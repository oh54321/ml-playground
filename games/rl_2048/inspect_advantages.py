"""Frozen-policy action-ranking audit. See ADVANTAGE_AUDIT.md for usage.

Three explicit phases: select boards, discover discrepancies, independently
confirm preselected action pairs. This module never creates an optimizer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
from statistics import NormalDist

import numpy as np
import torch

from games.rl_2048.agent import ResNet, ResNetConfig
from games.rl_2048.environment.game_2048 import Game2048, State2048


ACTION_NAMES = ('left', 'up', 'right', 'down')
VERSION = 1


def write_json(path: Path, data) -> None:
    """Replace a diagnostic result atomically; never used for input checkpoints."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def stream_seed(root: int, phase: int, item: int, trial: int, stream: int) -> int:
    """Disjoint deterministic RNG namespaces; no action ID so trials are paired."""
    sequence = np.random.SeedSequence([root, phase, item, trial, stream])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def sample_action(probabilities, rng) -> int:
    cumulative = np.asarray(probabilities, dtype=np.float64).cumsum()
    cumulative /= cumulative[-1]
    return int(np.searchsorted(cumulative, rng.random(), side='right'))


@dataclass(frozen=True)
class FrozenAgent:
    model: ResNet
    gamma: float
    gae_lambda: float
    reward_multiplier: float
    value_transform: str
    rollout_steps: int
    episode_limit: int

    @classmethod
    def load(cls, path: Path):
        # Only load trusted local checkpoints: existing checkpoint files include
        # optimizer/config objects and may require ordinary pickle loading.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = ResNet(ResNetConfig(**checkpoint['resnet_config']))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval().requires_grad_(False)
        cfg = checkpoint['ppo_config']
        multiplier = cfg['reward_multiplier']
        if multiplier <= 0 or not 0 <= cfg['gamma'] <= 1 or not 0 <= cfg['gae_lambda'] <= 1:
            raise ValueError('Requires positive reward multiplier and gamma/lambda in [0, 1]')
        if cfg['value_transform'] not in ('identity', 'signed_log1p', 'signed_sqrt1p'):
            raise ValueError('Unsupported checkpoint value transform')
        if cfg['steps_per_game_per_rollout'] < 1 or cfg['max_episode_steps'] < 1:
            raise ValueError('Checkpoint rollout and episode lengths must be positive')
        return cls(model, cfg['gamma'], cfg['gae_lambda'], multiplier,
                   cfg['value_transform'], cfg['steps_per_game_per_rollout'], cfg['max_episode_steps'])

    @torch.no_grad()
    def predict(self, games):
        states = torch.stack([game.state.to_tensor() for game in games])
        output = self.model(states)
        mask = torch.ones_like(output.action_logits, dtype=torch.bool)
        legal = [game.legal_actions() for game in games]
        for row, actions in enumerate(legal):
            if not actions:
                raise ValueError('Cannot query a policy on a terminal board')
            mask[row, actions] = False
        probabilities = output.action_logits.masked_fill(mask, -torch.inf).softmax(-1)
        value = output.value
        if self.value_transform == 'signed_log1p':
            value = value.sign() * value.abs().expm1()
        elif self.value_transform == 'signed_sqrt1p':
            value = value.sign() * ((value.abs() + 1).square() - 1)
        # Work in game-point units. Scaling all advantages by the same positive
        # reward multiplier does not change any action ranking.
        value = value / self.reward_multiplier
        if not torch.isfinite(value).all() or not torch.isfinite(probabilities).all():
            raise ValueError('Non-finite model output')
        return probabilities.numpy(), value.numpy(), legal


def trajectory_signals(rewards, values, gamma, gae_lambda, horizon):
    """Evaluate a complete terminal trajectory, including a truncated GAE prefix.

    values has length len(rewards)+1 and ends in terminal value zero. The prefix
    includes V(s_horizon) as a bootstrap; rewards after that boundary still enter
    the Monte Carlo measurement and the separate full-episode GAE measurement.
    """
    if not rewards or len(values) != len(rewards) + 1 or values[-1] != 0 or horizon < 1:
        raise ValueError('Expected a complete terminal trajectory and positive horizon')
    if not all(math.isfinite(float(x)) for x in (*rewards, *values)):
        raise ValueError('Non-finite trajectory')
    gae = 0.0
    prefix = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * gae
    for t in reversed(range(min(horizon, len(rewards)))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        prefix = delta + gamma * gae_lambda * prefix
    discounted = sum(gamma ** t * r for t, r in enumerate(rewards))
    return dict(game_score=float(sum(rewards)), discounted_return=float(discounted),
                gae=float(prefix), full_episode_gae=float(gae), steps=len(rewards),
                boundary_bootstrapped=horizon < len(rewards))


def game_from_board(board, seed):
    game = Game2048(seed=seed)
    game.state = State2048(grid=np.asarray(board, dtype=np.int64))
    game.total_score, game.done = 0, False
    # Initial random tiles were overwritten. Restart the spawn RNG so action
    # branches use the same future random stream, independent of construction.
    game.set_seed(seed)
    return game


def choose_boards(pools, counts, seed, crowded_cells, high_tile):
    """Game-balanced sampling; outcomes, scores, and advantages are not inputs.

    Representative games are chosen first; stress groups use remaining games.
    Exactly one board per source game across all groups; no duplicate grids.
    """
    rng = np.random.default_rng(seed)
    used_games, used_boards, selected = set(), set(), []
    predicates = {
        'representative': lambda b: True,
        'crowded': lambda b: b['empty_cells'] <= crowded_cells,
        'high_tile': lambda b: b['max_tile'] >= high_tile,
    }
    for group in ('representative', 'crowded', 'high_tile'):
        candidates = {}
        for game_id, pool in enumerate(pools):
            if game_id in used_games:
                continue
            eligible = [b for b in pool if predicates[group](b) and
                        tuple(v for row in b['board'] for v in row) not in used_boards]
            if eligible:
                candidates[game_id] = eligible
        if len(candidates) < counts[group]:
            raise ValueError(f'Need {counts[group]} {group} boards from unused games; '
                             f'only {len(candidates)} games qualify. Increase --games or reduce this group.')
        order = rng.permutation(list(candidates))
        taken = 0
        for game_id in order:
            if taken == counts[group]:
                break
            eligible = [b for b in candidates[game_id] if
                        tuple(v for row in b['board'] for v in row) not in used_boards]
            if not eligible:
                continue
            board = dict(eligible[int(rng.integers(len(eligible)))])
            board.update(id=len(selected), group=group, source_game=int(game_id))
            selected.append(board)
            used_games.add(int(game_id))
            used_boards.add(tuple(v for row in board['board'] for v in row))
            taken += 1
        if taken != counts[group]:
            raise ValueError(f'Duplicate boards prevent filling {group}; increase --games')
    return selected


def collect_board_pool(agent, games_count, root_seed, max_moves, batch_size):
    games = [Game2048(seed=stream_seed(root_seed, 0, i, 0, 0)) for i in range(games_count)]
    rngs = [np.random.default_rng(stream_seed(root_seed, 0, i, 0, 1)) for i in range(games_count)]
    pools = [[] for _ in games]
    active = list(range(games_count))
    for step in range(max_moves):
        if not active:
            break
        for start in range(0, len(active), batch_size):
            ids = active[start:start + batch_size]
            probabilities, _, legal = agent.predict([games[i] for i in ids])
            for row, i in enumerate(ids):
                game = games[i]
                if len(legal[row]) >= 2 and step < agent.episode_limit:
                    grid = game.state.grid
                    pools[i].append(dict(board=grid.tolist(), source_step=step,
                        source_seed=stream_seed(root_seed, 0, i, 0, 0),
                        empty_cells=int((grid == 0).sum()), max_tile=int(grid.max()),
                        gae_horizon=min(agent.rollout_steps - step % agent.rollout_steps,
                                        agent.episode_limit - step)))
                result = game.step(sample_action(probabilities[row], rngs[i]))
                assert result.action_applied
        active = [i for i in active if not games[i].done]
        if step % 250 == 249:
            print(f'Selection: {step+1} moves; {len(active)}/{games_count} games active', flush=True)
    if active:
        raise RuntimeError('Selection games hit --max-moves; refusing an incomplete candidate pool')
    return pools


def estimate(samples, z=1.959963984540054):
    values = np.asarray(samples, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite trials are required')
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return dict(mean=mean, se=se, n=len(values), interval=[mean-z*se, mean+z*se])


def compare_pair(actions, action_a, action_b, family_size=1):
    z = NormalDist().inv_cdf(1 - .05 / (2 * 2 * family_size))
    pair = dict(action_a=action_a, action_b=action_b)
    for key in ('discounted_return', 'gae', 'full_episode_gae'):
        a = np.array([t[key] for t in actions[str(action_a)]['trials']])
        b = np.array([t[key] for t in actions[str(action_b)]['trials']])
        pair[key] = estimate(a-b)
        if key != 'full_episode_gae':
            pair[key]['family_interval'] = estimate(a-b, z)['interval']
    mc, gae = pair['discounted_return'], pair['gae']
    pair['opposite_point_estimates'] = mc['mean'] * gae['mean'] < 0
    mlo, mhi = mc['family_interval']
    glo, ghi = gae['family_interval']
    pair['status'] = ('opposite' if (mlo > 0 and ghi < 0) or (mhi < 0 and glo > 0)
                      else 'aligned' if (mlo > 0 and glo > 0) or (mhi < 0 and ghi < 0)
                      else 'inconclusive')
    # Selection score uses discovery data only. Confirmation keeps this pair
    # fixed and uses an independent random stream, never reselecting a winner.
    pair['discrepancy_score'] = (min(abs(mc['mean'])/(mc['se']+1e-12),
                                      abs(gae['mean'])/(gae['se']+1e-12))
                                if pair['opposite_point_estimates'] else 0.)
    return pair


def summarize_trials(raw, probabilities, pairs, family_size):
    actions = {}
    for action, trials in raw.items():
        actions[str(action)] = dict(trials=trials, **{
            metric: estimate([t[metric] for t in trials])
            for metric in ('game_score', 'discounted_return', 'gae', 'full_episode_gae')})
    for action, trials in raw.items():
        for metric, name in [('discounted_return', 'mc_advantage'), ('gae', 'centered_gae')]:
            baseline = sum(probabilities[a]*np.array([t[metric] for t in rows]) for a,rows in raw.items())
            actions[str(action)][name] = estimate(np.array([t[metric] for t in trials])-baseline)
    comparisons = [compare_pair(actions, a, b, family_size) for a,b in pairs]
    return dict(actions=actions, comparisons=comparisons,
        best_mc_action=max(raw, key=lambda a: actions[str(a)]['discounted_return']['mean']),
        best_gae_action=max(raw, key=lambda a: actions[str(a)]['gae']['mean']))


def branch_trials(agent, board, trials, root_seed, phase, max_moves, batch_size):
    """All legal first actions; paired future spawn/action streams across actions."""
    template = game_from_board(board['board'], 0)
    legal = template.legal_actions()
    if len(legal) < 2:
        raise ValueError('Selected board needs at least two legal moves')
    keys = [(a, rep) for a in legal for rep in range(trials)]
    games = [game_from_board(board['board'], stream_seed(root_seed, phase, board['id'], rep, 0))
             for _,rep in keys]
    rngs = [np.random.default_rng(stream_seed(root_seed, phase, board['id'], rep, 1)) for _,rep in keys]
    rewards, values = [[] for _ in keys], [[] for _ in keys]
    active = list(range(len(keys)))
    for step in range(max_moves):
        if not active:
            break
        for start in range(0,len(active),batch_size):
            ids = active[start:start+batch_size]
            probs, predictions, _ = agent.predict([games[i] for i in ids])
            for row,i in enumerate(ids):
                values[i].append(float(predictions[row]))
                action = keys[i][0] if step == 0 else sample_action(probs[row],rngs[i])
                result = games[i].step(action)
                assert result.action_applied
                rewards[i].append(float(result.reward))
                if result.terminated:
                    values[i].append(0.)
        active = [i for i in active if not games[i].done]
        if step % 500 == 499:
            print(f"  Board {board['id']}: step {step+1}; {len(active)} branches active", flush=True)
    if active:
        raise RuntimeError('Branch trials hit --max-moves; no truncated results will be accepted')
    raw = {a:[] for a in legal}
    for i,(action,rep) in enumerate(keys):
        signals = trajectory_signals(rewards[i],values[i],agent.gamma,agent.gae_lambda,board['gae_horizon'])
        signals['trial'] = rep
        raw[action].append(signals)
        assert abs(signals['game_score']-games[i].total_score) < 1e-8
    return raw


def select(args):
    counts = dict(representative=args.representative, crowded=args.crowded, high_tile=args.high_tile)
    if any(v < 0 for v in counts.values()) or not 0 < sum(counts.values()) <= args.games:
        raise ValueError('Board counts must be nonnegative, positive in total, and no larger than --games')
    if not 0 <= args.crowded_cells <= 16 or args.high_tile_threshold < 2:
        raise ValueError('Invalid stress-test thresholds')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    snapshot = output/'checkpoint.pt'
    shutil.copyfile(args.checkpoint,snapshot)
    agent = FrozenAgent.load(snapshot)
    print('Collecting fresh sampled games; no branch results are used for selection.',flush=True)
    pools = collect_board_pool(agent,args.games,args.seed,args.max_moves,args.batch_size)
    boards = choose_boards(pools,counts,stream_seed(args.seed,0,0,0,2),
                           args.crowded_cells,args.high_tile_threshold)
    games = [game_from_board(b['board'],0) for b in boards]
    probabilities, values, legal = agent.predict(games)
    for b,p,v,actions in zip(boards,probabilities,values,legal):
        b.update(probabilities=p.tolist(),predicted_value_points=float(v),legal_actions=actions)
    manifest = dict(version=VERSION,source_checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(snapshot),seed=args.seed,gamma=agent.gamma,gae_lambda=agent.gae_lambda,
        reward_multiplier=agent.reward_multiplier,value_transform=agent.value_transform,
        rollout_steps=agent.rollout_steps,episode_limit=agent.episode_limit,
        torch_version=str(torch.__version__),numpy_version=np.__version__,device='cpu',
        selection=dict(games=args.games,counts=counts,crowded_cells=args.crowded_cells,
            high_tile_threshold=args.high_tile_threshold,eligible_states_per_game=[len(p) for p in pools],
            weighting='uniform eligible game, then uniform eligible time within game; no reused games',
            horizon='min(rollout_steps - source_step % rollout_steps, episode_limit - source_step)'),
        boards=boards)
    write_json(output/'boards.json',manifest)
    print(f'Saved {len(boards)} locked boards and checkpoint to {output}',flush=True)


def validate_manifest(folder):
    manifest = read_json(folder/'boards.json')
    if manifest['version'] != VERSION:
        raise ValueError('Unsupported manifest version')
    if sha256(folder/'checkpoint.pt') != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint snapshot changed since board selection')
    return manifest


def confirmation_plan(folder, manifest, requested, top):
    discovery = read_json(folder/'discovery'/'results.json')
    if discovery['settings']['manifest_sha256'] != sha256(folder/'boards.json'):
        raise ValueError('Discovery used a different board manifest')
    candidates = []
    for board in discovery['boards']:
        comparisons = board['comparisons']
        pair = max(comparisons,key=lambda p:p['discrepancy_score'])
        if pair['discrepancy_score'] == 0:
            pair = max(comparisons,key=lambda p:abs(p['discounted_return']['mean']))
        candidates.append(dict(board_id=board['id'],action_a=pair['action_a'],action_b=pair['action_b'],
            discovery_discrepancy_score=pair['discrepancy_score']))
    if requested is not None:
        ids = set(requested)
        if len(ids) != len(requested) or not ids.issubset({b['id'] for b in manifest['boards']}):
            raise ValueError('--boards must contain unique existing board IDs')
        selected = [p for p in candidates if p['board_id'] in ids]
    else:
        selected = sorted((p for p in candidates if p['discovery_discrepancy_score'] > 0),
                          key=lambda p:(-p['discovery_discrepancy_score'],p['board_id']))[:top]
    if not selected:
        raise ValueError('No opposite-ranking discovery candidates. Use --boards ID ... to '
                         'confirm an explicit board, or stop with an inconclusive/aligned discovery result.')
    return selected


def write_report(folder, result, manifest):
    phase = result['settings']['phase']
    lines = ['# Frozen-policy advantage audit', '',
        f"Phase: **{phase}**. Checkpoint: `{manifest['source_checkpoint']}`.",
        f"Gamma {manifest['gamma']}; GAE lambda {manifest['gae_lambda']}; "
        f"{result['settings']['trials']} paired trials per legal action. All values below are in game-point units.", '',
        'The actor and critic were frozen. After the forced first move, all actions were sampled from that actor. '
        'Returns evaluate this continuation policy, not optimal play. A fresh spawn/action RNG namespace is used '
        'for confirmation. Trials with the same board ID and trial number are paired across first actions.', '',
        'Primary comparison: discounted Monte Carlo return versus GAE cut at the selected board\'s next '
        'rollout/time-limit boundary. Complete games are still played to measure return. Full-episode GAE is '
        'reported separately. With gamma=1, discounted return equals remaining game score.', '',
        'Intervals are normal-approximation Monte Carlo intervals over paired trial differences. Status uses '
        'Bonferroni-adjusted intervals across BOTH primary metrics and all predeclared pairs in this phase. '
        'These are approximate, particularly for small samples or heavy-tailed returns. Discovery is exploratory; '
        'independent confirmation tests a fixed pair, not a newly selected winner. Inconclusive is not evidence '
        'of agreement. Zero observed variance in a small sample is not proof of certainty.', '',
        'Representative selection is game-balanced, not uniform over all training transitions. Stress groups '
        'are reported separately and are not estimates of population failure frequency. At most one board '
        'was selected per source game. Board selection used no branch-return or advantage measurements.', '',
        'Advantages are not normalized by a rollout-wide mean/std here: subtracting a common mean and dividing '
        'by a positive std preserves pairwise action rankings. This tests conditional mean learning signals, '
        'not the full clipped PPO update or entropy term.', '']
    for group in ('representative','crowded','high_tile'):
        rows = [b for b in result['boards'] if b['group']==group]
        if not rows:
            continue
        lines += [f'## {group}', '']
        for board in rows:
            lines += [f"### Board {board['id']}", '',
                f"Source game {board['source_game']}, move {board['source_step']}; "
                f"GAE horizon {board['gae_horizon']} transitions; critic value {board['predicted_value_points']:.1f}.",
                '', '```', *[' '.join(f'{v:5d}' for v in row) for row in board['board']], '```', '']
            for action,data in board['actions'].items():
                a=int(action)
                lines.append(f"- {ACTION_NAMES[a]} (policy {board['probabilities'][a]:.1%}): "
                    f"MC return {data['discounted_return']['mean']:.1f}; GAE {data['gae']['mean']:.1f}; "
                    f"full-episode GAE {data['full_episode_gae']['mean']:.1f}; "
                    f"policy-centered MC advantage {data['mc_advantage']['mean']:.1f}.")
            lines += ['']
            for pair in board['comparisons']:
                mc,gae=pair['discounted_return'],pair['gae']
                lines.append(f"- **{ACTION_NAMES[pair['action_a']]} minus {ACTION_NAMES[pair['action_b']]}: "
                    f"{pair['status']}**. MC difference {mc['mean']:+.1f} "
                    f"[{mc['family_interval'][0]:+.1f}, {mc['family_interval'][1]:+.1f}]; "
                    f"GAE difference {gae['mean']:+.1f} "
                    f"[{gae['family_interval'][0]:+.1f}, {gae['family_interval'][1]:+.1f}].")
            lines += ['']
    lines += ['## Reproducibility', '', 'The immutable selection is in ../boards.json and the frozen checkpoint '
              'is ../checkpoint.pt. settings.json records the fixed phase plan and manifest hash. Individual '
              'board JSON files contain every trial\'s return, GAE, full-episode GAE, length, and bootstrap flag. '
              'The root seed plus phase/board/trial/stream IDs reconstruct RNG streams. No optimizer or training '
              'update was run. Resume uses the original plan and only computes missing complete board files.', '']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def evaluate(args):
    folder=args.folder.resolve()
    manifest=validate_manifest(folder)
    phase='confirmation' if args.command=='confirm' else 'discovery'
    if phase=='confirmation':
        plan=confirmation_plan(folder,manifest,args.boards,args.top)
    else:
        plan=[dict(board_id=b['id'],action_a=a,action_b=c) for b in manifest['boards']
              for a,c in combinations(b['legal_actions'],2)]
    phase_dir=folder/phase
    settings=dict(version=VERSION,phase=phase,manifest_sha256=sha256(folder/'boards.json'),
        trials=args.trials,max_moves=args.max_moves,batch_size=args.batch_size,
        threads=args.threads,seed=manifest['seed'],rng_namespace=2 if phase=='confirmation' else 1,plan=plan)
    if args.resume:
        if read_json(phase_dir/'settings.json') != settings:
            raise ValueError('Resume settings or plan differ; use the original arguments')
    else:
        phase_dir.mkdir(exist_ok=False)
        # Persist the selected action pairs BEFORE running any confirmation trials.
        write_json(phase_dir/'settings.json',settings)
    agent=FrozenAgent.load(folder/'checkpoint.pt')
    results=[]
    for board in manifest['boards']:
        pairs=[(p['action_a'],p['action_b']) for p in plan if p['board_id']==board['id']]
        if not pairs:
            continue
        path=phase_dir/f"board_{board['id']:03d}.json"
        if path.exists():
            result=read_json(path)
            if result['settings_sha256'] != sha256(phase_dir/'settings.json'):
                raise ValueError('Existing board result does not match this phase')
        else:
            print(f"{phase}: board {board['id']} ({board['group']}), "
                  f"{len(board['legal_actions'])} actions x {args.trials} trials",flush=True)
            raw=branch_trials(agent,board,args.trials,manifest['seed'],settings['rng_namespace'],
                              args.max_moves,args.batch_size)
            result=dict(board,**summarize_trials(raw,board['probabilities'],pairs,len(plan)),
                        settings_sha256=sha256(phase_dir/'settings.json'))
            write_json(path,result)
        results.append(result)
    result=dict(settings=settings,boards=results)
    write_json(phase_dir/'results.json',result)
    write_report(phase_dir,result,manifest)
    assert sha256(folder/'checkpoint.pt')==manifest['checkpoint_sha256']
    print(f"Complete: {phase_dir/'report.md'}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    selection=commands.add_parser('select',help='Freeze checkpoint and select boards before measuring actions')
    selection.add_argument('checkpoint',type=Path)
    selection.add_argument('--output',type=Path,required=True)
    selection.add_argument('--games',type=int,default=96)
    selection.add_argument('--representative',type=int,default=12)
    selection.add_argument('--crowded',type=int,default=6)
    selection.add_argument('--high-tile',type=int,default=6)
    selection.add_argument('--crowded-cells',type=int,default=2)
    selection.add_argument('--high-tile-threshold',type=int,default=1024)
    selection.add_argument('--seed',type=int,default=20482026)
    for name in ('evaluate','confirm'):
        sub=commands.add_parser(name,help='Run discovery' if name=='evaluate' else 'Independently confirm fixed discovery pairs')
        sub.add_argument('folder',type=Path)
        sub.add_argument('--trials',type=int,default=64 if name=='evaluate' else 256)
        sub.add_argument('--resume',action='store_true')
        if name=='confirm':
            choice=sub.add_mutually_exclusive_group()
            choice.add_argument('--boards',type=int,nargs='+',help='Explicit discovery board IDs')
            choice.add_argument('--top',type=int,default=3,help='Top opposite-ranking discovery candidates')
    for sub in (selection,*[commands.choices[n] for n in ('evaluate','confirm')]):
        sub.add_argument('--max-moves',type=int,default=10000)
        sub.add_argument('--batch-size',type=int,default=256)
        sub.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if min(args.max_moves,args.batch_size,args.threads)<1:
        parser.error('max-moves, batch-size, and threads must be positive')
    if args.command=='select' and (args.games<1 or args.seed<0):
        parser.error('games must be positive and seed nonnegative')
    if args.command!='select' and args.trials<2:
        parser.error('At least two trials per action are required')
    if args.command=='confirm' and args.top is not None and args.top<1:
        parser.error('top must be positive')
    torch.set_num_threads(args.threads)
    if args.command=='select':
        select(args)
    else:
        evaluate(args)


if __name__=='__main__':
    main()
