"""Read-only checkpoint investigation; all experiments use copied models on CPU.

Run with --output NEW_DIRECTORY. No production settings/checkpoints are changed.
Phases can be rerun against the immutable snapshots saved by the first invocation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import numpy as np
import torch

from games.rl_2048.debug_learning import (
    make_ppo, predictions, metrics, estimate, continuations, one_step_branches,
)
from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.plot_training_diagnostics import _parse_raw_summary


RUNS = {
    'separate': 'run_20260921_003303*',
    'shared': 'run_20260920_122359*',
    'small': 'run_20260918_094712*',
}
SEED = 720391


def write(path, obj):
    def numpy_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=numpy_scalar))


def snapshot(folder):
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for label, pattern in RUNS.items():
        run, = Path('artifacts/ppo_2048').glob(pattern)
        checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        torch.save(checkpoint, folder / f'{label}.pt')
        summary = (run / 'summary.txt').read_text()
        raw = (run / 'raw_summary.txt').read_text()
        (folder / f'{label}_summary.txt').write_text(summary)
        (folder / f'{label}_raw_summary.txt').write_text(raw)
        records = _parse_raw_summary(folder / f'{label}_raw_summary.txt')
        evals = []
        for part in re.split(r'Outer iteration ', summary)[1:]:
            iteration = int(part.splitlines()[0])
            mean = re.search(r'Mean score: ([0-9.]+)', part)
            scores = re.search(r'Raw scores: (\[[^\]]+\])', part)
            if mean and scores:
                evals.append(dict(iteration=iteration, mean=float(mean[1]), scores=json.loads(scores[1])))
        metadata[label] = dict(source=str(run.resolve()), iteration=checkpoint['outer_iteration'],
            evaluation=checkpoint['evaluation'], model_config=checkpoint['resnet_config'],
            ppo_config=checkpoint['ppo_config'], lr=checkpoint['optimizer_state_dict']['param_groups'][0]['lr'],
            parameters=sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            training=records, evaluations=evals)
    write(folder / 'metadata.json', metadata)


def load(folder, label, seed=SEED):
    return make_ppo(torch.load(folder / f'{label}.pt', map_location='cpu', weights_only=False), seed)


def death_risks(game, legal):
    """Exact next-spawn terminal probability for each legal action."""
    result = {}
    for action in legal:
        grid, _, moved = game._simulate_move(game.state.grid, action)
        assert moved
        empty = np.argwhere(grid == 0)
        risk = 0.
        if len(empty) == 1:
            y, x = empty[0]
            for tile, probability in ((2, .9), (4, .1)):
                grid[y, x] = tile
                if not ((grid[:-1] == grid[1:]).any() or (grid[:, :-1] == grid[:, 1:]).any()):
                    risk += probability
        result[action] = risk
    return result


@torch.no_grad()
def evaluate_games(ppo, count, seed, mode):
    """Identical new initial seeds across models; separate action RNG streams.

    Returns complete episodes, systematic state samples, and per-game calibration
    aggregates. Aggregate uncertainty must resample GAMES, not correlated states.
    """
    games = [Game2048(seed=seed + i) for i in range(count)]
    rngs = [np.random.default_rng(seed + i + 50000000) for i in range(count)]
    traces = [[] for _ in games]
    active = list(range(count))
    for time in range(10000):
        if not active:
            break
        output = ppo.model(ppo._state_batch_tensor([games[i].state for i in active]))
        legal = [games[i].legal_actions() for i in active]
        mask = torch.ones_like(output.action_logits, dtype=torch.bool)
        for row, actions in enumerate(legal):
            mask[row, actions] = False
        logs = output.action_logits.masked_fill(mask, -torch.inf).log_softmax(-1)
        probabilities = logs.exp().numpy()
        values = ppo._untransform_predicted_value(output.value).numpy() / ppo.config.reward_multiplier
        for row, i in enumerate(active):
            game, probs = games[i], probabilities[row]
            cumulative = probs.astype(np.float64).cumsum()
            cumulative /= cumulative[-1]
            action = int(probs.argmax()) if mode == 'greedy' else int(np.searchsorted(cumulative, rngs[i].random(), side='right'))
            grid = game.state.grid
            empties = int((grid == 0).sum())
            risks = death_risks(game, legal[row]) if empties <= 1 else {a: 0. for a in legal[row]}
            min_risk = min(risks.values())
            avoidable_mass = sum(float(probs[a]) for a in risks if risks[a] > min_risk + 1e-8)
            max_tile = int(grid.max())
            entry = dict(step=time, board=grid.tolist(), score_before=game.total_score,
                value=float(values[row]), probabilities=probs.tolist(), action=action,
                empties=empties, max_tile=max_tile, legal_count=len(legal[row]),
                entropy=float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()),
                max_in_corner=bool(max_tile in (grid[0,0], grid[0,-1], grid[-1,0], grid[-1,-1])),
                death_risks={str(a):r for a,r in risks.items()},
                min_death_risk=min_risk, avoidable_action_mass=avoidable_mass,
                chosen_excess_risk=risks[action] - min_risk)
            result = game.step(action)
            assert result.action_applied
            entry['terminal'] = result.terminated
            traces[i].append(entry)
        active = [i for i in active if not games[i].done]
        if time % 500 == 499:
            print(f'  {mode}: step {time+1}, {len(active)}/{count} games still active', flush=True)
    if active:
        raise RuntimeError('Refusing truncated game evaluation')
    reports = []
    for i, (game, trace) in enumerate(zip(games, traces)):
        aggregates = {}
        for entry in trace:
            entry['remaining_score'] = game.total_score - entry['score_before']
            entry['moves_remaining'] = len(trace) - entry['step']
            strata = ['all', 'crowded' if entry['empties'] <= 2 else 'open',
                'late_tile' if entry['max_tile'] >= 1024 else 'early_tile']
            if entry['moves_remaining'] <= 20:
                strata.append('last_20_moves')
            for stratum in strata:
                a = aggregates.setdefault(stratum, dict(n=0, predicted_sum=0., return_sum=0., error_square_sum=0.))
                a['n'] += 1
                a['predicted_sum'] += entry['value']
                a['return_sum'] += entry['remaining_score']
                a['error_square_sum'] += (entry['value'] - entry['remaining_score']) ** 2
        # Keep systematic samples for repeated-continuation checks plus the final
        # eight decisions for qualitative inspection; calibration uses ALL states.
        saved = [e for e in trace if e['step'] % 16 == 0 or e['moves_remaining'] <= 8]
        opportunities = [e for e in trace if max(e['death_risks'].values()) > e['min_death_risk'] + 1e-8]
        reports.append(dict(seed=seed+i, score=game.total_score, moves=len(trace), max_tile=int(game.state.grid.max()),
            entropy=float(np.mean([e['entropy'] for e in trace])),
            empty_cells=float(np.mean([e['empties'] for e in trace])),
            max_in_corner=float(np.mean([e['max_in_corner'] for e in trace])),
            forced_fraction=float(np.mean([e['legal_count']==1 for e in trace])),
            risk_opportunities=len(opportunities), avoidable_mass_sum=sum(e['avoidable_action_mass'] for e in opportunities),
            avoidable_terminal=bool(trace[-1]['min_death_risk'] == 0),
            terminal_value=trace[-1]['value'], terminal_board=trace[-1]['board'],
            calibration=aggregates, samples=saved))
    return dict(mode=mode, score=estimate([g['score'] for g in reports]),
        median=float(np.median([g['score'] for g in reports])), games=reports)


def evaluations(folder, count):
    for label in RUNS:
        for mode in ('greedy', 'sampled'):
            path = folder / f'{label}_{mode}.json'
            if path.exists():
                continue
            print(f'Evaluating {label}: {mode}, {count} fresh games', flush=True)
            result = evaluate_games(load(folder, label), count, SEED + 10000, mode)
            write(path, result)
            print(label, mode, result['score'], flush=True)


def actor_epoch(ppo, rollout, mean, std):
    shuffled = ppo.rng.permutation(rollout)
    for start in range(0, len(shuffled), ppo.config.mini_batch_size):
        obj = ppo._compute_objective(shuffled[start:start + ppo.config.mini_batch_size], mean, std)
        loss = (obj.policy + ppo.config.entropy_coefficient * obj.negative_entropy).mean()
        ppo.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ppo.optimizer.step()


def updates(folder):
    from games.rl_2048.debug_learning import update_epoch
    for label in RUNS:
        output = folder / f'{label}_updates.json'
        if output.exists():
            continue
        print(f'Collecting native-size fresh rollout for {label}', flush=True)
        ppo = load(folder, label, SEED + 30000)
        rollout, mean, std = ppo.collect_batched_information()
        shuffle = copy.deepcopy(ppo.rng.bit_generator.state)
        torch.save(dict(rollout=rollout, mean=mean, std=std, shuffle=shuffle), folder / f'{label}_rollout.pt')
        logs, _ = predictions(ppo, rollout)
        targets = torch.tensor([s.value + s.advantage for s in rollout])
        initial = torch.nn.utils.parameters_to_vector(ppo.model.parameters()).detach().clone()
        result = dict(size=len(rollout), advantage_mean=mean, advantage_std=std, epochs=[], interventions={})
        old_actions = logs.argmax(-1)

        def measure(model):
            row = metrics(model, rollout, logs, targets, initial)
            new_logs, _ = predictions(model, rollout)
            row['greedy_action_change_fraction'] = float((new_logs.argmax(-1) != old_actions).float().mean())
            return row

        result['epochs'].append(dict(epoch=0, **measure(ppo)))
        for epoch in range(1, ppo.config.max_update_epochs + 1):
            during = ppo.run_epoch(rollout, mean, std, measure_gradient_diagnostics=epoch == 1)
            row = dict(epoch=epoch, during_epoch_kl=during.approximate_kl, **measure(ppo))
            result['epochs'].append(row)
            print(label, 'epoch', epoch, 'KL', row['full_kl'], 'EV', row['explained_variance'], flush=True)
            if epoch > ppo.config.min_update_epochs and during.approximate_kl > ppo.config.target_kl:
                break
        result['production_epochs'] = epoch
        for mode in ('actor_entropy_only', 'value_only', 'quarter_lr'):
            altered = load(folder, label, SEED + 30000)
            altered.rng.bit_generator.state = copy.deepcopy(shuffle)
            if mode == 'quarter_lr':
                for group in altered.optimizer.param_groups:
                    group['lr'] *= .25
                for _ in range(result['production_epochs']):
                    altered.run_epoch(rollout, mean, std)
            elif mode == 'actor_entropy_only':
                actor_epoch(altered, rollout, mean, std)
            else:
                update_epoch(altered, rollout, mean, std, 'value_only')
            result['interventions'][mode] = measure(altered)
        write(output, result)


def cases(folder):
    from types import SimpleNamespace
    rng = np.random.default_rng(SEED + 60000)
    selected = []
    # Symmetric coverage: one random crowded and one random high-tile state
    # from each model's sampled evaluation, selected without action return tests.
    for label in RUNS:
        data = json.loads((folder / f'{label}_sampled.json').read_text())
        pool = [s for g in data['games'] for s in g['samples'] if s['step'] % 16 == 0 and s['legal_count'] >= 2]
        for name, predicate in [('crowded', lambda s:s['empties']<=2), ('high_tile', lambda s:s['max_tile']>=1024)]:
            candidates = [s for s in pool if predicate(s)]
            chosen = candidates[int(rng.integers(len(candidates)))]
            selected.append(dict(source=label, selection=name, board=chosen['board']))
    # A known failure from the prior investigation, explicitly not an unbiased sample.
    selected.append(dict(source='previous investigation', selection='90% fatal right move',
        board=[[16,32,64,256],[8,16,32,64],[4,8,16,8],[2,2,8,2]]))
    steps = []
    for c in selected:
        game = Game2048(seed=0)
        game.state = State2048(grid=np.array(c['board']))
        mask = torch.ones(4, dtype=torch.bool)
        mask[game.legal_actions()] = False
        steps.append(SimpleNamespace(state=game.state, legal_actions_mask=mask))
    report = dict(cases=selected, models={})
    for label in RUNS:
        print(f'Common-board continuations: {label}', flush=True)
        ppo = load(folder, label)
        logs, values = predictions(ppo, steps)
        returns = continuations(ppo, steps, 64, SEED + 100000, greedy=False)
        report['models'][label] = [dict(probabilities=log.exp().tolist(), predicted_value=float(v)/ppo.config.reward_multiplier,
            returns=returns[(i,None)].tolist(), mc_value=estimate(returns[(i,None)]),
            one_step=one_step_branches(ppo, c['board']))
            for i,(c,log,v) in enumerate(zip(selected,logs,values))]
        write(folder / 'cases.json', report)


def confirm_common_actions(folder):
    """Independent confirmation after discovery: force the same first move.

    Cases 4 and 5 showed similar first-action preferences but different returns.
    Their IDs/actions are fixed before this second random stream is evaluated.
    """
    from types import SimpleNamespace
    discovery = json.loads((folder / 'cases.json').read_text())
    chosen = [(4, 2), (5, 3)]
    steps = []
    for case_id, action in chosen:
        state = State2048(grid=np.array(discovery['cases'][case_id]['board']))
        game = Game2048(seed=0)
        game.state = state
        assert action in game.legal_actions()
        mask = torch.ones(4, dtype=torch.bool)
        mask[action] = False  # Restricts only the forced FIRST move, not later moves.
        steps.append(SimpleNamespace(state=state, legal_actions_mask=mask))
    report = dict(cases=[dict(case_id=i, first_action=a) for i,a in chosen], models={})
    for label in RUNS:
        print('Confirming common first moves:',label,flush=True)
        scores = continuations(load(folder,label), steps, 128, SEED+700000, forced=True)
        report['models'][label] = [dict(case_id=i, first_action=a,
            estimate=estimate(scores[(local,a)]), samples=scores[(local,a)].tolist())
            for local,(i,a) in enumerate(chosen)]
        write(folder/'common_action_confirmation.json',report)


def final_checkpoint(folder, count):
    """Follow-up when the initially live separate run has finished."""
    source, = Path('artifacts/ppo_2048').glob(RUNS['separate'])
    checkpoint = torch.load(source/'final.pt', map_location='cpu', weights_only=False)
    torch.save(checkpoint, folder/'separate_final.pt')
    summary=(source/'raw_summary.txt').read_text()
    (folder/'separate_final_raw_summary.txt').write_text(summary)
    ppo=load(folder,'separate_final')
    evaluated=evaluate_games(ppo,count,SEED+10000,'greedy')
    write(folder/'separate_final_greedy.json',evaluated)
    captured=torch.load(folder/'separate_rollout.pt',map_location='cpu',weights_only=False)
    steps=captured['rollout']
    old_logs,_=predictions(load(folder,'separate'),steps)
    new_logs,_=predictions(ppo,steps)
    kl=(old_logs.exp()*(old_logs-new_logs).masked_fill(~torch.isfinite(old_logs),0)).sum(-1)
    result=dict(iteration=checkpoint['outer_iteration'],evaluation=checkpoint['evaluation'],
        fresh_greedy=evaluated['score'],training=_parse_raw_summary(folder/'separate_final_raw_summary.txt'),
        policy_change_since_900=dict(full_kl=float(kl.mean()),kl_p95=float(kl.quantile(.95)),
            greedy_action_change_fraction=float((old_logs.argmax(-1)!=new_logs.argmax(-1)).float().mean())))
    write(folder/'final_checkpoint.json',result)
    print('Final checkpoint fresh scores:',evaluated['score'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('snapshot','eval','updates','cases','confirm','final','all'), default='all')
    parser.add_argument('--games', type=int, default=96)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase in ('snapshot','all'):
        snapshot(args.output)
    if args.phase in ('eval','all'):
        evaluations(args.output, args.games)
    if args.phase in ('updates','all'):
        updates(args.output)
    if args.phase in ('cases','all'):
        cases(args.output)
    if args.phase == 'confirm':
        confirm_common_actions(args.output)
    if args.phase == 'final':
        final_checkpoint(args.output,args.games)


if __name__ == '__main__':
    main()
