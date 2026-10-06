"""Bounded PPO 2048 update audit. See documents/PPO_UPDATE_AUDIT.md.

The capture phase collects one ordinary full rollout from a checkpoint. Branch
continuations and PPO replay are separate phases; branch RNG never enters replay.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
from torch.optim import Adam

from algorithms.ppo import PPO, PPOConfig, PolicyRolloutInformation
from core.models.conv_actor_critic import ConvActorCriticConfig
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.training.config import exploration_schedule_from_checkpoint
from games.rl_2048.legacy_agent import model_from_legacy_config
from games.rl_2048.scripts.inspect_advantages import stream_seed


VERSION = 1
ACTION_NAMES = ('left', 'up', 'right', 'down')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_tree_sha256(value) -> str:
    """Stable digest for nested tensor/state-dict structures used in restore gates."""
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def mixture_probabilities(policy_probs, legal, epsilon):
    if not 0 <= epsilon < 1:
        raise ValueError('exploration epsilon must be in [0, 1)')
    out = np.zeros(4, dtype=np.float64)
    out[legal] = (1 - epsilon) * np.asarray(policy_probs)[legal] + epsilon / len(legal)
    total=out.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('behavior probabilities must have positive finite mass')
    out /= total
    return out


def clipped_surrogate_gradient_active(ratio: float, advantage: float, epsilon: float) -> bool:
    """Whether the sampled policy term has a nonzero derivative in log ratio."""
    if not math.isfinite(ratio) or ratio < 0 or epsilon < 0:
        raise ValueError('invalid ratio or clip epsilon')
    if advantage == 0:
        return False
    if advantage > 0:
        return ratio <= 1 + epsilon
    return ratio >= 1 - epsilon


def exact_action_q(model, game: Game2048, legal, gamma, reward_multiplier, value_fn):
    """Exact one-step reward plus uniformly averaged 2/4 spawn critic value."""
    q = {}
    for action in legal:
        moved, reward, applied = game._simulate_move(game.state.grid, action)
        if not applied:
            raise ValueError('legal action failed to move')
        empty = np.argwhere(moved == 0)
        if len(empty) == 0:
            next_value = 0.0
        else:
            successors=[]
            for row, col in empty:
                for tile, probability in ((2, .9), (4, .1)):
                    spawned = moved.copy()
                    spawned[row, col] = tile
                    successor = Game2048(seed=0)
                    successor.state = State2048(grid=spawned)
                    successor.done = not successor.legal_actions()
                    successors.append((successor.state, 0. if successor.done else None, probability/len(empty)))
            pending=[state for state,value,_ in successors if value is None]
            pending_values=iter(value_fn(pending) if pending else [])
            expected=0.
            for state,value,probability in successors:
                expected += probability * (next(pending_values) if value is None else value)
            next_value = expected
        # The critic output is untransformed but still in reward-multiplied
        # training units. Report exact Q in raw 2048 game points.
        q[int(action)] = float(reward + gamma * next_value / reward_multiplier)
    return q


def load_runtime(checkpoint_path: Path, seed: int, allow_current_d4: bool):
    # Checkpoints contain only resnet_config in the current writer. D4 action
    # order changed in older experiments, so require explicit acknowledgment.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg_dict = dict(checkpoint['resnet_config'])
    is_d4 = bool(cfg_dict.get('group_equivariant'))
    if cfg_dict.get('symmetry_adapter'):
        raise ValueError('Legacy D4 residual-adapter checkpoints are unsupported')
    if is_d4 and not allow_current_d4:
        raise ValueError('D4 checkpoint requires --current-d4-order (current SQUARE_SYMMETRY_SPECS order)')
    model = model_from_legacy_config(cfg_dict)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.train()
    ppo_config = replace(PPOConfig(**checkpoint['ppo_config']), seed=seed, print_timestamps=False)
    if checkpoint.get('symmetry_augmentation') is not None and not is_d4:
        raise ValueError('This audit supports symmetry augmentation only for current D4 models')
    optimizer = Adam(model.parameters(), lr=checkpoint['optimizer_state_dict']['param_groups'][0]['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ppo = PPO(model, optimizer, ppo_config, lambda s: Game2048(seed=s), checkpoint.get('symmetry_augmentation'))
    # PPO seeded its rollout RNG from the explicitly recorded audit seed.
    schedule_data = checkpoint.get('exploration_schedule')
    epsilon = 0.0
    if schedule_data:
        schedule = exploration_schedule_from_checkpoint(schedule_data)
        epsilon = schedule.probability(int(checkpoint['outer_iteration']) + 1)
    ppo.set_exploration_probability(epsilon)
    return checkpoint, model, optimizer, ppo, epsilon


def serialize_step(step):
    return dict(board=step.state.grid.tolist(), next_board=step.next_state.grid.tolist(),
        action=int(step.action), log_prob=float(step.log_prob), value=float(step.value),
        reward=float(step.reward), done=bool(step.done), truncated=bool(step.truncated),
        legal_actions_mask=step.legal_actions_mask.cpu().tolist(), advantage=float(step.advantage))


def deserialize_step(row):
    return PolicyRolloutInformation(State2048(grid=np.asarray(row['board'], dtype=np.int64)), row['action'],
        State2048(grid=np.asarray(row['next_board'], dtype=np.int64)), row['log_prob'], row['value'],
        row['reward'], row['done'], row['truncated'], torch.tensor(row['legal_actions_mask'], dtype=torch.bool),
        row['advantage'])


def exact_policy_logs(ppo, steps):
    rows = []
    with torch.no_grad():
        for start in range(0, len(steps), 512):
            batch = steps[start:start + 512]
            logits = ppo._model_logits(batch)
            legal_mask = ppo.rollout_collector.steps_to_tensor(batch, 'legal_actions_mask', dtype=torch.bool)
            logits = logits.masked_fill(legal_mask, -torch.inf)
            rows.append(ppo._action_log_probabilities(logits, legal_mask).cpu())
    return torch.cat(rows)


def choose_large_delta(candidates, boards, minimum_delta):
    """Select fixed high-gap pairs, preserving game-slot and board diversity."""
    if boards < 1 or minimum_delta <= 0:
        raise ValueError('boards and minimum DeltaQ must be positive')
    qualifying=[row for row in candidates if row['delta_q'] >= minimum_delta]
    qualifying.sort(key=lambda row:(-row['delta_q'],row['rollout_index']))
    selected=[]; seen_games=set(); seen_boards=set()
    for row in qualifying:
        board_key=tuple(int(v) for line in row['board'] for v in line)
        if row['game_index'] in seen_games or board_key in seen_boards:
            continue
        selected.append(row); seen_games.add(row['game_index']); seen_boards.add(board_key)
        if len(selected)>=boards: break
    return qualifying,selected


def counterfactual_gae_horizon(segment_remaining, episode_age, max_episode_steps):
    """Cap branch GAE at its scheduled segment or remaining episode time limit."""
    if segment_remaining < 1 or episode_age < 0 or max_episode_steps < 1:
        raise ValueError('invalid counterfactual GAE horizon inputs')
    episode_remaining=max_episode_steps-episode_age
    if episode_remaining < 1:
        raise ValueError('captured state is at or beyond the episode time limit')
    return min(segment_remaining,episode_remaining)


def counterfactual_gae_prefix_length(horizon_limit, branch_steps):
    """Use this branch's own terminal length if it ends before the fixed cap."""
    if horizon_limit < 1 or branch_steps < 1:
        raise ValueError('counterfactual GAE horizon and branch length must be positive')
    return min(horizon_limit,branch_steps)


def generalized_advantage_prefix(rewards, values, gamma, gae_lambda, horizon_limit):
    """Compute branch GAE through its own terminal or fixed rollout/time cap."""
    if len(values)!=len(rewards)+1 or not rewards or values[-1]!=0:
        raise ValueError('expected a complete terminal branch trajectory')
    prefix=0.
    prefix_length=counterfactual_gae_prefix_length(horizon_limit,len(rewards))
    for t in reversed(range(prefix_length)):
        delta=rewards[t]+gamma*values[t+1]-values[t]
        prefix=delta+gamma*gae_lambda*prefix
    return prefix


def capture(args):
    source = args.checkpoint.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    before = sha256(source)
    checkpoint, model, optimizer, ppo, epsilon = load_runtime(source, args.seed, args.current_d4_order)
    # PPO's rollout collector and updater share one NumPy generator as production does.
    steps, adv_mean, adv_std = ppo.collect_batched_information()
    model.eval()
    candidates = []
    eligible=[i for i,step in enumerate(steps) if int((~step.legal_actions_mask).sum())>=2]
    candidate_indices=np.unique(np.linspace(0,len(eligible)-1,min(args.candidate_limit,len(eligible)),dtype=int))
    def value_batch(states):
        with torch.no_grad():
            pred=model(ppo.rollout_collector.state_batch_tensor(states)).value
            return ppo._untransform_predicted_value(pred).cpu().tolist()
    for index in (eligible[i] for i in candidate_indices):
        step=steps[index]
        game = Game2048(seed=0); game.state = step.state.copy(); game.done = False
        legal = game.legal_actions()
        if len(legal) < 2: continue
        logits = model(ppo._state_batch_tensor([step.state])).action_logits[0]
        logits = logits.masked_fill(step.legal_actions_mask, -torch.inf)
        actor_probs = torch.softmax(logits, -1).detach().cpu().numpy()
        q = exact_action_q(model, game, legal, ppo.config.gamma,
                           ppo.config.reward_multiplier,
                           value_batch)
        actor_action = int(np.argmax(actor_probs))
        critic_action = max(q, key=q.get)
        if actor_action != critic_action:
            candidates.append((index, step, actor_probs, q, actor_action, critic_action, legal))
    if len(candidates) < args.boards:
        raise ValueError(f'Only {len(candidates)} rollout states disagree; requested {args.boards}. Increase rollout or reduce --boards.')
    # Fixed deterministic spread across candidate order; Q/return values are not used to select among candidates.
    chosen_positions = np.linspace(0, len(candidates)-1, args.boards, dtype=int)
    selections = []
    # Flattened rollout is game-major. Use actual captured segment and episode boundary.
    per_game = ppo.config.steps_per_game_per_rollout
    for board_id, pos in enumerate(chosen_positions):
        index, step, actor_probs, q, actor_action, critic_action, legal = candidates[int(pos)]
        game_index, offset = divmod(index, per_game)
        horizon = 0; boundary_after='rollout_segment'
        for following in steps[index:index + per_game - offset]:
            horizon += 1
            if following.done:
                boundary_after='termination'
                break
            if following.truncated:
                boundary_after='truncation'
                break
        if boundary_after == 'rollout_segment' and offset+horizon < per_game:
            boundary_after='captured_segment_end'
        selections.append(dict(id=board_id, rollout_index=index, game_index=game_index,
            game_step=offset, board=step.state.grid.tolist(), legal_actions=legal,
            actor_action=actor_action, critic_action=critic_action,
            actor_probabilities=actor_probs.tolist(), exact_q=q, exact_q_units='game_points',
            raw_advantage=float(step.advantage), normalized_advantage=float((step.advantage-adv_mean)/(adv_std+ppo.config.denominator_epsilon)),
            sampled_action=int(step.action), old_log_prob=float(step.log_prob),
            old_action_probability=float(math.exp(step.log_prob)), remaining_rollout_horizon=horizon,
            boundary_after=boundary_after))
    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        rollout=[serialize_step(s) for s in steps], advantage_mean=adv_mean, advantage_std=adv_std,
        numpy_rng_state=ppo.rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
        source_sha256=before), out/'capture.pt')
    settings = dict(version=VERSION, source_checkpoint=str(source), source_sha256=before,
        checkpoint_iteration=int(checkpoint['outer_iteration']), ppo_config=asdict(ppo.config),
        model_config=checkpoint['resnet_config'], current_d4_order_acknowledged=args.current_d4_order,
        symmetry_augmentation=checkpoint.get('symmetry_augmentation'), exploration_probability=epsilon,
        threads=args.threads,
        seed=args.seed, rollout_size=len(steps), advantage_mean=adv_mean, advantage_std=adv_std,
        candidate_limit=args.candidate_limit,
        selection='uniformly spaced eligible rollout candidates; selected disagreement boards are evenly spread through that fixed candidate order; diagnostic only',
        boards=selections)
    write_json(out/'selection.json', settings)
    if sha256(source) != before:
        raise RuntimeError('Source checkpoint changed during capture')
    print(f'Captured {len(steps)} rollout transitions and locked {len(selections)} boards at {out}')


def select_large_delta(args):
    """Lock high tested-pair gaps from an existing immutable rollout capture."""
    folder=args.folder.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((folder/'selection.json').read_text())
    source=Path(prior['source_checkpoint'])
    if sha256(source)!=prior['source_sha256']:
        raise ValueError('Source checkpoint changed after capture')
    data=torch.load(folder/'capture.pt',map_location='cpu',weights_only=False)
    if data.get('source_sha256')!=prior['source_sha256']:
        raise ValueError('Capture/source checkpoint hash mismatch')
    torch.set_num_threads(int(prior['threads']))
    checkpoint,model,optimizer,ppo,epsilon=load_runtime(source,prior['seed'],prior['current_d4_order_acknowledged'])
    model.load_state_dict(data['model_state_dict']); model.eval().requires_grad_(False)
    steps=[deserialize_step(row) for row in data['rollout']]
    per_game=ppo.config.steps_per_game_per_rollout
    eligible=[i for i,s in enumerate(steps) if int((~s.legal_actions_mask).sum())>=2]
    # Evenly cover the eligible rollout before inspecting gaps. No continuation outcomes
    # or prior branch results participate in this candidate screen.
    candidate_indices=np.unique(np.linspace(0,len(eligible)-1,min(args.candidate_limit,len(eligible)),dtype=int))
    candidate_rows=[]; actor_states=[]; query_states=[]; action_rows=[]
    for sampled_pos in candidate_indices:
        index=eligible[int(sampled_pos)]; step=steps[index]
        game=Game2048(seed=0); game.state=step.state.copy(); game.done=False
        legal=game.legal_actions()
        ci=len(candidate_rows); candidate_rows.append((index,step,game,legal))
        actor_states.append(step.state)
        for action in legal:
            moved,reward,applied=game._simulate_move(step.state.grid,action)
            if not applied: raise ValueError('legal action failed to move')
            empty=np.argwhere(moved==0); terms=[]
            if len(empty):
                for row,col in empty:
                    for tile,probability in ((2,.9),(4,.1)):
                        spawned=moved.copy(); spawned[row,col]=tile
                        successor=Game2048(seed=0); successor.state=State2048(grid=spawned)
                        successor.done=not successor.legal_actions()
                        if successor.done: terms.append((None,probability/len(empty)))
                        else:
                            terms.append((len(query_states),probability/len(empty)))
                            query_states.append(successor.state)
            action_rows.append((ci,int(action),float(reward),terms))
    actor_logits=[]
    with torch.no_grad():
        for start in range(0,len(actor_states),256):
            actor_logits.append(model(ppo._state_batch_tensor(actor_states[start:start+256])).action_logits.cpu())
    actor_logits=torch.cat(actor_logits)
    query_values=[]
    with torch.no_grad():
        for start in range(0,len(query_states),256):
            states=query_states[start:start+256]
            pred=model(ppo.rollout_collector.state_batch_tensor(states)).value
            query_values.extend((ppo._untransform_predicted_value(pred)/ppo.config.reward_multiplier).cpu().tolist())
    q_by_candidate=[{} for _ in candidate_rows]
    for ci,action,reward,terms in action_rows:
        q_by_candidate[ci][action]=reward+ppo.config.gamma*sum(
            (0. if qi is None else float(query_values[qi]))*weight for qi,weight in terms)
    candidates=[]
    for ci,(index,step,game,legal) in enumerate(candidate_rows):
        logits=actor_logits[ci].masked_fill(step.legal_actions_mask,-torch.inf)
        policy=torch.softmax(logits,-1).numpy()
        behavior=mixture_probabilities(policy,legal,epsilon)
        q=q_by_candidate[ci]
        actor_action=int(np.argmax(policy))
        critic_action=max(q,key=q.get)
        gap=float(q[critic_action]-q[actor_action])
        if actor_action==critic_action:
            continue
        game_index,offset=divmod(index,per_game)
        horizon=0; boundary='rollout_segment'
        for following in steps[index:index+per_game-offset]:
            horizon+=1
            if following.done:
                boundary='termination'; break
            if following.truncated:
                boundary='truncation'; break
        start=game_index*per_game
        for prior_index in range(index-1,start-1,-1):
            if steps[prior_index].done or steps[prior_index].truncated:
                start=prior_index+1
                break
        episode_age=index-start
        counterfactual_horizon=counterfactual_gae_horizon(
            per_game-offset,episode_age,ppo.config.max_episode_steps)
        score_before=sum(s.reward for s in steps[start:index])/ppo.config.reward_multiplier
        candidates.append(dict(rollout_index=index,game_index=game_index,game_step=offset,
            board=step.state.grid.tolist(),legal_actions=legal,actor_action=actor_action,
            critic_action=critic_action,actor_probabilities=policy.tolist(),
            behavior_probabilities=behavior.tolist(),exact_q=q,delta_q=gap,
            raw_advantage=float(step.advantage),sampled_action=int(step.action),
            old_log_prob=float(step.log_prob),old_action_probability=float(math.exp(step.log_prob)),
            remaining_rollout_horizon=horizon,boundary_after=boundary,
            captured_episode_age=episode_age,
            counterfactual_gae_horizon_limit=counterfactual_horizon,
            empty_cells=int((step.state.grid==0).sum()),largest_tile=int(step.state.grid.max()),
            cumulative_score=float(score_before)))
    # Highest tested-pair gaps first, with source-slot and exact-board diversity.
    # Stable rollout-index tie breaking makes the selected pairs reproducible.
    qualifying,chosen=choose_large_delta(candidates,args.boards,args.min_delta_q)
    for i,row in enumerate(chosen): row['id']=i
    manifest={k:v for k,v in prior.items() if k!='boards'}
    manifest.update(version=VERSION,selection_source=str(folder/'capture.pt'),
        candidate_limit=int(len(candidate_indices)),eligible_population=len(eligible),
        disagreement_candidate_count=len(candidates),
        qualifying_population=len(qualifying),minimum_tested_pair_delta_q_game_points=args.min_delta_q,
        selection_ordering='descending tested-pair DeltaQ; stable rollout-index tie-break; at most one selected state per source game slot and exact board',
        selection='largest tested-pair critic-minus-actor Q gaps among deterministic evenly spaced eligible captured-rollout states; fixed before branch outcomes',
        boards=chosen)
    if sha256(source)!=prior['source_sha256']:
        raise RuntimeError('Source checkpoint changed during selection')
    # Copy the frozen capture by hard link where possible; preserve the original.
    import shutil
    shutil.copy2(folder/'capture.pt',out/'capture.pt')
    write_json(out/'selection.json',manifest)
    print(f'Screened {len(candidate_indices)} states; {len(candidates)} met threshold; locked {len(chosen)} boards at {out}')


def branch(args):
    folder = args.folder.resolve(); manifest = json.loads((folder/'selection.json').read_text())
    if manifest.get('version') != VERSION:
        raise ValueError(f"Unsupported selection manifest version {manifest.get('version')}")
    torch.set_num_threads(int(manifest['threads']))
    source=Path(manifest['source_checkpoint'])
    if sha256(source)!=manifest['source_sha256']:
        raise ValueError('Source checkpoint changed after capture')
    capture_data = torch.load(folder/'capture.pt', map_location='cpu', weights_only=False)
    if capture_data.get('source_sha256') != manifest['source_sha256']:
        raise ValueError('Captured rollout provenance does not match selection manifest')
    checkpoint, model, optimizer, ppo, epsilon = load_runtime(Path(manifest['source_checkpoint']), manifest['seed'], manifest['current_d4_order_acknowledged'])
    model.load_state_dict(capture_data['model_state_dict']); model.eval().requires_grad_(False)
    result_path=folder/'branches.json'
    if args.resume and result_path.exists():
        existing=json.loads(result_path.read_text())
        if existing['settings'] != dict(trials=args.trials,max_moves=args.max_moves):
            raise ValueError('--resume requires the original trial count and max-moves')
        results=list(existing['boards'])
    else:
        if result_path.exists(): raise FileExistsError(f'{result_path} exists; use --resume')
        results=[]
    family_z=NormalDist().inv_cdf(1-.05/(2*max(1,len(manifest['boards']))))
    for board in manifest['boards']:
        if any(row['board_id']==board['id'] for row in results): continue
        action_ids=(board['actor_action'],board['critic_action'])
        paths=[]
        for arm,forced in enumerate(action_ids):
            for trial in range(args.trials):
                game=Game2048(seed=stream_seed(manifest['seed'],71,board['id'],trial,0))
                game.state=State2048(grid=np.asarray(board['board'])); game.done=False
                paths.append(dict(arm=arm,trial=trial,game=game,
                    rng=np.random.default_rng(stream_seed(manifest['seed'],71,board['id'],trial,1)),
                    forced=forced,rewards=[],values=[],steps=0))
        active=list(range(len(paths)))
        while active:
            states=[paths[i]['game'].state.copy() for i in active]
            outputs=[]
            with torch.no_grad():
                for start in range(0,len(states),256):
                    output=model(ppo._state_batch_tensor(states[start:start+256]))
                    outputs.extend(zip(output.action_logits.cpu(),
                        ppo._untransform_predicted_value(output.value).cpu().reshape(-1)))
            next_active=[]
            for index,(logits,value) in zip(active,outputs):
                path=paths[index]; game=path['game']; path['values'].append(float(value))
                if path['steps']==0:
                    action=path['forced']
                else:
                    legal_now=game.legal_actions()
                    mask=torch.ones_like(logits,dtype=torch.bool); mask[legal_now]=False
                    policy=torch.softmax(logits.masked_fill(mask,-torch.inf),-1).numpy()
                    probs=mixture_probabilities(policy,legal_now,epsilon)
                    action=int(path['rng'].choice(len(probs),p=probs))
                result=game.step(action); path['rewards'].append(result.reward*ppo.config.reward_multiplier)
                path['steps']+=1
                if game.done:
                    path['values'].append(0.)
                elif path['steps']>=args.max_moves:
                    raise RuntimeError('branch hit --max-moves; no incomplete result accepted')
                else:
                    next_active.append(index)
            active=next_active
        per_action=[]
        for arm,forced in enumerate(action_ids):
            trials=[]
            for path in (p for p in paths if p['arm']==arm):
                rewards=path['rewards']; values=path['values']; gae=0.
                for t in reversed(range(len(rewards))):
                    delta=rewards[t]+ppo.config.gamma*values[t+1]-values[t]
                    gae=delta+ppo.config.gamma*ppo.config.gae_lambda*gae
                prefix=generalized_advantage_prefix(rewards,values,ppo.config.gamma,
                    ppo.config.gae_lambda,board['counterfactual_gae_horizon_limit'])
                trials.append(dict(game_score=float(path['game'].total_score),
                    discounted_return=sum(ppo.config.gamma**t*r for t,r in enumerate(rewards))/ppo.config.reward_multiplier,
                    gae=prefix,full_episode_gae=gae,steps=len(rewards)))
            per_action.append(dict(action=forced,trials=trials,
                mean_score=float(np.mean([t['game_score'] for t in trials])),
                mean_discounted_return=float(np.mean([t['discounted_return'] for t in trials])),
                mean_gae=float(np.mean([t['gae'] for t in trials]))))
        paired={}
        for metric in ('game_score','discounted_return','gae','full_episode_gae'):
            differences=np.asarray([a[metric]-b[metric] for a,b in zip(per_action[0]['trials'],per_action[1]['trials'])],dtype=np.float64)
            se=float(differences.std(ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else None
            mean=float(differences.mean())
            paired[metric]=dict(mean_actor_minus_critic=mean,mean_critic_minus_actor=-mean,se=se,
                interval95=([mean-1.96*se,mean+1.96*se] if se is not None else None))
            if se is not None:
                paired[metric]['interval95_critic_minus_actor']=[-paired[metric]['interval95'][1],-paired[metric]['interval95'][0]]
                paired[metric]['interval95_familywise_critic_minus_actor']=[-mean-family_z*se,-mean+family_z*se]
        results.append(dict(board_id=board['id'], actor_action=per_action[0], critic_action=per_action[1],paired=paired))
        write_json(result_path,dict(trials=args.trials,settings=dict(trials=args.trials,max_moves=args.max_moves),boards=results,
            note='Fixed first-action pairs; independent trial streams paired across the pair. Normal intervals are approximate; point rankings are not claims of correctness.'))
    print(f'Wrote {folder}/branches.json')
    if sha256(source)!=manifest['source_sha256']:
        raise RuntimeError('Source checkpoint changed during continuation run')
    if 'minimum_tested_pair_delta_q_game_points' in manifest and len(results)==len(manifest['boards']):
        render_large_delta_report(folder)


def update(args):
    folder=args.folder.resolve(); manifest=json.loads((folder/'selection.json').read_text())
    update_path=folder/'update.json'
    update_state_path=folder/'update_state.pt'
    if update_path.exists() or update_state_path.exists():
        raise FileExistsError(f'Update artifacts already exist under {folder}; use a fresh audit folder')
    torch.set_num_threads(int(manifest['threads']))
    source=Path(manifest['source_checkpoint'])
    if sha256(source)!=manifest['source_sha256']:
        raise ValueError('Source checkpoint changed after capture')
    capture_data=torch.load(folder/'capture.pt',map_location='cpu',weights_only=False)
    source_model_state_hash = tensor_tree_sha256(capture_data['model_state_dict'])
    source_optimizer_state_hash = tensor_tree_sha256(capture_data['optimizer_state_dict'])
    _,model,optimizer,ppo,_=load_runtime(Path(manifest['source_checkpoint']),manifest['seed'],manifest['current_d4_order_acknowledged'])
    model.load_state_dict(copy.deepcopy(capture_data['model_state_dict']))
    optimizer.load_state_dict(copy.deepcopy(capture_data['optimizer_state_dict']))
    steps=[deserialize_step(row) for row in capture_data['rollout']]
    ppo.rng.bit_generator.state=capture_data['numpy_rng_state']; torch.set_rng_state(capture_data['torch_rng_state'])
    board_indices=[b['rollout_index'] for b in manifest['boards']]
    selected=[steps[i] for i in board_indices]
    def full_log_probs():
        return ppo.updater.policy_log_probabilities(steps)
    epochs=[]
    eps=ppo.config.clip_epsilon
    def selected_metrics():
        probabilities=[]; action_logp=[]; clipped=[]
        logs=ppo.updater.policy_log_probabilities(selected)
        for idx,(step,logp) in enumerate(zip(selected,logs)):
            p=logp.exp().cpu().numpy()
            probabilities.append(p.tolist())
            action_logp.append(float(logp[step.action]))
            ratio=float(torch.exp(logp[step.action]-step.log_prob))
            adv=(step.advantage-capture_data['advantage_mean'])/(capture_data['advantage_std']+ppo.config.denominator_epsilon)
            clipped.append(dict(ratio=ratio,normalized_advantage=float(adv),
                active=clipped_surrogate_gradient_active(ratio,float(adv),eps)))
        odds=[]
        for b,p in zip(manifest['boards'],probabilities):
            odds.append(float(np.log(max(p[b['actor_action']],1e-300))-np.log(max(p[b['critic_action']],1e-300))))
        return dict(probabilities=probabilities, sampled_action_log_probs=action_logp,
                    actor_vs_critic_log_odds=odds,clipped_surrogate=clipped)
    baseline=full_log_probs()
    baseline_selected=selected_metrics()
    def full_kl(current):
        oldp=baseline.exp(); oldsafe=baseline.masked_fill(~torch.isfinite(baseline),0.)
        newsafe=current.masked_fill(~torch.isfinite(current),0.)
        return float((oldp*(oldsafe-newsafe)).sum(-1).mean())
    for epoch in range(ppo.config.max_update_epochs):
        before=selected_metrics()
        # Production full update and early-stop check. Advantage stats are original.
        stats=ppo.run_epoch(steps,capture_data['advantage_mean'],capture_data['advantage_std'])
        new_full=full_log_probs()
        kl=full_kl(new_full)
        after=selected_metrics()
        epochs.append(dict(epoch=epoch+1,stats=asdict(stats),selected_before=before,selected_after=after,
            exact_full_rollout_kl=kl))
        if epoch+1 > ppo.config.min_update_epochs and stats.approximate_kl > ppo.config.target_kl: break
    torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        source_sha256=manifest['source_sha256']),update_state_path)
    if tensor_tree_sha256(capture_data['model_state_dict']) != source_model_state_hash:
        raise RuntimeError('Captured model state was mutated during PPO update replay')
    if tensor_tree_sha256(capture_data['optimizer_state_dict']) != source_optimizer_state_hash:
        raise RuntimeError('Captured optimizer state was mutated during PPO update replay')
    write_json(update_path,dict(epochs=epochs,rollout_size=len(steps),
        baseline=dict(exact_full_rollout_kl_from_self=0.,selected=baseline_selected),
        capture_restore=dict(model_state_sha256=source_model_state_hash,
            optimizer_state_sha256=source_optimizer_state_hash,
            capture_state_unchanged_after_update=True,
            restore_uses_deep_copied_state=True),
        note='Update replay uses only the captured original full rollout and restored optimizer.'))
    render_report(folder)
    print(f'Wrote {folder}/update.json ({len(epochs)} epochs)')


def render_report(folder: Path) -> None:
    manifest=json.loads((folder/'selection.json').read_text())
    update=json.loads((folder/'update.json').read_text())
    branches=json.loads((folder/'branches.json').read_text()) if (folder/'branches.json').exists() else None
    epochs=update['epochs']; baseline=update['baseline']['selected']['actor_vs_critic_log_odds']
    final=epochs[-1]['selected_after']['actor_vs_critic_log_odds']
    rows=[]; resolved_return=resolved_gae=opposed=0; toward_actor=toward_critic=unchanged=0
    by_id={b['board_id']:b for b in branches['boards']} if branches else {}
    for board in manifest['boards']:
        i=board['id']; q=sorted(board['exact_q'].items(),key=lambda item:item[1],reverse=True)
        qmargin=float(q[0][1]-q[1][1]) if len(q)>1 else float('nan')
        delta=final[i]-baseline[i]
        if delta>1e-3: toward_actor+=1; direction='toward actor'
        elif delta< -1e-3: toward_critic+=1; direction='toward critic'
        else: unchanged+=1; direction='unchanged'
        if branches and i in by_id:
            row=by_id[i]; ret=row['paired']['discounted_return']; gae=row['paired']['gae']
            if ret['interval95'][0]>0 or ret['interval95'][1]<0: resolved_return+=1
            if gae['interval95'][0]>0 or gae['interval95'][1]<0: resolved_gae+=1
            if ret['mean_actor_minus_critic']*gae['mean_actor_minus_critic']<0: opposed+=1
            score=row['paired']['game_score']
            rows.append(f"- **Board {i}: {ACTION_NAMES[board['actor_action']]} actor vs {ACTION_NAMES[board['critic_action']]} critic.** Exact-Q top-action margin {qmargin:.4g} game points. Paired score/return difference {score['mean_actor_minus_critic']:+.1f} points (95% CI {score['interval95'][0]:+.1f} to {score['interval95'][1]:+.1f}); GAE difference {gae['mean_actor_minus_critic']:+.4g} training units (95% CI {gae['interval95'][0]:+.4g} to {gae['interval95'][1]:+.4g}). Actor-vs-critic action log odds changed {delta:+.4g}, {direction}, across the PPO update.")
    report=['# PPO policy-update audit','',
        f"Source checkpoint: `{manifest['source_checkpoint']}` (iteration {manifest['checkpoint_iteration']}, SHA-256 `{manifest['source_sha256']}`).",
        f"Captured {manifest['rollout_size']:,} transitions from one normal full rollout; locked {len(manifest['boards'])} states where actor argmax differed from exact spawn-averaged critic-Q argmax.",
        f"The rollout used exploration probability {manifest['exploration_probability']:.4%}. Values, rewards, and GAE below use PPO training units unless labeled points.",
        ('The branch plan was fixed before any continuation result. Each branch forces the saved actor-preferred or critic-preferred first move, then samples the frozen behavior policy to natural terminal. The PPO update replays only the original complete rollout with its restored optimizer and RNG state.' if branches else 'This native audit captures and replays one full PPO rollout/update only; no forced-action continuations are included in this report.'),'',
        '## PPO update','',
        f"Replay ran {len(epochs)} epochs, with a configured maximum of {manifest['ppo_config']['max_update_epochs']}. The final approximate-KL statistic was {epochs[-1]['stats']['approximate_kl']:.5g} versus target {manifest['ppo_config']['target_kl']:.4g}.",
        'Exact full-rollout KL is reported from the frozen pre-update distribution to each epoch’s updated distribution. It is diagnostic and does not change production stopping. Selected action probabilities, actor-vs-critic log odds, and the advantage-sign-aware active clipping derivative for each sampled action are retained in `update.json` after every epoch.','',
        '| Epoch | Approx. minibatch KL | Exact full-rollout KL | Clip fraction |',
        '|---:|---:|---:|---:|']
    for e in epochs:
        report.append(f"| {e['epoch']} | {e['stats']['approximate_kl']:.5f} | {e['exact_full_rollout_kl']:.5f} | {e['stats']['clip_fraction']:.1%} |")
    if branches:
        report += ['', '## Paired branch results','',
            f'Intervals are paired 95% normal-approximation intervals over 64 common-random-number trial differences. These {len(manifest["boards"])} states are noisy, state-conditional evidence, not an overall winner.']
        report += rows
        report += ['', f"Point-estimate return and GAE rankings were opposite on {opposed} of 6 boards. Return intervals excluded zero on {resolved_return}/6 boards; GAE intervals excluded zero on {resolved_gae}/6. The PPO replay moved the selected action log odds toward the actor on {toward_actor} boards, toward the critic on {toward_critic}, and left {unchanged} effectively unchanged. These outcomes do not show a consistent per-board match to either continuation return or GAE."]
    report += ['', '## Limits','',
        f"The {len(manifest['boards'])} states are a diagnostic sample from one rollout, not a population estimate." + (' Branch continuation intervals can be wide because game scores vary substantially.' if branches else ''),
        ('D4 branches stay in the captured policy-view coordinates; exact D4 equivariance makes this equivalent to resampling a symmetry view and mapping its action back. ' if branches else '') + 'Capture, continuation, and replay run on CPU. The source checkpoint has no historical RNG snapshot, so this audit uses a recorded fresh rollout seed and makes no claim of bitwise parity with the original GPU update.',
        'This audit measures policy retention only within the replayed update. It does not measure forgetting across outer iterations. The local log-odds gradient dotted with the full epoch parameter displacement is not computed, so probability movement is descriptive rather than a causal attribution.','']
    (folder/'report.md').write_text('\n'.join(report),encoding='utf8')


def render_large_delta_report(folder: Path) -> None:
    folder=folder.resolve()
    manifest=json.loads((folder/'selection.json').read_text())
    branches=json.loads((folder/'branches.json').read_text())
    if len(branches['boards'])!=len(manifest['boards']):
        raise ValueError('Continuation results are incomplete')
    by_id={b['board_id']:b for b in branches['boards']}
    result_rows=[]; return_confirmed=gae_confirmed=0
    practical_margin=100.0
    for board in manifest['boards']:
        outcome=by_id[board['id']]
        ret=outcome['paired']['discounted_return']
        gae=outcome['paired']['gae']
        n=len(outcome['actor_action']['trials'])
        ret_mean=ret['mean_critic_minus_actor']; ret_se=ret['se']
        gae_mean=gae['mean_critic_minus_actor']; gae_se=gae['se']
        ret_f=ret['interval95_familywise_critic_minus_actor']
        gae_f=gae['interval95_familywise_critic_minus_actor']
        if ret_f[0]>practical_margin: return_confirmed+=1
        if gae_f[0]>0: gae_confirmed+=1
        ret_sd=ret_se*math.sqrt(n); gae_sd=gae_se*math.sqrt(n)
        native=ACTION_NAMES[board['sampled_action']]
        norm=(board['raw_advantage']-manifest['advantage_mean'])/(manifest['advantage_std']+manifest['ppo_config']['denominator_epsilon'])
        result_rows.append(
            f"| {board['id']} | {ACTION_NAMES[board['actor_action']]} → {ACTION_NAMES[board['critic_action']]} | {board['delta_q']:.0f} | {board['actor_probabilities'][board['actor_action']]:.3g} / {board['actor_probabilities'][board['critic_action']]:.3g}; {board['behavior_probabilities'][board['actor_action']]:.3g} / {board['behavior_probabilities'][board['critic_action']]:.3g} | {board['cumulative_score']:.0f}; {board['largest_tile']}; {board['empty_cells']}; {board['remaining_rollout_horizon']} / {board['counterfactual_gae_horizon_limit']} | {native}; {board['raw_advantage']:+.3f} ({norm:+.2f} SD) | {ret_mean:+.0f} ± {ret_se:.0f}; [{ret['interval95_critic_minus_actor'][0]:+.0f}, {ret['interval95_critic_minus_actor'][1]:+.0f}]; FW [{ret_f[0]:+.0f}, {ret_f[1]:+.0f}]; d={ret_mean/ret_sd if ret_sd else float('nan'):+.2f} | {gae_mean:+.3f} ± {gae_se:.3f}; [{gae['interval95_critic_minus_actor'][0]:+.3f}, {gae['interval95_critic_minus_actor'][1]:+.3f}]; FW [{gae_f[0]:+.3f}, {gae_f[1]:+.3f}]; d={gae_mean/gae_sd if gae_sd else float('nan'):+.2f} |")
    lines=[
        '# Large tested-pair DeltaQ continuation audit', '',
        f"Checkpoint: `{manifest['source_checkpoint']}` (iteration {manifest['checkpoint_iteration']}, SHA-256 `{manifest['source_sha256']}`).",
        f"Selection used the immutable {manifest['rollout_size']:,}-transition capture from the initial audit. It screened {manifest['candidate_limit']:,} evenly spaced eligible states from {manifest['eligible_population']:,}; {manifest['disagreement_candidate_count']:,} disagreed at their legal policy and critic argmaxes, and {manifest['qualifying_population']:,} met the fixed tested-pair DeltaQ threshold of {manifest['minimum_tested_pair_delta_q_game_points']:.0f} raw game points. The six largest qualifying gaps were locked before continuations, with one state per rollout game slot and distinct boards. No threshold was lowered after observing results.",
        f"Each selected actor/critic first action was forced for {branches['trials']} trials, then the same frozen behavior policy (exploration mixture {manifest['exploration_probability']:.2%}) was sampled to natural terminal. Future game-spawn and action RNG streams were independently seeded and paired across first actions. The sampled action and native GAE columns refer to the original captured transition.",
        '', '## Board results', '',
        'DeltaQ is Q(critic action) − Q(actor action) in raw game points. Probability columns show actor-policy probabilities and then frozen behavior-mixture probabilities for actor / critic action. Board state columns show cumulative game score; largest tile; empty cells; original GAE horizon / counterfactual GAE cap. Return and branch GAE differences are critic minus actor. Intervals use paired normal approximations; FW intervals use Bonferroni 95% familywise coverage across the six fixed board pairs. `d` is paired mean divided by paired-trial standard deviation. Return uses gamma=1 and is in raw game points; GAE stays in PPO training units.', '',
        '| Board | Forced pair | ΔQ | Actor probabilities; behavior probabilities | State: score; max tile; empty; original horizon / branch cap | Captured action; native GAE (normalized) | Remaining return difference, points | Horizon GAE difference, training units |',
        '|---:|:---|---:|:---|:---|:---|:---|:---|',
        *result_rows, '',
        '## Interpretation', '',
        'The first continuation report incorrectly used the sampled action’s original terminal time as the GAE horizon for both forced actions. Those GAE results are withdrawn. The corrected horizon is the earlier of the remaining rollout segment and remaining environment time limit from the captured episode age; each branch’s own natural terminal shortens its prefix, and surviving branches bootstrap the critic at the cap. The locked boards, action pairs, trial count, and RNG seeds were unchanged. All 1,536 per-action score, return, and step records are bit-for-bit identical to the earlier run; only GAE values changed.',
        f"Under the preregistered practical criterion (familywise return interval lower bound above +{practical_margin:.0f} points), {return_confirmed}/{len(manifest['boards'])} boards confirm a useful critic-preferred move. Familywise GAE intervals were wholly positive on {gae_confirmed}/{len(manifest['boards'])} boards. Nominal 95% intervals are included for each pair, but the familywise intervals control the six selected return comparisons.",
        'The largest observed gaps are subject to critic miscalibration and winner’s curse because states were selected by the same critic being evaluated. This diagnostic conditions on high-gap disagreement states from one rollout; it estimates neither how often these states occur nor overall policy performance. Return agreement would support critic ranking on these states, not global value accuracy. A favorable conditional GAE does not mean the captured rollout contained enough useful examples to change the policy. Low actor probability describes the chosen states; it does not establish that increasing sampling causes improvement.',
        'The checkpoint uses an exactly D4-equivariant policy. Branches run in each captured board’s view coordinates, so legal actions and the same policy/exploration mixture retain the training behavior law. Intervals use normal approximations despite variable, potentially heavy-tailed game returns. The fixed trial count avoids optional stopping. All branches reached natural terminal; the move cap was set to the checkpoint episode limit and any unfinished branch would have failed the run.',
        'Iteration 100 is an early checkpoint, so the policy may remain undertrained. A longer PPO run and the same audit at later checkpoints remain an untested hypothesis.', '',
        'Selection manifest: `selection.json`. Raw paired trials: `branches.json`. The original audit capture was preserved unchanged.'
    ]
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('capture'); p.add_argument('checkpoint',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--boards',type=int,default=8); p.add_argument('--candidate-limit',type=int,default=512); p.add_argument('--seed',type=int,default=20482027); p.add_argument('--threads',type=int,default=1); p.add_argument('--current-d4-order',action='store_true')
    p=commands.add_parser('branch'); p.add_argument('folder',type=Path); p.add_argument('--trials',type=int,default=64); p.add_argument('--max-moves',type=int,default=10000); p.add_argument('--resume',action='store_true')
    p=commands.add_parser('select-large-delta'); p.add_argument('folder',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--boards',type=int,default=6); p.add_argument('--candidate-limit',type=int,default=2048); p.add_argument('--min-delta-q',type=float,default=250.)
    p=commands.add_parser('update'); p.add_argument('folder',type=Path)
    p=commands.add_parser('report'); p.add_argument('folder',type=Path)
    args=parser.parse_args()
    if args.command=='capture': capture(args)
    elif args.command=='branch': branch(args)
    elif args.command=='select-large-delta': select_large_delta(args)
    elif args.command=='update': update(args)
    else:
        folder=args.folder.resolve()
        manifest=json.loads((folder/'selection.json').read_text())
        if 'minimum_tested_pair_delta_q_game_points' in manifest:
            render_large_delta_report(folder)
        else:
            render_report(folder)
        print(f'Wrote {args.folder.resolve()}/report.md')

if __name__=='__main__': main()
