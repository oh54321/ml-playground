"""Capture and dissect one PPO update without changing a training run.

Example: python -m games.rl_2048.debug_learning RUN/best.pt --output OUTPUT
Continuation returns are undiscounted remaining game scores, not final scores.
Every continuation runs to termination; hitting the safety cap is an error.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from algorithms.ppo import PPO, PPOConfig
from games.rl_2048.legacy_agent import model_from_legacy_config
from games.rl_2048.environment.game_2048 import Game2048, State2048


def make_ppo(checkpoint, seed):
    config = dict(checkpoint['ppo_config'])
    if 'update_epochs' in config:
        config['min_update_epochs'] = config['max_update_epochs'] = config.pop('update_epochs')
        config['target_kl'] = .02
    config.update(seed=seed, print_timestamps=False)
    model = model_from_legacy_config(checkpoint['resnet_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state_dict']))
    return PPO(model, optimizer, PPOConfig(**config), lambda seed: Game2048(seed=seed))


@torch.no_grad()
def predictions(ppo, steps):
    logs, values = [], []
    for i in range(0, len(steps), 1024):
        batch = steps[i:i + 1024]
        output = ppo.model(ppo._state_batch_tensor([s.state for s in batch]))
        mask = torch.stack([s.legal_actions_mask for s in batch])
        logs.append(output.action_logits.masked_fill(mask, -torch.inf).log_softmax(-1))
        values.append(ppo._untransform_predicted_value(output.value))
    return torch.cat(logs), torch.cat(values)


def metrics(ppo, steps, old_logs, targets, initial_parameters):
    logs, values = predictions(ppo, steps)
    probs = logs.exp()
    old_probs = old_logs.exp()
    safe_delta = (old_logs - logs).masked_fill(~torch.isfinite(old_logs), 0)
    kl = (old_probs * safe_delta).sum(-1)
    actions = torch.tensor([s.action for s in steps])
    ratio = (logs - old_logs)[torch.arange(len(steps)), actions].exp()
    errors = targets - values
    parameters = torch.nn.utils.parameters_to_vector(ppo.model.parameters()).detach()
    return dict(full_kl=kl.mean().item(), kl_p95=kl.quantile(.95).item(),
                clip_fraction=((ratio - 1).abs() > ppo.config.clip_epsilon).float().mean().item(),
                entropy=-(probs * logs.masked_fill(~torch.isfinite(logs), 0)).sum(-1).mean().item(),
                explained_variance=(1 - errors.var() / targets.var()).item(),
                value_rmse=errors.square().mean().sqrt().item(),
                relative_update=((parameters - initial_parameters).norm() / initial_parameters.norm()).item())


def select_cases(steps, logs):
    """Prespecified strata, selected without looking at continuation outcomes."""
    eligible = [i for i, s in enumerate(steps) if (~s.legal_actions_mask).sum() >= 2]
    chosen, reasons = [], []
    def add(candidates, reason):
        for i in candidates:
            if all(not np.array_equal(steps[i].state.grid, steps[j].state.grid) for j in chosen):
                chosen.append(i)
                reasons.append(reason)
                return
    add(sorted(eligible, key=lambda i: int((steps[i].state.grid == 0).sum())), 'fewest empty cells')
    confident = [i for i in eligible if logs[i].exp().max() >= .8]
    add(sorted(confident or eligible, key=lambda i: steps[i].advantage), 'confident policy, most negative observed GAE')
    add(sorted(eligible, key=lambda i: -steps[i].advantage), 'most positive observed GAE')
    for fraction in (.25, .5, .75):
        target = eligible[int(fraction * (len(eligible) - 1))]
        add(sorted(eligible, key=lambda i: abs(i - target)), f'rollout position {fraction}')
    return [(i, reason) for i, reason in zip(chosen, reasons)]


@torch.no_grad()
def continuations(ppo, cases, replicas, seed, forced=False, greedy=False):
    """Common random-number streams across actions/policies; never reuse training RNG.

    Each game gets separate spawn and action RNGs. Equal seeds pair trials, but
    different boards can still map a spawn draw to different cells.
    """
    games, action_rngs, keys = [], [], []
    scores = {}
    for case_id, step in enumerate(cases):
        actions = torch.where(~step.legal_actions_mask)[0].tolist() if forced else [None]
        for action in actions:
            scores[(case_id, action)] = np.zeros(replicas)
            for rep in range(replicas):
                spawn_seed = seed + case_id * 100000 + rep
                game = Game2048(seed=spawn_seed)
                game.state = step.state.copy()
                game.total_score, game.done = 0, False
                game.set_seed(spawn_seed)
                if action is not None:
                    assert game.step(action).action_applied
                games.append(game)
                action_rngs.append(np.random.default_rng(spawn_seed + 50000000))
                keys.append((case_id, action, rep))
    active = [i for i, g in enumerate(games) if not g.done]
    for _ in range(10000):
        if not active:
            break
        for start in range(0, len(active), 256):
            ids = active[start:start + 256]
            batch = [games[i] for i in ids]
            output = ppo.model(ppo._state_batch_tensor([g.state for g in batch]))
            mask = torch.ones_like(output.action_logits, dtype=torch.bool)
            for row, game in enumerate(batch):
                mask[row, game.legal_actions()] = False
            probs = output.action_logits.masked_fill(mask, -torch.inf).softmax(-1).numpy()
            for row, i in enumerate(ids):
                cumulative = probs[row].astype(np.float64).cumsum()
                cumulative /= cumulative[-1]
                action = int(probs[row].argmax()) if greedy else int(np.searchsorted(cumulative, action_rngs[i].random(), side='right'))
                assert games[i].step(action).action_applied
        active = [i for i in active if not games[i].done]
    if active:
        raise RuntimeError('Continuation exceeded 10,000 moves; refusing truncated returns')
    for game, (case_id, action, rep) in zip(games, keys):
        scores[(case_id, action)][rep] = game.total_score
    return scores


def estimate(samples):
    samples = np.asarray(samples)
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    return dict(mean=mean, se=se, ci95=[mean - 1.96 * se, mean + 1.96 * se], n=len(samples))


@torch.no_grad()
def one_step_branches(ppo, board):
    """Enumerate all next spawns: exact immediate death risk and critic lookahead.

    Lookahead includes a learned value, so is not an exact long-term return.
    """
    game = Game2048(seed=0)
    game.state = State2048(grid=np.asarray(board))
    branches = {}
    for action in game.legal_actions():
        grid, reward, moved = game._simulate_move(game.state.grid, action)
        assert moved
        empty = np.argwhere(grid == 0)
        outcomes = []
        for y, x in empty:
            for tile, probability in ((2, .9), (4, .1)):
                successor = grid.copy()
                successor[y, x] = tile
                game_after = Game2048(seed=0)
                game_after.state = State2048(grid=successor)
                done = not game_after.legal_actions()
                value = 0 if done else ppo._predict_value(game_after.state) / ppo.config.reward_multiplier
                outcomes.append(dict(board=successor.tolist(), probability=probability / len(empty),
                                     terminal=done, critic_value_score_units=value))
        branches[str(action)] = dict(immediate_reward=int(reward),
            terminal_probability=sum(o['probability'] for o in outcomes if o['terminal']),
            critic_one_step_q=reward + ppo.config.gamma * sum(o['probability'] * o['critic_value_score_units'] for o in outcomes),
            outcomes=outcomes)
    return branches


def gradient_conflict(ppo, rollout, mean, std, index):
    """First-order change of a sampled action's log probability under plain GD.

    All gradients are evaluated at the frozen starting model. Positive means
    increasing that action probability. These are not predicted Adam updates.
    """
    step = rollout[index]
    parameters = tuple(ppo.model.parameters())
    logits = ppo.model(step.state.to_tensor()[None]).action_logits[0]
    logp = logits.masked_fill(step.legal_actions_mask, -torch.inf).log_softmax(0)[step.action]
    logp_gradients = torch.autograd.grad(logp, parameters, allow_unused=True)
    directions = dict(policy=0., weighted_value=0.)
    for start in range(0, len(rollout), ppo.config.mini_batch_size):
        batch = rollout[start:start + ppo.config.mini_batch_size]
        objective = ppo._compute_objective(batch, mean, std)
        for name, loss in [('policy', objective.policy.mean()),
                           ('weighted_value', ppo.config.value_coefficient * objective.value.mean())]:
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=name == 'policy')
            dot = sum(float((a * b).sum()) for a, b in zip(logp_gradients, gradients) if a is not None and b is not None)
            directions[name] -= dot * len(batch) / len(rollout)
    local_objective = ppo._compute_objective([step], mean, std).policy.mean()
    local_gradients = torch.autograd.grad(local_objective, parameters, allow_unused=True)
    directions['single_example_policy'] = -sum(float((a * b).sum()) for a, b in zip(logp_gradients, local_gradients) if a is not None and b is not None)
    return dict(rollout_index=index, action=step.action, normalized_advantage=(step.advantage - mean) / (std + ppo.config.denominator_epsilon),
                log_probability_direction_under_negative_gradient=directions)


def update_epoch(ppo, rollout, mean, std, mode):
    # Same shuffle order and optimizer state for each intervention.
    original = ppo.config
    if mode == 'combined':
        return ppo.run_epoch(rollout, mean, std)
    shuffled = ppo.rng.permutation(rollout)
    for start in range(0, len(shuffled), original.mini_batch_size):
        obj = ppo._compute_objective(shuffled[start:start + original.mini_batch_size], mean, std)
        loss = obj.policy.mean() if mode == 'policy_only' else original.value_coefficient * obj.value.mean()
        ppo.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if original.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(ppo.model.parameters(), original.gradient_clip_norm)
        ppo.optimizer.step()
    return None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=830217)
    parser.add_argument('--replicas', type=int, default=32)
    parser.add_argument('--evaluation-replicas', type=int, default=16)
    parser.add_argument('--confirm-replicas', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torch.save(checkpoint, args.output / 'checkpoint_snapshot.pt')
    ppo = make_ppo(checkpoint, args.seed)
    print(f'Checkpoint iteration {checkpoint["outer_iteration"]}: collecting fresh rollout', flush=True)
    rollout, mean, std = ppo.collect_batched_information()
    shuffle_state = copy.deepcopy(ppo.rng.bit_generator.state)
    torch.save(dict(rollout=rollout, mean=mean, std=std, shuffle_state=shuffle_state,
                    torch_rng=torch.get_rng_state(), seed=args.seed), args.output / 'rollout.pt')
    old_logs, _ = predictions(ppo, rollout)
    targets = torch.tensor([s.value + s.advantage for s in rollout])
    initial_parameters = torch.nn.utils.parameters_to_vector(ppo.model.parameters()).detach().clone()
    old_sampled = old_logs[torch.arange(len(rollout)), torch.tensor([s.action for s in rollout])]
    stored = torch.tensor([s.log_prob for s in rollout])
    max_log_error = (stored - old_sampled).abs().max().item()
    assert max_log_error < 1e-4, f'Rollout log probability mismatch: {max_log_error}'
    terminal = [s for s in rollout if s.done]
    terminal_error = max(abs(s.advantage - (s.reward - s.value)) for s in terminal)
    assert terminal_error < 1e-5
    selected = select_cases(rollout, old_logs)
    cases = [rollout[i] for i, _ in selected]
    case_logs, case_values = predictions(ppo, cases)
    report = dict(checkpoint=str(args.checkpoint.resolve()), iteration=checkpoint['outer_iteration'],
                  checkpoint_evaluation=checkpoint['evaluation'], config=dataclasses.asdict(ppo.config),
                  seed=args.seed, rollout_size=len(rollout), advantage_mean=mean, advantage_std=std,
                  checks=dict(max_log_probability_error=max_log_error, terminal_gae_error=terminal_error),
                  cases=[], epochs=[], ablations=[])
    print('Estimating every legal action at six rollout boards', flush=True)
    branches = continuations(ppo, cases, args.replicas, 910000, forced=True)
    for case_id, ((index, reason), step) in enumerate(zip(selected, cases)):
        probs = case_logs[case_id].exp().numpy()
        actions = torch.where(~step.legal_actions_mask)[0].tolist()
        q_samples = {a: branches[(case_id, a)] for a in actions}
        v_samples = sum(probs[a] * q_samples[a] for a in actions)
        greedy = int(probs.argmax())
        best = max(actions, key=lambda a: q_samples[a].mean())
        report['cases'].append(dict(id=case_id, rollout_index=index, selection=reason,
            board=step.state.grid.tolist(), probabilities=probs.tolist(), sampled_action=step.action,
            greedy_action=greedy, best_estimated_action=best, raw_value=float(case_values[case_id]),
            value_score_units=float(case_values[case_id]) / ppo.config.reward_multiplier,
            observed_gae=step.advantage, normalized_gae=(step.advantage - mean) / (std + ppo.config.denominator_epsilon),
            mc_value=estimate(v_samples),
            mc_sampled_advantage=estimate(q_samples[step.action] - v_samples),
            greedy_gap=estimate(q_samples[best] - q_samples[greedy]),
            actions={str(a):dict(**estimate(q_samples[a]), samples=q_samples[a].tolist()) for a in actions}))
    write_json(args.output / 'results.json', report)
    # Confirm the two largest observed gaps with independent trials to reduce selection bias.
    chosen = sorted(range(len(cases)), key=lambda i: report['cases'][i]['greedy_gap']['mean'], reverse=True)[:2]
    print(f'Independent confirmation for cases {chosen}', flush=True)
    confirmation = continuations(ppo, [cases[i] for i in chosen], args.confirm_replicas, 1910000, forced=True)
    for local, case_id in enumerate(chosen):
        case = report['cases'][case_id]
        best, greedy = case['best_estimated_action'], case['greedy_action']
        actions = [int(a) for a in case['actions']]
        samples = {a:confirmation[(local, a)] for a in actions}
        v = sum(case['probabilities'][a] * samples[a] for a in actions)
        case['confirmation'] = dict(greedy_gap=estimate(samples[best] - samples[greedy]),
            mc_value=estimate(v), mc_sampled_advantage=estimate(samples[case['sampled_action']] - v),
            actions={str(a):dict(**estimate(samples[a]), samples=samples[a].tolist()) for a in actions})
    baseline_returns = {}
    for greedy in (False, True):
        baseline_returns[greedy] = continuations(ppo, cases, args.evaluation_replicas, 2910000, greedy=greedy)

    def evaluate(epoch, mode, during=None):
        result = dict(epoch=epoch, mode=mode, **metrics(ppo, rollout, old_logs, targets, initial_parameters))
        if during is not None:
            result['during_epoch_approx_kl'] = during.approximate_kl
        logs, values = predictions(ppo, cases)
        result['case_probabilities'] = logs.exp().tolist()
        result['case_values'] = values.tolist()
        for greedy in (False, True):
            returns = baseline_returns[greedy] if epoch == 0 else continuations(ppo, cases, args.evaluation_replicas, 2910000, greedy=greedy)
            label = 'greedy' if greedy else 'sampled'
            result[label + '_returns'] = [returns[(i, None)].tolist() for i in range(len(cases))]
            result[label + '_per_case_delta'] = [estimate(returns[(i, None)] - baseline_returns[greedy][(i, None)]) for i in range(len(cases))]
            result[label + '_mean_return'] = float(np.mean([returns[(i, None)].mean() for i in range(len(cases))]))
        return result

    report['epochs'].append(evaluate(0, 'combined'))
    ppo.rng.bit_generator.state = copy.deepcopy(shuffle_state)
    stop_epoch = None
    for epoch in range(1, ppo.config.max_update_epochs + 1):
        print(f'Combined update epoch {epoch}; evaluating exact metrics and continuations', flush=True)
        during = update_epoch(ppo, rollout, mean, std, 'combined')
        row = evaluate(epoch, 'combined', during)
        if stop_epoch is None and epoch > ppo.config.min_update_epochs and during.approximate_kl > ppo.config.target_kl:
            stop_epoch = epoch
        row['beyond_production_stop'] = stop_epoch is not None and epoch > stop_epoch
        report['epochs'].append(row)
        report['production_stop_epoch'] = stop_epoch
        write_json(args.output / 'results.json', report)
        print(f'epoch={epoch}, full KL={row["full_kl"]:.5f}, sampled return={row["sampled_mean_return"]:.1f}', flush=True)
    for mode in ('policy_only', 'value_only'):
        print(f'One-epoch intervention: {mode}', flush=True)
        ppo = make_ppo(checkpoint, args.seed)
        ppo.rng.bit_generator.state = copy.deepcopy(shuffle_state)
        update_epoch(ppo, rollout, mean, std, mode)
        report['ablations'].append(evaluate(1, mode))
        write_json(args.output / 'results.json', report)
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
