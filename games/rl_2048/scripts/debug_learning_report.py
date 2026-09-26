"""Render a captured learning-debug experiment as Markdown and ordinary PNGs."""
import argparse
import json
import math
from pathlib import Path

from games.rl_2048.utils.plot_training_diagnostics import _font, save_line_plot


NAMES = ['left', 'up', 'right', 'down']
COLORS = [(31,119,180), (214,39,40), (44,160,44), (148,103,189)]


def uncertainty(result):
    return f"{result['mean']:,.0f} (approx. 95% interval {result['ci95'][0]:,.0f} to {result['ci95'][1]:,.0f}; n={result['n']})"


def aggregate_delta(row, mode):
    # Conditional on the six fixed cases: combine independent per-case paired estimates.
    estimates = row[mode + '_per_case_delta']
    mean = sum(e['mean'] for e in estimates) / len(estimates)
    se = math.sqrt(sum(e['se'] ** 2 for e in estimates)) / len(estimates)
    return f"{mean:+,.0f} [{mean - 1.96 * se:+,.0f}, {mean + 1.96 * se:+,.0f}]"


def prepare(folder, comparison_checkpoint=None, gradient_case=None):
    """Run in the training environment; rendering only needs Pillow."""
    import torch
    from games.rl_2048.scripts.debug_learning import make_ppo, one_step_branches, predictions, gradient_conflict
    report = json.loads((folder / 'results.json').read_text())
    torch.set_num_threads(1)
    ppo = make_ppo(torch.load(folder / 'checkpoint_snapshot.pt', map_location='cpu', weights_only=False), report['seed'])
    one_step = {str(c['id']): one_step_branches(ppo, c['board']) for c in report['cases']}
    (folder / 'one_step_branches.json').write_text(json.dumps(one_step, indent=2))
    if gradient_case is not None:
        capture = torch.load(folder / 'rollout.pt', map_location='cpu', weights_only=False)
        index = report['cases'][gradient_case]['rollout_index']
        conflict = gradient_conflict(ppo, capture['rollout'], capture['mean'], capture['std'], index)
        conflict['case_id'] = gradient_case
        (folder / 'gradient_conflict.json').write_text(json.dumps(conflict, indent=2))
    if comparison_checkpoint is not None:
        checkpoint = torch.load(comparison_checkpoint, map_location='cpu', weights_only=False)
        comparison = make_ppo(checkpoint, report['seed'])
        rollout = torch.load(folder / 'rollout.pt', map_location='cpu', weights_only=False)['rollout']
        logs, values = predictions(comparison, [rollout[c['rollout_index']] for c in report['cases']])
        comparison_data = dict(checkpoint=str(comparison_checkpoint.resolve()), iteration=checkpoint['outer_iteration'],
            evaluation=checkpoint['evaluation'], cases=[dict(id=c['id'], probabilities=prob,
                value_score_units=v / comparison.config.reward_multiplier)
                for c, prob, v in zip(report['cases'], logs.exp().tolist(), values.tolist())])
        (folder / 'comparison.json').write_text(json.dumps(comparison_data, indent=2))


def render(folder):
    from PIL import Image, ImageDraw
    report = json.loads((folder / 'results.json').read_text())
    one_step = json.loads((folder / 'one_step_branches.json').read_text())
    comparison = json.loads((folder / 'comparison.json').read_text()) if (folder / 'comparison.json').exists() else None
    rows = report['epochs']
    for key, title in [('full_kl', 'Full rollout policy KL after each epoch'),
                       ('explained_variance', 'Fit to frozen rollout targets'),
                       ('sampled_mean_return', 'Sampled continuation score: six fixed boards'),
                       ('greedy_mean_return', 'Greedy continuation score: six fixed boards')]:
        save_line_plot(folder / f'{key}.png', [r['epoch'] for r in rows],
                       [(key, [r[key] for r in rows], COLORS[0])], title,
                       x_label='Completed update epochs (includes updates past stopping)', y_label=key)
    text = ["# One-rollout PPO investigation", "",
        f"Source: `{report['checkpoint']}`, iteration **{report['iteration']}**.", "",
        f"Saved-checkpoint evaluation: {report['checkpoint_evaluation']}. "
        f"Diagnostic rollout: {report['rollout_size']:,} transitions, seed {report['seed']}.", "",
        "## Scope and interpretation", "",
        "The checkpoint and Adam state were copied. Fresh environments were used; this is a controlled new update, "
        "not a reconstruction of the original training iteration. The captured rollout, seed, and shuffle state "
        "are saved in rollout.pt, and the model/optimizer snapshot is saved in checkpoint_snapshot.pt.", "",
        "Six boards were selected using fixed criteria, not continuation results. Every legal first action was "
        "followed by sampled actions from the frozen policy until game over. Values below are remaining scores "
        "in game points (training reward/value units divided by the reward multiplier). "
        "Two discovery gaps received independent confirmation. Intervals are normal-approximation Monte Carlo "
        "intervals, not a guarantee; discovery maxima are subject to selection bias.", "",
        "Epoch experiments reuse exactly the same rollout and shuffle stream. Full rollout metrics are measured "
        "after updates with the model fixed. Continuation trials use independent seeds from discovery/confirmation, "
        "with paired random streams across policy versions. These are six selected states, not a global policy "
        "evaluation. There are only 16 trials per state/mode by default; inspect per-state intervals in results.json.", "",
        "Policy-only and value-only interventions each start from the original model and Adam state, for one epoch. "
        "Adam momentum from previous training is retained, so they measure practical loss-removal interventions, "
        "not isolated raw-gradient effects. Entropy is omitted in both isolated interventions.", "",
        f"Production stopping would occur after epoch {report.get('production_stop_epoch') or report['config']['max_update_epochs']}. "
        "Later epochs are deliberate diagnostic counterfactuals. Production code is unchanged, including its "
        "strict > minimum-epoch condition.", "",
        f"Integrity checks: {report['checks']}.", "",
        "## Fixed-model metrics and continuation results", ""]
    for r in rows + report['ablations']:
        text.append(f"- {r['mode']}, epoch {r['epoch']}: full KL {r['full_kl']:.5f}; "
                    f"KL p95 {r['kl_p95']:.5f}; clip fraction {r['clip_fraction']:.3f}; "
                    f"EV {r['explained_variance']:.3f}; RMSE {r['value_rmse']:.3f}; "
                    f"entropy {r['entropy']:.3f}; cumulative parameter change {r['relative_update']:.6f}; "
                    f"sampled continuation {r['sampled_mean_return']:,.0f}; greedy continuation {r['greedy_mean_return']:,.0f}. "
                    f"Paired change versus baseline, approximate 95% intervals conditional on these six cases: "
                    f"sampled {aggregate_delta(r, 'sampled')}; greedy {aggregate_delta(r, 'greedy')}.")
    if (folder / 'gradient_conflict.json').exists():
        conflict = json.loads((folder / 'gradient_conflict.json').read_text())
        text += ['', f"Frozen-gradient test for case {conflict['case_id']}: {conflict['log_probability_direction_under_negative_gradient']}. "
                 "These numbers give the derivative of the sampled action's log probability along each negative "
                 "gradient direction, before any optimizer step. Positive means increasing probability. "
                 "They isolate cross-state gradient conflict from Adam; actual Adam steps also depend on momentum "
                 "and per-parameter scaling.", '']
    text += ['', '## Selected decisions', '']
    for c in report['cases']:
        i = c['id']
        image = Image.new('RGB', (640, 690), '#faf8ef')
        draw = ImageDraw.Draw(image)
        draw.text((20,12), f"Case {i}: rollout index {c['rollout_index']}", fill='#333333', font=_font(22))
        draw.text((20,45), f"Sampled: {NAMES[c['sampled_action']]} / greedy: {NAMES[c['greedy_action']]}", fill='#333333', font=_font(20))
        for y, row in enumerate(c['board']):
            for x, v in enumerate(row):
                box = (20+x*150, 85+y*150, 160+x*150, 225+y*150)
                color = '#cdc1b4' if v == 0 else '#eee4da' if v < 8 else '#f2b179' if v < 128 else '#edc850'
                draw.rounded_rectangle(box, radius=7, fill=color)
                draw.text((90+x*150,155+y*150), str(v) if v else '', anchor='mm', font=_font(30), fill='#333333')
        board_path = folder / f'case_{i}_board.png'
        image.save(board_path)
        save_line_plot(folder / f'case_{i}_probabilities.png', [r['epoch'] for r in rows],
            [(NAMES[a], [r['case_probabilities'][i][a] for r in rows], COLORS[a]) for a in range(4)],
            f'Case {i}: action probabilities after each epoch', x_label='Completed update epochs', y_label='Probability', y_min=0)
        text += [f"### Case {i}: {c['selection']}", '', f"![Board]({board_path.resolve()})", '',
            f"Sampled action: **{NAMES[c['sampled_action']]}**. Greedy action: **{NAMES[c['greedy_action']]}**. "
            f"Probabilities: {', '.join(f'{NAMES[a]} {p:.1%}' for a,p in enumerate(c['probabilities']))}.", '',
            f"Observed raw GAE: {c['observed_gae']:.3f}; normalized: {c['normalized_gae']:.3f}. "
            f"Predicted remaining score: {c['value_score_units']:,.0f}. "
            f"Monte Carlo remaining score: {uncertainty(c['mc_value'])}.", '',
            f"Monte Carlo sampled-action advantage: {uncertainty(c['mc_sampled_advantage'])} game points. "
            "A single noisy GAE sample can disagree with the mean without implying an implementation bug.", '']
        for a, result in c['actions'].items():
            branch = one_step[str(i)][a]
            text.append(f"- Force {NAMES[int(a)]}: {uncertainty(result)} remaining points. "
                        f"Exact immediate game-over probability: {branch['terminal_probability']:.0%}. "
                        f"One-step critic lookahead: {branch['critic_one_step_q']:,.0f} points.")
        if 'confirmation' in c:
            confirmation = c['confirmation']
            text += ['', f"Independent confirmation, preselected {NAMES[c['best_estimated_action']]} versus "
                f"greedy {NAMES[c['greedy_action']]}: difference {uncertainty(confirmation['greedy_gap'])}.",
                f"Confirmed sampled-action advantage: {uncertainty(confirmation['mc_sampled_advantage'])}; "
                f"confirmed value: {uncertainty(confirmation['mc_value'])}."]
        if comparison is not None:
            other = comparison['cases'][i]
            text += ['', f"Comparison checkpoint (iteration {comparison['iteration']}; different training history): "
                     + ', '.join(f'{NAMES[a]} {p:.1%}' for a,p in enumerate(other['probabilities']))
                     + f"; predicted remaining score {other['value_score_units']:,.0f}. "
                     "These states came from the larger policy, so this is not a matched game-score comparison."]
        text += ['', f"![Action probability evolution]({(folder / f'case_{i}_probabilities.png').resolve()})", '']
    text += ['## Files', '', 'Full numerical results and trial scores: results.json. '
             'These estimates evaluate the current continuation policy, not optimal play or a perfect action label.', '']
    (folder / 'report.md').write_text('\n'.join(text))
    print(folder / 'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--prepare', action='store_true', help='Enumerate spawns using the training environment')
    parser.add_argument('--comparison-checkpoint', type=Path)
    parser.add_argument('--gradient-case', type=int, help='Optional case ID for a full-rollout frozen-gradient test')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.folder, args.comparison_checkpoint, args.gradient_case)
    else:
        render(args.folder)
