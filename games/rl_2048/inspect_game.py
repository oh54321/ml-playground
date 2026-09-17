"""Replay one 2048 game while capturing PyTorch policy diagnostics.

This intentionally uses ordinary PyTorch forward hooks instead of a separate
interpretability framework.  The generated HTML shows the board, action
probabilities, value estimate, and the final trunk activation maps for every
move in the replay.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from games.rl_2048.agent import ResNet, ResNetConfig
from games.rl_2048.game import Game2048


ACTION_NAMES = ("left", "up", "right", "down")


class ActivationRecorder:
    """Capture selected intermediate tensors from a model forward pass."""

    def __init__(self, model: ResNet) -> None:
        self.activations: dict[str, torch.Tensor] = {}
        self.handles = [
            model.trunk.register_forward_hook(self._save("trunk")),
            model.trunk.stem.register_forward_hook(self._save("stem")),
            model.trunk.body.register_forward_hook(self._save("body")),
        ]

    def _save(self, name: str):
        def hook(_module, _inputs, output) -> None:
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[ResNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ResNet(ResNetConfig(**checkpoint["resnet_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _untransform_value(value: float, value_transform: str) -> float:
    if value_transform == "identity":
        return value
    magnitude = abs(value)
    if value_transform == "signed_log1p":
        raw_magnitude = torch.expm1(torch.tensor(magnitude)).item()
    else:
        raw_magnitude = (1 + magnitude) ** 2 - 1
    return (-raw_magnitude if value < 0 else raw_magnitude)


def _entropy(probabilities: torch.Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * positive.log()).sum().item())


def _find_game_seed(summary_path: Path, iteration: int, game_kind: str) -> tuple[int, int]:
    text = summary_path.read_text(encoding="utf-8")
    block_match = re.search(
        rf"Outer iteration {iteration}\n(?P<block>.*?)(?=\nOuter iteration |\Z)",
        text,
        flags=re.DOTALL,
    )
    if block_match is None:
        raise ValueError(f"No evaluation summary found for outer iteration {iteration}")
    label = "Best" if game_kind == "best" else "Worst"
    match = re.search(
        rf"{label} game: score (\d+), steps (\d+), seed (\d+)",
        block_match.group("block"),
    )
    if match is None:
        raise ValueError(f"No {game_kind}-game seed found for outer iteration {iteration}")
    return int(match.group(3)), int(match.group(1))


def replay_game(
    model: ResNet,
    checkpoint: dict,
    seed: int,
    device: torch.device,
) -> dict:
    """Replay a deterministic game and collect model diagnostics at each move."""
    game = Game2048(seed=seed)
    recorder = ActivationRecorder(model)
    value_transform = checkpoint["ppo_config"]["value_transform"]
    steps: list[dict] = []
    try:
        while not game.done:
            state = game.current_state().copy()
            legal_actions = game.legal_actions()
            state_tensor = state.to_tensor().unsqueeze(0).to(device)
            recorder.activations.clear()
            with torch.no_grad():
                output = model(state_tensor)
                logits = output.action_logits[0]
                masked_logits = logits.clone()
                illegal = torch.ones(4, dtype=torch.bool, device=device)
                illegal[legal_actions] = False
                masked_logits[illegal] = -torch.inf
                log_probabilities = F.log_softmax(masked_logits, dim=0)
                probabilities = log_probabilities.exp()
                action = int(masked_logits.argmax().item())

            legal_logits = masked_logits[~illegal]
            sorted_logits = torch.sort(legal_logits, descending=True).values
            margin = float((sorted_logits[0] - sorted_logits[1]).item()) if len(sorted_logits) > 1 else None
            activation = recorder.activations["trunk"][0].float().tolist()
            result = game.step(action)
            steps.append(
                {
                    "step": len(steps),
                    "board": state.grid.tolist(),
                    "legal_actions": legal_actions,
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "logits": [float(logits[i].item()) for i in range(4)],
                    "masked_logits": [
                        None if bool(illegal[i]) else float(masked_logits[i].item())
                        for i in range(4)
                    ],
                    "probabilities": [
                        0.0 if bool(illegal[i]) else float(probabilities[i].item())
                        for i in range(4)
                    ],
                    "entropy": _entropy(probabilities[~illegal]),
                    "logit_margin": margin,
                    "value_transformed": float(output.value[0].item()),
                    "value_raw": _untransform_value(float(output.value[0].item()), value_transform),
                    "reward": int(result.reward),
                    "total_score": int(game.total_score),
                    "trunk_activation": activation,
                }
            )
    finally:
        recorder.close()
    return {
        "seed": seed,
        "final_score": int(game.total_score),
        "steps": steps,
        "checkpoint_iteration": checkpoint.get("outer_iteration"),
        "value_transform": value_transform,
    }


def _render_html(report: dict, output_path: Path) -> None:
    report_json = json.dumps(report, separators=(",", ":"))
    title = html.escape(
        f"2048 policy inspection — score {report['final_score']} — seed {report['seed']}"
    )
    document = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; background: #f7f7f7; }}
h1 {{ font-size: 22px; margin: 0 0 8px; }}
.muted {{ color: #666; margin-bottom: 18px; }}
.controls {{ display: flex; gap: 12px; align-items: center; margin: 12px 0 20px; }}
button, input {{ font: inherit; }}
button {{ padding: 7px 12px; }}
#stepSlider {{ flex: 1; }}
.layout {{ display: grid; grid-template-columns: 300px minmax(360px, 1fr); gap: 22px; align-items: start; }}
.panel {{ background: white; border: 1px solid #ddd; padding: 16px; }}
.board {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; background: #bbada0; padding: 7px; }}
.tile {{ aspect-ratio: 1; display: grid; place-items: center; font-weight: 700; font-size: 20px; border-radius: 4px; color: #776e65; }}
.metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; margin-top: 14px; }}
.metric-label {{ color: #666; font-size: 12px; }}
.metric-value {{ font-variant-numeric: tabular-nums; font-size: 17px; }}
.prob-row {{ display: grid; grid-template-columns: 58px 1fr 70px; gap: 8px; align-items: center; margin: 8px 0; }}
.bar {{ height: 18px; background: #e8edf2; }}
.bar-fill {{ height: 100%; background: #3478b9; }}
.prob-row.selected .bar-fill {{ background: #d65a4a; }}
.prob-row.illegal {{ opacity: .35; }}
.prob-number {{ text-align: right; font-variant-numeric: tabular-nums; }}
.activation-grid {{ display: grid; grid-template-columns: repeat(4, minmax(90px, 1fr)); gap: 12px; }}
.activation-title {{ font-size: 13px; margin-bottom: 4px; }}
.activation-map {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; }}
.activation-cell {{ aspect-ratio: 1; min-width: 14px; }}
@media (max-width: 760px) {{ .layout {{ grid-template-columns: 1fr; }} .activation-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="muted">Use the slider or arrow buttons to inspect the deterministic policy at every move. Activation maps are the final 16-channel trunk output.</div>
<div class="controls">
  <button id="previous">Previous</button>
  <input id="stepSlider" type="range" min="0" max="0" value="0">
  <button id="next">Next</button>
  <span id="stepLabel"></span>
</div>
<div class="layout">
  <section class="panel">
    <div id="board" class="board"></div>
    <div id="metrics" class="metric-grid"></div>
  </section>
  <section class="panel">
    <h2>Action distribution</h2>
    <div id="probabilities"></div>
    <h2>Trunk activations</h2>
    <div id="activations" class="activation-grid"></div>
  </section>
</div>
<script id="report-data" type="application/json">{report_json}</script>
<script>
const report = JSON.parse(document.getElementById('report-data').textContent);
const steps = report.steps;
const slider = document.getElementById('stepSlider');
slider.max = Math.max(steps.length - 1, 0);
const actionNames = ['left', 'up', 'right', 'down'];
const tileColors = {{0:'#cdc1b4',2:'#eee4da',4:'#ede0c8',8:'#f2b179',16:'#f59563',32:'#f67c5f',64:'#f65e3b',128:'#edcf72',256:'#edcc61',512:'#edc850',1024:'#edc53f',2048:'#edc22e'}};

function tileColor(value) {{ return tileColors[value] || '#3c3a32'; }}
function formatNumber(value) {{ return value === null ? '—' : Number(value).toFixed(4); }}
function renderBoard(step) {{
  const board = document.getElementById('board');
  board.innerHTML = '';
  step.board.flat().forEach(value => {{
    const cell = document.createElement('div');
    cell.className = 'tile'; cell.style.background = tileColor(value);
    if (value >= 8) cell.style.color = '#f9f6f2';
    cell.textContent = value === 0 ? '' : value;
    board.appendChild(cell);
  }});
}}
function renderMetrics(step, index) {{
  const values = [
    ['move', `${{index + 1}} / ${{steps.length}}`],
    ['score after move', step.total_score],
    ['action', step.action_name],
    ['reward', step.reward],
    ['raw value', formatNumber(step.value_raw)],
    ['entropy', formatNumber(step.entropy)],
    ['top-logit margin', formatNumber(step.logit_margin)],
    ['legal moves', step.legal_actions.map(action => actionNames[action]).join(', ')],
  ];
  document.getElementById('metrics').innerHTML = values.map(([label, value]) => `<div><div class="metric-label">${{label}}</div><div class="metric-value">${{value}}</div></div>`).join('');
  document.getElementById('stepLabel').textContent = `move ${{index + 1}} / ${{steps.length}}`;
}}
function renderProbabilities(step) {{
  document.getElementById('probabilities').innerHTML = step.probabilities.map((probability, action) => {{
    const legal = step.legal_actions.includes(action);
    const selected = step.action === action;
    return `<div class="prob-row ${{selected ? 'selected' : ''}} ${{legal ? '' : 'illegal'}}"><span>${{actionNames[action]}}</span><div class="bar"><div class="bar-fill" style="width:${{(probability * 100).toFixed(2)}}%"></div></div><span class="prob-number">${{(probability * 100).toFixed(1)}}%</span></div>`;
  }}).join('');
}}
function renderActivations(step) {{
  const channels = step.trunk_activation;
  const max = Math.max(...channels.flat(2).map(value => Math.abs(value)), 1e-8);
  document.getElementById('activations').innerHTML = channels.map((channel, index) => {{
    const cells = channel.flat().map(value => `<div class="activation-cell" title="${{Number(value).toFixed(3)}}" style="background:rgba(34,119,180,${{Math.min(Math.abs(value) / max, 1).toFixed(3)}})"></div>`).join('');
    return `<div><div class="activation-title">channel ${{index}}</div><div class="activation-map">${{cells}}</div></div>`;
  }}).join('');
}}
function render(index) {{
  index = Math.max(0, Math.min(index, steps.length - 1));
  slider.value = index;
  const step = steps[index];
  renderBoard(step); renderMetrics(step, index); renderProbabilities(step); renderActivations(step);
}}
slider.addEventListener('input', event => render(Number(event.target.value)));
document.getElementById('previous').addEventListener('click', () => render(Number(slider.value) - 1));
document.getElementById('next').addEventListener('click', () => render(Number(slider.value) + 1));
document.addEventListener('keydown', event => {{ if (event.key === 'ArrowLeft') render(Number(slider.value) - 1); if (event.key === 'ArrowRight') render(Number(slider.value) + 1); }});
render(0);
</script>
</body>
</html>
'''
    output_path.write_text(document, encoding="utf-8")


def inspect_game(
    run_dir: Path,
    checkpoint_path: Path | None = None,
    seed: int | None = None,
    game_kind: str = "worst",
    output_dir: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    if checkpoint_path is None:
        checkpoint_path = run_dir / "final.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "best.pt"
    device = device or torch.device("cpu")
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_iteration = int(checkpoint["outer_iteration"])
    expected_score = None
    if seed is None:
        seed, expected_score = _find_game_seed(run_dir / "summary.txt", checkpoint_iteration, game_kind)
    report = replay_game(model, checkpoint, seed, device)
    report["game_kind"] = game_kind
    report["checkpoint_path"] = str(checkpoint_path)
    report["expected_score_from_summary"] = expected_score
    if output_dir is None:
        output_dir = run_dir / "worst_game_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = output_dir / "replay.html"
    _render_html(report, html_path)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--game", choices=("worst", "best"), default="worst")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    html_path = inspect_game(args.run_dir, args.checkpoint, args.seed, args.game, args.output)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
