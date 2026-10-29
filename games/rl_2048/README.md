# Reinforcement learning for 2048

The shared interfaces in `core/` describe discrete-action environments, tensor
observations, and actor-critic models. They are independent of the game and
learning algorithm.

From the repository root, install the dependencies in a Python 3.10+ environment:

```sh
python -m pip install -r games/rl_2048/requirements.txt
```

`Game2048(height=4, width=4)` defaults to a 4×4 board; other positive
rectangular dimensions are supported, such as `Game2048(height=6, width=6)`.
Actions are ordered left, up, right, down.
`step()` returns a reward and an immutable next-state snapshot.

Observations default to numeric log2 magnitudes (`numeric`), with zero for empty
cells. Choose `observation_encoding="numeric_onehot"` to append one-hot planes.
For an H×W board, this produces H×W+3 channels: one numeric plane, one empty-cell
indicator, and tile-exponent planes 1 through H×W+1. That is 19 channels for 4×4
and 39 for 6×6; it is not one-hot alone. The state owns this encoding and preserves
it across copies, game steps, and resets.

Run the tests from the repository root:

```sh
python -m unittest discover -s games/rl_2048/tests -v
```

## Model

The generic models live in `core/models/` and accept already-encoded tensors
with explicit input channels, spatial dimensions, and action counts. The 2048
state owns tile encoding; callers derive the model input channels from that
choice, as shown below.

The CNN training configuration uses separate policy and value trunks with four residual blocks,
32 channels, GroupNorm, and 32-channel heads. Both observation encodings
use the same model interface. `ConvActorCriticConfig` accepts `kernel_size`
(positive and odd, default 3) and `block_layers` (positive, default 2).
The kernel size applies to the stem and residual convolutions. GroupNorm is
always enabled, and policy and value trunks remain separate.

```python
from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding

encoding = ObservationEncoding.NUMERIC_ONEHOT
config = ConvActorCriticConfig(in_channels=encoding.in_channels(4, 4),
                              height=4, width=4, output_dim=4, n_channels=32)
model = ConvActorCriticModel(config)
game = Game2048(seed=123, observation_encoding=encoding)
output = model(game.state.to_tensor().unsqueeze(0))
```

The D4 model ties weights across rotations and reflections: these transforms
permute the policy's actions and leave its value prediction unchanged. It is
the default architecture, with 64 channels. Both models support either encoding,
so there are four architecture/encoding combinations. The two standard widths
do not imply equal parameter counts. The CNN supports rectangular observations;
D4 requires square observations and a supplied transitive D4 action mapping.
2048’s four-direction mappings are defined in `environment/symmetry_2048.py`.

Both models use the same kernel-size and block-layer settings, defaulting to
3×3 convolutions and two layers per residual block. D4 reuses expanded weights during inference and clears them before
training updates. Optional compilation preserves the model's parameters and
state dictionary.

## PPO

`algorithms/` collects rollouts, computes generalized advantage estimates, and
optimizes the clipped policy loss with a value loss and entropy regularization.
Illegal actions are masked during collection and updates. Terminal states stop
bootstrapping; time limits and rollout boundaries bootstrap from the next state.
Rollouts use the original board orientation for both architectures.

`PPO.run_outer_iteration()` collects one rollout and applies minibatch updates.
KL-based stopping is checked after each completed epoch, once
`min_update_epochs` have finished. Setting it to 2 guarantees at least two epochs;
`max_update_epochs` bounds the total. Rollout tensors are prepared once and reused
across update epochs.

## Train and evaluate

```sh
python -m games.rl_2048.train_2048
```

The default run uses D4 with numeric-plus-one-hot observations, 128 games,
1,024 steps per game per rollout, and 16,384-transition minibatches. Adam starts
at 3e-4, multiplied by 0.999 after each iteration through iteration 6,000, then
held at approximately 7.414e-7. This reproduces the historical 10,000-iteration
run's learning-rate schedule. Each iteration performs 2–8 update epochs,
stopping when the measured KL exceeds 0.02 after the minimum two epochs.
Training runs for 10,000 iterations and evaluates 500 seeded games every 50
iterations. Evaluation chooses the highest-scoring legal action; training samples
from the masked policy.

A short CPU run checks the complete training/evaluation path:

```sh
python -m games.rl_2048.train_2048 --device cpu --outer-iterations 1 \
  --num-parallel-games 2 --steps-per-game-per-rollout 4 \
  --evaluation-interval 1 --evaluation-games 1
```

Choose `--architecture cnn` or `--architecture d4`, independently of
`--observation-encoding numeric` or `--observation-encoding numeric_onehot`.
Choose board dimensions with `--height` and `--width`, for example
`--height 6 --width 6`. CNN also supports rectangular boards; D4 requires
equal height and width. Training requires at least three cells so new games
have a legal move. Each model is built for the selected board shape.
The standard widths are 32 for CNN and 64 for D4;
`--channels` overrides them. Use `--kernel-size` and `--block-layers` to change
the convolution size and depth within each residual block. The historical ablation model/schedule settings were:

```sh
python -m games.rl_2048.train_2048 --architecture cnn \
  --observation-encoding numeric_onehot --outer-iterations 6000 --scheduler-decay 0.999
python -m games.rl_2048.train_2048 --architecture d4 \
  --observation-encoding numeric --outer-iterations 6000 --scheduler-decay 0.999
```

`--scheduler-decay-until` sets the last iteration to decay, inclusive; afterward
the rate stays constant. Set `scheduler_decay_until=None` in `TRAINING_CONFIG`
to decay throughout training, or use `--scheduler-decay 1.0` for a constant rate
from the start. Both schedule settings are saved in each run's `config.json`.

Edit experiment defaults at the top of `train_2048.py`: `TRAINING_CONFIG` for
training and evaluation, `MODEL_CHANNELS` and `MODEL_SETTINGS` for the network,
and `PPO_SETTINGS` for the algorithm. `DEVICE = None` selects hardware automatically.
Command-line options override those values for a single run. `training/config.py`
defines and validates the training fields; it contains no experiment defaults.
Architecture and observation encoding are
recorded in each run configuration and checkpoint. CPU, CUDA, and MPS are supported. Device selection
prefers CUDA, then MPS, then CPU; use `--device` to choose explicitly. On CUDA,
`--compile-model` optionally compiles actor and critic calls on first use.

Each run writes to a new directory under `artifacts/ppo_2048/`: configuration,
`metrics.jsonl`, and per-game evaluation results. Generated artifacts are ignored
by Git. For optional TensorBoard logging:

```sh
python -m pip install tensorboard
python -m games.rl_2048.train_2048 --tensorboard
tensorboard --logdir artifacts/ppo_2048
```

## Checkpoints

Each run saves `latest.pt` at evaluation intervals and `best.pt` when its mean
evaluation score improves. `checkpoint_0500.pt`, `checkpoint_1000.pt`, etc. are
periodic recovery points; `--checkpoint-interval` changes their spacing.
`final.pt` is written after the requested training iterations finish. Best-model
selection is local to each run directory.

Continue a saved run by repeating its model, PPO, and schedule options:

```sh
python -m games.rl_2048.train_2048 \
  --warm-start-from artifacts/ppo_2048/<run>/latest.pt --outer-iterations 12000
```

For a CNN or numeric-only run, repeat its `--architecture`,
`--observation-encoding`, and any custom `--channels`, `--kernel-size`, or
`--block-layers` settings. Model and PPO
configuration must match the checkpoint, including board dimensions and rollout
sizes. Repeat `--height` and `--width` when resuming a non-default board. Recovery
restores model weights, Adam state, the current learning rate, scheduler counters,
and the completed iteration. Repeat `--scheduler-decay` and
`--scheduler-decay-until` from the saved run configuration to preserve its schedule.
The cutoff uses the absolute iteration number, so the default schedule stays
constant after iteration 6,000, including across restarts. To intentionally stop
decay earlier, use `--scheduler-decay 1.0`; it holds the restored rate constant.

`--outer-iterations` is the final iteration number, not an additional count.
Rollout games and random streams restart, so this is a warm start rather than an
exact continuation. Only checkpoints produced by this clean implementation are
supported. On interruption, resume from the most recent saved checkpoint.

## Run the demo locally

```sh
python -m games.rl_2048.demo --checkpoint artifacts/ppo_2048/<run>/best.pt
```

Open the address printed by the command. Play with arrow keys, WASD, swipe gestures, or
direction buttons. **Play model** lets the policy take over; **Pause model** and
**One step** let you inspect its decisions. Manual keyboard input pauses autoplay.
**New game** uses a fresh random seed. The speed slider sets a target of up to
50 moves per second; inference and rendering can limit the actual rate.
The latest move stays visible during autoplay, and **Move history** lists the
current game's moves and score gains. Click an entry (including **Starting board**)
to pause and view its board and score. **Return to live game** restores the current
board; browsing history never rewinds the game. Snapshots stay in memory until
**New game** or a server restart. Each move shows the merged board before
the new tile appears; the animation shortens at higher speeds and respects
your browser's reduced-motion setting.

The demo loads the architecture, board dimensions, observation encoding, and weights
from a clean training checkpoint. It uses CPU inference and greedily chooses the
highest-scoring legal action, just like evaluation. CNN and D4 checkpoints with
either encoding work without changing the demo code. It does not train or modify
the checkpoint.

Omit `--checkpoint` to play manually, or use `--port 8001` if port 8000 is occupied.
The server listens only on your computer and keeps one game in memory, shared
across its browser tabs. Restarting the server resets it. Stop with Ctrl+C.

## Host the demo on GitHub Pages

The static demo runs the game and model entirely in the visitor's browser using
ONNX Runtime Web. It needs no inference server. The included `demo_web/policy.onnx`
is an actor-only export of the saved best D4 numeric-plus-one-hot policy (about
2.3 MB); `model.json` records its board settings and checkpoint checksum. The
critic, optimizer, and training artifacts are not published. Each browser tab has
its own random game and move history; refreshing the page resets it.

Build and preview with Node.js 22 and any static HTTP server:

```sh
npm --prefix games/rl_2048/demo_web ci --ignore-scripts
npm --prefix games/rl_2048/demo_web test
npm --prefix games/rl_2048/demo_web run build
python -m http.server 8001 --bind 127.0.0.1 --directory games/rl_2048/demo_web/dist
```

Open the server's address in your browser. Serve `dist/`, rather than opening the
HTML file directly. The build includes the pinned runtime, its WebAssembly files,
and license notices; the deployed page makes no requests to an inference service
or external CDN. Model loading can take a moment, and speed depends on the
visitor's device. Manual play remains available if the model cannot load.

To publish, choose **Settings → Pages → Source → GitHub Actions** in the repository.
The workflow in `.github/workflows/demo-pages.yml` builds and deploys on pushes to
`rwu/ppo-2048` that change the demo, its browser tests, or the workflow. Allow that
branch in the `github-pages` environment's deployment rules if necessary. Change
the workflow's branch filter when moving the demo to another branch. GitHub shows
the published URL in the deployment; relative asset paths support repository
sites such as `/<repository>/`. The local Python demo remains available with the
same interface and controls.

To replace the browser policy, install the optional export dependencies in a
Python 3.11+ environment and export a clean checkpoint:

```sh
python -m pip install -r games/rl_2048/requirements-demo-export.txt
python -m games.rl_2048.export_policy artifacts/ppo_2048/<run>/best.pt
```

The exporter supports CNN and D4 with either observation encoding. It checks the
exported logits and legal action choices against PyTorch on 128 boards before
replacing `policy.onnx` and `model.json`. Rebuild the site after exporting. The
browser and local demos share `app.js`; `backend.js` runs browser inference, while
the Python server serves `server_backend.js` as its HTTP adapter.
