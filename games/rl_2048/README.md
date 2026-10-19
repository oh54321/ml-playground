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
recorded in each run configuration. CPU, CUDA, and MPS are supported. Device selection
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
