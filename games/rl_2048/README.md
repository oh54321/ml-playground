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

The CNN defaults to separate policy and value trunks with four residual blocks,
64 channels, GroupNorm, and 32-channel heads. The example below chooses 32 trunk
channels. Both observation encodings use the same model interface. `ConvActorCriticConfig` accepts `kernel_size`
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
permute the policy's actions and leave its value prediction unchanged. It uses
the same configuration interface, whose default width is 64 channels. Both models
support either encoding. The CNN supports rectangular observations;
D4 requires square observations and a supplied transitive D4 action mapping.
2048’s four-direction mappings are defined in `environment/symmetry_2048.py`.

Both models use the same kernel-size and block-layer settings, defaulting to
3×3 convolutions and two layers per residual block. D4 reuses expanded weights during inference and clears them before
training updates. Optional compilation preserves the model's parameters and
state dictionary.
