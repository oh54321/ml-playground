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
