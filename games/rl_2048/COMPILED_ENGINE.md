# Compiled 2048 game engine

The game uses Numba for the int64 board legality scan and fused move, merge,
spawn, and terminal checks. Install the pinned runtime with:

```sh
python -m pip install -r games/rl_2048/requirements-numba.txt
```

Numba compiles its small kernels on first use and caches compiled signatures on
disk. The public `Game2048` API and the NumPy helpers remain available. Boards
using integer dtypes other than `int64` use the existing NumPy step path so
their dtype and overflow behavior are preserved.
