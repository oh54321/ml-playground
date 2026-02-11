from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display as ipy_display


class GameDisplay:
    history: List[np.ndarray] = []

    def reset(self):
        self.history = []

    def add(self, frame: np.ndarray):
        self.history.append(frame)

    def _to_hwc_uint8(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))

        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255.0).round()
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def display(self, interval: int = 30) -> FuncAnimation:
        if len(self.history) == 0:
            raise ValueError("No frames in history. Run an episode first.")

        frames = [self._to_hwc_uint8(f) for f in self.history]
        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(frames[0])

        def update(i):
            im.set_data(frames[i])
            return (im,)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=interval,
            blit=True,
        )
        plt.close(fig)
        ipy_display(HTML(anim.to_jshtml()))
        return anim
