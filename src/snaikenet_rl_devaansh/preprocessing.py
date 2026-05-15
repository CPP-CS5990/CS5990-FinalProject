import numpy as np
import torch

from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientTileType, ClientDirection

NUM_TILE_TYPES   = len(ClientTileType)   # Empty, Wall, Food, Snake, Other_Snake
NUM_DIR_CHANNELS = 4                     # one channel per cardinal direction


def frame_to_tensor(frame: ClientGameStateFrame) -> torch.Tensor:
    """
    Converts a ClientGameStateFrame grid into a one-hot encoded Tensor.
    Returns shape: (NUM_TILE_TYPES, H, W) = (5, H, W)
    """
    grid = frame.grid_data
    H = len(grid)
    W = len(grid[0])

    raw = np.array(grid, dtype=np.int64)
    one_hot = np.zeros((NUM_TILE_TYPES, H, W), dtype=np.float32)
    for tile_type in ClientTileType:
        one_hot[tile_type] = (raw == tile_type).astype(np.float32)

    return torch.from_numpy(one_hot)


def direction_to_tensor(direction: ClientDirection, H: int, W: int) -> torch.Tensor:
    """
    Encodes the current heading as a (NUM_DIR_CHANNELS, H, W) spatial tensor.
    The channel for the active direction is filled with 1s; others are 0s.
    This gives the CNN an unambiguous signal about which way the snake is moving.
    """
    t = torch.zeros(NUM_DIR_CHANNELS, H, W, dtype=torch.float32)
    t[int(direction)].fill_(1.0)
    return t


class FrameStacker:
    """
    Keeps the last 'n_frames' grid observations stacked along the channel dimension.
    output shape: (NUM_TILE_TYPES * n_frames, H, W)
    """

    def __init__(self, n_frames: int = 3):
        self.n_frames = n_frames
        self._frames: list[torch.Tensor] = []

    def reset(self, first_frame: ClientGameStateFrame) -> torch.Tensor:
        """Call at the start of each episode — older slots are zero-padded."""
        t = frame_to_tensor(first_frame)
        C, H, W = t.shape
        zero = torch.zeros(C, H, W, dtype=torch.float32)
        self._frames = [zero] * (self.n_frames - 1) + [t]
        return self._get_stacked()

    def step(self, frame: ClientGameStateFrame) -> torch.Tensor:
        """Push a new frame and return the updated stacked observation."""
        self._frames.append(frame_to_tensor(frame))
        if len(self._frames) > self.n_frames:
            self._frames.pop(0)
        return self._get_stacked()

    def _get_stacked(self) -> torch.Tensor:
        return torch.cat(self._frames, dim=0)
