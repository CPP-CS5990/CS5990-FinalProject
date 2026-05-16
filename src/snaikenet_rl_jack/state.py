import numpy as np
import torch

from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientTileType

NUM_TILE_TYPES = len(ClientTileType)
VIEWPORT_DISTANCE: tuple[int, int] = (24, 24)
VIEWPORT_SIZE: tuple[int, int] = (
    2 * VIEWPORT_DISTANCE[0] + 1,
    2 * VIEWPORT_DISTANCE[1] + 1,
)


def encode_frame(frame: ClientGameStateFrame, device: torch.device) -> torch.Tensor:
    grid = np.asarray(frame.grid_data, dtype=np.int64)
    if grid.shape != VIEWPORT_SIZE:
        raise ValueError(
            f"Expected viewport size {VIEWPORT_SIZE} "
            f"(viewport-distance {VIEWPORT_DISTANCE}); got {tuple(grid.shape)}. "
            f"Start the server with "
            f"--viewport-distance {VIEWPORT_DISTANCE[0]} {VIEWPORT_DISTANCE[1]}."
        )
    channels = np.zeros((NUM_TILE_TYPES, *grid.shape), dtype=np.float32)
    for t in range(NUM_TILE_TYPES):
        channels[t] = grid == t
    return torch.from_numpy(channels).to(device)
