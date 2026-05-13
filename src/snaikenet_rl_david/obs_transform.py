import numpy as np

from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientTileType


NUM_TILE_TYPES = 5


def frame_to_raw_obs(frame: ClientGameStateFrame) -> dict:
    return {
        "grid": np.asarray(frame.grid_data, dtype=np.uint8),
        "player_length": np.array([frame.player_length], dtype=np.float32),
        "num_kills": np.array([frame.num_kills], dtype=np.float32),
        "closest_food": np.array(closest_food_offset(frame), dtype=np.float32),
    }


def closest_food_offset(frame: ClientGameStateFrame) -> tuple[float, float]:
    grid = np.asarray(frame.grid_data, dtype=np.uint8)

    h, w = grid.shape
    cy, cx = h // 2, w // 2

    food_positions = np.argwhere(grid == ClientTileType.FOOD)

    if len(food_positions) == 0:
        return 0.0, 0.0

    deltas = food_positions - np.array([cy, cx])
    distances = np.abs(deltas).sum(axis=1)
    dy, dx = deltas[np.argmin(distances)]

    return float(dy), float(dx)


def raw_obs_to_model_obs(raw_obs: dict) -> dict:
    grid = raw_obs["grid"]

    one_hot = np.zeros((NUM_TILE_TYPES, *grid.shape), dtype=np.float32)

    for tile_type in range(NUM_TILE_TYPES):
        one_hot[tile_type] = (grid == tile_type).astype(np.float32)

    h, w = grid.shape

    dy, dx = raw_obs["closest_food"]

    stats = np.array(
        [
            float(raw_obs["player_length"][0]) / 100.0,
            float(raw_obs["num_kills"][0]) / 10.0,
            float(dy) / h,
            float(dx) / w,
        ],
        dtype=np.float32,
    )

    return {
        "grid": one_hot,
        "stats": stats,
    }