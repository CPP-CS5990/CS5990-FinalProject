import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SnaikeNETObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        h, w = env.observation_space["grid"].shape
        self.num_channels = 5

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_channels, h, w),
                    dtype=np.float32,
                ),
                "stats": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                )
            }
        )

    def observation(self, obs):
        grid = obs["grid"]

        one_hot = np.zeros((self.num_channels, *grid.shape), dtype=np.float32)
        for tile_type in range(self.num_channels):
            one_hot[tile_type] = (grid == tile_type).astype(np.float32)

        dy, dx = self._food_direction(grid)

        stats = np.array(
            [
                float(obs["length"]) / 100.0,
                float(obs["kills"]) / 10.0,
                dy,
                dx,
            ],
            dtype=np.float32,
        )

        return {
            "grid": one_hot,
            "stats": stats,
        }

    def _food_direction(self, grid):
        snake_positions = np.argwhere(grid == 3)  # own snake
        food_positions = np.argwhere(grid == 2)  # food

        if len(snake_positions) == 0 or len(food_positions) == 0:
            return 0.0, 0.0

        snake = snake_positions[0]  # approximate head

        deltas = food_positions - snake
        dists = np.abs(deltas).sum(axis=1)
        nearest = deltas[np.argmin(dists)]

        dy, dx = nearest

        # normalize by grid size
        h, w = grid.shape
        dy = float(dy) / h
        dx = float(dx) / w

        return dy, dx
