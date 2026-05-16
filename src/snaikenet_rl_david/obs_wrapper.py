import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snaikenet_rl_david.obs_transform import raw_obs_to_model_obs, NUM_TILE_TYPES


class SnaikeNetDQNObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        h, w = env.observation_space["grid"].shape

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(NUM_TILE_TYPES, h, w),
                    dtype=np.float32,
                ),
                "stats": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, obs):
        return raw_obs_to_model_obs(obs)
