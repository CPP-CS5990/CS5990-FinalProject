import asyncio
import threading

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snaikenet_client.admin_client import SnaikenetAdminClient
from snaikenet_client.client.client import SnaikenetClient
from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.gym_event_handler import GymSnakeEventHandler
from snaikenet_client.types import ClientDirection

ACTION_TO_DIRECTION = {
    0: ClientDirection.NORTH,
    1: ClientDirection.SOUTH,
    2: ClientDirection.EAST,
    3: ClientDirection.WEST,
}


class SnaikenetGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            server_host: str = "localhost",
            server_tcp_port: int = 8888,
            admin_port: int = 8889,
            send_interval_ms: int = 13,
            step_timeout: float = 1.0,
            viewport_size: tuple[int, int] = (41, 41),
    ):
        super().__init__()

        self.step_timeout = step_timeout

        self.event_handler = GymSnakeEventHandler()
        self.client = SnaikenetClient(
            server_tcp_port=server_tcp_port,
            server_host=server_host,
            send_interval_ms=send_interval_ms,
            event_handler=self.event_handler,
            is_spectator=False,
        )

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
        )
        self._thread.start()

        self._started = False
        self._last_game_epoch = 0
        self._last_frame: ClientGameStateFrame | None = None
        self._previous_length = 0
        self._previous_kills = 0

        self.action_space = spaces.Discrete(4)

        # Will be finalized after the first GAME_START event.
        self.viewport_size = viewport_size
        viewport_width, viewport_height = self.viewport_size

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=4,
                    shape=(viewport_height, viewport_width),
                    dtype=np.int8,
                ),
                "length": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max,
                    shape=(),
                    dtype=np.int32,
                ),
                "kills": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max,
                    shape=(),
                    dtype=np.int32,
                ),
            }
        )

        self.admin_client = SnaikenetAdminClient(
            host=server_host,
            port=admin_port,
        )

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _start_client_if_needed(self):
        if self._started:
            return

        future = asyncio.run_coroutine_threadsafe(
            self.client.start(),
            self._loop,
        )
        future.result(timeout=10.0)

        actual_viewport_size = self.event_handler.wait_for_start(timeout=10.0)

        if actual_viewport_size != self.viewport_size:
            raise ValueError(
                f"Expected viewport_size={self.viewport_size}, "
                f"but server reported {actual_viewport_size}"
            )

        self._started = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._start_client_if_needed()

        previous_epoch = self._last_game_epoch

        future = asyncio.run_coroutine_threadsafe(
            self.admin_client.restart_game(),
            self._loop,
        )
        future.result(timeout=5.0)

        buffered = self.event_handler.wait_for_frame_in_new_epoch(
            previous_epoch=previous_epoch,
            timeout=self.step_timeout,
        )

        frame = buffered.frame
        self._last_frame = frame
        self._last_game_epoch = buffered.game_epoch

        obs = self._frame_to_obs(frame)
        info = self._frame_to_info(frame)

        return obs, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        if self._last_frame is None:
            raise RuntimeError("Call reset() before step().")

        previous_frame = self._last_frame
        previous_epoch = self._last_game_epoch
        previous_sequence = previous_frame.sequence_number

        direction = ACTION_TO_DIRECTION[int(action)]
        # self.client.set_direction(direction)
        self._loop.call_soon_threadsafe(
            self.client.set_direction,
            direction,
        )

        buffered = self.event_handler.wait_for_frame_after(
            previous_epoch=previous_epoch,
            previous_sequence_number=previous_sequence,
            timeout=self.step_timeout,
        )

        frame = buffered.frame
        self._last_frame = frame
        self._last_game_epoch = buffered.game_epoch

        obs = self._frame_to_obs(frame)

        if buffered.game_epoch != previous_epoch:
            reward = 0.0
            terminated = True
        else:
            reward = self._compute_reward(previous_frame, frame)
            terminated = self._is_terminated(frame)

        truncated = False
        info = self._frame_to_info(frame)

        return obs, reward, terminated, truncated, info

    def _frame_to_obs(self, frame: ClientGameStateFrame):
        grid = np.asarray(frame.grid_data, dtype=np.int8)

        return {
            "grid": grid,
            "length": np.asarray(frame.player_length, dtype=np.int32),
            "kills": np.asarray(frame.num_kills, dtype=np.int32),
        }

    def _distance_to_nearest_food(self, frame):
        grid = np.asarray(frame.grid_data)

        snake_positions = np.argwhere(grid == 3)
        food_positions = np.argwhere(grid == 2)

        if len(snake_positions) == 0 or len(food_positions) == 0:
            return None

        # assume one own-snake tile at length 1 or use closest own-snake tile as proxy
        snake = snake_positions[0]

        distances = np.abs(food_positions - snake).sum(axis=1)
        return float(distances.min())

    def _compute_reward(
            self,
            previous_frame: ClientGameStateFrame,
            frame: ClientGameStateFrame,
    ) -> float:
        reward = 0.0

        length_delta = frame.player_length - previous_frame.player_length
        kill_delta = frame.num_kills - previous_frame.num_kills

        reward += 100.0 * length_delta
        reward += 20.0 * kill_delta

        prev_dist = self._distance_to_nearest_food(previous_frame)
        curr_dist = self._distance_to_nearest_food(frame)

        reward -= 0.001

        if prev_dist is not None and curr_dist is not None:
            reward += 0.5 * (prev_dist - curr_dist)

        if previous_frame.is_alive and not frame.is_alive:
            reward -= 10.0

        return float(reward)

    def _is_terminated(self, frame: ClientGameStateFrame) -> bool:
        return not frame.is_alive or frame.is_spectating or self.event_handler.ended

    def _frame_to_info(self, frame: ClientGameStateFrame):
        return {
            "sequence_number": frame.sequence_number,
            "is_alive": frame.is_alive,
            "is_spectating": frame.is_spectating,
            "player_length": frame.player_length,
            "num_kills": frame.num_kills,
            "client_id": self.client.get_client_id(),
        }

    def close(self):
        if self._started:
            future = asyncio.run_coroutine_threadsafe(self.client.stop(), self._loop)
            try:
                future.result(timeout=5.0)
            except TimeoutError:
                pass
            self._started = False

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
