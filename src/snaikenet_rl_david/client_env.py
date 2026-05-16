import asyncio
import queue
import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snaikenet_agent.client_events import ClientOnlyEventHandler
from snaikenet_agent.obs_transform import frame_to_raw_obs
from snaikenet_client.client.client import SnaikenetClient
from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientDirection


NUM_TILE_TYPES = 5


class SnaikeNetClientEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        server_host: str = "localhost",
        server_tcp_port: int = 8888,
        viewport_size: int = 41,
        frame_timeout: float = 5.0,
        death_penalty: float = -100.0,
        food_reward: float = 100.0,
        closer_food_reward: float = 1.0,
        farther_food_penalty: float = -1.0,
        living_penalty: float = -0.01,
        max_steps_without_food: int = 1000,
    ):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=NUM_TILE_TYPES - 1,
                    shape=(viewport_size, viewport_size),
                    dtype=np.uint8,
                ),
                "player_length": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "num_kills": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "closest_food": spaces.Box(
                    low=-viewport_size,
                    high=viewport_size,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        self.server_host = server_host
        self.server_tcp_port = server_tcp_port
        self.viewport_size = viewport_size
        self.frame_timeout = frame_timeout

        self.death_penalty = death_penalty
        self.food_reward = food_reward
        self.closer_food_reward = closer_food_reward
        self.farther_food_penalty = farther_food_penalty
        self.living_penalty = living_penalty
        self.max_steps_without_food = max_steps_without_food

        self.event_handler = ClientOnlyEventHandler()
        self.client: SnaikenetClient | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.client_uuid: str | None = None

        self.prev_frame: ClientGameStateFrame | None = None
        self.last_food_sequence = 0

    def _start_client_thread(self):
        if self.thread is not None and self.thread.is_alive():
            return

        ready = threading.Event()

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self.client = SnaikenetClient(
                server_host=self.server_host,
                server_tcp_port=self.server_tcp_port,
                event_handler=self.event_handler,
            )

            self.loop.run_until_complete(self.client.start(uuid=self.client_uuid))
            self.client_uuid = self.client.get_client_id()

            ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

        if not ready.wait(timeout=10.0):
            raise TimeoutError("Timed out starting SnaikeNET client.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._start_client_thread()
        self._drain_events()

        frame = self._wait_for_alive_start_frame()

        self.prev_frame = frame
        self.last_food_sequence = frame.sequence_number

        obs = self._frame_to_obs(frame)
        info = self._frame_to_info(frame)

        return obs, info

    def step(self, action: int):
        if self.client is None or self.loop is None:
            raise RuntimeError("Call reset() before step().")

        direction = ClientDirection(int(action))

        self.loop.call_soon_threadsafe(
            self.client.set_direction,
            direction,
        )

        frame = self._wait_for_next_frame_or_end()

        if frame is None:
            obs = self._frame_to_obs(self.prev_frame)
            reward = self.death_penalty
            return obs, reward, True, False, {}

        prev = self.prev_frame
        reward = self._compute_reward(prev, frame)

        terminated = (
            not frame.is_alive
            or frame.is_spectating
            or frame.sequence_number - self.last_food_sequence
            > self.max_steps_without_food
        )

        self.prev_frame = frame

        obs = self._frame_to_obs(frame)
        info = self._frame_to_info(frame)

        return obs, reward, terminated, False, info

    def _wait_for_alive_start_frame(self) -> ClientGameStateFrame:
        while True:
            event = self.event_handler.event_queue.get(timeout=self.frame_timeout)

            if event.kind != "frame" or event.frame is None:
                continue

            frame = event.frame

            if frame.is_alive and not frame.is_spectating:
                return frame

    def _wait_for_next_frame_or_end(self) -> ClientGameStateFrame | None:
        deadline = time.monotonic() + self.frame_timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            try:
                event = self.event_handler.event_queue.get(timeout=remaining)
            except queue.Empty:
                return None

            if event.kind == "frame" and event.frame is not None:
                return event.frame

            if event.kind in {"game_end", "game_restart"}:
                return None

    def _drain_events(self):
        while True:
            try:
                self.event_handler.event_queue.get_nowait()
            except queue.Empty:
                break

    def _frame_to_obs(self, frame: ClientGameStateFrame | None) -> dict:
        if frame is None:
            return self._zero_obs()

        raw_obs = frame_to_raw_obs(frame)
        grid = raw_obs["grid"]

        padded = np.zeros((self.viewport_size, self.viewport_size), dtype=np.uint8)

        h, w = grid.shape
        copy_h = min(h, self.viewport_size)
        copy_w = min(w, self.viewport_size)

        padded[:copy_h, :copy_w] = grid[:copy_h, :copy_w]

        return {
            "grid": padded,
            "player_length": raw_obs["player_length"],
            "num_kills": raw_obs["num_kills"],
            "closest_food": raw_obs["closest_food"],
        }

    def _zero_obs(self):
        return {
            "grid": np.zeros((self.viewport_size, self.viewport_size), dtype=np.uint8),
            "player_length": np.array([0], dtype=np.float32),
            "num_kills": np.array([0], dtype=np.float32),
            "closest_food": np.array([0.0, 0.0], dtype=np.float32),
        }

    def _compute_reward(
        self,
        prev: ClientGameStateFrame | None,
        curr: ClientGameStateFrame,
    ) -> float:
        if prev is None:
            return 0.0

        if not curr.is_alive:
            return self.death_penalty

        reward = self.living_penalty

        length_diff = curr.player_length - prev.player_length
        if length_diff > 0:
            reward += self.food_reward * length_diff
            self.last_food_sequence = curr.sequence_number

        kill_diff = curr.num_kills - prev.num_kills
        if kill_diff > 0:
            reward += 100.0 * kill_diff

        prev_dist = self._closest_food_distance(prev)
        curr_dist = self._closest_food_distance(curr)

        if np.isfinite(prev_dist) and np.isfinite(curr_dist):
            if curr_dist < prev_dist:
                reward += self.closer_food_reward
            elif curr_dist > prev_dist:
                reward += self.farther_food_penalty

        return float(reward)

    @staticmethod
    def _closest_food_distance(frame: ClientGameStateFrame) -> float:
        raw_obs = frame_to_raw_obs(frame)
        dy, dx = raw_obs["closest_food"]

        if dy == 0 and dx == 0:
            grid = raw_obs["grid"]
            cy, cx = grid.shape[0] // 2, grid.shape[1] // 2

            if grid[cy, cx] != 2:
                return float("inf")

        return float(abs(dy) + abs(dx))

    @staticmethod
    def _frame_to_info(frame: ClientGameStateFrame) -> dict:
        return {
            "sequence_number": frame.sequence_number,
            "player_length": frame.player_length,
            "num_kills": frame.num_kills,
            "is_alive": frame.is_alive,
            "is_spectating": frame.is_spectating,
        }

    def close(self):
        if self.loop is not None and self.client is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.client.stop(),
                self.loop,
            )
            try:
                future.result(timeout=5.0)
            except TimeoutError:
                pass

        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread is not None:
            self.thread.join(timeout=5.0)

        self.loop = None
        self.thread = None
        self.client = None

        super().close()
