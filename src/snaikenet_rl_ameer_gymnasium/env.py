import asyncio
import queue
import threading
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snaikenet_client.client.client import SnaikenetClient
from snaikenet_client.client.client_event_handler import SnaikenetClientEventHandler
from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientDirection, ClientTileType

VIEWPORT_SIZE = 49  # 24 * 2 + 1
VIEWPORT_RADIUS = VIEWPORT_SIZE // 2
NUM_TILE_TYPES = 5  # EMPTY=0, WALL=1, FOOD=2, SNAKE=3, OTHER_SNAKE=4


@dataclass
class _GameEvent:
    kind: str
    frame: ClientGameStateFrame | None = None
    viewport_size: tuple[int, int] | None = None


class _EventHandler(SnaikenetClientEventHandler):
    def __init__(self):
        self.event_queue: queue.Queue[_GameEvent] = queue.Queue()

    def on_game_state_update(self, frame: ClientGameStateFrame):
        if not frame.is_spectating:
            self.event_queue.put(_GameEvent(kind="frame", frame=frame))

    def on_game_end(self):
        self.event_queue.put(_GameEvent(kind="game_end"))

    def on_game_restart(self):
        self.event_queue.put(_GameEvent(kind="game_restart"))

    def on_game_start(self, viewport_size: tuple[int, int]):
        self.event_queue.put(_GameEvent(kind="game_start", viewport_size=viewport_size))

    def on_game_about_to_start(self, seconds_until_start: int):
        self.event_queue.put(_GameEvent(kind="game_about_to_start"))


class SnaikeNetEnv(gym.Env):
    """Gymnasium environment that wraps the SnaikeNET client.

    Observation space:
        Dict with:
        - "grid": Box(0, 4, shape=(VIEWPORT_SIZE, VIEWPORT_SIZE), dtype=uint8)
            The viewport grid with tile types as integers.
        - "player_length": Box(0, inf, shape=(1,), dtype=float32)
        - "num_kills": Box(0, inf, shape=(1,), dtype=float32)
        - "closest_food": Box(-VIEWPORT_RADIUS, VIEWPORT_RADIUS, shape=(2,), dtype=float32)
            (dy, dx) offset from the snake's head to the closest visible food.
            (0, 0) when no food is in the viewport.

    Action space:
        Discrete(4): NORTH=0, SOUTH=1, EAST=2, WEST=3
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        server_host: str = "localhost",
        server_tcp_port: int = 8888,
        frame_timeout: float = 5.0,
        death_penalty: float = -150.0,
        food_reward: float = 150.0,
        kill_reward: float = 100.0,
        closer_to_food_reward: float = 1.0,
        farther_from_food_penalty: float = -1.5,
        living_penalty: float = -0.1,
        penalty_multiplier_ratio: float = 1000.0,
    ):
        super().__init__()

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=NUM_TILE_TYPES - 1,
                    shape=(VIEWPORT_SIZE, VIEWPORT_SIZE),
                    dtype=np.uint8,
                ),
                "player_length": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "num_kills": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "closest_food": spaces.Box(
                    low=-VIEWPORT_RADIUS,
                    high=VIEWPORT_RADIUS,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._server_host = server_host
        self._server_tcp_port = server_tcp_port
        self._frame_timeout = frame_timeout

        # Reward shaping parameters
        self._death_penalty = death_penalty
        self._food_reward = food_reward
        self._kill_reward = kill_reward
        self._closer_to_food_reward = closer_to_food_reward
        self._farther_from_food_penalty = farther_from_food_penalty
        self._living_penalty = living_penalty
        self._penalty_multiplier_ratio = penalty_multiplier_ratio

        self._event_handler = _EventHandler()
        self._client: SnaikenetClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client_uuid: str | None = None

        self._prev_frame: ClientGameStateFrame | None = None
        self._last_frame_food_eaten = 0

    def _start_client_thread(self):
        """Start the async client in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._event_handler = _EventHandler()
        ready = threading.Event()

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._client = SnaikenetClient(
                server_host=self._server_host,
                server_tcp_port=self._server_tcp_port,
                event_handler=self._event_handler,
            )
            self._loop.run_until_complete(self._client.start(uuid=self._client_uuid))
            self._client_uuid = self._client.get_client_id()
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        ready.wait(timeout=10.0)

    def _stop_client(self):
        """Stop the client and its event loop."""
        if self._loop is not None and self._client is not None:
            asyncio.run_coroutine_threadsafe(self._client.stop(), self._loop).result(
                timeout=5.0
            )
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._client = None

    def _wait_for_game_start(self):
        """Drain events until we get a game_start or first frame while alive."""
        while True:
            try:
                event = self._event_handler.event_queue.get(timeout=self._frame_timeout)
            except queue.Empty:
                continue
            if event.kind == "game_start":
                continue
            if event.kind == "game_about_to_start":
                continue
            if event.kind == "game_restart":
                continue
            if (
                event.kind == "frame"
                and event.frame is not None
                and event.frame.is_alive
                and not event.frame.is_spectating
                and event.frame.sequence_number < 5
            ):
                return event.frame

    def _wait_for_frame(self) -> ClientGameStateFrame | None:
        """Wait for the next game state frame. Returns None on game_end."""
        while True:
            try:
                event = self._event_handler.event_queue.get(timeout=self._frame_timeout)
            except queue.Empty:
                return None
            if event.kind == "frame" and event.frame is not None:
                return event.frame
            if event.kind == "game_end":
                return None
            if event.kind == "game_restart":
                return None

    def _frame_to_obs(self, frame: ClientGameStateFrame) -> dict:
        grid = np.array(frame.grid_data, dtype=np.uint8)
        # Pad or crop to VIEWPORT_SIZE x VIEWPORT_SIZE
        h, w = grid.shape
        padded = np.zeros((VIEWPORT_SIZE, VIEWPORT_SIZE), dtype=np.uint8)
        copy_h = min(h, VIEWPORT_SIZE)
        copy_w = min(w, VIEWPORT_SIZE)
        padded[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
        dy, dx = self._closest_food_coords(frame)
        return {
            "grid": padded,
            "player_length": np.array([frame.player_length], dtype=np.float32),
            "num_kills": np.array([frame.num_kills], dtype=np.float32),
            "closest_food": np.array([dy, dx], dtype=np.float32),
        }

    def _compute_reward(
        self, prev: ClientGameStateFrame, curr: ClientGameStateFrame
    ) -> float:
        penalty_multiplier = max(
            curr.sequence_number
            / (curr.player_length * self._penalty_multiplier_ratio),
            1,
        )
        # Death penalty
        if not curr.is_alive:
            return self._death_penalty * penalty_multiplier

        reward = 0

        # Kill reward
        kill_diff = curr.num_kills - prev.num_kills
        if kill_diff > 0:
            reward += self._kill_reward * kill_diff

        # Food eaten (length increased)
        length_diff = curr.player_length - prev.player_length
        if length_diff > 0:
            reward += self._food_reward * length_diff
            self._last_frame_food_eaten = curr.sequence_number
        # Doesn't make sense to calculate food distance
        # if it ate food this turn since the food distance would
        # definitely be higher on this frame since it just ate
        # the closest food
        else:
            # Food distance shaping
            prev_food_dist = self._closest_food_distance(prev)
            curr_food_dist = self._closest_food_distance(curr)
            if prev_food_dist != float("inf") and curr_food_dist != float("inf"):
                if curr_food_dist < prev_food_dist:
                    reward += self._closer_to_food_reward
                elif curr_food_dist >= prev_food_dist:
                    reward += self._farther_from_food_penalty * penalty_multiplier
            else:
                reward += self._farther_from_food_penalty * penalty_multiplier

        return reward

    @staticmethod
    def _closest_food_coords(frame: ClientGameStateFrame) -> tuple[int, int]:
        """Return (dy, dx) offset from snake head to closest food, or (0, 0) if none visible."""
        grid = frame.grid_data
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        cy, cx = height // 2, width // 2
        best_dist = float("inf")
        best = (0, 0)
        for i in range(height):
            for j in range(width):
                if grid[i][j] == ClientTileType.FOOD:
                    dy, dx = i - cy, j - cx
                    dist = abs(dy) + abs(dx)
                    if dist < best_dist:
                        best_dist = dist
                        best = (dy, dx)
        return best

    @classmethod
    def _closest_food_distance(cls, frame: ClientGameStateFrame) -> float:
        dy, dx = cls._closest_food_coords(frame)
        if dy == 0 and dx == 0:
            # Could mean food at head (already eaten) or no food visible.
            # Check the grid to disambiguate.
            grid = frame.grid_data
            cy = len(grid) // 2
            cx = len(grid[0]) // 2 if grid else 0
            if not grid or grid[cy][cx] != ClientTileType.FOOD:
                return float("inf")
        return abs(dy) + abs(dx)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Drain any stale events
        while not self._event_handler.event_queue.empty():
            try:
                self._event_handler.event_queue.get_nowait()
            except queue.Empty:
                break

        # Start client if not running
        if self._client is None:
            self._start_client_thread()

        frame = self._wait_for_game_start()
        self._prev_frame = frame
        return self._frame_to_obs(frame), {}

    def _set_direction_threadsafe(self, direction: ClientDirection):
        """Schedule set_direction on the client's event loop to avoid cross-thread transport access."""
        if self._loop is not None and self._client is not None:
            self._loop.call_soon_threadsafe(self._client.set_direction, direction)

    def step(self, action: int):
        direction = ClientDirection(action)
        send_start = time.perf_counter()
        self._set_direction_threadsafe(direction)
        send_elapsed = time.perf_counter() - send_start

        frame = self._wait_for_frame()

        process_start = time.perf_counter()
        if (
            frame is None
            or not frame.is_alive
            or frame.sequence_number - self._last_frame_food_eaten > 1000
        ):
            # Episode is over
            if frame is not None and self._prev_frame is not None:
                obs = self._frame_to_obs(frame)
                reward = self._compute_reward(self._prev_frame, frame)
            else:
                obs = (
                    self._frame_to_obs(self._prev_frame)
                    if self._prev_frame
                    else self._observation_space_sample()
                )
                reward = self._death_penalty
            self._prev_frame = None
            info = {
                "step_processing_time": send_elapsed
                + (time.perf_counter() - process_start)
            }
            return obs, reward, True, False, info

        reward = (
            self._compute_reward(self._prev_frame, frame) if self._prev_frame else 0.0
        )
        self._prev_frame = frame
        obs = self._frame_to_obs(frame)
        info = {
            "step_processing_time": send_elapsed + (time.perf_counter() - process_start)
        }
        return obs, reward, False, False, info

    def _observation_space_sample(self):
        return {
            "grid": np.zeros((VIEWPORT_SIZE, VIEWPORT_SIZE), dtype=np.uint8),
            "player_length": np.array([0], dtype=np.float32),
            "num_kills": np.array([0], dtype=np.float32),
            "closest_food": np.array([0, 0], dtype=np.float32),
        }

    def close(self):
        self._stop_client()
        super().close()
