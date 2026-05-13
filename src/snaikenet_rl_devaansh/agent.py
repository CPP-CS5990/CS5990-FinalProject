import time
import torch

from snaikenet_client.client.client_event_handler import DefaultSnaikenetClientEventHandler
from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientDirection

from snaikenet_rl_devaansh.networks import ActorCritic
from snaikenet_rl_devaansh.preprocessing import FrameStacker, NUM_TILE_TYPES
from snaikenet_rl_devaansh.rollout_buffer import RolloutBuffer
from snaikenet_rl_devaansh.ppo import PPO
from snaikenet_rl_devaansh.reward import compute_reward

N_FRAMES     = 2        # number of frames to stack
ROLLOUT_SIZE = 256      # steps collected before each PPO update

# Relative actions: 0 = straight, 1 = turn left, 2 = turn right
_TURN_LEFT = {
    ClientDirection.NORTH: ClientDirection.WEST,
    ClientDirection.WEST:  ClientDirection.SOUTH,
    ClientDirection.SOUTH: ClientDirection.EAST,
    ClientDirection.EAST:  ClientDirection.NORTH,
}
_TURN_RIGHT = {
    ClientDirection.NORTH: ClientDirection.EAST,
    ClientDirection.EAST:  ClientDirection.SOUTH,
    ClientDirection.SOUTH: ClientDirection.WEST,
    ClientDirection.WEST:  ClientDirection.NORTH,
}

def _resolve_action(action_idx: int, current_dir: ClientDirection) -> ClientDirection:
    if action_idx == 1:
        return _TURN_LEFT[current_dir]
    if action_idx == 2:
        return _TURN_RIGHT[current_dir]
    return current_dir


class PPOAgentEventHandler(DefaultSnaikenetClientEventHandler):
    """
    Connects the game client event loop to the PPO training pipeline

    - on_game_start         : initialise frame stacker (viewport size known here)
    - one_game_state_update : core step - preprocess --> act --> store --> maybe train
    - on_game_restart       : mark episode boundary; keep buffer + networks intact
    - on_game_end           : same as restart
    """

    def __init__(self):
        # Networks and PPO trainer - created once, never reset
        self.network: ActorCritic | None = None
        self.ppo: PPO | None = None
        self.buffer = RolloutBuffer(capacity=ROLLOUT_SIZE)

        self.stacker: FrameStacker = FrameStacker(n_frames=N_FRAMES)
        self._prev_frame: ClientGameStateFrame | None = None
        self._prev_state: torch.Tensor | None = None
        self._prev_action: int | None = None
        self._prev_log_prob: float | None = None
        self._prev_value: float | None = None

        self.update_count: int = 0
        self.session_start_update: int = 0
        self._current_dir: ClientDirection = ClientDirection.WEST

        self._episode_steps: int = 0
        self._episode_food: int = 0
        self._rollout_episodes: list[tuple[int, int]] = []
        self.metrics: list[tuple[int, float, float]] = []

        # Callback set by __main__ so the handler can send directions to the server
        self.send_direction = None   # type: ignore[assignment]
        # Callback invoked after each PPO update — set by __main__ for checkpointing
        self.on_ppo_update = None   # type: ignore[assignment]

    def on_game_start(self, viewport_size: tuple[int, int]):
        in_channels = NUM_TILE_TYPES * N_FRAMES  # 5 * 2 = 10

        if self.network is None:
            # First game - create networks
            self.network = ActorCritic(in_channels=in_channels)
            self.ppo = PPO(self.network)

        # Reset episode-level state (not the network buffer)
        self._prev_frame = None
        self._prev_state = None

    def on_game_state_update(self, frame: ClientGameStateFrame):
        # Tick per second counter
        # t0 = time.perf_counter()
        if self.network is None:
            return      # not initialized yet

        # 1. preprocess current frame
        if self._prev_state is None:
            curr_state = self.stacker.reset(frame)
        else:
            curr_state = self.stacker.step(frame)

        # 2. Store transition from the PREVIOUS step into the buffer
        if self._prev_frame is not None:
            # Episode metric tracking
            if self._prev_frame.is_alive:
                self._episode_steps += 1
                if frame.is_alive and frame.player_length > self._prev_frame.player_length:
                    self._episode_food += 1
            if not frame.is_alive and self._prev_frame.is_alive:
                self._rollout_episodes.append((self._episode_steps, self._episode_food))
                self._episode_steps = 0
                self._episode_food = 0

            reward = compute_reward(self._prev_frame, frame)
            done = not frame.is_alive

            self.buffer.add(
                state    = self._prev_state,
                action   = self._prev_action,
                log_prob = self._prev_log_prob,
                reward   = reward,
                value    = self._prev_value,
                done     = done,
            )

            # If buffer is full, run a PPO update
            if self.buffer.is_full():
                with torch.no_grad():
                    _, last_value = self.network.forward(curr_state.unsqueeze(0))
                last_val = last_value.item() if frame.is_alive else 0.0
                self.ppo.update(self.buffer, last_value=last_val)
                self.update_count += 1

                episodes = self._rollout_episodes or [(self._episode_steps, self._episode_food)]
                avg_steps = sum(e[0] for e in episodes) / len(episodes)
                avg_food  = sum(e[1] for e in episodes) / len(episodes)
                self.metrics.append((self.update_count, avg_steps, avg_food))
                self._rollout_episodes.clear()

                if self.on_ppo_update is not None:
                    self.on_ppo_update(self.update_count)

        # 3. Select action for the current state
        if frame.is_alive:
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(
                    curr_state.unsqueeze(0)
                )
            action_idx  = action.item()
            log_prob_val = log_prob.item()
            value_val    = value.item()

            resolved_dir = _resolve_action(action_idx, self._current_dir)
            self._current_dir = resolved_dir
            if self.send_direction is not None:
                self.send_direction(resolved_dir)
        else:
            # Dead this tick - pick a placeholder; won't be acted on
            action_idx, log_prob_val, value_val = 0, 0.0, 0.0

        # 4. Cache for next step
        self._prev_frame    = frame
        self._prev_state    = curr_state
        self._prev_action   = action_idx
        self._prev_log_prob = log_prob_val
        self._prev_value    = value_val

        """ 
        tick interval ms = elapsed_ms * 15
        TPS calculation formula: 1000/tick interval ms
        """

        # elapsed_ms = (time.perf_counter() - t0) * 1000
        # print(f"[agent] response: {elapsed_ms:.2f}ms")

    def on_game_restart(self):
        # New episode begins - reset episode state but keep buffer + networks
        if self._episode_steps > 0:
            self._rollout_episodes.append((self._episode_steps, self._episode_food))
        self._episode_steps = 0
        self._episode_food = 0
        self._prev_frame = None
        self._prev_state = None
        self._current_dir = ClientDirection.WEST

    def on_game_end(self):
        self.on_game_restart()