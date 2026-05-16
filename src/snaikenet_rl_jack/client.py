from pathlib import Path
from typing import Callable

import torch
from loguru import logger

from snaikenet_client.client.client_event_handler import SnaikenetClientEventHandler
from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientDirection
from snaikenet_rl_jack.agent import DoubleDQNAgent
from snaikenet_rl_jack.reward import compute_reward
from snaikenet_rl_jack.state import encode_frame


class JackAgentEventHandler(SnaikenetClientEventHandler):
    def __init__(
        self,
        device: torch.device,
        save_dir: Path,
        save_every: int = 5_000,
        optimize_every: int = 4,
        eval_only: bool = False,
        load_checkpoint: Path | None = None,
    ):
        super().__init__()
        self._device = device
        self._save_dir = save_dir
        self._save_every = save_every
        self._optimize_every = optimize_every
        self._eval_only = eval_only
        self._load_checkpoint = load_checkpoint

        self._agent: DoubleDQNAgent | None = None
        self._set_direction: Callable[[ClientDirection], None] | None = None

        self._prev_state: torch.Tensor | None = None
        self._prev_frame: ClientGameStateFrame | None = None
        self._prev_action: int | None = None

        self._frame_idx = 0
        self._episode_idx = 0
        self._episode_return = 0.0
        self._episode_length = 0

    def bind_set_direction(self, set_direction: Callable[[ClientDirection], None]):
        self._set_direction = set_direction

    def _ensure_agent(self, c: int, h: int, w: int):
        if self._agent is not None:
            return
        self._agent = DoubleDQNAgent(
            in_channels=c, height=h, width=w, device=self._device
        )
        if self._load_checkpoint and self._load_checkpoint.exists():
            self._agent.load(self._load_checkpoint)
            logger.info(
                f"Loaded checkpoint {self._load_checkpoint} at step {self._agent.step_count()}"
            )

    def on_game_state_update(self, frame: ClientGameStateFrame):
        if frame.is_spectating:
            return
        state = encode_frame(frame, self._device)
        c, h, w = state.shape
        self._ensure_agent(c, h, w)

        if (
            self._prev_state is not None
            and self._prev_frame is not None
            and self._prev_action is not None
        ):
            reward, done = compute_reward(self._prev_frame, frame)
            self._episode_return += reward
            self._episode_length += 1
            if not self._eval_only:
                self._agent.push(
                    self._prev_state, self._prev_action, reward, state, done
                )
                if self._frame_idx % self._optimize_every == 0:
                    self._agent.optimize_step()
                self._agent.increment_step()
                if (
                    self._agent.step_count() > 0
                    and self._agent.step_count() % self._save_every == 0
                ):
                    self._save_checkpoint()

        action = self._agent.select_action(state, greedy=self._eval_only)
        if self._set_direction is not None:
            self._set_direction(ClientDirection(action))

        self._prev_state = state
        self._prev_frame = frame
        self._prev_action = action
        self._frame_idx += 1

    def on_game_start(self, viewport_size: tuple[int, int]):
        logger.info(f"Game started; viewport={viewport_size}")
        self._reset_episode()

    def on_game_about_to_start(self, seconds_until_start: int):
        pass

    def on_game_end(self):
        self._log_episode_end("game_end")
        self._episode_idx += 1
        self._reset_episode()

    def on_game_restart(self):
        self._log_episode_end("game_restart")
        self._episode_idx += 1
        self._reset_episode()

    def _log_episode_end(self, kind: str):
        eps = self._agent.epsilon() if self._agent is not None else 0.0
        steps = self._agent.step_count() if self._agent is not None else 0
        logger.info(
            f"{kind} | episode={self._episode_idx} length={self._episode_length} "
            f"return={self._episode_return:.2f} eps={eps:.3f} train_steps={steps}"
        )

    def _reset_episode(self):
        self._prev_state = None
        self._prev_frame = None
        self._prev_action = None
        self._episode_return = 0.0
        self._episode_length = 0

    def _save_checkpoint(self):
        if self._agent is None:
            return
        path = self._save_dir / f"jack_dqn_{self._agent.step_count()}.pt"
        self._agent.save(path)
        logger.info(f"Saved checkpoint to {path}")
