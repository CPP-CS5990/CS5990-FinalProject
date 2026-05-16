import random
from pathlib import Path

import torch
import torch.nn.functional as F

from snaikenet_rl_jack.network import QNetwork, NUM_ACTIONS
from snaikenet_rl_jack.replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        device: torch.device,
        gamma: float = 0.95,
        lr: float = 5e-4,
        batch_size: int = 64,
        replay_capacity: int = 50_000,
        min_buffer: int = 1_000,
        target_sync_every: int = 500,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 15_000,
    ):
        self._device = device
        self._online = QNetwork(in_channels, height, width).to(device)
        self._target = QNetwork(in_channels, height, width).to(device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optimizer = torch.optim.Adam(self._online.parameters(), lr=lr)
        self._buffer = ReplayBuffer(replay_capacity)

        self._gamma = gamma
        self._batch_size = batch_size
        self._min_buffer = min_buffer
        self._target_sync_every = target_sync_every
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay_steps = eps_decay_steps

        self._step_count = 0

    def epsilon(self) -> float:
        frac = min(self._step_count / self._eps_decay_steps, 1.0)
        return self._eps_start + frac * (self._eps_end - self._eps_start)

    def select_action(self, state: torch.Tensor, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon():
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            q = self._online(state.to(self._device))
        return int(q.argmax(dim=1).item())

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        self._buffer.push(
            state.detach().cpu(), action, reward, next_state.detach().cpu(), done
        )

    def optimize_step(self) -> float | None:
        if len(self._buffer) < max(self._batch_size, self._min_buffer):
            return None
        states, actions, rewards, next_states, dones = self._buffer.sample(
            self._batch_size
        )
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device)

        q_sa = self._online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self._online(next_states).argmax(dim=1, keepdim=True)
            next_q = self._target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self._gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, target)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self._optimizer.step()
        return float(loss.item())

    def increment_step(self):
        self._step_count += 1
        if self._step_count % self._target_sync_every == 0:
            self._sync_target()

    def _sync_target(self):
        self._target.load_state_dict(self._online.state_dict())

    def step_count(self) -> int:
        return self._step_count

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self._online.state_dict(),
                "target": self._target.state_dict(),
                "step_count": self._step_count,
            },
            path,
        )

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self._device)
        self._online.load_state_dict(ckpt["online"])
        self._target.load_state_dict(ckpt["target"])
        self._step_count = ckpt.get("step_count", 0)
