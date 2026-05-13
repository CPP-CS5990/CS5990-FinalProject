import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from snaikenet_rl_ameer_gymnasium.env import VIEWPORT_SIZE


class DQN(nn.Module):
    """Deep Q-Network for the SnaikeNET environment.

    Processes the grid through convolutional layers and combines with
    scalar features (player_length, num_kills) before outputting Q-values.
    """

    def __init__(self, num_actions: int = 4):
        super().__init__()

        # Grid processing: input is 1-channel VIEWPORT_SIZE x VIEWPORT_SIZE
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate conv output size
        dummy = torch.zeros(1, 1, VIEWPORT_SIZE, VIEWPORT_SIZE)
        conv_out_size = self.conv(dummy).view(1, -1).shape[1]

        # Combine conv features with scalar features (length, kills, closest_food dy, dx)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = self.conv(grid)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, scalars], dim=1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, state: dict, action: int, reward: float, next_state: dict, done: bool):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(list(self._buffer), batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


def obs_to_tensors(obs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a gymnasium observation dict to tensors for the DQN."""
    grid = torch.from_numpy(obs["grid"]).float().unsqueeze(0) / 4.0  # normalize to [0, 1]
    scalars = torch.cat([
        torch.from_numpy(obs["player_length"]),
        torch.from_numpy(obs["num_kills"]),
        torch.from_numpy(obs["closest_food"]),
    ]).float()
    return grid.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device)


def batch_obs_to_tensors(batch: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of observations to tensors."""
    grids = torch.stack([
        torch.from_numpy(obs["grid"]).float().unsqueeze(0) / 4.0
        for obs in batch
    ]).to(device)
    scalars = torch.stack([
        torch.cat([
            torch.from_numpy(obs["player_length"]),
            torch.from_numpy(obs["num_kills"]),
            torch.from_numpy(obs["closest_food"]),
        ]).float()
        for obs in batch
    ]).to(device)
    return grids, scalars


class DQNAgent:
    def __init__(
        self,
        device: torch.device | None = None,
        lr: float = 5e-4,
        gamma: float = 0.90,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500_000,
        buffer_capacity: int = 200_000,
        batch_size: int = 64,
        target_update_freq: int = 10_000,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.steps_done = 0

    def select_action(self, obs: dict) -> int:
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * max(
            0, 1 - self.steps_done / self.epsilon_decay
        )
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            grids, scalars = obs_to_tensors(obs, self.device)
            q_values = self.policy_net(grids, scalars)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state: dict, action: int, reward: float, next_state: dict, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Perform one training step. Returns loss or None if buffer too small."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_grids, state_scalars = batch_obs_to_tensors(list(states), self.device)
        next_grids, next_scalars = batch_obs_to_tensors(list(next_states), self.device)

        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values
        q_values = self.policy_net(state_grids, state_scalars).gather(1, actions_t).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_grids, next_scalars).max(dim=1).values
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str, extra: dict | None = None):
        data = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
        }
        if extra:
            data.update(extra)
        torch.save(data, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.epsilon = checkpoint["epsilon"]