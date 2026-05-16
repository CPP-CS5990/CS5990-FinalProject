import torch


class RolloutBuffer:
    """
    Stores a fixed-length trajectory of experience for PPO updates.

    Collects 'capacity' steps (across however many episodes/deaths),
    then computes advantages via GAE and exposes mini-batches for training.
    """

    def __init__(
        self, capacity: int = 512, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._ptr = 0

        # Pre-allocated as plain lists; converted to tensors before training
        self.states: list[torch.Tensor] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.values.append(value)
            self.dones.append(done)
        else:
            # Overwrite oldest entry (entry buffer)
            idx = self._ptr % self.capacity
            self.states[idx] = state
            self.actions[idx] = action
            self.log_probs[idx] = log_prob
            self.rewards[idx] = reward
            self.values[idx] = value
            self.dones[idx] = done

        self._ptr += 1

    def is_full(self) -> bool:
        return self._ptr >= self.capacity

    def compute_advantages(
        self, last_value: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_value: Critic's value estimate for the state AFTER the last stored step
                        Pass 0.0 if the last step was terminal (done = true)

        Returns:
            advantages: (capacity,) tensor
            returns:    (capacity,) tensor     (used as critic targets)
        """
        advantages = torch.zeros(self.capacity)
        gae = 0.0

        for t in reversed(range(self.capacity)):
            next_value = last_value if t == self.capacity - 1 else self.values[t + 1]
            non_terminal = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + self.gamma * next_value * non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae

        returns = advantages + torch.tensor(self.values)
        return advantages, returns

    def get_tensors(self) -> dict[str, torch.Tensor]:
        """
        Convert stored lists to stacked tensors for training
        """
        return {
            "states": torch.stack(self.states),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "log_probs": torch.tensor(self.log_probs),
            "values": torch.tensor(self.values),
        }

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self._ptr = 0
