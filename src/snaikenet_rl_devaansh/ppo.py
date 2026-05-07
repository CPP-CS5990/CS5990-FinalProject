import torch
import torch.nn as nn
import torch.optim as optim

from snaikenet_rl_devaansh.networks import ActorCritic
from snaikenet_rl_devaansh.rollout_buffer import RolloutBuffer

class PPO:

    """
    PPO update logic.

    Hyperparameters:
        lr          - learning rate
        clip_eps    - PPO clipping epsilon (how far new policy can deviate from old)
        n_epochs    - how many passes over the rollout buffer per update
        batch_size  - mini-batch size for each gradient step
        vf_coef     - weight of critic (value) loss relative to actor loss
        ent_coef    - weight of entropy bonus (encourages exploration)
    """

    def __init__(
            self,
            network: ActorCritic,
            lr: float = 3e-4,
            clip_eps: float = 0.2,
            n_epochs: int = 4,
            batch_size: int = 64,
            vf_coef: float = 0.5,
            ent_coef: float = 0.02,
    ):
        self.network = network
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer, last_value: float):
        """
        Run PPO update using the data currently in 'buffer'.

        Steps:
            1. Compute GAE (Generalized Advantage Estimation) advantages and returns from the buffer.
            2. For n_epochs, shuffle the buffer and iterate in mini-batches
            3. For each mini-batch:
                a) Re-evaluate actions under the current policy.
                b) Compute clipped actor loss (PPO objective).
                c) Compute critic (value) loss.
                d) Compute entropy bonus.
                e) Backprop the combined loss.
            4. Clear the buffer
        """

        advantages, returns = buffer.compute_advantages(last_value)

        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        tensors =  buffer.get_tensors()
        states = tensors["states"]
        actions = tensors["actions"]
        old_log_probs = tensors["log_probs"]

        n = len(states)

        for _ in range(self.n_epochs):
            # Shuffle indices for each epoch
            indices = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_lp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Re-evaluate under current policy
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surrogate1 = ratio * batch_adv
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Critic loss
                critic_loss = nn.functional.mse_loss(new_values, batch_returns)

                # Entropy bonus (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm = 0.5)
                self.optimizer.step()

        buffer.clear()
