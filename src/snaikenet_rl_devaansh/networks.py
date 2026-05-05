import torch
import torch.nn as nn

class ActorCritic(nn.Module):

    """
    Shared CNN backbone with separate Actor and Critic heads.

    Input shape: (batch, 5 * n_frames, H, W)
    Actor output: (batch, 3)   - logits over straight/left/right
    Critic output: (batch, 1)  - scalar state value
    """

    def __init__(self, in_channels: int, n_actions: int = 3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        """
        We need to computer the flattened size after the CNN dynamically 
        because viewport size may vary. We do a dummy forward pass at init
        """

        self._cnn_out_size: int | None = None

        self.fc = nn.Sequential(
            nn.LazyLinear(256),   # LazyLinear infers input size pn first forward pass
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(256, n_actions)   # outputs logits
        self.critic_head = nn.Linear(256, 1)    # outputs scalar value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.fc(self.cnn(x))
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(
            self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return (action, log_prob, entropy, value).

        If 'action' is provided (during PPO update), evaluates that action
        instead of sampling a new one.
        """
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value