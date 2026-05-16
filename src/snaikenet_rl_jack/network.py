import torch
import torch.nn as nn

from snaikenet_client.types import ClientDirection

NUM_ACTIONS = len(ClientDirection)


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat_h = height // 4
        flat_w = width // 4
        flat = 32 * flat_h * flat_w
        self.head = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        z = self.conv(x)
        z = z.flatten(start_dim=1)
        return self.head(z)
