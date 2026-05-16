import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 5, 49, 49)
            conv_dim = self.conv(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(conv_dim + 4, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, grid, stats):
        x = self.conv(grid)
        x = torch.cat([x, stats], dim=1)
        return self.head(x)