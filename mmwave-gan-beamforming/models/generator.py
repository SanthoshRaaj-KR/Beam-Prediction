"""WGAN-GP generator (Phase 4). Shape math verified: lands exactly on [2,64,32].

4x2 seed + four stride-2 doublings:
  antenna axis    4 -> 8 -> 16 -> 32 -> 64
  subcarrier axis 2 -> 4 ->  8 -> 16 -> 32
Do NOT revert to a 4x4 seed -- that lands on a square 32x32 and needs a crop hack.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, cond_dim=2):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.fc = nn.Linear(z_dim + cond_dim, 256 * 4 * 2)      # -> [B,256,4,2]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # [B,128,8,4]
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(),  # [B,64,16,8]
            nn.ConvTranspose2d(64,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(),  # [B,32,32,16]
            nn.ConvTranspose2d(32,   16, 4, 2, 1), nn.BatchNorm2d(16),  nn.ReLU(),  # [B,16,64,32]
            nn.Conv2d(16, 2, 3, 1, 1), nn.Tanh(),                                   # [B,2,64,32]
        )

    def forward(self, z, c):
        x = self.fc(torch.cat([z, c], dim=1)).view(-1, 256, 4, 2)
        return self.net(x)
