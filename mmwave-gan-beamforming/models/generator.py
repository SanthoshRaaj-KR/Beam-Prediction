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
            nn.Conv2d(16, 2, 3, 1, 1),                                              # [B,2,64,32]
        )

    def forward(self, z, c):
        x = self.fc(torch.cat([z, c], dim=1)).view(-1, 256, 4, 2)
        x = self.net(x)
        # Project onto the unit-Frobenius sphere, the manifold the real data
        # actually lives on: Phase 1 divides every real sample by its own
        # ||.||_F, so each has ||x||_F == 1 exactly (RMS element 1/sqrt(4096)
        # = 0.0156 over [2,64,32]).
        #
        # This replaced a final nn.Tanh(), which was a real bug. Tanh bounds
        # each element to [-1,1], putting the generator's Frobenius norm one to
        # two orders of magnitude above the real data's. Real and fake were then
        # separable by that single scalar -- measured discriminability was 1.000
        # with std 0.000 -- so the critic saturated immediately and the
        # generator got almost no usable gradient (the Wasserstein estimate rose
        # from 0.12 to 0.62 over the back half of training instead of falling).
        # Matching the norm removes the trivial feature and forces the critic to
        # compare structure. Scale carries no information here anyway: labels
        # come from an argmax that is invariant to positive scaling, and the
        # saved per-sample norms are what spectral efficiency uses.
        return x / x.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1)
