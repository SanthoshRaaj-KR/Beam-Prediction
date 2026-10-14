"""WGAN-GP critic + gradient penalty (Phase 4).

NO BatchNorm in the critic -- it couples samples within a batch and makes the
gradient penalty ill-defined. GroupNorm(1, C) is LayerNorm-equivalent here and
avoids hardcoding exact [C,H,W] shapes. BatchNorm is not an acceptable swap.
"""
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, cond_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 + cond_dim, 32, 4, 2, 1), nn.LeakyReLU(0.2),                 # [B,32,32,16]
            nn.Conv2d(32,  64, 4, 2, 1), nn.GroupNorm(1,  64), nn.LeakyReLU(0.2),    # [B,64,16,8]
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(1, 128), nn.LeakyReLU(0.2),    # [B,128,8,4]
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(1, 256), nn.LeakyReLU(0.2),   # [B,256,4,2]
            nn.Flatten(), nn.Linear(256 * 4 * 2, 1),                                 # scalar, no sigmoid
        )

    def forward(self, x, c):
        # broadcast the condition as constant spatial channels
        cmap = c.view(-1, c.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, cmap], dim=1))


def gradient_penalty(critic, real, fake, cond, device):
    """Gulrajani et al. gradient penalty. create_graph=True is required."""
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(interp, cond)
    grads = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
