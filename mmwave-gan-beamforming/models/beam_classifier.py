"""CNN beam classifier (Phase 3). FROZEN -- all 21 ablation runs use this unchanged."""
import torch.nn as nn


class BeamCNN(nn.Module):
    """~4.3M params, dominated by the first Linear. Seconds/epoch on a T4.

    n_beams=32 for the beam task; n_beams=2 reuses it as the Phase 6
    real-vs-synthetic discriminator.
    """

    def __init__(self, n_beams=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            # [B,32,64,32]
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # [B,64,32,16]
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # [B,128,16,8]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # [B, 16384]
            nn.Linear(128 * 16 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_beams),                       # logits, NO softmax
        )

    def forward(self, x):
        return self.classifier(self.features(x))
