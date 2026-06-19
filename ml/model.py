"""
model.py — KeyMomentNet
-----------------------
Bidirectional LSTM that classifies the CENTER frame of a fixed-length window
of pose-landmark features into one of 5 classes:
    0 = background
    1 = foot_lift
    2 = foot_peak
    3 = foot_contact
    4 = ball_release
"""

import torch
import torch.nn as nn


class KeyMomentNet(nn.Module):
    def __init__(
        self,
        n_features: int = 198,   # 99 positions + 99 velocities
        hidden:     int = 128,
        n_layers:   int = 2,
        n_classes:  int = 5,
        window:     int = 45,    # frames in each input window
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.window = window
        self.mid    = window // 2

        self.lstm = nn.LSTM(
            n_features, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, window, n_features)
        returns : (batch, n_classes) logits
        """
        out, _ = self.lstm(x)           # (batch, window, hidden*2)
        center  = out[:, self.mid, :]   # center frame representation
        return self.head(center)


def load_model(path: str, device="cpu") -> KeyMomentNet:
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["config"]
    model = KeyMomentNet(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def save_model(model: KeyMomentNet, path: str, config: dict):
    torch.save({"config": config, "state_dict": model.state_dict()}, path)
