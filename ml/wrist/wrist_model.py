"""
wrist_model.py — WristNet
--------------------------
Bidirectional LSTM that predicts the normalized (x, y) wrist position
for the CENTER frame of a sliding window of pose-landmark features.

Same backbone as KeyMomentNet (ml/model.py) with a regression head instead
of a classification head.
"""

import torch
import torch.nn as nn


class WristNet(nn.Module):
    def __init__(
        self,
        n_features: int = 198,   # 99 normalized positions + 99 frame-diff velocities
        hidden:     int = 128,
        n_layers:   int = 2,
        window:     int = 31,    # frames per input window (~0.5 s at 60 fps)
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
            nn.Linear(64, 2),   # (x_norm, y_norm) in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, window, n_features)
        returns : (batch, 2) — predicted normalized (x, y)
        """
        out, _ = self.lstm(x)
        center = out[:, self.mid, :]
        return self.head(center)


def load_model(path: str, device: str = "cpu") -> WristNet:
    ckpt  = torch.load(path, map_location=device)
    model = WristNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def save_model(model: WristNet, path: str, config: dict):
    torch.save({"config": config, "state_dict": model.state_dict()}, path)
