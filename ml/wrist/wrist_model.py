"""
wrist_model.py — WristNet (YOLO-residual correction model)
------------------------------------------------------------
BiLSTM reads MediaPipe landmarks for temporal context, then predicts a small
CORRECTION (delta) to YOLO's wrist prediction for the center frame.

  output = yolo_xy + delta

This forces the model to "start from YOLO" and only deviate when MediaPipe
context gives a reason to.  Zero-initialising the output layer makes delta=0
at the start of training, so the model immediately benefits from YOLO rather
than having to re-discover it from scratch.

Input tensor shape: (batch, window, 201)
  [:, :,  0:198] — MediaPipe pos+velocity (fed to LSTM)
  [:, mid, 198:200] — YOLO (x_norm, y_norm)   center frame — added as residual
  [:, mid, 200:201] — YOLO confidence           center frame — fed to head
"""

import torch
import torch.nn as nn


class WristNet(nn.Module):
    def __init__(
        self,
        n_mp_features:   int   = 198,
        n_yolo_features: int   = 3,     # kept for checkpoint compatibility
        hidden:          int   = 128,
        n_layers:        int   = 2,
        window:          int   = 31,
        dropout:         float = 0.3,
    ):
        super().__init__()
        self.window   = window
        self.mid      = window // 2
        self.n_mp     = n_mp_features

        self.lstm = nn.LSTM(
            n_mp_features, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Predicts delta (correction to add to YOLO's xy).
        # YOLO confidence (1 dim) gates how much to deviate.
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        # Zero-init: initially delta=0 so model starts from "trust YOLO"
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, window, 201)
        returns : (batch, 2) — predicted normalized (x, y)
        """
        mp      = x[:, :, :self.n_mp]                        # (B, window, 198)
        yolo_xy = x[:, self.mid, self.n_mp:self.n_mp + 2]   # (B, 2)
        yolo_c  = x[:, self.mid, self.n_mp + 2:self.n_mp + 3]  # (B, 1)

        out, _ = self.lstm(mp)
        center = out[:, self.mid, :]                          # (B, 256)

        delta = self.head(torch.cat([center, yolo_c], dim=1))  # (B, 2)
        return yolo_xy + delta


def load_model(path: str, device: str = "cpu") -> WristNet:
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if "n_mp_features" not in config:
        config = dict(
            n_mp_features=198, n_yolo_features=3,
            hidden=config.get("hidden", 128),
            n_layers=config.get("n_layers", 2),
            window=config.get("window", 31),
            dropout=config.get("dropout", 0.3),
        )
    model = WristNet(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def save_model(model: WristNet, path: str, config: dict):
    torch.save({"config": config, "state_dict": model.state_dict()}, path)
