"""
train_wrist.py
--------------
Trains WristNet on the annotated wrist-position data.

Run from the project root:
    python ml/wrist/train_wrist.py
"""

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent))
from wrist_model import WristNet, save_model, load_model

# ── config ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = "ml/wrist/features"
MODELS_DIR   = "ml/wrist/models"
WINDOW       = 31
N_FEATURES   = 198
HIDDEN       = 128
N_LAYERS     = 2
DROPOUT      = 0.3
BATCH_SIZE   = 64
LR           = 1e-3
EPOCHS       = 80
PATIENCE     = 12
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── dataset ────────────────────────────────────────────────────────────────────

class WristDataset(Dataset):
    """
    Sliding-window dataset.
    Each sample: (WINDOW, N_FEATURES) → (2,) normalized (x, y) target.
    Occluded frames (NaN targets) are excluded.
    """

    def __init__(self, video_names: list, features_dir: str, window: int = 31):
        self.window = window
        half = window // 2
        X, y = [], []
        self.frame_dims = {}   # name → (w, h) for pixel-error reporting

        for name in video_names:
            feat_path = Path(features_dir) / f"{name}_features.npy"
            tgt_path  = Path(features_dir) / f"{name}_targets.npy"
            meta_path = Path(features_dir) / f"{name}_meta.npy"

            if not feat_path.exists():
                print(f"  [skip] features not found: {name}")
                continue

            features = np.load(feat_path).astype(np.float32)   # (n, 198)
            targets  = np.load(tgt_path).astype(np.float32)    # (n, 2)

            if meta_path.exists():
                meta = np.load(meta_path, allow_pickle=True).item()
                self.frame_dims[name] = (meta.get('frame_w', 1920),
                                         meta.get('frame_h', 1080))

            n   = len(features)
            pad = np.zeros((half, N_FEATURES), dtype=np.float32)
            padded = np.concatenate([pad, features, pad], axis=0)

            for i in range(n):
                if not np.isfinite(targets[i, 0]):  # skip occluded frames
                    continue
                X.append(padded[i: i + window])
                y.append(targets[i])

        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.float32))
        print(f"  {len(self.y)} samples from {len(video_names)} video(s)")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_names(features_dir: str, split: str) -> list:
    names = []
    for p in Path(features_dir).glob("*_meta.npy"):
        meta = np.load(p, allow_pickle=True).item()
        if meta.get("split", "").lower() == split.lower():
            names.append(meta["video_name"])
    return sorted(names)


# ── training / evaluation ──────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_n    = 0
    sum_px_err = 0.0   # approx pixel error at 1920×1080 (for readability)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            total_n    += len(y)
            err_x = (pred[:, 0] - y[:, 0]).abs() * 1920
            err_y = (pred[:, 1] - y[:, 1]).abs() * 1080
            sum_px_err += torch.sqrt(err_x ** 2 + err_y ** 2).sum().item()

    return total_loss / total_n, sum_px_err / total_n


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = ("mps"  if torch.backends.mps.is_available()  else
              "cuda" if torch.cuda.is_available()           else "cpu")
    print(f"Device: {device}")

    print("\nLoading train split …")
    train_names = load_names(FEATURES_DIR, "train")
    print(f"  {len(train_names)} video(s): {train_names}")
    train_ds = WristDataset(train_names, FEATURES_DIR, WINDOW)

    print("\nLoading val split …")
    val_names = load_names(FEATURES_DIR, "val")
    print(f"  {len(val_names)} video(s): {val_names}")
    val_ds = WristDataset(val_names, FEATURES_DIR, WINDOW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cfg = dict(n_features=N_FEATURES, hidden=HIDDEN, n_layers=N_LAYERS,
               window=WINDOW, dropout=DROPOUT)
    model = WristNet(**cfg).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_px = float('inf')
    no_improve  = 0
    best_path   = Path(MODELS_DIR) / "wrist_best.pt"

    print("\nTraining …")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_px = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_px = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train loss={tr_loss:.5f} px≈{tr_px:.1f}  "
              f"val loss={vl_loss:.5f} px≈{vl_px:.1f}  "
              f"({time.time()-t0:.1f}s)")

        if vl_px < best_val_px:
            best_val_px = vl_px
            save_model(model, str(best_path), cfg)
            print(f"    ✓ saved best  (val px≈{vl_px:.1f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    print(f"\nBest val pixel error: {best_val_px:.1f} px")

    # ── test evaluation ────────────────────────────────────────────────────────
    print("\nEvaluating on test split …")
    test_names = load_names(FEATURES_DIR, "test")
    print(f"  {len(test_names)} video(s): {test_names}")
    test_ds     = WristDataset(test_names, FEATURES_DIR, WINDOW)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_model = load_model(str(best_path), device=device).to(device)
    _, test_px = run_epoch(best_model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nTest pixel error: {test_px:.1f} px  (approx at 1920×1080)")
    print(f"Model saved → {best_path}")


if __name__ == "__main__":
    main()
