"""
train.py
--------
Loads extracted features from ml/features/, builds a sliding-window dataset,
trains KeyMomentNet, and saves the best checkpoint to ml/models/.

Run from the project root:
    python ml/train.py

Hyperparameters can be overridden at the top of this file.
"""

import os, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import KeyMomentNet, save_model

# ── config ────────────────────────────────────────────────────────────────────
FEATURES_DIR = "ml/features"
MODELS_DIR   = "ml/models"
WINDOW       = 45          # frames on each side
N_FEATURES   = 198
N_CLASSES    = 5
HIDDEN       = 128
N_LAYERS     = 2
DROPOUT      = 0.3
BATCH_SIZE   = 64
LR           = 1e-3
EPOCHS       = 60
PATIENCE     = 10          # early stopping patience (epochs without val improvement)
SEED         = 42

LABEL_NAMES = ["background", "foot_lift", "foot_peak", "foot_contact", "ball_release"]

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── dataset ───────────────────────────────────────────────────────────────────

class KeyMomentDataset(Dataset):
    """
    Sliding-window dataset.  Each sample is a (WINDOW, N_FEATURES) window
    whose center frame has a non-ignored label (label != -1).
    """

    def __init__(self, video_names: list[str], features_dir: str, window: int = 45):
        self.window = window
        half = window // 2
        X, y = [], []

        for name in video_names:
            lm_path = Path(features_dir) / f"{name}_landmarks.npy"
            lb_path = Path(features_dir) / f"{name}_labels.npy"
            if not lm_path.exists():
                print(f"  [skip] features not found: {name}")
                continue

            features = np.load(lm_path).astype(np.float32)   # (n, 198)
            labels   = np.load(lb_path).astype(np.int64)      # (n,)
            n        = len(features)

            # Pad with zeros so every labelled frame can have a full window
            pad = np.zeros((half, N_FEATURES), dtype=np.float32)
            padded = np.concatenate([pad, features, pad], axis=0)  # (n+window-1, 198)

            for i in range(n):
                if labels[i] == -1:
                    continue
                window_data = padded[i: i + window]  # (window, 198)
                X.append(window_data)
                y.append(int(labels[i]))

        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))
        print(f"  Dataset: {len(self.y)} samples | class dist: "
              + " | ".join(f"{LABEL_NAMES[c]}={int((self.y==c).sum())}" for c in range(N_CLASSES)))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_video_names(features_dir: str, split: str) -> list[str]:
    """Return video stems whose meta matches the requested split."""
    names = []
    for p in Path(features_dir).glob("*_meta.npy"):
        meta = np.load(p, allow_pickle=True).item()
        s = (meta.get("split") or "").strip().lower()
        if s.startswith(split.lower()):
            names.append(meta["video_name"])
    return names


def compute_class_weights(dataset: KeyMomentDataset) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    counts = torch.bincount(dataset.y, minlength=N_CLASSES).float()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (N_CLASSES * counts)
    return weights


# ── training / evaluation ─────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


def per_class_accuracy(model, loader, device):
    model.eval()
    counts  = torch.zeros(N_CLASSES, device=device)
    correct = torch.zeros(N_CLASSES, device=device)
    with torch.no_grad():
        for X, y in loader:
            X, y   = X.to(device), y.to(device)
            preds  = model(X).argmax(1)
            for c in range(N_CLASSES):
                mask = y == c
                counts[c]  += mask.sum()
                correct[c] += (preds[mask] == c).sum()
    accs = (correct / counts.clamp(min=1)).tolist()
    for c, (name, acc) in enumerate(zip(LABEL_NAMES, accs)):
        print(f"    {name:<15} n={int(counts[c]):>4}  acc={acc:.1%}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading train split …")
    train_names = load_video_names(FEATURES_DIR, "train")
    print(f"  {len(train_names)} train videos")
    train_ds = KeyMomentDataset(train_names, FEATURES_DIR, WINDOW)

    print("\nLoading val split …")
    val_names = load_video_names(FEATURES_DIR, "val")
    print(f"  {len(val_names)} val videos")
    val_ds = KeyMomentDataset(val_names, FEATURES_DIR, WINDOW)

    weights = compute_class_weights(train_ds).to(device)
    print(f"\nClass weights: {weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cfg = dict(n_features=N_FEATURES, hidden=HIDDEN, n_layers=N_LAYERS,
               n_classes=N_CLASSES, window=WINDOW, dropout=DROPOUT)
    model = KeyMomentNet(**cfg).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    no_improve   = 0
    best_path    = Path(MODELS_DIR) / "key_moments_best.pt"

    print("\nTraining …")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train loss={train_loss:.4f} acc={train_acc:.1%}  "
              f"val loss={val_loss:.4f} acc={val_acc:.1%}  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, str(best_path), cfg)
            print(f"    ✓ saved best model (val_acc={val_acc:.1%})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping after {epoch} epochs.")
                break

    print(f"\nBest val accuracy: {best_val_acc:.1%}")

    # ── test evaluation ──────────────────────────────────────────────────────
    print("\nLoading test split for final evaluation …")
    test_names = load_video_names(FEATURES_DIR, "test")
    test_ds    = KeyMomentDataset(test_names, FEATURES_DIR, WINDOW)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    from model import load_model
    best_model = load_model(str(best_path), device=device).to(device)
    test_loss, test_acc = run_epoch(best_model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.1%}")
    print("\nPer-class accuracy on test set:")
    per_class_accuracy(best_model, test_loader, device)
    print(f"\nModel saved to: {best_path}")


if __name__ == "__main__":
    main()
