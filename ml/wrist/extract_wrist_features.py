"""
extract_wrist_features.py
--------------------------
Reads annotated CSVs and converts them to normalized feature arrays for WristNet.

Run from the project root:
    python ml/wrist/extract_wrist_features.py

Outputs per video in ml/wrist/features/:
    <name>_features.npy  — (n_frames, 198) float32  (normalized pos + velocity)
    <name>_targets.npy   — (n_frames, 2)  float32  (x_norm, y_norm); NaN = occluded
    <name>_meta.npy      — dict: video_name, fps, frame_w, frame_h, split
"""

import csv
import os
import numpy as np
import cv2
from pathlib import Path

# All directories that may contain annotated CSVs
CSV_DIRS = [
    "Jun 26 Wrist Identification Training",
    "Jun 26 Wrist Identification Training/KG Wrist Tags 6-29-26",
    "labels",
]

# All directories to search for the matching video file
VIDEO_DIRS = [
    "Jun 26 Wrist Identification Training",
    "Machine Learning Videos - TAGGED",
]

FEATURES_DIR = "ml/wrist/features"

# Video-level train/val/test split (never split by frame — prevents data leakage)
SPLITS = {
    # Original 9
    '1.16.26_HA1':             'train',
    '1.19.26_VP1':             'train',
    '1.5.26_EH_Fastball1':     'train',
    '1.7.26_AL1':              'train',
    'CP_Lat_1_5.27.26':        'train',
    'EH_Lat_1_3.9.26':         'train',
    'Zara_Lat_1_3.9.26':       'train',
    'KG_Lat_1_6.18.26':        'val',
    'Cam_Lat_1_6.20.26':       'test',
    # Batch 2 (KG Wrist Tags 6-29-26)
    'Camryn_Lateral1_4.12.26': 'train',
    'Camryn_Lateral2_4.12.26': 'train',
    'Camryn_Lateral3_4.12.26': 'train',
    'HA_Lat_11_6.13.26':       'train',
    'HA_Lat_12_6.13.26':       'train',
    'HA_Lat_13_6.13.26':       'train',
    'HA_Lat_14_6.13.26':       'train',
    'HA_Lat_15_6.13.26':       'train',
    'HA_Lat_16_6.13.26':       'train',
    # Batch 3 (labels/)
    'AN_Lat_1_4.12.26':        'train',
    'AN_Lat_2_4.12.26':        'train',
    'AN_Lat_3_4.12.26':        'train',
    'EH_1_Lat_3.23.26':        'train',
    'EH_3_Lat_3.30.26':        'train',
    'EH_4_Lat_3.23.26':        'train',
    'EH_Lat_1_4.13.26':        'train',
    'EH_Lat_2_3.9.26':         'train',
    'EH_Lat_2_4.13.26':        'train',
    'HA_Lat_1_6.13.26':        'train',
}


def normalize_sequence(lm_list):
    """
    lm_list : list of (33, 3) float32 arrays or None, one per frame.
    Returns  : (n, 198) float32 — hip-centred, shoulder-width-scaled pos + velocity.
    Matches the normalization in processor.py and ml/extract_features.py exactly.
    """
    n   = len(lm_list)
    pos = np.zeros((n, 33, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for i, lm in enumerate(lm_list):
        if lm is not None:
            hip   = (lm[23] + lm[24]) / 2.0
            scale = np.linalg.norm(lm[11] - lm[12]) + 1e-6
            pos[i] = (lm - hip) / scale
            valid[i] = True

    # Linear interpolation across missing frames so velocity is well-defined
    xi = np.where(valid)[0]
    if len(xi) >= 2:
        inv = np.where(~valid)[0]
        if len(inv):
            for j in range(33):
                for k in range(3):
                    pos[inv, j, k] = np.interp(inv, xi, pos[xi, j, k])

    pos_flat = pos.reshape(n, 99)
    vel_flat = np.zeros_like(pos_flat)
    vel_flat[1:] = pos_flat[1:] - pos_flat[:-1]
    return np.concatenate([pos_flat, vel_flat], axis=1)   # (n, 198)


def get_video_dims(video_path):
    """Return (width, height, fps) from video file."""
    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps


def find_video(stem, video_dirs):
    """Search video_dirs for a file matching stem (case-insensitive, any video extension)."""
    for d in video_dirs:
        for f in Path(d).iterdir():
            if f.suffix.lower() in ('.mov', '.mp4', '.avi') and f.stem.lower() == stem.lower():
                return str(f)
    return None


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    csv_files = []
    for d in CSV_DIRS:
        csv_files.extend(sorted(Path(d).glob('*.csv')))
    print(f"Found {len(csv_files)} CSV files across {len(CSV_DIRS)} director(ies).")

    for csv_path in csv_files:
        name = csv_path.stem
        print(f"\n{name}")

        if name not in SPLITS:
            print(f"  WARN: not in SPLITS dict — skipping")
            continue

        feat_out = Path(FEATURES_DIR) / f"{name}_features.npy"
        tgt_out  = Path(FEATURES_DIR) / f"{name}_targets.npy"
        meta_out = Path(FEATURES_DIR) / f"{name}_meta.npy"

        if feat_out.exists() and tgt_out.exists():
            print(f"  cached — skipping (delete to re-extract)")
            continue

        # Need frame dimensions to normalize pixel labels → [0, 1]
        video_path = find_video(name, VIDEO_DIRS)
        if video_path is None:
            print(f"  ERROR: video file not found next to CSV")
            continue

        frame_w, frame_h, fps = get_video_dims(video_path)
        print(f"  {frame_w}x{frame_h} @ {fps:.1f} fps  split={SPLITS[name]}")

        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        n = len(rows)

        # Build per-frame landmark array from CSV columns lm0_x/y/z … lm32_x/y/z
        lm_list = []
        for row in rows:
            if all(row.get(f'lm{i}_x', '') != '' for i in range(33)):
                lm = np.array(
                    [[float(row[f'lm{i}_x']), float(row[f'lm{i}_y']), float(row[f'lm{i}_z'])]
                     for i in range(33)],
                    dtype=np.float32,
                )
                lm_list.append(lm)
            else:
                lm_list.append(None)

        # Build target array: normalized (x, y); NaN for occluded / missing
        targets = np.full((n, 2), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            if row.get('occluded') == '1' or row.get('true_wrist_x', '') == '':
                continue
            targets[i, 0] = float(row['true_wrist_x']) / frame_w
            targets[i, 1] = float(row['true_wrist_y']) / frame_h

        features = normalize_sequence(lm_list)

        np.save(feat_out, features)
        np.save(tgt_out,  targets)
        np.save(meta_out, {
            'video_name': name,
            'fps':        fps,
            'frame_w':    frame_w,
            'frame_h':    frame_h,
            'split':      SPLITS[name],
            'n_frames':   n,
        }, allow_pickle=True)

        n_valid = int(np.isfinite(targets[:, 0]).sum())
        print(f"  {n} frames | {n_valid} with labels | {n - n_valid} occluded/skipped")

    print("\nDone.")


if __name__ == '__main__':
    main()
