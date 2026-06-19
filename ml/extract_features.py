"""
extract_features.py
-------------------
Reads Key Moments Video Frame Labelling.xlsx, runs MediaPipe PoseLandmarker
on every tagged video, and saves per-video feature + label arrays to ml/features/.

Run from the project root:
    python ml/extract_features.py

Outputs (for each video):
    ml/features/<name>_landmarks.npy  — (n_frames, 198) float32
                                        first 99: normalized (x,y,z) per landmark
                                        last  99: frame-to-frame velocity
    ml/features/<name>_labels.npy     — (n_frames,) int8
                                        -1 = outside pitch window (ignored in training)
                                         0 = background
                                         1 = foot_lift
                                         2 = foot_peak
                                         3 = foot_contact
                                         4 = ball_release
    ml/features/<name>_meta.npy       — dict with fps, split, pitcher_id, moments_ms, etc.
"""

import os, sys, re, datetime
import numpy as np
import cv2
from pathlib import Path
from numbers_parser import Document

# MediaPipe Tasks API (same as processor.py)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── paths (relative to project root) ─────────────────────────────────────────
SPREADSHEET  = "Key Moments Video Frame Labelling.numbers"
VIDEO_DIR    = "Machine Learning Videos - TAGGED"
FEATURES_DIR = "ml/features"
MODEL_PATH   = "pose_landmarker_heavy.task"

# ── label constants ───────────────────────────────────────────────────────────
BACKGROUND   = 0
FOOT_LIFT    = 1
FOOT_PEAK    = 2
FOOT_CONTACT = 3
BALL_RELEASE = 4
IGNORE       = -1

MOMENT_COLS   = ["foot_lift_timestamp", "foot_peak_timestamp",
                 "foot_contact_timestamp", "ball_release_timestamp"]
MOMENT_IDS    = [FOOT_LIFT, FOOT_PEAK, FOOT_CONTACT, BALL_RELEASE]
LABEL_TOLERANCE = 2   # ±2 frames labelled as each key moment (~33 ms at 60 fps)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_ms(value):
    """Parse a timestamp value → integer milliseconds.
    Handles:
      '2,494 ms'              → 2494  (string with commas/spaces)
      datetime.timedelta(…)   → total milliseconds (Numbers duration cell)
    """
    if value is None:
        return None
    if isinstance(value, datetime.timedelta):
        return int(value.total_seconds() * 1000)
    s = str(value).strip()
    s = re.sub(r'[,\s]', '', s)
    s = re.sub(r'ms', '', s, flags=re.I)
    try:
        return int(float(s))
    except ValueError:
        return None


def find_video(name, video_dir):
    """Locate video file by stem (case-insensitive, any extension).
    Also tries with extra/missing dots removed so typos like
    'AN_Lat_2_.4.12.26' still match 'AN_Lat_2_4.12.26'.
    """
    import re as _re
    def norm(s):
        return _re.sub(r'\.{2,}|(?<=_)\.', '', s).lower()

    target = name.lower()
    target_norm = norm(name)
    for f in Path(video_dir).iterdir():
        if f.stem.lower() == target or norm(f.stem) == target_norm:
            return str(f)
    return None


def read_spreadsheet(path):
    """Return list of row dicts from Sheet2 of a Numbers file."""
    doc = Document(path)
    sheet = next(s for s in doc.sheets if s.name == "Sheet2")
    table = sheet.tables[0]
    rows = list(table.iter_rows(values_only=True))
    headers = rows[0]
    data = []
    for row in rows[1:]:
        if any(v is not None for v in row):
            data.append(dict(zip(headers, row)))
    return data


def normalize_sequence(raw):
    """
    raw : list of (33,3) arrays or None — per-frame MediaPipe landmarks
    Returns (n_frames, 198) float32:
        cols 0–98  : positions normalized to hip-center & shoulder width
        cols 99–197: frame-difference velocity (zero for first frame)
    """
    n = len(raw)
    pos = np.zeros((n, 33, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for i, lm in enumerate(raw):
        if lm is not None:
            hip_center = (lm[23] + lm[24]) / 2.0            # landmark 23=L_HIP 24=R_HIP
            scale = np.linalg.norm(lm[11] - lm[12]) + 1e-6  # shoulder width
            pos[i] = (lm - hip_center) / scale
            valid[i] = True

    # Interpolate missing frames so velocity is well-defined
    if valid.sum() >= 2:
        xi = np.where(valid)[0]
        for j in range(33):
            for k in range(3):
                pos[np.where(~valid)[0], j, k] = np.interp(
                    np.where(~valid)[0], xi, pos[xi, j, k])

    pos_flat = pos.reshape(n, 99)                            # (n, 99)
    vel_flat = np.zeros_like(pos_flat)
    vel_flat[1:] = pos_flat[1:] - pos_flat[:-1]             # (n, 99)

    return np.concatenate([pos_flat, vel_flat], axis=1)      # (n, 198)


def make_labels(moments_ms, fps, n_frames, pitch_start_ms, pitch_end_ms):
    """Build per-frame label array (-1 / 0-4)."""
    labels = np.full(n_frames, IGNORE, dtype=np.int8)

    # Pitch window → background
    s = max(0, int(pitch_start_ms / 1000 * fps) - 10)
    e = min(n_frames - 1, int(pitch_end_ms  / 1000 * fps) + 10)
    labels[s:e + 1] = BACKGROUND

    # Key moments overwrite background
    for ts_ms, cls in zip(moments_ms, MOMENT_IDS):
        if ts_ms is None:
            continue
        cf = int(round(ts_ms / 1000 * fps))
        lo = max(0, cf - LABEL_TOLERANCE)
        hi = min(n_frames - 1, cf + LABEL_TOLERANCE)
        labels[lo:hi + 1] = cls

    return labels


def run_mediapipe(video_path):
    """Run PoseLandmarker on all frames. Returns (list_of_(33,3)_or_None, fps)."""
    base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    landmarker = vision.PoseLandmarker.create_from_options(opts)
    results = []
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = int(frame_idx / fps * 1000)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = landmarker.detect_for_video(mp_img, ts_ms)
            if det.pose_landmarks:
                lm = det.pose_landmarks[0]
                coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
            else:
                coords = None
            results.append(coords)
            frame_idx += 1
    finally:
        landmarker.close()
        cap.release()

    return results, fps


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    rows = read_spreadsheet(SPREADSHEET)
    print(f"Found {len(rows)} rows in spreadsheet.")

    ok = skip = fail = 0
    for row in rows:
        name = row["video_path"]
        print(f"\n[{ok+skip+fail+1}/{len(rows)}] {name}")

        out_lm   = Path(FEATURES_DIR) / f"{name}_landmarks.npy"
        out_lb   = Path(FEATURES_DIR) / f"{name}_labels.npy"
        out_meta = Path(FEATURES_DIR) / f"{name}_meta.npy"

        if out_lm.exists() and out_lb.exists():
            print("  cached — skipping")
            skip += 1
            continue

        video_path = find_video(name, VIDEO_DIR)
        if video_path is None:
            print(f"  ERROR: video not found")
            fail += 1
            continue

        pitch_start = parse_ms(row["pitch_start_timestamp"])
        pitch_end   = parse_ms(row["pitch_end_timestamp"])
        moments_ms  = [parse_ms(row[col]) for col in MOMENT_COLS]

        if None in (pitch_start, pitch_end):
            print(f"  ERROR: could not parse pitch window timestamps")
            fail += 1
            continue

        # Validate moment ordering; mark bad moments as None.
        # ball_release is allowed up to 200 ms past pitch_end (common rounding error).
        valid_moments = list(moments_ms)
        prev = pitch_start
        for i, ts in enumerate(valid_moments):
            ceiling = pitch_end + (200 if i == 3 else 0)
            if ts is None or ts < prev or ts > ceiling:
                print(f"  WARN: moment {MOMENT_COLS[i]} ({ts} ms) out of order — ignored")
                valid_moments[i] = None
            else:
                prev = ts

        print(f"  Running MediaPipe …")
        try:
            raw_lm, fps = run_mediapipe(video_path)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            fail += 1
            continue

        n_frames = len(raw_lm)
        features = normalize_sequence(raw_lm)                # (n, 198)
        labels   = make_labels(valid_moments, fps, n_frames, pitch_start, pitch_end)

        np.save(out_lm, features)
        np.save(out_lb, labels)
        np.save(out_meta, {
            "video_name":     name,
            "fps":            fps,
            "n_frames":       n_frames,
            "split":          row.get("split (train, val, test)") or row.get("split"),
            "pitcher_id":     row["pitcher_id"],
            "moments_ms":     valid_moments,
            "pitch_start_ms": pitch_start,
            "pitch_end_ms":   pitch_end,
        }, allow_pickle=True)

        counts = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"  {n_frames} frames | labels: {counts}")
        ok += 1

    print(f"\nDone: {ok} extracted, {skip} cached, {fail} failed.")


if __name__ == "__main__":
    main()
