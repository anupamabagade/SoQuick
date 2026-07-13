#!/usr/bin/env python3
"""
annotate_wrist.py — Click-to-label wrist position for TCN training.

Usage:
    python annotate_wrist.py <video_path> --side RIGHT [--output labels/]

Controls:
    Left-click      Set true wrist at clicked pixel
    Y               Accept YOLO prediction as correct
    M               Accept MediaPipe prediction as correct
    S               Mark wrist as occluded (not visible)
    B               Go back one frame (undo)
    Space           Skip frame (exclude from dataset)
    Q               Save and quit
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_MODEL         = os.path.join(os.path.dirname(__file__), 'pose_landmarker_heavy.task')
VISIBILITY_THRESH  = 0.5
YOLO_WRIST_IDX     = {'RIGHT': 10, 'LEFT': 9}   # COCO keypoint indices
MP_WRIST_IDX       = {'RIGHT': 16, 'LEFT': 15}   # MediaPipe landmark indices


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess(video_path, side):
    """
    Single sequential pass: run YOLO + MediaPipe on every frame.
    Returns (frames_jpeg, yolo_preds, mp_preds, mp_landmarks, fps).
    Frames stored as JPEG bytes to keep memory reasonable.
    """
    yolo_wi    = YOLO_WRIST_IDX[side]
    mp_wi      = MP_WRIST_IDX[side]

    from ultralytics import YOLO
    yolo_model = YOLO('yolov8x-pose.pt')

    base_opts  = python.BaseOptions(model_asset_path=POSE_MODEL,
                                    delegate=python.BaseOptions.Delegate.CPU)
    lm_options = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(lm_options)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_jpeg  = []
    yolo_preds   = []   # (x_px, y_px, conf) or None
    mp_preds     = []   # (x_px, y_px, visibility) or None
    mp_landmarks = []   # (33, 3) float32 array or None

    print(f"Pre-processing {total} frames with YOLO + MediaPipe...")
    fidx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        ts_ms = int((fidx / fps) * 1000)

        # --- MediaPipe (must be sequential — VIDEO mode) ---
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), ts_ms
        )
        if result.pose_landmarks:
            lm      = result.pose_landmarks[0]
            lm_arr  = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
            wl      = lm[mp_wi]
            mp_landmarks.append(lm_arr)
            mp_preds.append((wl.x * w, wl.y * h, wl.visibility))
        else:
            mp_landmarks.append(None)
            mp_preds.append(None)

        # --- YOLO ---
        yresults = yolo_model(frame, verbose=False)
        yp = None
        if (yresults and yresults[0].keypoints is not None
                and len(yresults[0].keypoints.xy) > 0):
            kps_xy   = yresults[0].keypoints.xy
            kps_conf = yresults[0].keypoints.conf
            boxes    = yresults[0].boxes.xyxy
            areas    = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            best     = int(np.argmax(areas))
            yp = (
                float(kps_xy[best, yolo_wi, 0]),
                float(kps_xy[best, yolo_wi, 1]),
                float(kps_conf[best, yolo_wi]),
            )
        yolo_preds.append(yp)

        # Store as JPEG to keep RAM manageable
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames_jpeg.append(buf.tobytes() if ok else b'')

        fidx += 1
        if fidx % 10 == 0:
            pct = fidx / total * 100
            bar = '#' * int(pct / 2) + '.' * (50 - int(pct / 2))
            print(f"\r  [{bar}] {fidx}/{total} ({pct:.0f}%)", end='', flush=True)

    cap.release()
    landmarker.close()
    print(f"\n  Done — {fidx} frames processed.")
    return frames_jpeg, yolo_preds, mp_preds, mp_landmarks, fps


# ---------------------------------------------------------------------------
# Annotation UI
# ---------------------------------------------------------------------------

def annotate(video_name, frames_jpeg, yolo_preds, mp_preds, mp_landmarks,
             labels, csv_path):
    """
    OpenCV annotation window. Modifies `labels` dict in-place and auto-saves
    to CSV after every label so progress is never lost.
    """
    click_pos = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    cv2.namedWindow('Annotate Wrist', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Annotate Wrist', on_mouse)

    total = len(frames_jpeg)

    # Start at first unlabeled frame
    idx = 0
    while idx < total and idx in labels:
        idx += 1

    while 0 <= idx < total:
        raw   = np.frombuffer(frames_jpeg[idx], np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w  = frame.shape[:2]
        yp    = yolo_preds[idx]
        mpp   = mp_preds[idx]

        # YOLO — green
        if yp and yp[2] >= VISIBILITY_THRESH:
            cx, cy = int(yp[0]), int(yp[1])
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"YOLO {yp[2]:.2f}", (cx+14, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # MediaPipe — orange
        if mpp and mpp[2] >= VISIBILITY_THRESH:
            mx, my = int(mpp[0]), int(mpp[1])
            cv2.circle(frame, (mx, my), 8, (255, 100, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (mx, my), 12, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"MP {mpp[2]:.2f}", (mx+14, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # Existing label for this frame — cyan
        if idx in labels:
            lbl = labels[idx]
            if str(lbl.get('occluded')) == '1':
                cv2.putText(frame, "OCCLUDED [labeled]", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            elif lbl.get('true_wrist_x') not in ('', None):
                tx = int(float(lbl['true_wrist_x']))
                ty = int(float(lbl['true_wrist_y']))
                cv2.circle(frame, (tx, ty), 12, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 4,  (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(frame, "LABELED", (tx+14, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # HUD strip at the bottom
        hud = [
            f"Frame {idx+1}/{total}   Labeled: {len(labels)}/{total}",
            "Click=label  Y=YOLO  M=MediaPipe  S=occluded",
            "B=back  Space=skip  Q=save+quit",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-95), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, h-75 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Annotate Wrist', frame)
        click_pos[0] = None

        label_row = None
        advance   = True
        waiting   = True

        while waiting:
            key = cv2.waitKey(20) & 0xFF
            cp  = click_pos[0]

            if cp is not None:
                label_row  = _make_row(video_name, idx, cp[0], cp[1], 0,
                                       yp, mpp, mp_landmarks[idx])
                click_pos[0] = None
                waiting    = False

            elif key in (ord('y'), ord('Y')):
                if yp:
                    label_row = _make_row(video_name, idx, yp[0], yp[1], 0,
                                          yp, mpp, mp_landmarks[idx])
                waiting = False

            elif key in (ord('m'), ord('M')):
                if mpp:
                    label_row = _make_row(video_name, idx, mpp[0], mpp[1], 0,
                                          yp, mpp, mp_landmarks[idx])
                waiting = False

            elif key in (ord('s'), ord('S')):
                label_row = _make_row(video_name, idx, None, None, 1,
                                      yp, mpp, mp_landmarks[idx])
                waiting = False

            elif key in (ord('b'), ord('B')):
                idx     = max(0, idx - 1)
                advance = False
                waiting = False

            elif key == ord(' '):
                waiting = False  # advance=True, no label saved

            elif key in (ord('q'), ord('Q')):
                _save_csv(labels, csv_path)
                print(f"\nSaved {len(labels)} labels → {csv_path}")
                cv2.destroyAllWindows()
                sys.exit(0)

        if label_row is not None:
            labels[idx] = label_row
            _save_csv(labels, csv_path)

        if advance:
            idx += 1

    _save_csv(labels, csv_path)
    print(f"\nAll frames done! {len(labels)} labels → {csv_path}")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _make_row(video_name, frame_idx, true_x, true_y, occluded,
              yp, mpp, lm_arr):
    row = {
        'video_id':     video_name,
        'frame_idx':    frame_idx,
        'true_wrist_x': '' if (occluded or true_x is None) else round(float(true_x), 2),
        'true_wrist_y': '' if (occluded or true_y is None) else round(float(true_y), 2),
        'occluded':     int(occluded),
        'yolo_wrist_x': round(yp[0], 2)  if yp  else '',
        'yolo_wrist_y': round(yp[1], 2)  if yp  else '',
        'yolo_conf':    round(yp[2], 4)  if yp  else '',
        'mp_wrist_x':   round(mpp[0], 2) if mpp else '',
        'mp_wrist_y':   round(mpp[1], 2) if mpp else '',
        'mp_wrist_vis': round(mpp[2], 4) if mpp else '',
    }
    for i in range(33):
        if lm_arr is not None:
            row[f'lm{i}_x'] = round(float(lm_arr[i, 0]), 6)
            row[f'lm{i}_y'] = round(float(lm_arr[i, 1]), 6)
            row[f'lm{i}_z'] = round(float(lm_arr[i, 2]), 6)
        else:
            row[f'lm{i}_x'] = row[f'lm{i}_y'] = row[f'lm{i}_z'] = ''
    return row


def _save_csv(labels, path):
    if not labels:
        return
    fieldnames = list(next(iter(labels.values())).keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in sorted(labels.keys()):
            writer.writerow(labels[i])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Wrist annotation tool for TCN training')
    parser.add_argument('video', help='Path to pitching video clip')
    parser.add_argument('--side', required=True, choices=['LEFT', 'RIGHT'],
                        help='Pitching arm (LEFT or RIGHT)')
    parser.add_argument('--output', default='labels',
                        help='Directory for CSV output (default: labels/)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    csv_path   = os.path.join(args.output, f'{video_name}.csv')

    # Resume from prior session if CSV exists
    labels = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                labels[int(row['frame_idx'])] = row
        print(f"Resuming — {len(labels)} frames already labeled.")

    frames_jpeg, yolo_preds, mp_preds, mp_landmarks, fps = preprocess(
        args.video, args.side.upper()
    )
    annotate(video_name, frames_jpeg, yolo_preds, mp_preds, mp_landmarks,
             labels, csv_path)


if __name__ == '__main__':
    main()
