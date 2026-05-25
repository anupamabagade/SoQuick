# SoQuick — Biomechanical Pitching Analysis

A Streamlit web app that analyzes softball/baseball pitching video using MediaPipe pose estimation. Produces annotated slow-motion video with velocity, joint angles, and separation metrics.

## How to Run

```bash
streamlit run app.py
```

Requires ffmpeg installed on the system (used to re-encode output video for browser playback).

## Project Structure

- `app.py` — Streamlit UI. Handles upload, mode selection, calls processor, runs ffmpeg, displays result.
- `processor.py` — Core analysis engine. Contains `process_lateral` and `process_back`.
- `pose_landmarker_heavy.task` — MediaPipe pose model. Must be present in the project root.
- `back_view_dynamic_text.py` — Standalone (legacy) script for back-view analysis, not used by the app.
- `video_visualizer_dual_leg.py` — Standalone (legacy) script for lateral analysis, not used by the app.

## Analysis Modes

### Lateral (Trace)
Inputs: pitcher height (inches), pitching arm (Left/Right), slow-motion factor.

Tracks wrist velocity frame-by-frame and draws a heatmap trail. Detects individual pitches by velocity threshold crossings and marks peak MPH. Draws protractor angles on both legs (lead/drive), elbow, and hip-to-front-knee.

Landmark indices are set based on pitching arm:
- Right-handed: WRIST=16, ELBOW=14, SHOULDER=12, Lead Leg=left side (23,25,27,31), Drive Leg=right side (24,26,28,32)
- Left-handed: mirrored

### Back (Separation)
No extra inputs beyond slow-motion factor.

Measures shoulder-hip X-factor (rotation differential) per frame. Displays live angles for shoulder line and hip line, and their difference. Freezes the final frame for 3 seconds showing max separation and the timestamp it occurred.

## Key Constants in processor.py

| Constant | Default | Purpose |
|---|---|---|
| `SMOOTHING_FACTOR` | 0.35 | EMA weight on current frame for wrist position. Lower = smoother but more lag. |
| `VISIBILITY_THRESHOLD` | 0.5 | MediaPipe confidence below which a wrist frame is skipped entirely. Raise to 0.6 to be stricter. |
| `MAX_PHYSICAL_VELOCITY` | 35.0 m/s | Hard ceiling (~78 mph). Frames above this are treated as bad detections. |
| `MAX_VELOCITY_HEATMAP` | 35 m/s | Upper bound for heatmap color scaling (red = at or above this). |
| `FREEZE_DURATION_SEC` | 3 | Seconds the final summary frame is held at the end of back-view output. |

## Pitch Detection Logic (Lateral Mode)

- Pitch starts when wrist velocity exceeds `v_start_thresh` (3.5 m/s)
- Pitch ends when velocity stays below `v_stop_thresh` (5.0 m/s) for `stop_buffer` (25) consecutive frames
- During frames where the wrist is not visible or velocity is physically impossible, timers still advance using the last known velocity as a proxy

## Video Output

Raw output is written as XVID `.avi`, then re-encoded to H.264 `.mp4` via ffmpeg for browser compatibility. If ffmpeg is missing, the raw file will exist but the Streamlit video player will not display it.

## Dependencies

```
streamlit
opencv-python-headless
mediapipe
numpy
```
