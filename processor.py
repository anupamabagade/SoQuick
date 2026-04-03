import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# =============================================================================
# SoQuick processor.py  —  Hybrid YOLO + MediaPipe edition
#
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
# Problem with pure YOLO (COCO-17):
#   No foot-tip keypoints → ankle-flex angle is always a degenerate 180°.
#
# Problem with pure MediaPipe:
#   Wrist localisation loses precision during the fast windmill release,
#   producing jitter in the velocity trace.
#
# Solution — each model does what it is best at:
#
#   ┌──────────────────────────────────────────────────────┐
#   │  YOLOv8x-pose-p6  (1280 px, COCO-17)                │
#   │  → Wrist pixel position → velocity & trace           │
#   └──────────────────────────────────────────────────────┘
#   ┌──────────────────────────────────────────────────────┐
#   │  MediaPipe PoseLandmarker Heavy  (33-point schema)   │
#   │  → All joint angles (elbow, knee, ankle, hip)        │
#   │  → Foot-tip keypoints (indices 31/32) for true       │
#   │    ankle-flex angle                                   │
#   └──────────────────────────────────────────────────────┘
#
# Both models run on every frame.  The wrist EMA smoother sits on top of
# the YOLO output only; MediaPipe landmarks are used raw (MediaPipe's own
# internal temporal filtering already stabilises them).
# =============================================================================

# --- Constants ---------------------------------------------------------------
MS_TO_MPH            = 2.23694
MAX_VELOCITY_HEATMAP = 35
GRAVITY              = 9.80665
FREEZE_DURATION_SEC  = 3

# Wrist position EMA alpha (higher = more responsive, more noisy)
WRIST_SMOOTH = 0.55

# Pixel-per-metre calibration: MediaPipe nose (0) → foot tip (30)
# covers essentially the full standing height, so PPM_RATIO = 1.0
PPM_RATIO = 1.0

# MediaPipe landmark indices (33-point schema)
MP_IDX = {
    # Arms
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW":    13, "R_ELBOW":    14,
    "L_WRIST":    15, "R_WRIST":    16,
    # Hips / legs
    "L_HIP":   23, "R_HIP":   24,
    "L_KNEE":  25, "R_KNEE":  26,
    "L_ANKLE": 27, "R_ANKLE": 28,
    # Foot tips (only available in MediaPipe — this is the whole point)
    "L_FOOT":  31, "R_FOOT":  32,
    # Head
    "NOSE":     0,
    "L_HEEL":  29, "R_HEEL":  30,
}

# YOLOv8-pose COCO-17 indices (used ONLY for wrist tracking)
YOLO_KP = {
    "NOSE":        0,
    "L_SHOULDER":  5, "R_SHOULDER":  6,
    "L_ELBOW":     7, "R_ELBOW":     8,
    "L_WRIST":     9, "R_WRIST":    10,
    "L_HIP":      11, "R_HIP":      12,
    "L_KNEE":     13, "R_KNEE":     14,
    "L_ANKLE":    15, "R_ANKLE":    16,
}


# --- Thin KP wrapper ---------------------------------------------------------
# Gives YOLO keypoints the same .x/.y API as MediaPipe NormalizedLandmark.

class KP:
    __slots__ = ("x", "y", "z", "conf")
    def __init__(self, x: float, y: float, conf: float = 1.0):
        self.x    = float(x)
        self.y    = float(y)
        self.z    = 0.0
        self.conf = float(conf)


# --- YOLO helpers ------------------------------------------------------------

def _yolo_to_kp(raw_kps, frame_w: int, frame_h: int) -> list:
    return [KP(x / frame_w, y / frame_h, c) for x, y, c in raw_kps]


def _best_yolo_person(results, frame_w: int, frame_h: int):
    """Return COCO-17 KP list for the largest detected person, or None."""
    best_kps, best_area = None, -1.0
    for r in results:
        if r.keypoints is None or len(r.keypoints.data) == 0:
            continue
        boxes   = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        kps_all = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
        for i, kps in enumerate(kps_all):
            area = 0.0
            if len(boxes) > i:
                x1, y1, x2, y2 = boxes[i]
                area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_kps  = kps
    return None if best_kps is None else _yolo_to_kp(best_kps, frame_w, frame_h)


# --- Geometry helpers --------------------------------------------------------

def get_heatmap_color(velocity_metric, max_val=MAX_VELOCITY_HEATMAP):
    norm = min(velocity_metric / max_val, 1.0)
    return (int(255*(1-norm)), int(255*(1-abs(norm-0.5)*2)), int(255*norm))


def get_angle_3d(p1, p2, p3) -> float:
    """Interior angle (degrees) at vertex p2."""
    v1 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
    v2 = np.array([p3.x-p2.x, p3.y-p2.y, p3.z-p2.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    angle = np.degrees(np.arccos(np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)))
    return 360 - angle if angle > 180 else angle


def get_line_rotation(p1, p2) -> float:
    return np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))


def draw_sleek_label(img, text, pos, color=(255,255,255), base_scale=0.8, thickness_mult=1):
    h, w  = img.shape[:2]
    scale = max(0.45, (w/1000)*base_scale)
    thick = max(1, int(scale*2*thickness_mult))
    pad   = int(25*(w/1000))
    bw    = max(4, int(8*(w/1000)))
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = int(pos[0]), int(pos[1])
    if x == -1:  x = int((w-tw)/2)
    elif x == w: x = int(w-tw-pad-10)
    x = max(pad+5, min(x, w-tw-pad-10))
    ov = img.copy()
    cv2.rectangle(ov,  (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0,0,0), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x-pad, y-th-pad), (x-pad+bw, y+bl+pad), color, -1)
    cv2.putText(img, text, (x, y), font, scale, (255,255,255), thick, cv2.LINE_AA)
    return (tw, th)


def draw_protractor(img, p_center, p_start, p_end, angle_val, color):
    h, w   = img.shape[:2]
    center = (int(p_center.x*w), int(p_center.y*h))
    v1     = np.array([p_start.x-p_center.x, p_start.y-p_center.y])
    v2     = np.array([p_end.x-p_center.x,   p_end.y-p_center.y])
    sa     = np.degrees(np.arctan2(v1[1], v1[0]))
    ea     = np.degrees(np.arctan2(v2[1], v2[0]))
    diff   = ea - sa
    diff   = diff-360 if diff>180 else (diff+360 if diff<-180 else diff)
    ov = img.copy()
    cv2.ellipse(ov,  center, (40,40), 0, sa, sa+diff, color, -1)
    cv2.addWeighted(ov, 0.4, img, 0.6, 0, img)
    cv2.ellipse(img, center, (40,40), 0, sa, sa+diff, color, 2, cv2.LINE_AA)
    disp = angle_val if angle_val <= 180 else 360-angle_val
    cv2.putText(img, f"{int(disp)}", (center[0]+15, center[1]-15),
                1, 1, (255,255,255), 1, cv2.LINE_AA)


# =============================================================================
# Lateral engine
# =============================================================================

def process_lateral(input_path, output_path, p_height_inches, p_side, slow_mo_factor=2):
    p_height_m     = p_height_inches * 0.0254
    v_start_thresh = 3.5
    v_stop_thresh  = 5.0
    stop_buffer    = 25

    # --- MediaPipe landmark indices for this pitching side ---
    if p_side.upper() == "RIGHT":
        MP_WRIST    = MP_IDX["R_WRIST"];   MP_ELBOW    = MP_IDX["R_ELBOW"]
        MP_SHOULDER = MP_IDX["R_SHOULDER"]
        MP_L_HIP    = MP_IDX["L_HIP"];  MP_L_KNEE = MP_IDX["L_KNEE"]
        MP_L_ANKLE  = MP_IDX["L_ANKLE"]; MP_L_FOOT  = MP_IDX["L_FOOT"]
        MP_D_HIP    = MP_IDX["R_HIP"];  MP_D_KNEE = MP_IDX["R_KNEE"]
        MP_D_ANKLE  = MP_IDX["R_ANKLE"]; MP_D_FOOT  = MP_IDX["R_FOOT"]
    else:
        MP_WRIST    = MP_IDX["L_WRIST"];   MP_ELBOW    = MP_IDX["L_ELBOW"]
        MP_SHOULDER = MP_IDX["L_SHOULDER"]
        MP_D_HIP    = MP_IDX["L_HIP"];  MP_D_KNEE = MP_IDX["L_KNEE"]
        MP_D_ANKLE  = MP_IDX["L_ANKLE"]; MP_D_FOOT  = MP_IDX["L_FOOT"]
        MP_L_HIP    = MP_IDX["R_HIP"];  MP_L_KNEE = MP_IDX["R_KNEE"]
        MP_L_ANKLE  = MP_IDX["R_ANKLE"]; MP_L_FOOT  = MP_IDX["R_FOOT"]

    # --- YOLO wrist index for this pitching side ---
    YOLO_WRIST = YOLO_KP["R_WRIST"] if p_side.upper() == "RIGHT" else YOLO_KP["L_WRIST"]

    # ── Model initialisation ─────────────────────────────────────────────────
    # YOLO: wrist tracking only
    yolo_model = YOLO("yolov8x-pose-p6.pt")

    # MediaPipe: all joint angles + foot-tip keypoints
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    mp_options   = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(output_path, fourcc, int(fps/slow_mo_factor), (w, h))

    trail_history         = []
    peak_marker           = []
    prev_pos              = None
    smoothed_pos          = None   # YOLO wrist pixel-position EMA
    prev_vel              = 0.0
    low_speed_timer       = 0
    is_pitching           = False
    pitch_count           = 0
    current_pitch_buffer  = []
    current_x_coords      = []
    current_y_coords      = []

    with vision.PoseLandmarker.create_from_options(mp_options) as mp_landmarker:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_count / fps) * 1000)

            # ── 1. YOLO inference → wrist position ───────────────────────────
            yolo_results = yolo_model(frame, verbose=False, imgsz=1280)
            yolo_lm      = _best_yolo_person(yolo_results, w, h)

            # ── 2. MediaPipe inference → all joint angles ─────────────────────
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            mp_result = mp_landmarker.detect_for_video(mp_image, timestamp_ms)
            mp_lm = mp_result.pose_landmarks[0] if mp_result.pose_landmarks else None

            # ── 3. Pixel-per-metre from MediaPipe (nose → foot tip) ───────────
            ppm = 1.0
            if mp_lm is not None:
                nose_y   = mp_lm[0].y * h          # landmark 0 = nose
                foot_y   = mp_lm[30].y * h          # landmark 30 = foot tip
                height_px = abs(foot_y - nose_y)
                ppm = (height_px / (p_height_m * PPM_RATIO)) if height_px > 0 else 1.0

            # ── 4. Joint angles from MediaPipe ────────────────────────────────
            if mp_lm is not None:
                elbow_ang   = get_angle_3d(mp_lm[MP_SHOULDER], mp_lm[MP_ELBOW],   mp_lm[MP_WRIST])
                l_knee_ang  = get_angle_3d(mp_lm[MP_L_HIP],   mp_lm[MP_L_KNEE],  mp_lm[MP_L_ANKLE])
                d_knee_ang  = get_angle_3d(mp_lm[MP_D_HIP],   mp_lm[MP_D_KNEE],  mp_lm[MP_D_ANKLE])
                # True ankle-flex angle using MediaPipe foot-tip keypoints
                l_ankle_ang = get_angle_3d(mp_lm[MP_L_KNEE],  mp_lm[MP_L_ANKLE], mp_lm[MP_L_FOOT])
                d_ankle_ang = get_angle_3d(mp_lm[MP_D_KNEE],  mp_lm[MP_D_ANKLE], mp_lm[MP_D_FOOT])

                # Forward hip: whichever knee has greater x (closer to catcher)
                if mp_lm[MP_D_KNEE].x > mp_lm[MP_L_KNEE].x:
                    target_hip, target_knee = MP_D_HIP, MP_D_KNEE
                else:
                    target_hip, target_knee = MP_L_HIP, MP_L_KNEE
                hip_ang = get_angle_3d(mp_lm[MP_SHOULDER], mp_lm[target_hip], mp_lm[target_knee])

            # ── 5. Wrist velocity from YOLO ───────────────────────────────────
            if yolo_lm is not None:
                raw_wrist = np.array([yolo_lm[YOLO_WRIST].x * w,
                                      yolo_lm[YOLO_WRIST].y * h])
                if smoothed_pos is None:
                    smoothed_pos = raw_wrist
                else:
                    smoothed_pos = WRIST_SMOOTH * raw_wrist + (1-WRIST_SMOOTH) * smoothed_pos

                if prev_pos is not None:
                    dt    = 1.0 / fps
                    cur_v = (np.linalg.norm(smoothed_pos - prev_pos) / ppm) / dt

                    if cur_v > v_start_thresh and not is_pitching:
                        is_pitching = True
                        current_pitch_buffer, current_x_coords, current_y_coords = [], [], []

                    if is_pitching:
                        accel_m = abs(cur_v - prev_vel) / dt
                        current_pitch_buffer.append(
                            [pitch_count+1, timestamp_ms, cur_v, accel_m, accel_m/GRAVITY])
                        current_x_coords.append(smoothed_pos[0])
                        current_y_coords.append(smoothed_pos[1])
                        trail_history.append((int(smoothed_pos[0]), int(smoothed_pos[1]), cur_v))
                        low_speed_timer = low_speed_timer+1 if cur_v < v_stop_thresh else 0

                        if low_speed_timer > stop_buffer:
                            pitch_count += 1
                            v_list = [r[2] for r in current_pitch_buffer]
                            p_idx  = int(np.argmax(v_list))
                            peak_marker.append((
                                int(current_x_coords[p_idx]),
                                int(current_y_coords[p_idx]),
                                round(v_list[p_idx] * MS_TO_MPH, 1)
                            ))
                            is_pitching, low_speed_timer = False, 0

                    prev_vel = cur_v
                prev_pos = smoothed_pos.copy()

            # ── 6. Draw: angles & skeleton from MediaPipe ─────────────────────
            if mp_lm is not None:
                draw_protractor(frame, mp_lm[MP_L_KNEE],   mp_lm[MP_L_HIP],    mp_lm[MP_L_ANKLE], l_knee_ang,  (0,255,255))
                draw_protractor(frame, mp_lm[MP_L_ANKLE],  mp_lm[MP_L_KNEE],   mp_lm[MP_L_FOOT],  l_ankle_ang, (0,165,255))
                draw_protractor(frame, mp_lm[MP_D_KNEE],   mp_lm[MP_D_HIP],    mp_lm[MP_D_ANKLE], d_knee_ang,  (0,255,0))
                draw_protractor(frame, mp_lm[MP_D_ANKLE],  mp_lm[MP_D_KNEE],   mp_lm[MP_D_FOOT],  d_ankle_ang, (255,0,255))
                draw_protractor(frame, mp_lm[MP_ELBOW],    mp_lm[MP_SHOULDER], mp_lm[MP_WRIST],   elbow_ang,   (255,255,0))
                draw_protractor(frame, mp_lm[target_hip],  mp_lm[MP_SHOULDER], mp_lm[target_knee], hip_ang,    (0,128,255))

                def mp_line(i1, i2, col):
                    cv2.line(frame,
                             (int(mp_lm[i1].x*w), int(mp_lm[i1].y*h)),
                             (int(mp_lm[i2].x*w), int(mp_lm[i2].y*h)),
                             col, 2)
                mp_line(MP_SHOULDER, MP_ELBOW,      (255,255,0))
                mp_line(MP_ELBOW,    MP_WRIST,      (255,255,0))
                mp_line(MP_L_HIP,    MP_L_KNEE,     (0,255,255))
                mp_line(MP_L_KNEE,   MP_L_ANKLE,    (0,255,255))
                mp_line(MP_L_ANKLE,  MP_L_FOOT,     (0,165,255))
                mp_line(MP_D_HIP,    MP_D_KNEE,     (0,255,0))
                mp_line(MP_D_KNEE,   MP_D_ANKLE,    (0,255,0))
                mp_line(MP_D_ANKLE,  MP_D_FOOT,     (255,0,255))
                mp_line(MP_SHOULDER, target_hip,    (0,128,255))
                mp_line(target_hip,  target_knee,   (0,128,255))

            # ── 7. Draw: wrist trace from YOLO ────────────────────────────────
            if yolo_lm is not None and smoothed_pos is not None:
                wx, wy = int(smoothed_pos[0]), int(smoothed_pos[1])
                cv2.circle(frame, (wx, wy), 8,  (0,200,255), -1)
                cv2.circle(frame, (wx, wy), 11, (255,255,255), 2)

            # ── 8. HUD overlays ───────────────────────────────────────────────
            for i in range(1, len(trail_history)):
                cv2.line(frame, trail_history[i-1][:2], trail_history[i][:2],
                         get_heatmap_color(trail_history[i][2]), 10, cv2.LINE_AA)

            for px, py, mph in peak_marker:
                cv2.circle(frame, (px,py), 20, (0,0,255), 2)
                cv2.drawMarker(frame, (px,py), (0,255,255), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(frame, f"{mph} mph", (px+22,py+5), 2, 0.7, (0,255,255), 1, cv2.LINE_AA)

            cv2.rectangle(frame, (20,20), (450,140), (0,0,0), -1)
            cv2.putText(frame, f"PITCH: {pitch_count}",               (40,75),  2, 1.5, (0,255,255),   2)
            cv2.putText(frame, f"LIVE: {prev_vel*MS_TO_MPH:.1f} mph", (40,120), 2, 0.7, (255,255,255), 1)
            cv2.putText(frame, "LEAD LEG",  (300,100), 1, 0.8, (0,255,255), 1)
            cv2.putText(frame, "DRIVE LEG", (300,125), 1, 0.8, (0,255,0),   1)

            t_sec = frame_count / fps
            timer_txt = f"{int(t_sec//60):02}:{int(t_sec%60):02}.{int((t_sec%1)*100):02}"
            tw = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0][0]
            cv2.rectangle(frame, (w-tw-50,20), (w-20,70), (0,0,0), -1)
            cv2.putText(frame, timer_txt, (w-tw-40,55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()


# =============================================================================
# Back-view engine  (angles only → MediaPipe is sufficient, no YOLO needed)
# =============================================================================

def process_back(input_path, output_path, slow_mo_factor=2):
    """Back View Engine: Hip-Shoulder Separation (X-Factor).
    Uses MediaPipe only — YOLO wrist tracking is not relevant for this view.
    """
    L_SH  = MP_IDX["L_SHOULDER"]; R_SH  = MP_IDX["R_SHOULDER"]
    L_HIP = MP_IDX["L_HIP"];      R_HIP = MP_IDX["R_HIP"]

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    mp_options   = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(input_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(output_path, fourcc, int(fps/slow_mo_factor), (w, h))

    max_separation = 0.0
    max_x_time     = "00:00.00"
    final_frame    = None
    y_step         = h * 0.10
    margin_x       = int(w * 0.05)

    with vision.PoseLandmarker.create_from_options(mp_options) as mp_landmarker:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_count / fps) * 1000)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            mp_result = mp_landmarker.detect_for_video(mp_image, timestamp_ms)

            total_sec = frame_count / fps
            time_str  = f"{int(total_sec//60):02}:{int(total_sec%60):02}.{int((total_sec%1)*100):02}"

            if mp_result.pose_landmarks:
                lm         = mp_result.pose_landmarks[0]
                s_ang      = abs(get_line_rotation(lm[L_SH],  lm[R_SH]))
                h_ang      = abs(get_line_rotation(lm[L_HIP], lm[R_HIP]))
                separation = abs(s_ang - h_ang)

                if separation > max_separation:
                    max_separation = separation
                    max_x_time     = time_str

                lt = max(2, int(w/300))
                cv2.line(frame,
                         (int(lm[L_SH].x*w),  int(lm[L_SH].y*h)),
                         (int(lm[R_SH].x*w),  int(lm[R_SH].y*h)),
                         (255,0,255), lt, cv2.LINE_AA)
                cv2.line(frame,
                         (int(lm[L_HIP].x*w), int(lm[L_HIP].y*h)),
                         (int(lm[R_HIP].x*w), int(lm[R_HIP].y*h)),
                         (255,255,0), lt, cv2.LINE_AA)

                draw_sleek_label(frame, f"SHOULDER: {s_ang:.1f} DEG",        (margin_x, int(y_step)),     (255,0,255), 0.7)
                draw_sleek_label(frame, f"HIP: {h_ang:.1f} DEG",             (margin_x, int(y_step*2)),   (255,255,0), 0.7)
                draw_sleek_label(frame, f"SEPARATION: {separation:.1f} DEG", (margin_x, int(y_step*3.2)), (0,255,0),   1.0, 1.5)

            draw_sleek_label(frame, f"TIME: {time_str}", (w, int(y_step)), (180,180,180), 0.6)

            if frame_count == total_frames - 1:
                summary_txt = f"MAX SEPARATION: {max_separation:.1f} DEG | AT {max_x_time}"
                draw_sleek_label(frame, summary_txt, (-1, int(h-y_step)), (0,255,255), 0.8, 1.5)
                final_frame = frame.copy()

            out.write(frame)
            frame_count += 1

        if final_frame is not None:
            for _ in range(int((fps/slow_mo_factor) * FREEZE_DURATION_SEC)):
                out.write(final_frame)

    cap.release()
    out.release()