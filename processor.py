import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- Constants ---
MS_TO_MPH = 2.23694
MAX_VELOCITY_HEATMAP = 35
GRAVITY = 9.80665
SMOOTHING_FACTOR = 0.6
FREEZE_DURATION_SEC = 3

# --- YOLOv8-Pose Keypoint Index Map (COCO-17) ---
# 0:Nose  1:L-Eye  2:R-Eye  3:L-Ear  4:R-Ear
# 5:L-Shoulder  6:R-Shoulder  7:L-Elbow  8:R-Elbow
# 9:L-Wrist    10:R-Wrist
# 11:L-Hip     12:R-Hip
# 13:L-Knee    14:R-Knee
# 15:L-Ankle   16:R-Ankle
#
# NOTE: YOLOv8-pose uses the COCO-17 schema which does NOT include foot-tip
# keypoints (MediaPipe indices 31/32).  The ankle is used as the distal
# point for ankle-flex calculations; the metric remains directionally valid.

YOLO_KP = {
    "NOSE": 0,
    "L_SHOULDER": 5,  "R_SHOULDER": 6,
    "L_ELBOW": 7,     "R_ELBOW": 8,
    "L_WRIST": 9,     "R_WRIST": 10,
    "L_HIP": 11,      "R_HIP": 12,
    "L_KNEE": 13,     "R_KNEE": 14,
    "L_ANKLE": 15,    "R_ANKLE": 16,
}

CONF_THRESH = 0.3   # minimum per-keypoint confidence


# ---------------------------------------------------------------------------
# Thin wrapper so downstream code can use  lm[i].x / lm[i].y
# (same API as MediaPipe NormalizedLandmark, z is stubbed to 0)
# ---------------------------------------------------------------------------
class KP:
    __slots__ = ("x", "y", "z", "conf")
    def __init__(self, x: float, y: float, conf: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = 0.0
        self.conf = float(conf)


def _yolo_to_kp(raw_kps, frame_w: int, frame_h: int):
    """Convert a (17, 3) YOLOv8 keypoint array to a list of KP objects."""
    return [KP(x / frame_w, y / frame_h, c) for x, y, c in raw_kps]


def _best_person(results, frame_w: int, frame_h: int):
    """Return KP list for the largest detected person, or None."""
    best_kps, best_area = None, -1.0
    for r in results:
        if r.keypoints is None or len(r.keypoints.data) == 0:
            continue
        boxes   = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        kps_all = r.keypoints.data.cpu().numpy()          # (N, 17, 3)
        for i, kps in enumerate(kps_all):
            area = 0.0
            if len(boxes) > i:
                x1, y1, x2, y2 = boxes[i]
                area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_kps = kps
    if best_kps is None:
        return None
    return _yolo_to_kp(best_kps, frame_w, frame_h)


# ---------------------------------------------------------------------------
# Geometry helpers  (identical to originals)
# ---------------------------------------------------------------------------

def get_heatmap_color(velocity_metric, max_val=MAX_VELOCITY_HEATMAP):
    norm = min(velocity_metric / max_val, 1.0)
    return (int(255*(1-norm)), int(255*(1-abs(norm-0.5)*2)), int(255*norm))


def get_angle_3d(p1, p2, p3):
    v1 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
    v2 = np.array([p3.x-p2.x, p3.y-p2.y, p3.z-p2.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    angle = np.degrees(np.arccos(np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)))
    return 360 - angle if angle > 180 else angle


def get_line_rotation(p1, p2):
    return np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))


def draw_sleek_label(img, text, pos, color=(255,255,255), base_scale=0.8, thickness_mult=1):
    h, w = img.shape[:2]
    ui_scale  = max(0.45, (w/1000)*base_scale)
    thickness = max(1, int(ui_scale*2*thickness_mult))
    padding   = int(25*(w/1000))
    bar_width = max(4, int(8*(w/1000)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (txt_w, txt_h), baseline = cv2.getTextSize(text, font, ui_scale, thickness)
    x, y = int(pos[0]), int(pos[1])
    if x == -1:  x = int((w-txt_w)/2)
    elif x == w: x = int(w-txt_w-padding-10)
    if x+txt_w+padding > w: x = int(w-txt_w-padding-10)
    if x-padding < 0:       x = int(padding+5)
    overlay = img.copy()
    cv2.rectangle(overlay, (x-padding, y-txt_h-padding), (x+txt_w+padding, y+baseline+padding), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x-padding, y-txt_h-padding), (x-padding+bar_width, y+baseline+padding), color, -1)
    cv2.putText(img, text, (x, y), font, ui_scale, (255,255,255), thickness, cv2.LINE_AA)
    return (txt_w, txt_h)


def draw_protractor(img, p_center, p_start, p_end, angle_val, color):
    h, w = img.shape[:2]
    center = (int(p_center.x*w), int(p_center.y*h))
    v1 = np.array([p_start.x-p_center.x, p_start.y-p_center.y])
    v2 = np.array([p_end.x-p_center.x,   p_end.y-p_center.y])
    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
    end_angle   = np.degrees(np.arctan2(v2[1], v2[0]))
    diff = end_angle - start_angle
    if diff > 180:  diff -= 360
    elif diff < -180: diff += 360
    f_start, f_end = start_angle, start_angle+diff
    overlay = img.copy()
    cv2.ellipse(overlay, center, (40,40), 0, f_start, f_end, color, -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.ellipse(img, center, (40,40), 0, f_start, f_end, color, 2, cv2.LINE_AA)
    disp = angle_val if angle_val <= 180 else 360-angle_val
    cv2.putText(img, f"{int(disp)}", (center[0]+15, center[1]-15), 1, 1, (255,255,255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Lateral engine
# ---------------------------------------------------------------------------

def process_lateral(input_path, output_path, p_height_inches, p_side, slow_mo_factor=2):
    p_height_m = p_height_inches * 0.0254
    v_start_thresh, v_stop_thresh = 3.5, 5.0
    stop_buffer = 25

    # Pitching-side keypoint indices
    if p_side.upper() == "RIGHT":
        WRIST    = YOLO_KP["R_WRIST"]
        ELBOW    = YOLO_KP["R_ELBOW"]
        SHOULDER = YOLO_KP["R_SHOULDER"]
        L_HIP    = YOLO_KP["L_HIP"];   L_KNEE = YOLO_KP["L_KNEE"];   L_ANKLE = YOLO_KP["L_ANKLE"]
        D_HIP    = YOLO_KP["R_HIP"];   D_KNEE = YOLO_KP["R_KNEE"];   D_ANKLE = YOLO_KP["R_ANKLE"]
    else:
        WRIST    = YOLO_KP["L_WRIST"]
        ELBOW    = YOLO_KP["L_ELBOW"]
        SHOULDER = YOLO_KP["L_SHOULDER"]
        D_HIP    = YOLO_KP["L_HIP"];   D_KNEE = YOLO_KP["L_KNEE"];   D_ANKLE = YOLO_KP["L_ANKLE"]
        L_HIP    = YOLO_KP["R_HIP"];   L_KNEE = YOLO_KP["R_KNEE"];   L_ANKLE = YOLO_KP["R_ANKLE"]

    # No foot-tip kp in COCO-17 – reuse ankle as distal point
    L_FOOT = L_ANKLE
    D_FOOT = D_ANKLE

    # YOLOv8-pose model  ("n"=nano for speed; swap to "s"/"m" for accuracy)
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    margin_x = int(w * 0.05)
    y_step = h * 0.10

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, int(fps / slow_mo_factor), (w, h))

    trail_history, pitch_summaries, peak_marker = [], [], []
    prev_pos, smoothed_pos, prev_vel = None, None, 0
    low_speed_timer, is_pitching, pitch_count = 0, False, 0
    current_pitch_buffer, current_x_coords, current_y_coords = [], [], []

    # Height calibration: nose→ankle covers ~92 % of standing height
    PPM_RATIO = 0.92

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_count / fps) * 1000)
        results = model(frame, verbose=False)
        lm = _best_person(results, w, h)

        if lm is not None:
            # Pixel-per-metre from nose↔ankle span
            nose_y_px  = lm[YOLO_KP["NOSE"]].y * h
            ankle_y_px = max(lm[YOLO_KP["L_ANKLE"]].y, lm[YOLO_KP["R_ANKLE"]].y) * h
            height_px  = abs(ankle_y_px - nose_y_px)
            ppm = (height_px / (p_height_m * PPM_RATIO)) if height_px > 0 else 1.0

            # 1. Joint angles
            elbow_ang   = get_angle_3d(lm[SHOULDER], lm[ELBOW],  lm[WRIST])
            l_knee_ang  = get_angle_3d(lm[L_HIP],   lm[L_KNEE], lm[L_ANKLE])
            l_ankle_ang = get_angle_3d(lm[L_KNEE],  lm[L_ANKLE], lm[L_FOOT])
            d_knee_ang  = get_angle_3d(lm[D_HIP],   lm[D_KNEE], lm[D_ANKLE])
            d_ankle_ang = get_angle_3d(lm[D_KNEE],  lm[D_ANKLE], lm[D_FOOT])

            if lm[D_KNEE].x > lm[L_KNEE].x:
                target_hip, target_knee = D_HIP, D_KNEE
            else:
                target_hip, target_knee = L_HIP, L_KNEE
            hip_ang = get_angle_3d(lm[SHOULDER], lm[target_hip], lm[target_knee])

            # 2. Wrist velocity
            raw_pos = np.array([lm[WRIST].x * w, lm[WRIST].y * h])
            if smoothed_pos is None:
                smoothed_pos = raw_pos
            else:
                smoothed_pos = (SMOOTHING_FACTOR * raw_pos) + ((1-SMOOTHING_FACTOR) * smoothed_pos)

            if prev_pos is not None:
                dt    = 1.0 / fps
                cur_v = (np.linalg.norm(smoothed_pos - prev_pos) / ppm) / dt

                if cur_v > v_start_thresh and not is_pitching:
                    is_pitching = True
                    current_pitch_buffer, current_x_coords, current_y_coords = [], [], []
            
            
            # 1. Calculate Real-Time Torso Angle & Deviation
            torso_angle = get_line_rotation(lm[target_hip], lm[SHOULDER])
            real_time_sep_deg = abs(torso_angle) - 90

            # 2. Coordinates for Drawing
            s_px = (int(lm[SHOULDER].x * w), int(lm[SHOULDER].y * h))
            h_px_coord = (int(lm[target_hip].x * w), int(lm[target_hip].y * h))

            # --- 3. MAKE LINE MORE VISIBLE & THICKER ---
            # Draw a thick black outline first for high contrast against any background
            cv2.line(frame, s_px, h_px_coord, (0, 0, 0), 6, cv2.LINE_AA)
            # Draw the bright core line on top
            cv2.line(frame, s_px, h_px_coord, (0, 255, 0), 3, cv2.LINE_AA)

            # Draw the reference vertical (thin for less clutter)
            ref_top = (h_px_coord[0], h_px_coord[1] - 100)
            cv2.line(frame, h_px_coord, ref_top, (255, 255, 255), 1, cv2.LINE_4)

            # --- 4. MAKE DISPLAY SMALLER ---
            # Reduced base_scale from 0.8 to 0.5 for a more compact UI
            sep_color = (0, 255, 0) if real_time_sep_deg > 0 else (0, 0, 255)
            draw_sleek_label(
                frame, 
                f"SEP: {real_time_sep_deg:.1f} DEG", 
                (margin_x, int(y_step * 4.5)), 
                sep_color, 
                base_scale=0.5  # This makes the text and box smaller
            )

            # --- Draw protractors ---
            draw_protractor(frame, lm[L_KNEE],     lm[L_HIP],    lm[L_ANKLE],    l_knee_ang,  (0,255,255))
            draw_protractor(frame, lm[L_ANKLE],    lm[L_KNEE],   lm[L_FOOT],     l_ankle_ang, (0,165,255))
            draw_protractor(frame, lm[D_KNEE],     lm[D_HIP],    lm[D_ANKLE],    d_knee_ang,  (0,255,0))
            draw_protractor(frame, lm[D_ANKLE],    lm[D_KNEE],   lm[D_FOOT],     d_ankle_ang, (255,0,255))
            draw_protractor(frame, lm[ELBOW],      lm[SHOULDER], lm[WRIST],      elbow_ang,   (255,255,0))
            draw_protractor(frame, lm[target_hip], lm[SHOULDER], lm[target_knee], hip_ang,    (0,128,255))

            # --- Draw skeleton lines ---
            def l_line(i1, i2, col):
                cv2.line(frame,
                         (int(lm[i1].x*w), int(lm[i1].y*h)),
                         (int(lm[i2].x*w), int(lm[i2].y*h)),
                         col, 2)
            l_line(SHOULDER,   ELBOW,       (255,255,0))
            l_line(ELBOW,      WRIST,       (255,255,0))
            l_line(L_HIP,      L_KNEE,      (0,255,255))
            l_line(L_KNEE,     L_ANKLE,     (0,255,255))
            l_line(D_HIP,      D_KNEE,      (0,255,0))
            l_line(D_KNEE,     D_ANKLE,     (0,255,0))
            l_line(SHOULDER,   target_hip,  (0,128,255))
            l_line(target_hip, target_knee, (0,128,255))

            # Wrist tracking highlight (new visual cue)
            wx, wy = int(lm[WRIST].x*w), int(lm[WRIST].y*h)
            cv2.circle(frame, (wx, wy), 8,  (0,200,255), -1)
            cv2.circle(frame, (wx, wy), 11, (255,255,255), 2)

        # --- HUD: trail, peaks, dashboard ---
        for i in range(1, len(trail_history)):
            cv2.line(frame, trail_history[i-1][:2], trail_history[i][:2],
                     get_heatmap_color(trail_history[i][2]), 10, cv2.LINE_AA)

        for px, py, mph in peak_marker:
            cv2.circle(frame, (px,py), 20, (0,0,255), 2)
            cv2.drawMarker(frame, (px,py), (0,255,255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(frame, f"{mph} mph", (px+22, py+5), 2, 0.7, (0,255,255), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (20,20), (450,140), (0,0,0), -1)
        cv2.putText(frame, f"PITCH: {pitch_count}",               (40,75),  2, 1.5, (0,255,255),   2)
        cv2.putText(frame, f"LIVE: {prev_vel*MS_TO_MPH:.1f} mph", (40,120), 2, 0.7, (255,255,255), 1)
        cv2.putText(frame, "LEAD LEG",  (300,100), 1, 0.8, (0,255,255), 1)
        cv2.putText(frame, "DRIVE LEG", (300,125), 1, 0.8, (0,255,0),   1)

        t_sec = frame_count / fps
        timer_txt = f"{int(t_sec//60):02}:{int(t_sec%60):02}.{int((t_sec%1)*100):02}"
        tw = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0][0]
        cv2.rectangle(frame, (w-tw-50, 20), (w-20, 70), (0,0,0), -1)
        cv2.putText(frame, timer_txt, (w-tw-40, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# Back-view engine
# ---------------------------------------------------------------------------

def process_back(input_path, output_path, slow_mo_factor=2):
    """Back View Engine: Hip-Shoulder Separation (X-Factor) via YOLOv8-pose."""
    L_SH  = YOLO_KP["L_SHOULDER"]; R_SH  = YOLO_KP["R_SHOULDER"]
    L_HIP = YOLO_KP["L_HIP"];      R_HIP = YOLO_KP["R_HIP"]

    model = YOLO("yolov8s-pose.pt")

    cap = cv2.VideoCapture(input_path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    w, h        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, int(fps / slow_mo_factor), (w, h))

    max_separation = 0
    max_x_time     = "00:00.00"
    final_frame    = None
    y_step         = h * 0.10
    margin_x       = int(w * 0.05)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        lm = _best_person(results, w, h)

        total_sec = frame_count / fps
        time_str  = f"{int(total_sec//60):02}:{int(total_sec%60):02}.{int((total_sec%1)*100):02}"

        if lm is not None:
            s_ang      = abs(get_line_rotation(lm[L_SH],  lm[R_SH]))
            h_ang      = abs(get_line_rotation(lm[L_HIP], lm[R_HIP]))
            separation = abs(s_ang - h_ang)

            if separation > max_separation:
                max_separation = separation
                max_x_time     = time_str

            line_thick = max(2, int(w/300))
            cv2.line(frame,
                     (int(lm[L_SH].x*w),  int(lm[L_SH].y*h)),
                     (int(lm[R_SH].x*w),  int(lm[R_SH].y*h)),
                     (255,0,255), line_thick, cv2.LINE_AA)
            cv2.line(frame,
                     (int(lm[L_HIP].x*w), int(lm[L_HIP].y*h)),
                     (int(lm[R_HIP].x*w), int(lm[R_HIP].y*h)),
                     (255,255,0), line_thick, cv2.LINE_AA)

            draw_sleek_label(frame, f"SHOULDER: {s_ang:.1f} DEG",        (margin_x, int(y_step)),       (255,0,255), 0.7)
            draw_sleek_label(frame, f"HIP: {h_ang:.1f} DEG",             (margin_x, int(y_step*2)),     (255,255,0), 0.7)
            draw_sleek_label(frame, f"SEPARATION: {separation:.1f} DEG", (margin_x, int(y_step*3.2)),   (0,255,0),   1.0, 1.5)

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