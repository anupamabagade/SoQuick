import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_yolo_model = None   # loaded lazily on first use
_km_model   = None   # key-moment detection model, loaded lazily


def _get_km_model():
    global _km_model
    if _km_model is not False and _km_model is None:
        path = os.path.join(os.path.dirname(__file__), 'ml', 'models', 'key_moments_best.pt')
        if os.path.exists(path):
            try:
                import sys, torch
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))
                from model import load_model
                _km_model = load_model(path)
                _km_model.eval()
            except Exception:
                _km_model = False   # don't retry
        else:
            _km_model = False
    return _km_model if _km_model is not False else None


def _normalize_lm(raw_lm_list):
    """(n,33,3)-or-None list → (n,198) float32 feature array (pos + velocity)."""
    n = len(raw_lm_list)
    pos = np.zeros((n, 33, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, lm in enumerate(raw_lm_list):
        if lm is not None:
            hip = (lm[23] + lm[24]) / 2.0
            scale = np.linalg.norm(lm[11] - lm[12]) + 1e-6
            pos[i] = (lm - hip) / scale
            valid[i] = True
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
    return np.concatenate([pos_flat, vel_flat], axis=1)


def _predict_key_moments(raw_lm_buffer, fps):
    """
    Run the trained model over collected landmarks.
    Returns {key: frame_index} for whichever moments are confidently detected,
    or {} if the model isn't available.
    """
    import torch
    model = _get_km_model()
    if model is None or len(raw_lm_buffer) < model.window:
        return {}

    features = _normalize_lm(raw_lm_buffer)
    n = len(features)
    W = model.window
    half = W // 2
    pad = np.zeros((half, 198), dtype=np.float32)
    padded = np.concatenate([pad, features, pad], axis=0)

    probs = np.zeros((n, 5), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        bs = 64
        for s in range(0, n, bs):
            e = min(s + bs, n)
            batch = torch.from_numpy(
                np.stack([padded[i:i + W] for i in range(s, e)])
            )
            probs[s:e] = torch.softmax(model(batch), dim=-1).numpy()

    class_names = {1: 'foot_lift', 2: 'foot_peak', 3: 'foot_contact', 4: 'ball_release'}
    raw = {}
    for cls, name in class_names.items():
        best = int(np.argmax(probs[:, cls]))
        if probs[best, cls] > 0.3:
            raw[name] = best

    # Enforce monotone ordering
    ordered, prev = {}, 0
    for key in ['foot_lift', 'foot_peak', 'foot_contact', 'ball_release']:
        if key in raw and raw[key] > prev:
            ordered[key] = raw[key]
            prev = raw[key]
    return ordered

# --- Settings ---
MS_TO_MPH = 2.23694
MAX_VELOCITY_HEATMAP = 35
VISIBILITY_THRESHOLD = 0.5    # Skip wrist frames below this YOLO confidence
MAX_PHYSICAL_VELOCITY = 35.0  # m/s (~78 mph) — discard impossible spikes
MAX_JUMP_PX = 250             # reject frame if wrist jumps more than this many pixels (misidentification guard)

def get_heatmap_color(velocity_metric):
    norm = min(velocity_metric / MAX_VELOCITY_HEATMAP, 1.0)
    return (int(255*(1-norm)), int(255*(1-abs(norm-0.5)*2)), int(255*norm))

def get_angle_3d(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))

def get_line_rotation(p1, p2):
    """Calculates the 2D rotation angle (in degrees) of a line between two points."""
    return np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))

def draw_sleek_label(img, text, pos, color=(255, 255, 255), base_scale=0.8, thickness_mult=1):
    """Robust UI label with background box and accent bar."""
    h, w = img.shape[:2]
    ui_scale = max(0.45, (w / 1000) * base_scale)
    thickness = max(1, int(ui_scale * 2 * thickness_mult))
    padding = int(25 * (w / 1000))
    bar_width = max(4, int(8 * (w / 1000)))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (txt_w, txt_h), baseline = cv2.getTextSize(text, font, ui_scale, thickness)
    
    x, y = int(pos[0]), int(pos[1])
    
    # Alignment logic
    if x == -1: x = int((w - txt_w) / 2) # Center
    elif x == w: x = int(w - txt_w - padding - 10) # Right-align
    
    # Boundary checks
    if x + txt_w + padding > w: x = int(w - txt_w - padding - 10)
    if x - padding < 0: x = int(padding + 5)

    overlay = img.copy()
    cv2.rectangle(overlay, (x-padding, y-txt_h-padding), (x+txt_w+padding, y+baseline+padding), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x-padding, y-txt_h-padding), (x-padding+bar_width, y+baseline+padding), color, -1)
    cv2.putText(img, text, (x, y), font, ui_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return (txt_w, txt_h)

def draw_protractor(img, p_center, p_start, p_end, angle_val, color):
    """Draws an semi-transparent arc and angle text between three points."""
    h, w = img.shape[:2]
    center = (int(p_center.x * w), int(p_center.y * h))
    
    # Calculate vectors and angles for the arc
    v1 = np.array([p_start.x - p_center.x, p_start.y - p_center.y])
    v2 = np.array([p_end.x - p_center.x, p_end.y - p_center.y])
    
    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
    end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
    
    diff = end_angle - start_angle
    if diff > 180: diff -= 360
    elif diff < -180: diff += 360
    
    f_start, f_end = start_angle, start_angle + diff
    
    # Draw the transparent overlay
    overlay = img.copy()
    cv2.ellipse(overlay, center, (40, 40), 0, f_start, f_end, color, -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    
    # Draw the outline and text
    cv2.ellipse(img, center, (40, 40), 0, f_start, f_end, color, 2, cv2.LINE_AA)
    disp = angle_val if angle_val <= 180 else 360 - angle_val
    cv2.putText(img, f"{int(disp)}", (center[0]+15, center[1]-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
def process_lateral(input_path, output_path, p_height_inches, p_side, display_mode="All", slow_mo_factor=2):
    p_height_m = p_height_inches * 0.0254
    v_start_thresh, v_stop_thresh = 3.5, 5.0
    stop_buffer = 25
    
    # 1. FIXED LANDMARK MAPPING (Absolute Left vs Right)
    # MediaPipe indices are constant: Left=11,13,15,23,25,27,31 | Right=12,14,16,24,26,28,32
    L_SH, L_HIP, L_KNEE, L_ANKLE, L_FOOT = 11, 23, 25, 27, 31
    R_SH, R_HIP, R_KNEE, R_ANKLE, R_FOOT = 12, 24, 26, 28, 32
    
    # Arm side mapping for velocity trace (MediaPipe indices)
    WRIST = 16 if p_side.upper() == 'RIGHT' else 15
    SHOULDER = 12 if p_side.upper() == 'RIGHT' else 11
    ELBOW = 14 if p_side.upper() == 'RIGHT' else 13

    # COCO keypoint indices used by YOLOv8-pose (person's left/right, not camera's)
    YOLO_WRIST = 10 if p_side.upper() == 'RIGHT' else 9

    # Freeze frame: reach/stride foot = left for RH pitcher, right for LH pitcher
    REACH_FOOT_IDX  = L_FOOT  if p_side.upper() == 'RIGHT' else R_FOOT    # 31 or 32
    REACH_ANKLE_IDX = L_ANKLE if p_side.upper() == 'RIGHT' else R_ANKLE   # 27 or 28
    DRIVE_ANKLE_IDX = R_ANKLE if p_side.upper() == 'RIGHT' else L_ANKLE   # 28 or 27

    # Lazy-load YOLO only when wrist trace is needed (skipped in "Angles Only" mode)
    use_yolo = display_mode not in ["Angles Only"]
    if use_yolo:
        global _yolo_model
        if _yolo_model is None:
            from ultralytics import YOLO
            _yolo_model = YOLO('yolov8x-pose.pt')
        yolo = _yolo_model
    else:
        yolo = None

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_mo_factor, (w, h))

        trail_history, peak_marker = [], []
        prev_pos, prev_vel = None, 0
        is_pitching, pitch_count, low_speed_timer = False, 0, 0
        current_x_coords, current_y_coords, current_v_list = [], [], []

        # --- Freeze frame state ---
        gnd_samples, ground_y = [], None
        foot_phase = 'calibrating'   # → 'on_ground' → 'in_air' → 'landed'
        min_reach_y, min_reach_frame = 1.0, None
        freeze_frames, stride_info = {}, {}
        prev_was_pitching = False
        pitch_peak_v, pitch_peak_frame = 0.0, None
        raw_lm_buffer = []  # accumulated for ML key-moment detection

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            timestamp_ms = int((frame_count / fps) * 1000)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), timestamp_ms)

            # Collect landmarks for ML model (lightweight - just a numpy array)
            if result.pose_landmarks:
                raw_lm_buffer.append(np.array(
                    [[l.x, l.y, l.z] for l in result.pose_landmarks[0]], dtype=np.float32))
            else:
                raw_lm_buffer.append(None)

            # --- YOLO wrist detection (skipped in Angles Only mode) ---
            yolo_wrist_px, yolo_wrist_py, yolo_wrist_conf = None, None, 0.0
            if yolo is None:
                yolo_results = []
            else:
                yolo_results = yolo(frame, verbose=False)
            if (yolo_results and yolo_results[0].keypoints is not None
                    and len(yolo_results[0].keypoints.xy) > 0):
                kps_xy   = yolo_results[0].keypoints.xy    # (N, 17, 2) pixels
                kps_conf = yolo_results[0].keypoints.conf  # (N, 17)
                boxes    = yolo_results[0].boxes.xyxy      # (N, 4)
                # Pick the largest bounding box — the pitcher is the most prominent person
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                best  = int(np.argmax(areas))
                yolo_wrist_px   = float(kps_xy[best, YOLO_WRIST, 0])
                yolo_wrist_py   = float(kps_xy[best, YOLO_WRIST, 1])
                yolo_wrist_conf = float(kps_conf[best, YOLO_WRIST])

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                ppm = abs(lm[30].y * h - lm[0].y * h) / p_height_m

                # --- 1. DUAL LEG CALCULATIONS (No jumping) ---
                if display_mode in ["All", "Leg Angles Only"]:
                    # 1. Determine which hip is closer (Lower Z = Closer)
                    if lm[L_HIP].z < lm[R_HIP].z:
                        # Left Hip is facing camera
                        hip_label = "L-HIP"
                        hip_color = (0, 165, 255) # Orange
                        hip_raw = get_angle_3d(lm[L_SH], lm[L_HIP], lm[L_KNEE])
                        active_hip_ang = hip_raw if hip_raw <= 180 else 360 - hip_raw
                        # Draw Left Hip Protractor
                        draw_protractor(frame, lm[L_HIP], lm[L_SH], lm[L_KNEE], active_hip_ang, hip_color)
                    else:
                        # Right Hip is facing camera
                        hip_label = "R-HIP"
                        hip_color = (255, 0, 255) # Magenta
                        hip_raw = get_angle_3d(lm[R_SH], lm[R_HIP], lm[R_KNEE])
                        active_hip_ang = hip_raw if hip_raw <= 180 else 360 - hip_raw
                        # Draw Right Hip Protractor
                        draw_protractor(frame, lm[R_HIP], lm[R_SH], lm[R_KNEE], active_hip_ang, hip_color)
                    
                    # LEFT LEG Calculations
                    l_knee_r = get_angle_3d(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE])
                    l_ank_r = get_angle_3d(lm[L_KNEE], lm[L_ANKLE], lm[L_FOOT])
                    left_knee = l_knee_r if l_knee_r <= 180 else 360 - l_knee_r
                    left_ankle = l_ank_r if l_ank_r <= 180 else 360 - l_ank_r

                    # RIGHT LEG Calculations
                    r_knee_r = get_angle_3d(lm[R_HIP], lm[R_KNEE], lm[R_ANKLE])
                    r_ank_r = get_angle_3d(lm[R_KNEE], lm[R_ANKLE], lm[R_FOOT])
                    right_knee = r_knee_r if r_knee_r <= 180 else 360 - r_knee_r
                    right_ankle = r_ank_r if r_ank_r <= 180 else 360 - r_ank_r

                    # --- DRAW LEFT LEG (Yellow/Cyan) ---
                    draw_protractor(frame, lm[L_KNEE], lm[L_HIP], lm[L_ANKLE], left_knee, (0, 255, 255))
                    draw_protractor(frame, lm[L_ANKLE], lm[L_KNEE], lm[L_FOOT], left_ankle, (255, 255, 0))
                    cv2.line(frame, (int(lm[L_HIP].x*w), int(lm[L_HIP].y*h)), (int(lm[L_KNEE].x*w), int(lm[L_KNEE].y*h)), (0, 255, 255), 2)
                    cv2.line(frame, (int(lm[L_KNEE].x*w), int(lm[L_KNEE].y*h)), (int(lm[L_ANKLE].x*w), int(lm[L_ANKLE].y*h)), (0, 255, 255), 2)

                    # --- DRAW RIGHT LEG (Green/Magenta) ---
                    draw_protractor(frame, lm[R_KNEE], lm[R_HIP], lm[R_ANKLE], right_knee, (0, 255, 0))
                    draw_protractor(frame, lm[R_ANKLE], lm[R_KNEE], lm[R_FOOT], right_ankle, ((180, 105, 255)))
                    cv2.line(frame, (int(lm[R_HIP].x*w), int(lm[R_HIP].y*h)), (int(lm[R_KNEE].x*w), int(lm[R_KNEE].y*h)), (0, 255, 0), 2)
                    cv2.line(frame, (int(lm[R_KNEE].x*w), int(lm[R_KNEE].y*h)), (int(lm[R_ANKLE].x*w), int(lm[R_ANKLE].y*h)), (0, 255, 0), 2)

                # --- 2. ARM ANGLES ---
                if display_mode in ["All", "Arm Angles Only"]:
                    elbow_raw = get_angle_3d(lm[SHOULDER], lm[ELBOW], lm[WRIST])
                    elbow_ang = elbow_raw if elbow_raw <= 180 else 360 - elbow_raw
                    draw_protractor(frame, lm[ELBOW], lm[SHOULDER], lm[WRIST], elbow_ang, (255, 255, 0))
                    cv2.line(frame, (int(lm[SHOULDER].x*w), int(lm[SHOULDER].y*h)), (int(lm[ELBOW].x*w), int(lm[ELBOW].y*h)), (255, 255, 0), 2)
                    cv2.line(frame, (int(lm[ELBOW].x*w), int(lm[ELBOW].y*h)), (int(lm[WRIST].x*w), int(lm[WRIST].y*h)), (255, 255, 0), 2)

                # --- 3. WRIST TRACE & VELOCITY (position from YOLOv8-pose) ---
                if display_mode in ["All", "Wrist Trace & Velocity Only"]:
                    wrist_visible = (yolo_wrist_conf >= VISIBILITY_THRESHOLD
                                     and yolo_wrist_px is not None
                                     and yolo_wrist_px > 0)

                    if wrist_visible:
                        raw_pos = np.array([yolo_wrist_px, yolo_wrist_py])

                        if prev_pos is not None:
                            jump_px = np.linalg.norm(raw_pos - prev_pos)

                            if jump_px > MAX_JUMP_PX:
                                # Likely misidentification — keep prev_pos anchored to last good position
                                pass
                            else:
                                dt = 1 / fps
                                cur_v = (jump_px / ppm) / dt

                                if cur_v <= MAX_PHYSICAL_VELOCITY:
                                    if cur_v > v_start_thresh:
                                        is_pitching = True
                                        trail_history.append((int(raw_pos[0]), int(raw_pos[1]), cur_v))
                                        current_v_list.append(cur_v)
                                        current_x_coords.append(raw_pos[0])
                                        current_y_coords.append(raw_pos[1])

                                    if is_pitching and cur_v < v_stop_thresh:
                                        low_speed_timer += 1
                                        if low_speed_timer > stop_buffer:
                                            pitch_count += 1
                                            p_idx = np.argmax(current_v_list)
                                            peak_marker.append((int(current_x_coords[p_idx]), int(current_y_coords[p_idx]), round(current_v_list[p_idx] * MS_TO_MPH, 1)))
                                            is_pitching, low_speed_timer = False, 0
                                            current_v_list, current_x_coords, current_y_coords = [], [], []

                                    prev_vel = cur_v

                                prev_pos = raw_pos.copy()  # only update on accepted frames
                        else:
                            prev_pos = raw_pos.copy()

                    # --- DRAW YOLO WRIST DETECTION MARKER ---
                    if wrist_visible:
                        cx, cy = int(yolo_wrist_px), int(yolo_wrist_py)
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.circle(frame, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)

                    # --- DRAWING THE TRACE & PEAK MARKERS ---
                    # 1. Draw the Heatmap Trail
                    for i in range(1, len(trail_history)):
                        cv2.line(frame, trail_history[i-1][:2], trail_history[i][:2], 
                                 get_heatmap_color(trail_history[i][2]), 10, cv2.LINE_AA)
                    
                    # 2. Draw the Peak Markers (Circle + Text)
                    for px, py, mph in peak_marker:
                        # Draw a bright outer ring and a white center dot
                        cv2.circle(frame, (px, py), 12, (0, 255, 255), 3, cv2.LINE_AA) # Yellow ring
                        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA) # White center
                        
                        # Add the MPH text right above the point
                        cv2.putText(frame, f"{mph} MPH", (px - 40, py - 20), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # --- CLEANED DASHBOARD (Top Left) ---
            # Box size adjusted for better spacing
            cv2.rectangle(frame, (10, 20), (380, 450), (0, 0, 0), -1) 
            cv2.rectangle(frame, (10, 20), (380, 450), (100, 100, 100), 2) 

            # Header
            cv2.putText(frame, "DASHBOARD", (30, 65), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.line(frame, (30, 80), (350, 80), (150, 150, 150), 1)

            # --- Row 1: Camera Facing Hip ---
            if 'active_hip_ang' in locals():
                cv2.putText(frame, f"{hip_label}: {int(active_hip_ang)} deg", (30, 125), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) # MAGENTA

            # --- Row 2: Left Leg Data ---
            if 'left_knee' in locals():
                cv2.putText(frame, f"L-KNEE: {int(left_knee)} deg", (30, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # CYAN
                cv2.putText(frame, f"L-ANKLE: {int(left_ankle)} deg", (30, 215), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # YELLOW

            # --- Row 3: Right Leg Data ---
            if 'right_knee' in locals():
                cv2.putText(frame, f"R-KNEE: {int(right_knee)} deg", (30, 260), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)   # GREEN
                cv2.putText(frame, f"R-ANKLE: {int(right_ankle)} deg", (30, 305), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 105, 255), 2) # PURPLE/VIOLET

            # --- Row 4: Pitch Data ---
            if display_mode in ["All", "Wrist Trace & Velocity Only"]:
                cv2.putText(frame, f"SPEED: {prev_vel * 2.23694:.1f} MPH", (30, 370), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"COUNT: {pitch_count}", (30, 415), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- FREEZE FRAME DETECTION (after all drawing) ---
            if result.pose_landmarks:
                lm_f = result.pose_landmarks[0]
                rf_y  = lm_f[REACH_FOOT_IDX].y

                # Phase 1: calibrate ground level from first 20 detected frames
                if foot_phase == 'calibrating':
                    gnd_samples.append(rf_y)
                    if len(gnd_samples) == 20:
                        ground_y = float(np.median(gnd_samples))
                        foot_phase = 'on_ground'

                # Phase 2: wait for foot to lift (threshold: 3% of frame height above ground)
                elif foot_phase == 'on_ground':
                    if 'foot_lift' not in freeze_frames and rf_y < ground_y - 0.03:
                        freeze_frames['foot_lift'] = frame.copy()
                        foot_phase = 'in_air'
                        min_reach_y  = rf_y
                        min_reach_frame = frame.copy()

                # Phase 3: track foot in air — record peak, detect landing
                elif foot_phase == 'in_air':
                    if rf_y < min_reach_y:
                        min_reach_y = rf_y
                        min_reach_frame = frame.copy()
                    elif rf_y >= ground_y - 0.05:   # foot returned to ground level
                        if 'foot_peak' not in freeze_frames and min_reach_frame is not None:
                            freeze_frames['foot_peak'] = min_reach_frame
                        if 'foot_contact' not in freeze_frames:
                            fc = frame.copy()
                            ra = lm_f[REACH_ANKLE_IDX]
                            da = lm_f[DRIVE_ANKLE_IDX]
                            rx, ry = int(ra.x * w), int(ra.y * h)
                            dx, dy = int(da.x * w), int(da.y * h)
                            horiz_px = abs(ra.x * w - da.x * w)
                            full_px  = float(np.linalg.norm([ra.x*w - da.x*w, ra.y*h - da.y*h]))
                            stride_info = {
                                'horiz_ft': round(horiz_px / ppm * 3.28084, 1),
                                'full_ft':  round(full_px  / ppm * 3.28084, 1),
                            }
                            # Annotate stride measurement on the captured frame
                            cv2.line(fc, (rx, ry), (dx, dy), (0, 255, 255), 3, cv2.LINE_AA)
                            cv2.putText(fc, f"FEET APART: {stride_info['full_ft']} ft",
                                        ((rx+dx)//2 - 140, min(ry, dy) - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                            freeze_frames['foot_contact'] = fc
                        foot_phase = 'landed'

                # Ball release: update candidate frame on each new velocity peak
                if is_pitching and current_v_list:
                    if current_v_list[-1] > pitch_peak_v:
                        pitch_peak_v  = current_v_list[-1]
                        pitch_peak_frame = frame.copy()

                # Commit ball release when pitch transitions from active → ended
                if prev_was_pitching and not is_pitching:
                    if pitch_peak_frame is not None and 'ball_release' not in freeze_frames:
                        freeze_frames['ball_release'] = pitch_peak_frame
                    pitch_peak_v, pitch_peak_frame = 0.0, None
                prev_was_pitching = is_pitching

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        # ML key-moment override: if model is available, replace heuristic results
        ml_targets = _predict_key_moments(raw_lm_buffer, fps)
        if ml_targets:
            cap2 = cv2.VideoCapture(input_path)
            for key, fidx in ml_targets.items():
                cap2.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret2, raw_frame = cap2.read()
                if not ret2:
                    continue
                if key == 'foot_contact':
                    lm_arr = raw_lm_buffer[fidx]
                    if lm_arr is not None:
                        ppm_local = abs(lm_arr[30, 1] * h - lm_arr[0, 1] * h) / p_height_m
                        ra, da = lm_arr[REACH_ANKLE_IDX], lm_arr[DRIVE_ANKLE_IDX]
                        rx, ry = int(ra[0] * w), int(ra[1] * h)
                        dx, dy = int(da[0] * w), int(da[1] * h)
                        horiz_px = abs(ra[0] * w - da[0] * w)
                        full_px  = float(np.linalg.norm([(ra[0]-da[0])*w, (ra[1]-da[1])*h]))
                        stride_info = {
                            'horiz_ft': round(horiz_px / ppm_local * 3.28084, 1),
                            'full_ft':  round(full_px  / ppm_local * 3.28084, 1),
                        }
                        cv2.line(raw_frame, (rx, ry), (dx, dy), (0, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(raw_frame, f"FEET APART: {stride_info['full_ft']} ft",
                                    ((rx+dx)//2 - 140, min(ry, dy) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                freeze_frames[key] = raw_frame
            cap2.release()

        # Save freeze frames as JPEG images and return paths
        freeze_image_paths = []
        base = output_path.rsplit('.', 1)[0]
        for key, label in [
            ('foot_lift',    '1 — Back Foot Lift'),
            ('foot_peak',    '2 — Reach Foot Peak Height'),
            ('foot_contact', '3 — Reach Foot Contact'),
            ('ball_release', '4 — Ball Release'),
        ]:
            if key in freeze_frames:
                img_path = f'{base}_freeze_{key}.jpg'
                cv2.imwrite(img_path, freeze_frames[key])
                freeze_image_paths.append({
                    'label': label,
                    'path':  img_path,
                    'stride': stride_info if key == 'foot_contact' else {},
                })
        return freeze_image_paths
    finally:
        landmarker.close()

def process_back(input_path, output_path, slow_mo_factor=2):
    """Back View Engine: Hip-Shoulder Separation (X-Factor)."""
    # Define missing landmark indices
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Using mp4v for compatibility (app.py handles the ffmpeg conversion to fix black screen)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_mo_factor, (w, h))

        max_separation = 0
        max_x_time = "00:00.00"
        final_frame = None
        y_step = h * 0.10 
        margin_x = int(w * 0.05)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            timestamp_ms = int((frame_count / fps) * 1000)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), timestamp_ms)

            total_sec = frame_count / fps
            time_str = f"{int(total_sec//60):02}:{int(total_sec%60):02}.{int((total_sec%1)*100):02}"

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                
                # Calculate rotation of shoulder line and hip line
                s_ang = abs(get_line_rotation(lm[L_SH], lm[R_SH]))
                h_ang = abs(get_line_rotation(lm[L_HIP], lm[R_HIP]))
                separation = abs(s_ang - h_ang)

                if separation > max_separation:
                    max_separation = separation
                    max_x_time = time_str

                # Draw Visual Skeleton Lines
                line_thick = max(2, int(w/300))
                cv2.line(frame, (int(lm[L_SH].x*w), int(lm[L_SH].y*h)), (int(lm[R_SH].x*w), int(lm[R_SH].y*h)), (255, 0, 255), line_thick, cv2.LINE_AA)
                cv2.line(frame, (int(lm[L_HIP].x*w), int(lm[L_HIP].y*h)), (int(lm[R_HIP].x*w), int(lm[R_HIP].y*h)), (255, 255, 0), line_thick, cv2.LINE_AA)

                # Dashboard Labels
                draw_sleek_label(frame, f"SHOULDER: {s_ang:.1f} DEG", (margin_x, int(y_step)), (255, 0, 255), 0.7)
                draw_sleek_label(frame, f"HIP: {h_ang:.1f} DEG", (margin_x, int(y_step * 2)), (255, 255, 0), 0.7)
                draw_sleek_label(frame, f"SEPARATION: {separation:.1f} DEG", (margin_x, int(y_step * 3.2)), (0, 255, 0), 1.0, 1.5)

            # Timer Ticker
            draw_sleek_label(frame, f"TIME: {time_str}", (w, int(y_step)), (180, 180, 180), 0.6)

            # Final Summary Handling
            if frame_count == total_frames - 1:
                summary_txt = f"MAX SEPARATION: {max_separation:.1f} DEG | AT {max_x_time}"
                draw_sleek_label(frame, summary_txt, (-1, int(h - y_step)), (0, 255, 255), 0.8, 1.5)
                final_frame = frame.copy()

            out.write(frame)
            frame_count += 1

        # Freeze Frame Logic
        if final_frame is not None:
            for _ in range(int((fps / slow_mo_factor) * 3)): # Freeze for 3 seconds
                out.write(final_frame)

        cap.release()
        out.release()
    finally:
        landmarker.close()