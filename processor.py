import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Settings ---
MS_TO_MPH = 2.23694
SMOOTHING_FACTOR = 0.2  # Lower = Smoother trace
MAX_VELOCITY_HEATMAP = 35

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
    
    # Landmark Mapping
    if p_side.upper() == 'RIGHT':
        WRIST, ELBOW, SHOULDER = 16, 14, 12
        L_HIP, L_KNEE, L_ANKLE, L_FOOT = 23, 25, 27, 31
        D_HIP, D_KNEE, D_ANKLE, D_FOOT = 24, 26, 28, 32
    else:
        WRIST, ELBOW, SHOULDER = 15, 13, 11
        D_HIP, D_KNEE, D_ANKLE, D_FOOT = 23, 25, 27, 31
        L_HIP, L_KNEE, L_ANKLE, L_FOOT = 24, 26, 28, 32

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_mo_factor, (w, h))

        trail_history, peak_marker = [], []
        prev_pos, smoothed_pos, prev_vel = None, None, 0
        is_pitching, pitch_count, low_speed_timer = False, 0, 0
        current_x_coords, current_y_coords, current_v_list = [], [], []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            timestamp_ms = int((frame_count / fps) * 1000)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                ppm = abs(lm[30].y * h - lm[0].y * h) / p_height_m

                # --- 1. LEG ANGLES (Lead & Drive) ---
                if display_mode in ["All", "Leg Angles Only"]:
                    # Lead Leg
                    l_knee_ang = get_angle_3d(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE])
                    l_ankle_ang = get_angle_3d(lm[L_KNEE], lm[L_ANKLE], lm[L_FOOT])
                    draw_protractor(frame, lm[L_KNEE], lm[L_HIP], lm[L_ANKLE], l_knee_ang, (0, 255, 255))
                    draw_protractor(frame, lm[L_ANKLE], lm[L_KNEE], lm[L_FOOT], l_ankle_ang, (0, 165, 255))
                    
                    # Drive Leg
                    d_knee_ang = get_angle_3d(lm[D_HIP], lm[D_KNEE], lm[D_ANKLE])
                    d_ankle_ang = get_angle_3d(lm[D_KNEE], lm[D_ANKLE], lm[D_FOOT])
                    draw_protractor(frame, lm[D_KNEE], lm[D_HIP], lm[D_ANKLE], d_knee_ang, (0, 255, 0))
                    draw_protractor(frame, lm[D_ANKLE], lm[D_KNEE], lm[D_FOOT], d_ankle_ang, (255, 0, 255))

                    # Skeletal Lines for Legs
                    cv2.line(frame, (int(lm[L_HIP].x*w), int(lm[L_HIP].y*h)), (int(lm[L_KNEE].x*w), int(lm[L_KNEE].y*h)), (0, 255, 255), 2)
                    cv2.line(frame, (int(lm[L_KNEE].x*w), int(lm[L_KNEE].y*h)), (int(lm[L_ANKLE].x*w), int(lm[L_ANKLE].y*h)), (0, 255, 255), 2)
                    cv2.line(frame, (int(lm[D_HIP].x*w), int(lm[D_HIP].y*h)), (int(lm[D_KNEE].x*w), int(lm[D_KNEE].y*h)), (0, 255, 0), 2)
                    cv2.line(frame, (int(lm[D_KNEE].x*w), int(lm[D_KNEE].y*h)), (int(lm[D_ANKLE].x*w), int(lm[D_ANKLE].y*h)), (0, 255, 0), 2)

                # --- 2. ARM ANGLES (Throwing Elbow) ---
                if display_mode in ["All", "Arm Angles Only"]:
                    elbow_ang = get_angle_3d(lm[SHOULDER], lm[ELBOW], lm[WRIST])
                    draw_protractor(frame, lm[ELBOW], lm[SHOULDER], lm[WRIST], elbow_ang, (255, 255, 0))
                    cv2.line(frame, (int(lm[SHOULDER].x*w), int(lm[SHOULDER].y*h)), (int(lm[ELBOW].x*w), int(lm[ELBOW].y*h)), (255, 255, 0), 2)
                    cv2.line(frame, (int(lm[ELBOW].x*w), int(lm[ELBOW].y*h)), (int(lm[WRIST].x*w), int(lm[WRIST].y*h)), (255, 255, 0), 2)

                # --- 3. WRIST TRACE & VELOCITY ---
                if display_mode in ["All", "Wrist Trace & Velocity Only"]:
                    raw_pos = np.array([lm[WRIST].x * w, lm[WRIST].y * h])
                    if smoothed_pos is None: smoothed_pos = raw_pos
                    smoothed_pos = (SMOOTHING_FACTOR * raw_pos) + ((1 - SMOOTHING_FACTOR) * smoothed_pos)

                    if prev_pos is not None:
                        dt = 1 / fps
                        cur_v = (np.linalg.norm(smoothed_pos - prev_pos) / ppm) / dt
                        
                        if cur_v > v_start_thresh:
                            is_pitching = True
                            trail_history.append((int(smoothed_pos[0]), int(smoothed_pos[1]), cur_v))
                            current_v_list.append(cur_v)
                            current_x_coords.append(smoothed_pos[0]); current_y_coords.append(smoothed_pos[1])
                        
                        if is_pitching and cur_v < v_stop_thresh:
                            low_speed_timer += 1
                            if low_speed_timer > stop_buffer:
                                pitch_count += 1
                                p_idx = np.argmax(current_v_list)
                                peak_marker.append((int(current_x_coords[p_idx]), int(current_y_coords[p_idx]), round(current_v_list[p_idx]*MS_TO_MPH, 1)))
                                is_pitching, low_speed_timer, current_v_list, current_x_coords, current_y_coords = False, 0, [], [], []
                        
                        prev_vel = cur_v
                    prev_pos = smoothed_pos.copy()

                    # Draw Trails & Markers
                    for i in range(1, len(trail_history)):
                        cv2.line(frame, trail_history[i-1][:2], trail_history[i][:2], get_heatmap_color(trail_history[i][2]), 10, cv2.LINE_AA)
                    for px, py, mph in peak_marker:
                        cv2.putText(frame, f"{mph} mph", (px + 20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- HUD ---

            # --- ORIGINAL DASHBOARD (Top Left) ---
            # Define colors to match your protractors
            color_lead = (0, 255, 255)  # Yellow
            color_drive = (0, 255, 0)   # Green
            color_text = (255, 255, 255) # White

            # Background Box for the Dashboard
            cv2.rectangle(frame, (10, 20), (350, 280), (0, 0, 0), -1) # Solid black box
            cv2.rectangle(frame, (10, 20), (350, 280), (100, 100, 100), 2) # Grey border

            # Title
            cv2.putText(frame, "MECHANICS HUB", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_text, 2)
            cv2.line(frame, (30, 75), (320, 75), (150, 150, 150), 1)

            # Leg Metrics (Color Matched)
            if 'l_knee_ang' in locals():
                cv2.putText(frame, f"LEAD LEG: {int(l_knee_ang)} deg", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_lead, 2)
            if 'd_knee_ang' in locals():
                cv2.putText(frame, f"DRIVE LEG: {int(d_knee_ang)} deg", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_drive, 2)
            
            # Pitch Data
            cv2.putText(frame, f"PITCH COUNT: {pitch_count}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
            cv2.putText(frame, f"LIVE SPEED: {prev_vel * MS_TO_MPH:.1f} MPH", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

# Note: Keep your existing process_back function here, but update its codec to 'mp4v'ions(model_asset_path='pose_landmarker_heavy.task')
def process_back(input_path, output_path, slow_mo_factor=2):
    """Back View Engine: Hip-Shoulder Separation (X-Factor)."""
    # Define missing landmark indices
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
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