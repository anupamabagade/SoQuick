import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Constants ---
MS_TO_MPH = 2.23694
MAX_VELOCITY_HEATMAP = 35 
GRAVITY = 9.80665 
SMOOTHING_FACTOR = 0.6 
FREEZE_DURATION_SEC = 3

# --- Helpers ---
def get_heatmap_color(velocity_metric, max_val=MAX_VELOCITY_HEATMAP):
    norm = min(velocity_metric / max_val, 1.0)
    return (int(255*(1-norm)), int(255*(1-abs(norm-0.5)*2)), int(255*norm))

def get_angle_3d(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))

def get_line_rotation(p1, p2):
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
    h, w = img.shape[:2]
    center = (int(p_center.x * w), int(p_center.y * h))
    v1 = np.array([p_start.x - p_center.x, p_start.y - p_center.y])
    v2 = np.array([p_end.x - p_center.x, p_end.y - p_center.y])
    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
    end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
    diff = end_angle - start_angle
    if diff > 180: diff -= 360
    elif diff < -180: diff += 360
    f_start, f_end = start_angle, start_angle + diff
    overlay = img.copy()
    cv2.ellipse(overlay, center, (40, 40), 0, f_start, f_end, color, -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.ellipse(img, center, (40, 40), 0, f_start, f_end, color, 2, cv2.LINE_AA)
    disp = angle_val if angle_val <= 180 else 360 - angle_val
    cv2.putText(img, f"{int(disp)}", (center[0]+15, center[1]-15), 1, 1, (255, 255, 255), 1, cv2.LINE_AA)

# --- Engines ---

def process_lateral(input_path, output_path, p_height_inches, p_side, slow_mo_factor=2):
    p_height_m = p_height_inches * 0.0254
    v_start_thresh, v_stop_thresh = 3.5, 5.0
    stop_buffer = 25
    
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
        # Update 11:36 pm 03.06.26
        # Use avc1 for web compatibility, and adjust output FPS for slow motion
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps / slow_mo_factor, (w, h))

        # 'mp4v' is supposedly the most reliable software-based encoder for Linux servers
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, int(fps / slow_mo_factor), (int(w), int(h)))

        trail_history, pitch_summaries, peak_marker = [], [], []
        prev_pos, smoothed_pos, prev_vel = None, None, 0
        low_speed_timer, is_pitching, pitch_count = 0, False, 0
        current_pitch_buffer, current_x_coords, current_y_coords = [], [], []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            timestamp_ms = int((frame_count / fps) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                ppm = abs(lm[30].y * h - lm[0].y * h) / p_height_m

                # 1. Angles
                elbow_ang = get_angle_3d(lm[SHOULDER], lm[ELBOW], lm[WRIST])
                l_knee_ang = get_angle_3d(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE])
                l_ankle_ang = get_angle_3d(lm[L_KNEE], lm[L_ANKLE], lm[L_FOOT])
                d_knee_ang = get_angle_3d(lm[D_HIP], lm[D_KNEE], lm[D_ANKLE])
                d_ankle_ang = get_angle_3d(lm[D_KNEE], lm[D_ANKLE], lm[D_FOOT])
                hip_ang = get_angle_3d(lm[SHOULDER], lm[D_HIP], lm[D_KNEE])
                # s_ang = abs(get_line_rotation(lm[15], lm[16]))
                # h_ang = abs(get_line_rotation(lm[L_HIP], lm[D_HIP]))
                # separation = abs(s_ang - h_ang)

                # 2. Velocity Tracking
                raw_pos = np.array([lm[WRIST].x * w, lm[WRIST].y * h])
                if smoothed_pos is None: smoothed_pos = raw_pos
                else: smoothed_pos = (SMOOTHING_FACTOR * raw_pos) + ((1 - SMOOTHING_FACTOR) * smoothed_pos)

                if prev_pos is not None:
                    dt = 1 / fps
                    cur_v = (np.linalg.norm(smoothed_pos - prev_pos) / ppm) / dt
                    
                    if cur_v > v_start_thresh and not is_pitching:
                        is_pitching = True
                        current_pitch_buffer, current_x_coords, current_y_coords = [], [], []
                    
                    if is_pitching:
                        accel_m = abs(cur_v - prev_vel) / dt
                        current_pitch_buffer.append([pitch_count+1, timestamp_ms, cur_v, accel_m, accel_m/GRAVITY])
                        current_x_coords.append(smoothed_pos[0]); current_y_coords.append(smoothed_pos[1])
                        trail_history.append((int(smoothed_pos[0]), int(smoothed_pos[1]), cur_v))
                        
                        if cur_v < v_stop_thresh: low_speed_timer += 1
                        else: low_speed_timer = 0
                        
                        if low_speed_timer > stop_buffer:
                            pitch_count += 1
                            v_list = [r[2] for r in current_pitch_buffer]
                            p_idx = np.argmax(v_list)
                            peak_marker.append((int(current_x_coords[p_idx]), int(current_y_coords[p_idx]), round(v_list[p_idx]*MS_TO_MPH, 1)))
                            is_pitching, low_speed_timer = False, 0
                    prev_vel = cur_v
                prev_pos = smoothed_pos.copy()

                # --- Visualizations ---
                draw_protractor(frame, lm[L_KNEE], lm[L_HIP], lm[L_ANKLE], l_knee_ang, (0, 255, 255))
                draw_protractor(frame, lm[L_ANKLE], lm[L_KNEE], lm[L_FOOT], l_ankle_ang, (0, 165, 255))
                draw_protractor(frame, lm[D_KNEE], lm[D_HIP], lm[D_ANKLE], d_knee_ang, (0, 255, 0))
                draw_protractor(frame, lm[D_ANKLE], lm[D_KNEE], lm[D_FOOT], d_ankle_ang, (255, 0, 255))
                draw_protractor(frame, lm[ELBOW], lm[SHOULDER], lm[WRIST], elbow_ang, (255, 255, 0))
                draw_protractor(frame, lm[D_HIP], lm[SHOULDER], lm[D_KNEE], hip_ang, (0, 128, 255))

                # Skeleton Lines
                def l_line(p1, p2, col): cv2.line(frame, (int(lm[p1].x*w), int(lm[p1].y*h)), (int(lm[p2].x*w), int(lm[p2].y*h)), col, 2)
                l_line(SHOULDER, ELBOW, (255, 255, 0)); l_line(ELBOW, WRIST, (255, 255, 0))
                l_line(L_HIP, L_KNEE, (0, 255, 255)); l_line(L_KNEE, L_ANKLE, (0, 255, 255)); l_line(L_ANKLE, L_FOOT, (0, 165, 255))
                l_line(D_HIP, D_KNEE, (0, 255, 0)); l_line(D_KNEE, D_ANKLE, (0, 255, 0)); l_line(D_ANKLE, D_FOOT, (255, 0, 255))
                l_line(SHOULDER, D_HIP, (0, 128, 255)); l_line(D_HIP, D_KNEE, (0, 128, 255))

                
            # --- Overlays (HUD, Ticker, Trails) ---
            for i in range(1, len(trail_history)):
                cv2.line(frame, trail_history[i-1][:2], trail_history[i][:2], get_heatmap_color(trail_history[i][2]), 10, cv2.LINE_AA)
            
            for px, py, mph in peak_marker:
                cv2.circle(frame, (px, py), 20, (0, 0, 255), 2)
                cv2.drawMarker(frame, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(frame, f"{mph} mph", (px + 22, py + 5), 2, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            # Dashboard & Legend
            cv2.rectangle(frame, (20, 20), (450, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"PITCH: {pitch_count}", (40, 75), 2, 1.5, (0, 255, 255), 2)
            cv2.putText(frame, f"LIVE: {prev_vel * MS_TO_MPH:.1f} mph", (40, 120), 2, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, "LEAD LEG", (300, 100), 1, 0.8, (0, 255, 255), 1)
            cv2.putText(frame, "DRIVE LEG", (300, 125), 1, 0.8, (0, 255, 0), 1)

            #draw_sleek_label(frame, f"SEPARATION: {separation:.1f} DEG", (50, h - 50), (0, 255, 0), 0.8)    
            
            # Timer Ticker
            t_sec = frame_count / fps
            timer_txt = f"{int(t_sec//60):02}:{int(t_sec%60):02}.{int((t_sec%1)*100):02}"
            tw = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0][0]
            cv2.rectangle(frame, (w - tw - 50, 20), (w - 20, 70), (0, 0, 0), -1)
            cv2.putText(frame, timer_txt, (w - tw - 40, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()

def process_back(input_path, output_path, slow_mo_factor=2):

    """Back View Engine: Hip-Shoulder Separation (X-Factor)."""
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # slow_mo_factor adjusts output FPS for browser playback
        #out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps / slow_mo_factor, (w, h))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, int(fps / slow_mo_factor), (int(w), int(h)))

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
                
                # Calculate angles
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
            for _ in range(int((fps / slow_mo_factor) * FREEZE_DURATION_SEC)):
                out.write(final_frame)

        cap.release()
        out.release()

