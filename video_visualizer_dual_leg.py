import cv2
import mediapipe as mp
import numpy as np
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
model_path = 'pose_landmarker_heavy.task' 
input_video = '01.04.26_KG2.mp4'
output_video = '01.04.26_KG2_dual_leg_analysis.mp4'
output_summary_csv = '01.04.26_KG2_peak_vel.csv'
slow_motion_factor = 2

MS_TO_MPH = 2.23694
MAX_VELOCITY_HEATMAP = 35 
PITCHER_WRIST_SIDE = 'RIGHT' 
PITCHER_HEIGHT_INCHES = 72
PITCHER_HEIGHT_METERS = PITCHER_HEIGHT_INCHES * 0.0254 
GRAVITY = 9.80665 
VELOCITY_START_THRESHOLD = 3.5  
VELOCITY_STOP_THRESHOLD = 5   
STOP_FRAME_BUFFER = 25
SMOOTHING_FACTOR = 0.6 

# --- Leg Mapping Logic ---
# MediaPipe Indices: Left(23,25,27,31), Right(24,26,28,32)
if PITCHER_WRIST_SIDE.upper() == 'RIGHT':
    WRIST, ELBOW, SHOULDER = 16, 14, 12 
    # Lead Leg (Left)
    LEAD_HIP, LEAD_KNEE, LEAD_ANKLE, LEAD_FOOT = 23, 25, 27, 31
    # Drive Leg (Right)
    DRIVE_HIP, DRIVE_KNEE, DRIVE_ANKLE, DRIVE_FOOT = 24, 26, 28, 32
else:
    WRIST, ELBOW, SHOULDER = 15, 13, 11  
    # Lead Leg (Right)
    LEAD_HIP, LEAD_KNEE, LEAD_ANKLE, LEAD_FOOT = 24, 26, 28, 32
    # Drive Leg (Left)
    DRIVE_HIP, DRIVE_KNEE, DRIVE_ANKLE, DRIVE_FOOT = 23, 25, 27, 31

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

def draw_protractor(img, p_center, p_start, p_end, angle_val, color):
    center = (int(p_center.x * img.shape[1]), int(p_center.y * img.shape[0]))
    v1 = np.array([p_start.x - p_center.x, p_start.y - p_center.y])
    v2 = np.array([p_end.x - p_center.x, p_end.y - p_center.y])
    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
    end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
    diff = end_angle - start_angle
    if diff > 180: diff -= 360
    elif diff < -180: diff += 360
    f_start, f_end = start_angle, start_angle + diff
    if f_start > f_end: f_start, f_end = f_end, f_start
    overlay = img.copy()
    cv2.ellipse(overlay, center, (40, 40), 0, f_start, f_end, color, -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.ellipse(img, center, (40, 40), 0, f_start, f_end, color, 2, cv2.LINE_AA)
    disp = angle_val if angle_val <= 180 else 360 - angle_val
    cv2.putText(img, f"{int(disp)}", (center[0]+15, center[1]-15), 1, 1, (255, 255, 255), 1, cv2.LINE_AA)

# --- Tracking State ---
trail_history, pitch_summaries, peak_marker = [], [], []
prev_pos, smoothed_pos, prev_vel = None, None, 0
low_speed_timer, is_pitching, pitch_count = 0, False, 0
current_x_coords, current_y_coords, current_pitch_buffer = [], [], []

# --- Main Engine ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_motion_factor, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        timestamp_ms = int((frame_count / fps) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            ppm = abs(lm[30].y * height - lm[0].y * height) / PITCHER_HEIGHT_METERS

            # 1. Calculate Angles for Both Legs
            elbow_ang = get_angle_3d(lm[SHOULDER], lm[ELBOW], lm[WRIST])
            # Lead Leg Angles
            l_knee_ang = get_angle_3d(lm[LEAD_HIP], lm[LEAD_KNEE], lm[LEAD_ANKLE])
            l_ankle_ang = get_angle_3d(lm[LEAD_KNEE], lm[LEAD_ANKLE], lm[LEAD_FOOT])
            # Drive Leg Angles
            d_knee_ang = get_angle_3d(lm[DRIVE_HIP], lm[DRIVE_KNEE], lm[DRIVE_ANKLE])
            d_ankle_ang = get_angle_3d(lm[DRIVE_KNEE], lm[DRIVE_ANKLE], lm[DRIVE_FOOT])

            # 2. Wrist Velocity Tracking
            raw_pos = np.array([lm[WRIST].x * width, lm[WRIST].y * height])
            if smoothed_pos is None: smoothed_pos = raw_pos
            else: smoothed_pos = (SMOOTHING_FACTOR * raw_pos) + ((1 - SMOOTHING_FACTOR) * smoothed_pos)

            if prev_pos is not None:
                dt = 1 / fps
                cur_v = (np.linalg.norm(smoothed_pos - prev_pos) / ppm) / dt
                if cur_v > VELOCITY_START_THRESHOLD and not is_pitching:
                    is_pitching = True
                    current_pitch_buffer, current_x_coords, current_y_coords = [], [], []
                if is_pitching:
                    accel_m = abs(cur_v - prev_vel) / dt
                    current_pitch_buffer.append([pitch_count+1, timestamp_ms, cur_v, accel_m, accel_m/GRAVITY])
                    current_x_coords.append(smoothed_pos[0]); current_y_coords.append(smoothed_pos[1])
                    trail_history.append((int(smoothed_pos[0]), int(smoothed_pos[1]), cur_v))
                    if cur_v < VELOCITY_STOP_THRESHOLD: low_speed_timer += 1
                    else: low_speed_timer = 0
                    if low_speed_timer > STOP_FRAME_BUFFER:
                        pitch_count += 1
                        v_list = [r[2] for r in current_pitch_buffer]
                        p_idx = np.argmax(v_list)
                        peak_marker.append((int(current_x_coords[p_idx]), int(current_y_coords[p_idx]), round(v_list[p_idx]*MS_TO_MPH, 1)))
                        pitch_summaries.append([pitch_count, round(v_list[p_idx]*MS_TO_MPH, 1), round(max([r[4] for r in current_pitch_buffer]), 1)])
                        is_pitching, low_speed_timer = False, 0
                prev_vel = cur_v
            prev_pos = smoothed_pos.copy()

            # --- 3. DUAL LEG VISUALIZATION ---
            # LEAD LEG (Yellow Knee, Orange Ankle)
            draw_protractor(frame, lm[LEAD_KNEE], lm[LEAD_HIP], lm[LEAD_ANKLE], l_knee_ang, (0, 255, 255))
            draw_protractor(frame, lm[LEAD_ANKLE], lm[LEAD_KNEE], lm[LEAD_FOOT], l_ankle_ang, (0, 165, 255))
            # DRIVE LEG (Lime Knee, Magenta Ankle)
            draw_protractor(frame, lm[DRIVE_KNEE], lm[DRIVE_HIP], lm[DRIVE_ANKLE], d_knee_ang, (0, 255, 0))
            draw_protractor(frame, lm[DRIVE_ANKLE], lm[DRIVE_KNEE], lm[DRIVE_FOOT], d_ankle_ang, (255, 0, 255))
            # ARM (Cyan)
            draw_protractor(frame, lm[ELBOW], lm[SHOULDER], lm[WRIST], elbow_ang, (255, 255, 0))

            # Skeleton Lines
            # Arm
            cv2.line(frame, (int(lm[SHOULDER].x*width), int(lm[SHOULDER].y*height)), (int(lm[ELBOW].x*width), int(lm[ELBOW].y*height)), (255, 255, 0), 2)
            cv2.line(frame, (int(lm[ELBOW].x*width), int(lm[ELBOW].y*height)), (int(lm[WRIST].x*width), int(lm[WRIST].y*height)), (255, 255, 0), 2)
            # Lead Leg
            cv2.line(frame, (int(lm[LEAD_HIP].x*width), int(lm[LEAD_HIP].y*height)), (int(lm[LEAD_KNEE].x*width), int(lm[LEAD_KNEE].y*height)), (0, 255, 255), 2)
            cv2.line(frame, (int(lm[LEAD_KNEE].x*width), int(lm[LEAD_KNEE].y*height)), (int(lm[LEAD_ANKLE].x*width), int(lm[LEAD_ANKLE].y*height)), (0, 255, 255), 2)
            cv2.line(frame, (int(lm[LEAD_ANKLE].x*width), int(lm[LEAD_ANKLE].y*height)), (int(lm[LEAD_FOOT].x*width), int(lm[LEAD_FOOT].y*height)), (0, 165, 255), 2)
            # Drive Leg
            cv2.line(frame, (int(lm[DRIVE_HIP].x*width), int(lm[DRIVE_HIP].y*height)), (int(lm[DRIVE_KNEE].x*width), int(lm[DRIVE_KNEE].y*height)), (0, 255, 0), 2)
            cv2.line(frame, (int(lm[DRIVE_KNEE].x*width), int(lm[DRIVE_KNEE].y*height)), (int(lm[DRIVE_ANKLE].x*width), int(lm[DRIVE_ANKLE].y*height)), (0, 255, 0), 2)
            cv2.line(frame, (int(lm[DRIVE_ANKLE].x*width), int(lm[DRIVE_ANKLE].y*height)), (int(lm[DRIVE_FOOT].x*width), int(lm[DRIVE_FOOT].y*height)), (255, 0, 255), 2)

        # Drawing HUD & Trail
        for i in range(1, len(trail_history)):
            cv2.line(frame, trail_history[i-1][:2], trail_history[i][:2], get_heatmap_color(trail_history[i][2]), 10, cv2.LINE_AA)
        for px, py, mph in peak_marker:
            cv2.circle(frame, (px, py), 20, (0, 0, 255), 2)
            cv2.drawMarker(frame, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(frame, f"{mph} mph", (px + 22, py + 5), 2, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        # Dashboard
        cv2.rectangle(frame, (20, 20), (450, 140), (0, 0, 0), -1)
        cv2.putText(frame, f"PITCH: {pitch_count}", (40, 75), 2, 1.5, (0, 255, 255), 2)
        cv2.putText(frame, f"LIVE: {prev_vel * MS_TO_MPH:.1f} mph", (40, 120), 2, 0.7, (255, 255, 255), 1)
        
        # Leg Legend
        cv2.putText(frame, "LEAD LEG", (300, 100), 1, 0.8, (0, 255, 255), 1)
        cv2.putText(frame, "DRIVE LEG", (300, 125), 1, 0.8, (0, 255, 0), 1)

        # Ticker
        t_sec = frame_count / fps
        timer_txt = f"{int(t_sec//60):02}:{int(t_sec%60):02}.{int((t_sec%1)*100):02}"

        text_size = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        text_x = width - text_size[0] - 40
        cv2.rectangle(frame, (text_x - 10, 20), (width - 20, 70), (0, 0, 0), -1)

        cv2.putText(frame, timer_txt, (text_x, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(frame, timer_txt, (width-220, 55), 2, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

with open(output_summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Pitch Number', 'Peak Velocity (mph)', 'Peak G-Force'])
    writer.writerows(pitch_summaries)

cap.release()
out.release()