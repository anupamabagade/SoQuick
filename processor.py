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

def process_lateral(input_path, output_path, p_height_inches, p_side, display_mode="All", slow_mo_factor=2):
    p_height_m = p_height_inches * 0.0254
    yolo_model = YOLO("yolov8n-pose.pt") # YOLO is best for fast movement
    
    # Landmark indices
    if p_side.upper() == 'RIGHT':
        WRIST, ELBOW, SHOULDER = 16, 14, 12
        L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
        Y_WRIST = 10 # YOLO Right Wrist
    else:
        WRIST, ELBOW, SHOULDER = 15, 13, 11
        L_HIP, L_KNEE, L_ANKLE = 24, 26, 28
        Y_WRIST = 9 # YOLO Left Wrist

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # FIX FOR CRASH: Change codec to 'mp4v'
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_mo_factor, (w, h))

        trail_history = []
        smoothed_pos, prev_pos = None, None
        is_pitching = False

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. YOLO for Wrist (Smoothness)
            yolo_results = yolo_model(frame, verbose=False)[0]
            wrist_coords = None
            if yolo_results.keypoints:
                kps = yolo_results.keypoints.xy[0].cpu().numpy()
                if kps[Y_WRIST][0] > 0:
                    wrist_coords = kps[Y_WRIST]

            # 2. MediaPipe for Angles
            timestamp_ms = int((frame_count / fps) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                ppm = abs(lm[30].y * h - lm[0].y * h) / p_height_m

                if wrist_coords is not None:
                    if smoothed_pos is None: smoothed_pos = wrist_coords
                    smoothed_pos = (SMOOTHING_FACTOR * wrist_coords) + (1 - SMOOTHING_FACTOR) * smoothed_pos
                    
                    # Logic to clear trail if arm resets (Fixes "double printing")
                    cur_v = 0
                    if prev_pos is not None:
                        cur_v = (np.linalg.norm(smoothed_pos - prev_pos) / ppm) * fps
                    
                    if cur_v > 6.0 and not is_pitching:
                        is_pitching = True
                        trail_history = [] # CLEAR OLD TRACE
                    elif cur_v < 2.0:
                        is_pitching = False
                    
                    if is_pitching:
                        trail_history.append((int(smoothed_pos[0]), int(smoothed_pos[1]), cur_v))
                    prev_pos = smoothed_pos.copy()

                # --- CONDITIONAL DRAWING (Prevents Clutter) ---
                if display_mode in ["All", "Arm Angles Only"]:
                    elbow_ang = get_angle_3d(lm[SHOULDER], lm[ELBOW], lm[WRIST])
                    #cv2.putText(frame, f"Elbow: {int(elbow_ang)}", (50, 100), 1, 2, (255, 255, 0), 2)
                
                if display_mode in ["All", "Leg Angles Only"]:
                    knee_ang = get_angle_3d(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE])
                    #cv2.putText(frame, f"Knee: {int(knee_ang)}", (50, 150), 1, 2, (0, 255, 255), 2)

            # Draw Trace
            if display_mode in ["All", "Wrist Trace & Velocity Only"]:
                for i in range(1, len(trail_history)):
                    cv2.line(frame, trail_history[i-1][:2], trail_history[i][:2], get_heatmap_color(trail_history[i][2]), 8)

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