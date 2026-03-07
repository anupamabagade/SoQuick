import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
model_path = 'pose_landmarker_heavy.task' 
input_video = 'KPickens1009.mp4'
output_video = 'KPickens1009_Rotational_Analysis.mp4'
slow_motion_factor = 2
FREEZE_DURATION_SEC = 3 

# Landmark Mapping
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24

def draw_sleek_label(img, text, pos, color=(255, 255, 255), base_scale=0.8, thickness_mult=1):
    """
    Robust UI label: Handles resolution scaling and ensures integer coordinates 
    to prevent OpenCV 'Bad argument' errors.
    """
    h, w = img.shape[:2]
    
    # 1. Scaling Logic
    ui_scale = max(0.45, (w / 1000) * base_scale)
    thickness = max(1, int(ui_scale * 2 * thickness_mult))
    padding = int(25 * (w / 1000))
    bar_width = max(4, int(8 * (w / 1000)))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (txt_w, txt_h), baseline = cv2.getTextSize(text, font, ui_scale, thickness)
    
    # FORCE INTEGER COORDINATES
    x, y = int(pos[0]), int(pos[1])
    
    if x == -1:
        x = int((w - txt_w) / 2)
    if x + txt_w + padding > w:
        x = int(w - txt_w - padding - 10)
    if x - padding < 0:
        x = int(padding + 5)

    # 2. Define Rectangles as Integer Tuples
    rect_top_left = (int(x - padding), int(y - txt_h - padding))
    rect_bottom_right = (int(x + txt_w + padding), int(y + baseline + padding))
    bar_right_edge = (int(x - padding + bar_width), int(y + baseline + padding))

    # 3. Background Overlay
    overlay = img.copy()
    cv2.rectangle(overlay, rect_top_left, rect_bottom_right, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # 4. Scaled Accent Bar
    cv2.rectangle(img, rect_top_left, bar_right_edge, color, -1)
    
    # 5. Draw Text (Ensure x,y are ints)
    cv2.putText(img, text, (int(x), int(y)), font, ui_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return (txt_w, txt_h)

def get_line_rotation(p1, p2):
    return np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))

# --- Main Engine ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

max_separation = 0
max_x_time = "00:00.00"
final_frame = None

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_motion_factor, (w, h))

    # Scaling Factors
    y_step = h * 0.10 
    margin_x = int(w * 0.05)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        timestamp_ms = int((frame_count / fps) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        total_sec = frame_count / fps
        time_str = f"{int(total_sec//60):02}:{int(total_sec%60):02}.{int((total_sec%1)*100):02}"

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            
            s_ang = abs(get_line_rotation(lm[L_SH], lm[R_SH]))
            h_ang = abs(get_line_rotation(lm[L_HIP], lm[R_HIP]))
            separation = abs(s_ang - h_ang)

            if separation > max_separation:
                max_separation = separation
                max_x_time = time_str

            # Draw Lines
            line_thick = max(2, int(w/300))
            cv2.line(frame, (int(lm[L_SH].x*w), int(lm[L_SH].y*h)), (int(lm[R_SH].x*w), int(lm[R_SH].y*h)), (255, 0, 255), line_thick, cv2.LINE_AA)
            cv2.line(frame, (int(lm[L_HIP].x*w), int(lm[L_HIP].y*h)), (int(lm[R_HIP].x*w), int(lm[R_HIP].y*h)), (255, 255, 0), line_thick, cv2.LINE_AA)

            # Dashboard - Coordinates wrapped in int() for extra safety
            draw_sleek_label(frame, f"SHOULDER: {s_ang:.1f} DEG", (margin_x, int(y_step)), (255, 0, 255), 0.7)
            draw_sleek_label(frame, f"HIP: {h_ang:.1f} DEG", (margin_x, int(y_step * 2)), (255, 255, 0), 0.7)
            draw_sleek_label(frame, f"SEPARATION: {separation:.1f} DEG", (margin_x, int(y_step * 3.2)), (0, 255, 0), 1.0, 1.5)

        # Timer
        draw_sleek_label(frame, f"TIME: {time_str}", (w, int(y_step)), (180, 180, 180), 0.6)

        if frame_count == total_frames - 1:
            summary_txt = f"MAX SEPARATION: {max_separation:.1f} DEG | AT {max_x_time}"
            draw_sleek_label(frame, summary_txt, (-1, int(h - y_step)), (0, 255, 255), 0.8, 1.5)
            final_frame = frame.copy()

        out.write(frame)
        frame_count += 1

    if final_frame is not None:
        for _ in range(int((fps / slow_motion_factor) * FREEZE_DURATION_SEC)):
            out.write(final_frame)

    cap.release()
    out.release()