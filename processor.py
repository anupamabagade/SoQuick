import streamlit as st
import tempfile
import os
import subprocess
import processor 

st.set_page_config(page_title="Pitcher Analysis Portal", page_icon="⚾", layout="centered")

st.title("⚾ Softball Pitching Analysis")

with st.sidebar:
    st.header("Pitcher Profile")
    pitcher_height = st.number_input("Pitcher Height (Inches)", min_value=40, max_value=90, value=72)
    pitcher_side = st.radio("Throwing Hand", ["RIGHT", "LEFT"])
    slow_mo = st.checkbox("Slow Motion Output (2x)", value=True)

# UI to fix "Double Printing" clutter
view_mode = st.selectbox("Select Camera View", ["Lateral (Side) View", "Back View"])
display_mode = "All"
if view_mode == "Lateral (Side) View":
    display_mode = st.selectbox(
        "Select Measurements to Display",
        ["All", "Wrist Trace & Velocity Only", "Arm Angles Only", "Leg Angles Only"]
    )

uploaded_file = st.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    t_in.write(uploaded_file.read())
    t_in.close()
    
    raw_output = "raw_processed.mp4"
    final_output = "web_ready.mp4"

    if st.button("🚀 Run Analysis"):
        with st.spinner("Processing..."):
            try:
                if view_mode == "Lateral (Side) View":
                    processor.process_lateral(
                        t_in.name, raw_output, pitcher_height, pitcher_side, 
                        display_mode=display_mode, 
                        slow_mo_factor=2 if slow_mo else 1
                    )
                else:
                    processor.process_back(t_in.name, raw_output)

                # --- FIX FOR BLACK SCREEN ---
                # Convert to H.264 using ffmpeg so it plays in the browser
                subprocess.run([
                    'ffmpeg', '-y', '-i', raw_output, 
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', final_output
                ], check=True)

                if os.path.exists(final_output):
                    st.success("Analysis Complete!")
                    st.video(final_output)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(t_in.name): os.remove(t_in.name)

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