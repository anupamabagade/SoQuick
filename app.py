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