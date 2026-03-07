import streamlit as st
import tempfile
import os
import processor  # Ensure processor.py is in the same folder
import subprocess

# 1. Running analysis
# --- Page Config ---
st.set_page_config(
    page_title="Pitcher Analysis Portal",
    page_icon="⚾",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚾ Softball Pitching Analysis")
st.write("Upload a pitching clip to generate automated velocity and mechanical insights.")

# --- Sidebar: User Profile ---
with st.sidebar:
    st.header("Pitcher Profile")
    st.info("These metrics ensure accurate MPH and scaling calculations.")
    pitcher_height = st.number_input("Pitcher Height (Inches)", min_value=40, max_value=90, value=72)
    pitcher_side = st.radio("Throwing Hand", ["RIGHT", "LEFT"])
    
    st.divider()
    st.write("### Analysis Settings")
    slow_mo = st.checkbox("Slow Motion Output (2x)", value=True)

# --- Main UI: Toggle View ---
view_mode = st.selectbox(
    "Select Camera View",
    ["Lateral (Side) View", "Back View"],
    help="Choose Lateral for velocity/legs or Back View for hip-shoulder separation."
)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 1. Create temporary files for processing
    t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    t_in.write(uploaded_file.read())
    t_in.close() # Close to allow CV2 to open it
    
    output_filename = "analyzed_output.mp4"

    # 2. Start Analysis Button
    if st.button("🚀 Run Analysis"):
        with st.spinner(f"Processing {view_mode}... This may take a minute."):
            try:
                if view_mode == "Lateral (Side) View":
                    # Call the Lateral Engine
                    processor.process_lateral(
                        input_path=t_in.name,
                        output_path=output_filename,
                        p_height_inches=pitcher_height,
                        p_side=pitcher_side
                    )
                else:
                    # Call the Back View Engine
                    processor.process_back(
                        input_path=t_in.name,
                        output_path=output_filename
                    )

                # 3. Display Success & Video
                if os.path.exists(output_filename):
                    st.success("Analysis Complete!")
                    st.video(output_filename)
                    
                    # Download Button
                    with open(output_filename, "rb") as file:
                        st.download_button(
                            label="📥 Download Analyzed Video",
                            data=file,
                            file_name=f"PitchAnalysis_{view_mode.split()[0]}.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Analysis failed to generate output video.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                # Cleanup temporary file
                if os.path.exists(t_in.name):
                    os.remove(t_in.name)

else:
    st.info("Please upload a video file to begin.")

# --- Footer ---
st.divider()
st.caption("Powered by MediaPipe Pose Landmark Detection.")

# 2. Convert to Web-Friendly H.264
st.info("Optimizing video for web playback...")
subprocess.run([
    'ffmpeg', '-i', 'temp_output.mp4', 
    '-vcodec', 'libx264', 
    '-preset', 'ultrafast', 
    '-crf', '28', 
    'web_ready.mp4', '-y'
])

# 3. Display the converted video
st.video('web_ready.mp4')