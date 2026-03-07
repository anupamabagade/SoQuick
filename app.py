import streamlit as st
import os
import subprocess
import time
import processor  # Assuming your logic is in processor.py

# --- Page Config ---
st.set_page_config(page_title="SoQuick | Pitching Analysis", layout="wide")

st.title("⚾ SoQuick Biomechanical Portal")
st.markdown("---")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Analysis Settings")
    view_type = st.radio("Select Video View", ["Lateral (Side)", "Back (X-Factor)"])
    
    if view_type == "Lateral (Side)":
        pitcher_height = st.number_input("Pitcher Height (inches)", min_value=40, max_value=90, value=72)
        pitcher_side = st.selectbox("Pitching Arm", ["Right", "Left"])
    
    st.info("Note: Processing takes ~20-40 seconds depending on video length.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a pitching clip (MP4, MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    input_path = "input_video.mp4"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Define paths for the stages of processing
    raw_output = "temp_output.mp4"
    web_ready = "web_ready.mp4"

    # Cleanup old files from previous runs
    for f in [raw_output, web_ready]:
        if os.path.exists(f):
            os.remove(f)

    if st.button("🚀 Start Biomechanical Analysis"):
        status = st.status("Initializing Analysis Engine...", expanded=True)
        
        try:
            # 1. RUN THE ANALYSIS (The "Engine" Phase)
            status.update(label="Running Pose Estimation & Physics Engine...", state="running")
            
            if view_type == "Lateral (Side)":
                # Ensure these function names match your processor.py exactly
                processor.process_lateral(input_path, raw_output, pitcher_height, pitcher_side)
            else:
                processor.process_back(input_path, raw_output)

            # 2. SAFETY GATE: Verify the file was actually created
            if not os.path.exists(raw_output):
                status.update(label="Analysis Failed!", state="error")
                st.error("The analysis engine crashed or failed to save the video. Please check your logs.")
                st.stop()

            # 3. CONVERT FOR WEB (The "FFmpeg" Phase)
            status.update(label="Optimizing Video for Web Playback...", state="running")
            
            # We use subprocess to call the ffmpeg package installed via packages.txt
            conversion = subprocess.run([
                'ffmpeg', '-i', raw_output,
                '-vcodec', 'libx264',
                '-preset', 'ultrafast', # Keeps it fast on Streamlit's limited CPU
                '-crf', '28',           # Balanced quality/file size
                web_ready, '-y'
            ], capture_output=True, text=True)

            # Check if FFmpeg succeeded
            if conversion.returncode != 0:
                st.error("FFmpeg optimization failed.")
                st.code(conversion.stderr)
                st.stop()

            # 4. FINAL DISPLAY
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Analyzed Video")
                st.video(web_ready)
            
            with col2:
                st.subheader("Performance Metrics")
                # If your processor creates a CSV, you can display it here
                if os.path.exists("report.csv"):
                    st.download_button("Download Scouting Report", "report.csv")
                    st.dataframe("report.csv")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            status.update(label="Error Detected", state="error")

else:
    st.warning("Please upload a video file to begin.")