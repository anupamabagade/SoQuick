import streamlit as st
import os
import subprocess
import time
import processor  # Ensure your processor.py contains process_lateral and process_back

# --- UI Configuration ---
st.set_page_config(page_title="SoQuick | Biomechanical Portal", layout="wide")
st.title("⚾ SoQuick Pitching Analysis")

# --- Sidebar: The Toggle & Inputs ---
with st.sidebar:
    st.header("Analysis Controls")
    # This restores the toggle you were looking for
    view_type = st.radio("Select Analysis View", ["Lateral (Trace)", "Back (Separation)"])
    
    st.markdown("---")
    
    # Contextual inputs based on the toggle
    if view_type == "Lateral (Side)":
        st.subheader("Lateral Parameters")
        p_height = st.number_input("Pitcher Height (inches)", value=73)
        p_side = st.selectbox("Pitching Arm", ["Right", "Left"])
    else:
        st.subheader("Back View Parameters")
        st.info("Detecting Shoulder-Hip Separation (X-Factor)")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload Pitching Video", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    # 1. Save upload to disk
    input_path = "input_temp.mp4"
    raw_output = "raw_analyzed.mp4"
    web_ready = "web_ready.mp4"
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 Run Analysis"):
        # Cleanup old files
        for f in [raw_output, web_ready]:
            if os.path.exists(f): os.remove(f)
            
        status = st.status(f"Processing {view_type} View...", expanded=True)
        
        try:
            # 2. TRIGGER THE CORRECT LOGIC
            # This calls the specific functions in your processor.py
            if view_type == "Lateral (Side)":
                status.update(label="Calculating Velocity & Leg Drive...")
                processor.process_lateral(input_path, raw_output, p_height, p_side)
            else:
                status.update(label="Analyzing Rotational X-Factor...")
                processor.process_back(input_path, raw_output)

            # 3. SAFETY CHECK: Did the engine finish?
            if not os.path.exists(raw_output):
                st.error("Analysis engine failed to produce a file. Check processor.py logs.")
                st.stop()

            # 4. CONVERT FOR WEB (The "No-Video Bug" Fix)
            status.update(label="Optimizing Video for Browser Playback...")
            subprocess.run([
                'ffmpeg', '-i', raw_output,
                '-vcodec', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '28',
                web_ready, '-y'
            ], capture_output=True)

            # 5. DISPLAY RESULTS
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            st.subheader(f"Final {view_type} Analysis")
            st.video(web_ready)
            
            # Allow download of the analyzed file
            with open(web_ready, "rb") as file:
                st.download_button(
                    label="📥 Download Analyzed Video",
                    data=file,
                    file_name=f"SoQuick_{view_type.split()[0]}_Analysis.mp4",
                    mime="video/mp4"
                )

        except Exception as e:
            st.error(f"UI Error: {e}")