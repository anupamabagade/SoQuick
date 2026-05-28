import streamlit as st
import os
import subprocess
import time
import processor  # Ensure your processor.py contains process_lateral and process_back

# --- UI Configuration ---
st.set_page_config(page_title="SoQuick | Biomechanical Portal", layout="wide")
st.title("🥎 SoQuick Pitching Analysis")

# --- Sidebar: The Toggle & Inputs ---
with st.sidebar:
    st.header("Analysis Controls")
    # This restores the toggle you were looking for
    view_type = st.radio("Select Analysis View", ["Lateral (Trace)", "Back (Separation)"])
    
    st.markdown("---")
    
    # Contextual inputs based on the toggle
    if view_type == "Lateral (Trace)":
        st.subheader("Lateral Parameters")
        p_height = st.number_input("Pitcher Height (inches)", value=62)
        p_side = st.selectbox("Pitching Arm", ["Right", "Left"])
        display_mode = st.selectbox(
            "Measurements to Display",
            ["All", "Wrist Trace & Velocity Only", "Arm Angles Only", "Leg Angles Only"]
        )
        slow_mo = st.slider("Slow Motion Factor", min_value=1, max_value=4, value=2)
    else:
        slow_mo = st.slider("Slow Motion Factor", min_value=1, max_value=4, value=2)
        st.subheader("Back View Parameters")
        st.info("Detecting Shoulder-Hip Separation")

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
            freeze_frames = []
            if view_type == "Lateral (Trace)":
                status.update(label="Calculating Velocity & Leg Drive...")
                freeze_frames = processor.process_lateral(input_path, raw_output, p_height, p_side, display_mode=display_mode, slow_mo_factor=slow_mo) or []
            else:
                status.update(label="Analyzing pitch...")
                processor.process_back(input_path, raw_output, slow_mo_factor=slow_mo)

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

            # 6. FREEZE FRAMES (lateral mode only)
            if freeze_frames:
                st.markdown("---")
                st.subheader("Key Moments")
                for ff in freeze_frames:
                    st.markdown(f"**{ff['label']}**")
                    st.image(ff['path'], use_container_width=True)
                    if ff.get('stride'):
                        s = ff['stride']
                        st.caption(
                            f"Stride (feet apart, 2D): **{s['full_ft']} ft** &nbsp;|&nbsp; "
                            f"Horizontal distance: **{s['horiz_ft']} ft**"
                        )
                    st.markdown("")   # spacer

        except Exception as e:
            st.error(f"UI Error: {e}")