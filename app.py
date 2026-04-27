import streamlit as st
import tempfile
import os
import subprocess  # Added to handle the H.264 conversion
import processor 

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
    pitcher_height = st.number_input("Pitcher Height (Inches)", min_value=40, max_value=90, value=62)
    pitcher_side = st.radio("Throwing Hand", ["RIGHT", "LEFT"])
    
    st.divider()
    st.write("### Analysis Settings")
    # 1x to 4x slow motion slider
    slow_mo_val = st.slider("Slow Motion Factor", min_value=1, max_value=4, value=2)

# --- Main UI: View & Display Selection ---
view_mode = st.selectbox(
    "Select Camera View", 
    ["Lateral (Side) View", "Back View"],
    help="Choose Lateral for velocity/legs or Back View for hip-shoulder separation."
)

# Conditional Display Selection for Lateral View
display_mode = "All"
if view_mode == "Lateral (Side) View":
    display_mode = st.selectbox(
        "Select Measurements to Display",
        ["All", "Wrist Trace & Velocity Only", "Arm Angles Only", "Leg Angles Only"]
    )

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 1. Create temporary input file
    t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    t_in.write(uploaded_file.read())
    t_in.close() 
    
    # 2. Define filenames for the two-step process
    output_filename = "raw_analyzed.mp4" # Raw file from OpenCV (mp4v)
    final_output = "web_ready.mp4"      # Final file for Streamlit (H.264)

    if st.button("🚀 Run Analysis"):
        with st.spinner(f"Processing {view_mode}... This may take a moment."):
            try:
                # --- STEP 1: Run Physics Engine ---
                if view_mode == "Lateral (Side) View":
                    processor.process_lateral(
                        input_path=t_in.name, 
                        output_path=output_filename, 
                        p_height_inches=pitcher_height, 
                        p_side=pitcher_side,
                        display_mode=display_mode,
                        slow_mo_factor=slow_mo_val
                    )
                else:
                    processor.process_back(
                        input_path=t_in.name, 
                        output_path=output_filename, 
                        slow_mo_factor=slow_mo_val
                    )

                # --- STEP 2: Convert to Web-Compatible H.264 ---
                # This fixes the "Black Screen" issue in browsers
                subprocess.run([
                    'ffmpeg', '-y', '-i', output_filename, 
                    '-c:v', 'libx264', 
                    '-pix_fmt', 'yuv420p', 
                    '-preset', 'ultrafast', 
                    final_output
                ], check=True)

                # --- STEP 3: Display Results ---
                if os.path.exists(final_output):
                    st.success("Analysis Complete!")
                    st.video(final_output)
                    
                    with open(final_output, "rb") as file:
                        st.download_button(
                            label="📥 Download Analyzed Video",
                            data=file,
                            file_name=f"PitchAnalysis_{view_mode.split()[0]}.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Conversion failed. Video could not be optimized for web.")

            except Exception as e:
                st.error(f"Analysis Error: {e}")
            
            finally:
                # Cleanup: Remove temp and raw files, but keep final_output for display/download
                for f in [t_in.name, output_filename]:
                    if os.path.exists(f): 
                        os.remove(f)

else:
    st.info("Please upload a video file to begin.")

st.divider()
st.caption("Powered by MediaPipe Pose Landmark Detection.")