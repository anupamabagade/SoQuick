import streamlit as st
import tempfile
import os
import processor 

st.set_page_config(page_title="Pitcher Analysis Portal", page_icon="⚾", layout="centered")

st.title("⚾ Softball Pitching Analysis")

with st.sidebar:
    st.header("Pitcher Profile")
    pitcher_height = st.number_input("Pitcher Height (Inches)", min_value=40, max_value=90, value=72)
    pitcher_side = st.radio("Throwing Hand", ["RIGHT", "LEFT"])
    
    st.divider()
    st.write("### Analysis Settings")
    # Updated to your requested 1-4x slider
    slow_mo_val = st.slider("Slow Motion Factor", min_value=1, max_value=4, value=2)

view_mode = st.selectbox("Select Camera View", ["Lateral (Side) View", "Back View"])

# --- NEW: Display Selection Dropdown ---
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
    output_filename = "analyzed_output.mp4"

    if st.button("🚀 Run Analysis"):
        with st.spinner(f"Processing..."):
            try:
                if view_mode == "Lateral (Side) View":
                    processor.process_lateral(
                        t_in.name, output_filename, 
                        pitcher_height, pitcher_side, 
                        display_mode=display_mode, # Passing the selection
                        slow_mo_factor=slow_mo_val
                    )
                else:
                    processor.process_back(
                        t_in.name, output_filename, 
                        slow_mo_factor=slow_mo_val
                    )

                # ... inside the "Run Analysis" button logic ...

                # 3. Display Success & Video
                if os.path.exists(final_output):
                    st.success("Analysis Complete!")
                    st.video(final_output) # Always play the final, web-compatible version
                    
                    # Download Button should also use the final_output
                    with open(final_output, "rb") as file:
                        st.download_button(
                            label="📥 Download Analyzed Video",
                            data=file,
                            file_name=f"PitchAnalysis_{view_mode.split()[0]}.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Analysis failed to generate the web-ready output video.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                for f in [t_in.name, output_filename]: # Cleanup raw and temp files
                    if os.path.exists(f): 
                        os.remove(f)