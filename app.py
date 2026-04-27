import streamlit as st
import tempfile
import os
import processor  # Ensure processor.py is in the same folder

# --- Page Config ---
st.set_page_config(
    page_title="Pitcher Analysis Portal",
    page_icon="⚾",
    layout="centered"
)

st.title("⚾ Softball Pitching Analysis")

# --- Sidebar: User Profile ---
with st.sidebar:
    st.header("Pitcher Profile")
    pitcher_height = st.number_input("Pitcher Height (Inches)", min_value=40, max_value=90, value=72)
    pitcher_side = st.radio("Throwing Hand", ["RIGHT", "LEFT"])
    
    st.divider()
    st.write("### Analysis Settings")
    slow_mo = st.checkbox("Slow Motion Output (2x)", value=True)

# --- NEW: View & Display Toggles ---
view_mode = st.selectbox(
    "Select Camera View",
    ["Lateral (Side) View", "Back View"]
)

# New dropdown to stop "double printing" clutter
display_mode = "All"
if view_mode == "Lateral (Side) View":
    display_mode = st.selectbox(
        "Select Measurement Display",
        [
            "All",
            "Wrist trace + Peak velocity",
            "Arm angles (Elbow)",
            "Leg angles (Knee/Ankle)"
        ]
    )

# --- File Upload ---
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
                    # Pass display_mode to the processor
                    processor.process_lateral(
                        input_path=t_in.name,
                        output_path=output_filename,
                        p_height_inches=pitcher_height,
                        p_side=pitcher_side,
                        display_mode=display_mode,
                        slow_mo_factor=2 if slow_mo else 1
                    )
                else:
                    processor.process_back(
                        input_path=t_in.name,
                        output_path=output_filename,
                        slow_mo_factor=2 if slow_mo else 1
                    )

                if os.path.exists(output_filename):
                    st.success("Analysis Complete!")
                    st.video(output_filename) # If black screen persists, see processor notes
                    
                    with open(output_filename, "rb") as file:
                        st.download_button(
                            label="📥 Download Video",
                            data=file,
                            file_name="PitchAnalysis.mp4",
                            mime="video/mp4"
                        )
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if os.path.exists(t_in.name):
                    os.remove(t_in.name)