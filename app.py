import streamlit as st
import os
import subprocess
import processor

st.set_page_config(page_title="SoQuick | Biomechanical Portal", layout="wide")
st.title("SoQuick Pitching Analysis")


# ---------------------------------------------------------------------------
# Helper: process one video end-to-end, return freeze frames
# ---------------------------------------------------------------------------
def run_analysis(input_path, raw_output, web_ready, view_type, params):
    freeze_frames = []

    if view_type == "Lateral (Trace)":
        freeze_frames = processor.process_lateral(
            input_path, raw_output,
            params["height"], params["side"],
            display_mode=params["display_mode"],
            slow_mo_factor=params["slow_mo"],
        ) or []
    else:
        processor.process_back(input_path, raw_output, slow_mo_factor=params["slow_mo"])

    if not os.path.exists(raw_output):
        raise RuntimeError("Processor produced no output file.")

    subprocess.run(
        ["ffmpeg", "-i", raw_output, "-vcodec", "libx264",
         "-preset", "ultrafast", "-crf", "28", web_ready, "-y"],
        capture_output=True,
    )
    return freeze_frames


def show_results(col, label, web_ready, freeze_frames, view_type):
    """Render video, download button, and freeze frames inside a column."""
    with col:
        st.subheader(label)
        st.video(web_ready)
        with open(web_ready, "rb") as f:
            st.download_button(
                label="Download Video",
                data=f,
                file_name=f"SoQuick_{label.replace(' ', '_')}_{view_type.split()[0]}.mp4",
                mime="video/mp4",
                key=f"dl_{label}",
            )
        if freeze_frames:
            st.markdown("---")
            st.markdown("**Key Moments**")
            for ff in freeze_frames:
                st.markdown(f"*{ff['label']}*")
                st.image(ff["path"], use_container_width=True)
                if ff.get("stride"):
                    s = ff["stride"]
                    st.caption(
                        f"Feet apart (2D): **{s['full_ft']} ft** &nbsp;|&nbsp; "
                        f"Horizontal: **{s['horiz_ft']} ft**"
                    )
                st.markdown("")


# ---------------------------------------------------------------------------
# Sidebar — shared settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Analysis Controls")

    mode = st.radio("Mode", ["Single Video", "Compare Two Videos"])

    st.markdown("---")

    view_type = st.radio("Analysis View", ["Lateral (Trace)", "Back (Separation)"])

    st.markdown("---")

    slow_mo = st.slider("Slow Motion Factor", min_value=1, max_value=4, value=2)

    if view_type == "Lateral (Trace)":
        display_mode = st.selectbox(
            "Measurements to Display",
            ["All", "Wrist Trace & Velocity Only", "Arm Angles Only", "Leg Angles Only"],
        )
    else:
        display_mode = "All"
        st.info("Detecting Shoulder-Hip Separation")


# ---------------------------------------------------------------------------
# SINGLE VIDEO MODE
# ---------------------------------------------------------------------------
if mode == "Single Video":
    if view_type == "Lateral (Trace)":
        col_l, col_r = st.columns(2)
        with col_l:
            p_height = st.number_input("Pitcher Height (inches)", value=62, key="h_single")
        with col_r:
            p_side = st.selectbox("Pitching Arm", ["Right", "Left"], key="s_single")
    else:
        p_height, p_side = 62, "Right"

    uploaded_file = st.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

    if uploaded_file:
        input_path = "input_temp.mp4"
        raw_output = "raw_analyzed.mp4"
        web_ready  = "web_ready.mp4"

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Run Analysis"):
            for path in [raw_output, web_ready]:
                if os.path.exists(path):
                    os.remove(path)

            with st.status("Processing…", expanded=True) as status:
                try:
                    params = {
                        "height": p_height,
                        "side": p_side,
                        "display_mode": display_mode,
                        "slow_mo": slow_mo,
                    }
                    freeze_frames = run_analysis(
                        input_path, raw_output, web_ready, view_type, params
                    )
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    show_results(st, "Analysis", web_ready, freeze_frames, view_type)
                except Exception as e:
                    st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# COMPARISON MODE
# ---------------------------------------------------------------------------
else:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Pitcher A")
        if view_type == "Lateral (Trace)":
            p_height_a = st.number_input("Height (inches)", value=62, key="h_a")
            p_side_a   = st.selectbox("Pitching Arm", ["Right", "Left"], key="s_a")
        else:
            p_height_a, p_side_a = 62, "Right"
        uploaded_a = st.file_uploader("Upload Video A", type=["mp4", "mov", "avi"], key="up_a")

    with col_b:
        st.subheader("Pitcher B")
        if view_type == "Lateral (Trace)":
            p_height_b = st.number_input("Height (inches)", value=62, key="h_b")
            p_side_b   = st.selectbox("Pitching Arm", ["Right", "Left"], key="s_b")
        else:
            p_height_b, p_side_b = 62, "Right"
        uploaded_b = st.file_uploader("Upload Video B", type=["mp4", "mov", "avi"], key="up_b")

    both_ready = uploaded_a is not None and uploaded_b is not None
    if not both_ready:
        st.info("Upload both videos to enable comparison.")

    if both_ready and st.button("Run Both Analyses"):
        # Write uploads to disk
        with open("input_temp_A.mp4", "wb") as f:
            f.write(uploaded_a.getbuffer())
        with open("input_temp_B.mp4", "wb") as f:
            f.write(uploaded_b.getbuffer())

        # Clean up old outputs
        for path in ["raw_analyzed_A.mp4", "web_ready_A.mp4",
                     "raw_analyzed_B.mp4", "web_ready_B.mp4"]:
            if os.path.exists(path):
                os.remove(path)

        params_a = {"height": p_height_a, "side": p_side_a,
                    "display_mode": display_mode, "slow_mo": slow_mo}
        params_b = {"height": p_height_b, "side": p_side_b,
                    "display_mode": display_mode, "slow_mo": slow_mo}

        freeze_a, freeze_b = [], []
        error_a, error_b   = None, None

        with st.status("Processing Pitcher A…", expanded=True) as status:
            try:
                freeze_a = run_analysis(
                    "input_temp_A.mp4", "raw_analyzed_A.mp4", "web_ready_A.mp4",
                    view_type, params_a,
                )
                status.update(label="Pitcher A complete.")
            except Exception as e:
                error_a = str(e)
                status.update(label=f"Pitcher A failed: {e}", state="error")

        with st.status("Processing Pitcher B…", expanded=True) as status:
            try:
                freeze_b = run_analysis(
                    "input_temp_B.mp4", "raw_analyzed_B.mp4", "web_ready_B.mp4",
                    view_type, params_b,
                )
                status.update(label="Pitcher B complete.")
            except Exception as e:
                error_b = str(e)
                status.update(label=f"Pitcher B failed: {e}", state="error")

        # Display results side by side
        res_a, res_b = st.columns(2)

        if error_a:
            with res_a:
                st.error(f"Pitcher A error: {error_a}")
        elif os.path.exists("web_ready_A.mp4"):
            show_results(res_a, "Pitcher A", "web_ready_A.mp4", freeze_a, view_type)

        if error_b:
            with res_b:
                st.error(f"Pitcher B error: {error_b}")
        elif os.path.exists("web_ready_B.mp4"):
            show_results(res_b, "Pitcher B", "web_ready_B.mp4", freeze_b, view_type)
