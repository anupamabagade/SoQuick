"""
api.py — SoQuick FastAPI backend
---------------------------------
POST /analyze
  Accepts a pitching video + parameters, runs process_lateral,
  and returns the 4 key-moment freeze-frame images as base64 JSON.

Run locally:
    uvicorn api:app --reload --port 8000

Test:
    curl -s -X POST http://localhost:8000/analyze \
         -F "video=@mypitch.mp4" \
         -F "p_height=65" \
         -F "p_side=Right" \
         | python3 -m json.tool | head -60
"""

import os
import uuid
import base64
import glob

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import processor

app = FastAPI(title="SoQuick Analysis API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-test")
def model_test():
    """Tries to create a PoseLandmarker and reports success or the exact error."""
    import traceback, os
    model_path = os.environ.get("MEDIAPIPE_MODEL_PATH", "pose_landmarker_heavy.task")
    model_exists = os.path.exists(model_path)
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
        )
        lm = vision.PoseLandmarker.create_from_options(opts)
        lm.close()
        return {"status": "ok", "model": model_path, "model_exists": model_exists}
    except Exception as exc:
        return {"status": "error", "model": model_path, "model_exists": model_exists,
                "error": str(exc), "traceback": traceback.format_exc()}


@app.get("/debug")
def debug():
    import subprocess, platform
    ldconfig = subprocess.run(
        ["ldconfig", "-p"], capture_output=True, text=True
    ).stdout
    gles_cached = [l.strip() for l in ldconfig.splitlines() if "GLES" in l or "gles" in l.lower()]
    all_gl = subprocess.run(
        ["find", "/", "-name", "lib*GL*", "-not", "-path", "*/proc/*"],
        capture_output=True, text=True
    ).stdout.strip()
    pkgs = subprocess.run(
        ["dpkg", "-l"], capture_output=True, text=True
    ).stdout
    gl_pkgs = [l for l in pkgs.splitlines() if "gl" in l.lower() or "egl" in l.lower() or "gles" in l.lower()]
    return {
        "arch":         platform.machine(),
        "gles_ldconfig": gles_cached,
        "all_gl_files":  all_gl or "NONE",
        "gl_packages":   gl_pkgs,
    }


@app.post("/analyze")
async def analyze(
    video:    UploadFile = File(...),
    p_height: int        = Form(62),
    p_side:   str        = Form("Right"),
):
    job    = uuid.uuid4().hex[:8]
    tmp_in = f"/tmp/{job}_input.mp4"
    tmp_out = f"/tmp/{job}_raw.avi"

    with open(tmp_in, "wb") as f:
        f.write(await video.read())

    try:
        freeze_frames = processor.process_lateral(
            input_path      = tmp_in,
            output_path     = tmp_out,
            p_height_inches = p_height,
            p_side          = p_side,
            display_mode    = "Angles Only",  # skips YOLO/PyTorch — fits free-tier RAM
            slow_mo_factor  = 1,
        )
    except Exception as exc:
        import traceback
        _cleanup(tmp_in, tmp_out)
        return JSONResponse(status_code=500, content={
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    results = []
    for ff in freeze_frames:
        img_path = ff.get("path", "")
        if not os.path.exists(img_path):
            continue
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        stride = {k: float(v) for k, v in (ff.get("stride") or {}).items()}
        results.append({
            "label":     ff.get("label", ""),
            "image_b64": b64,
            "stride":    stride,
        })

    _cleanup(tmp_in, tmp_out)
    return {"freeze_frames": results}


def _cleanup(*paths):
    for p in paths:
        try: os.remove(p)
        except FileNotFoundError: pass
    for p in glob.glob("/tmp/*_freeze_*.jpg"):
        try: os.remove(p)
        except FileNotFoundError: pass
