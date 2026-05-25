import uuid
import os
import subprocess
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import processor

app = FastAPI(title="SoQuick Analysis API")


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    view_type: str = Form("lateral"),
    p_height: int = Form(62),
    p_side: str = Form("Right"),
    display_mode: str = Form("All"),
    slow_mo: int = Form(2),
):
    job = str(uuid.uuid4())[:8]
    input_path = f"/tmp/{job}_input.mp4"
    raw_output = f"/tmp/{job}_raw.mp4"
    web_ready  = f"/tmp/{job}_web.mp4"

    try:
        with open(input_path, "wb") as f:
            f.write(await video.read())

        if view_type == "lateral":
            processor.process_lateral(
                input_path, raw_output,
                p_height, p_side,
                display_mode=display_mode,
                slow_mo_factor=slow_mo,
            )
        elif view_type == "back":
            processor.process_back(input_path, raw_output, slow_mo_factor=slow_mo)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown view_type: {view_type}")

        if not os.path.exists(raw_output):
            raise HTTPException(status_code=500, detail="Processor produced no output file.")

        result = subprocess.run(
            ["ffmpeg", "-i", raw_output, "-vcodec", "libx264",
             "-preset", "ultrafast", "-crf", "28", web_ready, "-y"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="ffmpeg re-encode failed.")

        return FileResponse(
            web_ready,
            media_type="video/mp4",
            filename=f"SoQuick_{view_type}_analysis.mp4",
        )

    finally:
        for path in [input_path, raw_output]:
            if os.path.exists(path):
                os.remove(path)
