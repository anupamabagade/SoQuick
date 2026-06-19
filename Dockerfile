FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the YOLO pose model so it's baked into the image
# (avoids a 133 MB download on every cold start)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-pose.pt')"

COPY . .

EXPOSE 8000

# Increase keep-alive timeout to handle long analysis requests (2–3 min)
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300
