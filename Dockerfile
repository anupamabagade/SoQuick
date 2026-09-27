FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for MediaPipe (OpenGL/GLES) and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgles2 \
    libegl1 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLO nano (yolov8n-pose.pt) is auto-downloaded by ultralytics on first request (~6MB)
COPY . .

EXPOSE 8000

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300
