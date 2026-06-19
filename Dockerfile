FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps — ubuntu:22.04 has libgles2-mesa which MediaPipe requires
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgles2 \
    libgles2-mesa \
    libegl1 \
    libegl1-mesa \
    libegl-mesa0 \
    libglvnd0 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Verify libGLESv2 is present — fails the build early if the package didn't install
RUN ls /usr/lib/x86_64-linux-gnu/libGLESv2* && echo "libGLESv2 OK"

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Note: YOLO model not pre-downloaded — API uses display_mode="Angles Only" which skips YOLO
COPY . .

EXPOSE 8000

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300
