# Video to 3D: local app (CPU steps) that rents a RunPod GPU for the training step.
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libegl1 libgomp1 libglib2.0-0 libx11-6 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask==3.1.*
COPY . .
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8080
EXPOSE 8080
VOLUME /data
CMD ["python", "app/server.py"]
