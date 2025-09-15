FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgthread-2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

COPY weights /app/weights

RUN useradd -m appuser

ENV YOLO_CONFIG_DIR=/app/ultralytics

RUN mkdir -p /app/ultralytics /app/results && chown -R appuser:appuser /app/ultralytics /app/results

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
