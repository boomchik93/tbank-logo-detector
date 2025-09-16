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

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --default-timeout=120 --no-cache-dir \
        torch>=2.2.0 torchvision>=0.15.2 \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --default-timeout=120 --no-cache-dir -r requirements.txt

COPY app/ .  

RUN useradd -m appuser \
    && mkdir -p /logs \
    && chown -R appuser:appuser /logs


ENV YOLO_CONFIG_DIR=/app/ultralytics

RUN mkdir -p /app/ultralytics /app/results && chown -R appuser:appuser /app/ultralytics /app/results

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
