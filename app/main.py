from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from models import DetectionResponse, ErrorResponse
from detector import LogoDetector
from utils import pil_image_from_bytes, SUPPORTED_MIMES
from typing import List


app = FastAPI(title="T-Bank Logo Detector", version="1.0")

# Инициализация детектора.
# Путь до весов: weights/best.pt
detector = LogoDetector(weights_path="weights/best.pt", conf=0.25)

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(files: List[UploadFile] = File(...)):
    """
    Детекция логотипа Т-банка на изображении или в папке с изображениями.
    Args:
        files: Один или несколько файлов (JPEG, PNG, BMP, WEBP)
    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    detections_all = []

    for file in files:
        if file.content_type not in SUPPORTED_MIMES:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {file.content_type}")

        data = await file.read()
        try:
            img = pil_image_from_bytes(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

        detections = detector.detect(img)
        detections_all.extend(detections)

    return DetectionResponse(detections=detections_all)
