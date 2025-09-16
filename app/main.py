import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from models import DetectionResponse, ErrorResponse
from detector import LogoDetector
from utils import pil_image_from_bytes, SUPPORTED_MIMES
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("/logs/app.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="T-Bank Logo Detector", version="1.0")

logger.info("Инициализация LogoDetector...")
detector = LogoDetector(weights_path="weights/best.pt", conf=0.25)
logger.info("LogoDetector успешно инициализирован")

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
    logger.info("Получен запрос на детекцию. Количество файлов: %d", len(files))

    for file in files:
        logger.info("Обработка файла: %s (тип: %s)", file.filename, file.content_type)

        if file.content_type not in SUPPORTED_MIMES:
            logger.warning("Неподдерживаемый формат файла: %s", file.content_type)
            raise HTTPException(status_code=400, detail=f"Unsupported format: {file.content_type}")

        data = await file.read()
        try:
            img = pil_image_from_bytes(data)
        except Exception as e:
            logger.error("Ошибка обработки изображения %s: %s", file.filename, str(e))
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

        detections = detector.detect(img)
        logger.info("Файл %s: найдено %d объектов", file.filename, len(detections))
        detections_all.extend(detections)

    logger.info("Запрос успешно обработан. Всего найдено объектов: %d", len(detections_all))
    return DetectionResponse(detections=detections_all)
