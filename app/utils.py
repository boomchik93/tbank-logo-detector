import logging
from typing import Tuple
from PIL import Image
import io

logger = logging.getLogger(__name__)

SUPPORTED_MIMES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp"
}

def pil_image_from_bytes(data: bytes) -> Image.Image:
    """
    Конвертация массива байт в PIL.Image.
    Поддерживаемые форматы: JPEG, PNG, BMP, WEBP
    """
    logger.debug("Попытка открыть изображение из байтов (размер: %d байт)", len(data))
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        logger.info("Изображение успешно загружено (%s, %s)", img.format, img.size)
        return img
    except Exception as e:
        logger.exception("Ошибка при открытии изображения: %s", str(e))
        raise
