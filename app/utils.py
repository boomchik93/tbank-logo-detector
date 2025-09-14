from typing import Tuple
from PIL import Image
import io

SUPPORTED_MIMES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp"
}

def pil_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")
