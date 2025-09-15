"""Пакетная обработка директории с изображениями.
Для каждого изображения:
  - если модель находит хотя бы один логотип -> копируем файл в output_dir
  - если нет — игнорируем
По завершении печатаем отчет.
"""
import argparse
from pathlib import Path
from app.detector import LogoDetector
from PIL import Image
import shutil
import time

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def process_directory(input_dir: Path, output_dir: Path, weights: str = "weights/best.pt", conf: float = 0.25):
    detector = LogoDetector(weights_path=weights, conf=conf)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    detected = 0
    times = []

    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXT:
            continue
        total += 1
        start = time.time()
        try:
            img = Image.open(p).convert("RGB")
            dets = detector.detect(img)
        except Exception as e:
            print(f"Ошибка при обработке {p.name}: {e}")
            continue
        elapsed = time.time() - start
        times.append(elapsed)
        if dets:
            shutil.copy2(p, output_dir / p.name)
            detected += 1
    avg = sum(times)/len(times) if times else 0
    print("=== Отчет по обработке ===")
    print(f"Папка: {input_dir}")
    print(f"Всего файлов обработано: {total}")
    print(f"Файлов с детекцией: {detected}")
    print(f"Файлов без детекции: {total - detected}")
    print(f"Среднее время обработки изображения: {avg:.3f} сек")
    print(f"Максимальное время обработки (сек): {max(times) if times else 0:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Путь к директории с изображениями')
    parser.add_argument('--output', required=False, default='filtered', help='Путь к директории куда копировать удовлетворяющие файлы')
    parser.add_argument('--weights', required=False, default='weights/best.pt', help='Путь к весам модели (по умолчанию weights/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    process_directory(args.input, args.output, args.weights, args.conf)
