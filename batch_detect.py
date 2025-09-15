import argparse
from pathlib import Path
from app.detector import LogoDetector
from PIL import Image
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)


def bbox_to_xyxy(bbox):
    """
    Приводим bbox к формату [xmin, ymin, xmax, ymax].
    Универсально для разных структур BoundingBox.
    """
    if hasattr(bbox, "dict"):
        data = bbox.dict()
    elif hasattr(bbox, "__dict__"):
        data = bbox.__dict__
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [bbox[0], bbox[1], bbox[2], bbox[3]]
    else:
        raise ValueError(f"Неизвестный формат bbox: {bbox}")

    if all(k in data for k in ["xmin", "ymin", "xmax", "ymax"]):
        return [data["xmin"], data["ymin"], data["xmax"], data["ymax"]]

    if all(k in data for k in ["x_min", "y_min", "x_max", "y_max"]):
        return [data["x_min"], data["y_min"], data["x_max"], data["y_max"]]

    if all(k in data for k in ["left", "top", "width", "height"]):
        return [data["left"], data["top"], data["left"] + data["width"], data["top"] + data["height"]]

    if all(k in data for k in ["x", "y", "width", "height"]):
        return [data["x"], data["y"], data["x"] + data["width"], data["y"] + data["height"]]

    raise ValueError(f"Неизвестные поля в bbox: {data}")


def get_score(detection):
    if hasattr(detection, "confidence"):
        return detection.confidence
    elif hasattr(detection, "score"):
        return detection.score
    elif hasattr(detection, "conf"):
        return detection.conf
    else:
        return 1.0



def load_yolo_labels(label_path: Path, img_size):
    """
    Загружает аннотации YOLO формата (x_center, y_center, w, h в относительных координатах).
    Возвращает список боксов в формате [xmin, ymin, xmax, ymax].
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x, y, w, h = map(float, parts)
            W, H = img_size
            xmin = (x - w / 2) * W
            ymin = (y - h / 2) * H
            xmax = (x + w / 2) * W
            ymax = (y + h / 2) * H
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def validate_directory(input_dir: Path, labels_dir: Path, weights: str = "weights/best.pt", conf: float = 0.25):
    input_dir = Path(input_dir)
    labels_dir = Path(labels_dir)

    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    if not files:
        print("Нет изображений для валидации в папке:", input_dir)
        return
    print(f"Найдено {len(files)} изображений для валидации.")

    detector = LogoDetector(weights_path=weights, conf=conf, device=device)
    metric = MeanAveragePrecision().to(device)

    total = 0
    for p in files:
        total += 1
        img = Image.open(p).convert("RGB")

        dets = detector.detect(img)
        if dets:
            boxes = torch.tensor([bbox_to_xyxy(d.bbox) for d in dets], device=device, dtype=torch.float)
            scores = torch.tensor([get_score(d) for d in dets], device=device, dtype=torch.float)
            labels = torch.zeros(len(dets), device=device, dtype=torch.long)
        else:
            boxes = torch.empty((0, 4), device=device, dtype=torch.float)
            scores = torch.empty((0,), device=device, dtype=torch.float)
            labels = torch.empty((0,), device=device, dtype=torch.long)

        preds = {"boxes": boxes, "scores": scores, "labels": labels}


        labels_file = labels_dir / (p.stem + ".txt")
        gts = load_yolo_labels(labels_file, img.size)
        if gts:
            target_boxes = torch.tensor(gts, device=device, dtype=torch.float)
            target_labels = torch.zeros(len(gts), device=device, dtype=torch.long)
        else:
            target_boxes = torch.empty((0, 4), device=device, dtype=torch.float)
            target_labels = torch.empty((0,), device=device, dtype=torch.long)

        target = {"boxes": target_boxes, "labels": target_labels}
        metric.update([preds], [target])

    results = metric.compute()
    print("=== Результаты валидации ===")
    print(f"Обработано изображений: {total}")
    print(f"mAP@0.5: {results['map_50']:.3f}")
    print(f"mAP@[0.5:0.95]: {results['map']:.3f}")
    print(f"mAP@0.75: {results['map_75']:.3f}")
    precision = results['map_50']
    recall = results['mar_100'] if 'mar_100' in results else results['map_50']

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    print(f"F1-score @ IoU=0.5: {f1:.3f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Путь к директории с изображениями')
    parser.add_argument('--labels', required=True, help='Путь к YOLO-аннотациям (.txt)')
    parser.add_argument('--weights', default='weights/best.pt', help='Путь к весам модели')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    validate_directory(args.input, args.labels, args.weights, args.conf)
