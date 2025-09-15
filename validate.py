import argparse
from ultralytics import YOLO

def validate(weights: str, data: str):
    model = YOLO(weights)
    print(f"Запуск валидации: weights={weights}, data={data}")
    results = model.val(data=data, save_json=True)
    print("Результаты валидации (summaries):\n", results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/best.pt', help='Путь к весам')
    parser.add_argument('--data', required=True, help='Путь к data.yaml (YOLO формат)')
    args = parser.parse_args()
    validate(args.weights, args.data)
