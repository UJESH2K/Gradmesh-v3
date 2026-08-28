import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic YOLO OBB training runner")
    parser.add_argument("--data", required=True, help="Path to data YAML")
    parser.add_argument("--model", default="yolov8n-obb.pt", help="Model checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device id, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs_obb")
    parser.add_argument("--name", default="exp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
