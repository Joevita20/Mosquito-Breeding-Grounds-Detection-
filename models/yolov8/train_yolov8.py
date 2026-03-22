"""
train_yolov8.py
---------------
YOLOv8 training script for the MBG dataset, using SGD optimizer 
as described in the research paper.

Usage (local):
    python3 models/yolov8/train_yolov8.py

Usage (Colab - recommended):
    !python models/yolov8/train_yolov8.py --data /content/data.yaml --epochs 100

Requirements:
    pip install ultralytics
"""

import argparse
import sys
import os


def train(data_yaml: str, epochs: int, img_size: int, batch_size: int, device: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load YOLOv8 nano (fastest) - change to 's', 'm', 'l', 'x' for larger models
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        optimizer="SGD",      # Paper uses SGD for YOLOv8
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=device,
        project="runs/yolov8",
        name="mbg_train",
        save=True,
        plots=True,
        conf=0.25,
        iou=0.5,
        verbose=True,
    )
    print(f"\n[INFO] Training complete. Results saved to: {results.save_dir}")
    return results


def validate(weights: str, data_yaml: str, device: str):
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, device=device)
    print(f"\n[INFO] mAP50: {metrics.box.map50:.4f}")
    print(f"[INFO] Precision: {metrics.box.mp:.4f}")
    print(f"[INFO] Recall: {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the MBG dataset.")
    parser.add_argument("--data", type=str, default="data/data.yaml",
                        help="Path to YOLO data.yaml file.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for training.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to use: '0' for GPU, 'cpu' for CPU.")
    parser.add_argument("--validate_only", action="store_true",
                        help="Skip training and only run validation.")
    parser.add_argument("--weights", type=str, default="runs/yolov8/mbg_train/weights/best.pt",
                        help="Path to weights for validation.")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.weights, args.data, args.device)
    else:
        train(args.data, args.epochs, args.img_size, args.batch, args.device)
