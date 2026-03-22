"""
train_yolov5.py
---------------
YOLOv5 training script for the MBG dataset, using SGD optimizer 
as described in the research paper.

Usage (Colab - recommended):
    python train.py \
        --data /content/data.yaml \
        --cfg yolov5s.yaml \
        --weights yolov5s.pt \
        --optimizer SGD \
        --epochs 100 \
        --batch-size 16 \
        --img 640

Requirements:
    git clone https://github.com/ultralytics/yolov5
    pip install -r yolov5/requirements.txt
"""

import subprocess
import argparse
import sys
import os


def clone_yolov5(target_dir: str = "yolov5"):
    """Clone the YOLOv5 repo if not already present."""
    if not os.path.exists(target_dir):
        print("[INFO] Cloning YOLOv5 repository...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5", target_dir],
                       check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        os.path.join(target_dir, "requirements.txt")], check=True)
        print("[INFO] YOLOv5 repository ready.")
    else:
        print("[INFO] YOLOv5 already exists.")


def train_yolov5(data_yaml: str, epochs: int, img_size: int,
                 batch_size: int, device: str, yolov5_dir: str = "yolov5"):
    """
    Launch YOLOv5 training using its own train.py script.
    Optimizer: SGD as specified in the paper.
    """
    cmd = [
        sys.executable,
        os.path.join(yolov5_dir, "train.py"),
        "--data", data_yaml,
        "--weights", "yolov5s.pt",  # start from small pretrained
        "--cfg", os.path.join(yolov5_dir, "models", "yolov5s.yaml"),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--imgsz", str(img_size),
        "--optimizer", "SGD",       # Paper specifies SGD for YOLOv5
        "--device", device,
        "--project", "runs/yolov5",
        "--name", "mbg_train",
        "--save-period", "10",
    ]
    print(f"[INFO] Starting YOLOv5 training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("[INFO] YOLOv5 training complete.")


def validate_yolov5(weights: str, data_yaml: str, device: str, yolov5_dir: str = "yolov5"):
    """Run YOLOv5 validation."""
    cmd = [
        sys.executable,
        os.path.join(yolov5_dir, "val.py"),
        "--weights", weights,
        "--data", data_yaml,
        "--device", device,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv5 on the MBG dataset.")
    parser.add_argument("--data", type=str, default="data/data.yaml",
                        help="Path to YOLO data.yaml file.")
    parser.add_argument("--epochs", type=int, default=91,
                        help="Number of training epochs (paper: ~91 mins training time).")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--yolov5_dir", type=str, default="yolov5")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--weights", type=str, default="runs/yolov5/mbg_train/weights/best.pt")
    args = parser.parse_args()

    clone_yolov5(args.yolov5_dir)

    if args.validate_only:
        validate_yolov5(args.weights, args.data, args.device, args.yolov5_dir)
    else:
        train_yolov5(args.data, args.epochs, args.img_size, args.batch,
                     args.device, args.yolov5_dir)
