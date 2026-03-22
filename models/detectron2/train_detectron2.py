"""
train_detectron2.py
-------------------
Detectron2 (Faster R-CNN with FPN backbone) training script for the MBG dataset.
Optimizers: Adagrad and Adam as described in the research paper.

Usage (Colab - recommended):
    python3 models/detectron2/train_detectron2.py \
        --data_dir data/detectron2_format \
        --optimizer adam \
        --epochs 60

Requirements:
    pip install torch torchvision
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
"""

import argparse
import os
import json
import logging
import sys


CLASSES = ["tires", "water_tanks", "bottles", "buckets", "pools", "water_tubes"]
NUM_CLASSES = len(CLASSES)


def setup_detectron2():
    """Import detectron2 and verify installation."""
    try:
        import detectron2
        from detectron2.utils.logger import setup_logger
        setup_logger()
        return True
    except ImportError:
        print("[ERROR] Detectron2 not installed.")
        print("  Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        return False


def register_mbg_dataset(data_dir: str):
    """Register MBG dataset (COCO format) with Detectron2."""
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import DatasetCatalog, MetadataCatalog

    for split in ["train", "val"]:
        name = f"mbg_{split}"
        json_file = os.path.join(data_dir, split, "annotations.json")
        image_dir = os.path.join(data_dir, split, "images")

        if name not in DatasetCatalog.list():
            register_coco_instances(name, {}, json_file, image_dir)

        MetadataCatalog.get(name).set(thing_classes=CLASSES)
        print(f"[INFO] Registered dataset: {name}")


def get_config(optimizer: str, epochs: int, device: str):
    """Build a Detectron2 config for Mask R-CNN / Faster R-CNN with FPN."""
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    cfg.DATASETS.TRAIN = ("mbg_train",)
    cfg.DATASETS.TEST = ("mbg_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.MAX_ITER = epochs * 100  # approximate iterations
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.STEPS = (int(epochs * 100 * 0.6), int(epochs * 100 * 0.9))

    # Optimizer selection as per paper
    if optimizer.lower() == "adam":
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.OPTIMIZER = "ADAM"
    elif optimizer.lower() == "adagrad":
        cfg.SOLVER.BASE_LR = 0.001
        # Adagrad not natively in D2; use custom or fall back to SGD with scheduler
        cfg.SOLVER.OPTIMIZER = "ADAGRAD" if hasattr(cfg.SOLVER, "OPTIMIZER") else "SGD"
    else:  # SGD
        cfg.SOLVER.BASE_LR = 0.02
        cfg.SOLVER.MOMENTUM = 0.9

    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.TEST.EVAL_PERIOD = 500
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = f"runs/detectron2/mbg_{optimizer}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def train(data_dir: str, optimizer: str, epochs: int, device: str):
    if not setup_detectron2():
        sys.exit(1)

    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator

    register_mbg_dataset(data_dir)
    cfg = get_config(optimizer, epochs, device)

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print(f"[INFO] Detectron2 training complete. Weights -> {cfg.OUTPUT_DIR}")


def validate(data_dir: str, weights: str, optimizer: str, device: str):
    if not setup_detectron2():
        sys.exit(1)

    from detectron2.engine import DefaultPredictor
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    register_mbg_dataset(data_dir)
    cfg = get_config(optimizer, 1, device)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    evaluator = COCOEvaluator("mbg_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "mbg_val")
    predictor = DefaultPredictor(cfg)

    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(f"\n[INFO] Validation Results:\n{json.dumps(results, indent=2)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detectron2 on the MBG dataset.")
    parser.add_argument("--data_dir", type=str, default="data/detectron2_format",
                        help="Root directory with train/val subdirectories in COCO format.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adagrad", "sgd"],
                        default="adam", help="Optimizer (paper uses Adagrad and Adam).")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Approximate training epochs.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--weights", type=str, default="runs/detectron2/mbg_adam/model_final.pth")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.data_dir, args.weights, args.optimizer, args.device)
    else:
        train(args.data_dir, args.optimizer, args.epochs, args.device)
