"""
train_detr.py
-------------
DETR (Detection Transformer) training script for the MBG dataset,
as used in the research paper.

Reference: https://github.com/facebookresearch/detr

Usage (Colab - recommended):
    python3 models/detr/train_detr.py \
        --data_dir data/detr_format \
        --epochs 140 \
        --device cuda

Requirements:
    pip install torch torchvision
    # DETR uses timm and scipy:
    pip install timm scipy
"""

import argparse
import os
import sys
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


CLASSES = ["tires", "water_tanks", "bottles", "buckets", "pools", "water_tubes"]
NUM_CLASSES = len(CLASSES)


# ───────────────────────────────────────────────────────
# Minimal DETR Dataset wrapper for COCO-format annotations
# ───────────────────────────────────────────────────────

class MBGDETRDataset(Dataset):
    """
    Dataset class that wraps COCO-format annotations for DETR.
    Each item returns:
      - image: Tensor [3, H, W] (normalized)
      - target: dict with "boxes" [N,4] (cx, cy, w, h normalised) and "labels" [N]
    """

    def __init__(self, image_dir: str, annotation_file: str, img_size: int = 640):
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.image_dir = image_dir
        self.img_size = img_size
        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

        self.ids = list(self.images.keys())
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        image = self.transform(image)

        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO: x,y,w,h (top-left)
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            boxes.append([cx, cy, nw, nh])
            labels.append(ann["category_id"])  # 0-indexed class id

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


# ───────────────────────────────────────────────────────
# Simplified DETR wrapper using the official repo
# ───────────────────────────────────────────────────────

def clone_detr(target_dir: str = "detr_repo"):
    """Clone the official DETR repo."""
    if not os.path.exists(target_dir):
        os.system(f"git clone https://github.com/facebookresearch/detr {target_dir}")
    sys.path.insert(0, target_dir)


def build_detr_model(num_classes: int = NUM_CLASSES, device: str = "cuda"):
    """
    Build a DETR model:
    - Backbone: ResNet-50 (pretrained on ImageNet)
    - Transformer encoder-decoder
    - Set prediction (bipartite matching)
    """
    try:
        clone_detr()
        from models import build_model
        import argparse as ap

        # Minimal args for DETR
        args = ap.Namespace(
            lr=1e-4,
            lr_backbone=1e-5,
            batch_size=2,
            weight_decay=1e-4,
            epochs=300,
            lr_drop=200,
            clip_max_norm=0.1,
            frozen_weights=None,
            backbone="resnet50",
            dilation=False,
            position_embedding="sine",
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_queries=100,
            pre_norm=False,
            masks=False,
            aux_loss=True,
            set_cost_class=1,
            set_cost_bbox=5,
            set_cost_giou=2,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            eos_coef=0.1,
            dataset_file="coco",
            coco_path=None,
            coco_panoptic_path=None,
            remove_difficult=False,
            output_dir="runs/detr",
            device=device,
            seed=42,
            resume="",
            start_epoch=0,
            eval=False,
            num_workers=2,
            num_classes=num_classes,
        )
        model, criterion, postprocessors = build_model(args)
        print(f"[INFO] DETR model built successfully with {num_classes} classes.")
        return model, criterion, postprocessors
    except Exception as e:
        print(f"[WARN] Could not build DETR from official repo: {e}")
        print("[INFO] Using a minimal fallback DETR-like structure for testing.")
        return None, None, None


def train_detr(data_dir: str, epochs: int, device: str):
    """Full DETR training loop using the official DETR repository."""
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training DETR on device: {device_obj}")

    # Build datasets
    train_dataset = MBGDETRDataset(
        image_dir=os.path.join(data_dir, "train", "images"),
        annotation_file=os.path.join(data_dir, "train", "annotations.json")
    )
    val_dataset = MBGDETRDataset(
        image_dir=os.path.join(data_dir, "val", "images"),
        annotation_file=os.path.join(data_dir, "val", "annotations.json")
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    # Build model
    model, criterion, _ = build_detr_model(NUM_CLASSES, device)
    if model is None:
        print("[ERROR] Model could not be built. Please run on Colab with DETR repo cloned.")
        return

    model = model.to(device_obj)

    # Optimizer: AdamW (standard for Transformers)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200)

    best_loss = float("inf")
    os.makedirs("runs/detr", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        criterion.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device_obj)
            for t in targets:
                for k, v in t.items():
                    t[k] = v.to(device_obj)

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += losses.item()

        lr_scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "runs/detr/best.pt")

    print(f"[INFO] DETR training complete. Best weights -> runs/detr/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR on the MBG dataset.")
    parser.add_argument("--data_dir", type=str, default="data/detr_format")
    parser.add_argument("--epochs", type=int, default=140,
                        help="Training epochs (paper uses 140 mins for DETR).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train_detr(args.data_dir, args.epochs, args.device)
