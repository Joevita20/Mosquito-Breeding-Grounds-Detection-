"""
convert_to_yolo_format.py
-------------------------
Converts COCO-format annotations (used by MBG dataset) to YOLO format.
YOLO label format: class_id cx cy w h (all normalized 0-1)

Usage:
    python3 src/preprocessing/convert_to_yolo_format.py \
        --annotations data/raw/annotations.json \
        --images_dir data/frames \
        --output_dir data/yolo_format
"""

import json
import os
import argparse
import shutil
from pathlib import Path


CLASSES = ["tires", "water_tanks", "bottles", "buckets", "pools", "water_tubes"]


def coco_to_yolo(
    annotations_file: str,
    images_dir: str,
    output_dir: str,
    split: str = "train",
    val_fraction: float = 0.15,
    test_fraction: float = 0.10
):
    """
    Convert COCO annotations to YOLO format and split into train/val/test.
    """
    with open(annotations_file, "r") as f:
        coco = json.load(f)

    # Build index
    image_info = {img["id"]: img for img in coco["images"]}
    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    category_map = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    # Split image IDs
    all_ids = list(image_info.keys())
    n = len(all_ids)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test

    splits = {
        "train": all_ids[:n_train],
        "val": all_ids[n_train:n_train + n_val],
        "test": all_ids[n_train + n_val:],
    }

    for split_name, ids in splits.items():
        images_out = Path(output_dir) / split_name / "images"
        labels_out = Path(output_dir) / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img = image_info[img_id]
            img_file = img["file_name"]
            W, H = img["width"], img["height"]

            # Copy image
            src = Path(images_dir) / img_file
            dst = images_out / Path(img_file).name
            if src.exists():
                shutil.copy2(str(src), str(dst))

            # Write label file
            label_file = labels_out / (Path(img_file).stem + ".txt")
            with open(label_file, "w") as lf:
                for ann in annotations_by_image.get(img_id, []):
                    cls_id = category_map.get(ann["category_id"], 0)
                    x, y, w, h = ann["bbox"]
                    cx = (x + w / 2) / W
                    cy = (y + h / 2) / H
                    nw = w / W
                    nh = h / H
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        print(f"[INFO] {split_name}: {len(ids)} images -> {images_out}")

    print(f"\n[INFO] All splits written to {output_dir}")
    print("Update your data/data.yaml paths accordingly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format.")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to COCO annotations JSON file.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing raw images.")
    parser.add_argument("--output_dir", type=str, default="data/yolo_format")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    args = parser.parse_args()

    coco_to_yolo(args.annotations, args.images_dir, args.output_dir,
                 val_fraction=args.val_fraction, test_fraction=args.test_fraction)
