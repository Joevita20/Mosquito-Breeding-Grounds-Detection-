"""
augment_data.py
---------------
Applies augmentations identical to those described in the research paper:
  - Resize to 640x640
  - Auto-orientation (EXIF correction)
  - Horizontal/Vertical flipping
  - Grayscale (4% of images)
  - Saturation jitter (-25% to +25%)

Usage:
    python3 src/preprocessing/augment_data.py \
        --input_dir data/frames \
        --output_dir data/augmented \
        --n_augments 5
"""

import cv2
import os
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps


IMAGE_SIZE = (640, 640)
GRAYSCALE_PROB = 0.04
SATURATION_RANGE = (-0.25, 0.25)


def auto_orient(img: Image.Image) -> Image.Image:
    """Correct image orientation using EXIF data."""
    return ImageOps.exif_transpose(img)


def resize(img: Image.Image, size=IMAGE_SIZE) -> Image.Image:
    """Resize to target size."""
    return img.resize(size, Image.BILINEAR)


def flip_horizontal(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def flip_vertical(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def apply_grayscale(img: Image.Image) -> Image.Image:
    """Convert to grayscale but keep 3 channels."""
    gray = ImageOps.grayscale(img)
    return Image.merge("RGB", [gray, gray, gray])


def apply_saturation(img: Image.Image, factor_range=SATURATION_RANGE) -> Image.Image:
    """Jitter image saturation within [-25%, +25%]."""
    img_cv = np.array(img)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = 1.0 + random.uniform(*factor_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    img_cv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_cv)


def augment_image(img_path: str, output_dir: str, n: int = 5):
    """
    Applies n random augmentations to a single image and saves results.
    """
    img = Image.open(img_path).convert("RGB")
    img = auto_orient(img)
    img = resize(img)
    stem = Path(img_path).stem

    for i in range(n):
        augmented = img.copy()

        # Always apply a random flip
        flip_choice = random.choice(["horizontal", "vertical", "both", "none"])
        if flip_choice in ("horizontal", "both"):
            augmented = flip_horizontal(augmented)
        if flip_choice in ("vertical", "both"):
            augmented = flip_vertical(augmented)

        # 4% chance of grayscale
        if random.random() < GRAYSCALE_PROB:
            augmented = apply_grayscale(augmented)
        else:
            augmented = apply_saturation(augmented)

        out_path = os.path.join(output_dir, f"{stem}_aug{i:03d}.jpg")
        augmented.save(out_path)


def process_directory(input_dir: str, output_dir: str, n_augments: int = 5):
    os.makedirs(output_dir, exist_ok=True)
    images = list(Path(input_dir).rglob("*.jpg")) + \
             list(Path(input_dir).rglob("*.png")) + \
             list(Path(input_dir).rglob("*.jpeg"))

    print(f"[INFO] Found {len(images)} images. Augmenting with {n_augments}x...")
    for img_path in images:
        augment_image(str(img_path), output_dir, n_augments)
    print(f"[INFO] Augmentation complete. Output -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment MBG dataset images.")
    parser.add_argument("--input_dir", type=str, default="data/frames")
    parser.add_argument("--output_dir", type=str, default="data/augmented")
    parser.add_argument("--n_augments", type=int, default=5)
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.n_augments)
