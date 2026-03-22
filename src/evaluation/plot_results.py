"""
plot_results.py
---------------
Visualizations for model evaluation:
  - Confusion matrix heatmap
  - mAP bar chart comparison
  - Detection overlay on images

Usage:
    python3 src/evaluation/plot_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os


CLASSES = ["tires", "water_tanks", "bottles", "buckets", "pools", "water_tubes"]


def plot_confusion_matrix(cm: np.ndarray, class_names=CLASSES, model_name: str = "YOLOv8",
                          output_path: str = "results/confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved confusion matrix -> {output_path}")


def plot_map_comparison(results: dict, output_path: str = "results/map_comparison.png"):
    """
    Plot a bar chart comparing mAP of all models.

    Args:
        results: {"ModelName": {"map": 0.92, "precision": 0.94, ...}, ...}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(results.keys())
    maps = [results[m]["map"] for m in models]
    precisions = [results[m]["precision"] for m in models]
    recalls = [results[m]["recall"] for m in models]
    f1s = [results[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, maps, width, label="mAP", color="#2196F3")
    ax.bar(x - 0.5 * width, precisions, width, label="Precision", color="#4CAF50")
    ax.bar(x + 0.5 * width, recalls, width, label="Recall", color="#FF9800")
    ax.bar(x + 1.5 * width, f1s, width, label="F1-Score", color="#9C27B0")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison – MBG Dataset", fontsize=14)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved comparison chart -> {output_path}")


def plot_detections(image_path: str, detections: list, class_names=CLASSES,
                    output_path: str = "results/detection.png"):
    """
    Draw bounding box predictions on an image.

    Args:
        image_path: Path to the source image.
        detections: [{"class": int, "bbox": [x1,y1,x2,y2], "score": float}, ...]
        class_names: List of class label strings.
        output_path: Path to save the annotated image.
    """
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        score = det["score"]
        color = colors[cls][:3]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{class_names[cls]}: {score:.2f}",
                color="white", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))

    ax.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved detection image -> {output_path}")


# Example: Reproduce Table I from the paper
PAPER_RESULTS = {
    "YOLOv8":       {"precision": 0.94, "recall": 0.86, "f1": 0.90, "map": 0.92},
    "YOLOv5":       {"precision": 0.88, "recall": 0.74, "f1": 0.80, "map": 0.82},
    "Detectron2":   {"precision": 0.85, "recall": 0.80, "f1": 0.82, "map": 0.85},
    "DETR":         {"precision": 0.80, "recall": 0.75, "f1": 0.77, "map": 0.82},
    "Faster-RCNN":  {"precision": 0.67, "recall": 0.90, "f1": 0.77, "map": 0.71},
}

if __name__ == "__main__":
    plot_map_comparison(PAPER_RESULTS, output_path="results/map_comparison.png")
    print("[INFO] Run with your actual results dict to compare against paper baselines.")
