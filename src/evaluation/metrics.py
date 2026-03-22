"""
metrics.py
----------
Evaluation metrics for object detection:
  - Precision, Recall, F1-score
  - mAP (Mean Average Precision) at IoU=0.5
  - Confusion matrix generation

Usage:
    from src.evaluation.metrics import compute_map, compute_confusion_matrix
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


CLASSES = ["tires", "water_tanks", "bottles", "buckets", "pools", "water_tubes"]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate average precision using 11-point interpolation."""
    ap = 0.0
    for thresh in np.arange(0, 1.1, 0.1):
        prec_at_rec = precisions[recalls >= thresh]
        ap += max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0
    return ap / 11.0


def compute_map(
    gt_boxes: Dict[str, List],
    pred_boxes: Dict[str, List],
    num_classes: int = len(CLASSES),
    iou_threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute mAP across all classes.

    Args:
        gt_boxes: {image_id: [{"class": int, "bbox": [x1,y1,x2,y2]}]}
        pred_boxes: {image_id: [{"class": int, "bbox": [x1,y1,x2,y2], "score": float}]}
        num_classes: Number of object classes.
        iou_threshold: IoU threshold for true positive.

    Returns:
        mAP (float), per-class AP dict
    """
    aps = {}

    for cls_idx in range(num_classes):
        cls_name = CLASSES[cls_idx]
        # Collect all predictions for this class, sorted by confidence
        all_preds = []
        for img_id, preds in pred_boxes.items():
            for pred in preds:
                if pred["class"] == cls_idx:
                    all_preds.append({
                        "image_id": img_id,
                        "bbox": pred["bbox"],
                        "score": pred["score"]
                    })

        all_preds.sort(key=lambda x: x["score"], reverse=True)

        # Count ground truth boxes per image for this class
        gt_count = defaultdict(int)
        matched = defaultdict(list)
        for img_id, gts in gt_boxes.items():
            for gt in gts:
                if gt["class"] == cls_idx:
                    gt_count[img_id] += 1

        total_gt = sum(gt_count.values())
        if total_gt == 0:
            aps[cls_name] = 0.0
            continue

        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        for i, pred in enumerate(all_preds):
            img_id = pred["image_id"]
            gt_list = [g for g in gt_boxes.get(img_id, []) if g["class"] == cls_idx]

            best_iou = 0
            best_j = -1
            for j, gt in enumerate(gt_list):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j not in matched[img_id]:
                tp[i] = 1
                matched[img_id].append(best_j)
            else:
                fp[i] = 1

        cumtp = np.cumsum(tp)
        cumfp = np.cumsum(fp)
        recalls = cumtp / total_gt
        precisions = cumtp / (cumtp + cumfp + 1e-8)
        aps[cls_name] = compute_average_precision(recalls, precisions)

    map_score = np.mean(list(aps.values()))
    return map_score, aps


def compute_confusion_matrix(
    gt_labels: List[int],
    pred_labels: List[int],
    num_classes: int = len(CLASSES)
) -> np.ndarray:
    """
    Compute a confusion matrix.

    Args:
        gt_labels: Ground truth class indices.
        pred_labels: Predicted class indices.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for gt, pred in zip(gt_labels, pred_labels):
        cm[gt][pred] += 1
    return cm


def print_metrics_table(metrics: Dict[str, Dict[str, float]]):
    """Print a formatted metrics table matching Table I in the paper."""
    header = f"{'Model':<15} {'Precision':>12} {'Recall':>10} {'F1':>8} {'mAP':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for model, m in metrics.items():
        print(f"{model:<15} {m['precision']:>12.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>8.3f} {m['map']:>8.3f}")
    print("=" * len(header) + "\n")
