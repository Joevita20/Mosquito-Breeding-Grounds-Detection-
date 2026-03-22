"""
test_preprocessing.py
---------------------
Unit tests for data preprocessing utilities.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.augment_data import (
    flip_horizontal, flip_vertical, apply_grayscale, apply_saturation, resize
)
from src.evaluation.metrics import compute_iou, compute_map


class TestAugmentations(unittest.TestCase):

    def setUp(self):
        """Create a small dummy RGB image."""
        self.img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )

    def test_resize(self):
        resized = resize(self.img, (640, 640))
        self.assertEqual(resized.size, (640, 640))

    def test_flip_horizontal(self):
        flipped = flip_horizontal(self.img)
        self.assertEqual(flipped.size, self.img.size)

    def test_flip_vertical(self):
        flipped = flip_vertical(self.img)
        self.assertEqual(flipped.size, self.img.size)

    def test_grayscale(self):
        gray = apply_grayscale(self.img)
        arr = np.array(gray)
        # All three channels should be equal
        self.assertTrue(np.allclose(arr[:, :, 0], arr[:, :, 1]))
        self.assertTrue(np.allclose(arr[:, :, 1], arr[:, :, 2]))

    def test_saturation_preserves_size(self):
        result = apply_saturation(self.img, (-0.1, 0.1))
        self.assertEqual(result.size, self.img.size)


class TestMetrics(unittest.TestCase):

    def test_iou_perfect_overlap(self):
        box = [0, 0, 10, 10]
        self.assertAlmostEqual(compute_iou(box, box), 1.0)

    def test_iou_no_overlap(self):
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        self.assertAlmostEqual(compute_iou(box1, box2), 0.0)

    def test_iou_partial_overlap(self):
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = compute_iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_map_perfect(self):
        gt = {"img1": [{"class": 0, "bbox": [0, 0, 10, 10]}]}
        pred = {"img1": [{"class": 0, "bbox": [0, 0, 10, 10], "score": 0.99}]}
        map_score, aps = compute_map(gt, pred, num_classes=1)
        self.assertGreater(map_score, 0.5)

    def test_map_no_prediction(self):
        gt = {"img1": [{"class": 0, "bbox": [0, 0, 10, 10]}]}
        pred = {}
        map_score, aps = compute_map(gt, pred, num_classes=1)
        self.assertEqual(map_score, 0.0)


if __name__ == "__main__":
    unittest.main()
