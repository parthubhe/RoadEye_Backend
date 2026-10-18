"""Fast unit checks for target conversion; never loads model weights."""

from __future__ import annotations

import unittest

import numpy as np

from .prepare_teacher_dataset import iou, teacher_to_yolo, xywh_to_xyxy


class DistillationGeometryTests(unittest.TestCase):
    def test_box_round_trip(self):
        converted = teacher_to_yolo(np.array([10, 20, 30, 60]), 100, 100, 2)
        self.assertIsNotNone(converted)
        line, corners = converted
        self.assertEqual(corners, (10, 20, 30, 60))
        self.assertEqual(line.split()[0], "2")
        for actual, expected in zip(xywh_to_xyxy(line.split(), 100, 100), corners):
            self.assertAlmostEqual(actual, expected)

    def test_iou_overlap_filter(self):
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertGreater(iou((0, 0, 10, 10), (5, 5, 15, 15)), 0)

    def test_outside_and_small_boxes(self):
        self.assertIsNone(teacher_to_yolo(np.array([2, 2, 2.1, 2.1]), 100, 100, 0))
        line, corners = teacher_to_yolo(np.array([-5, -10, 20, 20]), 100, 100, 0)
        self.assertEqual(corners, (0.0, 0.0, 20.0, 20.0))
        self.assertEqual(line.split()[0], "0")


if __name__ == "__main__":
    unittest.main()
