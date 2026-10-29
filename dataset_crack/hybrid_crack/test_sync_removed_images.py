"""Focused tests for reconciling manually removed generated images."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from .data import HybridDataset, read_excluded_training_images, read_reviewed_masks
from .sync_removed_images import EXCLUDED_MASK_STATUS, EXCLUDED_OVERLAY_STATUS, EXCLUDED_STATUS, sync


class SyncRemovedImagesTest(unittest.TestCase):
    def test_dry_run_apply_and_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subset"
            root.mkdir()
            for split in ("train", "valid", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / "masks" / split).mkdir(parents=True)
            fields = ["split", "image", "kind", "mask_path", "status"]
            rows = [
                {"split": "train", "image": "bad.jpg", "kind": "positive",
                 "mask_path": "masks/train/bad.png", "status": "auto_pseudo"},
                {"split": "valid", "image": "good.jpg", "kind": "positive",
                 "mask_path": "masks/valid/good.png", "status": "auto_pseudo"},
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            for path in (root / "train/labels/bad.txt", root / "masks/train/bad.png",
                         root / "valid/images/good.jpg", root / "valid/labels/good.txt",
                         root / "masks/valid/good.png"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            train_label = root / "train/labels/bad.txt"
            train_mask = root / "masks/train/bad.png"

            self.assertEqual(sync(root, apply=False), [("train", "bad.jpg")])
            self.assertTrue(train_label.is_file())
            self.assertTrue(train_mask.is_file())
            self.assertEqual(sync(root, apply=True), [("train", "bad.jpg")])
            self.assertFalse(train_label.exists())
            self.assertFalse(train_mask.exists())
            self.assertTrue((root / ".rejected/train/labels/bad.txt").is_file())
            self.assertTrue((root / ".rejected/masks/train/bad.png").is_file())
            self.assertEqual(read_excluded_training_images(manifest), {"bad.jpg"})
            self.assertEqual(set(read_reviewed_masks(manifest, allow_pseudo=True)), {("valid", "good.jpg")})
            full_v5_images = Path(directory) / "full_v5/train/images"
            full_v5_images.mkdir(parents=True)
            (full_v5_images / "bad.jpg").write_bytes(b"fixture")
            (full_v5_images / "good.jpg").write_bytes(b"fixture")
            training = HybridDataset(full_v5_images, "train", {}, exclude_names={"bad.jpg"})
            self.assertEqual([path.name for path in training.paths], ["good.jpg"])
            with manifest.open(newline="", encoding="utf-8") as file:
                saved = list(csv.DictReader(file))
            self.assertEqual(saved[0]["status"], EXCLUDED_STATUS)
            self.assertEqual(saved[1]["status"], "auto_pseudo")
            self.assertEqual(sync(root, apply=True), [])

    def test_missing_mask_removes_generated_image_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subset"
            root.mkdir()
            for split in ("train", "valid", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / "masks" / split).mkdir(parents=True)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["split", "image", "kind", "mask_path", "status"])
                writer.writeheader()
                writer.writerow({"split": "train", "image": "bad_mask.jpg", "kind": "positive",
                                 "mask_path": "masks/train/bad_mask.png", "status": "auto_pseudo"})
            image = root / "train/images/bad_mask.jpg"
            label = root / "train/labels/bad_mask.txt"
            label.parent.mkdir(parents=True)
            image.write_bytes(b"original generated copy")
            label.write_bytes(b"label")
            self.assertEqual(sync(root, apply=False), [("train", "bad_mask.jpg")])
            self.assertTrue(image.exists())
            self.assertEqual(sync(root, apply=True), [("train", "bad_mask.jpg")])
            self.assertFalse(image.exists())
            self.assertFalse(label.exists())
            self.assertTrue((root / ".rejected/train/images/bad_mask.jpg").is_file())
            self.assertTrue((root / ".rejected/train/labels/bad_mask.txt").is_file())
            self.assertEqual(read_excluded_training_images(manifest), {"bad_mask.jpg"})
            with manifest.open(newline="", encoding="utf-8") as file:
                self.assertEqual(next(csv.DictReader(file))["status"], EXCLUDED_MASK_STATUS)
            self.assertEqual(sync(root, apply=True), [])

    def test_missing_full_review_overlay_excludes_row_not_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subset"
            root.mkdir()
            for split in ("train", "valid", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / "masks" / split).mkdir(parents=True)
            rows = [
                {"split": "train", "image": "bad.jpg", "kind": "positive",
                 "mask_path": "masks/train/bad.png", "status": "auto_pseudo"},
                {"split": "valid", "image": "good.jpg", "kind": "positive",
                 "mask_path": "masks/valid/good.png", "status": "auto_pseudo"},
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for path in (root / "train/images/bad.jpg", root / "train/labels/bad.txt",
                         root / "masks/train/bad.png", root / "valid/images/good.jpg",
                         root / "valid/labels/good.txt", root / "masks/valid/good.png"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            full = root / "review_overlays_20261017_120000"
            pilot = root / "review_overlays_20261017_130000"
            full.mkdir()
            pilot.mkdir()
            review_rows = [
                {"split": "train", "image": "bad.jpg", "overlay": "train/bad_overlay.jpg"},
                {"split": "valid", "image": "good.jpg", "overlay": "valid/good_overlay.jpg"},
            ]
            for folder, items in ((full, review_rows), (pilot, review_rows[:1])):
                with (folder / "review_index.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(review_rows[0]))
                    writer.writeheader()
                    writer.writerows(items)
            (full / "valid").mkdir()
            (full / "valid/good_overlay.jpg").write_bytes(b"fixture")
            self.assertEqual(sync(root, apply=False), [("train", "bad.jpg")])
            self.assertEqual(sync(root, apply=True), [("train", "bad.jpg")])
            self.assertFalse((root / "train/images/bad.jpg").exists())
            self.assertTrue((root / ".rejected/train/images/bad.jpg").is_file())
            self.assertTrue((root / "valid/images/good.jpg").is_file())
            self.assertEqual(read_excluded_training_images(manifest), {"bad.jpg"})
            with manifest.open(newline="", encoding="utf-8") as file:
                self.assertEqual(next(csv.DictReader(file))["status"], EXCLUDED_OVERLAY_STATUS)
            self.assertEqual(sync(root, apply=True), [])


if __name__ == "__main__":
    unittest.main()
