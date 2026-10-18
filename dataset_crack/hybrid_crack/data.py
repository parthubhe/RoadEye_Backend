"""Local v5 detection data with optional, explicitly reviewed crack masks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_paths(data_yaml: Path) -> dict[str, Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if list(config["names"].values()) != [
        "crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"
    ]:
        raise ValueError("Expected the original v5 five-class order")
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return {split: (root / config[key]).resolve() for split, key in (("train", "train"), ("valid", "val"), ("test", "test"))}


def read_labels(image_path: Path) -> np.ndarray:
    path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Expected YOLO box row at {path}:{line_number}")
        cls = int(values[0])
        box = [float(x) for x in values[1:]]
        if cls not in range(5) or not all(0 <= x <= 1 for x in box):
            raise ValueError(f"Invalid class or box at {path}:{line_number}")
        if box[2] <= 0 or box[3] <= 0:
            # Existing v5 has occasional zero-area boxes. They cannot train a detector.
            continue
        rows.append([float(cls), *box])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def read_reviewed_masks(manifest: Path, allow_pseudo: bool = False) -> dict[tuple[str, str], tuple[Path, str]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing review manifest: {manifest}. Generate it with make_review_queue.py")
    approved: dict[tuple[str, str], tuple[Path, str]] = {}
    accepted = {"approved", "auto_pseudo"} if allow_pseudo else {"approved"}
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        for line, row in enumerate(csv.DictReader(file), 2):
            if row.get("status", "").strip().lower() not in accepted:
                continue
            split, image_name = row["split"].strip(), row["image"].strip()
            kind, mask_name = row["kind"].strip(), row["mask_path"].strip()
            if split not in {"train", "valid", "test"} or kind not in {"positive", "negative"} or not mask_name:
                raise ValueError(f"Bad approved entry in {manifest}:{line}")
            mask_path = Path(mask_name)
            if not mask_path.is_absolute():
                mask_path = manifest.parent / mask_path
            mask_path = mask_path.resolve()
            if not mask_path.is_file():
                raise FileNotFoundError(f"Approved mask is missing: {mask_path}")
            key = split, image_name
            if key in approved:
                raise ValueError(f"Duplicate approved mask entry: {key}")
            approved[key] = mask_path, kind
    return approved


def read_excluded_training_images(manifest: Path) -> set[str]:
    """Images rejected from the generated subset must not train boxes in full v5."""
    excluded = set()
    rejected_statuses = {"excluded_by_manual_image_removal", "excluded_by_manual_mask_removal",
                         "excluded_by_manual_overlay_removal"}
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if (row.get("split", "").strip() == "train" and
                    row.get("status", "").strip().lower() in rejected_statuses):
                excluded.add(row["image"].strip())
    return excluded


class HybridDataset(Dataset):
    def __init__(self, images_dir: Path, split: str, reviewed: dict, imgsz: int = 512,
                 augment: bool = False, max_images: int | None = None, only_reviewed: bool = False,
                 exclude_names: set[str] | None = None):
        if imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32")
        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)
        excluded = exclude_names or set()
        self.paths = [p for p in sorted(images_dir.iterdir())
                      if p.suffix.lower() in IMAGE_SUFFIXES and p.name not in excluded]
        if only_reviewed:
            self.paths = [p for p in self.paths if (split, p.name) in reviewed]
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.split, self.imgsz, self.augment, self.reviewed = split, imgsz, augment, reviewed
        self.approved_count = sum((split, p.name) in reviewed for p in self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        labels = read_labels(image_path)
        reviewed = self.reviewed.get((self.split, image_path.name))
        if reviewed:
            mask_path, kind = reviewed
            mask = Image.open(mask_path)
            if mask.mode not in {"L", "1", "P"} or mask.size != image.size:
                raise ValueError(f"Mask must be single-channel and match {image_path}: {mask_path}")
            mask = mask.convert("L")
            values = set(np.unique(np.asarray(mask)).tolist())
            if not values.issubset({0, 1, 255}):
                raise ValueError(f"Mask must be binary 0/1 or 0/255: {mask_path}")
            positive = max(values) > 0
            if (kind == "positive") != positive:
                raise ValueError(f"Mask and approved kind disagree: {mask_path}")
            if (kind == "positive") != bool((labels[:, 0] == 0).any()):
                raise ValueError(f"Mask and YOLO crack-box presence disagree: {image_path}")
        else:
            mask = Image.new("L", image.size, 0)  # ignored by loss, NOT a negative label

        ratio = min(self.imgsz / width, self.imgsz / height)
        new_width, new_height = round(width * ratio), round(height * ratio)
        left, top = (self.imgsz - new_width) // 2, (self.imgsz - new_height) // 2
        image = ImageOps.expand(image.resize((new_width, new_height), Image.Resampling.BILINEAR),
                                (left, top, self.imgsz - new_width - left, self.imgsz - new_height - top), fill=(114, 114, 114))
        mask = ImageOps.expand(mask.resize((new_width, new_height), Image.Resampling.NEAREST),
                               (left, top, self.imgsz - new_width - left, self.imgsz - new_height - top), fill=0)
        if len(labels):
            labels[:, 1] = (labels[:, 1] * width * ratio + left) / self.imgsz
            labels[:, 2] = (labels[:, 2] * height * ratio + top) / self.imgsz
            labels[:, 3] *= width * ratio / self.imgsz
            labels[:, 4] *= height * ratio / self.imgsz
        if self.augment and torch.rand(()) < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
            labels[:, 1] = 1 - labels[:, 1]
        pixels = np.asarray(image, dtype=np.uint8).copy()
        mask_pixels = np.asarray(mask, dtype=np.uint8).copy()
        return {
            "img": torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255),
            "mask": torch.from_numpy((mask_pixels > 0).astype(np.float32)),
            "has_mask": torch.tensor(reviewed is not None),
            "cls": torch.from_numpy(labels[:, :1].copy()),
            "bboxes": torch.from_numpy(labels[:, 1:].copy()),
            "path": str(image_path),
        }


def collate(items: list[dict]) -> dict:
    return {
        "img": torch.stack([item["img"] for item in items]),
        "mask": torch.stack([item["mask"] for item in items]),
        "has_mask": torch.stack([item["has_mask"] for item in items]),
        "cls": torch.cat([item["cls"] for item in items]),
        "bboxes": torch.cat([item["bboxes"] for item in items]),
        "batch_idx": torch.cat([torch.full((len(item["cls"]),), i, dtype=torch.float32) for i, item in enumerate(items)]),
        "paths": [item["path"] for item in items],
    }
