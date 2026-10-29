"""Audit the materialized hybrid subset and write a reproducible count report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .data import IMAGE_SUFFIXES, dataset_paths, read_excluded_training_images, read_labels
from .sync_removed_images import ACTIVE_STATUSES, DEFAULT_SUBSET, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--full-data", type=Path, default=ROOT / "final_dataset_v5" / "data.yaml")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = root / "manifest.csv"
    rows, _ = load_manifest(root)
    subset_paths = dataset_paths((root / "data.yaml").resolve(strict=True))
    full_paths = dataset_paths(args.full_data.resolve(strict=True))
    expected_paths = {split: (root / split / "images").resolve() for split in ("train", "valid", "test")}
    if subset_paths != expected_paths:
        raise ValueError(f"Subset data.yaml does not point to the materialized images: {subset_paths}")
    active = [row for row in rows if row["status"].strip().lower() in ACTIVE_STATUSES]
    counts = Counter()
    sources = Counter()
    active_names = {split: set() for split in ("train", "valid", "test")}
    for row in active:
        split, name, kind = row["split"], row["image"], row["kind"]
        image_path = root / split / "images" / name
        label_path = root / split / "labels" / f"{Path(name).stem}.txt"
        mask_path = root / row["mask_path"]
        if not all(path.is_file() for path in (image_path, label_path, mask_path)):
            raise FileNotFoundError(f"Incomplete active row: {split}/{name}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if mask.mode != "L" or mask.size != image.size:
                raise ValueError(f"Bad mask mode/size: {mask_path}")
            values = np.unique(np.asarray(mask))
        if not set(values.tolist()).issubset({0, 255}):
            raise ValueError(f"Non-binary mask: {mask_path}")
        positive = len(values) > 1 or (len(values) == 1 and values[0] == 255)
        has_crack_box = bool((read_labels(image_path)[:, 0] == 0).any())
        if positive != (kind == "positive") or has_crack_box != (kind == "positive"):
            raise ValueError(f"Mask/box/kind disagreement: {split}/{name}")
        active_names[split].add(name)
        counts[(split, kind)] += 1
        sources[row.get("source_family") or Path(name).stem.rsplit("_", 1)[0]] += 1
    for split in active_names:
        images = {path.name for path in (root / split / "images").iterdir()
                  if path.suffix.lower() in IMAGE_SUFFIXES}
        labels = {path.stem for path in (root / split / "labels").glob("*.txt")}
        masks = {path.stem for path in (root / "masks" / split).glob("*.png")}
        stems = {Path(name).stem for name in active_names[split]}
        if images != active_names[split] or labels != stems or masks != stems:
            raise ValueError(f"Orphan or missing materialized files in {split}: "
                             f"images={len(images)}, labels={len(labels)}, masks={len(masks)}, active={len(stems)}")
    excluded_train = read_excluded_training_images(manifest)
    full_train_images = {path.name for path in full_paths["train"].iterdir()
                         if path.suffix.lower() in IMAGE_SUFFIXES}
    if not excluded_train.issubset(full_train_images):
        raise ValueError("Some excluded train filenames are absent from original v5")
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "subset_root": str(root), "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "total_manifest_rows": len(rows), "active_masks": len(active),
        "excluded_rows": len(rows) - len(active),
        "active_by_split_kind": {split: {kind: counts[(split, kind)] for kind in ("positive", "negative")}
                                 for split in ("train", "valid", "test")},
        "active_by_source_family": dict(sorted(sources.items())),
        "full_v5_train_images": len(full_train_images),
        "excluded_from_full_v5_train": len(excluded_train),
        "effective_full_v5_train_images": len(full_train_images - excluded_train),
        "warning": "No validation/test negative masks remain; mask false-positive rate cannot be evaluated there. SAM masks remain pseudo-labels, not verified ground truth.",
    }
    output = root / f"curation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
