"""Create v5 by filtering only close-up domains from the v2 output.

This is the closest possible apples-to-apples comparison with v2: it keeps
v2's exact source-prefixed filenames, split assignments, train oversampling,
class mapping, and empty-label policy.  Only ``pipeline_crack_detection_*``
and ``pipe_inspection_*`` images are removed because the visual audit found
them to be close-up/surface imagery rather than pipe-CCTV POV.

The v2 directory is read-only.  Images are hard-linked by default, labels are
copied, and an audit manifest records every retained source file.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXCLUDED_PREFIXES = ("pipeline_crack_detection_", "pipe_inspection_")
CLASSES = ["crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "dataset_crack" / "final_dataset_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_crack" / "final_dataset_v5")
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def materialize(source: Path, destination: Path, mode: str) -> str:
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def source_id(filename: str) -> str:
    # v2 names are <source>_<counter>.<extension>; preserve the full source
    # prefix rather than relying on the first underscore (sewer_o8use, etc.).
    known = (
        "sewer_o8use",
        "pipeline_crack_detection",
        "sewage_defect_detection",
        "sewer_defects",
        "pipe_inspection",
        "sewage",
    )
    for name in known:
        if filename.startswith(f"{name}_"):
            return name
    return filename.rsplit("_", 1)[0]


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"v2 input dataset does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []
    counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    materialized: Counter[str] = Counter()

    try:
        for split in SPLITS:
            source_images = input_dir / split / "images"
            source_labels = input_dir / split / "labels"
            if not source_images.is_dir() or not source_labels.is_dir():
                raise FileNotFoundError(f"Missing {split}/images or {split}/labels under {input_dir}")
            destination_images = output_dir / split / "images"
            destination_labels = output_dir / split / "labels"
            destination_images.mkdir(parents=True, exist_ok=True)
            destination_labels.mkdir(parents=True, exist_ok=True)

            for image in sorted(source_images.iterdir(), key=lambda path: path.name.lower()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                source = source_id(image.name)
                if image.name.startswith(EXCLUDED_PREFIXES):
                    excluded[source] += 1
                    continue

                label = source_labels / f"{image.stem}.txt"
                output_image = destination_images / image.name
                output_label = destination_labels / label.name
                materialized[materialize(image, output_image, args.image_mode)] += 1
                if label.is_file():
                    shutil.copy2(label, output_label)
                else:
                    output_label.write_text("", encoding="utf-8")
                counts[split] += 1
                manifest_rows.append(
                    {
                        "split": split,
                        "output_image": str(output_image.relative_to(output_dir)),
                        "output_label": str(output_label.relative_to(output_dir)),
                        "source_id": source,
                        "source_image": str(image),
                        "source_label": str(label) if label.is_file() else "",
                        "excluded": "false",
                    }
                )

        data_yaml = {
            "path": str(output_dir),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(CLASSES),
            "names": {index: name for index, name in enumerate(CLASSES)},
        }
        with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data_yaml, handle, sort_keys=False, allow_unicode=True)
        with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

        report = {
            "version": 5,
            "source_dataset": str(input_dir),
            "split_strategy": "Exact final_dataset_v2 split assignments",
            "classes": CLASSES,
            "images_by_split": dict(counts),
            "images_retained": sum(counts.values()),
            "excluded_by_domain": dict(excluded),
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "image_mode": args.image_mode,
            "image_materialization": dict(materialized),
            "empty_labels_retained": True,
            "note": "No v4 empty/unmapped filtering; only close-up domain images removed.",
        }
        with (output_dir / "build_report.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)
    except Exception:
        print(f"Build stopped; partial output remains at {output_dir}")
        raise

    print(f"Retained: train={counts['train']} valid={counts['valid']} test={counts['test']}")
    print(f"Excluded: {dict(excluded)}")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
