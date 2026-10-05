"""Create a non-destructive YOLO dataset with one class removed.

The default removes ``sediment_deposit`` instances from v5 while keeping the
images. If an image contains other classes, those boxes remain and class IDs
are compacted. Images whose labels become empty are retained as background
examples unless ``--drop-empty-images`` is supplied.

Example:
    python remove_class_instances.py --input final_dataset_v5 --output final_dataset_v5_no_sediment
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "final_dataset_v5")
    parser.add_argument("--output", type=Path, default=ROOT / "final_dataset_v5_no_sediment")
    parser.add_argument("--remove", default="sediment_deposit", help="class name or numeric class ID")
    parser.add_argument("--drop-empty-images", action="store_true")
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def resolve_names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    raise ValueError("data.yaml must contain a names list or numeric-keyed names mapping")


def resolve_remove_id(value: str, names: list[str]) -> int:
    if value.isdigit():
        class_id = int(value)
        if not 0 <= class_id < len(names):
            raise ValueError(f"Class ID {class_id} is outside names range 0..{len(names) - 1}")
        return class_id
    if value not in names:
        raise ValueError(f"Unknown class '{value}'. Available classes: {names}")
    return names.index(value)


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


def transform_label(source: Path, destination: Path, remove_id: int) -> tuple[int, int]:
    kept: list[str] = []
    removed = 0
    malformed = 0
    if source.is_file():
        for raw in source.read_text(encoding="utf-8").splitlines():
            fields = raw.split()
            if len(fields) != 5:
                malformed += 1
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                malformed += 1
                continue
            if class_id == remove_id:
                removed += 1
                continue
            if class_id > remove_id:
                fields[0] = str(class_id - 1)
            kept.append(" ".join(fields))
    destination.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed, malformed


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    source_yaml = input_dir / "data.yaml"
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {source_yaml}")
    names = resolve_names(data)
    remove_id = resolve_remove_id(str(args.remove), names)
    kept_names = [name for index, name in enumerate(names) if index != remove_id]

    output_dir.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    materialized: Counter[str] = Counter()
    removed_instances = 0
    malformed_labels = 0
    dropped_images = 0

    try:
        for split in SPLITS:
            image_dir = input_dir / split / "images"
            label_dir = input_dir / split / "labels"
            if not image_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(f"Expected {split}/images and {split}/labels under {input_dir}")
            output_images = output_dir / split / "images"
            output_labels = output_dir / split / "labels"
            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)

            for image in sorted(image_dir.iterdir(), key=lambda path: path.name.lower()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                source_label = label_dir / f"{image.stem}.txt"
                temporary_label = output_labels / source_label.name
                removed, malformed = transform_label(source_label, temporary_label, remove_id)
                if args.drop_empty_images and temporary_label.read_text(encoding="utf-8").strip() == "":
                    temporary_label.unlink()
                    dropped_images += 1
                    continue
                materialized[materialize(image, output_images / image.name, args.image_mode)] += 1
                removed_instances += removed
                malformed_labels += malformed
                counts[split] += 1

        output_data = {
            "path": str(output_dir),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(kept_names),
            "names": {index: name for index, name in enumerate(kept_names)},
        }
        with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(output_data, handle, sort_keys=False, allow_unicode=True)

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(input_dir),
            "output_dataset": str(output_dir),
            "removed_class": names[remove_id],
            "removed_class_id_before_filter": remove_id,
            "classes_after_filter": kept_names,
            "images_by_split": dict(counts),
            "removed_instances": removed_instances,
            "dropped_empty_images": dropped_images,
            "malformed_label_rows_skipped": malformed_labels,
            "empty_images_retained": not args.drop_empty_images,
            "image_materialization": dict(materialized),
        }
        with (output_dir / "filter_report.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)
    except Exception:
        print(f"Filtering stopped; partial output remains at {output_dir}")
        raise

    print(f"Removed class: {names[remove_id]} ({remove_id})")
    print(f"Retained classes: {kept_names}")
    print(f"Images: {dict(counts)}; removed instances: {removed_instances}; dropped images: {dropped_images}")
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
