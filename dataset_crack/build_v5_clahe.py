"""Create a non-destructive CLAHE copy of ``final_dataset_v5``.

CLAHE is applied to the L channel in CIE Lab space. Labels, split membership,
and provenance files are copied unchanged. Existing output directories are
never overwritten.

Examples
--------
    python build_v5_clahe.py
    python build_v5_clahe.py --clip-limit 2.5 --tile-size 8
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

try:  # Works both as ``python dataset_crack/build_v5_clahe.py`` and as a module.
    from build_v4_clahe import (
        SPLITS,
        collect_split_images,
        ensure_safe_paths,
        transform_image,
        verify_split,
        write_data_yaml,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    from dataset_crack.build_v4_clahe import (
        SPLITS,
        collect_split_images,
        ensure_safe_paths,
        transform_image,
        verify_split,
        write_data_yaml,
    )


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "final_dataset_v5")
    parser.add_argument("--output", type=Path, default=ROOT / "final_dataset_v5_clahe")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_limit <= 0 or args.tile_size <= 0 or args.workers <= 0:
        raise ValueError("--clip-limit, --tile-size, and --workers must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    temporary_dir = output_dir.with_name(f"{output_dir.name}.__clahe_building__")
    ensure_safe_paths(input_dir, output_dir, temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    split_counts: dict[str, int] = {}

    try:
        for split in SPLITS:
            source_images = collect_split_images(input_dir, split)
            output_images = temporary_dir / split / "images"
            output_labels = temporary_dir / split / "labels"
            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)

            for source_image in source_images:
                source_label = input_dir / split / "labels" / f"{source_image.stem}.txt"
                shutil.copy2(source_label, output_labels / source_label.name)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        transform_image,
                        source_image,
                        output_images / source_image.name,
                        args.clip_limit,
                        args.tile_size,
                        args.jpeg_quality,
                    )
                    for source_image in source_images
                ]
                for future in futures:
                    future.result()

            split_counts[split] = len(source_images)
            verify_split(temporary_dir, split, split_counts[split])
            print(f"{split}: {split_counts[split]} images enhanced")

        write_data_yaml(input_dir, temporary_dir, output_dir)
        copied_provenance: list[str] = []
        for filename in ("manifest.csv", "build_report.yaml"):
            source = input_dir / filename
            if source.is_file():
                shutil.copy2(source, temporary_dir / filename)
                copied_provenance.append(filename)

        report = {
            "source_dataset": str(input_dir),
            "output_dataset": str(output_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "CLAHE on L channel in CIE Lab colour space",
            "parameters": {
                "clip_limit": args.clip_limit,
                "tile_grid_size": [args.tile_size, args.tile_size],
                "jpeg_quality": args.jpeg_quality,
                "workers": args.workers,
            },
            "opencv_version": cv2.__version__,
            "images_by_split": split_counts,
            "labels": "Copied byte-for-byte from source dataset.",
            "copied_provenance": copied_provenance,
        }
        with (temporary_dir / "clahe_report.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)
        os.replace(temporary_dir, output_dir)
    except Exception:
        print(f"CLAHE build stopped; partial files, if any, remain in {temporary_dir}")
        raise

    print(f"Wrote CLAHE dataset: {output_dir}")


if __name__ == "__main__":
    main()
