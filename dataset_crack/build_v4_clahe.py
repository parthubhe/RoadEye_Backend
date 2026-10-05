"""Build a CLAHE-enhanced copy of ``final_dataset_v4`` safely.

The source dataset is never modified.  That matters because v4 images are
hard-linked to their original source images: writing CLAHE output in place
would silently alter the source data too.

Example:

    python build_v4_clahe.py
    python build_v4_clahe.py --clip-limit 2.5 --tile-size 8

The result is ``final_dataset_v4_clahe`` by default.  Labels are copied
unchanged because CLAHE only changes pixel intensities, not image geometry.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "final_dataset_v4")
    parser.add_argument("--output", type=Path, default=ROOT / "final_dataset_v4_clahe")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE contrast cap (default: 2.0)")
    parser.add_argument("--tile-size", type=int, default=8, help="CLAHE grid width/height (default: 8)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG encoding quality, 1-100 (default: 95)")
    parser.add_argument("--workers", type=int, default=4, help="parallel image workers (default: 4)")
    return parser.parse_args()


def apply_clahe_bgr(image_bgr: Any, clip_limit: float, tile_size: int) -> Any:
    """Apply CLAHE only to L in CIE Lab, preserving colour information."""

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced_lab = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_clahe(image: Any, clip_limit: float, tile_size: int) -> Any:
    """Enhance grayscale, BGR, or BGRA image data without changing its shape."""

    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape}")
    if image.shape[2] == 3:
        return apply_clahe_bgr(image, clip_limit, tile_size)
    if image.shape[2] == 4:
        enhanced_bgr = apply_clahe_bgr(image[:, :, :3], clip_limit, tile_size)
        return cv2.merge((*cv2.split(enhanced_bgr), image[:, :, 3]))
    raise ValueError(f"Unsupported channel count {image.shape[2]}")


def encoding_options(path: Path, jpeg_quality: int) -> list[int]:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    return []


def transform_image(
    source: Path,
    destination: Path,
    clip_limit: float,
    tile_size: int,
    jpeg_quality: int,
) -> tuple[str, tuple[int, ...]]:
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {source}")
    enhanced = apply_clahe(image, clip_limit, tile_size)
    if enhanced.shape != image.shape:
        raise RuntimeError(f"CLAHE changed image shape for {source}: {image.shape} -> {enhanced.shape}")
    if not cv2.imwrite(str(destination), enhanced, encoding_options(destination, jpeg_quality)):
        raise RuntimeError(f"Could not write enhanced image: {destination}")
    return source.name, tuple(image.shape)


def collect_split_images(input_dir: Path, split: str) -> list[Path]:
    image_dir = input_dir / split / "images"
    label_dir = input_dir / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {split}/images and {split}/labels under {input_dir}")

    images = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in label_dir.glob("*.txt") if path.is_file()}
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    if missing_labels or orphan_labels:
        raise ValueError(
            f"{split} has mismatched image/label stems: "
            f"missing_labels={len(missing_labels)}, orphan_labels={len(orphan_labels)}"
        )
    return images


def write_data_yaml(input_dir: Path, build_dir: Path, final_output_dir: Path) -> None:
    source_yaml = input_dir / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {source_yaml}")
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {source_yaml}")

    # Retain v4's taxonomy but make training unambiguously use CLAHE images.
    data["path"] = str(final_output_dir)
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    with (build_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def verify_split(build_dir: Path, split: str, expected_count: int) -> None:
    image_dir = build_dir / split / "images"
    label_dir = build_dir / split / "labels"
    images = {path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    labels = {path.stem for path in label_dir.glob("*.txt") if path.is_file()}
    if len(images) != expected_count or images != labels:
        raise RuntimeError(
            f"CLAHE validation failed for {split}: images={len(images)}, labels={len(labels)}, "
            f"expected={expected_count}"
        )


def ensure_safe_paths(input_dir: Path, output_dir: Path, temporary_dir: Path) -> None:
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ; CLAHE is never applied in place.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if temporary_dir.exists():
        raise FileExistsError(
            f"A previous incomplete CLAHE build exists: {temporary_dir}. "
            "Inspect or remove it manually before retrying."
        )


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
        for name in ("manifest.csv", "build_report.yaml"):
            source = input_dir / name
            if source.is_file():
                shutil.copy2(source, temporary_dir / name)
                copied_provenance.append(name)

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
