"""Stage one visually reviewed SAM mask without changing the original dataset.

Use --confirmed only after looking at the mask overlaid on the source image.
This script does not judge whether the mask actually follows every crack.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .data import dataset_paths, read_labels

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "hybrid_crack_masks" / "review_queue.csv")
    parser.add_argument("--data", type=Path, default=ROOT / "final_dataset_v5" / "data.yaml")
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--image", required=True, help="Image filename as shown in the review CSV")
    parser.add_argument("--mask", type=Path, help="Visually reviewed binary mask PNG for a crack-positive image")
    parser.add_argument("--confirmed", action="store_true", help="I visually checked all visible cracks and false regions")
    parser.add_argument("--max-mask-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not args.confirmed:
        parser.error("Visual review is required: pass --confirmed only after checking the overlay")
    manifest = args.manifest.resolve(strict=True)
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    matches = [row for row in rows if row["split"] == args.split and row["image"] == args.image]
    if len(matches) != 1:
        raise ValueError("The split/image must match exactly one row in the review queue")
    row = matches[0]
    if row["status"].strip().lower() != "pending":
        raise ValueError("Only pending rows may be approved; existing approvals are not overwritten")
    image_path = dataset_paths(args.data.resolve())[args.split] / args.image
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    has_crack_box = bool((read_labels(image_path)[:, 0] == 0).any())
    if has_crack_box != (row["kind"] == "positive"):
        raise ValueError("Queue kind and original crack-box labels disagree")
    size = Image.open(image_path).size
    if row["kind"] == "positive":
        if not args.mask:
            parser.error("A reviewed positive mask needs --mask")
        source = Image.open(args.mask.resolve(strict=True))
        if source.mode not in {"L", "1", "P"} or source.size != size:
            raise ValueError("Source mask must be single-channel and same size as the image")
        pixels = np.asarray(source.convert("L"), dtype=np.uint8)
        if not set(np.unique(pixels).tolist()).issubset({0, 1, 255}):
            raise ValueError("Source mask must contain only 0/1 or 0/255")
        fraction = float((pixels > 0).mean())
        if fraction <= 0 or fraction > args.max_mask_fraction:
            raise ValueError(f"Mask occupies {fraction:.1%} of the image; review before accepting")
    else:
        if args.mask:
            parser.error("For a visually verified negative, do not provide --mask")
        pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
        fraction = 0.0
    target = Path(row["mask_path"])
    if not target.is_absolute():
        target = manifest.parent / target
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing mask: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pixels > 0).astype(np.uint8) * 255, "L").save(target)
    row["status"] = "approved"
    temporary = manifest.with_suffix(".pending.csv")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "image", "kind", "mask_path", "status"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    print(f"Approved {args.split}/{args.image}: {target} ({fraction:.2%} crack pixels)")


if __name__ == "__main__":
    main()
