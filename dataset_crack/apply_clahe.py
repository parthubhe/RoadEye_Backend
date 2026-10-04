"""
Applies CLAHE (local contrast enhancement) to every image in a YOLO-format
dataset and writes a parallel copy. Labels are copied unchanged -- CLAHE
only remaps pixel intensity, it never moves anything spatially, so existing
bounding boxes stay valid.

Usage:
    python apply_clahe.py <input_dataset_dir> <output_dataset_dir>

Expects the standard train/valid/test + images/labels structure (same
layout as sewer_remapped).
"""
import argparse
import shutil
from pathlib import Path
import cv2


def apply_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def process_split(input_dir, output_dir, split, clip_limit, tile_grid_size):
    img_in = input_dir / split / "images"
    lbl_in = input_dir / split / "labels"
    if not img_in.exists():
        return 0
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in img_in.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  could not read {img_path.name}, skipping")
            continue
        enhanced = apply_clahe(img, clip_limit, tile_grid_size)
        cv2.imwrite(str(img_out / img_path.name), enhanced)

        lbl_path = lbl_in / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)
        else:
            (lbl_out / (img_path.stem + ".txt")).touch()
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--clip-limit", type=float, default=2.0)
    ap.add_argument("--tile-size", type=int, default=8)
    args = ap.parse_args()

    tile_grid = (args.tile_size, args.tile_size)

    for split in ["train", "valid", "test"]:
        n = process_split(args.input_dir, args.output_dir, split, args.clip_limit, tile_grid)
        print(f"{split}: {n} images processed")

    src_yaml = args.input_dir / "data.yaml"
    if src_yaml.exists():
        shutil.copy2(src_yaml, args.output_dir / "data.yaml")
        print("Copied data.yaml")


if __name__ == "__main__":
    main()
