"""Draw CSV false-positive/false-negative boxes on copied error images.

Example:
    python draw_error_boxes.py --analysis-dir runs/pipe_proto/..._error_analysis
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def read_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["image"]].append(row)
    return grouped


def find_image(source: str, analysis_dir: Path, kind: str) -> Path | None:
    original = Path(source)
    if original.is_file():
        return original
    folder = analysis_dir / f"{kind}_images"
    matches = list(folder.glob(f"*_{original.name}"))
    return matches[0] if matches else None


def draw(image_path: Path, fp_rows: list[dict[str, str]], fn_rows: list[dict[str, str]], output: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return
    for row in fp_rows:
        box = tuple(int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2"))
        label = f"FP: {row['predicted_class']} {float(row['confidence']):.2f}"
        cv2.rectangle(image, box[:2], box[2:], (0, 0, 255), 2)
        cv2.putText(image, label, (box[0], max(18, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    for row in fn_rows:
        box = tuple(int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2"))
        label = f"FN: {row['true_class']}"
        cv2.rectangle(image, box[:2], box[2:], (255, 0, 0), 2)
        cv2.putText(image, label, (box[0], min(image.shape[0] - 5, box[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def main() -> None:
    cfg = parse_args()
    analysis = cfg.analysis_dir.resolve()
    fp_csv, fn_csv = analysis / "false_positives.csv", analysis / "false_negatives.csv"
    if not fp_csv.is_file() or not fn_csv.is_file():
        raise FileNotFoundError(f"Expected CSV files in {analysis}")
    out = (cfg.output or (analysis / "annotated_error_images")).resolve()
    if out.exists():
        number = 2
        base = out
        while out.exists():
            out = base.parent / f"{base.name}_{number}"
            number += 1
    fp, fn = read_rows(fp_csv), read_rows(fn_csv)
    all_images = sorted(set(fp) | set(fn))
    for index, source in enumerate(all_images, 1):
        source_path = find_image(source, analysis, "false_positive") or find_image(source, analysis, "false_negative")
        if source_path is None:
            print(f"Missing copied image for: {source}")
            continue
        target = out / f"{index:05d}_{Path(source).name}"
        draw(source_path, fp.get(source, []), fn.get(source, []), target)
    print(f"Annotated images written to: {out}")
    print("Red = false-positive prediction; blue = false-negative ground-truth box")


if __name__ == "__main__":
    main()
