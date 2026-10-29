"""List and optionally copy per-image errors for a trained parallel YOLO model.

Example:
    python analyze_parallel_errors.py \
        --weights runs/pipe_proto/yolo26s_v5_clahe_parallel_simam_cbam_se/weights/best.pt \
        --data final_dataset_v5_clahe/data.yaml

Predictions are matched to ground-truth boxes of the same class using greedy
confidence order and an IoU threshold. Unmatched predictions are reported as
false positives; unmatched labels are reported as false negatives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "parallel_error_analysis")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--batch", type=int, default=1, help="Bounded prediction batch; no training is performed")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--copy-images", action="store_true")
    return p.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def image_list(data: dict, data_path: Path, split: str = "test") -> list[Path]:
    value = data.get(split)
    if not value:
        raise ValueError(f"Dataset YAML has no {split} split")
    root = Path(data.get('path') or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    source = Path(value)
    if not source.is_absolute():
        source = (root / source).resolve()
    if source.is_file() and source.suffix.lower() == ".txt":
        images = [Path(line.strip()) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [(p if p.is_absolute() else source.parent / p).resolve() for p in images]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    raise FileNotFoundError(f"{split} split not found: {source}")


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    try:
        idx = next(i for i, part in enumerate(parts) if part.lower() == "images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except StopIteration:
        return image.with_suffix(".txt")


def unique_output(path: Path) -> Path:
    candidate = path
    number = 2
    while candidate.exists():
        candidate = path.parent / f"{path.name}_{number}"
        number += 1
    return candidate


def load_labels(path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing label file (use an empty file for background images): {path}')
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f'Invalid detection annotation: {path}')
        cls, xc, yc, w, h = map(float, fields)
        if not all(math.isfinite(v) for v in (cls, xc, yc, w, h)) or cls != int(cls):
            raise ValueError(f'Invalid detection annotation: {path}')
        records.append((int(cls), [(xc - w / 2) * width, (yc - h / 2) * height,
                                    (xc + w / 2) * width, (yc + h / 2) * height]))
    return records


def match_errors(predictions, labels, threshold):
    """One-to-one, same-class matching in descending confidence order."""
    used, unmatched = set(), []
    for pred_cls, pred_box, confidence in sorted(predictions, key=lambda item: item[2], reverse=True):
        candidates = [(iou(pred_box, gt_box), idx) for idx, (gt_cls, gt_box) in enumerate(labels)
                      if idx not in used and gt_cls == pred_cls]
        if candidates and max(candidates)[0] >= threshold:
            used.add(max(candidates)[1])
        else:
            unmatched.append((pred_cls, pred_box, confidence))
    return unmatched, [(cls, box) for idx, (cls, box) in enumerate(labels) if idx not in used]


def predict_images(model, images, cfg):
    for start in range(0, len(images), cfg.batch):
        chunk = images[start:start + cfg.batch]
        results = model.predict(source=[str(image) for image in chunk], conf=cfg.conf, iou=cfg.iou,
                                imgsz=cfg.imgsz, device=cfg.device, verbose=False, batch=cfg.batch)
        if len(results) != len(chunk):
            raise RuntimeError('Prediction count does not match input batch')
        for image, result in zip(chunk, results):
            if Path(result.path).resolve() != image.resolve():
                raise RuntimeError('Prediction image order mismatch')
            yield image, result


def main() -> None:
    cfg = args()
    if not 0 < cfg.conf < 1 or not 0 < cfg.iou <= 1:
        raise ValueError("--conf must be in (0,1), and --iou must be in (0,1]")
    if cfg.batch < 1:
        raise ValueError('--batch must be positive')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'notebook/workspace/ultralytics_config'))
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    import ultralytics
    import ultralytics.nn.tasks as tasks
    try:
        from attention_modules import ParallelAttention
    except ModuleNotFoundError:
        from dataset_crack.attention_modules import ParallelAttention
    tasks.ParallelAttention = ParallelAttention
    weights, data_path = cfg.weights.resolve(), cfg.data.resolve()
    if not weights.is_file() or not data_path.is_file():
        raise FileNotFoundError("Both --weights and --data must exist")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    names = [names.get(i, names.get(str(i))) for i in range(len(names))] if isinstance(names, dict) else list(names)
    images = image_list(data, data_path, cfg.split)
    if not images or len(set(images)) != len(images):
        raise ValueError('Empty split or duplicate image paths')
    for image in images:
        if not image.is_file() or not label_path(image).is_file():
            raise FileNotFoundError(f'Missing image/label pair: {image}')
    out = unique_output(cfg.output.resolve())
    out.mkdir(parents=True, exist_ok=False)
    fp_dir, fn_dir = out / "false_positive_images", out / "false_negative_images"
    if cfg.copy_images:
        fp_dir.mkdir()
        fn_dir.mkdir()
    fp_rows, fn_rows = [], []
    model = YOLO(str(weights))
    if [model.names[i] for i in range(len(model.names))] != names:
        raise ValueError('Checkpoint taxonomy does not match dataset')
    print(f'Analyzing {len(images)} {cfg.split} images; output: {out}', flush=True)
    for number, (image, result) in enumerate(predict_images(model, images, cfg), 1):
        height, width = result.orig_shape
        labels = load_labels(label_path(image), width, height)
        if any(cls < 0 or cls >= len(names) for cls, _ in labels):
            raise ValueError(f'Invalid class in {label_path(image)}')
        predictions = []
        if result.boxes is not None:
            for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                predictions.append((int(cls), box, float(conf)))
        unmatched_predictions, unmatched_labels = match_errors(predictions, labels, cfg.iou)
        for pred_cls, box, confidence in unmatched_predictions:
            fp_rows.append({"image": str(image), "predicted_class": names[pred_cls] if pred_cls < len(names) else str(pred_cls),
                            "confidence": f"{confidence:.6f}", "x1": f"{box[0]:.1f}", "y1": f"{box[1]:.1f}",
                            "x2": f"{box[2]:.1f}", "y2": f"{box[3]:.1f}"})
        for gt_cls, box in unmatched_labels:
            fn_rows.append({"image": str(image), "true_class": names[gt_cls] if gt_cls < len(names) else str(gt_cls),
                            "x1": f"{box[0]:.1f}", "y1": f"{box[1]:.1f}", "x2": f"{box[2]:.1f}", "y2": f"{box[3]:.1f}"})
        if cfg.copy_images:
            if unmatched_predictions:
                shutil.copy2(image, fp_dir / f"{number:05d}_{image.name}")
            if unmatched_labels:
                shutil.copy2(image, fn_dir / f"{number:05d}_{image.name}")
        if number % 100 == 0 or number == len(images):
            print(f"Processed {number}/{len(images)} images", flush=True)
    with (out / "false_positives.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "predicted_class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader(); writer.writerows(fp_rows)
    with (out / "false_negatives.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "true_class", "x1", "y1", "x2", "y2"])
        writer.writeheader(); writer.writerows(fn_rows)
    fp_images, fn_images = {row['image'] for row in fp_rows}, {row['image'] for row in fn_rows}
    for filename, members in [('false_positive_images.txt', fp_images), ('false_negative_images.txt', fn_images),
                              ('all_error_images.txt', fp_images | fn_images)]:
        (out / filename).write_text(''.join(f'{path}\n' for path in sorted(members)), encoding='utf-8')
    summary = {
        'status': 'complete', 'split': cfg.split, 'images_processed': len(images),
        'checkpoint': str(weights), 'checkpoint_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'data_yaml': str(data_path), 'confidence': cfg.conf, 'matching_iou': cfg.iou,
        'prediction_iou': cfg.iou, 'imgsz': cfg.imgsz, 'batch': cfg.batch,
        'ultralytics': ultralytics.__version__, 'matching': 'greedy confidence order, same class, one-to-one',
        'fp_instances': len(fp_rows), 'fn_instances': len(fn_rows),
        'fp_unique_images': len(fp_images), 'fn_unique_images': len(fn_images),
        'both_fp_fn_images': len(fp_images & fn_images), 'any_error_unique_images': len(fp_images | fn_images),
        'fp_by_class': dict(Counter(row['predicted_class'] for row in fp_rows)),
        'fn_by_class': dict(Counter(row['true_class'] for row in fn_rows)),
        'note': 'Diagnostic disagreements with labels, not an automatic deletion list. Training errors are in-sample. '
                'Original annotations are used without silently dropping zero-size boxes; counts need not match validator confusion matrices.',
    }
    (out / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"False positives: {len(fp_rows)}")
    print(f"False negatives: {len(fn_rows)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
