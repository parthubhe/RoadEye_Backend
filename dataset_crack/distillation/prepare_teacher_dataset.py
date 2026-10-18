"""Build a non-overwriting v5 training overlay with RF-DETR hard targets.

Ground-truth boxes are retained. Only high-confidence teacher detections that
do not overlap a ground-truth box are added. Validation and test images/labels
are referenced unchanged; the teacher never predicts on those splits here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "final_dataset_v5" / "data.yaml"
DEFAULT_TEACHER = ROOT / "runs" / "pipe_proto" / "rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7" / "checkpoint_best_total.pth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denominator = area_a + area_b - overlap
    return overlap / denominator if denominator > 0 else 0.0


def xywh_to_xyxy(parts: list[str], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(parts) < 5:
        return None
    try:
        cx, cy, bw, bh = [float(value) for value in parts[1:5]]
    except ValueError:
        return None
    return ((cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height)


def teacher_to_yolo(box: np.ndarray, width: int, height: int, class_id: int) -> tuple[str, tuple[float, float, float, float]] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    line = f"{class_id} {(x1 + x2) / (2 * width):.7f} {(y1 + y2) / (2 * height):.7f} {(x2 - x1) / width:.7f} {(y2 - y1) / height:.7f}"
    return line, (x1, y1, x2, y2)


def load_source(data_yaml: Path):
    settings = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(settings.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train = root / settings.get("train", "train/images")
    val = root / settings.get("val", "valid/images")
    test = root / settings.get("test", "test/images")
    for path in (train, val, test, train.parent / "labels"):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset path: {path}")
    names = settings.get("names")
    if isinstance(names, dict):
        names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
    if names != ["crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"]:
        raise ValueError("Expected the unchanged five-class v5 data.yaml")
    images = sorted(path for path in train.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Training images have duplicate stems; YOLO labels would collide")
    return root, train, val, test, names, images


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--output", type=Path, default=ROOT / "distillation_outputs" / "v5_rfdetr50_to_yolo26s_targets")
    p.add_argument("--confidence", type=float, default=0.85)
    p.add_argument("--overlap-iou", type=float, default=0.30)
    p.add_argument("--max-additions", type=int, default=3)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted overlay with identical settings")
    p.add_argument("--dry-run", action="store_true", help="Check paths and counts without loading RF-DETR or writing files")
    return p


def main() -> None:
    args = parser().parse_args()
    if not 0.01 <= args.confidence <= 0.99 or not 0 <= args.overlap_iou <= 1 or args.max_additions < 1:
        raise ValueError("Invalid confidence, overlap IoU, or maximum additions")
    data_yaml, teacher, output = args.data.resolve(), args.teacher.resolve(), args.output.resolve()
    if not data_yaml.is_file() or not teacher.is_file():
        raise FileNotFoundError("The v5 data.yaml and 50-epoch RF-DETR teacher checkpoint must exist")
    root, train, val, test, names, images = load_source(data_yaml)
    if args.dry_run:
        print(f"Teacher: {teacher}\nDataset: {root}\nTrain images: {len(images)}\nValidation: {val}\nTest: {test}\nOutput: {output}\nNo files written.")
        return
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output or use --resume.")
    fingerprint = {
        "source_yaml": str(data_yaml), "teacher_checkpoint": str(teacher),
        "teacher_sha256": sha256(teacher), "train_images": len(images),
        "confidence": args.confidence, "overlap_iou": args.overlap_iou,
        "max_additions": args.max_additions, "classes": names,
        "method": "high-confidence hard-target response distillation; original ground truth retained",
    }
    config_file = output / "generation_config.json"
    if args.resume:
        if not config_file.is_file() or json.loads(config_file.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("Existing overlay does not match teacher, dataset, or thresholds; cannot resume safely")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "train" / "images").mkdir(parents=True)
        (output / "train" / "labels").mkdir(parents=True)
        config_file.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    import torch
    from rfdetr import RFDETRNano

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    model = RFDETRNano(pretrain_weights=str(teacher), num_classes=5, resolution=512, device=args.device)
    if args.device == "cuda":
        model.inference(compile=False, dtype=torch.float16, inplace=True)
    out_images, out_labels = output / "train" / "images", output / "train" / "labels"
    for index, image_path in enumerate(images, 1):
        dest_image = out_images / image_path.name
        dest_label = out_labels / f"{image_path.stem}.txt"
        if not dest_image.exists():
            try:
                os.link(image_path, dest_image)
            except OSError:
                shutil.copy2(image_path, dest_image)
        if dest_label.exists():
            continue
        original_label = train.parent / "labels" / dest_label.name
        original_lines = original_label.read_text(encoding="utf-8").splitlines() if original_label.is_file() else []
        original_lines = [line.strip() for line in original_lines if line.strip()]
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            prediction = model.predict(rgb, threshold=args.confidence)
        if isinstance(prediction, list):
            prediction = prediction[0]
        occupied = [box for line in original_lines if (box := xywh_to_xyxy(line.split(), width, height)) is not None]
        candidates = sorted(zip(prediction.xyxy, prediction.confidence, prediction.class_id), key=lambda item: -float(item[1]))
        added = []
        for box, score, class_id in candidates:
            class_id = int(class_id)
            if not (0 <= class_id < 5) or float(score) < args.confidence:
                continue
            converted = teacher_to_yolo(box, width, height, class_id)
            if converted is None:
                continue
            line, corners = converted
            if any(iou(corners, existing) >= args.overlap_iou for existing in occupied):
                continue
            added.append(line)
            occupied.append(corners)
            if len(added) >= args.max_additions:
                break
        temp = dest_label.with_suffix(".txt.tmp")
        temp.write_text("\n".join(original_lines + added) + ("\n" if original_lines or added else ""), encoding="utf-8")
        os.replace(temp, dest_label)
        if index % 250 == 0:
            print(f"Processed {index}/{len(images)} training images", flush=True)

    overlay_yaml = {
        "path": str(output), "train": "train/images", "val": str(val), "test": str(test),
        "nc": 5, "names": {index: name for index, name in enumerate(names)},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(overlay_yaml, sort_keys=False), encoding="utf-8")
    counts = Counter()
    for source_image in images:
        original = train.parent / "labels" / f"{source_image.stem}.txt"
        enhanced = out_labels / f"{source_image.stem}.txt"
        before = len([line for line in original.read_text(encoding="utf-8").splitlines() if line.strip()]) if original.is_file() else 0
        after_lines = [line for line in enhanced.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in after_lines[before:]:
            counts[names[int(line.split()[0])]] += 1
    summary = {"train_images": len(images), "teacher_boxes_added": sum(counts.values()), "added_by_class": dict(counts), "validation_and_test_unchanged": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Ready: {output / 'data.yaml'}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
