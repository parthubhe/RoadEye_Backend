"""Train YOLO26s with parallel SimAM, CBAM, and SE attention.

The same script can train on normal v5, v5 CLAHE, or another compatible
five-class YAML. Existing run directories are never overwritten.

Examples:
    python train_parallel.py --dataset v5_clahe
    python train_parallel.py --data final_dataset_v5/data.yaml --epochs 80
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from multiprocessing import freeze_support
from pathlib import Path

import yaml
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

try:
    from attention_modules import ParallelAttention
except ModuleNotFoundError:  # pragma: no cover - module invocation
    from dataset_crack.attention_modules import ParallelAttention


ROOT = Path(__file__).resolve().parent
MODEL_YAML = ROOT / "yolo26s-parallel.yaml"
GENERATED_MODELS = ROOT / "generated_attention_models"
DATASETS = {
    "v5": ROOT / "final_dataset_v5" / "data.yaml",
    "v5_clahe": ROOT / "final_dataset_v5_clahe" / "data.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="v5_clahe")
    parser.add_argument("--data", type=Path, default=None, help="Optional custom data.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "pipe_proto")
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analyze-errors",
        action="store_true",
        help="After training, record per-image false positives/negatives and copy error images",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=None,
        help="Optional output directory for post-training error analysis",
    )
    return parser.parse_args()


def register_module() -> None:
    tasks.ParallelAttention = ParallelAttention


def unique_name(project: Path, requested: str) -> str:
    candidate = requested
    suffix = 2
    while (project / candidate).exists():
        candidate = f"{requested}_{suffix}"
        suffix += 1
    return candidate


def dataset_token(path: Path, requested: str) -> str:
    text = str(path.parent if path.name == "data.yaml" else path).lower()
    if "final_dataset_v5_clahe" in text:
        return "v5_clahe"
    if "final_dataset_v5" in text:
        return "v5"
    return re.sub(r"[^a-z0-9]+", "_", requested.lower()).strip("_") or "custom"


def class_count(data_path: Path) -> int:
    with data_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names")
    if isinstance(names, dict):
        count = len(names)
    elif isinstance(names, (list, tuple)):
        count = len(names)
    else:
        raise ValueError(f"Dataset YAML must contain a names list/dict: {data_path}")
    if count < 1:
        raise ValueError(f"Dataset YAML contains no classes: {data_path}")
    return count


def model_yaml_for_classes(nc: int) -> Path:
    """Create a class-count-specific copy without modifying the source YAML."""
    with MODEL_YAML.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["nc"] = nc
    GENERATED_MODELS.mkdir(parents=True, exist_ok=True)
    target = GENERATED_MODELS / f"yolo26s_parallel_{nc}c.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs, imgsz, and batch must be positive; workers cannot be negative")
    if not MODEL_YAML.is_file():
        raise FileNotFoundError(f"Missing model YAML: {MODEL_YAML}")

    register_module()
    data_path = (args.data if args.data is not None else DATASETS[args.dataset]).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {data_path}")

    nc = class_count(data_path)
    model_yaml = model_yaml_for_classes(nc)
    model = YOLO(str(model_yaml))
    if not args.no_pretrained:
        model.load("yolo26s.pt")

    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    token = dataset_token(data_path, args.dataset)
    requested = args.name or f"yolo26s_{token}_parallel_simam_cbam_se"
    run_name = unique_name(project, requested)
    print(f"data: {data_path}")
    print(f"classes: {nc}")
    print(f"model: {model_yaml}")
    print(f"run: {project / run_name}")

    if args.dry_run:
        print("Dry run complete; no training started.")
        return

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        cls_pw=0.0,
        patience=100,
        close_mosaic=10,
        device=args.device,
        workers=args.workers,
        seed=0,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=False,
        plots=True,
    )

    if args.analyze_errors:
        best = project / run_name / "weights" / "best.pt"
        error_output = args.error_output or (project / f"{run_name}_error_analysis")
        command = [
            sys.executable,
            str(ROOT / "analyze_parallel_errors.py"),
            "--weights", str(best),
            "--data", str(data_path),
            "--output", str(error_output),
            "--device", str(args.device),
            "--imgsz", str(args.imgsz),
            "--workers", str(args.workers),
            "--copy-images",
        ]
        print("Starting post-training error analysis...")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    freeze_support()
    main()
