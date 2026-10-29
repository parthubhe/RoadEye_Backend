"""Evaluate the four trained v5 models on their untouched test split.

The baseline was trained on normal v5 images, while the attention models were
trained on v5 CLAHE images. Each model is therefore evaluated against the
matching test image domain. Results are written to separate, non-overwriting
directories and an aggregate CSV is created after all evaluations finish.

Run from the repository root, for example:

    venv\\Scripts\\python.exe dataset_crack\\evaluate_v5_test.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

try:  # Supports both script and module invocation.
    from attention_modules import CBAM, SimAM, SqueezeExcitation
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    from dataset_crack.attention_modules import CBAM, SimAM, SqueezeExcitation


ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = ROOT / "runs" / "pipe_proto"
DEFAULT_DATA = {
    "normal": ROOT / "final_dataset_v5" / "data.yaml",
    "clahe": ROOT / "final_dataset_v5_clahe" / "data.yaml",
}
DEFAULT_MODELS = {
    "baseline": ROOT / "runs" / "pipe_proto" / "yolo26s_v5_cctv" / "weights" / "best.pt",
    "simam": ROOT / "runs" / "pipe_proto" / "yolo26s_v5_clahe_simam" / "weights" / "best.pt",
    "cbam": ROOT / "runs" / "pipe_proto" / "yolo26s_v5_clahe_cbam" / "weights" / "best.pt",
    "se": ROOT / "runs" / "pipe_proto" / "yolo26s_v5_clahe_se" / "weights" / "best.pt",
}


def register_attention_modules() -> None:
    """Make custom layers available while Ultralytics loads checkpoints."""

    tasks.SimAM = SimAM
    tasks.CBAM = CBAM
    tasks.SqueezeExcitation = SqueezeExcitation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_MODELS["baseline"])
    parser.add_argument("--simam", type=Path, default=DEFAULT_MODELS["simam"])
    parser.add_argument("--cbam", type=Path, default=DEFAULT_MODELS["cbam"])
    parser.add_argument("--se", type=Path, default=DEFAULT_MODELS["se"])
    parser.add_argument("--normal-data", type=Path, default=DEFAULT_DATA["normal"])
    parser.add_argument("--clahe-data", type=Path, default=DEFAULT_DATA["clahe"])
    parser.add_argument("--dry-run", action="store_true", help="Load/check all checkpoints without evaluating")
    return parser.parse_args()


def available_name(project: Path, requested: str) -> str:
    candidate = requested
    number = 2
    while (project / candidate).exists():
        candidate = f"{requested}_{number}"
        number += 1
    return candidate


def available_file(directory: Path, requested: str) -> Path:
    candidate = directory / requested
    stem, suffix = candidate.stem, candidate.suffix
    number = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{number}{suffix}"
        number += 1
    return candidate


def metric_value(metrics: object, name: str) -> float:
    value = getattr(getattr(metrics, "box"), name)
    return float(value)


def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.imgsz <= 0 or args.workers < 0:
        raise ValueError("batch and imgsz must be positive; workers cannot be negative")

    register_attention_modules()
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    specs = [
        ("baseline", args.baseline, args.normal_data, "yolo26s_v5_cctv_test"),
        ("simam", args.simam, args.clahe_data, "yolo26s_v5_clahe_simam_test"),
        ("cbam", args.cbam, args.clahe_data, "yolo26s_v5_clahe_cbam_test"),
        ("se", args.se, args.clahe_data, "yolo26s_v5_clahe_se_test"),
    ]

    for label, checkpoint, data_yaml, _ in specs:
        checkpoint = checkpoint.resolve()
        data_yaml = data_yaml.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint: {checkpoint}")
        if not data_yaml.is_file():
            raise FileNotFoundError(f"Missing {label} test YAML: {data_yaml}")

    rows: list[dict[str, object]] = []
    for label, checkpoint, data_yaml, requested_name in specs:
        checkpoint = checkpoint.resolve()
        data_yaml = data_yaml.resolve()
        run_name = available_name(project, requested_name)
        print(f"\n[{label}] checkpoint: {checkpoint}")
        print(f"[{label}] data:       {data_yaml} (split=test)")
        print(f"[{label}] output:     {project / run_name}")
        model = YOLO(str(checkpoint))
        if args.dry_run:
            print(f"[{label}] checkpoint loaded successfully")
            continue

        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            project=str(project),
            name=run_name,
            exist_ok=False,
        )
        rows.append(
            {
                "model": label,
                "checkpoint": str(checkpoint),
                "dataset": "v5" if label == "baseline" else "v5_clahe",
                "split": "test",
                "precision": metric_value(metrics, "mp"),
                "recall": metric_value(metrics, "mr"),
                "mAP50": metric_value(metrics, "map50"),
                "mAP50_95": metric_value(metrics, "map"),
                "run": str(project / run_name),
            }
        )

    if args.dry_run:
        print("\nDry run complete; no test evaluation started.")
        return

    summary_path = available_file(project, f"v5_test_comparison_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTest-set comparison:")
    print(f"{'Model':<10} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    for row in rows:
        print(
            f"{row['model']:<10} {float(row['precision']):8.4f} {float(row['recall']):8.4f} "
            f"{float(row['mAP50']):8.4f} {float(row['mAP50_95']):10.4f}"
        )
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    freeze_support()
    main()
