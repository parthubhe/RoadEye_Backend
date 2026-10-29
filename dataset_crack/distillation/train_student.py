"""Fine-tune YOLO26s on RF-DETR hard-target overlay without altering v5."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT = ROOT / "runs" / "pipe_proto" / "yolo26s_v5_cctv" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "distillation_outputs" / "v5_rfdetr50_to_yolo26s_targets" / "data.yaml"
PROJECT = ROOT / "runs" / "pipe_proto"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--weights", type=Path, default=DEFAULT_STUDENT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--name", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    data, weights, project = args.data.resolve(), args.weights.resolve(), args.project.resolve()
    if not data.is_file() or not weights.is_file():
        raise FileNotFoundError("Prepared overlay data.yaml and YOLO26s baseline best.pt must exist")
    if not (data.parent / "summary.json").is_file():
        raise ValueError("This is not a completed teacher-generated overlay; summary.json is missing")
    summary = json.loads((data.parent / "summary.json").read_text(encoding="utf-8"))
    if summary.get("teacher_boxes_added", 0) == 0:
        raise ValueError("No teacher boxes were added; this would not be a distillation experiment. Review the teacher threshold and create a new overlay.")
    settings = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    if settings.get("nc") != 5 or Path(settings.get("path", "")).resolve() != data.parent:
        raise ValueError("Prepared overlay YAML is not the expected five-class dataset")
    source = yaml.safe_load((ROOT / "final_dataset_v5" / "data.yaml").read_text(encoding="utf-8"))
    source_root = Path(source["path"]).resolve()
    if Path(settings.get("val", "")).resolve() != (source_root / source["val"]).resolve() or Path(settings.get("test", "")).resolve() != (source_root / source["test"]).resolve():
        raise ValueError("Validation/test paths must point to the unchanged v5 splits")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.workers < 0:
        raise ValueError("Invalid training hyperparameters")
    run_name = args.name or f"pipescout_kd_yolo26s_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = project / run_name
    if target.exists():
        raise FileExistsError(f"Refusing to reuse existing run: {target}")
    print(f"Student: {weights}\nTraining data: {data}\nRun: {target}\nEpochs: {args.epochs}; batch: {args.batch}; imgsz: {args.imgsz}")
    if args.dry_run:
        print("Dry run only; no training started.")
        return

    from dataset_crack.webapp.engine import register_checkpoint_modules

    register_checkpoint_modules()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, optimizer="AdamW", lr0=0.0005,
        lrf=0.01, cos_lr=True, patience=15, mosaic=0.0, mixup=0.0,
        seed=0, deterministic=True, project=str(project), name=run_name,
        exist_ok=False, plots=True,
    )
    target = Path(model.trainer.save_dir).resolve()
    provenance = {
        "method": "RF-DETR50 high-confidence hard-target response distillation; original GT retained",
        "teacher_dataset": str(data), "student_initial_weights": str(weights),
        "source_v5": str(source_root), "epochs_requested": args.epochs,
        "batch": args.batch, "imgsz": args.imgsz, "seed": 0,
        "test_split_used_in_training": False,
    }
    (target / "distillation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Training complete. Best checkpoint: {target / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    freeze_support()
    main()
