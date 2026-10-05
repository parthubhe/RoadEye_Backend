"""Train the v5 five-class CCTV comparison model with YOLO26s or YOLO26m.

The default settings intentionally match the saved v2 run (80 epochs,
640px, AdamW, cosine LR, seed 0, no class weighting) so the data-domain
change is measurable.  Choose ``--model m`` only as a capacity ablation.
"""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "final_dataset_v5" / "data.yaml")
    parser.add_argument("--model", choices=("s", "m"), default="s")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "pipe_proto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing v5 YAML: {data_path}. Run build_v5_dataset.py first.")
    model_name = f"yolo26{args.model}.pt"
    run_name = args.name or f"yolo26{args.model}_v5_cctv"
    model = YOLO(model_name)
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
        project=str(args.project.resolve()),
        name=run_name,
        plots=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
