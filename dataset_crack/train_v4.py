"""Train the plain pretrained YOLO26n baseline on the pipe-CCTV v4 dataset.

The default data file is ``final_dataset_v4/data.yaml``.  It deliberately
starts a fresh YOLO26n run; do not resume a v3 checkpoint after changing the
class taxonomy.

Examples:

    python train_v4.py
    python train_v4.py --epochs 150 --batch 12 --imgsz 640
    python train_v4.py --data closeup_crack_v4/data.yaml --name yolo26n_closeup_v4
"""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "final_dataset_v4" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0, help="CUDA device, e.g. 0; use cpu only for diagnostics")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--name", default="yolo26n_v4_pipe_cctv")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "pipe_proto")
    parser.add_argument("--cls-pw", type=float, default=0.25, help="rare-class weighting; set 0 for an unweighted ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"v4 data YAML not found: {data_path}\n"
            "Run `python build_v4_datasets.py --dataset pipe_cctv` first."
        )

    # Keep the model explicitly at Nano for the first v4 ablation.  The
    # rebuilt dataset, not extra model capacity, is what this run is testing.
    model = YOLO("yolo26s.pt")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        cls_pw=args.cls_pw,
        patience=40,
        close_mosaic=10,
        device=args.device,
        workers=args.workers,
        seed=42,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        plots=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
