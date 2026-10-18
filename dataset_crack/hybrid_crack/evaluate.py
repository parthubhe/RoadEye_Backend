"""Evaluate a hybrid checkpoint: full five-class box mAP plus reviewed mask Dice/IoU.

Box and mask metrics have different denominators and must be reported separately.
The test split should be run once after final model selection on validation data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from .data import HybridDataset, collate, dataset_paths, read_reviewed_masks
from .model import load_hybrid
from .train import ROOT, validate_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "final_dataset_v5" / "data.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "hybrid_crack_masks" / "review_queue.csv")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "pipe_proto")
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = args.checkpoint.resolve(strict=True)
    data_yaml = args.data.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    run_name = f"hybrid_{checkpoint.parent.name}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.project.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    reviewed = read_reviewed_masks(manifest)
    mask_set = HybridDataset(dataset_paths(data_yaml)[args.split], args.split, reviewed,
                             args.imgsz, only_reviewed=True)
    if mask_set.approved_count:
        loader = DataLoader(mask_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate)
        mask_metrics = validate_masks(load_hybrid(str(checkpoint), device), loader, device)
    else:
        mask_metrics = None

    detector_wrapper = YOLO("yolo26n.yaml")
    detector_wrapper.model = load_hybrid(str(checkpoint), device).detector
    detector_wrapper.model.names = {0: "crack", 1: "corrosion_rust", 2: "sediment_deposit",
                                    3: "root_intrusion", 4: "joint_defect"}
    box = detector_wrapper.val(data=str(data_yaml), split="val" if args.split == "valid" else "test",
                               imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                               device=str(device), half=False, plots=False, verbose=False,
                               project=str(run_dir), name="boxes", exist_ok=False)
    result = {
        "checkpoint": str(checkpoint), "data": str(data_yaml), "split": args.split,
        "box_full_split": {"precision": float(box.box.mp), "recall": float(box.box.mr),
                           "map50": float(box.box.map50), "map50_95": float(box.box.map)},
        "mask_reviewed_subset": mask_metrics,
        "warning": "Mask Dice/IoU and box AP are different metrics; compare only on identical held-out splits.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
