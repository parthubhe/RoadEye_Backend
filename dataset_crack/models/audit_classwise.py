r"""Score candidate YOLO checkpoints on their matching v5 validation image domain.

Run from the RoadEye root:
    venv\Scripts\python.exe dataset_crack\models\audit_classwise.py

This audit selects class-specialist release candidates on validation, not test.
It writes a new evidence directory and never changes checkpoint or dataset files.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_crack.webapp.engine import register_checkpoint_modules


ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "pipe_proto"
NOTEBOOK_RUNS = ROOT / "notebook" / "runs"
CANDIDATES = {
    "baseline": (RUNS / "yolo26s_v5_cctv/weights/best.pt", ROOT / "final_dataset_v5/data.yaml"),
    "simam_clahe": (RUNS / "yolo26s_v5_clahe_simam/weights/best.pt", ROOT / "final_dataset_v5_clahe/data.yaml"),
    "cbam_clahe": (RUNS / "yolo26s_v5_clahe_cbam/weights/best.pt", ROOT / "final_dataset_v5_clahe/data.yaml"),
    "se_clahe": (RUNS / "yolo26s_v5_clahe_se/weights/best.pt", ROOT / "final_dataset_v5_clahe/data.yaml"),
    "parallel": (RUNS / "yolo26s_v5_parallel_simam_cbam_se/weights/best.pt", ROOT / "final_dataset_v5/data.yaml"),
    "p2_weighted_fusion": (MODELS / "p2_weighted_fusion/best.pt", ROOT / "final_dataset_v5/data.yaml"),
    "directional_e5": (NOTEBOOK_RUNS / "E5_directional_v5_normal_seed0_20260928_185759_471e27/weights/best.pt", ROOT / "final_dataset_v5/data.yaml"),
}


def main() -> None:
    config_dir = MODELS / "evidence" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    from ultralytics import YOLO

    register_checkpoint_modules()
    for name, (checkpoint, data) in CANDIDATES.items():
        if not checkpoint.is_file() or not data.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint or dataset YAML: {checkpoint}, {data}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    audit = MODELS / "evidence" / f"classwise_validation_{stamp}"
    audit.mkdir(parents=True, exist_ok=False)
    results = {"split": "val", "selection_basis": "per-class AP50 on original v5 validation image IDs",
               "models": {}}
    for name, (checkpoint, data) in CANDIDATES.items():
        print(f"Evaluating {name} on {data.parent.name} validation", flush=True)
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(data), split="val", imgsz=640, batch=8, workers=0,
            device=0 if torch.cuda.is_available() else "cpu", plots=False,
            project=str(audit), name=name, exist_ok=False, verbose=False,
        )
        row = {
            "checkpoint": str(checkpoint), "data": str(data),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "per_class": {
                str(model.names[int(class_id)]): {
                    "AP50": float(metrics.box.ap50[index]),
                    "AP50_95": float(metrics.box.ap[index]),
                    "precision": float(metrics.box.p[index]),
                    "recall": float(metrics.box.r[index]),
                }
                for index, class_id in enumerate(metrics.box.ap_class_index)
            },
        }
        results["models"][name] = row
        print(f"{name}: mAP50={row['mAP50']:.4f}, mAP50-95={row['mAP50_95']:.4f}", flush=True)
        (audit / "classwise_validation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Validation evidence: {audit / 'classwise_validation.json'}", flush=True)


if __name__ == "__main__":
    main()
