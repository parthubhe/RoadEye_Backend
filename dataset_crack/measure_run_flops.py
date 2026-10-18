"""Measure YOLO checkpoint parameters and approximate GFLOPs for the run report.

Only load checkpoints created by this project: YOLO .pt loading uses pickle.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from dataset_crack.webapp.engine import register_checkpoint_modules


ROOT = Path(__file__).resolve().parent
RUN_ROOTS = (ROOT / "runs" / "pipe_proto", ROOT / "runs" / "detect" / "runs" / "pipe_proto", ROOT / "notebook" / "runs")
OUTPUT = ROOT / "run_flops_measurements.json"


def main() -> None:
    register_checkpoint_modules()
    from ultralytics.utils.torch_utils import get_flops

    records = {}
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for run in sorted(root.iterdir()):
            if not run.is_dir() or not (run / "results.csv").is_file():
                continue
            checkpoint = run / "weights" / "best.pt"
            if not checkpoint.is_file():
                continue
            key = run.relative_to(ROOT).as_posix()
            try:
                args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8")) if (run / "args.yaml").is_file() else {}
                imgsz = int(args.get("imgsz", 640))
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = saved.get("ema") or saved.get("model")
                model = model.float().eval()
                records[key] = {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "gflops": round(float(get_flops(model, imgsz=imgsz)), 3),
                    "imgsz": imgsz,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "method": "Ultralytics get_flops (THOP), nominal forward pass at saved imgsz",
                }
                print(f"{key}: {records[key]['gflops']} GFLOPs")
            except Exception as exc:
                records[key] = {"error": str(exc), "checkpoint_bytes": checkpoint.stat().st_size}
                print(f"{key}: ERROR {exc}")
            finally:
                if "saved" in locals():
                    del saved
                if "model" in locals():
                    del model
    OUTPUT.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
