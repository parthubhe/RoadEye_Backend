"""Count parameters in trusted local checkpoints and refresh the UI cache.

This unpickles YOLO .pt files. Run only for checkpoints created by this project.
"""

from __future__ import annotations

import gc
import json

import torch

from .engine import register_checkpoint_modules
from .registry import PARAMETERS_FILE, discover_models


def main() -> None:
    register_checkpoint_modules()
    counts = {}
    for identity, spec in discover_models().items():
        if not spec.checkpoint or not spec.checkpoint.is_file():
            continue
        try:
            if spec.family == "RF-DETR Nano":
                checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
                # RF-DETR's serialized model map contains the same parameter
                # count as its instantiated nn.Module for these Nano runs.
                count = sum(tensor.numel() for tensor in checkpoint["model"].values())
            else:
                checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
                model = checkpoint.get("ema") or checkpoint.get("model")
                count = sum(parameter.numel() for parameter in model.parameters())
            counts[identity] = count
            print(f"{spec.label}: {count:,}")
        except Exception as exc:
            print(f"SKIP {spec.label}: {exc}")
        finally:
            if "checkpoint" in locals():
                del checkpoint
            if "model" in locals():
                del model
            gc.collect()
    PARAMETERS_FILE.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(counts)} counts to {PARAMETERS_FILE}")


if __name__ == "__main__":
    main()
