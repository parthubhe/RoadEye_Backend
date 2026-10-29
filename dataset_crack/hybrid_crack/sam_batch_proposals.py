r"""Generate *unreviewed* SAM3.1 candidates for queued crack-positive images.

Loads the local SAM3.1 checkpoint once per batch. Existing YOLO boxes are
prompts, not trusted pixel labels. Output is never marked approved automatically.
Run with the ComfyUI Python environment, not the RoadEye training venv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

from PIL import Image

from dataset_crack.segment_crack_sam31 import (
    DEFAULT_CHECKPOINT, DEFAULT_SAM_CODE, load_crack_boxes, save_results,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "hybrid_crack_masks" / "review_queue.csv")
    parser.add_argument("--dataset", type=Path, default=ROOT / "final_dataset_v5")
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--max-images", type=int, default=10, help="Start with a small pilot; process more in later runs")
    parser.add_argument("--out", type=Path, default=ROOT / "hybrid_crack_masks" / "proposals")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam-code", type=Path, default=DEFAULT_SAM_CODE)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--with-text", action="store_true", help="Also test the short 'pipe wall crack' prompt")
    args = parser.parse_args()
    if args.max_images < 1 or not 1 <= args.top_n <= 10:
        parser.error("max-images must be positive and top-n 1..10")
    manifest = args.manifest.resolve(strict=True)
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    pending = [row for row in rows if row["split"] == args.split and row["kind"] == "positive"
               and row["status"].strip().lower() == "pending"]
    output_root = args.out.resolve() / args.split
    targets = []
    for row in pending:
        image_path = args.dataset.resolve() / args.split / "images" / row["image"]
        output = output_root / image_path.stem
        if output.exists():
            if not (output / "summary.json").is_file():
                raise RuntimeError(f"Incomplete existing proposal folder; inspect manually: {output}")
            continue  # non-overwriting; never change previously generated proposals
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        targets.append((image_path, output))
        if len(targets) == args.max_images:
            break
    if not targets:
        print("No pending positive rows without an existing proposal directory")
        return

    sys.path.insert(0, str(args.sam_code.resolve(strict=True)))
    import torch
    from safetensors import safe_open
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM3.1 candidate generation")
    model = build_sam3_image_model(device="cuda", eval_mode=True,
                                   checkpoint_path=str(args.checkpoint.resolve(strict=True)),
                                   load_from_HF=False, enable_segmentation=True,
                                   enable_inst_interactivity=False, compile=False)
    with safe_open(str(args.checkpoint.resolve()), framework="pt", device="cpu") as source:
        loaded = {key.removeprefix("detector.") for key in source.keys() if key.startswith("detector.")}
    coverage = len(loaded & set(model.state_dict())) / max(1, len(model.state_dict()))
    if coverage < 0.90:
        raise RuntimeError(f"SAM checkpoint key coverage {coverage:.1%} is too low")
    processor = Sam3Processor(model, resolution=1008, device="cuda",
                              confidence_threshold=args.threshold)
    for number, (image_path, output) in enumerate(targets, 1):
        image = Image.open(image_path).convert("RGB")
        labels = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        boxes = load_crack_boxes(labels)
        if not boxes:
            print(f"Skipping {image_path.name}: no original crack boxes")
            continue
        output.mkdir(parents=True, exist_ok=False)
        reports = []
        for box_index, box in enumerate(boxes):
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = processor.set_image(image)
                state = processor.add_boxes_prompts([box], [True], state)
                torch.cuda.synchronize()
            reports.append(save_results(image, boxes, state, output / f"box_{box_index}",
                                        args.top_n, 0.75, time.perf_counter() - started))
            del state
        if args.with_text:
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = processor.set_image(image)
                state = processor.set_text_prompt("pipe wall crack", state)
                torch.cuda.synchronize()
            reports.append(save_results(image, boxes, state, output / "text_pipe_wall_crack",
                                        args.top_n, 0.75, time.perf_counter() - started))
            del state
        (output / "summary.json").write_text(json.dumps({"image": str(image_path),
            "boxes": boxes, "experiments": [report["experiment"] for report in reports],
            "review_status": "UNREVIEWED — never use these candidates directly as ground truth"}, indent=2),
            encoding="utf-8")
        print(f"[{number}/{len(targets)}] {image_path.name}: {len(boxes)} box prompts -> {output}", flush=True)
        if number % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
