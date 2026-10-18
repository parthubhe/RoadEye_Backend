r"""Compare SAM3.1 crack prompts and top-N masks on one image.

PowerShell, from RoadEye:
  & 'C:\Users\parth\Documents\ComfyUI\.venv\Scripts\python.exe' dataset_crack\segment_crack_sam31.py --top-n 4

Loads the existing local checkpoint; does not download anything. Saves each
candidate and contact sheets for visual review. Outputs are NOT ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "final_dataset_v5/test/images/sewage_000004.jpg"
DEFAULT_CHECKPOINT = Path(r"D:\ComfyUI_Assets\Models\checkpoints\sam3.1_multiplex_fp16.safetensors")
DEFAULT_SAM_CODE = Path(r"C:\Users\parth\Documents\ComfyUI\custom_nodes\comfyui-easy-sam3")
DEFAULT_PROMPTS = ("crack", "pipe wall crack", "all cracks present in the image", "fracture in concrete")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--labels", type=Path, help="YOLO labels; inferred from image path by default")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--sam-code", type=Path, default=DEFAULT_SAM_CODE)
    p.add_argument("--prompt", action="append", dest="prompts", help="Repeat for custom text prompts")
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--dedup-iou", type=float, default=0.75)
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--no-text-prompts", action="store_true")
    p.add_argument("--no-box-prompts", action="store_true")
    p.add_argument("--pixel-box", action="append", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Repeat to test a tighter manual box in image pixels")
    p.add_argument("--assemble-mask", action="append", type=Path,
                   help="Repeat with visually selected candidate masks to combine them without rerunning SAM")
    p.add_argument("--output-root", type=Path, default=ROOT / "sam31_preview")
    args = p.parse_args()
    if not 1 <= args.top_n <= 20 or not 0 <= args.threshold < 1 or not 0 <= args.dedup_iou <= 1:
        p.error("Expected top-n 1..20 and threshold/dedup-iou in [0,1]")
    args.prompts = args.prompts or list(DEFAULT_PROMPTS)
    return args


def load_crack_boxes(path: Path) -> list[list[float]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "0":
            result.append([float(value) for value in parts[1:5]])
    return result


def coordinates(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    return (max(0, int((cx - bw / 2) * width)), max(0, int((cy - bh / 2) * height)),
            min(width, int((cx + bw / 2) * width)), min(height, int((cy + bh / 2) * height)))


def draw_overlay(image: Image.Image, mask: np.ndarray, boxes: list[list[float]]) -> Image.Image:
    base = np.asarray(image, dtype=np.uint8)
    color = base.copy()
    color[mask] = (0.45 * base[mask] + 0.55 * np.array([255, 72, 30])).astype(np.uint8)
    shown = Image.fromarray(color)
    draw = ImageDraw.Draw(shown)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = coordinates(box, image.width, image.height)
        draw.rectangle((x1, y1, x2, y2), outline="#ffdf3d", width=2)
        draw.text((x1 + 3, y1 + 3), str(i), fill="#ffdf3d", stroke_width=1, stroke_fill="black")
    return shown


def iou(a: np.ndarray, b: np.ndarray) -> float:
    both = np.logical_and(a, b).sum()
    either = np.logical_or(a, b).sum()
    return float(both / either) if either else 0.0


def coverage(mask: np.ndarray, boxes: list[list[float]]) -> list[float]:
    """Fraction of each YOLO box covered by a mask, NOT mask accuracy."""
    h, w = mask.shape
    values = []
    for box in boxes:
        x1, y1, x2, y2 = coordinates(box, w, h)
        values.append(round(float(mask[y1:y2, x1:x2].sum() / max(1, (x2 - x1) * (y2 - y1))), 4))
    return values


def make_sheet(tiles: list[tuple[str, Image.Image]], path: Path) -> None:
    w, h = tiles[0][1].size
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    bar = 28
    sheet = Image.new("RGB", (columns * w, rows * (h + bar)), "#f5f4f0")
    draw = ImageDraw.Draw(sheet)
    for i, (caption, tile) in enumerate(tiles):
        x, y = (i % columns) * w, (i // columns) * (h + bar)
        draw.text((x + 7, y + 7), caption[:90], fill="#102c36")
        sheet.paste(tile, (x, y + bar))
    sheet.save(path)


def save_results(image: Image.Image, boxes: list[list[float]], state: dict,
                 target: Path, top_n: int, dedup_iou: float, elapsed: float) -> dict:
    target.mkdir(parents=True, exist_ok=False)
    scores = state["scores"].detach().float().cpu().numpy().reshape(-1)
    masks = state["masks"].detach().cpu().numpy().reshape(-1, image.height, image.width).astype(bool)
    pred_boxes = state["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
    ranked = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    top = ranked[:top_n]
    distinct = []
    for index in ranked:
        if all(iou(masks[index], masks[previous]) < dedup_iou for previous in distinct):
            distinct.append(index)
            if len(distinct) == top_n:
                break
    empty = np.zeros((image.height, image.width), dtype=bool)
    tiles = [("Original; yellow = existing crack boxes", draw_overlay(image, empty, boxes))]
    candidates = []
    for rank, index in enumerate(top, 1):
        mask = masks[index]
        name = f"top{rank:02d}_idx{index:03d}"
        mask_file, overlay_file = target / f"{name}_mask.png", target / f"{name}_overlay.png"
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(mask_file)
        picture = draw_overlay(image, mask, boxes)
        picture.save(overlay_file)
        tiles.append((f"Top {rank}: score {scores[index]:.3f}, {mask.sum():,} px", picture))
        candidates.append({
            "rank": rank, "index": index, "score": round(float(scores[index]), 5),
            "mask_pixels": int(mask.sum()), "predicted_bbox_xyxy": pred_boxes[index].round(1).tolist(),
            "box_coverage_not_accuracy": coverage(mask, boxes),
            "mask": str(mask_file), "overlay": str(overlay_file),
        })
    unions = {}
    for name, indices in (("top_n_union", top), ("distinct_top_n_union", distinct)):
        mask = np.logical_or.reduce(masks[indices]) if indices else empty.copy()
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(target / f"{name}_mask.png")
        picture = draw_overlay(image, mask, boxes)
        picture.save(target / f"{name}_overlay.png")
        tiles.append((f"{name}: {mask.sum():,} px", picture))
        unions[name] = {"indices": indices, "mask_pixels": int(mask.sum()),
                        "box_coverage_not_accuracy": coverage(mask, boxes)}
    sheet = target / "contact_sheet.png"
    make_sheet(tiles, sheet)
    report = {"experiment": target.name, "seconds": round(elapsed, 2),
              "detections_above_threshold": len(scores), "top_n": top_n,
              "candidates": candidates, "unions": unions, "contact_sheet": str(sheet)}
    (target / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{target.name}: {len(scores)} candidates, top-{len(top)} union {unions['top_n_union']['mask_pixels']} px, {elapsed:.2f}s", flush=True)
    return report


def main() -> None:
    args = arguments()
    image_path = args.image.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    sam_code = args.sam_code.resolve(strict=True)
    labels = (args.labels or image_path.parent.parent / "labels" / f"{image_path.stem}.txt").resolve()
    image = Image.open(image_path).convert("RGB")
    boxes = load_crack_boxes(labels)
    manual_boxes = []
    for pixel_box in args.pixel_box or []:
        x1, y1, x2, y2 = pixel_box
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError(f"Invalid --pixel-box {pixel_box} for image {image.size}")
        manual_boxes.append([(x1 + x2) / (2 * image.width), (y1 + y2) / (2 * image.height),
                             (x2 - x1) / image.width, (y2 - y1) / image.height])
    output = args.output_root.resolve() / f"{image_path.stem}_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir(parents=True, exist_ok=False)
    if args.assemble_mask:
        selected = []
        for path in args.assemble_mask:
            path = path.resolve(strict=True)
            candidate = np.asarray(Image.open(path).convert("L")) > 0
            if candidate.shape != (image.height, image.width):
                raise ValueError(f"Mask size {candidate.shape} does not match image {image.size}: {path}")
            selected.append((path, candidate))
        combined = np.logical_or.reduce([mask for _, mask in selected])
        mask_file = output / "assembled_mask.png"
        overlay_file = output / "assembled_overlay.png"
        Image.fromarray(combined.astype(np.uint8) * 255, "L").save(mask_file)
        draw_overlay(image, combined, boxes).save(overlay_file)
        (output / "assembled_report.json").write_text(json.dumps({
            "image": str(image_path), "selected_masks": [str(path) for path, _ in selected],
            "mask_pixels": int(combined.sum()), "box_coverage_not_accuracy": coverage(combined, boxes),
            "warning": "Visually selected proposal only; not a verified ground-truth mask.",
        }, indent=2), encoding="utf-8")
        print(f"Assembled {len(selected)} masks: {overlay_file}", flush=True)
        return

    sys.path.insert(0, str(sam_code))
    import torch
    from safetensors import safe_open
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this preview")
    started = time.perf_counter()
    model = build_sam3_image_model(device="cuda", eval_mode=True, checkpoint_path=str(checkpoint),
                                   load_from_HF=False, enable_segmentation=True,
                                   enable_inst_interactivity=False, compile=False)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        loaded_keys = {key.removeprefix("detector.") for key in source.keys() if key.startswith("detector.")}
    model_keys = set(model.state_dict())
    key_coverage = len(loaded_keys & model_keys) / max(1, len(model_keys))
    if key_coverage < 0.90:
        raise RuntimeError(f"Checkpoint coverage {key_coverage:.1%} is too low")
    load_seconds = time.perf_counter() - started
    processor = Sam3Processor(model, resolution=args.resolution, device="cuda",
                              confidence_threshold=args.threshold)
    torch.cuda.reset_peak_memory_stats()
    print(f"Model loaded in {load_seconds:.1f}s; {key_coverage:.1%} key coverage; {len(boxes)} YOLO crack boxes", flush=True)
    reports = []

    def experiment(name: str, prompt: str | None = None, prompt_boxes: list[list[float]] | None = None) -> None:
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = processor.set_image(image)
            if prompt is not None:
                state = processor.set_text_prompt(prompt, state)
            if prompt_boxes:
                state = processor.add_boxes_prompts(prompt_boxes, [True] * len(prompt_boxes), state)
            torch.cuda.synchronize()
        result = save_results(image, boxes, state, output / name, args.top_n, args.dedup_iou,
                              time.perf_counter() - start)
        result["prompt"] = prompt
        result["prompt_boxes_xywh"] = prompt_boxes
        reports.append(result)
        del state

    if not args.no_text_prompts:
        for number, prompt in enumerate(args.prompts, 1):
            safe_name = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:45]
            experiment(f"text_{number}_{safe_name}", prompt=prompt)
    if boxes and not args.no_box_prompts:
        for number, box in enumerate(boxes):
            experiment(f"box_{number}", prompt_boxes=[box])
        experiment("text_crack_plus_all_boxes", prompt="crack", prompt_boxes=boxes)
    for number, box in enumerate(manual_boxes):
        experiment(f"manual_box_{number}", prompt_boxes=[box])
    summary = {"image": str(image_path), "labels": str(labels), "checkpoint": str(checkpoint),
               "sam_code": str(sam_code), "crack_boxes_xywh": boxes, "manual_boxes_xywh": manual_boxes,
               "top_n": args.top_n, "threshold": args.threshold, "dedup_iou": args.dedup_iou,
               "resolution": args.resolution, "load_seconds": round(load_seconds, 2),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
               "note": "Scores are model confidence, not mask correctness. Unions require manual review.",
               "experiments": reports}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(reports)} experiments: {output}", flush=True)


if __name__ == "__main__":
    main()
