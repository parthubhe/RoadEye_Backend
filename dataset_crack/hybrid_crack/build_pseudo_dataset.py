r"""Materialize the selected v5 subset with SAM3.1 crack pseudo-masks.

Run with the local ComfyUI Python after make_review_queue has written a NEW
2,000-row manifest. One SAM load serves the whole batch. Safe to rerun after
an interruption: completed masks are checked and never replaced.

These masks are automatically generated pseudo-labels, NOT verified ground
truth. YOLO boxes and original split membership are copied without editing v5.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_crack.hybrid_crack.data import dataset_paths
from dataset_crack.hybrid_crack.make_review_queue import source_family
from dataset_crack.segment_crack_sam31 import (
    DEFAULT_CHECKPOINT, DEFAULT_SAM_CODE, coordinates, draw_overlay, load_crack_boxes,
)


def save_manifest(path: Path, rows: list[dict], fields: list[str]) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", suffix=".csv",
                                     prefix=".manifest_", dir=path.parent, delete=False) as file:
        temp = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # On Windows, an editor/indexer/antivirus can briefly open the destination
    # without delete-sharing. The completed temp file is kept for every retry;
    # never regenerate a SAM mask merely because this rename was momentarily busy.
    for attempt in range(1, 31):
        try:
            os.replace(temp, path)
            if attempt > 1:
                print(f"Manifest update succeeded after {attempt} attempts", flush=True)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) != 5 or attempt == 30:
                raise PermissionError(
                    f"Could not update {path} after {attempt} attempts. "
                    f"The complete replacement is preserved at {temp}; close any "
                    "program holding manifest.csv, then rerun this command."
                ) from error
            if attempt == 1:
                print(f"Manifest briefly locked; retrying: {path}", flush=True)
            time.sleep(min(0.2 * attempt, 2.0))


def copy_checked(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.is_file():
            matching = filecmp.cmp(source, target, shallow=False)
        else:
            matching = target.stat().st_size == 0
        if not matching:
            raise RuntimeError(f"Existing copied file differs from source: {target}")
        return
    if source.is_file():
        shutil.copy2(source, target)
    else:
        target.touch(exist_ok=False)  # an unlabeled YOLO image has an empty label file

def validate_mask(path: Path, image_size: tuple[int, int], positive: bool) -> None:
    with Image.open(path) as image:
        if image.size != image_size or image.mode != "L":
            raise ValueError(f"Wrong mask size/mode at {path}")
        pixels = np.asarray(image)
    if not np.isin(pixels, (0, 255)).all() or bool(pixels.any()) != positive:
        raise ValueError(f"Invalid or empty-positive pseudo-mask: {path}")


def pick_candidate(state: dict, box: list[float], size: tuple[int, int], top_n: int,
                   min_inside: float, max_image_fraction: float) -> tuple[np.ndarray, dict]:
    width, height = size
    scores = state["scores"].detach().float().cpu().numpy().reshape(-1)
    masks = state["masks"].detach().cpu().numpy().reshape(-1, height, width) > 0
    if len(scores) == 0:
        raise RuntimeError("SAM returned no candidates for a crack box")
    x1, y1, x2, y2 = coordinates(box, width, height)
    ranked = np.argsort(scores)[::-1][:top_n]
    choices = []
    for index in ranked:
        mask = masks[index]
        area = int(mask.sum())
        if area == 0:
            continue
        inside = float(mask[y1:y2, x1:x2].sum() / area)
        fraction = area / (width * height)
        choices.append((int(index), float(scores[index]), inside, fraction))
    if not choices:
        raise RuntimeError("All top SAM candidates are empty")
    eligible = [item for item in choices if item[2] >= min_inside and item[3] <= max_image_fraction]
    selected = max(eligible or choices, key=lambda item: item[1] + 0.10 * item[2])
    index, score, inside, fraction = selected
    flags = []
    if inside < min_inside:
        flags.append("mask_mostly_outside_prompt_box")
    if fraction > max_image_fraction:
        flags.append("unusually_large_mask")
    return masks[index], {"candidate_index": index, "sam_score_not_accuracy": round(score, 5),
                           "mask_inside_box_fraction": round(inside, 5),
                           "image_mask_fraction": round(fraction, 5), "quality_flags": flags,
                           "top_n_considered": len(choices)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Fresh 2,000-row manifest generated by make_review_queue")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1]
                        / "final_dataset_v5" / "data.yaml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam-code", type=Path, default=DEFAULT_SAM_CODE)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min-inside", type=float, default=0.55)
    parser.add_argument("--max-image-fraction", type=float, default=0.20)
    parser.add_argument("--limit", type=int, help="Only process this many unfinished rows, for a pilot")
    parser.add_argument("--save-overlays", action="store_true", help="Save visual overlays as well as final masks")
    args = parser.parse_args()
    if not 1 <= args.top_n <= 10 or not 0 <= args.threshold < 1:
        parser.error("Expected top-n 1..10 and threshold in [0,1)")
    if not 0 <= args.min_inside <= 1 or not 0 < args.max_image_fraction <= 1:
        parser.error("Invalid geometry thresholds")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    manifest = args.manifest.resolve(strict=True)
    root = manifest.parent
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows, fields = list(reader), list(reader.fieldnames or [])
    required = {"split", "image", "kind", "mask_path", "status"}
    if not required.issubset(fields) or len(rows) != 2000:
        raise ValueError("Expected a fresh 2,000-row mask manifest with split/image/kind/mask_path/status")
    for field in ("source_family", "sam_selections_json", "quality_flags"):
        if field not in fields:
            fields.append(field)
    keys = [(row["split"], row["image"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate image in manifest")
    mask_names = [row["mask_path"] for row in rows]
    if len(mask_names) != len(set(mask_names)):
        raise ValueError("Two selected images would write to the same mask PNG")
    paths = dataset_paths(args.data.resolve(strict=True))
    checkpoint = args.checkpoint.resolve(strict=True)
    sam_code = args.sam_code.resolve(strict=True)
    signature = {"data": str(args.data.resolve()), "checkpoint": str(checkpoint),
                 "top_n": args.top_n, "threshold": args.threshold,
                 "min_inside": args.min_inside, "max_image_fraction": args.max_image_fraction}
    config_path = root / "sam_build_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError(f"Resuming with different SAM settings is unsafe: {config_path}")
    else:
        config_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    targets = [row for row in rows if row["status"].strip().lower() in {"pending", "failed"}]
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        print("No unfinished rows. The 2,000-image pseudo-mask dataset is already materialized.")
        return
    print(f"Processing {len(targets)} unfinished rows; {len(rows) - len(targets)} already completed", flush=True)

    # Import SAM only when a positive image actually needs inference.
    processor = torch = None
    started = time.perf_counter()
    for number, row in enumerate(targets, 1):
        split, name, kind = row["split"], row["image"], row["kind"]
        if split not in paths or kind not in {"positive", "negative"} or Path(name).name != name:
            raise ValueError(f"Invalid manifest row: {row}")
        source_image = paths[split] / name
        source_label = source_image.parent.parent / "labels" / f"{source_image.stem}.txt"
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        target_image = root / split / "images" / name
        target_label = root / split / "labels" / f"{source_image.stem}.txt"
        mask_path = (root / row["mask_path"]).resolve()
        if root not in mask_path.parents:
            raise ValueError(f"Mask target escapes output directory: {mask_path}")
        row["source_family"] = source_family(source_image)
        with Image.open(source_image) as file:
            image = file.convert("RGB")
        boxes = load_crack_boxes(source_label)
        if bool(boxes) != (kind == "positive"):
            raise ValueError(f"Manifest kind disagrees with original crack boxes: {source_image}")
        copy_checked(source_image, target_image)
        copy_checked(source_label, target_label)
        if mask_path.exists():
            validate_mask(mask_path, image.size, kind == "positive")
            row["quality_flags"] = row.get("quality_flags") or "existing_mask_verified_on_resume"
        else:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            selections = []
            union = np.zeros((image.height, image.width), dtype=bool)
            if kind == "positive":
                if processor is None:
                    sys.path.insert(0, str(sam_code))
                    import torch as torch_module
                    from safetensors import safe_open
                    from sam3.model_builder import build_sam3_image_model
                    from sam3.model.sam3_image_processor import Sam3Processor
                    torch = torch_module
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is required for local SAM3.1 pseudo-mask generation")
                    model = build_sam3_image_model(device="cuda", eval_mode=True,
                        checkpoint_path=str(checkpoint), load_from_HF=False,
                        enable_segmentation=True, enable_inst_interactivity=False, compile=False)
                    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
                        loaded = {key.removeprefix("detector.") for key in source.keys()
                                  if key.startswith("detector.")}
                    coverage = len(loaded & set(model.state_dict())) / max(1, len(model.state_dict()))
                    if coverage < 0.90:
                        raise RuntimeError(f"SAM checkpoint key coverage is too low: {coverage:.1%}")
                    processor = Sam3Processor(model, resolution=1008, device="cuda",
                                              confidence_threshold=args.threshold)
                    print(f"SAM loaded once; checkpoint key coverage {coverage:.1%}", flush=True)
                failure = None
                for box in boxes:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        state = processor.set_image(image)
                        state = processor.add_boxes_prompts([box], [True], state)
                    try:
                        selected, metadata = pick_candidate(state, box, image.size, args.top_n,
                            args.min_inside, args.max_image_fraction)
                    except RuntimeError as error:
                        failure = str(error)
                        del state
                        break
                    union |= selected
                    selections.append(metadata)
                    del state
                if failure or not union.any():
                    row["status"] = "failed"
                    row["quality_flags"] = failure or "no_nonempty_mask"
                    save_manifest(manifest, rows, fields)
                    print(f"[{number}/{len(targets)}] FAILED {split}/{name}: {row['quality_flags']}", flush=True)
                    continue  # no invented blank positive mask
            else:
                selections.append({"quality_flags": ["unverified_negative_no_crack_box"]})
            temp = mask_path.with_name(mask_path.stem + ".partial.png")
            Image.fromarray(union.astype(np.uint8) * 255, "L").save(temp)
            validate_mask(temp, image.size, kind == "positive")
            os.replace(temp, mask_path)
            row["sam_selections_json"] = json.dumps(selections, separators=(",", ":"))
            flags = [flag for item in selections for flag in item["quality_flags"]]
            row["quality_flags"] = ";".join(flags)
            if args.save_overlays:
                overlay = root / "overlays" / split / f"{source_image.stem}.jpg"
                overlay.parent.mkdir(parents=True, exist_ok=True)
                if not overlay.exists():
                    draw_overlay(image, union, boxes).save(overlay, quality=85)
        row["status"] = "auto_pseudo"
        save_manifest(manifest, rows, fields)
        if number % 25 == 0 or number == len(targets):
            if torch is not None:
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[{number}/{len(targets)}] {split}/{name}; elapsed {time.perf_counter() - started:.0f}s", flush=True)

    import yaml
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(yaml.safe_dump({"path": str(root), "train": "train/images",
            "val": "valid/images", "test": "test/images", "names": {
            0: "crack", 1: "corrosion_rust", 2: "sediment_deposit",
            3: "root_intrusion", 4: "joint_defect"}}, sort_keys=False), encoding="utf-8")
    print(f"Dataset: {root}")
    print(f"Manifest: {manifest}")
    print(f"Statuses: {dict(Counter(row['status'] for row in rows))}")
    print("WARNING: auto_pseudo masks have NOT been human-verified. Do not report mask metrics as ground truth.")


if __name__ == "__main__":
    main()
