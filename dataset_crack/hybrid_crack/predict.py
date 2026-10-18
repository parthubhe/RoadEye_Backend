"""Run unified box + crack-mask inference on one local image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.nn import functional as F

from .model import load_hybrid

NAMES = ["crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Non-existing output overlay PNG")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--box-conf", type=float, default=0.25)
    parser.add_argument("--mask-conf", type=float, default=0.50)
    parser.add_argument("--show-crack-boxes", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    image_path, out = args.image.resolve(strict=True), args.out.resolve()
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite inference output: {out}")
    if args.imgsz % 32 or not (0 <= args.box_conf <= 1 and 0 <= args.mask_conf <= 1):
        parser.error("imgsz must be multiple of 32; thresholds must be in [0,1]")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = load_hybrid(str(args.checkpoint.resolve(strict=True)), device)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ratio = min(args.imgsz / width, args.imgsz / height)
    resized = round(width * ratio), round(height * ratio)
    left, top = (args.imgsz - resized[0]) // 2, (args.imgsz - resized[1]) // 2
    padded = ImageOps.expand(image.resize(resized, Image.Resampling.BILINEAR),
                             (left, top, args.imgsz - resized[0] - left, args.imgsz - resized[1] - top),
                             fill=(114, 114, 114))
    tensor = torch.from_numpy(np.asarray(padded).copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    with torch.inference_mode():
        detections, mask_logits = model(tensor)
        detections = detections[0] if isinstance(detections, tuple) else detections
        mask_prob = F.interpolate(mask_logits.float(), size=(args.imgsz, args.imgsz),
                                  mode="bilinear", align_corners=False)[0, 0].sigmoid().cpu().numpy()
    mask_crop = mask_prob[top:top + resized[1], left:left + resized[0]]
    original_mask = Image.fromarray((mask_crop * 255).astype(np.uint8), "L").resize((width, height), Image.Resampling.BILINEAR)
    binary_mask = np.asarray(original_mask) >= round(args.mask_conf * 255)
    rgb = np.asarray(image).copy()
    rgb[binary_mask] = (0.5 * rgb[binary_mask] + 0.5 * np.array([255, 83, 55])).astype(np.uint8)
    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    boxes = []
    for x1, y1, x2, y2, confidence, class_id in detections[0].detach().cpu().tolist():
        class_id = int(class_id)
        if confidence < args.box_conf or class_id not in range(5):
            continue
        if class_id == 0 and not args.show_crack_boxes:
            continue
        coordinates = [max(0, min(width, (x1 - left) / ratio)), max(0, min(height, (y1 - top) / ratio)),
                       max(0, min(width, (x2 - left) / ratio)), max(0, min(height, (y2 - top) / ratio))]
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        draw.rectangle(coordinates, outline="#38d8ff", width=3)
        draw.text((coordinates[0], max(0, coordinates[1] - 14)), f"{NAMES[class_id]} {confidence:.2f}",
                  fill="#ffffff", stroke_width=2, stroke_fill="#06303c")
        boxes.append({"class": NAMES[class_id], "confidence": round(confidence, 4), "xyxy": coordinates})
    out.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out)
    out.with_suffix(".json").write_text(json.dumps({"image": str(image_path), "checkpoint": str(args.checkpoint),
                                                    "boxes": boxes, "crack_mask_pixels": int(binary_mask.sum()),
                                                    "mask_threshold": args.mask_conf}, indent=2), encoding="utf-8")
    print(f"Saved {out}; {len(boxes)} shown boxes and {int(binary_mask.sum())} crack-mask pixels")


if __name__ == "__main__":
    main()
