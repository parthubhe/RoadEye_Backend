r"""Render generated crack masks on their source images for manual review.

The crack mask is magenta. Existing YOLO boxes are drawn in class-specific
colors, including cyan for joint_defect. No labels or masks are changed.

Run from RoadEye:
  venv\Scripts\python.exe -m dataset_crack.hybrid_crack.render_mask_overlays
"""

from __future__ import annotations

import argparse
import csv
import html
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import read_labels
from .sync_removed_images import ACTIVE_STATUSES, DEFAULT_SUBSET, load_manifest

CLASS_NAMES = ("crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect")
CLASS_COLORS = ("#ffe066", "#ff974c", "#79d78a", "#c6a3ff", "#51dbff")
MASK_COLOR = np.array([255, 25, 125], dtype=np.float32)


def box_pixels(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = map(float, box)
    return (max(0, round((cx - bw / 2) * width)), max(0, round((cy - bh / 2) * height)),
            min(width, round((cx + bw / 2) * width)), min(height, round((cy + bh / 2) * height)))


def render(image_path: Path, mask_path: Path, row: dict) -> tuple[Image.Image, dict]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    with Image.open(mask_path) as source:
        if source.size != image.size or source.mode not in {"L", "1", "P"}:
            raise ValueError("Mask size/mode does not match image")
        mask_values = np.asarray(source.convert("L"))
    if not np.isin(mask_values, (0, 1, 255)).all():
        raise ValueError("Mask must be binary")
    mask = mask_values > 0
    labels = read_labels(image_path)
    expected_positive = row["kind"].strip().lower() == "positive"
    if mask.any() != expected_positive:
        raise ValueError("Mask positivity disagrees with manifest kind")
    width, height = image.size
    pixels = np.asarray(image, dtype=np.uint8).copy()
    if mask.any():
        pixels[mask] = (0.30 * pixels[mask] + 0.70 * MASK_COLOR).astype(np.uint8)
    annotated = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    joint_region = np.zeros(mask.shape, dtype=bool)
    for label in labels:
        class_id = int(label[0])
        x1, y1, x2, y2 = box_pixels(label[1:5], width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        if class_id == 4:
            joint_region[y1:y2, x1:x2] = True
        color = CLASS_COLORS[class_id]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=3)
        caption = CLASS_NAMES[class_id]
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        top = max(0, y1 - text_height - 7)
        draw.rectangle((x1, top, min(width - 1, x1 + text_width + 8), top + text_height + 6), fill="#112c38")
        draw.text((x1 + 4, top + 3), caption, fill=color, font=font)
    joint_fraction = float((mask & joint_region).sum() / max(1, mask.sum())) if mask.any() else 0.0
    header_height = 55
    canvas = Image.new("RGB", (width, height + header_height), "#102b36")
    canvas.paste(annotated, (0, header_height))
    head = ImageDraw.Draw(canvas)
    family = row.get("source_family") or image_path.stem.rsplit("_", 1)[0]
    head.text((10, 8), f"{row['split']}/{row['image']}  |  {family}  |  {row['kind']}",
              fill="white", font=font)
    summary = (f"CRACK MASK: {int(mask.sum()):,} px (magenta)  |  "
               f"mask inside joint boxes: {joint_fraction:.1%}"
               if mask.any() else "BLANK CRACK MASK / NEGATIVE  |  existing class boxes still shown")
    head.text((10, 29), summary, fill="#ff9dc6" if mask.any() else "#a8dce8", font=font)
    return canvas, {"mask_pixels": int(mask.sum()), "joint_overlap_fraction": round(joint_fraction, 5),
                    "image_width": width, "image_height": height,
                    "box_classes": ";".join(CLASS_NAMES[int(item[0])] for item in labels)}


def write_index(output: Path, records: list[dict], errors: list[dict]) -> None:
    fields = ("split", "image", "kind", "source_family", "mask_pixels",
              "joint_overlap_fraction", "image_width", "image_height", "box_classes", "overlay")
    with (output / "review_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    if errors:
        with (output / "render_errors.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("split", "image", "error"))
            writer.writeheader()
            writer.writerows(errors)
    cards = []
    for record in sorted(records, key=lambda item: item["joint_overlap_fraction"], reverse=True):
        title = html.escape(f"{record['split']}/{record['image']}")
        overlay = html.escape(record["overlay"], quote=True)
        family = html.escape(record["source_family"])
        joint = float(record["joint_overlap_fraction"])
        cards.append(f'<article class="card" data-search="{title.lower()} {family.lower()}" '
                     f'data-split="{record["split"]}" data-kind="{record["kind"]}" data-joint="{joint}">'
                     f'<a href="{overlay}" target="_blank"><img loading="lazy" src="{overlay}" alt="{title}"></a>'
                     f'<div><strong>{title}</strong><span>{family} · {record["kind"]} · joint overlap {joint:.1%}</span></div></article>')
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PipeScout crack-mask review</title><style>
body{font:15px system-ui,sans-serif;margin:0;background:#edf2f2;color:#18303a}header{padding:20px 28px;background:#102b36;color:white;position:sticky;top:0;z-index:1}h1{font-size:22px;margin:0 0 6px}p{margin:0 0 12px;color:#b7ccd2}input,select,label{font:inherit;margin-right:12px;padding:7px}.controls{display:flex;flex-wrap:wrap;gap:7px;align-items:center}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px;padding:20px}.card{background:white;border:1px solid #cbd9dc}.card img{width:100%;height:260px;object-fit:contain;background:#102b36}.card div{padding:10px}.card strong{display:block;font-size:13px;overflow-wrap:anywhere}.card span{display:block;color:#526b75;font-size:12px;margin-top:4px}
</style><header><h1>Crack-mask review</h1><p>Magenta = generated crack mask. Yellow = crack box. Cyan = joint-defect box. Joint overlap is a review hint, not proof of an error. Click an image to inspect it full-size.</p>
<div class="controls"><input id="search" placeholder="Search filename or source"><select id="split"><option value="">All splits</option><option>train</option><option>valid</option><option>test</option></select><select id="kind"><option value="">All masks</option><option>positive</option><option>negative</option></select><label><input type="checkbox" id="joint"> Joint-box overlap ≥ 10%</label><span id="count"></span></div></header><main>""" + "\n".join(cards) + """</main><script>
const cards=[...document.querySelectorAll('.card')];const s=document.getElementById('search'),sp=document.getElementById('split'),k=document.getElementById('kind'),j=document.getElementById('joint'),n=document.getElementById('count');
function update(){let shown=0;for(const c of cards){const ok=c.dataset.search.includes(s.value.toLowerCase())&&(!sp.value||c.dataset.split===sp.value)&&(!k.value||c.dataset.kind===k.value)&&(!j.checked||Number(c.dataset.joint)>=.1);c.hidden=!ok;if(ok)shown++}n.textContent=shown+' / '+cards.length+' images'}
for(const control of [s,sp,k,j])control.addEventListener('input',update);update();</script></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--out", type=Path, help="New output folder; defaults to a timestamped folder inside the subset")
    parser.add_argument("--limit", type=int, help="Render only the first N eligible images for a pilot")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.root.resolve(strict=True)
    output = (args.out or root / f"review_overlays_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing overlays: {output}")
    rows, _ = load_manifest(root)
    active = [row for row in rows if row["status"].strip().lower() in ACTIVE_STATUSES]
    if args.limit:
        active = active[:args.limit]
    output.mkdir(parents=True, exist_ok=False)
    records, errors = [], []
    for number, row in enumerate(active, 1):
        split, name = row["split"], row["image"]
        image_path = root / split / "images" / name
        mask_path = root / row["mask_path"]
        if not image_path.is_file() or not mask_path.is_file():
            errors.append({"split": split, "image": name, "error": "generated image or mask is missing"})
            continue
        try:
            annotated, metrics = render(image_path, mask_path, row)
            relative = Path(split) / f"{Path(name).stem}_overlay.jpg"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(target, quality=90, subsampling=0)
            records.append({"split": split, "image": name, "kind": row["kind"],
                            "source_family": row.get("source_family") or Path(name).stem.rsplit("_", 1)[0],
                            **metrics, "overlay": relative.as_posix()})
        except Exception as error:
            errors.append({"split": split, "image": name, "error": str(error)})
        if number % 100 == 0 or number == len(active):
            print(f"[{number}/{len(active)}] rendered={len(records)}, errors={len(errors)}", flush=True)
    write_index(output, records, errors)
    print(f"Review page: {output / 'index.html'}")
    print(f"Annotated images: {len(records)}; skipped/errors: {len(errors)}")


if __name__ == "__main__":
    main()
