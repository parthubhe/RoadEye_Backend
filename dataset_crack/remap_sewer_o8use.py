"""
Collapses the raw 54-class sewer.v2-sw2-2.2k Roboflow export (YOLO format)
down to a 5-class taxonomy: crack, corrosion_rust, sediment_deposit,
root_intrusion, joint_defect. Everything else (BX/ZW/QF/TL/CR/AJ/CQ/SL) is
dropped from the labels entirely -- images are kept either way, they just
won't have those boxes anymore.

Expects the standard Roboflow YOLO export structure:
    <input_dir>/data.yaml
    <input_dir>/train/images, <input_dir>/train/labels
    <input_dir>/valid/images, <input_dir>/valid/labels
    <input_dir>/test/images,  <input_dir>/test/labels

Usage:
    python remap_sewer_o8use.py <input_dir> <output_dir>
"""
import argparse
import shutil
from pathlib import Path
import yaml

BASE_MAP = {
    "PL": "crack",
    "FS": "corrosion_rust",
    "CJ": "sediment_deposit",
    "JG": "sediment_deposit",
    "SG": "root_intrusion",
    "TJ": "joint_defect",
    "CK": "joint_defect",
}
NEW_CLASSES = ["crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"]
NEW_ID = {name: i for i, name in enumerate(NEW_CLASSES)}


def old_id_to_new_id(old_names):
    """old_names: list of the 54 raw class names, index-ordered from data.yaml"""
    mapping = {}
    for old_id, name in enumerate(old_names):
        base = name.split("_")[0]
        new_name = BASE_MAP.get(base)
        mapping[old_id] = NEW_ID[new_name] if new_name else None
    return mapping


def find_dir(input_dir: Path, split: str, subfolder: str):
    """Find <split>/<subfolder> anywhere under input_dir, regardless of nesting depth."""
    matches = list(input_dir.rglob(f"{split}/{subfolder}"))
    return matches[0] if matches else None


def remap_split(input_dir: Path, output_dir: Path, split: str, id_map: dict):
    img_in = find_dir(input_dir, split, "images")
    lbl_in = find_dir(input_dir, split, "labels")
    if img_in is None or lbl_in is None:
        print(f"  [{split}] could not find images/labels folders anywhere under {input_dir} -- skipping")
        return 0, 0
    print(f"  [{split}] found images: {img_in}")
    print(f"  [{split}] found labels: {lbl_in}")
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_images, n_boxes = 0, 0
    for lbl_path in lbl_in.glob("*.txt"):
        new_lines = []
        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_id = id_map.get(old_id)
            if new_id is None:
                continue  # dropped class
            new_lines.append(" ".join([str(new_id)] + parts[1:]))
            n_boxes += 1

        (lbl_out / lbl_path.name).write_text("\n".join(new_lines))

        # match the image regardless of extension
        stem = lbl_path.stem
        matches = list(img_in.glob(f"{stem}.*"))
        if matches:
            shutil.copy2(matches[0], img_out / matches[0].name)
            n_images += 1

    return n_images, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    args = ap.parse_args()

    yaml_matches = list(args.input_dir.rglob("data.yaml"))
    if not yaml_matches:
        raise FileNotFoundError(f"No data.yaml found anywhere under {args.input_dir}")
    print(f"Using data.yaml: {yaml_matches[0]}")
    with open(yaml_matches[0]) as f:
        data_yaml = yaml.safe_load(f)
    old_names = data_yaml["names"]
    id_map = old_id_to_new_id(old_names)

    dropped = sorted({old_names[i].split("_")[0] for i, v in id_map.items() if v is None})
    print(f"Keeping: {list(BASE_MAP.keys())} -> {NEW_CLASSES}")
    print(f"Dropping: {dropped}")

    total_imgs, total_boxes = 0, 0
    for split in ["train", "valid", "test"]:
        n_i, n_b = remap_split(args.input_dir, args.output_dir, split, id_map)
        print(f"{split}: {n_i} images, {n_b} boxes kept")
        total_imgs += n_i
        total_boxes += n_b

    new_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }
    with open(args.output_dir / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f, default_flow_style=False)

    print(f"\nTotal: {total_imgs} images, {total_boxes} boxes across {len(NEW_CLASSES)} classes")
    print(f"Wrote {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()