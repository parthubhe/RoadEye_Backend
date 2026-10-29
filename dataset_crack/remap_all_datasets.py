"""
Reads dataset_classes.yaml and remaps every listed source dataset's raw
classes onto the unified 5-class taxonomy, writing each to its own output
folder in consistent train/valid/test + images/labels structure.

Usage:
    python remap_all_datasets.py <extracted_datasets_root> <output_root> --config dataset_classes.yaml

Expects <extracted_datasets_root>/<dataset_key>/ to contain, anywhere at
any nesting depth, a data.yaml plus train/valid/test image+label folders --
exactly what extracting a Roboflow YOLO export zip gives you.
"""
import argparse
from pathlib import Path
import shutil
import yaml

UNIFIED_CLASSES = ["crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"]
UNIFIED_ID = {name: i for i, name in enumerate(UNIFIED_CLASSES)}


def find_dir(root, split, subfolder):
    matches = list(root.rglob(f"{split}/{subfolder}"))
    return matches[0] if matches else None


def find_yaml(root):
    matches = list(root.rglob("data.yaml"))
    return matches[0] if matches else None


def resolve_class(raw_name, mapping):
    """Exact match first, then strip a trailing _N severity suffix (e.g. PL_3 -> PL)."""
    if raw_name in mapping:
        return mapping[raw_name]
    base = raw_name.rsplit("_", 1)[0]
    if base in mapping:
        return mapping[base]
    return None  # not listed at all -> drop


def build_id_map(raw_names, mapping):
    id_map = {}
    for old_id, name in enumerate(raw_names):
        unified_name = resolve_class(name, mapping)
        id_map[old_id] = UNIFIED_ID[unified_name] if unified_name else None
    return id_map


def remap_split(dataset_root, output_root, split, id_map):
    img_in = find_dir(dataset_root, split, "images")
    lbl_in = find_dir(dataset_root, split, "labels")
    if img_in is None or lbl_in is None:
        return 0, 0
    img_out = output_root / split / "images"
    lbl_out = output_root / split / "labels"
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
                continue
            new_lines.append(" ".join([str(new_id)] + parts[1:]))
            n_boxes += 1
        (lbl_out / lbl_path.name).write_text("\n".join(new_lines))

        stem = lbl_path.stem
        matches = list(img_in.glob(f"{stem}.*"))
        if matches:
            shutil.copy2(matches[0], img_out / matches[0].name)
            n_images += 1
    return n_images, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets_root", type=Path)
    ap.add_argument("output_root", type=Path)
    ap.add_argument("--config", type=Path, default=Path("dataset_classes.yaml"))
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for dataset_key, dataset_cfg in config["datasets"].items():
        dataset_dir = args.datasets_root / dataset_key
        if not dataset_dir.exists():
            print(f"[{dataset_key}] SKIPPED -- not found at {dataset_dir}")
            continue

        yaml_path = find_yaml(dataset_dir)
        if yaml_path is None:
            print(f"[{dataset_key}] SKIPPED -- no data.yaml found under {dataset_dir}")
            continue

        with open(yaml_path) as f:
            raw_names = yaml.safe_load(f)["names"]

        mapping = dataset_cfg["map"]
        id_map = build_id_map(raw_names, mapping)
        kept = sum(1 for v in id_map.values() if v is not None)
        print(f"\n[{dataset_key}] {kept}/{len(raw_names)} raw classes kept (using {yaml_path})")

        out_dir = args.output_root / dataset_key
        total_imgs, total_boxes = 0, 0
        for split in ["train", "valid", "test"]:
            n_i, n_b = remap_split(dataset_dir, out_dir, split, id_map)
            if n_i:
                print(f"  {split}: {n_i} images, {n_b} boxes")
            total_imgs += n_i
            total_boxes += n_b
        print(f"  TOTAL: {total_imgs} images, {total_boxes} boxes")


if __name__ == "__main__":
    main()
