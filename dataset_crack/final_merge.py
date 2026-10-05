"""
Pools every remapped source into one final dataset -- fresh shuffled
train/valid/test split at the GROUP level (near-duplicate images always
land together, never split across sets), source-prefixed filenames so
nothing collides.

Usage:
    python final_merge.py <output_dir> \
        --sources name1=path1 name2=path2 ... \
        --flat-sources name3=path3 ... \
        --repeat name1=4
"""
import argparse
import random
import re
import shutil
from pathlib import Path
import yaml

UNIFIED_CLASSES = ["crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"]


def group_key(img_path):
    """Extract a grouping identifier so near-duplicate images always land in
    the same split together, never split across train/valid/test.

    Handles two real patterns found in these datasets:
      1. Roboflow's universal ".rf.<hash>" suffix -- multiple files sharing
         the part before this are Roboflow's own augmented copies of the
         exact same source image.
      2. "<videoID>_f<frameNumber>_jpg" and "<videoID>-mp4-t-<timestamp>"
         -- video-derived frames, grouped by video ID so nearby frames of
         the same clip don't split across sets either.

    Falls back to the de-hashed base name for anything matching neither
    pattern -- still fixes the Roboflow-duplicate case even then.
    """
    stem = img_path.stem
    base = stem.split(".rf.")[0] if ".rf." in stem else stem

    if "-mp4-t-" in base:
        return base.split("-mp4-t-")[0]

    match = re.match(r"^(.+)_f\d+_jpg$", base)
    if match:
        return match.group(1)

    return base


def collect_pairs(source_dir, flat=False):
    pairs = []
    if flat:
        splits = [(source_dir / "images", source_dir / "labels")]
    else:
        splits = []
        for split in ["train", "valid", "test"]:
            img_dir = source_dir / split / "images"
            lbl_dir = source_dir / split / "labels"
            if img_dir.exists():
                splits.append((img_dir, lbl_dir))

    for img_dir, lbl_dir in splits:
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            pairs.append((img_path, lbl_path if lbl_path.exists() else None))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--sources", nargs="*", default=[],
                     help="name=path pairs, each with train/valid/test/images,labels")
    ap.add_argument("--flat-sources", nargs="*", default=[],
                     help="name=path pairs with just a flat images/labels pair")
    ap.add_argument("--repeat", nargs="*", default=[],
                     help="name=count pairs -- include this source's images N times, "
                          "to rebalance classes that only exist in a small source "
                          "against classes flooded by much larger ones")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    random.seed(args.seed)
    repeat_counts = {}
    for entry in args.repeat:
        name, count = entry.split("=", 1)
        repeat_counts[name] = int(count)

    # Step 1: collect UNIQUE pairs only -- no repeats applied yet
    unique_pairs = []
    for entry in args.sources:
        name, path = entry.split("=", 1)
        pairs = collect_pairs(Path(path), flat=False)
        print(f"{name}: {len(pairs)} unique images")
        unique_pairs += [(name, i, l) for i, l in pairs]
    for entry in args.flat_sources:
        name, path = entry.split("=", 1)
        pairs = collect_pairs(Path(path), flat=True)
        print(f"{name} (flat): {len(pairs)} unique images")
        unique_pairs += [(name, i, l) for i, l in pairs]

    # Step 2: group by near-duplicate cluster, then shuffle+split at the
    # GROUP level -- guarantees near-duplicate frames/augmented copies never
    # end up split across train/valid/test.
    groups = {}
    for item in unique_pairs:
        source_name, img_path, lbl_path = item
        key = (source_name, group_key(img_path))
        groups.setdefault(key, []).append(item)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    n_groups = len(group_keys)
    n_test_groups = int(n_groups * args.test_frac)
    n_val_groups = int(n_groups * args.val_frac)

    test_items = [item for k in group_keys[:n_test_groups] for item in groups[k]]
    val_items = [item for k in group_keys[n_test_groups:n_test_groups + n_val_groups] for item in groups[k]]
    train_items = [item for k in group_keys[n_test_groups + n_val_groups:] for item in groups[k]]

    print(f"\nGrouped {len(unique_pairs)} images into {n_groups} near-duplicate-safe groups "
          f"(avg {len(unique_pairs) / n_groups:.1f} images/group)")

    # Step 3: apply repeats ONLY within train. Valid/test never see a
    # duplicate of anything, so there is no leakage regardless of repeat factor.
    train_assignment = []
    for item in train_items:
        source_name = item[0]
        n_repeat = repeat_counts.get(source_name, 1)
        for _ in range(n_repeat):
            train_assignment.append(item)

    print(f"After oversampling: train={len(train_assignment)} (from {len(train_items)} unique), "
          f"valid={len(val_items)} (unique, no repeats), test={len(test_items)} (unique, no repeats)")

    counts = {"train": 0, "valid": 0, "test": 0}
    for split, items in [("test", test_items), ("valid", val_items), ("train", train_assignment)]:
        img_out_dir = args.output_dir / split / "images"
        lbl_out_dir = args.output_dir / split / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        for source_name, img_path, lbl_path in items:
            new_name = f"{source_name}_{counts[split]:06d}"
            shutil.copy2(img_path, img_out_dir / f"{new_name}{img_path.suffix}")
            if lbl_path and lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out_dir / f"{new_name}.txt")
            else:
                (lbl_out_dir / f"{new_name}.txt").touch()
            counts[split] += 1

    print(f"\nFinal split: train={counts['train']} valid={counts['valid']} test={counts['test']}")

    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    with open(args.output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"Wrote {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()