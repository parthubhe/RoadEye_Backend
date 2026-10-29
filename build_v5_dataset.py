"""Build the v5 comparability dataset using the exact v2 split strategy.

v5 intentionally mirrors ``final_merge.py``: source-prefixed output names,
randomized group-level split, 15% validation, 15% test, and train-only
oversampling of ``sewer_o8use`` by four.  The only source-level change is the
exclusion of the two close-up domains documented in ``v5_taxonomy.yaml``.

The remapped input sources are not modified.  Empty labels are retained so
this experiment does not repeat v4's aggressive empty/unmapped-image filter.

Run from any directory:

    python dataset_crack/build_v5_dataset.py
    python dataset_crack/build_v5_dataset.py --seed 42 --image-mode copy
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
# Import the exact v2 helper functions regardless of whether this script is
# launched from the repository root or as ``python dataset_crack/...``.
sys.path.insert(0, str(ROOT / "dataset_crack"))
from final_merge import collect_pairs, group_key  # noqa: E402


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "v5_taxonomy.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_crack" / "final_dataset_v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--image-mode", choices=("copy", "hardlink"), default="hardlink")
    return parser.parse_args()


def source_pairs(source_dir: Path) -> list[tuple[Path, Path | None]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Configured source does not exist: {source_dir}")
    pairs = collect_pairs(source_dir, flat=False)
    if not pairs:
        raise RuntimeError(f"Configured source contains no images: {source_dir}")
    return pairs


def materialize_image(source: Path, destination: Path, mode: str) -> str:
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {config_path}")
    classes = list(config["classes"])
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    if not 0 < args.val_frac < 1 or not 0 < args.test_frac < 1 or args.val_frac + args.test_frac >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")

    all_items: list[tuple[str, Path, Path | None]] = []
    repeats: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for source in config["sources"]:
        source_id = str(source["id"])
        source_dir = (config_path.parent / source["path"]).resolve()
        pairs = source_pairs(source_dir)
        all_items.extend((source_id, image, label) for image, label in pairs)
        repeats[source_id] = int(source.get("repeat_train", 1))
        source_counts[source_id] = len(pairs)
        print(f"{source_id}: {len(pairs)} images")

    groups: dict[tuple[str, str], list[tuple[str, Path, Path | None]]] = defaultdict(list)
    for item in all_items:
        source_id, image_path, _ = item
        groups[(source_id, group_key(image_path))].append(item)

    rng = random.Random(args.seed)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    n_test = int(len(group_keys) * args.test_frac)
    n_valid = int(len(group_keys) * args.val_frac)
    split_groups = {
        "test": group_keys[:n_test],
        "valid": group_keys[n_test : n_test + n_valid],
        "train": group_keys[n_test + n_valid :],
    }

    assignments: dict[str, list[tuple[str, Path, Path | None]]] = {}
    for split, keys in split_groups.items():
        items = [item for key in keys for item in groups[key]]
        if split == "train":
            items = [item for item in items for _ in range(repeats[item[0]])]
        assignments[split] = items

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    manifest_rows: list[dict[str, str | int]] = []
    for split in SPLITS:
        image_dir = output_dir / split / "images"
        label_dir = output_dir / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, (source_id, image_path, label_path) in enumerate(assignments[split]):
            stem = f"{source_id}_{index:06d}"
            image_out = image_dir / f"{stem}{image_path.suffix.lower()}"
            label_out = label_dir / f"{stem}.txt"
            materialization = materialize_image(image_path, image_out, args.image_mode)
            if label_path is not None and label_path.is_file():
                shutil.copy2(label_path, label_out)
            else:
                label_out.write_text("", encoding="utf-8")
            counts[split] += 1
            manifest_rows.append(
                {
                    "split": split,
                    "output_image": str(image_out.relative_to(output_dir)),
                    "output_label": str(label_out.relative_to(output_dir)),
                    "source_id": source_id,
                    "source_image": str(image_path),
                    "source_label": str(label_path) if label_path else "",
                    "group_id": f"{source_id}:{group_key(image_path)}",
                    "materialization": materialization,
                }
            )

    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(classes),
        "names": {index: name for index, name in enumerate(classes)},
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False, allow_unicode=True)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "version": 5,
        "split_strategy": "v2 final_merge.py: random group split, source-specific groups",
        "seed": args.seed,
        "validation_fraction": args.val_frac,
        "test_fraction": args.test_frac,
        "classes": classes,
        "source_images_before_train_repeat": source_counts,
        "train_repeat": repeats,
        "images_by_split": dict(counts),
        "groups_by_split": {split: len(split_groups[split]) for split in SPLITS},
        "output_dir": str(output_dir),
        "image_mode": args.image_mode,
        "excluded_sources": config.get("excluded_sources", []),
        "note": "Empty labels retained; only close-up domain sources excluded.",
    }
    with (output_dir / "build_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)

    print(f"\nGrouped {len(all_items)} source images into {len(groups)} groups")
    print(f"Final split: train={counts['train']} valid={counts['valid']} test={counts['test']}")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
