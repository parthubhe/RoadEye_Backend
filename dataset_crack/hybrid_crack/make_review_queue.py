"""Create a non-overwriting, split-preserving SAM-assisted mask review queue.

Rows are candidates, never approved labels. Do not train until each selected
mask has been checked and its row status changed to approved.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data import IMAGE_SUFFIXES, dataset_paths, read_labels

ROOT = Path(__file__).resolve().parents[1]


def spread_by_source(paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    """Round-robin coarse filename families; original splits are unchanged."""
    if count > len(paths):
        raise ValueError(f"Requested {count} images from a pool of only {len(paths)}")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[path.stem.rsplit("_", 1)[0]].append(path)
    for group in groups.values():
        rng.shuffle(group)
    families = list(groups)
    rng.shuffle(families)
    result = []
    while len(result) < count:
        for family in families:
            if groups[family]:
                result.append(groups[family].pop())
                if len(result) == count:
                    break
    return result


def source_family(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def select_positives(paths: list[Path], count: int, sewage_fraction: float | None,
                     rng: random.Random) -> list[Path]:
    if sewage_fraction is None:
        return spread_by_source(paths, count, rng)
    sewage = [path for path in paths if source_family(path) == "sewage"]
    other = [path for path in paths if source_family(path) != "sewage"]
    sewage_count = round(count * sewage_fraction)
    return (spread_by_source(sewage, sewage_count, rng)
            + spread_by_source(other, count - sewage_count, rng))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "final_dataset_v5" / "data.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "hybrid_crack_masks" / "review_queue.csv")
    parser.add_argument("--train-positive", type=int, default=1400)
    parser.add_argument("--train-negative", type=int, default=200)
    parser.add_argument("--valid-positive", type=int, default=160)
    parser.add_argument("--valid-negative", type=int, default=40)
    parser.add_argument("--test-positive", type=int, default=160)
    parser.add_argument("--test-negative", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sewage-positive-fraction", type=float, default=None,
                        help="Fraction of positive selections from the sewage_* filename family; "
                             "e.g. 0.10. Does not change the original detection dataset or split.")
    parser.add_argument("--exclude-audit-manifest", type=Path,
                        help="Exclude images marked needs_label_audit in an existing manifest")
    args = parser.parse_args()
    if any(value < 0 for value in (args.train_positive, args.train_negative, args.valid_positive,
                                    args.valid_negative, args.test_positive, args.test_negative)):
        parser.error("Sample counts must be non-negative")
    if args.sewage_positive_fraction is not None and not 0 <= args.sewage_positive_fraction <= 1:
        parser.error("--sewage-positive-fraction must be between 0 and 1")
    excluded = set()
    if args.exclude_audit_manifest:
        with args.exclude_audit_manifest.resolve(strict=True).open(newline="", encoding="utf-8-sig") as file:
            excluded = {(row["split"], row["image"]) for row in csv.DictReader(file)
                        if row.get("status", "").strip().lower() == "needs_label_audit"}
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Review queue already exists; refusing to overwrite: {out}")
    rng = random.Random(args.seed)
    rows = []
    paths = dataset_paths(args.data.resolve())
    for split in ("train", "valid", "test"):
        positive, negative = [], []
        for path in sorted(paths[split].iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if (split, path.name) in excluded:
                continue
            (positive if (read_labels(path)[:, 0] == 0).any() else negative).append(path)
        pairs = (("positive", positive, getattr(args, f"{split}_positive")),
                 ("negative", negative, getattr(args, f"{split}_negative")))
        for kind, pool, count in pairs:
            selected = (select_positives(pool, count, args.sewage_positive_fraction, rng)
                        if kind == "positive" else spread_by_source(pool, count, rng))
            for path in selected:
                rows.append({"split": split, "image": path.name, "kind": kind,
                             "mask_path": f"masks/{split}/{path.stem}.png", "status": "pending"})
            by_source = dict(Counter(source_family(path) for path in selected))
            print(f"{split} {kind}: {len(selected)} selected; sources {by_source}")
        print(f"{split}: {len(positive)} crack-box positives, {len(negative)} candidates without crack boxes")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "image", "kind", "mask_path", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pending review entries: {out}")
    print("WARNING: 'negative' means no crack BOX is labeled; visually verify before approving a blank mask.")
    print("WARNING: the v5 split is inherited; this queue does not prove video/site-group independence.")


if __name__ == "__main__":
    main()
