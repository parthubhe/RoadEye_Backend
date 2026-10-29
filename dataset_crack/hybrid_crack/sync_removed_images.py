r"""Reconcile manually deleted masks, images, or annotated review overlays.

Delete unwanted files from the latest full review_overlays_*/<split>/ folder,
from <subset>/masks/<split>/, or from <subset>/<split>/images/. This command
finds those missing files, moves their remaining generated image, label, and
mask into a recoverable .rejected/ folder, and marks the manifest row excluded.
It never edits the original final_dataset_v5 images, labels, or split.

Run from the RoadEye root:
  venv\Scripts\python.exe -m dataset_crack.hybrid_crack.sync_removed_images --apply

Omit --apply to preview exactly which rows would be changed.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .build_pseudo_dataset import save_manifest
from .data import IMAGE_SUFFIXES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = ROOT / "hybrid_crack_masks" / "v5_2k_sam_pseudo_sewage10"
EXCLUDED_STATUS = "excluded_by_manual_image_removal"  # retained for earlier image-first reviews
EXCLUDED_MASK_STATUS = "excluded_by_manual_mask_removal"
EXCLUDED_OVERLAY_STATUS = "excluded_by_manual_overlay_removal"
ACTIVE_STATUSES = {"auto_pseudo", "approved"}


def load_manifest(root: Path) -> tuple[list[dict], list[str]]:
    manifest = root / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows, fields = list(reader), list(reader.fieldnames or [])
    if not {"split", "image", "mask_path", "status"}.issubset(fields):
        raise ValueError(f"Invalid mask manifest: {manifest}")
    if any(row["status"].strip().lower() in {"pending", "failed"} for row in rows):
        raise RuntimeError("Dataset generation is incomplete; finish it before syncing manual removals")
    seen = set()
    for row in rows:
        split, name = row["split"].strip(), row["image"].strip()
        if split not in {"train", "valid", "test"} or Path(name).name != name or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsafe split or image name in manifest: {split}/{name}")
        key = split, name
        if key in seen:
            raise ValueError(f"Duplicate image in manifest: {key}")
        seen.add(key)
        expected_mask = Path("masks") / split / f"{Path(name).stem}.png"
        if Path(row["mask_path"]) != expected_mask:
            raise ValueError(f"Unexpected mask path for {split}/{name}: {row['mask_path']}")
    return rows, fields


def review_overlays(root: Path, rows: list[dict], explicit: Path | None = None) -> tuple[Path | None, dict]:
    """Use only a full review export; never treat the small pilot as a rejection list."""
    active = {(row["split"], row["image"]) for row in rows
              if row["status"].strip().lower() in ACTIVE_STATUSES}
    candidates = ([explicit.resolve(strict=True)] if explicit else
                  sorted(root.glob("review_overlays_*/review_index.csv"), reverse=True))
    for index in candidates:
        if index.name != "review_index.csv" or index.parent.parent.resolve() != root.resolve():
            raise ValueError(f"Review index must be directly inside this generated subset: {index}")
        with index.open(newline="", encoding="utf-8-sig") as file:
            records = list(csv.DictReader(file))
        lookup = {}
        for item in records:
            split, name = item.get("split", ""), item.get("image", "")
            expected = Path(split) / f"{Path(name).stem}_overlay.jpg"
            relative = Path(item.get("overlay", ""))
            if split not in {"train", "valid", "test"} or Path(name).name != name or relative != expected:
                raise ValueError(f"Unsafe or inconsistent review-index row: {item}")
            key = split, name
            if key in lookup:
                raise ValueError(f"Duplicate review-index row: {key}")
            lookup[key] = relative
        if active.issubset(lookup):
            return index.parent, lookup
        if explicit:
            raise ValueError(f"Review index does not cover all active manifest rows: {index}")
    return None, {}


def find_missing(root: Path, rows: list[dict], kind: str = "all",
                 review_dir: Path | None = None, review_lookup: dict | None = None) -> list[tuple[dict, str]]:
    missing = []
    review_lookup = review_lookup or {}
    for row in rows:
        if row["status"].strip().lower() not in ACTIVE_STATUSES:
            continue
        if kind != "all" and row.get("kind", "").strip().lower() != kind:
            continue
        image = root / row["split"] / "images" / row["image"]
        mask = root / row["mask_path"]
        if not mask.is_file():
            missing.append((row, "mask"))
        elif not image.is_file():
            missing.append((row, "image"))
        elif review_dir is not None and not (review_dir / review_lookup[row["split"], row["image"]]).is_file():
            missing.append((row, "review_overlay"))
    return missing


def artifact_pairs(root: Path, row: dict, review_dir: Path | None = None,
                   review_lookup: dict | None = None) -> list[tuple[Path, Path]]:
    split, stem = row["split"], Path(row["image"]).stem
    quarantine = root / ".rejected"
    pairs = [
        (root / split / "images" / row["image"],
         quarantine / split / "images" / row["image"]),
        (root / split / "labels" / f"{stem}.txt",
         quarantine / split / "labels" / f"{stem}.txt"),
        (root / "masks" / split / f"{stem}.png",
         quarantine / "masks" / split / f"{stem}.png"),
    ]
    if review_dir is not None:
        relative = review_lookup[row["split"], row["image"]]
        pairs.append((review_dir / relative,
                      quarantine / "review_overlays" / review_dir.name / relative))
    return pairs


def sync(root: Path, apply: bool = False, allow_large_batch: bool = False,
         kind: str = "all", review_index: Path | None = None,
         no_review_overlays: bool = False) -> list[tuple[str, str]]:
    rows, fields = load_manifest(root)
    for split in ("train", "valid", "test"):
        if not (root / "masks" / split).is_dir() or not (root / split / "images").is_dir():
            raise RuntimeError(f"Expected active image and mask folders for {split}; refusing to treat an entire missing folder as manual review")
    if kind not in {"all", "positive", "negative"}:
        raise ValueError(f"Unknown kind: {kind}")
    if review_index and no_review_overlays:
        raise ValueError("Choose either --review-index or --no-review-overlays")
    review_dir, review_lookup = ((None, {}) if no_review_overlays else
                                 review_overlays(root, rows, explicit=review_index))
    if review_dir:
        print(f"Scanning full annotated review export: {review_dir}")
    missing = find_missing(root, rows, kind=kind, review_dir=review_dir,
                           review_lookup=review_lookup)
    selections = [(row["split"], row["image"]) for row, _ in missing]
    print(f"Scanned {len(rows)} manifest rows ({kind}); {len(missing)} manually removed mask/image pair(s) found.")
    for row, reason in missing[:30]:
        print(f"  {row['split']}/{row['image']} (missing {reason})")
    if len(selections) > 30:
        print(f"  ... plus {len(selections) - 30} more")
    if not missing or not apply:
        if missing:
            print("Dry run only. Rerun with --apply to exclude these rows and quarantine matching files.")
        return selections
    if len(missing) > 200 and not allow_large_batch:
        raise RuntimeError(f"{len(missing)} missing rows is unusually large; inspect the folders first. "
                           "Pass --allow-large-batch only if this many deletions were intentional.")

    # Check every target before touching anything. A destination already in use
    # could belong to a different review pass, so never overwrite it.
    for row, _ in missing:
        for source, target in artifact_pairs(root, row, review_dir, review_lookup):
            if source.is_symlink() or target.is_symlink() or (source.exists() and target.exists()):
                raise RuntimeError(f"Conflicting or symlinked artifact; inspect manually: {source} -> {target}")

    quarantine = root / ".rejected"
    backup_dir = quarantine / "manifest_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"manifest_{stamp}_{uuid.uuid4().hex[:8]}.csv"
    shutil.copy2(root / "manifest.csv", backup)
    for row, reason in missing:
        for source, target in artifact_pairs(root, row, review_dir, review_lookup):
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
        row["status"] = (EXCLUDED_MASK_STATUS if reason == "mask" else
                         EXCLUDED_OVERLAY_STATUS if reason == "review_overlay" else EXCLUDED_STATUS)
    save_manifest(root / "manifest.csv", rows, fields)
    print(f"Excluded {len(missing)} row(s); remaining generated image/label/mask files moved into {quarantine}")
    print(f"Manifest backup: {backup}")
    print("Original final_dataset_v5 was not modified. Full-v5 hybrid training skips excluded TRAIN images.")
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--apply", action="store_true", help="Perform the cleanup; otherwise preview only")
    parser.add_argument("--allow-large-batch", action="store_true", help="Permit more than 200 exclusions in one pass")
    parser.add_argument("--kind", choices=("all", "positive", "negative"), default="all",
                        help="Restrict cleanup to a mask kind; useful when deciding what to do with blank negatives")
    parser.add_argument("--review-index", type=Path,
                        help="Use this full review_index.csv instead of auto-detecting the latest full export")
    parser.add_argument("--no-review-overlays", action="store_true",
                        help="Ignore overlay deletions; scan only main masks and images")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    allowed_parent = (ROOT / "hybrid_crack_masks").resolve(strict=True)
    if root.parent != allowed_parent:
        parser.error(f"Refusing a root outside a single generated subset under {allowed_parent}")
    sync(root, apply=args.apply, allow_large_batch=args.allow_large_batch, kind=args.kind,
         review_index=args.review_index, no_review_overlays=args.no_review_overlays)


if __name__ == "__main__":
    main()
