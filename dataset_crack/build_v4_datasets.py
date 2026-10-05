"""Build the domain-separated RoadEye crack/pipe datasets for version 4.

The input datasets are never modified.  The builder creates:

* ``final_dataset_v4``: in-pipe CCTV defects, with 11 explicit classes.
* ``closeup_crack_v4``: close-up/surface defects, with 3 explicit classes.

Both outputs use a deterministic, group-disjoint split.  Frames from the
same video, Roboflow variants of one source image, and matching original stems
from different sources are kept in the same split.  The generated manifest and
report make every output example traceable to its source image and label.

Run from ``dataset_crack`` or any directory:

    python build_v4_datasets.py --dry-run
    python build_v4_datasets.py

Use ``--dataset pipe_cctv`` if only the model-training dataset is needed.
Outputs are refused when the target directory is non-empty; this prevents
accidentally overwriting a previous experiment.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "valid", "test")


@dataclass(frozen=True)
class Sample:
    """One source image together with its already-remapped label rows."""

    source_id: str
    image_path: Path
    label_path: Path | None
    group_id: str
    mapped_lines: tuple[str, ...]
    raw_boxes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("v4_taxonomy.yaml"),
        help="taxonomy YAML; defaults to dataset_crack/v4_taxonomy.yaml",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "pipe_cctv", "closeup_crack"),
        default="all",
        help="which configured dataset to build",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="directory in which output_dir values from the YAML are created",
    )
    parser.add_argument("--seed", type=int, default=42, help="split seed")
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink images to save space (default), or make full copies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="audit, map, and split data without creating any files",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def source_group_key(image_path: Path) -> str:
    """Return a source-agnostic key for video/image near-duplicate groups.

    The Roboflow ``.rf.<hash>`` suffix denotes alternate versions of the same
    source image.  Video-derived frames are grouped by video/inspection ID,
    not individual frame, so adjacent frames cannot leak across splits.
    """

    stem = image_path.stem
    base = stem.split(".rf.", 1)[0]

    if "-mp4-t-" in base:
        return f"video:{base.split('-mp4-t-', 1)[0]}"

    match = re.match(r"^(.+)_f\d+_jpg$", base)
    if match:
        return f"video:{match.group(1)}"

    return f"image:{base}"


def normalize_class_map(source_cfg: dict[str, Any], class_to_id: dict[str, int]) -> dict[int, int]:
    """Convert YAML raw-ID -> class-name mapping to raw-ID -> output-ID."""

    raw_mapping = source_cfg.get("class_map")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError(f"Source {source_cfg.get('id')!r} has no class_map")

    result: dict[int, int] = {}
    for raw_id, target_name in raw_mapping.items():
        try:
            raw_int = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid raw class ID {raw_id!r} in source {source_cfg.get('id')!r}") from exc
        if target_name not in class_to_id:
            raise ValueError(
                f"Source {source_cfg.get('id')!r} maps raw class {raw_int} to unknown class {target_name!r}"
            )
        result[raw_int] = class_to_id[target_name]
    return result


def remap_label(
    label_path: Path | None,
    raw_to_output: dict[int, int],
    on_unmapped: str,
) -> tuple[tuple[str, ...], int, Counter[int]]:
    """Remap the class ID while preserving YOLO box or polygon coordinates."""

    if label_path is None or not label_path.exists():
        return (), 0, Counter()

    mapped_lines: list[str] = []
    unmapped = Counter()
    raw_boxes = 0

    for number, original_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = original_line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Malformed YOLO label ({len(parts)} fields) at {label_path}:{number}")
        try:
            raw_class = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Non-integer class ID at {label_path}:{number}: {parts[0]!r}") from exc

        # Validate numeric coordinates early.  Polygon labels are preserved;
        # Ultralytics can use them to derive boxes for a detect model.
        try:
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric coordinate at {label_path}:{number}") from exc
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"Out-of-range normalized coordinate at {label_path}:{number}")

        raw_boxes += 1
        output_class = raw_to_output.get(raw_class)
        if output_class is None:
            unmapped[raw_class] += 1
            if on_unmapped == "error":
                raise ValueError(
                    f"Unmapped raw class {raw_class} at {label_path}:{number}. "
                    "Add it to v4_taxonomy.yaml or intentionally exclude the source."
                )
            continue

        mapped_lines.append(" ".join([str(output_class), *parts[1:]]))

    return tuple(mapped_lines), raw_boxes, unmapped


def collect_source_samples(
    source_cfg: dict[str, Any],
    config_root: Path,
    class_to_id: dict[str, int],
) -> tuple[list[Sample], dict[str, Any]]:
    """Read all original splits from one configured source into one pool."""

    source_id = str(source_cfg["id"])
    source_root = (config_root / source_cfg["path"]).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source {source_id!r} does not exist: {source_root}")

    raw_to_output = normalize_class_map(source_cfg, class_to_id)
    include_empty = bool(source_cfg.get("include_empty_labels", True))
    on_unmapped = str(source_cfg.get("on_unmapped", "error"))
    if on_unmapped not in {"error", "drop"}:
        raise ValueError(f"Source {source_id!r} has invalid on_unmapped={on_unmapped!r}")

    samples: list[Sample] = []
    report: dict[str, Any] = {
        "source_path": str(source_root),
        "images_seen": 0,
        "images_kept": 0,
        "images_skipped_empty_after_mapping": 0,
        "missing_labels": 0,
        "raw_boxes": 0,
        "mapped_boxes": 0,
        "unmapped_boxes": Counter(),
    }

    for original_split in SPLIT_NAMES:
        image_dir = source_root / original_split / "images"
        label_dir = source_root / original_split / "labels"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory for {source_id!r}: {image_dir}")

        for image_path in sorted(image_dir.iterdir(), key=lambda path: path.name.lower()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            report["images_seen"] += 1
            candidate_label = label_dir / f"{image_path.stem}.txt"
            label_path = candidate_label if candidate_label.is_file() else None
            if label_path is None:
                report["missing_labels"] += 1

            mapped_lines, raw_boxes, unmapped = remap_label(label_path, raw_to_output, on_unmapped)
            report["raw_boxes"] += raw_boxes
            report["mapped_boxes"] += len(mapped_lines)
            report["unmapped_boxes"].update(unmapped)

            if not mapped_lines and not include_empty:
                report["images_skipped_empty_after_mapping"] += 1
                continue

            samples.append(
                Sample(
                    source_id=source_id,
                    image_path=image_path,
                    label_path=label_path,
                    group_id=source_group_key(image_path),
                    mapped_lines=mapped_lines,
                    raw_boxes=raw_boxes,
                )
            )
            report["images_kept"] += 1

    return samples, report


def class_vector(samples: list[Sample], n_classes: int) -> list[int]:
    counts = [0] * n_classes
    for sample in samples:
        for line in sample.mapped_lines:
            counts[int(line.split(maxsplit=1)[0])] += 1
    return counts


def assign_groups(
    groups: dict[str, list[Sample]],
    ratios: dict[str, float],
    n_classes: int,
    seed: int,
) -> dict[str, list[Sample]]:
    """Make a deterministic, source-stratified, group-disjoint split.

    Source stratification matters here: each source has a different visual
    domain and label vocabulary.  A global class objective can otherwise put
    all crack-only examples in train while using another source to fill val
    and test.  Every source is therefore targeted independently for image and
    class proportions, while a group that occurs in multiple sources still
    remains indivisible.
    """

    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split YAML must contain exactly {SPLIT_NAMES}, got {sorted(ratios)}")
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must be positive and sum to 1.0, got {ratios}")

    by_source: dict[str, list[Sample]] = defaultdict(list)
    for samples in groups.values():
        for sample in samples:
            by_source[sample.source_id].append(sample)

    source_class_totals: dict[str, list[int]] = {}
    source_targets: dict[str, dict[str, dict[str, Any]]] = {}
    for source_id, source_samples in by_source.items():
        total_images = len(source_samples)
        total_classes = class_vector(source_samples, n_classes)
        source_class_totals[source_id] = total_classes
        source_targets[source_id] = {
            split: {
                "images": total_images * ratios[split],
                "classes": [count * ratios[split] for count in total_classes],
            }
            for split in SPLIT_NAMES
        }

    rng = random.Random(seed)
    group_items = list(groups.items())
    random_rank = {group_id: rng.random() for group_id, _ in group_items}

    # Rare-class groups are placed first, which gives val/test a chance to
    # receive them before abundant groups fill their source-level targets.
    def difficulty(item: tuple[str, list[Sample]]) -> tuple[float, int, float]:
        group_id, samples = item
        by_source_in_group: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            by_source_in_group[sample.source_id].append(sample)
        rarity = 0.0
        for source_id, source_samples in by_source_in_group.items():
            vector = class_vector(source_samples, n_classes)
            totals = source_class_totals[source_id]
            for index, count in enumerate(vector):
                if totals[index]:
                    rarity += count / totals[index]
        return (-rarity, -len(samples), random_rank[group_id])

    group_items.sort(key=difficulty)
    assignments: dict[str, list[Sample]] = {split: [] for split in SPLIT_NAMES}
    current_images = {
        source_id: {split: 0 for split in SPLIT_NAMES} for source_id in by_source
    }
    current_classes = {
        source_id: {split: [0] * n_classes for split in SPLIT_NAMES} for source_id in by_source
    }

    for _, samples in group_items:
        group_by_source: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            group_by_source[sample.source_id].append(sample)

        def cost(split: str) -> float:
            total_delta = 0.0
            for source_id, source_samples in group_by_source.items():
                image_target = max(source_targets[source_id][split]["images"], 1.0)
                image_before = current_images[source_id][split]
                image_after = image_before + len(source_samples)
                image_delta = (
                    ((image_after - image_target) / image_target) ** 2
                    - ((image_before - image_target) / image_target) ** 2
                )

                vector = class_vector(source_samples, n_classes)
                class_deltas: list[float] = []
                for index, count in enumerate(vector):
                    class_target = source_targets[source_id][split]["classes"][index]
                    if not class_target:
                        continue
                    class_before = current_classes[source_id][split][index]
                    class_after = class_before + count
                    class_deltas.append(
                        ((class_after - class_target) / class_target) ** 2
                        - ((class_before - class_target) / class_target) ** 2
                    )

                # Image count and the average active-class proportion have
                # equal influence.  Averaging avoids a source with more named
                # classes overwhelming another source's image balance.
                total_delta += image_delta + (sum(class_deltas) / max(len(class_deltas), 1))
            return total_delta

        chosen_split = min(
            SPLIT_NAMES,
            key=lambda split: (
                cost(split),
                sum(
                    current_images[source_id][split]
                    / max(source_targets[source_id][split]["images"], 1.0)
                    for source_id in group_by_source
                ),
                split,
            ),
        )
        assignments[chosen_split].extend(samples)
        for source_id, source_samples in group_by_source.items():
            current_images[source_id][chosen_split] += len(source_samples)
            vector = class_vector(source_samples, n_classes)
            for index, count in enumerate(vector):
                current_classes[source_id][chosen_split][index] += count

    return assignments


def ensure_output_is_safe(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}\n"
            "Choose a new --output-root or move the existing experiment first."
        )


def materialize_image(source: Path, destination: Path, image_mode: str) -> str:
    """Materialize an image without modifying the input source."""

    if image_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        # This only occurs when paths are on different volumes or the file
        # system does not allow hard links.  The report makes the fallback
        # visible instead of silently losing provenance.
        shutil.copy2(source, destination)
        return "copy_fallback"


def report_class_counts(assignments: dict[str, list[Sample]], class_names: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        vector = class_vector(assignments[split], len(class_names))
        result[split] = {name: vector[index] for index, name in enumerate(class_names)}
    return result


def build_dataset(
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    seed: int,
    image_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    class_names = list(dataset_cfg.get("class_names", []))
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError(f"Dataset {dataset_name!r} must have unique, non-empty class_names")
    class_to_id = {name: index for index, name in enumerate(class_names)}

    source_reports: dict[str, dict[str, Any]] = {}
    samples: list[Sample] = []
    for source_cfg in dataset_cfg.get("sources", []):
        source_samples, source_report = collect_source_samples(source_cfg, config_path.parent, class_to_id)
        source_id = str(source_cfg["id"])
        source_reports[source_id] = source_report
        samples.extend(source_samples)

    if not samples:
        raise RuntimeError(f"Dataset {dataset_name!r} has no usable images")

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)

    ratios = {key: float(value) for key, value in config["split"].items()}
    assignments = assign_groups(groups, ratios, len(class_names), seed)
    output_dir = (output_root / dataset_cfg["output_dir"]).resolve()
    split_class_counts = report_class_counts(assignments, class_names)

    source_counts_by_split: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        source_counts_by_split[split] = dict(sorted(Counter(sample.source_id for sample in assignments[split]).items()))

    report: dict[str, Any] = {
        "version": int(config.get("version", 4)),
        "dataset": dataset_name,
        "description": dataset_cfg.get("description", ""),
        "output_dir": str(output_dir),
        "seed": seed,
        "image_mode_requested": image_mode,
        "classes": class_names,
        "input_images_kept": len(samples),
        "groups": len(groups),
        "group_split_counts": {
            split: len({sample.group_id for sample in assignments[split]}) for split in SPLIT_NAMES
        },
        "image_split_counts": {split: len(assignments[split]) for split in SPLIT_NAMES},
        "class_split_counts": split_class_counts,
        "source_images_by_split": source_counts_by_split,
        "source_audit": {
            source_id: {
                **{key: value for key, value in source_report.items() if key != "unmapped_boxes"},
                "unmapped_boxes": dict(sorted(source_report["unmapped_boxes"].items())),
            }
            for source_id, source_report in source_reports.items()
        },
        "excluded_sources": config.get("excluded_sources", []),
    }

    if dry_run:
        return report

    ensure_output_is_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    image_materialization = Counter()

    for split in SPLIT_NAMES:
        image_dir = output_dir / split / "images"
        label_dir = output_dir / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        ordered_samples = sorted(
            assignments[split],
            key=lambda sample: (sample.group_id, sample.source_id, sample.image_path.name.lower()),
        )
        for index, sample in enumerate(ordered_samples):
            stem = f"{sample.source_id}_{index:06d}"
            image_output = image_dir / f"{stem}{sample.image_path.suffix.lower()}"
            label_output = label_dir / f"{stem}.txt"
            materialization = materialize_image(sample.image_path, image_output, image_mode)
            image_materialization[materialization] += 1
            label_output.write_text("\n".join(sample.mapped_lines), encoding="utf-8")
            manifest_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "output_image": str(image_output.relative_to(output_dir)),
                    "output_label": str(label_output.relative_to(output_dir)),
                    "source_id": sample.source_id,
                    "source_image": str(sample.image_path),
                    "source_label": str(sample.label_path) if sample.label_path else "",
                    "group_id": sample.group_id,
                    "raw_boxes": sample.raw_boxes,
                    "mapped_boxes": len(sample.mapped_lines),
                }
            )

    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False, allow_unicode=True)

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    report["image_materialization"] = dict(sorted(image_materialization.items()))
    with (output_dir / "build_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)

    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"\n[{report['dataset']}]")
    print(f"  images kept: {report['input_images_kept']} across {report['groups']} safe groups")
    for split in SPLIT_NAMES:
        class_counts = report["class_split_counts"][split]
        class_summary = ", ".join(f"{name}={count}" for name, count in class_counts.items())
        print(
            f"  {split}: {report['image_split_counts'][split]} images, "
            f"{report['group_split_counts'][split]} groups; {class_summary}"
        )
    for source_id, source_report in report["source_audit"].items():
        print(
            f"  source {source_id}: kept={source_report['images_kept']}/"
            f"{source_report['images_seen']}, skipped_empty={source_report['images_skipped_empty_after_mapping']}, "
            f"mapped_boxes={source_report['mapped_boxes']}"
        )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    configured_datasets = config.get("datasets")
    if not isinstance(configured_datasets, dict):
        raise ValueError("v4 taxonomy YAML has no datasets mapping")
    if "split" not in config:
        raise ValueError("v4 taxonomy YAML has no split mapping")

    selected = list(configured_datasets) if args.dataset == "all" else [args.dataset]
    missing = [name for name in selected if name not in configured_datasets]
    if missing:
        raise ValueError(f"Selected dataset(s) absent from config: {missing}")

    output_root = args.output_root.resolve() if args.output_root else config_path.parent
    for dataset_name in selected:
        report = build_dataset(
            dataset_name=dataset_name,
            dataset_cfg=configured_datasets[dataset_name],
            config=config,
            config_path=config_path,
            output_root=output_root,
            seed=args.seed,
            image_mode=args.image_mode,
            dry_run=args.dry_run,
        )
        print_report(report)
        if args.dry_run:
            print("  dry run: no files written")
        else:
            print(f"  wrote: {report['output_dir']}")


if __name__ == "__main__":
    main()
