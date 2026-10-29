"""Build a non-destructive, model-error-filtered YOLO dataset view.

The resulting test set is selected using model predictions and is unsuitable
for reporting independent generalization performance. Originals are preserved.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = ROOT / "runs/pipe_proto/yolo26s_v5_parallel_simam_cbam_se_error_analysis"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build(source, analysis, output, dry_run=False):
    source, analysis, output = source.resolve(), analysis.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source dataset")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): v for k, v in names.items()}
    reasons = defaultdict(Counter)
    csv_info = {}
    for filename, reason in [("false_positives.csv", "FP"), ("false_negatives.csv", "FN")]:
        path = analysis / filename
        count = 0
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                image = Path(row["image"])
                if not image.is_absolute():
                    raise ValueError(f"Expected absolute image path: {image}")
                image = image.resolve()
                if not image.is_relative_to(source) or not image.is_file():
                    raise ValueError(f"CSV image is absent or outside source: {image}")
                reasons[image][reason] += 1
                count += 1
        csv_info[reason] = {"path": str(path), "rows": count,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    splits, stats, exclusions, discovered = {}, {}, [], set()
    for split in ("train", "val", "test"):
        image_dir = Path(config[split])
        if not image_dir.is_absolute():
            image_dir = source / image_dir
        image_dir = image_dir.resolve()
        if not image_dir.is_relative_to(source) or not image_dir.is_dir() or image_dir.name != "images":
            raise ValueError(f"Expected an images directory within source: {image_dir}")
        images = sorted(p.resolve() for p in image_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
        kept = []
        before, after = Counter(), Counter()
        for image in images:
            discovered.add(image)
            label = image_dir.parent / "labels" / image.relative_to(image_dir).with_suffix(".txt")
            classes = []
            if label.exists():
                for line in label.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5 or int(fields[0]) not in names:
                        raise ValueError(f"Invalid detection label: {label}")
                    classes.append(int(fields[0]))
            before.update(classes)
            if image in reasons:
                exclusions.append({"split": split, "image": str(image),
                                   "relative_image": image.relative_to(source).as_posix(),
                                   "label": str(label), "fp_instances": reasons[image]["FP"],
                                   "fn_instances": reasons[image]["FN"]})
            else:
                kept.append(image)
                after.update(classes)
        if not kept:
            raise ValueError(f"Filtering would empty split: {split}")
        splits[split] = kept
        stats[split] = {"before": len(images), "excluded": len(images) - len(kept),
                        "retained": len(kept),
                        "instances_before": {name: before[i] for i, name in names.items()},
                        "instances_retained": {name: after[i] for i, name in names.items()}}
    if set(reasons) - discovered:
        raise ValueError("Some CSV images were not found in configured splits")
    report = {"purpose": "Model-selected diagnostic subset; biased evaluation, not an independent benchmark",
              "source": str(source), "csv_inputs": csv_info, "excluded_unique_images": len(reasons),
              "split_counts": stats, "original_files_deleted": 0}
    print(json.dumps(report, indent=2))
    if dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    new_config = {"path": str(source), "nc": len(names), "names": names}
    for split, images in splits.items():
        manifest = output / f"{split}.txt"
        manifest.write_text("".join(f"{p.as_posix()}\n" for p in images), encoding="utf-8")
        new_config[split] = manifest.as_posix()
    (output / "data.yaml").write_text(yaml.safe_dump(new_config, sort_keys=False), encoding="utf-8")
    (output / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "excluded_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "relative_image", "label", "fp_instances", "fn_instances"])
        writer.writeheader()
        writer.writerows(exclusions)
    (output / "README.md").write_text(
        "# Error-filtered diagnostic dataset\n\n"
        "This YOLO dataset view excludes the union of images in the FP/FN CSV files. "
        "It references the original images and labels through absolute-path manifests; no originals were deleted.\n\n"
        "Selection uses predictions from the parallel model. Scores on the retained test subset "
        "are selection-biased and must not replace full-v5 test metrics or be presented as improved generalization. "
        "FP/FN status alone is not evidence that an image or annotation is defective.\n\n"
        "Do not move excluded test images into training. Train and validation lists are unchanged "
        "when the input CSVs contain only test images; retraining is unnecessary to measure this filtering effect.\n\n"
        "See filter_report.json for counts and excluded_images.csv for provenance. "
        "These manifests are local references, not a standalone dataset upload.\n", encoding="utf-8")
    print(f"Created: {output / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "final_dataset_v5")
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=ROOT / "v5_error_filtered_diagnostic")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args.source, args.analysis, args.output, args.dry_run)
