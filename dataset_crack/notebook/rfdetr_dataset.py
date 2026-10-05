"""Prepare the pinned RoadEye v5 YOLO dataset for RF-DETR without editing its snapshot.

This file is embedded inside RFDETR_Nano_v5.ipynb; the notebook needs no repo clone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import yaml

REPO_ID = 'AkumaDachi/roadeeye-sewer-defects-v5'
REVISION = 'd88c0ed4f0fd39253eb3866bf2feb9dcf808bae5'
NAMES = ['crack', 'corrosion_rust', 'sediment_deposit', 'root_intrusion', 'joint_defect']
EXPECTED_COUNTS = {'train': 9486, 'valid': 2014, 'test': 1986}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def locate_dataset(snapshot: Path) -> Path:
    """Find the single full-v5 root inside a Hugging Face snapshot."""
    snapshot = Path(snapshot)
    candidates = [
        item.parent for item in snapshot.rglob('data.yaml')
        if all((item.parent / split / 'images').is_dir() and
               (item.parent / split / 'labels').is_dir()
               for split in ('train', 'valid', 'test'))
    ]
    if len(candidates) != 1:
        raise ValueError(f'Expected one full v5 dataset root, found {candidates}')
    return candidates[0]


def _names(yaml_path: Path) -> list[str]:
    raw = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    names = raw.get('names')
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if names != NAMES or raw.get('nc', 5) != 5:
        raise ValueError(f'Expected full v5 five-class taxonomy, found {names}')
    return names


def _label_lines(content: bytes, label: Path, split: str, skipped: list[dict]) -> list[str]:
    retained = []
    for line_no, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'Invalid YOLO label row {label}:{line_no}')
        try:
            cls = int(parts[0])
            x, y, w, h = (float(part) for part in parts[1:])
        except ValueError as exc:
            raise ValueError(f'Invalid YOLO label row {label}:{line_no}') from exc
        if cls not in range(5):
            raise ValueError(f'Unexpected class ID at {label}:{line_no}')
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f'Non-finite box at {label}:{line_no}')
        # RF-DETR documents normalized coordinates in [0, 1] and requires area > 0.
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            skipped.append({'split': split, 'label': str(label), 'line': line_no,
                            'value': line, 'reason': 'zero-size or out-of-range YOLO box'})
            continue
        retained.append(line)
    return retained


def prepare_overlay(source: Path, overlay: Path, *, expected_counts=EXPECTED_COUNTS,
                    link_images: bool = True) -> dict:
    """Hard-link images and copy labels to a separate RF-DETR YOLO directory.

    RF-DETR resolves YAML image paths and rejects directory symlinks that escape
    the dataset root. Hard links share image bytes but keep each path under it.
    Set link_images=False only for small cross-filesystem/local smoke tests.
    """
    source, overlay = Path(source).resolve(), Path(overlay).resolve()
    if overlay == source or overlay.is_relative_to(source) or source.is_relative_to(overlay):
        raise ValueError('The RF-DETR overlay must be separate from the snapshot')
    _names(source / 'data.yaml')
    overlay.mkdir(parents=True, exist_ok=True)
    counts, hashes, skipped = {}, {}, []
    for split in ('train', 'valid', 'test'):
        images_dir = source / split / 'images'
        labels_dir = source / split / 'labels'
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f'Missing {split} image/label directories')
        images = sorted(p for p in images_dir.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        counts[split] = len(images)
        if expected_counts is not None and len(images) != expected_counts[split]:
            raise ValueError(f'Unexpected {split} count {len(images)}; expected {expected_counts[split]}')
        if any(image.parent != images_dir for image in images):
            raise ValueError(f'RF-DETR YOLO loader requires flat {split}/images; nested images found')
        target_split = overlay / split
        target_split.mkdir(exist_ok=True)
        target_images = target_split / 'images'
        if target_images.is_symlink():
            raise ValueError(f'Directory symlinks are incompatible with RF-DETR: {target_images}')
        target_images.mkdir(exist_ok=True)
        target_labels = target_split / 'labels'
        target_labels.mkdir(exist_ok=True)
        sha = hashlib.sha256()
        for image in images:
            relative = image.relative_to(images_dir)
            label = labels_dir / relative.with_suffix('.txt')
            if not label.is_file():
                raise FileNotFoundError(f'Missing label file: {label}')
            output_image = target_images / relative
            output_image.parent.mkdir(parents=True, exist_ok=True)
            if output_image.exists():
                if link_images and not os.path.samefile(image, output_image):
                    raise FileExistsError(f'Existing image is not linked to source: {output_image}')
            elif link_images:
                try:
                    os.link(image.resolve(), output_image)
                except OSError as exc:
                    raise OSError(f'Cannot hard-link {image} to {output_image}; '
                                  'put the Hub cache and working directory on one filesystem') from exc
            else:
                shutil.copy2(image, output_image)
            content = label.read_bytes()
            kept = _label_lines(content, label.relative_to(source), split, skipped)
            output_label = target_labels / relative.with_suffix('.txt')
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text(''.join(f'{line}\n' for line in kept), encoding='utf-8')
            sha.update(relative.as_posix().encode('utf-8'))
            sha.update(b'\0' + content + b'\0')
        hashes[split] = sha.hexdigest()
    config = {'path': str(overlay), 'nc': 5, 'names': dict(enumerate(NAMES)),
              'train': 'train/images', 'val': 'valid/images', 'test': 'test/images'}
    (overlay / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    report = {'source': str(source), 'overlay': str(overlay), 'counts': counts,
              'classes': NAMES, 'source_filename_label_sha256': hashes,
              'removed_label_rows': skipped, 'removed_label_row_count': len(skipped),
              'image_files_removed': 0, 'source_modified': False,
              'image_overlay_method': 'hardlink' if link_images else 'copy',
              'preprocessing': 'normal v5, no offline CLAHE; invalid YOLO box rows omitted in overlay only'}
    (overlay / 'dataset_provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
