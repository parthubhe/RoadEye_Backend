"""In-place, reversible v6 manifest filtering: all FP images + 80% of FN images.

Dry-run by default. Original v5 image/label files are never deleted.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_errors(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return Counter(Path(row['image']).resolve() for row in rows)


def csv_bytes(fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix='.filter_', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    dataset = ROOT / 'final_dataset_v6'
    analysis = ROOT / 'runs/pipe_proto/yolo26s_v6_train_error_analysis'
    originals = {name: (dataset / name).read_bytes() for name in
                 ('data.yaml', 'train.txt', 'val.txt', 'test.txt', 'filter_report.json', 'excluded_images.csv', 'README.md')}
    report = json.loads(originals['filter_report.json'])
    if 'training_filter' in report:
        raise ValueError('Training filter already applied; refusing to resample or apply twice.')
    cfg = yaml.safe_load(originals['data.yaml'])
    if Path(cfg['train']).resolve() != (dataset / 'train.txt').resolve():
        raise ValueError('Unexpected training manifest path')
    summary = json.loads((analysis / 'analysis_summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'complete' or summary['split'] != 'train':
        raise ValueError('Expected completed training-set analysis')
    fp = read_errors(analysis / 'false_positives.csv')
    fn = read_errors(analysis / 'false_negatives.csv')
    train = [Path(line).resolve() for line in originals['train.txt'].decode('utf-8').splitlines() if line.strip()]
    source = Path(report['source']).resolve()
    training_root = source / 'train/images'
    if len(set(train)) != len(train) or len(train) != summary['images_processed']:
        raise ValueError('Training manifest differs from analyzed split')
    if (set(fp) | set(fn)) - set(train):
        raise ValueError('Error CSV contains images outside the current training list')
    if any(not p.is_relative_to(training_root) or not p.is_file() for p in train):
        raise ValueError('Training source path invalid')
    assert len(fp) == summary['fp_unique_images'] and len(fn) == summary['fn_unique_images']
    selected_fn = set(random.Random(args.seed).sample(sorted(fn, key=lambda p: p.as_posix()), math.ceil(0.8 * len(fn))))
    removed = set(fp) | selected_fn
    retained = [path for path in train if path not in removed]
    if not retained:
        raise ValueError('Refusing to empty the training split')
    names = cfg['names']
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    before, after = Counter(), Counter()
    for image in train:
        label = source / 'train/labels' / image.relative_to(training_root).with_suffix('.txt')
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                fields = line.split()
                if len(fields) != 5 or int(fields[0]) not in names:
                    raise ValueError(f'Invalid label: {label}')
                cls = int(fields[0])
                before[cls] += 1
                if image not in removed:
                    after[cls] += 1
    if any(after[i] == 0 for i in names):
        raise ValueError('Filtering would remove all training instances of a class')
    audit = {
        'policy': 'Remove all unique FP images, plus a seeded sample of ceil(80% of all unique FN images); take union.',
        'seed': args.seed, 'requested_fn_fraction': 0.8,
        'before': len(train), 'retained': len(retained), 'removed_unique_images': len(removed),
        'fp_images_removed': len(fp), 'fn_images_total': len(fn), 'fn_images_sampled': len(selected_fn),
        'sampled_fn_already_fp': len(selected_fn & set(fp)),
        'additional_fn_only_images_removed': len(selected_fn - set(fp)),
        'effective_fn_images_removed': len(removed & set(fn)),
        'effective_fn_images_retained': len(set(fn) - removed),
        'analysis_checkpoint': summary['checkpoint'], 'analysis_checkpoint_sha256': summary['checkpoint_sha256'],
        'analysis_confidence': summary['confidence'], 'analysis_matching_iou': summary['matching_iou'],
        'csv_inputs': {kind: {'path': str(analysis / filename), 'sha256': digest((analysis / filename).read_bytes())}
                       for kind, filename in [('FP', 'false_positives.csv'), ('FN', 'false_negatives.csv')]},
        'original_files_deleted': 0,
        'note': 'FP/FN overlap can make effective FN removal exceed 80%. Selection is model-based, not verified data cleansing.',
    }
    print(json.dumps(audit, indent=2), flush=True)
    if not args.apply:
        print('Dry run only; pass --apply to modify existing v6 manifests.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    backup = dataset / 'backups' / ('before_training_filter_' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    for name, data in originals.items():
        if (dataset / name).read_bytes() != data:
            raise RuntimeError(f'File changed during preparation: {name}')
        shutil.copy2(dataset / name, backup / name)
    audit['backup'] = str(backup)
    audit['applied_utc'] = datetime.now(timezone.utc).isoformat()
    report['training_filter'] = audit
    report['split_counts']['train'] = {
        'before': len(train), 'excluded': len(removed), 'retained': len(retained),
        'instances_before': {name: before[i] for i, name in names.items()},
        'instances_retained': {name: after[i] for i, name in names.items()},
    }
    with io.StringIO(originals['excluded_images.csv'].decode('utf-8'), newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        exclusions = list(reader)
    for image in sorted(removed):
        exclusions.append({'split': 'train', 'image': str(image), 'relative_image': image.relative_to(source).as_posix(),
                           'label': str(source / 'train/labels' / image.relative_to(training_root).with_suffix('.txt')),
                           'fp_instances': fp[image], 'fn_instances': fn[image]})
    report['excluded_unique_images'] = len({row['image'] for row in exclusions})
    selection = [{'image': str(image), 'fp_instances': fp[image], 'fn_instances': fn[image],
                  'selected_in_80pct_fn_sample': int(image in selected_fn), 'excluded_from_v6': int(image in removed)}
                 for image in sorted(set(fp) | set(fn))]
    updates = {
        'train.txt': ''.join(f'{image.as_posix()}\n' for image in retained).encode('utf-8'),
        'filter_report.json': json.dumps(report, indent=2).encode('utf-8'),
        'excluded_images.csv': csv_bytes(fields, exclusions),
        'training_filter_selection.csv': csv_bytes(list(selection[0]), selection),
        'README.md': originals['README.md'] + (
            '\n\n## In-place training filter\n\n'
            f'Training now contains {len(retained)} images, down from {len(train)}. '
            f'Excluded all {len(fp)} unique FP images plus {len(selected_fn)} sampled FN images '
            f'(80% rounded up, seed {args.seed}); after overlap, {len(removed)} unique training images were excluded. '
            f'Validation and filtered test are unchanged. Full v5 originals are preserved. Backup: {backup.name}. '
            'See training_filter_selection.csv for each sampling/removal decision. '
            'Retraining is required for the existing model to reflect the changed training membership.\n').encode('utf-8'),
    }
    try:
        for name, data in updates.items():
            atomic_write(dataset / name, data)
        for name in ('data.yaml', 'val.txt', 'test.txt'):
            assert (dataset / name).read_bytes() == originals[name]
    except Exception:
        for name in updates:
            if name in originals:
                atomic_write(dataset / name, originals[name])
        raise
    print(f'Updated v6 in place: {len(train)} -> {len(retained)} training images. Backup: {backup}')


if __name__ == '__main__':
    main()
