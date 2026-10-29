"""Resolve the normal v5 Hub snapshot; repair machine-specific YAML paths locally."""
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

REPO_ID = 'AkumaDachi/roadeeye-sewer-defects-v5'
REVISION = 'd88c0ed4f0fd39253eb3866bf2feb9dcf808bae5'
NAMES = ['crack', 'corrosion_rust', 'sediment_deposit', 'root_intrusion', 'joint_defect']
COUNTS = {'train': 9486, 'val': 2014, 'test': 1986}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def prepare_dataset(workspace, repo_id=REPO_ID, revision=REVISION, local_data=None,
                    strict_counts=True):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    commit = None
    if local_data:
        source = Path(local_data).resolve()
        source = source.parent if source.is_file() else source
    else:
        from huggingface_hub import HfApi, snapshot_download
        token = os.getenv('HF_TOKEN') or None  # also allows existing local HF authentication
        info = HfApi().repo_info(repo_id=repo_id, repo_type='dataset', revision=revision, token=token)
        commit = info.sha
        snapshot = Path(snapshot_download(
            repo_id=repo_id, repo_type='dataset', revision=commit, token=token,
            cache_dir=str(workspace / 'hf_cache'),
            allow_patterns=['data.yaml', 'train/*', 'valid/*', 'val/*', 'test/*',
                            'final_dataset_v5/*', 'README.md'],
            ignore_patterns=['*.cache', '*.pt'], max_workers=4,
        ))
        candidates = [p.parent for p in snapshot.rglob('data.yaml')
                      if (p.parent / 'train/images').is_dir() and (p.parent / 'test/images').is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one v5 root with data.yaml/train/test; found {candidates}')
        source = candidates[0]  # do not resolve individual symlinks out of snapshot structure
    raw = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'))
    names = raw.get('names', [])
    names = [names.get(i, names.get(str(i))) for i in range(len(names))] if isinstance(names, dict) else names
    if names != NAMES:
        raise ValueError(f'Expected normal v5 five-class taxonomy, got {names}')
    cfg = {'path': str(source), 'nc': 5, 'names': dict(enumerate(NAMES))}
    counts, hashes, label_warnings = {}, {}, []
    for split, candidates in {'train': ['train'], 'val': ['valid', 'val'], 'test': ['test']}.items():
        folder = next((source / name for name in candidates if (source / name / 'images').is_dir()), None)
        if folder is None or not (folder / 'labels').is_dir():
            raise ValueError(f'Missing image/label directories for {split} in {source}')
        images = sorted(p for p in (folder / 'images').rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f'Empty split: {split}')
        counts[split] = len(images)
        digest = hashlib.sha256()
        for image in images:
            relative = image.relative_to(folder / 'images')
            label = folder / 'labels' / relative.with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'Missing label file: {label}; use an empty file for true background images')
            content = label.read_bytes()
            for line_number, line in enumerate(content.decode('utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5 or fields[0] not in {'0', '1', '2', '3', '4'}:
                    raise ValueError(f'Invalid YOLO detection annotation: {label}')
                coords = list(map(float, fields[1:]))
                # Match Ultralytics' 1% coordinate tolerance; do not silently clean the published v5.
                if not all(math.isfinite(c) and -0.01 <= c <= 1.01 for c in coords):
                    raise ValueError(f'Invalid normalized box: {label}')
                if min(coords[2:]) <= 0 or not all(0 <= c <= 1 for c in coords):
                    label_warnings.append({'split': split, 'label': str(label.relative_to(source)),
                                           'line': line_number, 'issue': 'zero/nonpositive size or tolerance-bound coordinate'})
            digest.update(relative.as_posix().encode())
            digest.update(b'\0' + content + b'\0')
        hashes[split] = digest.hexdigest()
        cfg[split] = str(folder / 'images')
    if strict_counts and counts != COUNTS:
        raise ValueError(f'Unexpected full-v5 counts {counts}; expected {COUNTS}. '
                         'Check the upload. Use --allow-count-mismatch only for deliberate smoke-test fixtures.')
    resolved = workspace / 'data_v5_resolved.yaml'
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    provenance = {'repo_id': None if local_data else repo_id, 'requested_revision': revision,
                  'resolved_commit': commit, 'local_root': str(source), 'counts': counts,
                  'filename_label_sha256': hashes, 'preprocessing': 'normal v5, no CLAHE',
                  'label_warnings': label_warnings}
    (workspace / 'dataset_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Dataset ready: {counts}; revision: {commit or "local"}', flush=True)
    if label_warnings:
        print(f'WARNING: {len(label_warnings)} suspicious label rows preserved for consistent comparisons; '
              'see dataset_provenance.json. Dataset was not relabelled.', flush=True)
    return resolved, provenance
