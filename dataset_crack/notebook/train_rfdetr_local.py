r"""Train RF-DETR Nano on the existing local, five-class RoadEye v5 dataset.

Run from the RoadEye root:
    venv\Scripts\python.exe dataset_crack\notebook\train_rfdetr_local.py --check
    venv\Scripts\python.exe dataset_crack\notebook\train_rfdetr_local.py

The source dataset is never edited. A reusable working dataset hard-links its
images and copies its labels, omitting invalid zero-area boxes with a report.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from rfdetr_dataset import EXPECTED_COUNTS, EXTENSIONS, NAMES, prepare_overlay


DATASET_CRACK = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = DATASET_CRACK / 'final_dataset_v5'
DEFAULT_WORKSPACE = DATASET_CRACK / 'rfdetr_local_workspace'
DEFAULT_RUNS = DATASET_CRACK / 'runs' / 'pipe_proto'
REQUIRED_RFDETR_VERSION = '1.11.0'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_SOURCE,
                        help='Local full five-class v5 directory; never downloaded.')
    parser.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE,
                        help='Working YOLO directory and pretrained model cache.')
    parser.add_argument('--runs', type=Path, default=DEFAULT_RUNS,
                        help='Parent of unique RF-DETR run directories.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--resolution', type=int, default=512,
                        help='512 for a new local run; 384 matches the original Kaggle setup.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum-steps', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0,
                        help='0 is reliable on Windows; try 2 if CPU loading limits throughput.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path, default=None,
                        help='Full last.ckpt from an earlier run; never use best .pth here.')
    parser.add_argument('--eval-test', action='store_true',
                        help='Evaluate the held-out test split after model selection.')
    parser.add_argument('--check', action='store_true',
                        help='Read-only dataset and hardware check; no install or training.')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1 or args.workers < 0:
        parser.error('epochs, batch size and accumulation must be positive; workers must be nonnegative')
    if args.resolution < 256 or args.resolution % 32:
        parser.error('RF-DETR Nano detection resolution must be >= 256 and divisible by 32')
    return args


def verify_source(source: Path) -> dict[str, int]:
    """Check the exact full-v5 layout before creating a run or working files."""
    if not (source / 'data.yaml').is_file():
        raise FileNotFoundError(f'Local v5 data.yaml not found: {source / "data.yaml"}')
    data = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'))
    names = data.get('names')
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if names != NAMES or data.get('nc') != len(NAMES):
        raise ValueError(f'Expected five v5 classes {NAMES}; found {names}')
    counts = {}
    for split, expected in EXPECTED_COUNTS.items():
        images_dir = source / split / 'images'
        labels_dir = source / split / 'labels'
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f'Missing {split}/images or {split}/labels in {source}')
        images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if len(images) != expected:
            raise ValueError(f'{split}: expected {expected} images, found {len(images)}')
        for image in images:
            label = labels_dir / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(f'Missing label for {image}: {label}')
        counts[split] = len(images)
    return counts


def hardware_report() -> None:
    print(f'Python {platform.python_version()} | PyTorch {torch.__version__}')
    if not torch.cuda.is_available():
        print('CUDA GPU: unavailable')
        return
    props = torch.cuda.get_device_properties(0)
    print(f'CUDA GPU: {props.name} | total VRAM: {props.total_memory / 2**30:.1f} GiB')
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f'System RAM: {memory.total / 2**30:.1f} GiB total, '
              f'{memory.available / 2**30:.1f} GiB available now')
        if memory.available < 5 * 2**30:
            print('WARNING: Free more system RAM before training; aim for at least 5 GiB available.')
    except ImportError:
        pass


def load_rfdetr_nano():
    try:
        actual = importlib.metadata.version('rfdetr')
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit('Install RF-DETR first:\n'
                         '  venv\\Scripts\\python.exe -m pip install "rfdetr[train]==1.11.0"') from exc
    if actual != REQUIRED_RFDETR_VERSION:
        raise SystemExit(f'Expected rfdetr=={REQUIRED_RFDETR_VERSION}, found {actual}. '
                         'Use the pinned version for Kaggle checkpoint compatibility.')
    from rfdetr import RFDETRNano
    return RFDETRNano


def save_json(path: Path, content: object) -> None:
    path.write_text(json.dumps(content, indent=2, default=str) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.data.resolve()
    counts = verify_source(source)
    print(f'Local dataset: {source}')
    print(f'Splits: {counts} | classes: {", ".join(NAMES)} | preprocessing: normal, no CLAHE')
    hardware_report()
    if args.check:
        print('Read-only check passed. No dataset files or training runs were created.')
        return 0
    if not torch.cuda.is_available():
        raise SystemExit('Training requires CUDA. Check the GPU driver and PyTorch installation.')
    if torch.cuda.get_device_properties(0).total_memory < 6 * 2**30:
        raise SystemExit('Less than 6 GiB VRAM detected. This script targets the 8 GiB local GPU.')
    workspace = args.workspace.resolve()
    runs = args.runs.resolve()
    if workspace == source or workspace.is_relative_to(source) or source.is_relative_to(workspace):
        raise ValueError('The working directory must be separate from the source dataset')
    os.environ['RF_HOME'] = str(workspace / 'model_cache')
    RFDETRNano = load_rfdetr_nano()
    resume_path = None
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.is_file() or resume_path.suffix != '.ckpt':
            raise FileNotFoundError(f'--resume requires an existing full .ckpt file: {resume_path}')

    workspace.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    prepared = workspace / 'final_dataset_v5_rfdetr_yolo'
    provenance = prepare_overlay(source, prepared)
    if provenance['counts'] != EXPECTED_COUNTS:
        raise ValueError('Prepared dataset does not match the full v5 split counts')
    print(f'RF-DETR working dataset: {prepared}')
    print(f'Invalid label rows omitted only in working copy: {provenance["removed_label_row_count"]}')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    run_dir = runs / (f'rfdetr_nano_v5_normal_{args.resolution}_seed{args.seed}_'
                      f'{stamp}_{uuid.uuid4().hex[:6]}')
    run_dir.mkdir(parents=True, exist_ok=False)
    train_config = dict(
        dataset_file='yolo', dataset_dir=str(prepared), output_dir=str(run_dir),
        epochs=args.epochs, resolution=args.resolution, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps, eval_batch_size=1,
        num_workers=args.workers, device='cuda', amp_dtype='auto', seed=args.seed,
        multi_scale=False,  # Predictable peak VRAM on the 8 GiB card.
        use_ema=True, early_stopping=True, early_stopping_patience=10,
        skip_best_epochs=2, checkpoint_interval=5, tensorboard=False,
        log_per_class_metrics=True,
    )
    if resume_path is not None:
        train_config['resume'] = str(resume_path)
    experiment = {
        'architecture': 'RFDETRNano', 'rfdetr_version': REQUIRED_RFDETR_VERSION,
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'source_dataset': str(source), 'prepared_dataset': str(prepared),
        'dataset_provenance': provenance, 'train_config': train_config,
        'test_is_opt_in': True,
    }
    save_json(run_dir / 'roadeeye_experiment.json', experiment)
    shutil.copy2(prepared / 'dataset_provenance.json', run_dir / 'dataset_provenance.json')
    print(f'Unique run: {run_dir}')
    print(f'Nominal effective batch: {args.batch_size * args.grad_accum_steps}')
    print('Starting RF-DETR Nano fine-tuning from pretrained weights or the supplied full checkpoint...')

    model = RFDETRNano(gradient_checkpointing=True)
    try:
        model.train(**train_config)
    except KeyboardInterrupt:
        print(f'Interrupted. Inspect {run_dir / "last.ckpt"} to resume.')
        raise
    del model
    gc.collect()
    torch.cuda.empty_cache()

    best = run_dir / 'checkpoint_best_total.pth'
    if not best.is_file():
        raise FileNotFoundError(f'RF-DETR did not save a best checkpoint: {best}')
    best_model = RFDETRNano.from_checkpoint(str(best))
    eval_config = dict(dataset_file='yolo', dataset_dir=str(prepared),
                       resolution=args.resolution, batch_size=1, eval_batch_size=1,
                       num_workers=args.workers, device='cuda', amp_dtype='auto')
    validation = best_model.evaluate(split='val', **eval_config)
    save_json(run_dir / 'best_validation_metrics.json', validation)
    print(f'Best-checkpoint validation metrics: {validation}')
    if args.eval_test:
        test = best_model.evaluate(split='test', **eval_config)
        save_json(run_dir / 'full_v5_test_metrics.json', test)
        print(f'Held-out full-v5 test metrics: {test}')
    print(f'Finished. Checkpoints and metrics: {run_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
