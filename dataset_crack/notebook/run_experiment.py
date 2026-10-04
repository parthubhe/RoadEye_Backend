"""Common E1-E6 runner. Training uses validation; test evaluation is explicit."""
import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds

from architectures import EXPERIMENTS, write_config
from dataset import prepare_dataset, REPO_ID, REVISION
from experiment_trainer import ExperimentTrainer
from modules import register_modules

ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_DATA = ROOT.parent / 'final_dataset_v5'


def arguments(argv=None, default_experiment=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=EXPERIMENTS, default=default_experiment, required=default_experiment is None)
    p.add_argument('--simam-base', choices=['E1', 'E2', 'E3', 'E4', 'E5'], default='E3')
    p.add_argument('--repo-id', default=REPO_ID)
    p.add_argument('--revision', default=REVISION, help='Use the same resolved HF commit for every experiment')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--local-data', type=Path, default=DEFAULT_LOCAL_DATA,
                        help='Local v5 root or data.yaml (default: dataset_crack/final_dataset_v5, relative to this script)')
    source.add_argument('--download-data', action='store_true',
                        help='Explicitly use the pinned Hugging Face dataset instead of local v5')
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    p.add_argument('--project', type=Path, default=ROOT / 'runs')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pretrained', default='auto', help='auto or path to matching COCO yolo26s.pt/yolo26m.pt')
    p.add_argument('--scratch', action='store_true', help='Explicit random initialization; not a comparable pretrained run')
    p.add_argument('--dry-run', action='store_true', help='Build/forward-check without downloading dataset or weights')
    p.add_argument('--allow-count-mismatch', action='store_true')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-split', choices=['val', 'test'], default='val')
    p.add_argument('--evaluate-only', type=Path, help='Evaluate an existing best.pt; no training')
    p.add_argument('--resume', type=Path, help='Resume an interrupted last.pt in its original run directory')
    args = p.parse_args(argv)
    args.local_data = None if args.download_data else args.local_data.resolve()
    return args


def check_dataset_source(args):
    """Fail locally before building a run; never silently fall back to a download."""
    if args.local_data is None:
        print(f'Dataset source: Hugging Face {args.repo_id} @ {args.revision}', flush=True)
        return
    source = args.local_data
    yaml_path = source if source.is_file() else source / 'data.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f'Local v5 dataset YAML not found: {yaml_path}. '
            'Use --local-data "path/to/final_dataset_v5" or explicitly pass --download-data. '
            'No dataset download was attempted.')
    print(f'Dataset source: local {source}', flush=True)


def run(argv=None, default_experiment=None):
    args = arguments(argv, default_experiment)
    if ultralytics.__version__ != '8.4.120':
        raise RuntimeError('Install requirements.txt: architecture/parser support was tested with ultralytics==8.4.120')
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz < 64 or args.imgsz % 32 or args.workers < 0:
        raise ValueError('Positive epochs/batch, imgsz >=64 divisible by 32, workers >=0 required')
    if args.resume and args.evaluate_only:
        raise ValueError('Choose resume OR evaluate-only')
    if ',' in args.device:
        raise ValueError('These notebook runners use one GPU per experiment; use separate sessions for parallel experiments')
    if not args.dry_run:
        check_dataset_source(args)
    register_modules()
    init_seeds(args.seed, deterministic=True)
    suffix = f'_{args.simam_base}' if args.experiment == 'E6' else ''
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    name = f'{args.experiment}{suffix}_{EXPERIMENTS[args.experiment]}_v5_normal_seed{args.seed}_{stamp}_{uuid4().hex[:6]}'
    work = args.workspace.resolve() / name
    work.mkdir(parents=True, exist_ok=False)
    weights = None if args.scratch else args.pretrained
    if weights not in (None, 'auto') and Path(weights).is_file():
        weights = str(Path(weights).resolve())
    model_yaml = write_config(args.experiment, work, args.simam_base, weights)
    if args.dry_run:
        model = YOLO(str(model_yaml)).model.cpu().eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 128, 128))
        info = {'experiment': args.experiment, 'simam_base': args.simam_base,
                'parameters': sum(p.numel() for p in model.parameters()),
                'strides': model.stride.tolist(), 'config': str(model_yaml)}
        print(json.dumps(info, indent=2))
        return work
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU runtime in Colab/Kaggle, or pass --device cpu for a smoke test')
    data_yaml, provenance = prepare_dataset(args.workspace.resolve() / 'dataset', args.repo_id, args.revision, args.local_data,
                                            not args.allow_count_mismatch)
    common = dict(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device,
                  workers=args.workers, plots=True, project=str(args.project.resolve()), exist_ok=False)
    started = time.perf_counter()
    if args.evaluate_only:
        model = YOLO(str(args.evaluate_only.resolve()))
        if model.model.yaml.get('experiment') != args.experiment:
            raise ValueError('Evaluation experiment does not match checkpoint')
        if args.experiment == 'E6' and model.model.yaml.get('simam_base') != args.simam_base:
            raise ValueError('Pass --simam-base matching the evaluated E6 checkpoint')
    elif args.resume:
        model = YOLO(str(args.resume.resolve()))
        if model.model.yaml.get('experiment') != args.experiment:
            raise ValueError('Resume experiment does not match checkpoint')
        if not model.ckpt or model.ckpt.get('epoch', -1) < 0 or model.ckpt.get('optimizer') is None:
            raise ValueError('Checkpoint is completed/stripped; --resume requires an interrupted last.pt with optimizer state')
        if args.experiment == 'E6' and model.model.yaml.get('simam_base') != args.simam_base:
            raise ValueError('E6 parent must match resumed checkpoint')
        model.train(trainer=ExperimentTrainer, resume=True,
                    save_dir=str(args.resume.resolve().parent.parent), **common)
    else:
        model = YOLO(str(model_yaml))
        model.train(
            trainer=ExperimentTrainer, **common, name=name, pretrained=False,
            epochs=args.epochs, optimizer='AdamW', lr0=0.002, lrf=0.01,
            cos_lr=True, cls_pw=0.0, patience=100, close_mosaic=10,
            seed=args.seed, deterministic=True, amp=args.amp, save_period=10,
            # Explicit common augmentation values, independent of COCO checkpoint recipes.
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
            scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.0, copy_paste=0.0,
        )
    elapsed = time.perf_counter() - started
    run_dir = Path(model.trainer.save_dir) if not args.evaluate_only else work
    best = Path(model.trainer.best) if not args.evaluate_only else args.evaluate_only.resolve()
    metadata = {'experiment': args.experiment, 'simam_base': args.simam_base if args.experiment == 'E6' else None,
                'dataset': provenance, 'checkpoint': str(best), 'elapsed_before_evaluation_seconds': elapsed,
                'ultralytics': ultralytics.__version__, 'torch': torch.__version__, 'python': platform.python_version(),
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'cli': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (run_dir / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Bundle custom modules/config so custom checkpoints remain usable after cloud sessions end.
    support = run_dir / 'support'
    support.mkdir(exist_ok=True)
    for filename in ('modules.py', 'architectures.py', 'experiment_trainer.py', 'yolo26_base.yaml', 'requirements.txt'):
        shutil.copy2(ROOT / filename, support / filename)
    shutil.copy2(model_yaml, support / model_yaml.name)
    metrics = model.val(**common, name=name + '_' + args.eval_split, split=args.eval_split)
    summary = {'experiment': args.experiment, 'simam_base': args.simam_base if args.experiment == 'E6' else None,
               'seed': args.seed, 'split': args.eval_split,
               'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
               'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
               'speed_ms': metrics.speed, 'checkpoint': str(best),
               'per_class': [{ 'name': model.names[int(cls)], 'AP50': float(metrics.box.ap50[i]),
                              'AP50_95': float(metrics.box.ap[i]), 'precision': float(metrics.box.p[i]),
                              'recall': float(metrics.box.r[i])}
                             for i, cls in enumerate(metrics.box.ap_class_index)]}
    (Path(metrics.save_dir) / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Checkpoints and metadata: {run_dir}')
    print(f'Evaluation: {metrics.save_dir}')
    return run_dir


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    run()
