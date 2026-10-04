"""Train plain YOLO26s on local v6; model-selected test results are diagnostic only.

Uses the current v6 manifests and reports their actual split exclusion counts.
No custom attention, CLAHE preprocessing, dataset download or v5 modification.
"""
import argparse
import json
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'final_dataset_v6/data.yaml')
    parser.add_argument('--weights', type=Path, default=ROOT.parent / 'yolo26s.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project', type=Path, default=ROOT / 'runs/pipe_proto')
    parser.add_argument('--evaluate-filtered-test', action='store_true',
                        help='After training evaluate best.pt on the model-selected diagnostic test subset')
    parser.add_argument('--dry-run', action='store_true', help='Check paths and print settings without loading/training a model')
    return parser.parse_args()


def main():
    args = arguments()
    data, weights = args.data.resolve(), args.weights.resolve()
    if not data.is_file():
        raise FileNotFoundError(f'{data}: run dataset_crack/build_v6_dataset.py first')
    if not weights.is_file():
        raise FileNotFoundError(f'Local baseline weights not found: {weights}. Pass --weights path/to/yolo26s.pt')
    report_path = data.parent / 'filter_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 64 or args.imgsz % 32 or args.workers < 0:
        raise ValueError('Positive epochs/batch, imgsz >=64 divisible by 32, and workers >=0 required')
    print('V6 is a model-error-selected diagnostic dataset, NOT an independent test benchmark.')
    for split, stats in report['split_counts'].items():
        print(f"{split}: {stats['before']} -> {stats['retained']} images; excluded {stats['excluded']}")
    if report['split_counts']['train']['excluded'] == 0:
        print('IMPORTANT: no training images were removed. Retraining is optional; the training data are unchanged.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    name = f'yolo26s_v6_normal_error_filtered_diagnostic_seed{args.seed}_{stamp}_{uuid4().hex[:8]}'
    settings = dict(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=args.seed, deterministic=True,
        optimizer='AdamW', lr0=0.002, lrf=0.01, cos_lr=True, cls_pw=0.0,
        patience=100, close_mosaic=10, amp=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
        scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        project=str(args.project.resolve()), name=name, exist_ok=False,
        plots=True, save=True, save_period=10,
    )
    if args.dry_run:
        print(json.dumps({'weights': str(weights), 'training': settings,
                          'evaluate_filtered_test': args.evaluate_filtered_test}, indent=2))
        return
    from ultralytics import YOLO
    model = YOLO(str(weights))
    model.train(**settings)
    run_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)
    metadata = {'experiment': 'plain YOLO26s v6 diagnostic', 'filter_report': report,
                'weights': str(weights), 'best_checkpoint': str(best), 'settings': settings,
                'warning': 'See filter_report for the current training exclusions; filtered test is selection-biased.'}
    (run_dir / 'v6_experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if args.evaluate_filtered_test:
        metrics = YOLO(str(best)).val(
            data=str(data), split='test', imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, plots=True,
            project=str(args.project.resolve()), name=name + '_filtered_test', exist_ok=False,
        )
        summary = {'split': 'model-selected filtered test (diagnostic only)',
                   'checkpoint': str(best), 'mAP50': float(metrics.box.map50),
                   'mAP50_95': float(metrics.box.map), 'precision': float(metrics.box.mp),
                   'recall': float(metrics.box.mr)}
        (Path(metrics.save_dir) / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
    print(f'Run saved to: {run_dir}')


if __name__ == '__main__':
    freeze_support()
    main()
