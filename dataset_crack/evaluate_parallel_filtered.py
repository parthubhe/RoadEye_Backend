"""Evaluate the same checkpoint on full v5 and its model-selected diagnostic subset."""

import argparse
import json
from pathlib import Path

import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from attention_modules import ParallelAttention, SimAM, CBAM, SqueezeExcitation

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    for module in (ParallelAttention, SimAM, CBAM, SqueezeExcitation):
        setattr(tasks, module.__name__, module)
    checkpoint = ROOT / 'runs/pipe_proto/yolo26s_v5_parallel_simam_cbam_se/weights/best.pt'
    for label, dataset in (
        ('full_v5', ROOT / 'final_dataset_v5/data.yaml'),
        ('filtered_diagnostic', ROOT / 'v5_error_filtered_diagnostic/data.yaml'),
    ):
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(dataset), split='test', imgsz=640, batch=args.batch,
            device=args.device, workers=0, plots=True,
            project=str(ROOT / 'runs/pipe_proto'),
            name=f'yolo26s_parallel_{label}_test', exist_ok=False,
        )
        result = {
            'checkpoint': str(checkpoint), 'data': str(dataset), 'split': 'test',
            'subset': label, 'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr), 'mAP50': float(metrics.box.map50),
            'mAP50_95': float(metrics.box.map), 'speed_ms': metrics.speed,
            'note': 'Filtered subset was selected using model errors; its scores are selection-biased.',
            'per_class': [
                {'class': model.names[int(cls)], 'precision': float(metrics.box.p[i]),
                 'recall': float(metrics.box.r[i]), 'mAP50': float(metrics.box.ap50[i]),
                 'mAP50_95': float(metrics.box.ap[i])}
                for i, cls in enumerate(metrics.box.ap_class_index)
            ],
        }
        output = Path(metrics.save_dir) / 'evaluation_summary.json'
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'Summary: {output}', flush=True)


if __name__ == '__main__':
    main()
