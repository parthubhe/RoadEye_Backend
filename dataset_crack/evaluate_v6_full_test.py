"""Evaluate the v6 baseline checkpoint on the unchanged, full v5 test split."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = 'yolo26s_v6_normal_error_filtered_diagnostic_seed0_20260929_184244_3e235d15'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'runs/pipe_proto' / RUN / 'weights/best.pt')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    # Keep evaluation's framework settings inside the project workspace.
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'notebook/workspace/ultralytics_config'))
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    data = ROOT / 'final_dataset_v5/data.yaml'
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data), split='test', imgsz=640, batch=args.batch,
        device=args.device, workers=0, plots=True,
        project=str(ROOT / 'runs/pipe_proto'),
        name=weights.parent.parent.name + '_full_v5_test', exist_ok=False,
    )
    summary = {
        'checkpoint': str(weights), 'data': str(data), 'split': 'full unchanged v5 test',
        'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
        'speed_ms': metrics.speed,
        'per_class': [
            {'class': model.names[int(cls)], 'precision': float(metrics.box.p[i]),
             'recall': float(metrics.box.r[i]), 'mAP50': float(metrics.box.ap50[i]),
             'mAP50_95': float(metrics.box.ap[i])}
            for i, cls in enumerate(metrics.box.ap_class_index)
        ],
    }
    output = Path(metrics.save_dir) / 'evaluation_summary.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Summary saved to: {output}')


if __name__ == '__main__':
    main()
