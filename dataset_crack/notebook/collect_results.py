"""Combine saved validation summaries after local runs or extracting cloud result ZIPs."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parent / 'runs')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    args = p.parse_args()
    rows = []
    for path in args.project.rglob('summary.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('split') == args.split:
            rows.append({key: data.get(key) for key in ['experiment', 'simam_base', 'seed', 'split', 'precision', 'recall', 'mAP50', 'mAP50_95']} | {'file': str(path)})
    rows.sort(key=lambda x: x['mAP50_95'], reverse=True)
    if not rows:
        raise SystemExit('No matching evaluation summaries found')
    output = args.project / f'comparison_{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())
