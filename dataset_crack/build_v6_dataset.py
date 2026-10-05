"""Create a local v6 diagnostic dataset excluding the exact FP/FN CSV image union.

Uses YOLO image manifests; all original v5 images and labels are preserved.
The supplied CSVs contain test errors, not training errors.
"""
import argparse
from pathlib import Path

from create_error_filtered_dataset import DEFAULT_ANALYSIS, ROOT, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'final_dataset_v5')
    parser.add_argument('--analysis', type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument('--output', type=Path, default=ROOT / 'final_dataset_v6')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    build(args.source, args.analysis, args.output, args.dry_run)


if __name__ == '__main__':
    main()
