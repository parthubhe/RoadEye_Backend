"""Run a predefined batch sequentially on one GPU. Separate cloud sessions can run concurrently."""
import argparse
import subprocess
import sys
from pathlib import Path

GROUPS = {'A': ['E1', 'E2'], 'B': ['E3', 'E4'], 'C': ['E5', 'E6']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                epilog='E1-E6 are experiment IDs, not batch names. For E5 alone: python train_e5.py')
    p.add_argument('--group', choices=GROUPS, required=True,
                   help='A: E1/E2; B: E3/E4; C: E5/E6. Uses local v5 by default.')
    p.add_argument('training_args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    forwarded = args.training_args
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    if any(x in forwarded for x in ['--experiment', '--resume', '--evaluate-only']):
        p.error('Use individual experiment scripts for resume/evaluation; batch chooses experiment IDs')
    for experiment in GROUPS[args.group]:
        command = [sys.executable, str(Path(__file__).with_name('run_experiment.py')),
                   '--experiment', experiment, *forwarded]
        print(f'Running {experiment}', flush=True)
        subprocess.run(command, check=True)
