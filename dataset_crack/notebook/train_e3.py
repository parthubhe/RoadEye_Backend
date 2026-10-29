"""YOLO26s with P2 and weighted residual fusion at all five merge points."""
from multiprocessing import freeze_support
from run_experiment import run

if __name__ == '__main__':
    freeze_support()
    run(default_experiment='E3')
