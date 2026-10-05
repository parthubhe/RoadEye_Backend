"""YOLO26s with learned weighted-sum residual fusion at the four PAN merge points. This is not a full BiFPN."""
from multiprocessing import freeze_support
from run_experiment import run

if __name__ == '__main__':
    freeze_support()
    run(default_experiment='E2')
