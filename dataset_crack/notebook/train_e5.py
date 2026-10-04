"""YOLO26s with one residual depthwise horizontal/vertical convolution block at P3."""
from multiprocessing import freeze_support
from run_experiment import run

if __name__ == '__main__':
    freeze_support()
    run(default_experiment='E5')
