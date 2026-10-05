"""YOLO26s with an additional P2/4 detection branch. The original P3-P5 branches are retained."""
from multiprocessing import freeze_support
from run_experiment import run

if __name__ == '__main__':
    freeze_support()
    run(default_experiment='E1')
