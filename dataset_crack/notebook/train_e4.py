"""Standard YOLO26m: a capacity comparison without added attention."""
from multiprocessing import freeze_support
from run_experiment import run

if __name__ == '__main__':
    freeze_support()
    run(default_experiment='E4')
