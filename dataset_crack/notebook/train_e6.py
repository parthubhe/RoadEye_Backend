"""One SimAM block on the finest detection feature of a configurable E1-E5 parent. Default E3 is provisional; choose using validation."""
from multiprocessing import freeze_support
from run_experiment import run

if __name__ == '__main__':
    freeze_support()
    run(default_experiment='E6')
