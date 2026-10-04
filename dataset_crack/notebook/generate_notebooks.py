"""Rebuild six standalone notebooks and thin Python entry points from local source.

Generated notebook bootstrap cells embed the shared implementation, so users only
need to upload one .ipynb to Colab/Kaggle; no GitHub checkout is required.
"""
import json
from pathlib import Path

from architectures import EXPERIMENTS, write_config

ROOT = Path(__file__).resolve().parent
FILES = ['modules.py', 'architectures.py', 'dataset.py', 'experiment_trainer.py',
         'run_experiment.py', 'yolo26_base.yaml', 'requirements.txt']
DESCRIPTIONS = {
    'E1': 'YOLO26s with an additional P2/4 detection branch. The original P3-P5 branches are retained.',
    'E2': 'YOLO26s with learned weighted-sum residual fusion at the four PAN merge points. This is not a full BiFPN.',
    'E3': 'YOLO26s with P2 and weighted residual fusion at all five merge points.',
    'E4': 'Standard YOLO26m: a capacity comparison without added attention.',
    'E5': 'YOLO26s with one residual depthwise horizontal/vertical convolution block at P3.',
    'E6': 'One SimAM block on the finest detection feature of a configurable E1-E5 parent. Default E3 is provisional; choose using validation.',
}


def cell(kind, source):
    data = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        data.update(execution_count=None, outputs=[])
    return data


def main():
    files = {name: (ROOT / name).read_text(encoding='utf-8') for name in FILES}
    bootstrap = '''from pathlib import Path
import sys, os
BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else (Path('/content') if Path('/content').exists() else Path.cwd())
PACKAGE = BASE / 'roadeeye_experiments'
PACKAGE.mkdir(parents=True, exist_ok=True)
FILES = ''' + repr(files) + '''
for name, content in FILES.items():
    (PACKAGE / name).write_text(content, encoding='utf-8')
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
print('Experiment code:', PACKAGE)
'''
    auth = '''# Add HF_TOKEN as a Colab secret or Kaggle secret with READ access to the private dataset.
# Public repositories work without a token. Never paste tokens into saved cells.
import os
if not os.getenv('HF_TOKEN'):
    try:
        if str(BASE) == '/kaggle/working':
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        elif str(BASE) == '/content':
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        else:
            token = None
    except Exception:
        token = None
    if token:
        os.environ['HF_TOKEN'] = token
    del token
# If secrets are unavailable and the dataset is private, uncomment:
# from getpass import getpass
# os.environ['HF_TOKEN'] = getpass('HF read token: ')
'''
    install = '''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PACKAGE / 'requirements.txt')], check=True)
import torch, ultralytics
print('Torch:', torch.__version__, '| Ultralytics:', ultralytics.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - enable GPU runtime')
'''
    for exp, label in EXPERIMENTS.items():
        launcher = f'''"""{DESCRIPTIONS[exp]}"""
from multiprocessing import freeze_support
from run_experiment import run

if __name__ == '__main__':
    freeze_support()
    run(default_experiment='{exp}')
'''
        (ROOT / f'train_{exp.lower()}.py').write_text(launcher, encoding='utf-8')
        write_config(exp, ROOT / 'configs')
        config = f'''EXPERIMENT = '{exp}'
SIMAM_BASE = 'E3'  # E6 only: E1, E2, E3, E4 or E5; choose using validation
REPO_ID = 'AkumaDachi/roadeeye-sewer-defects-v5'
REVISION = 'd88c0ed4f0fd39253eb3866bf2feb9dcf808bae5'  # Verified full-v5 snapshot; same for every experiment
EPOCHS = 80
IMGSZ = 640
BATCH = 16  # If OOM, use 8 or 4 across comparable runs; record the change
SEED = 0
WORKERS = 2
PROJECT = BASE / 'roadeeye_runs'
WORKSPACE = BASE / 'roadeeye_workspace'
LOCAL_DATA = None  # Optional already-downloaded v5 dataset root
RESUME = None  # Path to interrupted run's weights/last.pt
MOUNT_DRIVE = False  # Colab: set True to store runs/checkpoints persistently on Drive
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = Path('/content/drive/MyDrive/RoadEye/experiments')
PROJECT.mkdir(parents=True, exist_ok=True)
'''
        prepare = '''from dataset import prepare_dataset
DATA_YAML, DATA_INFO = prepare_dataset(WORKSPACE / 'dataset', REPO_ID, REVISION, LOCAL_DATA)
print('Class names: crack, corrosion_rust, sediment_deposit, root_intrusion, joint_defect')
print('Resolved revision:', DATA_INFO['resolved_commit'])
# Pin the resolved commit for this session; copy it into the other notebooks too.
if DATA_INFO['resolved_commit']:
    REVISION = DATA_INFO['resolved_commit']
'''
        training = '''from run_experiment import run
RUN_ARGS = [
    '--experiment', EXPERIMENT, '--simam-base', SIMAM_BASE,
    '--repo-id', REPO_ID, '--revision', REVISION,
    '--workspace', str(WORKSPACE), '--project', str(PROJECT),
    '--epochs', str(EPOCHS), '--imgsz', str(IMGSZ), '--batch', str(BATCH),
    '--workers', str(WORKERS), '--seed', str(SEED), '--device', '0',
]
if LOCAL_DATA:
    RUN_ARGS += ['--local-data', str(LOCAL_DATA)]
else:
    RUN_ARGS += ['--download-data']  # Cloud notebooks explicitly opt into the Hub snapshot.
if RESUME:
    RUN_ARGS += ['--resume', str(RESUME)]
RUN_DIR = run(RUN_ARGS)  # Trains, then evaluates VALIDATION. No automatic test-set selection.
BEST = RUN_DIR / 'weights/best.pt'
print('Best checkpoint:', BEST)
'''
        test = '''# Optional final full-test evaluation, after choosing configurations on validation.
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    from modules import register_modules
    from ultralytics import YOLO
    import json
    register_modules()
    model = YOLO(str(BEST))
    metrics = model.val(data=str(DATA_YAML), split='test', imgsz=IMGSZ, batch=BATCH,
                        device=0, workers=WORKERS, plots=True, project=str(PROJECT),
                        name=RUN_DIR.name + '_full_test', exist_ok=False)
    result = {'experiment': EXPERIMENT, 'simam_base': SIMAM_BASE if EXPERIMENT == 'E6' else None,
              'seed': SEED, 'split': 'test', 'checkpoint': str(BEST), 'mAP50': float(metrics.box.map50),
              'mAP50_95': float(metrics.box.map), 'precision': float(metrics.box.mp),
              'recall': float(metrics.box.mr)}
    (Path(metrics.save_dir) / 'summary.json').write_text(json.dumps(result, indent=2))
    print(result)
'''
        archive = '''# Export weights, configs, CSVs, plots and custom modules. Dataset cache is excluded.
import shutil
archive_path = shutil.make_archive(str(BASE / ('results_' + RUN_DIR.name)), 'zip', str(PROJECT))
print('Save/download this before ending the runtime:', archive_path)
# Colab browser download (optional):
# from google.colab import files
# files.download(archive_path)
# Kaggle: Save Version and retain output files; download the ZIP from Output.
'''
        cells = [
            cell('markdown', f'# {exp}: {label}\n\n{DESCRIPTIONS[exp]}\n\n'
                 'Normal full v5, five classes, 80 epochs at 640 by default. E0 is intentionally omitted. '
                 'No accuracy improvement is assumed.\n\n'
                 '**Colab:** enable GPU under Runtime. **Kaggle:** enable GPU and Internet in notebook settings. '
                 'Import this notebook and run cells in order; it includes all custom code. Use a read-access HF_TOKEN secret '
                 'if your dataset is private. One experiment per GPU session.\n\n'
                 'Keep the same dataset commit, seed, batch and training settings across comparisons. '
                 'Full training is not started until the training cell. Estimated time depends on allocated GPU.'),
            cell('markdown', '## 1. Extract embedded source files'), cell('code', bootstrap),
            cell('markdown', '## 2. Install tested dependencies (keep platform GPU PyTorch)'), cell('code', install),
            cell('markdown', '## 3. Authenticate to the dataset'), cell('code', auth),
            cell('markdown', '## 4. Configure experiment and persistent output'), cell('code', config),
            cell('markdown', '## 5. Download v5, verify classes/counts, fix YAML paths'), cell('code', prepare),
            cell('markdown', '## 6. Train and evaluate validation'), cell('code', training),
            cell('markdown', '## 7. Optional final evaluation on the full test split'), cell('code', test),
            cell('markdown', '## 8. Export results before ending the cloud session'), cell('code', archive),
        ]
        notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python', 'version': '3.11'},
                     'accelerator': 'GPU', 'colab': {'name': f'{exp}_{label}.ipynb', 'provenance': []}},
                    'nbformat': 4, 'nbformat_minor': 5}
        for i, c in enumerate(cells):
            c['id'] = f'{exp.lower()}-cell-{i:02d}'
        (ROOT / f'{exp}_{label}.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    print('Generated six self-contained notebooks, six launchers and six model YAMLs.')


if __name__ == '__main__':
    main()
