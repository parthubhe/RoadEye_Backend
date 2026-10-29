"""Generate a standalone Colab/Kaggle RF-DETR notebook from the local dataset helper."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'RFDETR_Nano_v5.ipynb'


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    helper = (ROOT / 'rfdetr_dataset.py').read_text(encoding='utf-8')
    cells = [
        cell('markdown', '''# RoadEye v5 — RF-DETR Nano

Fine-tune a COCO-pretrained RF-DETR Nano on the **full five-class normal v5 dataset** from Hugging Face. Upload this notebook directly to **Colab or Kaggle**, enable a GPU and enable Internet access. Kaggle users should save a notebook version with outputs before the session ends. The complete training run may exceed a single free GPU session; preserve `last.ckpt` if you plan to resume.

The v5 Hugging Face revision is pinned. A local working copy omits zero-size or out-of-range label rows and writes a report; image files and the downloaded snapshot stay intact. Validation selects the checkpoint. The full test set is evaluated only if you change `RUN_TEST` to `True` after choosing your configuration.

RF-DETR uses COCO-style mAP reporting. Check metric names and evaluation settings before comparing numerical values with Ultralytics YOLO. No result above 90% is assumed.

References: [RF-DETR v1.11.0 training](https://github.com/roboflow/rf-detr/blob/1.11.0/docs/learn/train/index.md), [YOLO dataset layout](https://github.com/roboflow/rf-detr/blob/1.11.0/docs/learn/train/dataset-formats.md), [memory settings](https://github.com/roboflow/rf-detr/blob/1.11.0/docs/learn/train/advanced.md).
'''),
        cell('code', '''# 1. Install in the active Colab/Kaggle runtime. Restart the runtime if it had an older rfdetr loaded.
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'rfdetr[train]==1.11.0', 'huggingface_hub>=0.35,<2', 'PyYAML>=6,<7'], check=True)
import importlib.metadata
import torch
assert importlib.metadata.version('rfdetr') == '1.11.0'
if not torch.cuda.is_available():
    raise RuntimeError('Enable a GPU runtime before training.')
print('RF-DETR:', importlib.metadata.version('rfdetr'), '| PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0), '| VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
'''),
        cell('code', '''# 2. Configure the experiment. On an 8 GB GPU, batch 1 + accumulation is a conservative start.
from pathlib import Path
import os, sys, json, uuid, platform
from datetime import datetime, timezone

BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else (
    Path('/content') if Path('/content').exists() else Path.cwd())
WORKSPACE = BASE / 'roadeeye_rfdetr_workspace'
PROJECT = BASE / 'roadeeye_rfdetr_runs'
WORKSPACE.mkdir(parents=True, exist_ok=True)
PROJECT.mkdir(parents=True, exist_ok=True)
os.environ['RF_HOME'] = str(WORKSPACE / 'model_cache')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

MODEL_SIZE = 'nano'       # Optional: 'small' for a separate capacity experiment
RESOLUTION = 384         # Nano default. Change to 512 after the first run if memory permits.
EPOCHS = 40             # Total target epochs; RF-DETR may stop early on validation.
SEED = 0
NUM_WORKERS = 2
RUN_TEST = False        # Turn on only after model selection on validation.
RESUME_FROM = None      # Existing run's last.ckpt to continue after interruption.
INCLUDE_LAST_CKPT_IN_ZIP = True
DOWNLOAD_ZIP_IN_COLAB = False

vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
BATCH_SIZE = 1 if vram_gib < 12 else 4
GRAD_ACCUM_STEPS = 16 // BATCH_SIZE
EVAL_BATCH_SIZE = 1 if vram_gib < 12 else 2
if MODEL_SIZE not in ('nano', 'small') or RESOLUTION % 32:
    raise ValueError('Use nano/small and a resolution divisible by 32.')
print('Batch:', BATCH_SIZE, 'accumulation:', GRAD_ACCUM_STEPS,
      'eval batch:', EVAL_BATCH_SIZE, 'resolution:', RESOLUTION)
'''),
        cell('code', '''# 3. Read HF_TOKEN from a secret if the repository requires it. Do not paste a token into the notebook.
if not os.getenv('HF_TOKEN'):
    token = None
    try:
        if BASE.as_posix() == '/kaggle/working':
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        elif BASE.as_posix() == '/content':
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
    except (ImportError, KeyError, ValueError, OSError):
        pass
    if token:
        os.environ['HF_TOKEN'] = token
    del token
print('Hugging Face authentication:', 'token available' if os.getenv('HF_TOKEN') else 'anonymous')
'''),
        cell('code', '''# 4. Embed the dataset helper. This notebook is self-contained; no repository checkout is needed.
PACKAGE = WORKSPACE / 'notebook_support'
PACKAGE.mkdir(parents=True, exist_ok=True)
HELPER_SOURCE = ''' + repr(helper) + '''
(PACKAGE / 'rfdetr_dataset.py').write_text(HELPER_SOURCE, encoding='utf-8')
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from rfdetr_dataset import REPO_ID, REVISION, NAMES, EXPECTED_COUNTS, locate_dataset, prepare_overlay
print('Dataset:', REPO_ID, '@', REVISION)
'''),
        cell('code', '''# 5. Download the exact published v5 revision. About 27,000 files; allow time and disk space.
from huggingface_hub import snapshot_download
SNAPSHOT = Path(snapshot_download(
    repo_id=REPO_ID, repo_type='dataset', revision=REVISION,
    cache_dir=str(WORKSPACE / 'hf_cache'), token=os.getenv('HF_TOKEN') or None,
    allow_patterns=['data.yaml', 'train/**', 'valid/**', 'test/**',
                    'final_dataset_v5/**'], ignore_patterns=['*.cache', '*.pt'],
    max_workers=8,
))
SOURCE = locate_dataset(SNAPSHOT)
print('Source root:', SOURCE)
'''),
        cell('code', '''# 6. Make a portable YOLO dataset directory for RF-DETR using hard-linked images.
# This shares image bytes with the Hub cache but keeps all paths inside the new dataset root.
DATASET_DIR = WORKSPACE / ('v5_rfdetr_' + REVISION[:12])
provenance = prepare_overlay(SOURCE, DATASET_DIR)
provenance.update({'repo_id': REPO_ID, 'revision': REVISION})
(DATASET_DIR / 'dataset_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
assert provenance['counts'] == EXPECTED_COUNTS
assert (DATASET_DIR / 'test/images').is_dir()
print('Full-v5 split counts:', provenance['counts'])
print('Five classes:', ', '.join(NAMES))
print('Invalid box rows omitted in overlay:', provenance['removed_label_row_count'])
for row in provenance['removed_label_rows']:
    print('  ', row['split'], row['label'], 'line', row['line'])
print('Local dataset YAML:', DATASET_DIR / 'data.yaml')
'''),
        cell('code', '''# 7. Fine-tune from the official COCO checkpoint. A fresh run always gets a unique folder.
from rfdetr import RFDETRNano, RFDETRSmall
MODEL_CLASS = {'nano': RFDETRNano, 'small': RFDETRSmall}[MODEL_SIZE]
resume_path = None
if RESUME_FROM:
    resume_path = Path(RESUME_FROM).resolve()
    if not resume_path.is_file() or resume_path.suffix != '.ckpt':
        raise FileNotFoundError('RESUME_FROM must be an existing full .ckpt checkpoint.')
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
RUN_DIR = PROJECT / f'rfdetr_{MODEL_SIZE}_v5_normal_{RESOLUTION}_seed{SEED}_{stamp}_{uuid.uuid4().hex[:6]}'
RUN_DIR.mkdir(parents=True, exist_ok=False)

config = dict(dataset_file='yolo', dataset_dir=str(DATASET_DIR), output_dir=str(RUN_DIR),
              epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
              eval_batch_size=EVAL_BATCH_SIZE, resolution=RESOLUTION, device='cuda',
              num_workers=NUM_WORKERS, amp_dtype='auto', seed=SEED,
              checkpoint_interval=5, early_stopping=True, early_stopping_patience=10,
              skip_best_epochs=2, use_ema=True, tensorboard=False,
              log_per_class_metrics=True)
if RESUME_FROM:
    config['resume'] = str(resume_path)
metadata = {'model': MODEL_SIZE, 'rfdetr': importlib.metadata.version('rfdetr'),
            'torch': torch.__version__, 'python': platform.python_version(),
            'gpu': torch.cuda.get_device_name(0), 'repo_id': REPO_ID, 'revision': REVISION,
            'dataset_provenance': provenance, 'training_config': config,
            'test_evaluation_is_manual': True}
(RUN_DIR / 'roadeeye_experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Training output:', RUN_DIR)
print('Expected nominal batch:', BATCH_SIZE * GRAD_ACCUM_STEPS)
model = MODEL_CLASS(gradient_checkpointing=True)
model.train(**config)
print('Training finished. Best checkpoint:', RUN_DIR / 'checkpoint_best_total.pth')
'''),
        cell('code', '''# 8. Evaluate the best validation-selected checkpoint. Test evaluation is opt-in above.
best_checkpoint = RUN_DIR / 'checkpoint_best_total.pth'
if not best_checkpoint.is_file():
    raise FileNotFoundError(f'Best checkpoint missing: {best_checkpoint}')
best_model = MODEL_CLASS.from_checkpoint(str(best_checkpoint))
evaluation_args = dict(dataset_file='yolo', dataset_dir=str(DATASET_DIR),
                       resolution=RESOLUTION, device='cuda',
                       batch_size=EVAL_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
                       num_workers=NUM_WORKERS)
validation_metrics = best_model.evaluate(split='val', **evaluation_args)
(RUN_DIR / 'best_validation_metrics.json').write_text(
    json.dumps(validation_metrics, indent=2, default=str), encoding='utf-8')
print('Best-checkpoint validation metrics:', validation_metrics)
if RUN_TEST:
    if provenance['counts']['test'] != EXPECTED_COUNTS['test'] or not (DATASET_DIR / 'test/images').is_dir():
        raise FileNotFoundError('The independent test split is missing.')
    test_metrics = best_model.evaluate(split='test', **evaluation_args)
    (RUN_DIR / 'full_v5_test_metrics.json').write_text(
        json.dumps(test_metrics, indent=2, default=str), encoding='utf-8')
    print('FULL v5 test metrics:', test_metrics)
'''),
        cell('code', '''# 9. Export a portable run archive. Kaggle: save a notebook version with outputs.
from zipfile import ZipFile, ZIP_DEFLATED
files_to_export = [RUN_DIR / 'checkpoint_best_total.pth',
                   RUN_DIR / 'roadeeye_experiment.json',
                   RUN_DIR / 'training_config.json',
                   RUN_DIR / 'best_validation_metrics.json',
                   RUN_DIR / 'full_v5_test_metrics.json']
if INCLUDE_LAST_CKPT_IN_ZIP:
    files_to_export.append(RUN_DIR / 'last.ckpt')
ZIP_PATH = BASE / f'{RUN_DIR.name}_results.zip'
with ZipFile(ZIP_PATH, 'w', ZIP_DEFLATED, compresslevel=3) as archive:
    for path in files_to_export:
        if path.is_file():
            archive.write(path, path.name)
    archive.write(DATASET_DIR / 'dataset_provenance.json', 'dataset_provenance.json')
print('Saved:', ZIP_PATH, '| MiB:', round(ZIP_PATH.stat().st_size / 2**20, 1))
if DOWNLOAD_ZIP_IN_COLAB and BASE.as_posix() == '/content':
    from google.colab import files
    files.download(str(ZIP_PATH))
'''),
        cell('markdown', '''## Running again

- For a memory error, use `BATCH_SIZE = 1`, `EVAL_BATCH_SIZE = 1` and `RESOLUTION = 384` in cell 2, restart the runtime, and rerun. Gradient checkpointing is already enabled.
- For an interrupted run, set `RESUME_FROM` to the run's `last.ckpt`, then rerun the training cell. A lightweight `.pth` file does not restore the optimizer schedule.
- To evaluate the full test split after choosing a configuration, run `RUN_TEST = True` in a new cell, then rerun **cell 8** and cell 9 to refresh the archive.
- The dataset contains full v5 train/valid/test splits. This notebook does not use the diagnostic v6 subset. The temporary RF-DETR overlay may omit malformed annotation rows; the provenance file lists each one.
'''),
    ]
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU',
        'colab': {'name': OUTPUT.name, 'provenance': []},
    }, 'nbformat': 4, 'nbformat_minor': 5}
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Generated {OUTPUT}')


if __name__ == '__main__':
    main()
