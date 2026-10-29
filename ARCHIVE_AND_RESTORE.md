# Project artifact archive

The source code and reports are retained in GitHub. Large local datasets, model
checkpoints, training artifacts and the virtual environment are being retired
after backup verification. The inference server needs model files restored
before it can run again.

Hugging Face repository:
https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5

Existing releases remain under `models/`. Additional checkpoints, including
older pothole models, are stored under `models/archive_20261029/weights/`.
`models/archive_20261029/model_inventory.json` maps original paths to remote
files and SHA-256 checksums. Duplicate local files share one remote backup.

After creating a Python environment, use `requirements-pipescout.txt` for the
PipeScout dependency snapshot. The older root `requirements.txt` belongs to the
original pothole setup. Select an appropriate PyTorch/CUDA build for the target
machine. To list archived model paths:

```powershell
python scripts\restore_archived_models.py
```

Restore all checkpoints from one run by matching part of its original path:

```powershell
python scripts\restore_archived_models.py --match rfdetr_nano_v5_normal_512_seed0_20261003
python scripts\restore_archived_models.py --match yolo26s_v5_cctv/
```

The restore script verifies hashes and writes files to their original locations.
The published normal v5 dataset is available in the same Hugging Face repository.
Historical locally generated dataset variants are not all published datasets.
`cleanup_archive/` is a local audit directory excluded from Git; consult its
verification records and deletion log for the exact cleanup outcome.
