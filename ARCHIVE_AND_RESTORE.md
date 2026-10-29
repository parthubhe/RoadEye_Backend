# Project artifact archive

The source code and reports are retained in GitHub. Large local datasets, model
checkpoints, training artifacts and the virtual environment were removed on
29 October 2026 after model-backup verification. The inference server needs model files restored
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

## Completed cleanup

- 116 local model files (107 unique SHA-256 hashes) verified in Hugging Face.
- 102 previously missing unique checkpoints uploaded; five already existed.
- Archive inventory revision: `6d3eec65c2ccbbe2564a2eb0f6881fdacb9ef6e3`.
- 389,802 dataset/output/cache files and 116 local model files removed, plus the local `venv`.
- About 41.7 GiB of actual disk space recovered; 104.9 GiB free immediately afterward.
- Source code, reports, presentations, bundled web demos, and small tracked dataset metadata retained.
- Standalone graphs, previews, historical data variants, and mask payloads were discarded without
  a new evidence archive, as requested. The published normal v5 data remains on Hugging Face.
- Local deletion audit is compressed in `cleanup_archive/deletion_audit.zip`;
  the verified weight mapping is `cleanup_archive/verified_model_backups.json`.

The final scan found no `.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors`, `.tflite`,
or `.engine` model files in the project. The retained workspace payload was about
84 MiB excluding Git history. Shared caches or model libraries outside this workspace were untouched.
