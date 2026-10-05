r"""Extract only the publishable E3 checkpoint and its support files.

Run from the RoadEye root:
    venv\Scripts\python.exe dataset_crack\models\extract_e3.py

The original Kaggle results ZIP is left untouched. Existing destinations are
never overwritten, and the large periodic/last training checkpoints are skipped.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT / "results_E3_p2_weighted_fusion_v5_normal_seed0_20260928_165814_0161fd.zip"
ROOT_IN_ZIP = "E3_p2_weighted_fusion_v5_normal_seed0_20260928_165814_0161fd"
VAL_IN_ZIP = ROOT_IN_ZIP + "_val"
DESTINATION = Path(__file__).resolve().parent / "p2_weighted_fusion"
FILES = {
    f"{ROOT_IN_ZIP}/weights/best.pt": "best.pt",
    f"{ROOT_IN_ZIP}/support/modules.py": "support/modules.py",
    f"{ROOT_IN_ZIP}/support/architectures.py": "support/architectures.py",
    f"{ROOT_IN_ZIP}/support/experiment_trainer.py": "support/experiment_trainer.py",
    f"{ROOT_IN_ZIP}/support/yolo26_base.yaml": "support/yolo26_base.yaml",
    f"{ROOT_IN_ZIP}/support/yolo26s_E3.yaml": "support/yolo26s_E3.yaml",
    f"{ROOT_IN_ZIP}/support/requirements.txt": "support/requirements.txt",
    f"{ROOT_IN_ZIP}/experiment_metadata.json": "experiment_metadata.json",
    f"{VAL_IN_ZIP}/summary.json": "validation_summary.json",
}


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"E3 archive not found: {ARCHIVE}")
    if any((DESTINATION / relative).exists() for relative in FILES.values()):
        raise FileExistsError(f"Refusing to overwrite an existing E3 release file in {DESTINATION}")
    with ZipFile(ARCHIVE) as archive:
        missing = set(FILES) - set(archive.namelist())
        if missing:
            raise ValueError(f"E3 archive is missing required entries: {sorted(missing)}")
        for entry, relative in FILES.items():
            target = DESTINATION / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            print(f"Extracted {relative} ({target.stat().st_size:,} bytes)")
    (DESTINATION / "release_manifest.json").write_text(
        json.dumps({"source_archive": ARCHIVE.name, "checkpoint": "best.pt",
                    "validation_only": True, "test_evaluation_available": False}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
