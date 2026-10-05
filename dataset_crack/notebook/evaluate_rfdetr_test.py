r"""Evaluate the two completed local RF-DETR Nano runs on the held-out v5 test split.

Run from the RoadEye root:
    venv\Scripts\python.exe dataset_crack\notebook\evaluate_rfdetr_test.py

This does not train or modify the dataset. It refuses to replace existing test
results and records the checkpoint hash and dataset provenance used for each run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from train_rfdetr_local import load_rfdetr_nano


DATASET_CRACK = Path(__file__).resolve().parents[1]
RUNS = DATASET_CRACK / "runs" / "pipe_proto"
PREPARED = DATASET_CRACK / "rfdetr_local_workspace" / "final_dataset_v5_rfdetr_yolo"
DEFAULT_RUNS = (
    "rfdetr_nano_v5_normal_512_seed0_20261002_193202_f6bd13",
    "rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7",
)
PARTIAL_RUN = "rfdetr_nano_v5_normal_512_seed0_20261001_181318_e7e98d"
EXPECTED_TEST_IMAGES = 1986
REQUIRED_METRICS = (
    "test/mAP_50", "test/mAP_50_95", "test/precision", "test/recall",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_dataset_provenance() -> dict:
    provenance_path = PREPARED / "dataset_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("counts", {}).get("test") != EXPECTED_TEST_IMAGES:
        raise ValueError("Prepared dataset provenance does not identify the full v5 test split")
    if any(row.get("split") == "test" for row in provenance.get("removed_label_rows", [])):
        raise ValueError("Prepared dataset removed a test annotation")
    for kind in ("images", "labels"):
        actual = sum(p.is_file() for p in (PREPARED / "test" / kind).iterdir())
        if actual != EXPECTED_TEST_IMAGES:
            raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} test {kind}, found {actual}")
    return provenance


def main() -> None:
    # RF-DETR prints box-drawing tables even for non-interactive evaluations.
    # Windows' default cp1252 stdout cannot encode those characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", choices=(*DEFAULT_RUNS, PARTIAL_RUN),
                        help="Evaluate only this run; omit to evaluate both completed runs")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local evaluation")

    dataset = test_dataset_provenance()
    RFDETRNano = load_rfdetr_nano()
    for name in args.run or DEFAULT_RUNS:
        run = RUNS / name
        checkpoint = run / "checkpoint_best_total.pth"
        if name == PARTIAL_RUN:
            checkpoint = run / "checkpoint_best_ema.pth"
        output = run / "full_v5_test_metrics.json"
        evidence = run / "full_v5_test_evaluation.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if output.exists() or evidence.exists():
            raise FileExistsError(f"Existing evaluation will not be overwritten: {output} or {evidence}")
        run_provenance = json.loads((run / "dataset_provenance.json").read_text(encoding="utf-8"))
        if run_provenance["source_filename_label_sha256"]["test"] != dataset["source_filename_label_sha256"]["test"]:
            raise ValueError(f"Test split no longer matches the run provenance: {name}")

        print(f"Evaluating {name} on {EXPECTED_TEST_IMAGES} unchanged v5 test images", flush=True)
        model = RFDETRNano.from_checkpoint(str(checkpoint))
        metrics = model.evaluate(
            split="test", dataset_file="yolo", dataset_dir=str(PREPARED),
            resolution=512, batch_size=1, eval_batch_size=1,
            num_workers=args.workers, device="cuda", amp_dtype="auto",
        )
        if not isinstance(metrics, dict) or any(key not in metrics for key in REQUIRED_METRICS):
            raise ValueError(f"Unexpected RF-DETR test metrics: {metrics}")
        metrics = {key: float(value) for key, value in metrics.items()}
        record = {
            "split": "test",
            "test_images": EXPECTED_TEST_IMAGES,
            "dataset": str(PREPARED),
            "test_filename_label_sha256": dataset["source_filename_label_sha256"]["test"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"resolution": 512, "batch_size": 1, "eval_batch_size": 1,
                       "num_workers": args.workers, "device": "cuda", "amp_dtype": "auto"},
            "metrics": metrics,
        }
        with output.open("x", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
        with evidence.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        print(f"Saved {output}: mAP50={metrics['test/mAP_50']:.4f}, "
              f"mAP50-95={metrics['test/mAP_50_95']:.4f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
