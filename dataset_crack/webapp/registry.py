"""Local checkpoint inventory and evidence-backed display metadata."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DATASET_ROOT.parent
PIPE_RUNS = DATASET_ROOT / "runs" / "pipe_proto"
LEGACY_RUNS = DATASET_ROOT / "runs" / "detect" / "runs" / "pipe_proto"
EXPERIMENT_RUNS = DATASET_ROOT / "notebook" / "runs"
PARAMETERS_FILE = Path(__file__).resolve().parent / "model_parameters.json"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    family: str
    checkpoint: Path | None
    run_dir: Path | None
    dataset: str
    status: str
    img_size: int
    metrics: dict[str, float | None]
    parameters: int | None
    notes: str = ""
    test_metrics: dict[str, float] = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "dataset": self.dataset,
            "status": self.status,
            "available": bool(self.checkpoint and self.checkpoint.is_file()),
            "img_size": self.img_size,
            "metrics": self.metrics,
            "test_metrics": self.test_metrics,
            "parameters": self.parameters,
            "notes": self.notes,
            "run": str(self.run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") if self.run_dir else None,
        }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _best_yolo_csv(path: Path) -> dict[str, float | None]:
    if not path.is_file():
        return {"precision": None, "recall": None, "map50": None, "map5095": None, "epoch": None, "logged_epochs": 0}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("metrics/mAP50-95(B)")]
    if not rows:
        return {"precision": None, "recall": None, "map50": None, "map5095": None, "epoch": None, "logged_epochs": 0}
    row = max(rows, key=lambda item: float(item["metrics/mAP50-95(B)"]))
    return {
        "precision": float(row["metrics/precision(B)"]),
        "recall": float(row["metrics/recall(B)"]),
        "map50": float(row["metrics/mAP50(B)"]),
        "map5095": float(row["metrics/mAP50-95(B)"]),
        "epoch": int(row["epoch"]),
        "logged_epochs": len(rows),
    }


def _rf_metrics(path: Path) -> dict[str, float | None]:
    values = _read_json(path)
    return {
        "precision": values.get("val/precision"),
        "recall": values.get("val/recall"),
        "map50": values.get("val/mAP_50"),
        "map5095": values.get("val/mAP_50_95"),
        "epoch": None,
        "logged_epochs": None,
    }


def _run_args(run: Path) -> dict:
    path = run / "args.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def _dataset_from_args(run: Path, name: str) -> str:
    if name.startswith(("E5_", "E6_")):
        return "v5 normal"
    data = _run_args(run)
    if data:
        try:
            dataset = str(data.get("data", "")).lower()
            if "final_dataset_v6" in dataset:
                return "v6 filtered"
            if "final_dataset_v5_clahe" in dataset:
                return "v5 CLAHE"
            if "final_dataset_v5" in dataset:
                return "v5 normal"
            if "final_dataset_v4_clahe" in dataset:
                return "v4 CLAHE"
            if "final_dataset_v4" in dataset:
                return "v4 CCTV"
            if "final_dataset_v3" in dataset:
                return "v3 merged"
            if "final_dataset_v2" in dataset:
                return "v2 merged"
            if "sewer_remapped_clahe" in dataset:
                return "legacy CLAHE"
            if "sewer_remapped" in dataset:
                return "legacy sewer remapped"
            if "final_merged" in dataset:
                return "older merged"
            if dataset:
                return Path(dataset).parent.name or "legacy"
        except (OSError, ValueError):
            pass
    if "v6" in name:
        return "v6 filtered"
    if "clahe" in name:
        return "v5 CLAHE" if "v5" in name else "v4 CLAHE"
    if "v5" in name or name.startswith("E"):
        return "v5 normal"
    return "legacy / verify taxonomy"


def _label(name: str) -> str:
    if name.startswith("rfdetr_nano_v5"):
        if "20261003" in name:
            return "RF-DETR Nano · v5 · 50 epochs"
        if "20261002" in name:
            return "RF-DETR Nano · v5 · 40 epochs"
        return "RF-DETR Nano · earlier partial run"
    if name == "yolo26s_v5_cctv":
        return "YOLO26s · v5 baseline"
    if name == "yolo26s_v5_parallel_simam_cbam_se":
        return "YOLO26s · parallel SimAM + CBAM + SE"
    if name.startswith("yolo26s_v5_clahe_"):
        attention = name.rsplit("_", 1)[-1]
        return "YOLO26s · " + {"simam": "SimAM", "cbam": "CBAM", "se": "SE"}.get(attention, attention) + " · CLAHE"
    if name.startswith("E5_directional"):
        return "E5 · YOLO26s directional block"
    if name.startswith("E6_"):
        return "E6 · YOLO26s P2 + SimAM (incomplete)"
    if name.startswith("yolo26s_v6_"):
        return "YOLO26s · v6 error-filtered diagnostic · " + name[-13:]
    if name.startswith("yolo26n_v4_"):
        return "YOLO26n · v4 CCTV + CLAHE · run " + (name.rsplit("-", 1)[-1] if "-" in name else "1")
    if name.startswith("yolo26n_final_merged"):
        return "YOLO26n · older merged dataset · " + name.rsplit("-", 1)[-1]
    if name.startswith("yolo26n_cbam"):
        return "YOLO26n · legacy CBAM · " + name.split("cbam_", 1)[-1].replace("_", " ")
    if name.startswith("yolo26n_simam"):
        return "YOLO26n · legacy SimAM · " + name.split("simam_", 1)[-1].replace("_", " ")
    if name.startswith("yolo26n_v1"):
        return "YOLO26n · legacy baseline · " + name.rsplit("-", 1)[-1]
    return name.replace("_", " ").replace("yolo26", "YOLO26")


def _id(prefix: str, name: str) -> str:
    return prefix + "__" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _test_evidence() -> dict[str, dict[str, float]]:
    found = {}
    comparisons = sorted(PIPE_RUNS.glob("v5_test_comparison_*.csv"))
    if comparisons:
        with comparisons[-1].open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "test":
                    continue
                run_name = {
                    "baseline": "yolo26s_v5_cctv", "simam": "yolo26s_v5_clahe_simam",
                    "cbam": "yolo26s_v5_clahe_cbam", "se": "yolo26s_v5_clahe_se",
                }.get(row["model"])
                if run_name:
                    found[run_name] = {key: float(row[column]) for key, column in (
                        ("precision", "precision"), ("recall", "recall"),
                        ("map50", "mAP50"), ("map5095", "mAP50_95"),
                    )}
    for run_name, evidence in (
        ("yolo26s_v5_parallel_simam_cbam_se", PIPE_RUNS / "yolo26s_parallel_full_v5_test-2" / "evaluation_summary.json"),
        ("yolo26s_v6_normal_error_filtered_diagnostic_seed0_20260929_184244_3e235d15", PIPE_RUNS / "yolo26s_v6_normal_error_filtered_diagnostic_seed0_20260929_184244_3e235d15_full_v5_test" / "evaluation_summary.json"),
    ):
        values = _read_json(evidence)
        if values.get("split") in {"test", "full unchanged v5 test"}:
            found[run_name] = {key: float(values[column]) for key, column in (
                ("precision", "precision"), ("recall", "recall"),
                ("map50", "mAP50"), ("map5095", "mAP50_95"),
            )}
    return found


def discover_models() -> dict[str, ModelSpec]:
    parameters = _read_json(PARAMETERS_FILE)
    test_evidence = _test_evidence()
    found: dict[str, ModelSpec] = {}

    for root, prefix in ((PIPE_RUNS, "pipe"), (LEGACY_RUNS, "legacy"), (EXPERIMENT_RUNS, "experiment")):
        if not root.is_dir():
            continue
        for run in sorted(root.iterdir()):
            if not run.is_dir():
                continue
            name = run.name
            if prefix == "pipe" and name.startswith("rfdetr_nano_"):
                checkpoint = run / "checkpoint_best_total.pth"
                if not checkpoint.is_file():
                    checkpoint = run / "checkpoint_best_ema.pth"
                if not checkpoint.is_file():
                    continue
                identity = _id(prefix, name)
                metrics = _rf_metrics(run / "best_validation_metrics.json")
                config = _read_json(run / "training_config.json")
                size = int(config.get("model_config", {}).get("resolution", 512))
                status = "complete" if metrics["map50"] is not None else "partial / no final validation"
                found[identity] = ModelSpec(
                    identity, _label(name), "RF-DETR Nano", checkpoint, run, "v5 normal", status, size,
                    metrics, parameters.get(identity),
                    "RF-DETR validation uses its own evaluator; test result is not available." if status == "complete" else "Earlier checkpoint; final validation unavailable.",
                )
                continue

            checkpoint = run / "weights" / "best.pt"
            if not checkpoint.is_file():
                continue
            # Skip validation/error-analysis directories even if a checkpoint is copied there.
            if name.endswith(("_test", "_val", "_error_analysis")):
                continue
            identity = _id(prefix, name)
            metrics = _best_yolo_csv(run / "results.csv")
            args = _read_json(run / "v6_experiment.json")
            run_args = _run_args(run)
            size = int(args.get("settings", {}).get("imgsz", run_args.get("imgsz", 640)))
            planned_epochs = int(run_args.get("epochs", args.get("settings", {}).get("epochs", 0)) or 0)
            logged_epochs = int(metrics.get("logged_epochs") or 0)
            status = "complete"
            if logged_epochs == 0:
                status = "checkpoint / no metrics"
            elif name.startswith("E6_") or (planned_epochs and logged_epochs < planned_epochs):
                status = f"stopped {logged_epochs}/{planned_epochs}" if planned_epochs else "incomplete"
            label = _label(name)
            trained_model = str(run_args.get("model", "")).lower()
            if name.startswith("yolo26n_v4_") and "yolo26s" in trained_model:
                label = label.replace("YOLO26n", "YOLO26s", 1)
            notes = ""
            if "v6" in name:
                notes = "Model-selected filtering; not a fair independent benchmark."
            elif prefix == "legacy":
                notes = "Historical run; verify class taxonomy before comparing with v5."
            elif name.startswith("E6_"):
                notes = "Training stopped with CUDA OOM; checkpoint is experimental."
            elif "clahe" in name:
                notes = "Lab-L CLAHE (clip 2.0, grid 8×8) is applied to raw input automatically."
            found[identity] = ModelSpec(
                identity, label, "YOLO26", checkpoint, run,
                _dataset_from_args(run, name), status, size, metrics,
                parameters.get(identity), notes, test_evidence.get(name, {}),
            )

    # E3 was trained on cloud, but its local checkpoint was not retained.
    if not any(item.id.startswith("experiment__e3_") for item in found.values()):
        found["cloud__e3"] = ModelSpec(
            "cloud__e3", "E3 · YOLO26s P2 + weighted fusion", "YOLO26", None, None,
            "v5 normal", "checkpoint not local", 640,
            {"precision": 0.7981, "recall": 0.7667, "map50": 0.8219, "map5095": 0.6501, "epoch": 80},
            None, "Measured in the archived report; cannot run inference until its checkpoint is copied locally.",
        )
    return found
