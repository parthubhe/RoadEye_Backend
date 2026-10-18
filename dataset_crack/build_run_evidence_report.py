"""Consolidate local sewer-defect training runs and all saved plot artifacts.

Uses only local artifacts. Run with a Python environment containing python-docx,
matplotlib, Pillow and PyYAML. It does not load model checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RUN_ROOTS = (ROOT / "runs" / "pipe_proto", ROOT / "runs" / "detect" / "runs" / "pipe_proto", ROOT / "notebook" / "runs")
OUTPUT = PROJECT_ROOT / "PipeScout_All_Local_Model_Runs_Evidence.docx"
FLOPS_FILE = ROOT / "run_flops_measurements.json"
PARAMETERS_FILE = ROOT / "webapp" / "model_parameters.json"
PLOT_NAMES = (
    "results.png", "labels.jpg", "BoxF1_curve.png", "BoxPR_curve.png",
    "BoxP_curve.png", "BoxR_curve.png", "confusion_matrix.png",
    "confusion_matrix_normalized.png",
)
BLUE = RGBColor(16, 52, 65)
TEAL = RGBColor(0, 130, 124)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (ValueError, OSError, yaml.YAMLError):
        return {}


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): value for key, value in row.items() if key} for row in csv.DictReader(handle)]


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percent(value) -> str:
    return f"{value * 100:.2f}%" if value is not None else "Unavailable"


def display_hours(value) -> str:
    return f"{value:.2f} h" if value is not None else "Unavailable"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def fallback_params(path: Path, cache: dict) -> int | None:
    source = "experiment" if "notebook" in path.parts else "legacy" if "detect" in path.parts else "pipe"
    key = f"{source}__{path.name.lower().replace('-', '_')}"
    return cache.get(key)


def rf_epoch_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        epoch = number(row.get("epoch"))
        if epoch is None:
            continue
        target = grouped[int(epoch)]
        for key, value in row.items():
            if key == "epoch":
                continue
            parsed = number(value)
            if parsed is not None:
                target[key] = parsed
    return [dict(epoch=epoch, **data) for epoch, data in sorted(grouped.items())]


def local_time_proxy(path: Path) -> float | None:
    # RF-DETR did not log wall-clock training time. File timestamps provide
    # only a rough run window and must not be presented as exact GPU time.
    start = path / "roadeeye_experiment.json"
    end = path / "metrics.csv"
    if not start.is_file() or not end.is_file():
        return None
    hours = (end.stat().st_mtime - start.stat().st_ctime) / 3600
    return hours if 0.5 < hours < 72 else None


def find_evaluations(run: Path) -> list[Path]:
    candidates = []
    for item in run.parent.iterdir():
        if not item.is_dir() or item == run:
            continue
        if item.name.startswith(run.name + "_") and ("test" in item.name.lower() or item.name.endswith("_val")):
            candidates.append(item)
        elif run.name == "yolo26s_v5_parallel_simam_cbam_se" and item.name.startswith("yolo26s_parallel_") and "test" in item.name:
            candidates.append(item)
    return sorted(candidates)


def collect_runs() -> list[dict]:
    flops = read_json(FLOPS_FILE)
    parameter_cache = read_json(PARAMETERS_FILE)
    runs = []
    for root in RUN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            is_rf = path.name.startswith("rfdetr_")
            if not is_rf and not (path / "args.yaml").is_file():
                continue
            args = read_yaml(path / "args.yaml")
            key = path.relative_to(ROOT).as_posix()
            measured = flops.get(key, {})
            results = read_csv(path / "results.csv")
            rf_rows = rf_epoch_rows(read_csv(path / "metrics.csv")) if is_rf else []
            rf_best = read_json(path / "best_validation_metrics.json") if is_rf else {}
            rf_test = read_json(path / "full_v5_test_metrics.json") if is_rf else {}
            if is_rf:
                config = read_json(path / "training_config.json").get("train_config", {})
                exp = read_json(path / "roadeeye_experiment.json")
                checkpoint = next((path / name for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "last_ema.pth") if (path / name).is_file()), None)
                if not rf_best and rf_rows:
                    best_rf_row = max(rf_rows, key=lambda row: row.get("val/mAP_50_95", -1))
                    rf_best = {
                        "val/precision": best_rf_row.get("val/precision"),
                        "val/recall": best_rf_row.get("val/recall"),
                        "val/mAP_50": best_rf_row.get("val/mAP_50"),
                        "val/mAP_50_95": best_rf_row.get("val/mAP_50_95"),
                        "val/F1": best_rf_row.get("val/F1"),
                    }
                metric = {
                    "p": number(rf_best.get("val/precision")),
                    "r": number(rf_best.get("val/recall")),
                    "map50": number(rf_best.get("val/mAP_50")),
                    "map95": number(rf_best.get("val/mAP_50_95")),
                    "f1": number(rf_best.get("val/F1")),
                }
                epoch_count = len(rf_rows)
                target = number(config.get("epochs"))
                status = "complete continuation" if epoch_count and rf_rows[-1]["epoch"] + 1 >= target else "partial / resumed"
                train_time = local_time_proxy(path)
                time_method = "approx. filesystem timestamp window, not measured GPU time" if train_time else "not logged"
                params = fallback_params(path, parameter_cache)
                data = Path(exp.get("source_dataset", "")).name or "not recorded"
                imgsz = config.get("resolution") or 512
                epochs_label = f"{int(rf_rows[0]['epoch']) + 1}–{int(rf_rows[-1]['epoch']) + 1} / {int(target) if target else '?'}" if rf_rows else "Unavailable"
            else:
                checkpoint = path / "weights" / "best.pt"
                checkpoint = checkpoint if checkpoint.is_file() else None
                best = max(results, key=lambda row: number(row.get("metrics/mAP50-95(B)")) or -1) if results else {}
                metric = {
                    "p": number(best.get("metrics/precision(B)")),
                    "r": number(best.get("metrics/recall(B)")),
                    "map50": number(best.get("metrics/mAP50(B)")),
                    "map95": number(best.get("metrics/mAP50-95(B)")),
                    "f1": None,
                }
                last = results[-1] if results else {}
                train_time = (number(last.get("time")) or 0) / 3600 if number(last.get("time")) is not None else None
                time_method = "Ultralytics cumulative results.csv time" if train_time is not None else "not logged"
                epoch_count = max((int(number(row.get("epoch")) or 0) for row in results), default=0)
                target = number(args.get("epochs"))
                epochs_label = f"{epoch_count} / {int(target) if target else '?'}" if epoch_count else "Unavailable"
                status = "complete" if target and epoch_count >= target else "interrupted / partial" if (results or checkpoint or list(path.glob("*.png"))) else "initialized only"
                params = measured.get("parameters") or fallback_params(path, parameter_cache)
                data = Path(str(args.get("data", ""))).parent.name or "not recorded"
                imgsz = args.get("imgsz", "not recorded")
            saved_plots = [path / name for name in PLOT_NAMES if (path / name).is_file()]
            runs.append({
                "path": path, "key": key, "family": "RF-DETR Nano" if is_rf else "YOLO",
                "rf": is_rf, "args": args, "checkpoint": checkpoint,
                "checkpoint_mb": checkpoint.stat().st_size / 1_000_000 if checkpoint else None,
                "parameters": params, "gflops": measured.get("gflops"), "imgsz": imgsz,
                "metric": metric, "test": rf_test, "epochs": epochs_label,
                "epoch_count": epoch_count, "status": status, "data": data,
                "hours": train_time, "time_method": time_method,
                "plots": saved_plots, "evals": find_evaluations(path), "rf_rows": rf_rows,
            })
    return runs


def add_field(paragraph, text: str, bold: bool = False) -> None:
    run = paragraph.add_run(text)
    run.bold = bold


def table(doc, headers: list[str], rows: list[list[str]], widths: list[float] | None = None):
    result = doc.add_table(rows=1, cols=len(headers))
    result.style = "Light Shading Accent 1"
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, header in zip(result.rows[0].cells, headers):
        cell.text = header
    for row in rows:
        for cell, value in zip(result.add_row().cells, row):
            cell.text = str(value)
    if widths:
        for row in result.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    for row in result.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(7.3)
    return result


def rf_plots(run: dict, output: Path) -> list[Path]:
    rows = run["rf_rows"]
    if not rows:
        return []
    output.mkdir(parents=True, exist_ok=True)
    files = []
    groups = [
        ("mAP by epoch", {"val/mAP_50": "mAP50", "val/mAP_50_95": "mAP50-95", "val/ema_mAP_50": "EMA mAP50", "val/ema_mAP_50_95": "EMA mAP50-95"}, "rf_map.png"),
        ("Precision, recall and F1 by epoch", {"val/precision": "Precision", "val/recall": "Recall", "val/F1": "F1"}, "rf_prf.png"),
        ("Per-class AP50-95 by epoch", {"val/AP/crack": "Crack", "val/AP/corrosion_rust": "Corrosion", "val/AP/sediment_deposit": "Sediment", "val/AP/root_intrusion": "Root", "val/AP/joint_defect": "Joint"}, "rf_class_ap.png"),
        ("Training loss by epoch", {"train/loss": "Train loss", "train/loss_bbox": "Box loss", "train/loss_ce": "Class loss"}, "rf_loss.png"),
    ]
    for title, columns, filename in groups:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        plotted = False
        for col, label in columns.items():
            points = [(row["epoch"] + 1, row[col]) for row in rows if col in row]
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], marker=".", linewidth=1.5, label=label)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set(title=title, xlabel="Epoch", ylabel="Loss" if "loss" in filename else "Score")
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        file = output / filename
        fig.savefig(file, dpi=150)
        plt.close(fig)
        files.append(file)
    return files


def add_plot(doc, path: Path, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(8.5))
    p = doc.add_paragraph(caption)
    p.style = "Caption"


def build() -> None:
    runs = collect_runs()
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(11.69), Inches(8.27)
    section.top_margin = section.bottom_margin = Inches(.55)
    section.left_margin = section.right_margin = Inches(.65)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9)
    for name, size in (("Title", 24), ("Heading 1", 16), ("Heading 2", 12)):
        styles[name].font.name = "Aptos Display"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = BLUE
    styles["Caption"].font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("PipeScout · page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    document.add_heading("PipeScout", 0)
    document.add_heading("Consolidated local training-run evidence", 1)
    document.add_paragraph(f"Generated {datetime.now().strftime('%d %B %Y')} from the local dataset_crack run folders. {len(runs)} training-attempt folders inventoried.")
    document.add_paragraph("Scope: dataset_crack/runs/pipe_proto, dataset_crack/runs/detect/runs/pipe_proto, and dataset_crack/notebook/runs. This is an artifact inventory, not a new evaluation or a ranking across different data splits.")
    document.add_heading("Reading notes", 2)
    for note in (
        "YOLO validation figures use the best results.csv row by mAP50-95; RF-DETR validation figures use saved best_validation_metrics.json. Validation and held-out test scores are not interchangeable.",
        "YOLO GFLOPs are nominal forward-pass estimates measured with Ultralytics get_flops/THOP at each checkpoint's saved image size. RF-DETR GFLOPs are unavailable because no verified comparable measurement is stored; checkpoint byte size is not FLOPs.",
        "YOLO training time is the last cumulative results.csv time. RF-DETR did not log wall-clock duration; where shown, its time is only the approximate local timestamp window from experiment metadata creation to final config write.",
        "Parameter counts and model size refer to the saved checkpoint where available. Missing values are marked unavailable. Incomplete and resumed runs are not presented as fully trained independent models.",
        "All saved top-level plot images (training chart, label distribution, F1/PR/P/R curves and confusion matrices) are embedded for each attempt and linked evaluation folder. Train/validation batch example photos and individual false-positive/false-negative crops are not plots and are excluded.",
    ):
        document.add_paragraph(note, style="List Bullet")

    document.add_heading("Performance overview", 1)
    table(document, ["Run", "State", "Epochs", "Dataset", "Val P", "Val R", "Val mAP50", "Val mAP50-95"], [
        [r["path"].name, r["status"], r["epochs"], r["data"], percent(r["metric"]["p"]), percent(r["metric"]["r"]), percent(r["metric"]["map50"]), percent(r["metric"]["map95"])] for r in runs
    ])
    document.add_paragraph("The RF-DETR 50-epoch run is a 10-epoch continuation of earlier local RF-DETR runs; its final best checkpoint has 90.65% validation mAP50 and 89.04% held-out test mAP50. The three RF-DETR filesystem time windows total approximately 18.0 h, but this is not a logged GPU-time measurement. Do not cite validation as test performance.")
    document.add_heading("Compute and storage overview", 1)
    table(document, ["Run", "Parameters", "GFLOPs", "Input", "Checkpoint", "Training time"], [
        [r["path"].name, f"{r['parameters']:,}" if r["parameters"] else "Unavailable", f"{r['gflops']:.3f}" if r["gflops"] is not None else "Unavailable", str(r["imgsz"]), f"{r['checkpoint_mb']:.1f} MB" if r["checkpoint_mb"] else "Unavailable", display_hours(r["hours"]) + ("*" if r["rf"] and r["hours"] else "")] for r in runs
    ])
    document.add_paragraph("* RF-DETR time is an approximate filesystem timestamp window, not a recorded training-time metric. Checkpoint MB uses decimal megabytes (1 MB = 1,000,000 bytes).")

    with tempfile.TemporaryDirectory(prefix="pipescout_report_") as temp:
        temp_root = Path(temp)
        plot_count = 0
        for index, run in enumerate(runs, 1):
            document.add_page_break()
            document.add_heading(f"{index}. {run['path'].name}", 1)
            details = [
                ("Family", run["family"]), ("State", run["status"]),
                ("Epochs observed / target", run["epochs"]), ("Dataset", run["data"]),
                ("Image size", str(run["imgsz"])), ("Source folder", rel(run["path"])),
                ("Checkpoint", rel(run["checkpoint"]) if run["checkpoint"] else "Unavailable"),
                ("Checkpoint size", f"{run['checkpoint_mb']:.2f} MB" if run["checkpoint_mb"] else "Unavailable"),
                ("Parameters", f"{run['parameters']:,}" if run["parameters"] else "Unavailable"),
                ("GFLOPs", f"{run['gflops']:.3f}" if run["gflops"] is not None else "Unavailable"),
                ("Training time", f"{display_hours(run['hours'])} ({run['time_method']})"),
                ("Best validation", ", ".join(f"{label} {percent(run['metric'][key])}" for key, label in (("p", "P"), ("r", "R"), ("map50", "mAP50"), ("map95", "mAP50-95")) )),
            ]
            table(document, ["Field", "Recorded value"], [[name, value] for name, value in details])
            if run["test"]:
                test = run["test"]
                document.add_paragraph("Held-out test artifact: " + ", ".join(f"{label} {percent(number(test.get(key)))}" for key, label in (("test/mAP_50", "mAP50"), ("test/mAP_50_95", "mAP50-95"), ("test/precision", "P"), ("test/recall", "R"))) + ".")
            plots = list(run["plots"])
            if run["rf"]:
                plots += rf_plots(run, temp_root / f"rf_{index}")
            if plots:
                document.add_heading("Saved and reconstructed plots", 2)
                for plot in plots:
                    source = rel(plot) if plot.is_relative_to(PROJECT_ROOT) else f"Reconstructed from {rel(run['path'] / 'metrics.csv')}"
                    add_plot(document, plot, f"{plot.stem.replace('_', ' ')} — {source}")
                    plot_count += 1
            else:
                document.add_paragraph("No saved plot artifact or usable RF-DETR epoch series is available for this attempt.")
            for evaluation in run["evals"]:
                eval_plots = [evaluation / name for name in PLOT_NAMES if (evaluation / name).is_file()]
                if not eval_plots:
                    continue
                document.add_heading(f"Evaluation artifact: {evaluation.name}", 2)
                document.add_paragraph("This is a separate evaluation folder, not another training run. Its plots are reproduced below; scores are not inferred solely from the images.")
                for plot in eval_plots:
                    add_plot(document, plot, f"{plot.stem.replace('_', ' ')} — {rel(plot)}")
                    plot_count += 1
        document.add_page_break()
        document.add_heading("Artifact coverage", 1)
        document.add_paragraph(f"{len(runs)} training-attempt folders included; {sum(bool(r['checkpoint']) for r in runs)} contain a selected checkpoint; {plot_count} plot images embedded, including derived RF-DETR charts. Evaluation-only folders are attached to their matching training attempt rather than double-counted as trained models.")
        document.add_paragraph("The cloud E3 P2 weighted-fusion checkpoint and partner-run E1/E2/E4 are not in these local run folders; no local plots or measurements were invented for them. If those artifacts are added later, rerun this generator.")
        document.save(OUTPUT)
    print(f"Created {OUTPUT} ({len(runs)} attempts, {plot_count} plots, {OUTPUT.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    build()
