"""Export a distilled YOLO student to INT8 ONNX with train-only calibration.

The export never replaces the original best.pt. It validates the ONNX graph,
checks the 30 MB target and optionally measures validation-set metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "final_dataset_v5" / "data.yaml"
DEFAULT_OUT = ROOT / "distillation_outputs" / "exports"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Distilled run's weights/best.pt")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Original v5 YAML; train split used for calibration")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--name", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fraction", type=float, default=0.05, help="Train-split calibration fraction; default ~474 frames")
    p.add_argument("--size-limit-mb", type=float, default=30.0)
    p.add_argument("--device", default="cpu", help="CPU keeps GPU free and exports a portable ONNX file")
    p.add_argument("--skip-validation", action="store_true", help="Export only; validation is recommended")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    checkpoint, data = args.checkpoint.resolve(), args.data.resolve()
    if not checkpoint.is_file() or not data.is_file():
        raise FileNotFoundError("Student checkpoint and original v5 data.yaml must exist")
    if args.fraction <= 0 or args.fraction > 1 or args.size_limit_mb <= 0 or args.imgsz < 32:
        raise ValueError("Invalid calibration fraction, size limit, or image size")
    if checkpoint.name != "best.pt":
        raise ValueError("Pass the selected student weights/best.pt, not last.pt")
    if data != DEFAULT_DATA.resolve():
        raise ValueError("For this experiment calibration must use the unchanged final_dataset_v5/data.yaml")
    name = args.name or f"pipescout_kd_int8_onnx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = args.output_root.resolve() / name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    print(f"Student checkpoint: {checkpoint}\nCalibration: {data} train split, fraction={args.fraction}\nOutput: {output}")
    if args.dry_run:
        print("Dry run only; no export started.")
        return
    try:
        import onnx
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install export dependencies first: venv\\Scripts\\python.exe -m pip install "onnx>=1.16,<2" onnxruntime') from exc

    from dataset_crack.webapp.engine import register_checkpoint_modules

    register_checkpoint_modules()
    from ultralytics import YOLO

    output.mkdir(parents=True, exist_ok=False)
    copied = output / "student.pt"
    shutil.copy2(checkpoint, copied)
    model = YOLO(str(copied))
    exported = Path(model.export(
        format="onnx", quantize=8, data=str(data), split="train",
        fraction=args.fraction, imgsz=args.imgsz, batch=8,
        device=args.device, simplify=False, nms=False,
    )).resolve()
    if not exported.is_file() or exported.parent != output:
        raise RuntimeError(f"Export did not produce an ONNX file inside {output}: {exported}")
    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph)
    quant_ops = sum(node.op_type in {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"} for node in graph.graph.node)
    if quant_ops == 0:
        raise RuntimeError("Exported graph contains no quantization operators; refusing to call it INT8")
    size_mb = exported.stat().st_size / 1_000_000
    report = {
        "source_checkpoint": str(checkpoint), "calibration_yaml": str(data),
        "calibration_split": "train", "calibration_fraction": args.fraction,
        "exported_onnx": str(exported), "size_mb_decimal": round(size_mb, 3),
        "size_limit_mb_decimal": args.size_limit_mb, "under_limit": size_mb < args.size_limit_mb,
        "quantization_operators": quant_ops, "validation": None,
    }
    print(f"Exported: {exported} ({size_mb:.2f} MB); quantization operators: {quant_ops}")
    if not args.skip_validation:
        metrics = YOLO(str(exported)).val(data=str(data), split="val", imgsz=args.imgsz, batch=1, device="cpu", plots=False, verbose=False)
        report["validation"] = {
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        }
        print(f"Validation mAP50: {report['validation']['map50']:.4f}; mAP50-95: {report['validation']['map50_95']:.4f}")
    (output / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if size_mb >= args.size_limit_mb:
        raise RuntimeError(f"Valid INT8 ONNX export is {size_mb:.2f} MB, not under {args.size_limit_mb:.2f} MB. See {output / 'export_report.json'}")
    print(f"Size target met. Report: {output / 'export_report.json'}")


if __name__ == "__main__":
    main()
