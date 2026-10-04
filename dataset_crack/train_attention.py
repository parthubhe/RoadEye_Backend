"""Train a controlled YOLO26 attention-module ablation on v5.

One entry point is used for all variants so the data split, optimizer,
augmentation, seed, and evaluation settings stay identical.  Attention is
inserted after the P3/8 feature block, which is the small-object branch most
relevant to thin cracks.

Examples
--------
    python train_attention.py --attention simam --model s --dataset v5
    python train_attention.py --attention cbam --model s --dataset v5_clahe
    python train_attention.py --attention se --model m --data final_dataset_v5/data.yaml

The convenience wrappers ``train_simam.py``, ``train_cbam.py``, and
``train_se.py`` select the corresponding attention by default.
"""

from __future__ import annotations

import argparse
import copy
import re
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

try:  # Works both as a script and as ``python -m dataset_crack.train_attention``.
    from attention_modules import CBAM, SimAM, SqueezeExcitation
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    from dataset_crack.attention_modules import CBAM, SimAM, SqueezeExcitation


ROOT = Path(__file__).resolve().parent
BASE_MODEL_YAML = ROOT.parent / "venv" / "Lib" / "site-packages" / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
DATASETS = {
    "v5": ROOT / "final_dataset_v5" / "data.yaml",
    "v5_clahe": ROOT / "final_dataset_v5_clahe" / "data.yaml",
}
ATTENTIONS = ("none", "simam", "cbam", "se")
MODEL_SCALES = ("n", "s", "m")


def register_attention_modules() -> None:
    """Expose custom classes to Ultralytics' YAML parser."""

    tasks.SimAM = SimAM
    tasks.CBAM = CBAM
    tasks.SqueezeExcitation = SqueezeExcitation


def parse_args(default_attention: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", choices=ATTENTIONS, default=default_attention or "none")
    parser.add_argument("--model", choices=MODEL_SCALES, default="s", help="YOLO26 scale (default: s)")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="v5")
    parser.add_argument("--data", type=Path, default=None, help="Optional custom data.yaml; overrides --dataset")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--name", default=None, help="Explicit run name; an available suffix is chosen on collision")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "pipe_proto")
    parser.add_argument("--no-pretrained", action="store_true", help="Train the generated architecture from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Build/load the model and exit without training")
    return parser.parse_args()


def p3_channels(scale: str) -> int:
    """Return the width-scaled output channels of YOLO26 layer 16."""

    return {"n": 64, "s": 128, "m": 256}[scale]


def shifted_from(value: Any, insertion_index: int) -> Any:
    """Shift layer references after inserting one layer."""

    if isinstance(value, int):
        return value + 1 if value > insertion_index else value
    if isinstance(value, list):
        return [shifted_from(item, insertion_index) for item in value]
    return value


def attention_layer(attention: str, channels: int) -> list[Any]:
    if attention == "simam":
        return [-1, 1, "SimAM", [channels]]
    if attention == "cbam":
        return [-1, 1, "CBAM", [channels, 16, 7]]
    if attention == "se":
        return [-1, 1, "SqueezeExcitation", [channels, 16]]
    raise ValueError(f"Unsupported attention: {attention}")


def build_model_yaml(scale: str, attention: str) -> Path | None:
    """Create a scale-aware five-class YOLO26 YAML for an attention run."""

    if attention == "none":
        return None
    if not BASE_MODEL_YAML.is_file():
        raise FileNotFoundError(f"Ultralytics base model YAML not found: {BASE_MODEL_YAML}")

    config = yaml.safe_load(BASE_MODEL_YAML.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Unexpected model YAML structure: {BASE_MODEL_YAML}")
    config["nc"] = 5
    config["scale"] = scale

    layers = copy.deepcopy(config["backbone"] + config["head"])
    insertion_index = 16  # P3/8 C3k2 output in the official YOLO26 graph.
    rewritten: list[list[Any]] = []
    for index, layer in enumerate(layers):
        if index == insertion_index + 1:
            rewritten.append(attention_layer(attention, p3_channels(scale)))
        current = copy.deepcopy(layer)
        if index > insertion_index:
            current[0] = shifted_from(current[0], insertion_index)
        if current[2] == "Detect":
            current[0] = [17 if ref == 16 else ref for ref in current[0]]
        rewritten.append(current)

    config["backbone"] = rewritten[: len(config["backbone"])]
    config["head"] = rewritten[len(config["backbone"]) :]
    output_dir = ROOT / "generated_attention_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"yolo26{scale}_{attention}_v5.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if not output_path.exists():
        output_path.write_text(rendered, encoding="utf-8")
    elif output_path.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(
            f"Generated model config differs from existing file: {output_path}. "
            "Move it aside before regenerating."
        )
    return output_path


def dataset_path(args: argparse.Namespace) -> Path:
    path = (args.data if args.data is not None else DATASETS[args.dataset]).resolve()
    if not path.is_file():
        hint = "Run build_v5_clahe.py first" if args.data is None and args.dataset == "v5_clahe" else "Check --data"
        raise FileNotFoundError(f"Missing dataset YAML: {path}. {hint}.")
    return path


def dataset_token(path: Path, requested: str) -> str:
    text = str(path.parent if path.name == "data.yaml" else path).lower()
    if "final_dataset_v5_clahe" in text:
        return "v5_clahe"
    if "final_dataset_v5" in text:
        return "v5"
    token = re.sub(r"[^a-z0-9]+", "_", requested.lower()).strip("_")
    return token or "custom"


def available_run_name(project: Path, requested: str) -> str:
    """Return a non-colliding name without ever deleting or overwriting runs."""

    candidate = requested
    suffix = 2
    while (project / candidate).exists():
        candidate = f"{requested}_{suffix}"
        suffix += 1
    return candidate


def main(default_attention: str | None = None) -> None:
    args = parse_args(default_attention)
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs, imgsz, and batch must be positive; workers cannot be negative")

    register_attention_modules()
    data_path = dataset_path(args)
    model_yaml = build_model_yaml(args.model, args.attention)
    if args.attention == "none":
        model = YOLO(f"yolo26{args.model}.pt")
    else:
        model = YOLO(str(model_yaml))
        if not args.no_pretrained:
            model.load(f"yolo26{args.model}.pt")

    data_name = dataset_token(data_path, args.dataset)
    attention_name = "baseline" if args.attention == "none" else args.attention
    requested_name = args.name or f"yolo26{args.model}_{data_name}_{attention_name}"
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    run_name = available_run_name(project, requested_name)
    print(f"data: {data_path}")
    print(f"model: yolo26{args.model} ({'pretrained' if not args.no_pretrained else 'from scratch'})")
    print(f"attention: {args.attention}; run: {project / run_name}")

    if args.dry_run:
        print("Dry run complete; no training started.")
        return

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        cls_pw=0.0,
        patience=100,
        close_mosaic=10,
        device=args.device,
        workers=args.workers,
        seed=0,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
