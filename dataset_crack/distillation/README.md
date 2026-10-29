# PipeScout: RF-DETR teacher → compact YOLO26s student

These scripts implement **hard-target response distillation**: the 50-epoch
RF-DETR Nano teacher adds only high-confidence, non-overlapping boxes to a
*copy-on-write training overlay*. All original ground-truth rows remain. The
student starts from the existing v5 YOLO26s checkpoint, then exports to
statically calibrated INT8 ONNX. This is not soft-logit/feature distillation,
and an accuracy improvement is not guaranteed.

The original `final_dataset_v5` is never modified. Teacher inference and
quantization calibration use **train images only**. Validation and test remain
the unchanged v5 splits; do not tune thresholds against the test set.

Run from the RoadEye repository root in PowerShell:

```powershell
# Optional: check paths and image count; does not start inference.
venv\Scripts\python.exe -m dataset_crack.distillation.prepare_teacher_dataset --dry-run

# Stage 1: generate train-only teacher targets (~9,486 train images).
venv\Scripts\python.exe -m dataset_crack.distillation.prepare_teacher_dataset

# If interrupted, use the same settings with --resume; do not delete the overlay.
venv\Scripts\python.exe -m dataset_crack.distillation.prepare_teacher_dataset --resume

# Inspect the number of additions before training. A small number means
# this hard-target method has little extra supervision to contribute.
Get-Content dataset_crack\distillation_outputs\v5_rfdetr50_to_yolo26s_targets\summary.json

# Stage 2: fine-tune the saved YOLO26s baseline for 40 epochs in a new run.
venv\Scripts\python.exe -m dataset_crack.distillation.train_student `
  --name pipescout_kd_yolo26s_v5_rfdetr50
```

Stage 2 prints the new run path. For ONNX export, install the missing
dependencies in the **same** training venv first:

```powershell
venv\Scripts\python.exe -m pip install "onnx>=1.16,<2" onnxruntime

# Stage 3: INT8 ONNX export, ~5% of v5 train for calibration, then validation.
venv\Scripts\python.exe -m dataset_crack.distillation.export_student_int8 `
  --checkpoint "dataset_crack\runs\pipe_proto\pipescout_kd_yolo26s_v5_rfdetr50\weights\best.pt" `
  --name pipescout_kd_yolo26s_v5_rfdetr50_int8
```

The exporter saves an `export_report.json` with actual decimal MB, a
quantization-operator check, and validation P/R/mAP scores. It fails loudly if
the artifact is not under 30 MB. Its `--skip-validation` option skips the
validation pass, but size and graph checks still run. The original `.pt`
checkpoint and v5 dataset are retained.

For a fair comparison, validate the original teacher and baseline student on
the same unchanged v5 test split after the final student/export choice. INT8
ONNX may not speed up every Raspberry Pi CPU workload; benchmark FPS and memory
on the actual Pi before claiming a deployment improvement. Training and
calibration here are separate from that deployment benchmark.

The default overlay and export paths are deliberately separate from previous
runs. Each stage refuses to overwrite an existing completed output. Use
`--output` on Stage 1 or `--name` on Stages 2–3 for a new experiment. Stage 1
can resume only if the existing overlay's teacher hash and thresholds match.

Ultralytics' ONNX INT8 export uses static ONNX Runtime quantization with
calibration data; this script explicitly sets `split="train"` so held-out
validation and test frames are not used for calibration. See the
[Ultralytics export reference](https://docs.ultralytics.com/modes/export/).
