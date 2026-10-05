# RoadEye - Unified Pothole Detection System

## Overview

RoadEye is an integrated framework for pothole detection and road condition analysis using computer vision and deep learning. The system unifies multiple datasets (Roboflow, Kaggle, Mendeley) and provides end-to-end training, inference, and evaluation pipelines for various architectures such as YOLOv8 and RF-DETR.

This repository contains the training and augmentation scripts for the models used in the main project:
https://github.com/astralranger/road-eye

---

## Sewer-defect detection server

The `dataset_crack/webapp` FastAPI server is a separate, local pipe-inspection
demo. It detects five classes: `crack`, `corrosion_rust`, `sediment_deposit`,
`root_intrusion`, and `joint_defect`. The browser UI supports image and video
uploads, a browser webcam, a USB camera attached to the server, and optional
RTSP/HTTP streams. The [v5 dataset and released models](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/tree/main/models)
are on Hugging Face in a **dataset repository**, so downloads must use
`--repo-type dataset`.

| Released checkpoint | Hugging Face file | Local path recognized by the server |
|---|---|---|
| RF-DETR Nano, 50 epochs | [checkpoint_best_total.pth](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/rfdetr_nano_50e/checkpoint_best_total.pth) | `dataset_crack/runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7/checkpoint_best_total.pth` |
| YOLO26s v5 baseline | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/yolo26s_v5_baseline/best.pt) | `dataset_crack/runs/pipe_proto/yolo26s_v5_cctv/weights/best.pt` |
| YOLO26s + SimAM + CLAHE | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/yolo26s_v5_clahe_simam/best.pt) | `dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_simam/weights/best.pt` |
| YOLO26s + CBAM + CLAHE | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/yolo26s_v5_clahe_cbam/best.pt) | `dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_cbam/weights/best.pt` |
| YOLO26s E3, P2 + weighted fusion | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/p2_weighted_fusion/best.pt) | `dataset_crack/notebook/runs/E3_p2_weighted_fusion_v5_normal/weights/best.pt` |

The server scans these **run folders**, not `dataset_crack/models/`. Do not
place downloaded weights in the latter folder and expect them to appear in the
selector. The E3 checkpoint was verified to load with this project's custom
modules. Its source/configuration files are also in the [E3 release folder](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/tree/main/models/p2_weighted_fusion).
The SimAM and CBAM support modules are already included in this GitHub checkout.

### Set up on Windows

Run these commands from the repository root. If you already have the training
`venv`, keep it and install only the missing packages. For a fresh environment,
create it first and install the PyTorch/torchvision build matching your CPU or
NVIDIA setup from the [official PyTorch installer](https://pytorch.org/get-started/locally/).
The tested project environment uses `ultralytics==8.4.120` and
`rfdetr==1.11.0`:

```powershell
py -3.13 -m venv venv                 # skip if venv already exists
venv\Scripts\python.exe -m pip install --upgrade pip
# Install compatible torch and torchvision using the PyTorch installer first.
venv\Scripts\python.exe -m pip install ultralytics==8.4.120 rfdetr==1.11.0 "huggingface_hub>=1.0,<3.0"
venv\Scripts\python.exe -m pip install -r dataset_crack\webapp\requirements.txt
```

Do **not** install the repository-root `requirements.txt` for this server: it
contains machine-specific package URLs from a different environment.

Download the released `models/` folder, then copy each checkpoint and the
RF-DETR evaluation metadata to the paths the server scans. The Hugging Face
download is staged under your Windows temporary directory; weights are not
committed to Git. If the repository is private, authenticate with `hf auth
login` first. If Xet downloads stall on your network, set
`$env:HF_HUB_DISABLE_XET = "1"` before downloading.

```powershell
$repo = 'AkumaDachi/roadeeye-sewer-defects-v5'
$release = Join-Path $env:TEMP 'roadeye_v5_release'
venv\Scripts\python.exe -c "from huggingface_hub.cli.hf import main; main()" download $repo --repo-type dataset --include 'models/**' --local-dir $release
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }

$rf = 'dataset_crack/runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7'
$copies = @(
    @('models/rfdetr_nano_50e/checkpoint_best_total.pth', "$rf/checkpoint_best_total.pth"),
    @('models/rfdetr_nano_50e/best_validation_metrics.json', "$rf/best_validation_metrics.json"),
    @('models/rfdetr_nano_50e/full_v5_test_metrics.json', "$rf/full_v5_test_metrics.json"),
    @('models/rfdetr_nano_50e/full_v5_test_evaluation.json', "$rf/full_v5_test_evaluation.json"),
    @('models/yolo26s_v5_baseline/best.pt', 'dataset_crack/runs/pipe_proto/yolo26s_v5_cctv/weights/best.pt'),
    @('models/yolo26s_v5_baseline/args.yaml', 'dataset_crack/runs/pipe_proto/yolo26s_v5_cctv/args.yaml'),
    @('models/yolo26s_v5_clahe_simam/best.pt', 'dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_simam/weights/best.pt'),
    @('models/yolo26s_v5_clahe_simam/args.yaml', 'dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_simam/args.yaml'),
    @('models/yolo26s_v5_clahe_cbam/best.pt', 'dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_cbam/weights/best.pt'),
    @('models/yolo26s_v5_clahe_cbam/args.yaml', 'dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_cbam/args.yaml'),
    @('models/p2_weighted_fusion/best.pt', 'dataset_crack/notebook/runs/E3_p2_weighted_fusion_v5_normal/weights/best.pt')
)
foreach ($pair in $copies) {
    $destination = $pair[1]
    $source = Join-Path $release $pair[0]
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing downloaded file: $source" }
    if (Test-Path -LiteralPath $destination) { Write-Host "Keeping existing file: $destination"; continue }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
```

Start the server and open <http://127.0.0.1:8000>. Its model selector and
comparison table are available in the page; API docs are at
<http://127.0.0.1:8000/docs>.

```powershell
venv\Scripts\python.exe -m uvicorn dataset_crack.webapp.app:app --host 127.0.0.1 --port 8000
```

The published release does not include YOLO `results.csv` training logs, so a
fresh clone can run those checkpoints but may show blank YOLO validation
columns in the server. The [model release index](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/README.md)
records their measured scores; RF-DETR validation and test metadata are
included in the download. CLAHE checkpoints automatically enhance **raw**
input frames; do not pre-apply CLAHE. The 50-epoch RF-DETR result is 90.65%
validation mAP50 but 89.04% on the held-out v5 test set.

For USB/IP-camera settings, video-job storage, endpoint details, and safety
notes, see the [server README](dataset_crack/webapp/README.md). Keep the server
on localhost unless you add authentication and other deployment safeguards.

---

## Datasets

The project leverages multiple datasets:

* Roboflow Pothole Dataset
* Mendeley Pothole Dataset
* Kaggle Pothole Dataset

Unified and augmented datasets:

* YOLO format: https://huggingface.co/datasets/AkumaDachi/RoadEye_Yolo_Aug
* RF-DETR format: https://huggingface.co/datasets/AkumaDachi/RoadEye_RFDETR_Aug

---

## Models

Pretrained models and resources:

* Hugging Face (RF-DETR Medium model):
  https://huggingface.co/AkumaDachi/RoadEYE_RFDETR_Medium

* Inference images and videos:
  https://drive.google.com/drive/folders/1A9bsHYJtYSah6HvPjLE9KpoYdbDtrors?usp=sharing

* Large model (training + inference):
  https://drive.google.com/drive/folders/11Z4Ri7jdXBfsuCX4G6Ahdo7UW7JDeQNj?usp=drive_link

* Medium model (training + inference):
  https://drive.google.com/drive/folders/1HoBO5hgAdMVwkSq8ort8UezmWEZDyf5E?usp=sharing

* Final inference pipeline (Colab notebook – includes SegFormer + ROI):
  https://colab.research.google.com/drive/1K62OVa3uZtw51wlQMypFd_v6qdIyKvYC?usp=sharing

---

## Main Repository

* Full system (frontend, backend, and app):
  https://github.com/astralranger/road-eye

---

## Training Scripts

* `train_roadeye.py` – YOLOv8 training
* `train_rfdetr_roadeye.py` – RF-DETR training
* `validate_unified_dataset.py` – Cross-dataset validation
* `predict_roadeye.py` – Inference on images and videos

---

## Final Inference

For the complete pipeline, use:

* `RoadEyeFinal.ipynb`

This notebook includes:

* Cityscapes SegFormer model for road segmentation
* Region of Interest (ROI) extraction
* Integrated pothole detection pipeline

---
