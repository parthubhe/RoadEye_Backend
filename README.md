# RoadEye - Unified Pothole Detection System

## Overview
RoadEye is an integrated framework for pothole detection and road condition analysis using computer vision and deep learning. The system unifies multiple datasets (Roboflow, Kaggle, Mendeley) and provides end‑to‑end training, inference, and evaluation pipelines for various architectures (YOLOv8, RF‑DETR, etc.).

## Datasets
- **Roboflow Pothole Dataset** – `RoadEyeUnified_RFDETR/` and `RoadEyeUnified_SingleClass_*`
- **Mendeley Pothole Dataset** – `MultiweatherPotholeDataset/`
- **Kaggle Pothole Dataset** – `potholes0_SampleAnnotation_Kaggle.xml`
- **Custom Synthetic Datasets** – `NightPotholesDataset/`, `RainyPotholesDataset/`

Each dataset includes images, YOLO‑style labels, and task‑specific metadata.

## Models
- YOLOv8 (nano, small, medium, large) – `yolo11n.pt`, `yolo11s.pt`
- RF‑DETR (medium) – `rf-detr-medium.pth`
- Unified models for single‑class and multi‑class detection

## Training Scripts
- `train_roadeye.py` – YOLOv8 training
- `train_rfdetr_roadeye.py` – RF‑DETR training
- `validate_unified_dataset.py` – Cross‑dataset validation
- `predict_roadeye.py` – Inference on images / video

## Evaluation
- mAP, Precision‑Recall curves, confusion matrices
- Visualization scripts: `visualize_augmented_samples.py`, `merge_all_datasets.py`

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/RoadEye.git
cd RoadEye

# Install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train a model (example)
python train_roadeye.py --data RoadEyeUnified/roadeye.yaml --epochs 50

# Run inference
python predict_roadeye.py --weights runs/detect/train/weights/best.pt --source assets/sample.jpg
```

## Directory Structure
```
├── datasets/            # Raw and processed datasets
├── runs/                # Training runs and predictions
├── scripts/             # Training / inference utilities
├── assets/              # Sample images, annotations
└── README.md            # This file
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
[MIT License](LICENSE)

---

*Generated with ❤️ using Python, PyTorch, and Ultralytics YOLO.*