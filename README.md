# RoadEye - Unified Pothole Detection System

## Overview

RoadEye is an integrated framework for pothole detection and road condition analysis using computer vision and deep learning. The system unifies multiple datasets (Roboflow, Kaggle, Mendeley) and provides end-to-end training, inference, and evaluation pipelines for various architectures such as YOLOv8 and RF-DETR.

This repository contains the training and augmentation scripts for the models used in the main project:
https://github.com/astralranger/road-eye

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
