# RoadEye v5 trained models

This folder is the model index for the [RoadEye v5 sewer-defect dataset](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5). The checkpoints and supporting files below have been published under that dataset repository's `models/` directory. To run them in the GitHub project's FastAPI server, see the [server setup guide](https://github.com/parthubhe/RoadEye_Backend#sewer-defect-detection-server).

All checkpoints detect the same five classes, in this order: `crack`, `corrosion_rust`, `sediment_deposit`, `root_intrusion`, `joint_defect`. Normal v5 has 9,486 training, 2,014 validation, and 1,986 held-out test images. CLAHE checkpoints use the corresponding CLAHE-processed v5 image split, not the normal pixels.

| Model | Checkpoint link | Input | Validation mAP50 / mAP50-95 | Test mAP50 / mAP50-95 | Why publish it |
|---|---|---|---:|---:|---|
| RF-DETR Nano, 50 epochs | [checkpoint_best_total.pth](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/rfdetr_nano_50e/checkpoint_best_total.pth) | Normal v5, 512 px | 90.65% / 76.51% | **89.04% / 73.85%** | Best held-out test result among these releases. |
| YOLO26s baseline, 80 epochs | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/yolo26s_v5_baseline/best.pt) | Normal v5, 640 px | 82.56% / 65.58% | 82.14% / 65.05% | Main YOLO reference. |
| YOLO26s E3, P2 + weighted fusion, 80 epochs | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/p2_weighted_fusion/best.pt) | Normal v5, 640 px | 82.18% / 65.02% | 80.78% / 63.83% | Near the baseline on validation, but lower on test; useful architecture ablation. |
| YOLO26s + SimAM + CLAHE, 80 epochs | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/yolo26s_v5_clahe_simam/best.pt) | CLAHE v5, 640 px | 82.44% / 64.47% | 80.85% / 63.03% | Validation AP50 gains on crack, roots, and joints. |
| YOLO26s + CBAM + CLAHE, 80 epochs | [best.pt](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/yolo26s_v5_clahe_cbam/best.pt) | CLAHE v5, 640 px | 82.60% / 63.23% | 81.55% / 62.50% | Highest validation root-intrusion AP50 among the YOLO variants audited. |

The class-specialist claims use **validation AP50**, not test-set selection:

| Class AP50 | Baseline | P2 + fusion | SimAM + CLAHE | CBAM + CLAHE |
|---|---:|---:|---:|---:|
| Crack | 70.07% | 69.21% | **70.92%** | 70.65% |
| Root intrusion | 93.09% | 95.15% | 94.49% | **95.52%** |
| Joint defect | 79.03% | 79.64% | **80.09%** | 78.47% |

These are single-run differences; the higher class AP50 values do **not** mean higher overall test mAP or more reliable real-world performance. The normal and CLAHE validation sets share image identities but differ in pixels. YOLO and RF-DETR also use different resolutions and evaluators, so cross-family scores need protocol qualification. The 90.65% RF-DETR figure is **validation** mAP50; its test mAP50 is 89.04%.

## Loading and provenance

- RF-DETR uses `rfdetr==1.11.0`. Use its `checkpoint_best_total.pth`, not `last.ckpt`, for inference. Its folder should also contain the validation/test JSON and dataset provenance.
- YOLO checkpoints were trained with `ultralytics==8.4.120`. Baseline needs the installed Ultralytics package. SimAM and CBAM checkpoints also need [`models/support/attention_modules.py`](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/support/attention_modules.py) and their architecture YAMLs registered before loading. E3's folder includes the custom P2/weighted-fusion support source and YAML.
- [`classwise_validation_summary.json`](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/classwise_validation_summary.json) records the AP50 values used for the specialist comparison. E3's folder includes its separate [test summary](https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5/blob/main/models/p2_weighted_fusion/test_summary.json).
- `.pt` checkpoints contain Python-serialized objects; load only files from sources you trust. Verify the dataset/source-image and pretrained-weight licensing before making this repository public or selecting a license for the weights.

## Upload from the RoadEye project root

Authenticate once with `hf auth login`, then preview the exact files and destinations:

```powershell
.\dataset_crack\models\publish_hf_models.ps1
```

When the preview is correct, publish the selected checkpoints, metadata, support code, this README, and the validation evidence to the **existing dataset repository**:

```powershell
.\dataset_crack\models\publish_hf_models.ps1 -Publish
```

The script uses `hf upload ... --repo-type dataset` for explicit paths only. It never uploads whole training runs, `last.ckpt`, temporary images, or the local validation-audit output directory. The dataset repository remains a dataset repository; for model-page discoverability, publish separate model repositories and link them back here.

If you prefer individual CLI commands for the five model weights, these are the exact destinations. The release script above additionally uploads this README, validation/test evidence, YAMLs, and custom code, so it is the recommended complete upload:

```powershell
$repo = 'AkumaDachi/roadeeye-sewer-defects-v5'
hf upload $repo 'dataset_crack/runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7/checkpoint_best_total.pth' 'models/rfdetr_nano_50e/checkpoint_best_total.pth' --repo-type dataset
hf upload $repo 'dataset_crack/runs/pipe_proto/yolo26s_v5_cctv/weights/best.pt' 'models/yolo26s_v5_baseline/best.pt' --repo-type dataset
hf upload $repo 'dataset_crack/models/p2_weighted_fusion' 'models/p2_weighted_fusion' --repo-type dataset
hf upload $repo 'dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_simam/weights/best.pt' 'models/yolo26s_v5_clahe_simam/best.pt' --repo-type dataset
hf upload $repo 'dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_cbam/weights/best.pt' 'models/yolo26s_v5_clahe_cbam/best.pt' --repo-type dataset
```
