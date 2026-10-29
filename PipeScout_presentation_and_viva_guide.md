# PipeScout — presentation script and viva guide

Evidence checked: 28 October 2026. Percentages below are measured results unless explicitly described as unavailable. The RF-DETR confusion-matrix diagnostic is stored beside its checkpoint under `test_confusion_*`.

## 1. Presentation script

### Opening and objective

“Our project, PipeScout, assists visual inspection of underground pipes and sewers. It detects five defect categories: crack, corrosion or rust, sediment deposit, root intrusion, and joint defect. The system accepts still images, uploaded videos, and live camera feeds, and displays defect locations, predicted classes, confidence scores, and inference timing.

We developed both an accuracy-oriented model, RF-DETR Nano, and a smaller YOLO26s student for deployment experiments. We also investigated attention mechanisms, feature fusion, contrast enhancement, and crack segmentation. These are separate experiments; they are not all combined in the final RF-DETR model.”

### Data collection and taxonomy

“We assembled existing annotated sewer and pipe image exports rather than collecting every training image ourselves. Our merged data used the source families named sewage, sewage_defect_detection, sewer_defects, and sewer_o8use. Earlier versions also included pipe_inspection and pipeline_crack_detection.

The source annotations had different class names and IDs. We converted them into a common five-class YOLO taxonomy. Every label row contains a class ID followed by normalized bounding-box center coordinates, width, and height. Source prefixes prevent filename collisions and help trace data provenance.

The final v5 dataset contains 13,486 image files: 9,486 training, 2,014 validation, and 1,986 test. These counts describe files, not independently verified unique pipes or scenes.”

Source-family file counts in v5: sewage 2,400; sewage_defect_detection 980; sewer_defects 978; sewer_o8use 9,128. These are not four proven-independent datasets: some source exports overlap, and earlier source repetition is possible.

### Dataset development, v2 to v5

“The earlier merge mixed close-up crack photographs with internal CCTV pipe views. These have different scales, textures, and perspectives. We also encountered ambiguous class mappings, repeated images, and background-only labels.

The stricter v4 branch restricted sources, expanded the taxonomy to 11 categories, removed empty labels after mapping, and used grouped splitting. It kept only 2,628 images. This changed several factors together and substantially reduced training data, so its lower scores cannot be attributed solely to splitting or preprocessing.

For v5, we returned to the five-class v2 assignments and removed the two close-up domain families: 4,571 pipeline_crack_detection files and 348 pipe_inspection files. We retained empty labels and did not apply v4's aggressive source filtering. This produced a controlled domain-focused comparison with the earlier five-class setup.

The merge script contains filename-based grouping for recognized Roboflow duplicates and video naming patterns. However, v5 inherited v2 assignments and retained overlapping source families. We therefore do not claim that every pipe, video, or duplicate cluster is isolated across its splits.”

A remaining annotation concern: the legacy `sewage` mapping maps several anonymous numeric classes to crack. That is a documented limitation, not a verified ontology conversion. Do not describe all remapped labels as expert-validated.

### Preprocessing and augmentation

“We prepared a normal v5 dataset and a separate CLAHE version with identical image identities and bounding-box labels. CLAHE improves local contrast by equalizing small regions while clipping histogram peaks to limit noise amplification. We applied it to the L, or lightness, channel in Lab color space, using clip limit 2.0 and an 8 by 8 tile grid. Eight by eight describes the number of tiles, not tiles eight pixels wide.

The main YOLO baseline used 640-pixel training inputs, normalization, color jitter, translation, scale variation, horizontal flips, and mosaic augmentation. Mosaic was disabled for the final ten epochs. Mixup, vertical flips, shear, and perspective augmentation were disabled in the saved run.

RF-DETR used normal v5 pixels at resolution 512. Its separate working copy omitted four invalid zero-area label rows: three training and one validation row. No test label row or image was removed. Model-specific resizing and normalization happen in its own data pipeline.”

Do not claim every architecture used identical augmentations or pixel transformations. For RF-DETR, saved settings include scale jitter and disabled multi-scale scheduling; exact default transforms are package-version dependent.

### YOLO baseline and attention experiments

“We fine-tuned pretrained YOLO26s for 80 epochs using AdamW, initial learning rate 0.002, cosine learning-rate decay, batch size 16, and seed zero. Its locally counted model has 9,951,734 parameters and an estimated 22.757 GFLOPs at 640-pixel input. Logged training time was about 3.19 hours.

We evaluated three attention mechanisms. Squeeze-and-excitation learns which feature channels matter. CBAM first applies channel attention and then spatial attention. SimAM assigns feature importance using per-channel activation statistics without adding learned parameters.

We also implemented a parallel branch: the same feature map passes through SimAM, CBAM, and SE; their outputs and the original feature map are concatenated, projected through a 1 by 1 convolution, and added through a learnable residual scale. This was intended to combine complementary attention, but the full-test result did not beat the baseline.”

| Model | Validation mAP50 | Validation mAP50–95 | Test mAP50 | Test mAP50–95 |
|---|---:|---:|---:|---:|
| YOLO26s baseline | 82.54% | 65.61% | 82.14% | 65.05% |
| YOLO26s + CBAM + CLAHE | 82.66% | 63.50% | 81.55% | 62.50% |
| YOLO26s + SimAM + CLAHE | 82.45% | 64.47% | 80.85% | 63.03% |
| YOLO26s + SE + CLAHE | 82.24% | 62.82% | 81.53% | 62.96% |
| Parallel SimAM + CBAM + SE, normal v5 | 82.13% | 63.88% | 80.73% | 62.52% |
| E3: P2 + weighted fusion | 82.19% | 65.01% | 80.78% | 63.83% |

YOLO validation numbers above use the best CSV row by mAP50–95. Separate checkpoint re-evaluations can differ slightly; do not mix rows silently. Attention+CLAHE versus normal baseline changes two factors, so it does not isolate attention's causal effect.

### E1–E6 architecture ablations

| Experiment | What we changed | Expected benefit | Evidence status |
|---|---|---|---|
| E1 | Added P2, stride-4 detection output | Preserve finer details for small/thin defects | Defined; no verified completed result located locally |
| E2 | Learned weighted fusion in the YOLO neck | Adapt the contribution of feature branches | Defined; no verified completed result located locally |
| E3 | Combined P2 and weighted fusion | Fine detail plus adaptive feature fusion | Completed; tied roughly on validation, lower on test |
| E4 | YOLO26m capacity comparison | More representation capacity | Defined; no verified completed result located locally |
| E5 | Residual horizontal/vertical depthwise 7-pixel convolutions | Learn elongated crack patterns | Completed; validation mAP50 82.02%, mAP50–95 63.72% |
| E6 | Added one SimAM block to a selected experiment, locally E3 | Refine useful features without three attention branches | Interrupted after four epochs with memory limitations; not a valid completed comparison |

Our weighted fusion projects branches, applies softmax-normalized learned weights, expands the summed feature, and adds a residual. It is not a complete BiFPN implementation. E5 uses fixed directional kernels, not deformable convolution. SPPF already exists in the YOLO26 base; it is not an original contribution we invented.

### Error analysis and filtered-data branch

“We inspected false positives and false negatives to distinguish low contrast, blur, class confusion, missing labels, and poor localization. A false negative can occur even when the defect looks obvious to a human: confidence can be low, the predicted class can differ, or the box may fail the IoU requirement.

We also tried v6 error-based filtering. Scores obtained after removing evaluation examples selected from model errors describe an easier, selected subset. They cannot establish an improvement on the original benchmark. Full unchanged v5 evaluations remain the main comparison.”

### RF-DETR and final accuracy result

“We then compared YOLO against RF-DETR Nano, a transformer-based detector. Our saved configuration uses a windowed DINOv2-small encoder, 16-pixel patches, a 256-dimensional hidden representation, two decoder layers, and 300 object queries. The local parameter count is 30,333,500.

We trained on an RTX 4070 Laptop GPU with eight GiB VRAM. Batch size one and 16 accumulation steps gave a nominal effective batch size of 16. We enabled mixed precision, gradient checkpointing, and exponential moving-average weights, with zero loader workers for Windows reliability. We first trained 40 epochs, then resumed to 50 total epochs.

The final checkpoint re-evaluation achieved 90.65 percent validation mAP50 and 76.51 percent validation mAP50–95. On all 1,986 unchanged v5 test images, it achieved 89.04 percent mAP50 and 73.85 percent mAP50–95. Thus, the 90-percent result belongs to validation; the held-out split result is 89.04 percent.”

| RF-DETR Nano 50e metric | Validation | Test |
|---|---:|---:|
| mAP50 | 90.65% | 89.04% |
| mAP50–95 | 76.51% | 73.85% |
| mAP75 | 80.63% | 77.91% |
| Precision | 89.90% | 86.34% |
| Recall | 84.81% | 83.89% |
| F1 | 87.22% | 85.07% |
| mAR | 85.65% | 84.11% |

Compared with the YOLO26s baseline, test mAP50 increased by 6.90 percentage points and mAP50–95 by 8.80 points. Resolution and evaluator differ between families, so this is our observed pipeline comparison rather than a perfectly controlled architecture-only comparison. Repeated development against the same test benchmark also limits its status as a fresh final holdout.

Crack remained weaker than roots and corrosion in RF-DETR's per-class training logs. Do not substitute the per-class AP50–95 logs for per-class AP50.

### RF-DETR confusion matrix, measured 28 October 2026

The full 1,986-image v5 test split was run using RF-DETR50, 512-pixel inference, CUDA FP16, confidence 0.25, and one-to-one greedy matching at box IoU 0.50. Matching is class-agnostic so wrong-class spatial matches appear off the diagonal. Rows are predicted classes; columns are true classes. This is a new fixed-threshold diagnostic, not a reconstruction of the official evaluator's swept AP or F1 metrics.

| Predicted / True | Crack | Corrosion | Sediment | Roots | Joint | Background |
|---|---:|---:|---:|---:|---:|---:|
| Crack | 1108 | 0 | 7 | 0 | 7 | 553 |
| Corrosion | 0 | 366 | 0 | 0 | 0 | 39 |
| Sediment | 11 | 0 | 124 | 0 | 0 | 32 |
| Roots | 0 | 0 | 3 | 61 | 0 | 9 |
| Joint | 7 | 0 | 0 | 0 | 313 | 176 |
| Background | 270 | 28 | 18 | 9 | 43 | — |

“The diagonal gives correctly localized and classified objects. The background column contains unmatched predictions, and the background row contains unmatched annotations. Background is bookkeeping, not a sixth trained class. Its bottom-right cell is undefined because object detection does not enumerate all true-negative background regions.

Crack has 1,108 correct matches. It has 553 unmatched crack predictions and 270 unmatched crack annotations. Including wrong-class matches, its total false positives are 567 and total false negatives are 288, giving 66.15 percent precision and 79.37 percent recall at this threshold. Joint defect is the next substantial false-positive source: 176 unmatched predictions, or 183 false positives including class confusion.

Most errors here are unmatched detections or missed annotations rather than swaps between defect classes. Unmatched predictions may reflect localization failures, duplicate detections, or genuinely unlabeled objects as well as clean-background false alarms; visual review is required to separate these causes.”

| Class | TP | FP including class errors | FN including class errors | Precision at 0.25 | Recall at 0.25 |
|---|---:|---:|---:|---:|---:|
| Crack | 1108 | 567 | 288 | 66.15% | 79.37% |
| Corrosion | 366 | 39 | 28 | 90.37% | 92.89% |
| Sediment | 124 | 43 | 28 | 74.25% | 81.58% |
| Roots | 61 | 12 | 9 | 83.56% | 87.14% |
| Joint | 313 | 183 | 50 | 63.10% | 86.23% |

Artifacts: `dataset_crack/runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7/test_confusion_20261028_095944/`. Image: `confusion_matrix.png`; raw counts: `confusion_matrix.csv`; exact matching settings and statistics: `summary.json`. Script: `dataset_crack/notebook/rfdetr_confusion_matrix.py`.

### Distillation and INT8 deployment branch

“For a smaller deployable model, we used RF-DETR as an offline teacher. It predicted on training images only. We preserved original labels and added teacher detections with confidence at least 0.85 only if they overlapped no retained box by IoU 0.30 or more. We allowed at most three additions per image.

Only six new boxes passed the filter: four cracks, one sediment deposit, and one corrosion box. We then fine-tuned the existing YOLO26s baseline for 40 epochs, with AdamW at learning rate 0.0005, cosine decay, and no mosaic or mixup.

The best student validation result was 85.64 percent mAP50, 71.16 percent mAP50–95, 80.28 percent precision, and 80.00 percent recall. Its checkpoint is 20.32 MB; logged training time is about 97 minutes. Because only six labels changed, the gain cannot be attributed solely to distillation without a matching fine-tuning control.

We exported the trained student to INT8 ONNX. The file is 10.04 MB. Calibration uses five percent of the original training split, approximately 474 images. We have the export file, but no completed saved INT8 accuracy report; the 85.64 percent figure belongs to the PyTorch student. This is teacher-assisted hard-label fine-tuning followed by post-training quantization, not quantization-aware training.”

### Crack segmentation research branch

“We also tested whether pixel-level supervision helps cracks, whose irregular narrow shape is poorly represented by rectangular boxes. SAM3.1 served as an offline annotation assistant using original crack boxes and candidate-mask selection. We considered up to four candidates per box, selected one, and combined selected masks per image.

We generated 2,000 candidate examples and manually excluded poor cases, leaving 1,455 image-mask pairs: 1,226 training, 112 validation, and 117 test. Hybrid box training used 9,112 retained original training images; mask loss applied only where a usable mask existed.

The implemented hybrid shares a YOLO26n backbone and adds a small semantic crack mask head. It retains five-class box supervision, including auxiliary crack boxes, and adds a single binary crack mask. It is not five-class instance segmentation. SAM is absent during inference.

The best validation pseudo-mask Dice was 67.63 percent and IoU 53.60 percent at epoch 48. These measure agreement with pseudo-labels, not expert ground truth. No validation/test negative masks remained after curation, so negative-image mask false positives were not measurable. This branch did not replace RF-DETR as the demonstrated accuracy model.”

### Server and demonstration

“PipeScout uses FastAPI to expose inference to a lightweight HTML, CSS, and JavaScript interface. Images are decoded, processed by the selected local checkpoint, annotated, and returned with confidence and timing. Uploaded videos run as background jobs with progress polling. WebSockets support browser-camera frames and server-side USB or opt-in network-camera streams.

The interface provides drag-and-drop upload, test examples, model selection, a confidence slider, model metrics, parameter counts, and result downloads. A model manager keeps one checkpoint loaded and serializes inference to control VRAM. Live delivered FPS includes pipeline overhead; it should not be confused with pure GPU inference speed.

Our deployment objective is assisted inspection, with a human reviewing the detections. The contributions are the integrated inspection workflow, documented dataset experiments, architecture ablations, error analysis, and a smaller deployment branch. We did not invent YOLO, RF-DETR, CLAHE, or the individual attention mechanisms.”

## 2. Architecture and mathematics for questions

### YOLO26

- Backbone: convolutional features, C3k2 blocks, SPPF, and C2PSA. Local edges and textures become semantic features.
- Neck: FPN/PAN-style upsampling, concatenation, and downsampling combine fine spatial detail with deeper semantic context.
- Head: class scores and box regression at P3/8, P4/16, and P5/32; E1/E3 add P2/4. At 640 input, feature grids correspond to 80, 40, 20, and optionally 160 pixels per side.
- Our YAML has `end2end: true` and `reg_max: 1`; YOLO26 supports one-to-one end-to-end predictions and removes DFL. The installed loss uses CIoU and an L1 regression term for this configuration. Do not recite older YOLO DFL/objectness formulas as the exact implementation here.
- SPPF sequentially applies max pooling to enlarge contextual receptive fields efficiently. For stride-one 5x5 pools, stacked receptive-field extents correspond to 5, 9, and 13. It is contextual pooling, not dataset resizing.

### RF-DETR

- ViT patch embedding converts image patches into feature vectors; positional information preserves location. Windowed attention controls backbone computation.
- Object queries are learned detection slots, not preassigned class labels or a requirement to find 300 defects.
- Decoder self-attention lets queries interact. Cross-attention gathers relevant image features; deformable attention samples selected locations rather than densely attending to every location.
- Classification and box heads predict a set of detections. Hungarian matching chooses a one-to-one prediction/ground-truth assignment during training; this matching algorithm is not rerun as object assignment at inference.
- Training combines classification loss with L1 box loss and generalized IoU loss, including auxiliary supervision. The saved config enables IoU-aware BCE classification and grouped DETR training.
- DINOv2 supplies pretrained visual representations. Our superior score is observed evidence; better context and pretraining are plausible explanations, not experimentally isolated causes.

### Equations worth knowing

1. **YOLO coordinates:** `x_norm = x_center / W`, `y_norm = y_center / H`, `w_norm = box_width / W`, `h_norm = box_height / H`. This makes labels independent of stored image resolution.
2. **IoU:** `intersection_area / union_area`. Correct detection requires localization and class agreement under the evaluation protocol.
3. **Precision:** `TP/(TP+FP)`. **Recall:** `TP/(TP+FN)`. **F1:** `2PR/(P+R)` for one common aggregation and threshold. Independently averaged class metrics need not satisfy this relation when combined afterward.
4. **AP:** area under a class precision-recall curve. **mAP:** mean class AP. `mAP50` uses box IoU 0.50. `mAP50–95` averages over IoU 0.50, 0.55, ..., 0.95 and over classes. It is not classification accuracy.
5. **GIoU:** `IoU - (area(C)-area(union))/area(C)`, where C is the smallest enclosing rectangle. RF-DETR uses a loss based on `1-GIoU`.
6. **CIoU:** combines overlap, normalized center distance, and an aspect-ratio penalty. Our installed YOLO box loss uses CIoU; DIoU omits the aspect-ratio term.
7. **Transformer attention:** `softmax(QK^T/sqrt(d_k)) V`. Queries ask which features matter; keys determine similarity; values provide information. RF-DETR's deformable operator uses sparse learned sampling, so this dense equation explains the principle rather than its complete implementation.
8. **Weighted fusion:** `a_i = exp(w_i)/sum_j exp(w_j)`, `F_out = F_in + g * Expand(sum_i a_i * Project_i(F_i))`. The residual helps retain useful pretrained features.
9. **SE:** global average pooling produces a channel summary; a small bottleneck network and sigmoid create channel gates. **CBAM:** channel gates followed by a spatial gate from channel-average/max maps. **SimAM:** gates depend on squared activation deviation normalized by within-channel variance; it has no learned gate parameters.
10. **Mask Dice:** `2|P intersect G|/(|P|+|G|)`. **Mask IoU:** `|P intersect G|/|P union G|`. The code adds smoothing constants and averages per image; a transformed mean Dice is not exactly the mean IoU.
11. **Hybrid objective:** `L = L_detection + lambda * (L_weighted_BCE + L_Dice)`. Pixel BCE uses positive weight 8; mask loss is disabled where masks are absent. Missing masks are unknown labels, not automatic blank masks.
12. **INT8 affine quantization:** `q = clip(round(x/s)+z, qmin, qmax)`, reconstructed value `x_hat = s(q-z)`. Calibration estimates appropriate ranges. Some operators can remain floating point; ONNX alone does not imply INT8.
13. **Gradient accumulation:** nominal effective batch = microbatch x accumulation steps x device count. For local RF-DETR, 1 x 16 x 1 = 16. This does not make all behavior identical to a physical batch of 16.
14. **EMA weights:** `theta_ema = beta * theta_ema + (1-beta) * theta`. Smoothing model weights can stabilize validation performance.
15. **Parameters versus FLOPs:** parameters count stored learned values; FLOPs estimate arithmetic for a given input size. File size depends on precision and checkpoint contents. Runtime RAM, latency, and FPS also depend on execution backend, hardware, preprocessing, and buffering.

### Short viva answers

**Why is crack difficult?** Thin irregular shapes, low contrast, joints/shadows with similar appearance, broad or inconsistent boxes, and legacy source-label ambiguity. Class imbalance alone is not a sufficient explanation.

**Why did more attention not win?** Added gates can duplicate existing feature selection, suppress useful weak features, and increase optimization complexity. The test results determine whether it helped.

**Why not delete every error image?** An error is not proof of bad annotation. Deleting evaluation errors makes the benchmark easier; valid hard training examples teach robustness.

**Why not compare directly with a 92.3% paper?** Different data, class definitions, splits, IoU criteria, evaluator settings, and leakage controls can change mAP. A defensible head-to-head needs a shared benchmark and protocol.

**Does confidence 0.9 mean 90% accurate?** It is the model score for that prediction. Without calibration evidence, it is not a guaranteed correctness probability.

**Is the student a compressed RF-DETR network?** It is a separate YOLO network learning from teacher labels. RF-DETR layers are not simply shrunk or copied.

**Did we use QAT?** No. The present pipeline uses teacher-assisted fine-tuning and subsequent post-training INT8 export.

**Does ONNX ensure Raspberry Pi real time?** No. It provides a deployable graph; actual supported operators, CPU kernels, input size, and measured latency determine feasibility.

**Is the hybrid our best model?** No verified box-mAP improvement is stored for it; its reported results are pseudo-mask agreement. RF-DETR remains the strongest verified detection pipeline among the compared releases.

## 3. File locations and use

Workspace root: `D:\Study Material\Sem VI\Projects\PBL2\RoadEye`. Run commands below from this root.

| Purpose | Location relative to root | How to use it |
|---|---|---|
| Full normal v5 | `dataset_crack/final_dataset_v5/` | `train`, `valid`, `test` contain `images` and `labels` |
| Dataset entry point | `dataset_crack/final_dataset_v5/data.yaml` | Pass this YAML to YOLO training/evaluation |
| Data provenance | `dataset_crack/final_dataset_v5/build_report.yaml`, `manifest.csv` | Inspect splits, domain exclusions, source paths |
| Raw/staged exports | `dataset_crack/merge_staging/`, `sewer.v2-sw2-2.2k.yolo26/` | Original source metadata and images |
| Legacy mapping | `dataset_crack/dataset_classes.yaml`, `remap_all_datasets.py`, `remap_sewer_o8use.py` | Inspect class conversion before rebuilding |
| Legacy merge | `dataset_crack/final_merge.py` | Filename-based grouping, split creation, optional train repetition |
| CLAHE creation | `dataset_crack/build_v5_clahe.py` | Creates separate processed pixels |
| YOLO baseline | `dataset_crack/train_v5.py` | Fine-tunes YOLO26s or m |
| Attention implementations | `dataset_crack/attention_modules.py` | SimAM, CBAM, SE, parallel branch |
| Attention training | `dataset_crack/train_attention.py`, `train_parallel.py` | Choose compatible normal/CLAHE data |
| E1–E6 definitions | `dataset_crack/notebook/architectures.py`, `modules.py`, `configs/` | Architecture configurations and custom layers |
| E1–E6 runners | `dataset_crack/notebook/train_e1.py` through `train_e6.py`, matching notebooks | Local or cloud experiments |
| RF-DETR local training | `dataset_crack/notebook/train_rfdetr_local.py` | Local v5 data and run configuration |
| RF-DETR cloud notebook | `dataset_crack/notebook/RFDETR_Nano_v5.ipynb` | Colab/Kaggle dataset download and training |
| RF-DETR working data | `dataset_crack/rfdetr_local_workspace/final_dataset_v5_rfdetr_yolo/` | Loader-compatible overlay and validation report |
| Standard training runs | `dataset_crack/runs/pipe_proto/` | Checkpoints, CSV curves, evaluation JSON |
| Local E5/E6 runs | `dataset_crack/notebook/runs/` | Experiment results outside the standard runs directory |
| Final RF-DETR50 | `dataset_crack/runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7/` | Inference: `checkpoint_best_total.pth`; resume: `last.ckpt` |
| Final RF-DETR metrics | Same RF-DETR50 folder: `best_validation_metrics.json`, `full_v5_test_metrics.json`, `full_v5_test_evaluation.json` | Scores and evaluation provenance |
| RF-DETR confusion matrix | Same folder: newest `test_confusion_*/` | `confusion_matrix.png`, `.csv`, `summary.json` |
| YOLO26s baseline | `dataset_crack/runs/pipe_proto/yolo26s_v5_cctv/` | `weights/best.pt` for inference; `results.csv` for training history |
| E3 release/support | `dataset_crack/models/p2_weighted_fusion/` | Recovered model metadata and custom support code |
| Teacher label generation | `dataset_crack/distillation/prepare_teacher_dataset.py` | Offline RF-DETR predictions and conservative label additions |
| Teacher labels | `dataset_crack/distillation_outputs/v5_rfdetr50_to_yolo26s_targets/` | `data.yaml`, `summary.json`, generated train labels |
| Student trainer | `dataset_crack/distillation/train_student.py` | Fine-tune baseline YOLO26s on teacher overlay |
| Student checkpoint | `dataset_crack/runs/pipe_proto/pipescout_kd_yolo26s_v5_rfdetr50/weights/best.pt` | PyTorch student inference |
| INT8 export script | `dataset_crack/distillation/export_student_int8.py` | Train-only calibration and ONNX export |
| INT8 artifact | `dataset_crack/distillation_outputs/exports/pipescout_kd_yolo26s_v5_rfdetr50_int8/student_int8.onnx` | Requires compatible ONNX Runtime inference and preprocessing |
| Hybrid model/trainer | `dataset_crack/hybrid_crack/model.py`, `train.py` | Custom shared detector and crack mask head |
| Curated pseudo masks | `dataset_crack/hybrid_crack_masks/v5_2k_sam_pseudo_sewage10/` | Masks, copied images, review overlays, manifest |
| Hybrid checkpoints | `dataset_crack/runs/pipe_proto/yolo26n_v5_hybrid_sam3_curated_20261017/` | `best_hybrid.pt` for custom inference; `last_training.pt` for resuming |
| FastAPI entry | `dataset_crack/webapp/app.py` | Upload, jobs, WebSocket camera endpoints |
| Inference manager | `dataset_crack/webapp/engine.py` | Loading, preprocessing, inference, annotation |
| Model table | `dataset_crack/webapp/registry.py`, `model_parameters.json` | Availability and saved-score discovery |
| Frontend | `dataset_crack/webapp/static/index.html`, `app.js`, `styles.css` | Browser page, interactions, styling |
| Sample images/results | `dataset_crack/training_examples_final_25/`, `training_examples_rfdetr50_20261027_084336/` | Training-image visual examples and prediction contact sheet |

```powershell
# Launch the local interface
venv\Scripts\python.exe -m uvicorn dataset_crack.webapp.app:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001`; API documentation is `http://127.0.0.1:8001/docs`.

YOLO checkpoints can be loaded with Ultralytics; custom-attention/E3 checkpoints also need their Python modules registered. RF-DETR checkpoints require RF-DETR, not `YOLO(...)`. Hybrid checkpoints require `hybrid_crack.model.load_hybrid`. `.onnx` requires an ONNX-capable runner and matching preprocessing/postprocessing.

Hugging Face dataset/model repository: https://huggingface.co/datasets/AkumaDachi/roadeeye-sewer-defects-v5

Published model index: `dataset_crack/models/README.md`. Downloaded weights must be placed at the paths expected by `webapp/registry.py`, or the registry must be updated to their new locations; placing arbitrary files in a folder does not automatically configure every runtime.

## References for architecture explanations

- Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/
- RF-DETR official implementation: https://github.com/roboflow/rf-detr
- Our actual settings are the saved local YAML/JSON configurations. Current upstream defaults may differ from the package versions used for training.
