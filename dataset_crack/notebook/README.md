# RoadEye E1-E6 experiment notebooks

## RF-DETR baseline on the published v5 dataset

[RFDETR_Nano_v5.ipynb](RFDETR_Nano_v5.ipynb) is a standalone Colab/Kaggle notebook
for RF-DETR Nano (with an optional Small setting). It downloads the pinned Hugging
Face v5 revision, verifies all five classes and the full 9,486/2,014/1,986
train/validation/test image counts, then trains on the normal, non-CLAHE data.
Enable GPU and Internet access, open the notebook, and run its cells in order.
The full test split is evaluated only when `RUN_TEST` is explicitly enabled.
Each new run receives a unique output folder; set `RESUME_FROM` to a saved
`last.ckpt` to continue one. Preserve the checkpoint before the cloud session
expires. Colab/Kaggle GPU time is not guaranteed to cover all 40 epochs.

RF-DETR's YOLO loader needs image paths inside the dataset root. The notebook
therefore hard-links image files into a separate working directory and copies
labels there. It records and omits only invalid box rows (including four
known zero-area rows) in that working copy; the Hugging Face snapshot stays
unchanged. This baseline should not be compared to YOLO metrics without
matching evaluation definitions and the held-out split.

### RF-DETR Nano on the local RTX 4070 Laptop GPU

[`train_rfdetr_local.py`](train_rfdetr_local.py) uses the existing
`dataset_crack/final_dataset_v5`, with no Hugging Face download. From the
RoadEye root in PowerShell:

```powershell
venv\Scripts\python.exe -m pip install "rfdetr[train]==1.11.0"
venv\Scripts\python.exe dataset_crack\notebook\train_rfdetr_local.py --check
venv\Scripts\python.exe dataset_crack\notebook\train_rfdetr_local.py
```

Defaults for the local 8 GiB GPU: Nano, 512 px fixed scale, batch 1,
gradient accumulation 16, mixed precision, gradient checkpointing, 40 maximum
epochs, and zero data-loader workers for Windows reliability. It warns when
less than 5 GiB system RAM is available. These are safe starting settings,
not a guarantee of the highest accuracy or that every driver/installation
will fit. The default 512 px is a **different experiment** from the 384 px
Kaggle run; for a closer match use `--resolution 384`.

The script writes a new uniquely named run under `dataset_crack/runs/pipe_proto`
and keeps the original v5 source unchanged. Invalid zero-area label rows are
omitted only in its reusable working copy; provenance is saved with each run.
To resume an interrupted run from a full checkpoint, or deliberately score
the held-out test set after choosing a model on validation:

```powershell
venv\Scripts\python.exe dataset_crack\notebook\train_rfdetr_local.py --resume "path\to\last.ckpt" --resolution 512
venv\Scripts\python.exe dataset_crack\notebook\train_rfdetr_local.py --eval-test
```

The second command trains a new model **and then** scores the test set; do not
use it for repeated model selection. To continue the 384 px Kaggle run, obtain
its `last.ckpt` and pass `--resume ... --resolution 384` with otherwise matching
training settings. A `.pth` best-weights file does not restore optimizer state.

Six training scripts and six standalone notebooks for **normal v5, five classes**.
E0 is intentionally omitted. These are experiments, not established accuracy improvements.

Dataset: `AkumaDachi/roadeeye-sewer-defects-v5`.
Verified Hub revision: `d88c0ed4f0fd39253eb3866bf2feb9dcf808bae5`.
Counts at that revision: **9,486 train / 2,014 validation / 1,986 test**.
Classes: crack, corrosion_rust, sediment_deposit, root_intrusion, joint_defect.
No CLAHE or error-filtered subset is used; the uploaded split membership is preserved.
This does not independently verify pipe identities or prove absence of near-duplicate leakage.

## Experiments

| Batch | ID | Script | Notebook | Architecture |
|---|---|---|---|---|
| A | E1 | train_e1.py | E1_p2.ipynb | YOLO26s + additional P2 head |
| A | E2 | train_e2.py | E2_weighted_fusion.ipynb | YOLO26s + weighted residual fusion |
| B | E3 | train_e3.py | E3_p2_weighted_fusion.ipynb | P2 + weighted residual fusion |
| B | E4 | train_e4.py | E4_yolo26m.ipynb | Standard YOLO26m capacity baseline |
| C | E5 | train_e5.py | E5_directional.ipynb | YOLO26s + directional convolution at P3 |
| C | E6 | train_e6.py | E6_single_simam.ipynb | Chosen E1-E5 parent + one SimAM block |

E6 defaults to E3 **provisionally**; pass `--simam-base E1` etc. after selecting a parent on validation.
E6 starts from the matching COCO pretrained model, not from the parent's fine-tuned checkpoint,
so it has the same training budget. SPPF and C2PSA are inherited YOLO26 components.

Local architecture measurements (unfused, five classes, 640-pixel FLOP estimate):

| Experiment | Parameters | Estimated GFLOPs |
|---|---:|---:|
| E1 | 10,097,544 | 30.35 |
| E2 | 11,107,074 | 28.08 |
| E3 | 11,319,447 | 39.14 |
| E4 | 21,780,598 | 74.99 |
| E5 | 9,986,551 | 23.21 |
| E6, E3 parent | 11,319,447 | 39.14 plus uncounted SimAM operations |

SimAM adds no parameters, but does consume compute. The profiler does not account for all
functional/elementwise operations; these estimates are not measured GPU latency or training time.

### Exact design choices

- E1 appends a top-down P2 branch from P3 and the stride-4 backbone feature. Detect order
  is P3/P4/P5/P2 (strides 8/16/32/4) to preserve the original head widths and weight mapping.
  It is an extra high-resolution branch, not a full four-level neck redesign.
- E2 projects the two inputs of each PAN merge to a common width, learns softmax-normalized
  scalar weights, sums the projected features, and projects back with a residual connection.
  This is **weighted PAN fusion**, not a reproduction of EfficientDet's complete BiFPN.
- E3 uses the E2 fusion at all four existing merge points and its extra P2 merge.
- E5 uses depthwise 1x7 and 7x1 convolutions, pointwise fusion, and a residual path at P3.
  It is not dynamic snake convolution or a claimed new published method.
- E6 places one parameter-free SimAM at P2 for E1/E3 parents, or at P3 for E2/E4/E5.
- All original-layer weights are transferred by an explicit source-to-target index map;
  only shape-compatible tensors are copied. New modules and incompatible class outputs
  initialize fresh. `pretrained_transfer.json` confirms that the full backbone transferred.
- A custom trainer repeats this transfer when Ultralytics reconstructs the training model.
  `args.yaml` says `pretrained: false` because the default loader is bypassed; mapped loading
  still occurs unless you explicitly pass `--scratch`.

## Run locally (PowerShell from RoadEye root)

```powershell
venv\Scripts\python.exe -m pip install -r dataset_crack\notebook\requirements.txt
venv\Scripts\python.exe dataset_crack\notebook\train_e1.py
```

All six scripts and the batch runner use the existing `dataset_crack/final_dataset_v5`
automatically, regardless of your terminal's working directory. This is normal v5, all five
classes, without CLAHE. No Hub login or dataset download is needed locally. If the dataset
is missing, the script stops instead of silently downloading. Pretrained model weights may
still download if they are not cached (or supply `--pretrained path/to/yolo26s.pt`).
To select a different local location or explicitly download the pinned Hub snapshot:

```powershell
venv\Scripts\python.exe dataset_crack\notebook\train_e3.py --local-data dataset_crack\final_dataset_v5
venv\Scripts\python.exe dataset_crack\notebook\train_e3.py --download-data
venv\Scripts\python.exe dataset_crack\notebook\train_e6.py --simam-base E1
```

Run batches sequentially on one GPU:

```powershell
venv\Scripts\python.exe dataset_crack\notebook\run_batch.py --group A
venv\Scripts\python.exe dataset_crack\notebook\run_batch.py --group B
venv\Scripts\python.exe dataset_crack\notebook\run_batch.py --group C -- --simam-base E3
```

Extra arguments follow `--`, for example `--group A -- --batch 8 --seed 1`.
There is no batch E: E5 is an experiment, run with `python train_e5.py` from this folder;
batch C runs E5 and E6. Cloud notebooks explicitly select Hub downloads by default;
set their `LOCAL_DATA` variable to use an already-downloaded dataset instead.
Use independent Colab/Kaggle sessions to run different experiments concurrently; multiple
training processes sharing one GPU can increase memory pressure and total runtime.

## Colab and Kaggle

1. Upload/import the desired `.ipynb`. It embeds every required source file; no repo clone,
   manual YAML upload, or local Drive path is needed.
2. Enable a GPU. On Kaggle also enable Internet for package and dataset downloads.
3. For the private Hub repository, add **HF_TOKEN** in Colab Secrets or Kaggle Secrets and
   allow notebook access. Use a token that can read this dataset. Never put a literal token
   in notebook code. A hidden `getpass` fallback is included as commented code.
4. Run installation and configuration cells. Keep the supplied dataset commit identical
   across experiments. The bootstrap refuses a different installed Ultralytics version;
   restart the session if you had imported another version before installation.
5. Dataset setup verifies names, counts, and label syntax and writes a local portable YAML.
   It replaces stale Windows paths without modifying the Hub snapshot.
   Existing zero-size boxes or tolerance-bound coordinates are reported in dataset_provenance.json
   and preserved; this suite does not silently clean or relabel the published v5 dataset.
6. Run the training cell. E6's `SIMAM_BASE` is configurable. Use one GPU per session.
7. Export the ZIP before the session ends. In Colab set `MOUNT_DRIVE=True` to write runs
   directly to Drive. Kaggle outputs remain in `/kaggle/working`; save the notebook version
   with its output files. Dataset downloads remain outside the run archive.

The cloud GPU could be faster or slower than your laptop. No training time is guaranteed.
P2 and YOLO26m may require `BATCH=8` or `4`. Keep the same batch across comparable runs
where practical and record reductions; the default nominal batch size is 64 for optimizer
accumulation, but different microbatches can still change BatchNorm/optimization behavior.

## Training and evaluation

Defaults: 80 epochs, 640 pixels, batch 16, AdamW, lr0=0.002, lrf=0.01, cosine schedule,
seed 0, deterministic=True, patience 100, AMP on, close_mosaic=10, two workers.
HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, horizontal flip 0.5, mosaic 1.0;
vertical flip, rotations, shear, perspective, mixup and copy-paste are zero.
Other defaults are tied to Ultralytics 8.4.120. GPU PyTorch is supplied by the platform.

Every fresh run has an experiment/dataset/seed/time/UUID name. Existing runs are preserved.
Saved outputs include best.pt, last.pt, periodic checkpoints (every 10 epochs), results.csv,
plots/confusion matrices, per-class summary.json, dependency/hardware metadata, dataset
revision and filename/label hashes, weight-transfer report, and custom module source.

Training selects checkpoints using **validation**, followed by a validation evaluation.
The optional notebook test cell is off by default. Final full-test evaluation from a script:

```powershell
venv\Scripts\python.exe dataset_crack\notebook\train_e1.py --evaluate-only "path\to\best.pt" --eval-split test
```

Use validation to select configurations; then evaluate the chosen candidates on the full
unchanged test set. The earlier model-selected filtered subset is not used here.
For finalists, repeat seeds 0/1/2 and compare mean and variability, crack AP/recall, latency
and parameter cost as well as overall mAP. Select by mAP50-95 and inspect false positives.

Collect saved evaluation summaries, including extracted cloud result folders:

```powershell
venv\Scripts\python.exe dataset_crack\notebook\collect_results.py --project dataset_crack\notebook\runs
```

### Resume an interrupted run

Set notebook `RESUME` or pass `--resume path/to/weights/last.pt` with the matching experiment
and E6 base. Resume intentionally continues that run directory. Use an interrupted checkpoint
with optimizer state; a completed/stripped best.pt cannot resume optimizer training.
Restore the run/support files and regenerate the dataset YAML on a new cloud session first.

Custom checkpoints require this folder (or the exported `support` folder) on `sys.path`,
then `from modules import register_modules; register_modules()` before `YOLO(best_pt)`.

## Verification and rebuilding

```powershell
venv\Scripts\python.exe dataset_crack\notebook\train_e1.py --dry-run
venv\Scripts\python.exe dataset_crack\notebook\verify_experiments.py --training-smoke
venv\Scripts\python.exe dataset_crack\notebook\generate_notebooks.py
```

The verifier checks all six architectures plus all E6 parent choices: finite forward/backward
passes, new-module gradients, correct strides, pretrained backbone mapping (when matching
weights exist), checkpoint reload, notebook-cell Python syntax, and optional tiny synthetic
end-to-end training. Synthetic results are never research metrics. See verification_report.json.
Full v5 training and cloud execution have not been performed by this setup step.
Regenerate notebooks after editing shared Python modules so embedded sources stay in sync.

References: [Ultralytics training](https://docs.ultralytics.com/modes/train/),
[Hugging Face snapshot download](https://huggingface.co/docs/huggingface_hub/en/guides/download),
[EfficientDet weighted fusion](https://arxiv.org/abs/1911.09070).
Ultralytics-derived configuration/code use is subject to its AGPL-3.0/commercial licensing.
