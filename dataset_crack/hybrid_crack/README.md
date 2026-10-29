# YOLO26n five-box + crack-mask experiment

This is **one shared YOLO26n model**, not a YOLO + SAM inference pipeline. It
keeps the five-class YOLO box detector (including crack as auxiliary box
supervision) and adds a small binary **semantic** crack-mask head. At inference
it returns the other four classes as boxes and all crack pixels as a mask.
Separate crack instances are not represented by this mask head. SAM3.1 is only
an offline annotation assistant.

The code targets the installed Ultralytics 8.4.120 YOLO26n layer layout. Run
the smoke test again after changing Ultralytics. It uses P1/2, P2/4 and P3/8
features, 2,512,183 parameters in the tested build, and a single forward pass.
An untrained fp16 deployment checkpoint serialized to about 5.05 MiB; the
actual trained checkpoint and runtime memory should still be measured.
The reviewed-mask loss is **disabled for every image lacking an approved mask**.
All original v5 training images still contribute the five-class box loss.

### Materialize the 2,000-image SAM pseudo-mask training subset

The original review queue above is unchanged. This separate workflow creates
the actual copied images, YOLO labels, and binary mask PNGs in a new directory.
The `sewage_*` **filename family** supplies 10% of crack-positive samples in
each original split (140 train, 16 valid, 16 test); the other three filename
families supply the remaining positives. The 280 images without crack boxes
are sampled from the other families. Nothing is removed from the original v5
detection dataset or moved between splits. From the repository root:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.make_review_queue `
  --out dataset_crack\hybrid_crack_masks\v5_2k_sam_pseudo_sewage10\manifest.csv `
  --sewage-positive-fraction 0.10 `
  --exclude-audit-manifest dataset_crack\hybrid_crack_masks\review_queue.csv

& 'C:\Users\parth\Documents\ComfyUI\.venv\Scripts\python.exe' `
  -m dataset_crack.hybrid_crack.build_pseudo_dataset `
  --manifest dataset_crack\hybrid_crack_masks\v5_2k_sam_pseudo_sewage10\manifest.csv `
  --top-n 4
```

The second command loads SAM once, compares up to four candidates for each
existing crack box, selects one candidate per box, unions those selections
for the image, and writes the mask at `masks/{split}/{image-stem}.png`.
It writes a copied 2,000-image subset at `{split}/images` with corresponding
`{split}/labels`, a standalone `data.yaml`, and the provenance manifest. It
can be stopped and rerun with the same command: completed masks are checked
and not replaced. `--limit 25` on the second command is an optional first
batch; rerun without it to finish all 2,000. `--save-overlays` also stores
visual overlays but is not required for the training masks.

Every generated row is marked `auto_pseudo`, **not** `approved`. A plausible
SAM score does not establish pixel accuracy; broad boxes, partial masks, and
unlabeled cracks remain possible. Review `quality_flags` in the manifest and
visually audit representatives from every source family, especially the
`sewage_*` family. Do not treat the auto-generated validation/test masks as
research-grade ground truth or use their Dice/IoU as a final claim.

For this hybrid experiment, keep `--data` pointed at the **full original v5**
dataset so all original training frames contribute box supervision, and point
`--manifest` at the generated subset. Because these masks are unreviewed,
training requires explicit opt-in:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.train `
  --data dataset_crack\final_dataset_v5\data.yaml `
  --manifest dataset_crack\hybrid_crack_masks\v5_2k_sam_pseudo_sewage10\manifest.csv `
  --allow-pseudo-masks --name yolo26n_v5_hybrid_pseudo2k
```

That run's validation-mask score measures agreement with pseudo-labels, not
ground truth. The copied subset's `data.yaml` is available if you deliberately
want a 2,000-image box-only ablation, but using it for the main hybrid run
would discard most of the v5 box training data.

### Exclude manually rejected generated images in one pass

After generation has finished, delete unwanted **mask PNGs** from
`v5_2k_sam_pseudo_sewage10/masks/train`, `masks/valid`, or `masks/test`, **or**
delete unwanted annotated JPEGs from the latest full `review_overlays_*`
folder. The sync tool maps missing overlays back to their mask rows using that
folder's `review_index.csv`; it ignores smaller pilot exports.
Do not delete from `final_dataset_v5` or edit the manifest by hand. Then run:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.sync_removed_masks --apply
```

The script scans the manifest for missing generated masks (and also missing
generated images or review overlays), moves any remaining corresponding generated image/label/mask
into `.rejected/`, saves a manifest backup, and marks the rows excluded. It is
safe to rerun. Omit `--apply` for a read-only preview. It refuses to process
more than 200 missing rows or an entire missing split folder without further
inspection. Training against the full v5 dataset also excludes the rejected
**train** images from box supervision;
the original v5 files remain intact for reproducibility. Rejected validation
and test masks are omitted from mask scoring, but the original full validation
and test box sets remain unchanged. Start a new training run after changing
the manifest; an earlier checkpoint is not retroactively updated.

To verify all surviving active rows, file counts, mask dimensions, class
presence, and the full-v5 training exclusions after cleanup, run:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.validate_curated_dataset
```

It writes a timestamped `curation_report_*.json` beside the manifest. The
generated subset's `data.yaml` is directly usable for a 2,000-image-subset
ablation; the recommended hybrid training command above deliberately keeps
the full original v5 box set and uses the curated mask manifest.

To review the remaining SAM masks **on the images**, with all five original
YOLO classes drawn as colored boxes, run:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.render_mask_overlays
```

It creates a new timestamped `review_overlays_*/index.html` and one annotated
JPEG per active mask. Magenta is the SAM crack mask, yellow is the crack box,
and cyan is the joint-defect box. The index can filter by split, filename,
positive/negative kind, and overlap with joint boxes. Overlap is a triage hint,
not proof that SAM segmented a joint. Delete bad masks from the main `masks/`
folder, not from the overlay folder, then rerun the sync command. Rerun the
renderer afterward to make a fresh review set.

Blank masks are crack-negative examples: the mask head uses them to learn
background and validation needs them to measure false positives. If all
validation/test blank masks are removed, negative-mask specificity cannot be
measured on those subsets; the training script warns about this.

## 1. Review masks

From the RoadEye repository root, the 2,000-row candidate queue already exists
at `dataset_crack/hybrid_crack_masks/review_queue.csv`. It contains 1,400
positive + 200 potential negative train images, 160 + 40 validation, and 160
+ 40 test. Existing v5 split membership is preserved. Run this only if you
want a new queue at a **different** path:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.make_review_queue --out dataset_crack\hybrid_crack_masks\another_review_queue.csv
```

The first SAM batch smoke test flagged `train/sewage_005837.jpg` as
`needs_label_audit`: its two crack boxes appear to cover bright protrusions
rather than clear cracks. That row is excluded from further proposal batches
and cannot be approved until the original labeling is reviewed.
The next trial, `train/sewer_o8use_003235.jpg`, contains a visible damaged
crack/spall region; its leading SAM candidate covers part of it but also broad
texture. It too needs manual mask correction before approval. These two trials
show why top-N union or confidence is not a substitute for annotation QA.

`negative` means no crack **box** is present, not a guaranteed crack-free
image. Visually verify every negative. The v5 split itself may contain
near-duplicate CCTV frames across splits; audit camera/video/site identity
before treating its test result as independent generalization.

To prepare multiple **unreviewed** SAM proposals without reloading SAM for
each image, run a ten-image pilot first. This never approves a queue row:

```powershell
& 'C:\Users\parth\Documents\ComfyUI\.venv\Scripts\python.exe' -m dataset_crack.hybrid_crack.sam_batch_proposals --split train --max-images 10 --with-text
```

Repeat with a larger `--max-images` only after checking the first batch's
quality and VRAM. Proposal folders are not overwritten; subsequent runs skip
existing folders. Review per-box `contact_sheet.png` files before approving.

For a positive image, run the existing local SAM experiment script in the
ComfyUI Python environment. Replace `NAME.jpg` with a filename from the queue:

```powershell
& 'C:\Users\parth\Documents\ComfyUI\.venv\Scripts\python.exe' dataset_crack\segment_crack_sam31.py --image dataset_crack\final_dataset_v5\train\images\NAME.jpg --no-text-prompts --top-n 4
```

It tests the original crack boxes. If a box encloses too much pipe, also try
one or more tight pixel boxes with `--no-box-prompts --pixel-box X1 Y1 X2 Y2`.
The text prompt `--prompt "pipe wall crack"` can rescue some branches, but its
highest-score candidate is **not** automatically best. Review the overlays,
select the masks that cover all real cracks without pipe rims or joints, and
combine them using `--assemble-mask PATH` repeated once per selected mask.
The final mask must be a single-channel PNG in the original image dimensions,
with only 0/255 values. The candidate score or top-N union alone is not proof
of correctness.

After visual review, approve one queue row without changing the source images
or labels. The example paths below are placeholders:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.approve_mask --split train --image NAME.jpg --mask "dataset_crack\sam31_preview\RUN\assembled_mask.png" --confirmed
```

For a visually confirmed image with **no** crack box and **no** visible crack,
omit `--mask`:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.approve_mask --split train --image NEGATIVE_NAME.jpg --confirmed
```

The approval script refuses to overwrite existing masks or approvals and
rejects a positive mask occupying more than 20% of the frame by default.
Correct incomplete original crack boxes before approving a contradictory mask.

## 2. Train and evaluate

The defaults are conservative for the local 3 GB GPU: 512 px, batch 2,
gradient accumulation 8, AdamW, and 50 epochs. Run names never overwrite a
different run. Start with a smoke test (no mask data needed):

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.train --smoke-test
```

Training requires approved train masks and approved positive validation masks.
The full 2,000-image review is not required before a small pilot, but more
representative reviewed masks improve the experiment's validity:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.train --name yolo26n_v5_hybrid_crack_pilot --epochs 50 --imgsz 512 --batch 2 --accumulate 8
```

If it runs out of memory, reduce `--batch 1` and increase `--accumulate 16`.
To continue an interrupted run to a **total** of 60 epochs:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.train --resume dataset_crack\runs\pipe_proto\yolo26n_v5_hybrid_crack_pilot\last_training.pt --epochs 60 --imgsz 512 --batch 2 --accumulate 8
```

`best_hybrid.pt` is selected by reviewed validation-mask Dice, **not** by box
mAP. `last_hybrid.pt` is the final epoch. The full optimizer/resume checkpoint
is `last_training.pt`. Evaluate on validation during development:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.evaluate --checkpoint dataset_crack\runs\pipe_proto\yolo26n_v5_hybrid_crack_pilot\best_hybrid.pt --split valid
```

After all design choices are frozen, evaluate once on the original full test
set for five-class box metrics plus only the separately reviewed test masks
for crack Dice/IoU:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.evaluate --checkpoint dataset_crack\runs\pipe_proto\yolo26n_v5_hybrid_crack_pilot\best_hybrid.pt --split test
```

One-image unified inference:

```powershell
venv\Scripts\python.exe -m dataset_crack.hybrid_crack.predict --checkpoint dataset_crack\runs\pipe_proto\yolo26n_v5_hybrid_crack_pilot\best_hybrid.pt --image dataset_crack\final_dataset_v5\test\images\sewage_000004.jpg --out dataset_crack\runs\pipe_proto\hybrid_example.png
```

The output shows four non-crack box classes plus a crack-pixel overlay. Add
`--show-crack-boxes` to display the fifth detection class as boxes too.

## Comparison and limits

For an apples-to-apples **box** control, train the same script with
`--mask-weight 0 --name yolo26n_v5_hybrid_box_control`, then compare
`last_hybrid.pt` from both runs on the same full validation or test split.
The control's untrained mask output is meaningless. Also compare with a
standard YOLO26n detection-only run at the same resolution, epochs and split.

Crack-mask Dice/IoU and box mAP50 are distinct metrics; never add or compare
them as percentages of the same thing. Good SAM proposals are not pixel-level
ground truth until reviewed. A successful training run does not guarantee a
90% five-class detection mAP50, particularly when YOLO26n is smaller than the
previous YOLO26s and RF-DETR models. This is an experiment, not a claimed gain.
