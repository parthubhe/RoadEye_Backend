"""Train YOLO26n five-class boxes plus a crack-only semantic mask head.

Approved masks contribute mask loss by default. SAM pseudo-masks can be used
only with explicit --allow-pseudo-masks; all v5 train images contribute box
loss. Designed conservatively for a roughly 3 GB GPU (512 px, batch 2).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import HybridDataset, collate, dataset_paths, read_excluded_training_images, read_reviewed_masks
from .model import HybridCrackModel, deployment_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "final_dataset_v5" / "data.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "hybrid_crack_masks" / "review_queue.csv")
    parser.add_argument("--allow-pseudo-masks", action="store_true",
                        help="Explicitly train with unreviewed SAM auto_pseudo masks; validation mask scores are not ground truth")
    parser.add_argument("--weights", type=Path, default=ROOT.parent / "yolo26n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "pipe_proto")
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs, including resumed epochs")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, help="Path to last_training.pt in an existing run")
    parser.add_argument("--smoke-test", action="store_true", help="One synthetic forward/backward; no writes or dataset needed")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}


@torch.no_grad()
def validate_masks(model: HybridCrackModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    positive_dice, positive_iou, negative_fp_fraction, negative_any_fp = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        _, logits = model(batch["img"])
        truth = F.interpolate(batch["mask"].unsqueeze(1), size=logits.shape[-2:], mode="nearest") > 0
        prediction = logits.sigmoid() >= 0.5
        for i in range(len(prediction)):
            pred, target = prediction[i], truth[i]
            if target.any():
                overlap = (pred & target).sum().item()
                positive_dice.append((2 * overlap + 1) / (pred.sum().item() + target.sum().item() + 1))
                positive_iou.append((overlap + 1) / ((pred | target).sum().item() + 1))
            else:
                fraction = pred.float().mean().item()
                negative_fp_fraction.append(fraction)
                negative_any_fp.append(float(fraction > 0.001))
    if not positive_dice:
        raise ValueError("Validation set must include at least one approved positive crack mask")
    return {"positive_dice": float(np.mean(positive_dice)),
            "positive_iou": float(np.mean(positive_iou)),
            "negative_fp_fraction": float(np.mean(negative_fp_fraction)) if negative_fp_fraction else None,
            "negative_any_fp_rate": float(np.mean(negative_any_fp)) if negative_any_fp else None,
            "reviewed_positive": len(positive_dice), "reviewed_negative": len(negative_fp_fraction)}


def smoke_test() -> None:
    model = HybridCrackModel(pretrained=None).train()
    images = torch.rand(2, 3, 128, 128)
    masks = torch.zeros(2, 128, 128)
    masks[0, 30:90, 49:51] = 1
    batch = {"img": images, "mask": masks, "has_mask": torch.tensor([True, False]),
             "cls": torch.tensor([[0.]], dtype=torch.float32),
             "bboxes": torch.tensor([[0.39, 0.47, 0.08, 0.50]], dtype=torch.float32),
             "batch_idx": torch.tensor([0.], dtype=torch.float32)}
    predictions, logits = model(images)
    loss, det, seg = model.losses(batch, predictions, logits)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite smoke-test loss")
    loss.backward()
    print(f"Smoke test OK; parameters={model.parameter_count():,}; mask shape={tuple(logits.shape)}; "
          f"det_loss={det.item():.4f}; seg_loss={seg.item():.4f}")


def main() -> None:
    args = arguments()
    if args.smoke_test:
        smoke_test()
        return
    if min(args.epochs, args.batch, args.accumulate) < 1 or args.workers < 0:
        raise ValueError("epochs, batch and accumulate must be positive; workers non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    paths = dataset_paths(args.data.resolve())
    reviewed = read_reviewed_masks(args.manifest.resolve(), allow_pseudo=args.allow_pseudo_masks)
    excluded_train = read_excluded_training_images(args.manifest.resolve())
    train_set = HybridDataset(paths["train"], "train", reviewed, args.imgsz, augment=True,
                              exclude_names=excluded_train)
    val_set = HybridDataset(paths["valid"], "valid", reviewed, args.imgsz, only_reviewed=True)
    if train_set.approved_count < 1 or val_set.approved_count < 1:
        raise ValueError("Need usable train and validation masks; approve them or explicitly use --allow-pseudo-masks")
    if not any(kind == "positive" for (split, _), (_, kind) in reviewed.items() if split == "valid"):
        raise ValueError("Validation manifest needs usable crack-positive masks")
    if not any(kind == "negative" for (split, _), (_, kind) in reviewed.items() if split == "valid"):
        print("WARNING: no validation crack-negative masks remain; mask false-positive rate cannot be measured or used for checkpoint selection.", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", collate_fn=collate)

    if args.resume:
        resume_path = args.resume.resolve(strict=True)
        run_dir = resume_path.parent
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if (saved["imgsz"] != args.imgsz or saved["data"] != str(args.data.resolve()) or
                saved["manifest"] != str(args.manifest.resolve()) or
                saved["mask_weight"] != args.mask_weight or
                saved.get("allow_pseudo_masks", False) != args.allow_pseudo_masks):
            raise ValueError("Resume must use the same image size, data, review manifest and mask weight")
        model = HybridCrackModel(pretrained=None)
        model.load_state_dict(saved["model"])
        start_epoch = saved["epoch"] + 1
        best_score = saved["best_score"]
        if args.epochs <= start_epoch:
            raise ValueError(f"--epochs must exceed completed epochs ({start_epoch})")
    else:
        name = args.name or f"yolo26n_v5_hybrid_crack_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = args.project.resolve() / name
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
        model = HybridCrackModel(pretrained=str(args.weights.resolve(strict=True)))
        start_epoch, best_score = 0, float("-inf")
        run_dir.mkdir(parents=True, exist_ok=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scheduler.T_max = args.epochs  # extend cosine schedule when --epochs increases on resume
        scaler.load_state_dict(saved["scaler"])
    config = {"architecture": "yolo26n_5box_crack_semantic_v1", "data": str(args.data.resolve()),
              "manifest": str(args.manifest.resolve()), "imgsz": args.imgsz, "epochs": args.epochs,
              "batch": args.batch, "accumulate": args.accumulate, "lr": args.lr,
              "mask_weight": args.mask_weight, "seed": args.seed, "device": str(device),
              "allow_pseudo_masks": args.allow_pseudo_masks,
              "excluded_train_images": len(excluded_train),
              "train_images": len(train_set), "train_reviewed_masks": train_set.approved_count,
              "val_reviewed_masks": val_set.approved_count, "parameters": model.parameter_count()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run: {run_dir}\n{len(train_set)} box-labeled training images; "
          f"{train_set.approved_count} usable train masks; {val_set.approved_count} usable val masks; "
          f"{len(excluded_train)} manually excluded train images; "
          f"{model.parameter_count():,} parameters", flush=True)
    if args.allow_pseudo_masks:
        print("WARNING: SAM auto_pseudo masks are unreviewed. Validation mask Dice/IoU are pseudo-label agreement, not ground-truth accuracy.", flush=True)
    metrics_file = run_dir / "hybrid_metrics.csv"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(3, dtype=np.float64)
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predictions, logits = model(batch["img"])
                loss, detection, segmentation = model.losses(batch, predictions, logits, args.mask_weight)
            scaler.scale(loss / args.accumulate).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            totals += (loss.item(), detection.item(), segmentation.item())
            if (step + 1) % 200 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} step {step + 1}/{len(train_loader)} "
                      f"loss={totals[0] / (step + 1):.3f}", flush=True)
        scheduler.step()
        val = validate_masks(model, val_loader, device)
        score = val["positive_dice"] - 0.1 * (val["negative_any_fp_rate"] or 0)
        row = {"epoch": epoch + 1, "train_loss": totals[0] / len(train_loader),
               "train_detection_loss": totals[1] / len(train_loader),
               "train_mask_loss": totals[2] / len(train_loader), "val_mask_score": score,
               **val, "lr": scheduler.get_last_lr()[0]}
        with metrics_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        if score > best_score:
            best_score = score
            torch.save(deployment_checkpoint(model, epoch=epoch + 1, validation=val,
                                             data=str(args.data.resolve()), imgsz=args.imgsz),
                       run_dir / "best_hybrid.pt")
        torch.save(deployment_checkpoint(model, epoch=epoch + 1, validation=val,
                                         data=str(args.data.resolve()), imgsz=args.imgsz),
                   run_dir / "last_hybrid.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_score": best_score, "data": str(args.data.resolve()),
                    "manifest": str(args.manifest.resolve()), "mask_weight": args.mask_weight,
                    "allow_pseudo_masks": args.allow_pseudo_masks,
                    "imgsz": args.imgsz}, run_dir / "last_training.pt")
        print(f"epoch {epoch + 1}/{args.epochs}: box-loss={row['train_detection_loss']:.3f}; "
              f"mask-loss={row['train_mask_loss']:.3f}; val positive Dice={val['positive_dice']:.3f}; "
              f"val positive IoU={val['positive_iou']:.3f}; neg FP-image rate={val['negative_any_fp_rate']}",
              flush=True)
    print(f"Done. Best unified checkpoint: {run_dir / 'best_hybrid.pt'}")
    print("Selection used validation-mask Dice, not box mAP; assess both heads on an untouched, human-reviewed test set.")


if __name__ == "__main__":
    main()
