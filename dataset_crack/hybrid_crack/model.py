"""One-pass hybrid detector: YOLO26n boxes plus a lightweight crack mask head.

The mask is semantic (all crack pixels), not a separate mask for each crack.
SAM is used only to prepare labels; it is not part of this deployed model.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.nn.tasks import DetectionModel


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class CrackMaskHead(nn.Module):
    """Fuse P3/8, P2/4 and P1/2 features; output half-resolution logits."""

    def __init__(self):
        super().__init__()
        self.p3 = nn.Conv2d(64, 32, 1)
        self.p2 = nn.Conv2d(64, 32, 1)
        self.mid = ConvBlock(32, 24)
        self.p1 = nn.Conv2d(16, 24, 1)
        self.fine = ConvBlock(24, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        x = self.p2(p2) + F.interpolate(self.p3(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.mid(x)
        x = self.p1(p1) + F.interpolate(x, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.fine(x))


class HybridCrackModel(nn.Module):
    """A single shared YOLO26n network with box and binary-mask outputs.

    Hook indices are checked because they are specific to Ultralytics YOLO26n.
    A dependency upgrade must be smoke-tested before training or inference.
    """

    def __init__(self, pretrained: str | None = "yolo26n.pt"):
        super().__init__()
        self.detector = DetectionModel("yolo26n.yaml", nc=5, verbose=False)
        if pretrained:
            source = YOLO(pretrained).model
            if source.yaml.get("scale") != "n":
                raise ValueError("Expected YOLO26n pretrained weights")
            self.detector.load(source, verbose=False)
        self.detector.args = get_cfg(DEFAULT_CFG, {"task": "detect", "box": 7.5, "cls": 0.5, "dfl": 1.5})
        self.mask_head = CrackMaskHead()
        self._features: dict[int, torch.Tensor] = {}
        expected = {0: "Conv", 2: "C3k2", 16: "C3k2", 23: "Detect"}
        if len(self.detector.model) != 24 or any(
            type(self.detector.model[i]).__name__ != name for i, name in expected.items()
        ):
            raise RuntimeError("YOLO26n layer layout changed; update the feature taps before training")
        self._hooks = [
            self.detector.model[i].register_forward_hook(self._capture(i)) for i in (0, 2, 16)
        ]

    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            self._features[index] = output
        return hook

    def forward(self, images: torch.Tensor):
        self._features.clear()
        detection = self.detector(images)
        if set(self._features) != {0, 2, 16}:
            raise RuntimeError("Missing feature taps; YOLO26n architecture may have changed")
        mask_logits = self.mask_head(self._features[0], self._features[2], self._features[16])
        self._features.clear()
        return detection, mask_logits

    def losses(self, batch: dict, detection, mask_logits: torch.Tensor, mask_weight: float = 1.0):
        raw_detection, _ = self.detector.loss(batch, preds=detection)
        detection_loss = raw_detection.sum() / batch["img"].shape[0]
        reviewed = batch["has_mask"].bool()
        if reviewed.any():
            logits = mask_logits[reviewed].float()
            targets = F.interpolate(batch["mask"][reviewed].unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")
            positive_weight = torch.tensor(8.0, device=logits.device)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
            probs = logits.sigmoid().flatten(1)
            truth = targets.flatten(1)
            positive = truth.sum(1) > 0
            if positive.any():
                intersection = (probs[positive] * truth[positive]).sum(1)
                dice = 1 - ((2 * intersection + 1) / (probs[positive].sum(1) + truth[positive].sum(1) + 1)).mean()
            else:
                dice = bce.new_zeros(())
            mask_loss = bce + dice
        else:
            # An absent mask is *unknown*, never a background target.
            mask_loss = mask_logits.sum() * 0
        return detection_loss + mask_weight * mask_loss, detection_loss.detach(), mask_loss.detach()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_hybrid(path: str, device: str | torch.device = "cpu") -> HybridCrackModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "yolo26n_5box_crack_semantic_v1":
        raise ValueError("Not a compatible hybrid checkpoint")
    model = HybridCrackModel(pretrained=None)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def deployment_checkpoint(model: HybridCrackModel, **metadata) -> dict:
    return {
        "architecture": "yolo26n_5box_crack_semantic_v1",
        "model": {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": deepcopy(metadata),
    }
