"""Lightweight attention blocks used by the YOLO26 ablation scripts.

The classes deliberately accept the input channel count as their first
argument because Ultralytics' YAML parser does not automatically inject
``c1`` for third-party modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Parameter-free SimAM attention (Yang et al., ICML 2021)."""

    def __init__(self, c1: int | None = None, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        n = max(height * width - 1, 1)
        x_minus_mu_sq = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = x_minus_mu_sq / (
            4 * (x_minus_mu_sq.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * torch.sigmoid(y)


class SqueezeExcitation(nn.Module):
    """Channel-only squeeze-and-excitation block."""

    def __init__(self, c1: int, reduction: int = 16):
        super().__init__()
        hidden = max(int(c1) // int(reduction), 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel plus spatial attention)."""

    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        hidden = max(int(c1) // int(reduction), 1)
        if kernel_size not in (3, 7) or kernel_size % 2 == 0:
            raise ValueError("CBAM kernel_size must be 3 or 7")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_gate = torch.sigmoid(self.channel_mlp(self.avg_pool(x)) + self.channel_mlp(self.max_pool(x)))
        x = x * channel_gate
        spatial_mean = x.mean(dim=1, keepdim=True)
        spatial_max = x.amax(dim=1, keepdim=True)
        spatial_gate = torch.sigmoid(self.spatial(torch.cat((spatial_mean, spatial_max), dim=1)))
        return x * spatial_gate


class ParallelAttention(nn.Module):
    """Parallel SimAM + CBAM + SE fusion with a residual path.

    All three branches see the same P3 feature map. Their outputs, together
    with the unmodified feature map, are concatenated and projected back to
    the original channel count. A learnable residual scale keeps the
    pretrained YOLO path active while the new fusion learns.
    """

    def __init__(
        self,
        c1: int,
        reduction: int = 16,
        kernel_size: int = 7,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.simam = SimAM(c1)
        self.cbam = CBAM(c1, reduction, kernel_size)
        self.se = SqueezeExcitation(c1, reduction)
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * 4, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((x, self.simam(x), self.cbam(x), self.se(x)), dim=1)
        return x + self.residual_scale * self.fuse(fused)
