"""Importable checkpoint classes; all preserve input channel count and resolution."""
import torch
from torch import nn
from ultralytics.nn.modules import Conv


class WeightedFusion(nn.Module):
    """Project concatenated branches, learn a normalized weighted sum, add a residual.

    This adapts weighted fusion inside the YOLO PAN neck; it is NOT a full BiFPN.
    Output channels equal the concatenated input so no parser patch is necessary.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = list(channels)
        width = min(channels)
        self.project = nn.ModuleList(Conv(c, width, 1) for c in channels)
        self.weights = nn.Parameter(torch.zeros(len(channels)))
        self.expand = Conv(width, sum(channels), 1)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        parts = x.split(self.channels, dim=1)
        weights = self.weights.softmax(dim=0)
        fused = sum(w * layer(part) for w, layer, part in zip(weights, self.project, parts))
        return x + self.gain * self.expand(fused)


class DirectionalConv(nn.Module):
    """Residual depthwise horizontal/vertical 7-pixel kernels; not deformable convolution."""
    def __init__(self, channels, kernel=7):
        super().__init__()
        self.horizontal = nn.Conv2d(channels, channels, (1, kernel), padding=(0, kernel // 2),
                                    groups=channels, bias=False)
        self.vertical = nn.Conv2d(channels, channels, (kernel, 1), padding=(kernel // 2, 0),
                                  groups=channels, bias=False)
        self.fuse = Conv(2 * channels, channels, 1)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.gain * self.fuse(torch.cat((self.horizontal(x), self.vertical(x)), 1))


class SimAM(nn.Module):
    def __init__(self, channels=None, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        n = max(x.shape[-1] * x.shape[-2] - 1, 1)
        delta = (x - x.mean(dim=(2, 3), keepdim=True)).square()
        energy = delta / (4 * (delta.sum(dim=(2, 3), keepdim=True) / n + self.epsilon)) + 0.5
        return x * energy.sigmoid()


def register_modules():
    import ultralytics.nn.tasks as tasks
    for module in (WeightedFusion, DirectionalConv, SimAM):
        setattr(tasks, module.__name__, module)
