"""
SimAM -- parameter-free attention module (Yang et al., ICML 2021).

No learnable weights: computes per-position importance directly from each
channel's own statistics (how much a neuron's activation differs from the
surrounding ones), rather than learning a small MLP the way CBAM does.
Nothing new to converge, which is the point given the CBAM ablation's result.

Insert into a model yaml as:
    - [-1, 1, SimAM, []]

Register before building the model (same pattern as the CBAM patch):
    import ultralytics.nn.tasks as tasks
    from simam import SimAM
    tasks.SimAM = SimAM
"""
import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, c1=None, e_lambda=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)
