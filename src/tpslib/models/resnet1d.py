"""M2: 1D ResNet over the log-time feature channels, joined with the scalar inputs.

Stem conv -> 4 stages (32-64-128-128 channels, kernel 5, GELU; stages 2-4 halve the length) ->
global average pooling -> concat scalars -> 128-unit head -> mean and log-variance per target.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """Two 1D convolutions with batch norm and GELU plus a (projected) skip connection."""
    def __init__(self, c_in: int, c_out: int, k: int, stride: int):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
        self.skip = (nn.Identity() if (c_in == c_out and stride == 1) else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)))

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class ResNet1D(nn.Module):
    """M2 network: features (B, C, L) and scalars (B, S) -> mean and log-variance per target."""
    def __init__(self, in_channels: int = 3, n_scalars: int = 7, channels=(32, 64, 128, 128),
                 kernel_size: int = 5, head_hidden: int = 128, n_targets: int = 3,
                 logvar_bounds=(-14.0, 4.0)):
        super().__init__()
        k = kernel_size
        self.stem = nn.Sequential(nn.Conv1d(in_channels, channels[0], k, padding=k // 2, bias=False),
                                  nn.BatchNorm1d(channels[0]), nn.GELU())
        blocks, c_prev = [], channels[0]
        for i, c in enumerate(channels):
            blocks.append(ResBlock(c_prev, c, k, stride=1 if i == 0 else 2))
            c_prev = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(c_prev + n_scalars, head_hidden), nn.GELU(),
                                  nn.Linear(head_hidden, 2 * n_targets))
        self.n_targets = n_targets
        self.lv_min, self.lv_max = logvar_bounds

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor):
        h = self.blocks(self.stem(seq)).mean(dim=-1)
        out = self.head(torch.cat([h, scalars], dim=-1))
        mean, logvar = out[:, : self.n_targets], out[:, self.n_targets:]
        # soft bounds keep the variance away from 0 / infinity early in training
        logvar = self.lv_min + (self.lv_max - self.lv_min) * torch.sigmoid(logvar)
        return mean, logvar


def build_model(cfg: dict) -> ResNet1D:
    """Construct the network described by the `model` section of a training config."""
    m = cfg["model"]
    if m["name"] != "resnet1d":
        raise ValueError(f"unknown model {m['name']!r}")
    return ResNet1D(in_channels=int(m["in_channels"]), n_scalars=int(m["n_scalars"]),
                    channels=tuple(m["channels"]), kernel_size=int(m["kernel_size"]),
                    head_hidden=int(m["head_hidden"]), n_targets=len(m["targets"]),
                    logvar_bounds=tuple(m.get("logvar_bounds", (-14.0, 4.0))))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
