"""Small raw-EEG development control following the EEGNet-8,2 architecture.

Architecture reference: Lawhern et al., 2018, doi:10.1088/1741-2552/aace8c;
https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
This is a PyTorch implementation, not an exact replication of an emotion study.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EEGNetControl(nn.Module):
    def __init__(self, channels=32, samples=512, dropout=.5):
        super().__init__()
        if channels < 1 or samples < 32 or samples % 32:
            raise ValueError("EEGNet requires observed electrodes and samples divisible by 32")
        self.channels, self.samples = channels, samples
        # Explicit asymmetric padding preserves length for even temporal kernels.
        self.temporal = nn.Sequential(nn.ZeroPad2d((31, 32, 0, 0)),
                                      nn.Conv2d(1, 8, (1, 64), bias=False),
                                      nn.BatchNorm2d(8, eps=1e-3, momentum=.01))
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.block1 = nn.Sequential(nn.BatchNorm2d(16, eps=1e-3, momentum=.01), nn.ELU(),
                                    nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.ZeroPad2d((7, 8, 0, 0)),
                                    nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
                                    nn.Conv2d(16, 16, 1, bias=False),
                                    nn.BatchNorm2d(16, eps=1e-3, momentum=.01), nn.ELU(),
                                    nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        # Register the classifier BEFORE optimizer construction. Return logits
        # directly for CrossEntropyLoss; do not softmax twice.
        self.classifier = nn.Linear(16 * (samples // 32), 2)
        self.apply_constraints()

    def forward(self, waveforms):
        if waveforms.ndim != 3 or waveforms.shape[1:] != (self.channels, self.samples):
            raise ValueError("EEGNet expects batch x observed electrodes x declared samples")
        x = self.temporal(waveforms.unsqueeze(1))
        x = self.block1(self.spatial(x))
        return self.classifier(self.block2(x).flatten(1))

    @torch.no_grad()
    def apply_constraints(self):
        for weight, limit in [(self.spatial.weight, 1.), (self.classifier.weight, .25)]:
            norms = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
            factor = (limit / norms).clamp_max(1.)
            weight.mul_(factor.reshape(-1, *([1] * (weight.ndim - 1))))


class TrainingChannelScaler:
    """One frozen mean/std per electrode, fitted exclusively on training windows."""
    def __init__(self, mean, scale, count):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.count = int(count)
        if (self.mean.ndim != 1 or self.mean.shape != self.scale.shape or self.count < 1
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or (self.scale <= 0).any()):
            raise ValueError("Invalid channel statistics")

    @classmethod
    def fit(cls, training, cache):
        if training.empty or set(training.split) != {"train"}:
            raise ValueError("Scaler fitting accepts nonempty training rows only")
        count, mean, m2 = 0, None, None
        for filename, rows in training.groupby("cache_file", sort=True):
            saved = np.load(cache / filename, mmap_mode="r", allow_pickle=False)
            x = np.asarray(saved[rows.cache_index.astype(int)], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("Non-finite training EEG")
            n = x.shape[0] * x.shape[2]
            block_mean = x.mean(axis=(0, 2))
            block_m2 = x.var(axis=(0, 2)) * n
            if mean is None:
                mean, m2, count = block_mean, block_m2, n
            else:
                delta = block_mean - mean
                m2 += block_m2 + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
        return cls(mean, np.maximum(np.sqrt(m2 / count), 1e-6), count)

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, count=self.count)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            return cls(values["mean"], values["scale"], values["count"])

    def tensors(self, device):
        return (torch.as_tensor(self.mean, device=device, dtype=torch.float32)[None, :, None],
                torch.as_tensor(self.scale, device=device, dtype=torch.float32)[None, :, None])


def normalized_waveforms(waveforms, statistics, mode="train-channel"):
    if mode == "train-channel":
        mean, scale = statistics
        return (waveforms - mean) / scale
    if mode == "window":
        centered = waveforms - waveforms.mean(dim=-1, keepdim=True)
        return centered / centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    raise ValueError("Unknown normalization mode")
