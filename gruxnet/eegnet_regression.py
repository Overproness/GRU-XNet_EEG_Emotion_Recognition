"""Explicit scalar adaptation of the author-checked EEGNet classifier backbone.

This is a local regression reference, not a reproduction of published scores.
The original classifier and its source/author audit remain unchanged.
"""
import math

import torch
from torch import nn

from .eegnet_control import EEGNetControl


class EEGNetRegression(EEGNetControl):
    def __init__(self):
        super().__init__(classes=2, samples=512, channels=30, dropout=.5, context=False)
        self.head = nn.Linear(256, 1)
        bound = math.sqrt(6 / 257)
        nn.init.uniform_(self.head.weight, -bound, bound)
        nn.init.zeros_(self.head.bias)

    def forward_windows(self, windows):
        if windows.ndim != 3 or windows.shape[1:] != (30,512):
            raise ValueError("Require physical channel/time geometry 30 by 512")
        return super().forward(windows).squeeze(-1)

    def forward(self, trials):
        if trials.ndim != 4 or trials.shape[1:] != (7,30,512):
            raise ValueError("Require seven windows per original trial")
        windows = self.forward_windows(trials.reshape(-1,30,512))
        # One scalar/loss per trial; never repeat the rating as seven examples.
        return windows.reshape(len(trials),7).mean(dim=1)


def original_scale_prediction(normalized, offset):
    if normalized.ndim != 1 or offset.shape != normalized.shape:
        raise ValueError("Trial output/offset mismatch")
    return 7 * normalized + offset
