"""EEGNet-8,2 port checked against the pinned authors' Keras release.

Reference: Lawhern et al. (2018), doi:10.1088/1741-2552/aace8c.
Source: https://github.com/vlawhern/arl-eegmodels
Port and experiment controls are local; original optimizer trajectories and
published scores are not reproduced. See the accompanying author-code audit.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EEGNetControl(nn.Module):
    def __init__(self, classes, samples=5120, channels=14, dropout=.5, context=False):
        super().__init__()
        if samples % 32 or classes < 2:
            raise ValueError('Samples must be divisible by 32; at least two classes')
        self.context = context
        self.temporal = nn.Conv2d(1, 8, (1, 64), bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-3, momentum=.01)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-3, momentum=.01)
        self.depthwise = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-3, momentum=.01)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(16*(samples//32), classes)
        self.channels, self.samples = channels, samples
        # Keras Glorot fans refer to its ungrouped HWIO/depthwise HWIM tensors.
        for layer, fan_in, fan_out in ((self.temporal, 64, 512),
                                      (self.spatial, channels*8, channels*2),
                                      (self.depthwise, 16*16, 16),
                                      (self.pointwise, 16, 16),
                                      (self.head, self.head.in_features, classes)):
            nn.init.uniform_(layer.weight, -math.sqrt(6/(fan_in+fan_out)),
                             math.sqrt(6/(fan_in+fan_out)))
        nn.init.zeros_(self.head.bias)

    def forward(self, x, log_prior=None):
        if x.shape[1:] != (self.channels, self.samples):
            raise ValueError('Expected physical electrodes and full declared prefix')
        # TensorFlow SAME with even kernels puts the extra zero on the right.
        x = self.bn1(self.temporal(F.pad(x.unsqueeze(1), (31, 32))))
        x = self.drop(F.avg_pool2d(F.elu(self.bn2(self.spatial(x))), (1, 4)))
        x = self.pointwise(self.depthwise(F.pad(x, (7, 8))))
        x = self.drop(F.avg_pool2d(F.elu(self.bn3(x)), (1, 8)))
        # Authors' channels-last Flatten: time-major, then feature filters.
        logits = self.head(x.permute(0, 2, 3, 1).reshape(len(x), -1))
        if self.context:
            if log_prior is None or log_prior.shape != logits.shape:
                raise ValueError('Source-only participant-excluded prior required')
            logits = logits+log_prior
        return logits

    @torch.no_grad()
    def constrain(self):
        # Keras max_norm's default axis=0: electrodes for HWIM, inputs for IO.
        for weight, dimensions, bound in ((self.spatial.weight, (2,), 1.),
                                          (self.head.weight, (1,), .25)):
            norm = torch.sqrt(torch.sum(weight.square(), dim=dimensions, keepdim=True))
            weight.mul_(torch.clamp(norm, max=bound)/(norm+1e-7))

    @torch.no_grad()
    def copy_author_weights(self, weights):
        def copy(layer, values):
            for field, value in values.items():
                getattr(layer, field).copy_(torch.as_tensor(value))
        copy(self.temporal, {'weight': weights['conv2d'][0].transpose(3, 2, 0, 1)})
        copy(self.spatial, {'weight': weights['depthwise_conv2d'][0].transpose(2, 3, 0, 1).reshape(16, 1, self.channels, 1)})
        separable = weights['separable_conv2d']
        copy(self.depthwise, {'weight': separable[0].transpose(2, 3, 0, 1).reshape(16, 1, 1, 16)})
        copy(self.pointwise, {'weight': separable[1].transpose(3, 2, 0, 1)})
        for layer, name in ((self.bn1, 'batch_normalization'),
                            (self.bn2, 'batch_normalization_1'),
                            (self.bn3, 'batch_normalization_2')):
            copy(layer, dict(zip(('weight', 'bias', 'running_mean', 'running_var'), weights[name])))
        copy(self.head, {'weight': weights['dense'][0].T, 'bias': weights['dense'][1]})
