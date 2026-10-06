"""Full-width historical architecture controls; independent of compact studies.

Grouped convolutions/BatchNorm have separate parameters/statistics for every
electrode. This vectorizes the original ModuleList, without sharing channels.
Input adaptation is declared separately; no historical-score reproduction claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class IndependentCNN(nn.Module):
    def __init__(self, global_pool=False):
        super().__init__()
        self.global_pool = global_pool
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for previous, current in ((1, 32), (32, 64), (64, 128)):
            self.convs.append(nn.Conv2d(14*previous, 14*current, 3, padding=1, groups=14))
            self.norms.append(nn.BatchNorm2d(14*current))
        self.drop = nn.Dropout(.5)

    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = F.max_pool2d(F.relu(norm(conv(x))), 2)
        if self.global_pool: x = x.mean((-2, -1), keepdim=True)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(self, reference=False):
        super().__init__()
        self.q = nn.Linear(256, 256); self.k = nn.Linear(256, 256)
        self.v = nn.Linear(256, 256); self.o = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256); self.drop = nn.Dropout(.5)
        self.reference = reference

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [m(x).reshape(b, t, 4, 64).transpose(1, 2) for m in (self.q, self.k, self.v)]
        a = self.drop(torch.softmax(q@k.transpose(-2, -1)/8, dim=-1))
        z = self.o((a@v).transpose(1, 2).reshape(b, t, 256))
        return self.norm(x+(z if self.reference else self.drop(z)))


class FullControl(nn.Module):
    def __init__(self, name, classes):
        super().__init__()
        if name not in ('gru', 'lstm', 'cbsatt_local', 'gru_context'):
            raise ValueError(name)
        self.name = name; self.cnn = IndependentCNN(global_pool=name == 'cbsatt_local')
        reference = name == 'cbsatt_local'
        recurrent = nn.GRU if name in ('gru', 'gru_context') else nn.LSTM
        self.recurrent = recurrent(14*128*(1 if reference else 4), 128,
                                  num_layers=1 if reference else 2,
                                  batch_first=True, bidirectional=True,
                                  dropout=0 if reference else .5)
        self.attention = Attention(reference)
        self.head = (nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.5), nn.Linear(128, classes))
                     if reference else nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(.5),
                                                     nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.5), nn.Linear(128, classes)))

    def forward(self, x, log_prior=None):
        x = self.cnn(x)
        if self.name == 'cbsatt_local':
            x = x.mean((-2, -1)).unsqueeze(1)
        else:
            b, _, f, t = x.shape
            x = x.reshape(b, 14, 128, f, t).permute(0, 4, 1, 2, 3).reshape(b, t, -1)
        x, _ = self.recurrent(x)
        logits = self.head(self.attention(x).mean(1))
        if self.name == 'gru_context':
            if log_prior is None or log_prior.shape != logits.shape:
                raise ValueError('A source-only log prior is required')
            logits = logits+log_prior
        return logits


def spectrogram(waveforms):
    """40s, 128Hz; preserve absolute amplitude for train-only normalization."""
    if waveforms.shape[1:] != (14, 5120):
        raise ValueError('Expected common14 first40s prefixes')
    b = len(waveforms)
    z = torch.stft(waveforms.to(torch.float64).reshape(-1, 5120), 128, hop_length=64,
                   window=torch.hann_window(128, device=waveforms.device, dtype=torch.float64),
                   center=False, return_complex=True)
    return torch.log1p(z.abs()[:, 4:41]/64).reshape(b, 14, 37, 79).float()


def copy_legacy(model, legacy):
    """Map original local weights to the vectorized implementation for QA."""
    with torch.no_grad():
        for j in range(3):
            convs = [getattr(m, f'conv{j+1}') for m in legacy.channel_cnns]
            norms = [getattr(m, f'bn{j+1}') for m in legacy.channel_cnns]
            for k in ('weight', 'bias'):
                getattr(model.cnn.convs[j], k).copy_(torch.cat([getattr(m, k) for m in convs]))
            for k in ('weight', 'bias', 'running_mean', 'running_var'):
                getattr(model.cnn.norms[j], k).copy_(torch.cat([getattr(m, k) for m in norms]))
            model.cnn.norms[j].num_batches_tracked.copy_(norms[0].num_batches_tracked)
        model.recurrent.load_state_dict((legacy.lstm if model.name == 'cbsatt_local' else legacy.bigru).state_dict())
        for local, old in ((model.attention.q, legacy.attention.W_q), (model.attention.k, legacy.attention.W_k),
                           (model.attention.v, legacy.attention.W_v), (model.attention.o, legacy.attention.W_o)):
            local.load_state_dict(old.state_dict())
        model.attention.norm.load_state_dict((legacy.layer_norm if model.name == 'cbsatt_local' else legacy.attention.layer_norm).state_dict())
        if model.name == 'cbsatt_local':
            model.head[0].load_state_dict(legacy.fc1.state_dict()); model.head[3].load_state_dict(legacy.fc2.state_dict())
        else:
            model.head.load_state_dict(legacy.classifier.state_dict())
