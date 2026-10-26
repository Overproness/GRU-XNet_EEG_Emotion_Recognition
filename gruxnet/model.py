"""Compact, explicitly named hardware variant; not the historical 95.91% model."""
import torch
from torch import nn


def time_frequency(waveforms, mask, augment=False):
    observed = mask.to(waveforms.dtype).unsqueeze(-1)
    centered = waveforms - waveforms.mean(dim=-1, keepdim=True)
    rms = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / rms * observed
    if augment:
        gain = .9 + .2 * torch.rand((*normalized.shape[:2], 1), device=normalized.device)
        normalized = normalized * gain + .03 * torch.randn_like(normalized) * observed
        keep = torch.rand(mask.shape, device=mask.device) >= .05
        # Never remove the last real electrode.
        empty = ~(mask & keep).any(dim=1)
        keep[torch.arange(len(mask), device=mask.device)[empty], mask.float().argmax(dim=1)[empty]] = True
        mask = mask & keep
        normalized = normalized * mask.unsqueeze(-1)
    b, c, length = normalized.shape
    hann = torch.hann_window(128, device=normalized.device)
    spectrum = torch.stft(normalized.reshape(b*c, length).float(), n_fft=128, hop_length=32,
                          window=hann, center=False, return_complex=True)
    features = torch.log1p(spectrum[:, 4:41].abs() / hann.sum()).reshape(b, c, 37, -1)
    return features, mask


class CompactGRUXNet(nn.Module):
    def __init__(self, channels=14, recurrent="gru", attention=True, frequency_pooling="flatten"):
        super().__init__()
        if recurrent not in ("gru", "lstm", "none"):
            raise ValueError(recurrent)
        self.channels = channels
        if frequency_pooling not in ("flatten", "mean"):
            raise ValueError(frequency_pooling)
        self.frequency_pooling = frequency_pooling
        layers = []
        incoming = 1
        for outgoing in (8, 16, 32):
            # Grouped convolutions have distinct parameters per physical electrode.
            layers.extend([nn.Conv2d(channels*incoming, channels*outgoing, 3, padding=1, groups=channels),
                           nn.GroupNorm(channels, channels*outgoing), nn.GELU(), nn.MaxPool2d((2, 1))])
            incoming = outgoing
        self.cnn = nn.Sequential(*layers)
        frequency_features = 4 if frequency_pooling == "flatten" else 1
        self.fusion = nn.Sequential(nn.Linear(channels*32*frequency_features, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.3))
        if recurrent == "none":
            self.recurrent = None
        else:
            cell = nn.GRU if recurrent == "gru" else nn.LSTM
            self.recurrent = cell(128, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=.3)
        self.attention = nn.MultiheadAttention(128, 4, dropout=.3, batch_first=True) if attention else None
        self.norm = nn.LayerNorm(128)
        self.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, features, mask):
        if features.ndim != 4 or features.shape[1] != self.channels or features.shape[2] != 37 or mask.shape != features.shape[:2]:
            raise ValueError("Expected features [batch, canonical channels, frequency, time] and a channel mask")
        if not mask.any(dim=1).all():
            raise ValueError("Every example needs an observed channel")
        b, c, _, _ = features.shape
        masked = features * mask[:, :, None, None]
        encoded = self.cnn(masked)
        # Zero absent channels AFTER biased convolutions and normalization as well.
        encoded = encoded.reshape(b, c, 32, encoded.shape[-2], encoded.shape[-1])
        encoded = encoded.mean(dim=3) if self.frequency_pooling == "mean" else encoded.flatten(2, 3)
        encoded = encoded * mask[:, :, None, None]
        sequence = self.fusion(encoded.permute(0, 3, 1, 2).reshape(b, encoded.shape[-1], -1))
        if self.recurrent is not None:
            sequence, _ = self.recurrent(sequence)
        if self.attention is not None:
            attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
            sequence = self.norm(sequence + attended)
        return self.classifier(sequence.mean(dim=1))
