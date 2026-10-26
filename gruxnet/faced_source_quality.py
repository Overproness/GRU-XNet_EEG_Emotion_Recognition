"""Selective FACED BDF decoding and descriptive source-only quality checks.

No behavioral reader, target decoder, fitted transform, or trial rejection.
The caller must authenticate the file and authorize exact identities first.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.signal import welch

from .faced_proposal_preprocessing import COMMON_CHANNELS


QUALITY_RULES = {
    'raw_rail_fraction_above': 0.0001,
    'raw_identical_adjacent_fraction_above': 0.95,
    'processed_channel_sd_below_uv': 0.1,
    'processed_channel_sd_above_uv': 100.0,
    'processed_channel_peak_to_peak_above_uv': 1000.0,
    'interpretation': 'Engineering review flags, not validated artifact labels or automatic exclusions',
}


def decode_signed24(payload, reference=False):
    if len(payload) % 3:
        raise ValueError('Truncated 24-bit sample')
    b = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    u = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    values = (u ^ 0x800000) - 0x800000
    if reference:
        alternate = np.fromiter(
            (int.from_bytes(payload[i:i+3], 'little', signed=True)
             for i in range(0, len(payload), 3)), dtype=np.int32, count=len(values))
        if not np.array_equal(values, alternate):
            raise ValueError('Signed decoder disagrees with independent scalar conversion')
    return values


def read_interval(stream, header, start, stop, channels, reference=False):
    """Read only selected channels and samples, including partial boundary records."""
    c = header['columns']
    sizes = np.asarray(c['samples_per_record'], dtype=np.int64)
    channels = list(channels)
    if (header['sample_width'] != 3 or header['continuity'] not in ('BDF', '24BIT')
            or len(channels) != len(set(channels)) or not channels
            or any(i < 0 or i >= len(sizes) for i in channels)):
        raise ValueError('Unsupported BDF layout or channel selection')
    if len(set(sizes.tolist())) != 1 or sizes[0] <= 0:
        raise ValueError('Require equal positive original channel sample counts')
    per_record = int(sizes[0])
    total = header['record_count'] * per_record
    if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= total:
        raise ValueError('Interval outside recording geometry')
    result = np.empty((len(channels), stop-start), dtype=np.int32)
    record_bytes = 3 * int(sizes.sum())
    prefixes = 3 * np.r_[0, np.cumsum(sizes[:-1])]
    ranges = []
    for record in range(start // per_record, (stop-1) // per_record + 1):
        lo, hi = max(start, record*per_record), min(stop, (record+1)*per_record)
        for j, channel in enumerate(channels):
            offset = int(header['header_bytes'] + record*record_bytes + prefixes[channel]
                         + 3*(lo-record*per_record))
            stream.seek(offset)
            payload = stream.read(3*(hi-lo))
            if len(payload) != 3*(hi-lo):
                raise ValueError('Truncated selected sample range')
            result[j, lo-start:hi-start] = decode_signed24(payload, reference)
            ranges.append({'channel_index': channel, 'start_sample': lo,
                           'stop_sample': hi, 'offset': offset, 'bytes': len(payload)})
    ledger = json.dumps(ranges, sort_keys=True, separators=(',', ':')).encode()
    return result, {
        'ranges': ranges, 'range_ledger_sha256': hashlib.sha256(ledger).hexdigest(),
        'sample_values_decoded': int(result.size), 'signal_bytes_decoded': int(result.size*3),
        'extra_boundary_samples_decoded': 0,
        'all_counts_scalar_reference_checked': reference,
    }


def selected_header(header, channels):
    return {**header,
            'sample_rates_hz': [header['sample_rates_hz'][i] for i in channels],
            'columns': {k: [v[i] for i in channels] for k, v in header['columns'].items()}}


def quality_metrics(counts, microvolts, windows, rate, rules=QUALITY_RULES):
    """Describe every channel, retaining all windows and all trials."""
    counts, microvolts = np.asarray(counts), np.asarray(microvolts)
    if (counts.shape != (30, 32*rate) or microvolts.shape != counts.shape
            or windows.shape != (7, 30, 512) or not np.isfinite(microvolts).all()
            or not np.isfinite(windows).all()):
        raise ValueError('Quality input geometry/nonfinite failure')
    core = windows.transpose(1, 0, 2).reshape(30, -1).astype(np.float64)
    referenced = microvolts-microvolts.mean(axis=0, keepdims=True)
    f, p = welch(referenced, fs=rate, nperseg=4*rate, noverlap=2*rate, axis=-1)
    pf, pp = welch(core, fs=128, nperseg=512, noverlap=256, axis=-1)

    def power(freqs, psd, lo, hi):
        selected = (freqs >= lo) & (freqs < hi)
        return psd[:, selected].sum(axis=-1)*(freqs[1]-freqs[0])

    native_low = power(f, p, 1, 45)
    native_line = power(f, p, 49, 51)
    processed_line = power(pf, pp, 49, 51)
    processed_low = power(pf, pp, 1, 45)
    rows = []
    for i, channel in enumerate(COMMON_CHANNELS):
        sd = float(core[i].std())
        ptp = float(np.ptp(core[i]))
        rails = float(np.mean((counts[i] == -8388608) | (counts[i] == 8388607)))
        identical = float(np.mean(counts[i, 1:] == counts[i, :-1]))
        flags = []
        if rails > rules['raw_rail_fraction_above']: flags.append('adc_rail')
        if identical > rules['raw_identical_adjacent_fraction_above']: flags.append('stationary_counts')
        if sd < rules['processed_channel_sd_below_uv']: flags.append('low_processed_variation')
        if sd > rules['processed_channel_sd_above_uv']: flags.append('high_processed_variation')
        if ptp > rules['processed_channel_peak_to_peak_above_uv']: flags.append('high_processed_excursion')
        row = {
            'channel': channel, 'raw_median_uv': float(np.median(microvolts[i])),
            'raw_sd_uv': float(microvolts[i].std()), 'raw_peak_to_peak_uv': float(np.ptp(microvolts[i])),
            'raw_rail_fraction': rails, 'raw_identical_adjacent_fraction': identical,
            'processed_sd_uv': sd, 'processed_peak_to_peak_uv': ptp,
            'processed_abs_q99_uv': float(np.quantile(np.abs(core[i]), .99)),
            'native_car_49_51_to_1_45_power_ratio': float(native_line[i]/max(native_low[i], 1e-30)),
            'processed_49_51_to_1_45_power_ratio': float(processed_line[i]/max(processed_low[i], 1e-30)),
            'review_flags': flags,
        }
        for band, lo, hi in [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]:
            row[f'processed_{band}_power_uv2'] = float(power(pf, pp, lo, hi)[i])
        rows.append(row)
    by_window_sd = windows.astype(np.float64).std(axis=-1)
    return {
        'channels': rows, 'flagged_channels': [r['channel'] for r in rows if r['review_flags']],
        'processed_channel_sd_uv_min_median_max': [float(x) for x in np.quantile(core.std(axis=-1),[0,.5,1])],
        'window_median_channel_sd_uv': np.median(by_window_sd, axis=1).tolist(),
        'common_average_max_abs_uv': float(np.max(np.abs(core.mean(axis=0)))),
        'valid_windows': 7, 'windows_dropped': 0, 'finite_processed_values': True,
        'scope': 'Descriptive pilot quality; no confirmed neural origin, artifact freedom, target information or population prevalence',
    }
