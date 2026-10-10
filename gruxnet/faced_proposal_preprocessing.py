"""Synthetic-testable proposal adapter, with no raw-file reader or outcome access.

This is a proposed offline trial-local pipeline, not author-cleaned FACED or
permission to process reserved recordings. Originals are never edited here.
"""
from __future__ import annotations
from math import gcd
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

ALIASES={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
COMMON_CHANNELS=['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6',
    'Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6','Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']
UNIT_REPAIR_SUBJECTS={f'sub-{i:03d}' for i in [4,5,*range(8,36),58,59,60]}


def common_indices(labels):
    mapped=[ALIASES.get(name,name) for name in labels]
    if len(mapped)!=len(set(mapped)):
        raise ValueError('Duplicate channel aliases')
    if set(mapped)-set(COMMON_CHANNELS)-{'A1','A2','HEOR','HEOL'} or not set(COMMON_CHANNELS)<=set(mapped):
        raise ValueError('Unknown or missing scalp channel')
    return [mapped.index(name) for name in COMMON_CHANNELS]


def calibrate_common_counts(counts,subject,header):
    """Convert signed integer ADC counts to calibrated microvolts and 30 channels.

    The eventual file loader must first bind header and body to qualification
    receipts. This pure function accepts a qualified technical header projection.
    """
    counts=np.asarray(counts)
    c=header['columns']
    if counts.ndim!=2 or counts.shape[0]!=len(c['labels']) or counts.dtype.kind not in 'iu':
        raise ValueError('Expected channel-by-sample integer ADC counts')
    units=set(c['units'])
    if units=={'?V'}:
        if subject not in UNIT_REPAIR_SUBJECTS:
            raise ValueError('Undocumented corrupted unit label')
    elif units!={'uV'} or subject in UNIT_REPAIR_SUBJECTS:
        raise ValueError('Unit literal differs from qualified original')
    if set(header['sample_rates_hz']) not in ({250.0},{1000.0}):
        raise ValueError('Unsupported original acquisition rate')
    pmin=np.asarray(c['physical_min'],dtype=np.float64)[:,None]
    pmax=np.asarray(c['physical_max'],dtype=np.float64)[:,None]
    dmin=np.asarray(c['digital_min'],dtype=np.float64)[:,None]
    dmax=np.asarray(c['digital_max'],dtype=np.float64)[:,None]
    if any(not np.isfinite(v).all() for v in (pmin,pmax,dmin,dmax)) or np.any(pmax<=pmin) or np.any(dmax<=dmin):
        raise ValueError('Invalid header calibration')
    if np.any(counts<dmin) or np.any(counts>dmax):
        raise ValueError('ADC counts outside declared digital range')
    indices=common_indices(c['labels'])
    microvolts=pmin+(counts.astype(np.float64)-dmin)*(pmax-pmin)/(dmax-dmin)
    return microvolts[indices]


def trial_windows(microvolts,sampling_rate):
    """32 seconds from ceil(start_marker*fs); return 7x30x512 offline windows.

    Never filters/resamples a concatenated recording or another trial's context.
    No fitted scaler, ICA, interpolation or data-selected rejection is performed.
    """
    if sampling_rate not in (250,1000):
        raise ValueError('Unsupported acquisition rate')
    values=np.asarray(microvolts,dtype=np.float64)
    if values.shape!=(30,32*sampling_rate) or not np.isfinite(values).all():
        raise ValueError('Expected finite 32-second, 30-channel trial-local segment')
    referenced=values-values.mean(axis=0,keepdims=True)
    sos=butter(4,[.5,45.0],btype='bandpass',fs=sampling_rate,output='sos')
    # Two-pass zero-phase filtering is offline and has a squared magnitude response.
    filtered=sosfiltfilt(sos,referenced,axis=-1,padtype='odd')
    divisor=gcd(sampling_rate,128)
    resized=resample_poly(filtered,128//divisor,sampling_rate//divisor,axis=-1,
                          window=('kaiser',5.0),padtype='line')
    if resized.shape!=(30,4096):
        raise ValueError('Resampling geometry changed')
    core=resized[:,256:3840]  # 2..30 seconds, 28-second fixed observation.
    return np.stack([core[:,i*512:(i+1)*512] for i in range(7)]).astype(np.float32)
