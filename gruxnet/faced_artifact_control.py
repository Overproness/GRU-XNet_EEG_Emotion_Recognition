"""One fixed pre-CAR amplitude/interpolation control, not author-code reproduction.

No recording/rating reader, collective fitted transform or target information.
MNE is used on manufactured identity data to qualify spatial weights only.
"""
from functools import lru_cache
from math import gcd
import numpy as np
from scipy.linalg import pinv
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from scipy.special import eval_legendre
import mne
from mne.bem import fit_sphere_to_headshape
from .faced_proposal_preprocessing import COMMON_CHANNELS
from .faced_source_notch import notched_late_trial_windows

RULES = {'pre_car_core_sd_above_uv': 100., 'pre_car_core_peak_to_peak_above_uv': 1000.,
         'maximum_interpolated_channels': 3, 'threshold_scope': 'Fixed signal-only engineering rules; not validated artifact labels'}


@lru_cache(maxsize=1)
def geometry():
    if mne.__version__ != '1.10.0': raise ValueError('Qualify a changed MNE runtime separately')
    info = mne.create_info(COMMON_CHANNELS, 128., 'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))
    radius, origin, _ = fit_sphere_to_headshape(info, units='m', verbose=False)
    positions = np.asarray([c['loc'][:3] for c in info['chs']])
    if positions.shape != (30,3) or not np.isfinite(positions).all(): raise ValueError('Invalid common-scalp geometry')
    return {'channel_names': list(COMMON_CHANNELS), 'positions_head_m': positions.tolist(),
            'sphere_origin_head_m': origin.tolist(), 'sphere_radius_m': float(radius),
            'mne_version': mne.__version__, 'montage': 'standard_1020',
            'scope': 'Standard nominal geometry, not individual digitization or authenticated FACED author coordinates'}


def independent_spline_weights(positions, origin, bad_indices):
    """Perrin spherical spline system, independently evaluated Legendre series."""
    bad = list(bad_indices); good = [i for i in range(30) if i not in bad]
    p = np.asarray(positions, float)-np.asarray(origin, float)
    p = p/np.linalg.norm(p,axis=1,keepdims=True)
    def kernel(dot):
        return sum((2*n+1)/(4*np.pi*n**4*(n+1)**4)*eval_legendre(n,dot) for n in range(1,51))
    g = kernel(p[good]@p[good].T)+np.eye(len(good))*1e-5
    system = np.block([[g,np.ones((len(good),1))],[np.ones((1,len(good))),np.zeros((1,1))]])
    weights = np.c_[kernel(p[bad]@p[good].T),np.ones(len(bad))]@pinv(system)[:,:-1]
    return weights, good


@lru_cache(maxsize=64)
def qualified_weights(bad_indices):
    bad = tuple(bad_indices)
    if not 1 <= len(bad) <= 3 or tuple(sorted(set(bad))) != bad or any(i<0 or i>=30 for i in bad):
        raise ValueError('Outside fixed interpolation capacity')
    g = geometry(); weights, good = independent_spline_weights(g['positions_head_m'],g['sphere_origin_head_m'],bad)
    info = mne.create_info(COMMON_CHANNELS,128.,'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw = mne.io.RawArray(np.eye(30),info,verbose=False)
    raw.info['bads'] = [COMMON_CHANNELS[i] for i in bad]
    raw.interpolate_bads(origin=tuple(g['sphere_origin_head_m']),method={'eeg':'spline'},mode='accurate',reset_bads=True,verbose=False)
    public = raw.get_data(picks=list(bad))[:,good]
    discrepancy = float(np.max(np.abs(public-weights)))
    if (not np.isfinite(weights).all() or discrepancy > 1e-9
            or np.max(np.abs(weights.sum(axis=1)-1)) > 1e-9 or np.max(np.abs(weights)) > 5):
        raise ValueError('Spatial weight qualification failed')
    return weights, tuple(good), discrepancy


def pre_car_windows(values, rate):
    """Existing temporal operations without CAR, for detection only."""
    x = np.asarray(values,dtype=np.float64)
    if rate not in (250,1000) or x.shape != (30,32*rate) or not np.isfinite(x).all():
        raise ValueError('Require finite qualified late32s scalp input')
    b,a=iirnotch(50,30,fs=rate); filtered=filtfilt(b,a,x,axis=-1,padtype='odd')
    sos=butter(4,[.5,45.],btype='bandpass',fs=rate,output='sos')
    filtered=sosfiltfilt(sos,filtered,axis=-1,padtype='odd')
    divisor=gcd(rate,128)
    resized=resample_poly(filtered,128//divisor,rate//divisor,axis=-1,window=('kaiser',5.),padtype='line')
    core=resized[:,256:3840]
    if core.shape != (30,3584): raise ValueError('Detection interval changed')
    return np.stack([core[:,i*512:(i+1)*512] for i in range(7)]).astype(np.float32)


def detect_high_channels(windows):
    if np.asarray(windows).shape != (7,30,512) or not np.isfinite(windows).all():
        raise ValueError('Invalid pre-CAR detection windows')
    core=windows.transpose(1,0,2).reshape(30,-1).astype(float)
    sd=core.std(axis=1); excursion=np.ptp(core,axis=1)
    bad=np.flatnonzero((sd>RULES['pre_car_core_sd_above_uv'])|(excursion>RULES['pre_car_core_peak_to_peak_above_uv'])).tolist()
    return bad,sd,excursion


def interpolate_input(values,bad):
    x=np.asarray(values,dtype=np.float64)
    weights,good,discrepancy=qualified_weights(tuple(bad))
    repaired=x.copy(); repaired[bad]=weights@x[list(good)]
    if not np.array_equal(repaired[list(good)],x[list(good)]) or not np.isfinite(repaired).all():
        raise ValueError('Interpolation changed donors or produced nonfinite values')
    return repaired,weights,discrepancy


def artifact_trial_windows(values,rate):
    """Retain all trials/windows. More than three detections explicitly holds QC."""
    detection=pre_car_windows(values,rate);bad,sd,excursion=detect_high_channels(detection)
    metadata={'detected_channels':[COMMON_CHANNELS[i] for i in bad], 'detected_indices':bad,
              'pre_car_sd_uv':sd.tolist(),'pre_car_peak_to_peak_uv':excursion.tolist(),
              'repaired_channels':[],'interpolation_weight_max_discrepancy':0.,
              'repair_capacity_exceeded':len(bad)>3,'donors_unchanged':True,
              'trials_or_windows_dropped':0,'author_code_reproduction':False}
    if not bad or len(bad)>3:
        windows=notched_late_trial_windows(values,rate)
        metadata['decision']='unchanged' if not bad else 'held_beyond_repair_capacity'
    else:
        repaired,weights,discrepancy=interpolate_input(values,bad)
        windows=notched_late_trial_windows(repaired,rate)
        metadata.update({'decision':'fixed_spherical_interpolation_before_CAR',
            'repaired_channels':[COMMON_CHANNELS[i] for i in bad],
            'interpolation_weight_max_discrepancy':discrepancy,
            'interpolation_weights':weights.tolist()})
    return windows,metadata
