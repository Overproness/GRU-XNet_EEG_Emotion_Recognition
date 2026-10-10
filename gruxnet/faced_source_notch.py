"""One fixed source-quality follow-up; not a silently revised primary pipeline."""
import numpy as np
from scipy.signal import filtfilt, iirnotch
from .faced_proposal_preprocessing_v2 import late_trial_windows


def notched_late_trial_windows(microvolts, sampling_rate):
    values=np.asarray(microvolts,dtype=np.float64)
    if sampling_rate not in (250,1000) or values.shape!=(30,32*sampling_rate) or not np.isfinite(values).all():
        raise ValueError('Expected finite declared 32-second scalp interval')
    referenced=values-values.mean(axis=0,keepdims=True)
    b,a=iirnotch(50,30,fs=sampling_rate)
    cleaned=filtfilt(b,a,referenced,axis=-1,padtype='odd')
    # The unchanged base adapter also enforces its own common-average invariant.
    return late_trial_windows(cleaned,sampling_rate)
