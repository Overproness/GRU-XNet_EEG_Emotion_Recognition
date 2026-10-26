"""Source-only linear reference decomposition; never repairs an observed trial."""
import numpy as np
from .faced_source_notch import notched_late_trial_windows


def reference_decomposition(values, rate, dominant_channel):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (30,32*rate) or not np.isfinite(x).all() or not 0 <= dominant_channel < 30:
        raise ValueError('Require one declared finite scalp interval and channel')
    observed = notched_late_trial_windows(x, rate)
    isolated = np.zeros_like(x); isolated[dominant_channel] = x[dominant_channel]
    contribution = notched_late_trial_windows(isolated, rate)
    remainder = notched_late_trial_windows(x-isolated, rate)
    discrepancy = float(np.max(np.abs(observed.astype(float)-contribution.astype(float)-remainder.astype(float))))
    # Output is float32; the linear filters themselves operate in float64.
    scale = max(float(np.max(np.abs(observed))), float(np.max(np.abs(contribution))), 1.)
    if discrepancy > 5e-6*scale: raise ValueError('Reference decomposition failed linearity check')
    other = [i for i in range(30) if i != dominant_channel]
    metrics = {'window_median_other_channel_sd_uv': {}, 'core_median_other_channel_sd_uv': {}}
    for name, array in [('observed',observed), ('dominant_channel_contribution',contribution), ('remainder',remainder)]:
        metrics['window_median_other_channel_sd_uv'][name] = np.median(array[:,other].astype(float).std(axis=-1),axis=1).tolist()
        core = array.transpose(1,0,2).reshape(30,-1).astype(float)
        metrics['core_median_other_channel_sd_uv'][name] = float(np.median(core[other].std(axis=-1)))
    metrics['maximum_signal_linearity_discrepancy_uv'] = discrepancy
    metrics['interpretation'] = 'Algebraic channel contribution through the existing reference/filter, not validated artifact removal. Setting a channel to zero is a diagnostic counterfactual and is never used for fitting or adopted preprocessing.'
    return observed, metrics
