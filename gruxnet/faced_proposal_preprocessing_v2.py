"""Source-documented late-viewing supplement to the synthetic proposal adapter.

Only computes a proposed interval; contains no raw-file reader or rating access.
The previous early-viewing candidate remains preserved for review.
"""
from math import ceil, floor
from .faced_proposal_preprocessing import (
    ALIASES, COMMON_CHANNELS, UNIT_REPAIR_SUBJECTS, calibrate_common_counts,
    common_indices, trial_windows,
)


def late_segment_bounds(start_seconds,end_seconds,sampling_rate,recording_samples):
    if sampling_rate not in (250,1000):
        raise ValueError('Unsupported original rate')
    stop=floor(end_seconds*sampling_rate)
    start=stop-32*sampling_rate
    if start<ceil(start_seconds*sampling_rate) or stop>recording_samples or start<0:
        raise ValueError('Late segment exceeds its own original video span')
    return start,stop


def late_trial_windows(microvolts,sampling_rate):
    """Input segment end-32..end; output core approximately end-30..end-2."""
    return trial_windows(microvolts,sampling_rate)
