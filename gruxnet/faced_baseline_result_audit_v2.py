"""Finite-scalar amendment; original unexecuted audit algebra is preserved."""
import numpy as np
from .faced_baseline_result_audit import checked_panel,independent_flags
from .faced_baseline_result_audit import audit_ridge as original_audit_ridge

def audit_ridge(x,y,mean,scale,coefficient,intercept,alpha=100.):
    for value in (intercept,alpha):
        if isinstance(value,(bool,str)) or np.asarray(value).ndim!=0 or not np.isfinite(value):
            raise ValueError('Finite numeric scalar intercept/alpha required')
    if alpha<=0: raise ValueError('Positive ridge penalty required')
    return original_audit_ridge(x,y,mean,scale,coefficient,intercept,alpha)
