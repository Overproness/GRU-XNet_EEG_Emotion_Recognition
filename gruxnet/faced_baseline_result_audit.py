"""Independent result algebra; no fitting or filesystem access."""
from __future__ import annotations
import numpy as np


def checked_panel(rows, expected, families):
    """Validate the entire declared identity rectangle before outcome conversion."""
    keys = []
    required = {'fit','epoch','arm','subject','vid','family','target','prediction_raw'}
    for row in rows:
        if set(row) != required:
            raise PermissionError('Unexpected prediction schema')
        key = (row['fit'], int(row['epoch']), row['arm'], row['subject'], int(row['vid']))
        if key not in expected or row['family'] != families[key[3:]]:
            raise PermissionError('Undeclared fit/checkpoint/arm/person/material')
        keys.append(key)
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise PermissionError('Incomplete or duplicate prediction rectangle')
    result = {}
    for key,row in zip(keys,rows):
        y,p = float(row['target']), float(row['prediction_raw'])
        if not np.isfinite([y,p]).all() or not 0 <= y <= 7:
            raise ValueError('Invalid finite original-scale outcome/prediction')
        result[key] = (y,p)
    return result


def audit_ridge(x, y, mean, scale, coefficient, intercept, alpha=100.):
    """Reconstruct scaler moments and training equations without solving/refitting."""
    x,y = np.asarray(x,float), np.asarray(y,float)
    mean,scale,coefficient = [np.asarray(v,float) for v in (mean,scale,coefficient)]
    if (x.ndim != 2 or y.shape != (len(x),) or any(v.shape != (x.shape[1],) for v in (mean,scale,coefficient))
            or not all(np.isfinite(v).all() for v in (x,y,mean,scale,coefficient)) or np.any(scale <= 0)):
        raise ValueError('Invalid finite ridge geometry')
    expected_mean = x.mean(axis=0)
    expected_scale = np.sqrt(x.var(axis=0))
    expected_scale[expected_scale == 0] = 1.
    if not np.allclose(mean,expected_mean,atol=1e-11,rtol=1e-11) or not np.allclose(scale,expected_scale,atol=1e-11,rtol=1e-11):
        raise ValueError('Source scaler moments differ')
    z = (x-mean)/scale
    centered = z-z.mean(axis=0)
    rhs = centered.T@(y-y.mean())
    lhs = centered.T@(centered@coefficient)+alpha*coefficient
    residual = float(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(rhs),1.))
    if residual > 1e-10 or abs(float(intercept)-(y.mean()-z.mean(axis=0)@coefficient)) > 1e-10:
        raise ValueError('Ridge training equations/intercept differ')
    return residual


def independent_flags(baseline, middle, final, maes):
    result = []
    if any(x > .8*baseline for x in final): result.append('insufficient_source_learning')
    if any(b < .9*a for a,b in zip(middle,final)): result.append('budget_limited_convergence')
    if abs(maes[0]-maes[1]) > .15: result.append('held_seed_divergence')
    if max(final) > 2*max(min(final),1e-8): result.append('source_seed_divergence')
    return result
