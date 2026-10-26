"""Fixed FACED covariance/readout diagnostics; no original-data or target I/O."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.linalg import solve
from scipy.signal import get_window
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from .faced_classical_contract import BANDS

SHRINKAGE = 0.10
FEATURE_WIDTH = 2330
NORMAL_KERNEL_ALPHA = 0.1
CAPACITY_KERNEL_ALPHA = 1e-6
CONDITIONS = {
    'covariance_linear': ('linear', ('covariance',), 1000.0),
    'context_covariance_linear': ('linear', ('context', 'covariance'), 100.0),
    'context_kernel': ('rbf', ('context',), NORMAL_KERNEL_ALPHA),
    'spectral_kernel': ('rbf', ('spectral',), NORMAL_KERNEL_ALPHA),
    'context_spectral_kernel': ('rbf', ('context', 'spectral'), NORMAL_KERNEL_ALPHA),
    'covariance_kernel': ('rbf', ('covariance',), NORMAL_KERNEL_ALPHA),
    'context_covariance_kernel': ('rbf', ('context', 'covariance'), NORMAL_KERNEL_ALPHA),
}


def band_covariances(windows):
    """Real cross spectra integrated in the unchanged five Welch bands.

    Equal average of seven Hann/detrended windows, one matrix per band/trial.
    Diagonals equal arithmetic-mean integrated Welch power, not mean log power.
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.shape != (7, 30, 512) or not np.isfinite(x).all():
        raise ValueError('Require seven finite, unchanged common-channel windows')
    taper = get_window('hann', 512, fftbins=True)
    spectrum = np.fft.rfft((x-x.mean(axis=-1, keepdims=True))*taper, axis=-1)
    frequencies = np.fft.rfftfreq(512, 1/128)
    matrices = []
    for _, lo, hi in BANDS:
        selected = spectrum[..., (frequencies >= lo) & (frequencies < hi)]
        # Every selected bin is strictly between DC and Nyquist.
        matrix = np.einsum('wif,wjf->ij', selected, selected.conj()).real
        matrix *= 2*0.25/(7*128*np.sum(taper*taper))
        matrices.append((matrix+matrix.T)/2)
    result = np.stack(matrices)
    if result.shape != (5, 30, 30) or not np.isfinite(result).all():
        raise ValueError('Invalid band covariance')
    return result


def covariance_features(windows):
    matrices = band_covariances(windows)
    vectors = []
    traces = []
    minimum_eigenvalues = []
    upper = np.triu_indices(30)
    weights = np.where(upper[0] == upper[1], 1.0, np.sqrt(2.0))
    for matrix in matrices:
        trace = float(np.trace(matrix))
        if not np.isfinite(trace) or trace <= 0:
            raise ValueError('Nonpositive covariance power holds representation preparation')
        normalized = (1-SHRINKAGE)*matrix/trace + SHRINKAGE*np.eye(30)/30
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
            raise ValueError('Covariance is not positive definite')
        logged = (eigenvectors*np.log(eigenvalues))@eigenvectors.T
        vectors.append(logged[upper]*weights)
        traces.append(trace)
        minimum_eigenvalues.append(float(eigenvalues.min()))
    result = np.concatenate(vectors+[np.log10(traces)])
    if result.shape != (FEATURE_WIDTH,) or not np.isfinite(result).all():
        raise ValueError('Invalid log-covariance representation')
    return result, {'band_power_trace_uv2': traces,
                    'minimum_shrunken_eigenvalues': minimum_eigenvalues,
                    'shrinkage': SHRINKAGE, 'feature_width': FEATURE_WIDTH}


def exact_pairs(requested, expected):
    pairs = list(requested)
    intended = list(expected)
    if not pairs or len(pairs) != len(set(pairs)) or len(intended) != len(set(intended)) or set(pairs) != set(intended):
        raise PermissionError('Requested identities differ from the declared set')
    return pairs


def source_matrix(pairs, features, blocks):
    if tuple(blocks) not in (('context',), ('spectral',), ('covariance',),
                            ('context', 'spectral'), ('context', 'covariance')):
        raise ValueError('Undeclared representation blocks')
    widths = {'context': 13, 'spectral': 150, 'covariance': FEATURE_WIDTH}
    arrays = [np.stack([features[b][p] for p in pairs]).astype(np.float64) for b in blocks]
    if any(a.shape != (len(pairs), widths[b]) or not np.isfinite(a).all() for a, b in zip(arrays, blocks)):
        raise ValueError('Invalid declared feature; no imputation or row removal')
    return np.concatenate(arrays, axis=1), [widths[b] for b in blocks]


def fit_readout(requested_pairs, expected_pairs, features, targets, kind, blocks, alpha):
    """Identity guard precedes feature/target reads; every fit is source only."""
    pairs = exact_pairs(requested_pairs, expected_pairs)
    allowed = {'linear': (100.0, 1000.0), 'rbf': (NORMAL_KERNEL_ALPHA, CAPACITY_KERNEL_ALPHA)}
    if kind not in allowed or alpha not in allowed[kind]:
        raise ValueError('Undeclared readout/regularization')
    x, widths = source_matrix(pairs, features, blocks)
    y = np.asarray([targets[p] for p in pairs], dtype=float)
    if y.shape != (len(pairs),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 7)):
        raise ValueError('Invalid source target')
    scaler = StandardScaler().fit(x)
    block_weights = np.concatenate([np.full(w, 1/np.sqrt(w) if kind == 'rbf' else 1.0) for w in widths])
    z = scaler.transform(x)*block_weights
    if kind == 'rbf':
        distances = pdist(z, metric='sqeuclidean')
        positive = distances[distances > 0]
        if not positive.size:
            raise ValueError('No distinct source inputs; hold kernel fitting')
        median_squared_distance = float(np.median(positive))
        gamma = 1/median_squared_distance
        kernel = np.exp(-gamma*cdist(z, z, metric='sqeuclidean'))
    else:
        gamma = None
        median_squared_distance = None
        kernel = z@z.T
    target_mean = float(y.mean())
    system = kernel + alpha*np.eye(len(pairs))
    dual = solve(system, y-target_mean, assume_a='pos')
    residual = float(np.max(np.abs(system@dual-(y-target_mean))))
    if not np.isfinite(dual).all() or residual > 1e-6:
        raise ValueError('Readout equation check fails')
    raw = target_mean+kernel@dual
    return {'source_pairs': pairs, 'source_identity_sha256': hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest(),
            'kind': kind, 'blocks': tuple(blocks), 'alpha': float(alpha), 'scaler': scaler,
            'block_weights': block_weights, 'source_z': z, 'gamma': gamma,
            'median_source_squared_distance': median_squared_distance, 'target_mean': target_mean,
            'dual': dual, 'source_y': y, 'normal_equation_residual': residual,
            'training_mae': float(np.mean(np.abs(y-np.clip(raw, 0, 7)))),
            'training_raw_mae': float(np.mean(np.abs(y-raw)))}


def predict_readout(fitted, requested_pairs, expected_pairs, features):
    pairs = exact_pairs(requested_pairs, expected_pairs)
    x, _ = source_matrix(pairs, features, fitted['blocks'])
    z = fitted['scaler'].transform(x)*fitted['block_weights']
    if fitted['kind'] == 'rbf':
        kernel = np.exp(-fitted['gamma']*cdist(z, fitted['source_z'], metric='sqeuclidean'))
    else:
        kernel = z@fitted['source_z'].T
    prediction = np.clip(fitted['target_mean']+kernel@fitted['dual'], 0, 7)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    return prediction
