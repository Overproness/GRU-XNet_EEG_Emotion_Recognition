"""Matched, source-selected DEAP feature/context controls on reused cohorts."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .data import DEAP_CHANNELS, COMMON_CHANNELS, deap_reference, roots, verify_deap_labels
from .prepare import preprocess_signal
from .heldout_tuning import partitions
from .preprocessing_diagnostic_v2 import (SOURCES as PREVIOUS_SOURCES, STUDY as PARENT,
    atomic, sha, stamp, logpower, selection_key, check_cache)

REPO = Path(__file__).resolve().parents[1]
STUDY = 'deap_baseline_folds_2026-10-09'
REPRESENTATIONS = ('stimulus_absolute', 'stimulus_relative', 'baseline_only', 'baseline_relative')
MODELS = (*REPRESENTATIONS, *(f'{r}_context' for r in REPRESENTATIONS), 'context_logistic', 'prior')
C_VALUES = (.01, .1, 1., 10.)
SOURCE_ROLES = ('train', 'validation_unseen', 'validation_familiar')
ROLES = (*SOURCE_ROLES, 'test')
ARMS = ('exposed', 'unexposed')
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES, 'scripts/prestimulus_control.py',
    'gruxnet/deap_baseline_folds.py', 'scripts/deap_baseline_folds.py',
    'scripts/audit_deap_baseline_folds.py', 'scripts/analyze_deap_baseline_folds.py',
    'scripts/export_deap_baseline_folds.py')))


def cell_id(job):
    return f'deap_g{job["group"]}_r{job["rotation"]}_f{job["fold"]}_{job["arm"]}'


def full_job(job):
    return dict(job, model='baseline_relative', initialization=42)


def make_features(stimulus, baseline):
    if stimulus.shape != (32, 5120) or baseline.shape != (32, 384):
        raise ValueError('Expected measured native DEAP stimulus/baseline shapes')
    windows = stimulus.reshape(32, 10, 512).transpose(1, 0, 2)
    absolute = logpower(windows)
    rest = logpower(baseline)
    values = {'stimulus_absolute': absolute.mean(0),
        'stimulus_relative': (absolute-logsumexp(absolute, axis=-1, keepdims=True)).mean(0),
        'baseline_only': rest,
        # Subtract before window averaging, exactly as the inspected diagnostic.
        'baseline_relative': (absolute-rest[None]).mean(0)}
    return {name: value.reshape(-1).astype(np.float64) for name, value in values.items()}


def fast_prior(table, train, tested, crossfit=False):
    """Laplace1 video/global prior; exclude ALL recipient-person training labels."""
    source = table.iloc[train]
    global_counts = np.bincount(source.label.to_numpy(dtype=int), minlength=2)
    video_counts = {key: np.bincount(part.label.to_numpy(dtype=int), minlength=2)
                    for key, part in source.groupby('material_key', sort=True)}
    person_counts = {key: np.bincount(part.label.to_numpy(dtype=int), minlength=2)
                     for key, part in source.groupby('subject_id', sort=True)}
    person_video = {(person, video): np.bincount(part.label.to_numpy(dtype=int), minlength=2)
                    for (person, video), part in source.groupby(['subject_id', 'material_key'], sort=True)}
    result = []
    for row in table.iloc[tested].itertuples():
        all_counts = global_counts.copy()
        counts = video_counts.get(row.material_key, np.zeros(2, dtype=int)).copy()
        if crossfit:
            all_counts -= person_counts.get(row.subject_id, np.zeros(2, dtype=int))
            counts -= person_video.get((row.subject_id, row.material_key), np.zeros(2, dtype=int))
        if all_counts.sum() == 0 or np.any(counts < 0):
            raise ValueError('No valid other-participant source prior')
        chosen = counts if counts.sum() else all_counts
        chosen = chosen.astype(np.float64)+1
        result.append(chosen/chosen.sum())
    return np.asarray(result, dtype=np.float64)


def metric(target, probability):
    target = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != (len(target), 2) or set(target) != {0, 1}:
        raise ValueError('Missing binary class or malformed probabilities')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probability values')
    np.testing.assert_allclose(p.sum(1), 1., rtol=0, atol=1e-12)
    correct = p.argmax(1) == target
    loss = -np.log(np.clip(p[np.arange(len(p)), target], 1e-12, 1.))
    return {'balanced_accuracy': float(np.mean([correct[target == c].mean() for c in (0, 1)])),
        'balanced_log_loss': float(np.mean([loss[target == c].mean() for c in (0, 1)])),
        'accuracy': float(correct.mean()), 'n': len(target)}


def prediction_frame(table, indexes, probability, job, model, role, c=None):
    columns = ['trial_id', 'subject_id', 'material_key', 'label', 'original_label']
    result = table.iloc[indexes][columns].copy().reset_index(drop=True)
    result['group'] = job['group']; result['source_session'] = 1
    result['material_rotation'] = job['rotation']; result['fold'] = job['fold']
    result['arm'] = job['arm']; result['model'] = model; result['role'] = role
    if c is not None: result['C'] = c
    result['p0'] = probability[:, 0]; result['p1'] = probability[:, 1]
    return result


def inputs_for(features, table, idx, job, model, roles):
    """Only requested roles are retrieved; fitting never requests outer-test input."""
    result = {}
    for role in roles:
        rows = idx[role]
        columns = []
        if model in REPRESENTATIONS:
            columns.append(features[model][rows])
        elif model.endswith('_context'):
            columns.append(features[model[:-len('_context')]][rows])
        elif model != 'context_logistic':
            raise ValueError('Unknown fitted model')
        if model.endswith('_context') or model == 'context_logistic':
            q = fast_prior(table, idx['train'], rows, crossfit=role == 'train')
            columns.append(np.log(q))
        result[role] = np.concatenate(columns, axis=1)
    return result


def estimator(c):
    return LogisticRegression(C=c, class_weight='balanced', max_iter=4000, tol=1e-6, random_state=42)


def direct_probability(x, parameters, position):
    z = (x-parameters['mean'])/parameters['scale']
    q = expit(z @ parameters['coef'][position]+parameters['intercept'][position])
    return np.column_stack([1-q, q])


def prepare(output, root, data_root):
    """Fixed label-free per-recording transforms; no learned corpus-level scaler."""
    cache = output/'inputs'
    if (cache/'prepared.json').exists():
        return load_inputs(output)
    if cache.exists(): raise FileExistsError('Preserve incomplete preparation')
    base = pd.read_csv(root/'cache_full_context_deap/trials.csv')
    records = {r['trial_id']: r for r in json.loads((root/'cache_common14/lineage.json').read_text())
               if r['dataset'] == 'DEAP'}
    parent_cache = root/PARENT/'inputs/deap'
    check_cache(parent_cache)
    old = pd.read_csv(parent_cache/'trials.csv')
    positions = dict(zip(old.trial_id, old.index))
    native = np.load(parent_cache/'native.npy', mmap_mode='r', allow_pickle=False)
    baseline_cache = np.load(parent_cache/'baseline.npy', mmap_mode='r', allow_pickle=False)
    channels = list(DEAP_CHANNELS); picks = [channels.index(c) for c in COMMON_CHANNELS]
    source_root = roots(data_root)['DEAP']
    ratings, rating_info = deap_reference(source_root)
    arrays = {name: np.empty((len(base), 128), dtype=np.float64) for name in REPRESENTATIONS}
    checked = {}; overlap = 0
    for person, rows in base.groupby('subject_id', sort=True):
        first = records[rows.iloc[0].trial_id]; path = Path(first['source'])
        if sha(path) != first['source_sha256']: raise ValueError('Changed mirror source')
        checked[path.name] = first['source_sha256']
        with path.open('rb') as stream: values = pickle.load(stream, encoding='latin1')
        if values['data'].shape != (40, 40, 8064): raise ValueError('Wrong original shape')
        reference = ratings[int(person.rsplit('S', 1)[1])]
        verify_deap_labels(values['labels'], reference)
        for i, row in rows.iterrows():
            trial = int(row.trial_id.rsplit('T', 1)[1])-1
            if row.original_label != reference[trial, 0] or row.label != int(row.original_label > 5):
                raise ValueError('Changed corrected label')
            raw = values['data'][trial, :32]
            stimulus, _ = preprocess_signal(raw[:, 384:], 128, channels, channels)
            stimulus = stimulus[:, :5120]
            rest, _ = preprocess_signal(raw[:, :384], 128, channels, channels)
            record = records[row.trial_id]; common_path = root/'cache_common14'/record['cache_file']
            if sha(common_path) != record['cache_sha256']: raise ValueError('Changed common prefix')
            common = np.load(common_path, allow_pickle=False)[:10].transpose(1, 0, 2).reshape(14, 5120)
            np.testing.assert_array_equal(stimulus[picks], common)
            feature = make_features(stimulus, rest)
            if row.trial_id in positions:
                previous = positions[row.trial_id]
                np.testing.assert_array_equal(stimulus, native[previous])
                np.testing.assert_array_equal(rest, baseline_cache[previous])
                expected = make_features(native[previous], baseline_cache[previous])
                for name in REPRESENTATIONS: np.testing.assert_array_equal(feature[name], expected[name])
                overlap += 1
            for name in REPRESENTATIONS: arrays[name][i] = feature[name]
        print(json.dumps({'prepared_person': person, 'trials': len(rows)}), flush=True)
        del values
    if len(base) != 1264 or overlap != 740 or base.original_label.eq(5).any():
        raise ValueError('Unexpected complete/overlap populations')
    if not all(np.isfinite(x).all() for x in arrays.values()): raise ValueError('Nonfinite feature')
    cache.mkdir()
    base.to_csv(cache/'trials.csv', index=False)
    for name, values in arrays.items(): np.save(cache/f'{name}.npy', values, allow_pickle=False)
    files = ['trials.csv', *(f'{name}.npy' for name in REPRESENTATIONS)]
    atomic(cache/'prepared.json', {'plan_sha256': sha(output/'plan.json'), 'source_files': checked,
        'file_sha256': {name: sha(cache/name) for name in files}, 'channels': channels,
        'shape': [1264, 128], 'common_prefixes_exact': 1264, 'native_baseline_overlap_exact': overlap,
        'metadata_sha256': rating_info['sha256'] if 'sha256' in rating_info else sha(Path(rating_info['file'])),
        'first_party_DEAP_recording_authentication': False,
        'scope': 'Per-trial fixed label-free features of all original retained trials. Source/test separation and training-only learned scalers apply independently in each fit; this cache prepares test features before selection but never selects from them.'})
    return base, arrays


def load_inputs(output):
    cache = output/'inputs'; info = json.loads((cache/'prepared.json').read_text())
    if info['plan_sha256'] != sha(output/'plan.json'): raise ValueError('Changed input declaration')
    for name, checksum in info['file_sha256'].items():
        if sha(cache/name) != checksum: raise ValueError('Changed feature input')
    return pd.read_csv(cache/'trials.csv'), {name: np.load(cache/f'{name}.npy', allow_pickle=False)
                                          for name in REPRESENTATIONS}


def fit_model(output, base, features, job, model):
    folder = output/'cells'/cell_id(job)/model
    if (folder/'verification.json').exists(): return
    folder.mkdir(parents=True, exist_ok=True)
    table, idx = partitions(base, full_job(job))
    config_sha = sha(output/'config.json')
    if not (folder/'selection.json').exists():
        if list(folder.iterdir()): raise FileExistsError('Preserve incomplete unsealed fit')
        inputs = inputs_for(features, table, idx, job, model, SOURCE_ROLES)
        scaler = StandardScaler().fit(inputs['train'])
        z = {r: scaler.transform(inputs[r]) for r in SOURCE_ROLES}
        fitted = []; candidates = []; frames = []
        for c in C_VALUES:
            classifier = estimator(c).fit(z['train'], table.iloc[idx['train']].label)
            if np.any(classifier.n_iter_ >= 4000): raise ValueError('Logistic budget exhausted')
            np.testing.assert_array_equal(classifier.classes_, [0, 1])
            probabilities = {r: classifier.predict_proba(z[r]) for r in SOURCE_ROLES}
            metrics = {r: metric(table.iloc[idx[r]].label, probabilities[r]) for r in SOURCE_ROLES}
            frames.extend(prediction_frame(table, idx[r], probabilities[r], job, model, r, c) for r in SOURCE_ROLES)
            candidates.append({'id': str(c), 'C': c, 'metrics': metrics, 'iterations': int(classifier.n_iter_[0])})
            fitted.append(classifier)
        pd.concat(frames, ignore_index=True).to_csv(folder/'candidates_source.csv', index=False)
        np.savez(folder/'parameters.npz', mean=scaler.mean_, scale=scaler.scale_,
            coef=np.stack([m.coef_[0] for m in fitted]), intercept=np.array([m.intercept_[0] for m in fitted]))
        chosen = min(candidates, key=selection_key)
        # This immutable selection exists before retrieving ANY test features/prior.
        atomic(folder/'selection.json', {'job': job, 'model': model, 'config_sha256': config_sha,
            'selected': chosen['id'], 'candidates': candidates, 'created_utc': stamp(),
            'split_trials': {r: table.iloc[idx[r]].trial_id.tolist() for r in ROLES},
            'parameters_sha256': sha(folder/'parameters.npz'),
            'candidate_predictions_sha256': sha(folder/'candidates_source.csv')})
    selection = json.loads((folder/'selection.json').read_text())
    if selection['job'] != job or selection['model'] != model or selection['config_sha256'] != config_sha:
        raise ValueError('Changed sealed model')
    if selection['parameters_sha256'] != sha(folder/'parameters.npz'):
        raise ValueError('Changed sealed coefficients')
    if selection['candidate_predictions_sha256'] != sha(folder/'candidates_source.csv'):
        raise ValueError('Changed sealed source outcomes')
    inputs = inputs_for(features, table, idx, job, model, ROLES)
    parameters = np.load(folder/'parameters.npz', allow_pickle=False)
    position = next(i for i, c in enumerate(selection['candidates']) if c['id'] == selection['selected'])
    frames = []; metrics = {}
    for role in ROLES:
        p = direct_probability(inputs[role], parameters, position)
        metrics[role] = metric(table.iloc[idx[role]].label, p)
        frames.append(prediction_frame(table, idx[role], p, job, model, role))
    pd.concat(frames, ignore_index=True).to_csv(folder/'selected_predictions.csv', index=False)
    atomic(folder/'record.json', {'job': job, 'model': model, 'selected': selection['selected'],
        'selection_sha256': sha(folder/'selection.json'), 'metrics': metrics,
        'artifact_sha256': {'selected_predictions.csv': sha(folder/'selected_predictions.csv'),
                           'candidates_source.csv': sha(folder/'candidates_source.csv')}})


def fit_prior(output, base, job):
    folder = output/'cells'/cell_id(job)/'prior'
    if (folder/'verification.json').exists(): return
    if folder.exists(): raise FileExistsError('Preserve incomplete prior control')
    folder.mkdir(parents=True)
    table, idx = partitions(base, full_job(job))
    atomic(folder/'selection.json', {'job': job, 'model': 'prior', 'config_sha256': sha(output/'config.json'),
        'created_utc': stamp(), 'selected': 'fixed_Laplace1', 'split_trials': {r: table.iloc[idx[r]].trial_id.tolist() for r in ROLES}})
    frames = []; metrics = {}
    for role in ROLES:
        p = fast_prior(table, idx['train'], idx[role], crossfit=role == 'train')
        metrics[role] = metric(table.iloc[idx[role]].label, p)
        frames.append(prediction_frame(table, idx[role], p, job, 'prior', role))
    pd.concat(frames, ignore_index=True).to_csv(folder/'selected_predictions.csv', index=False)
    atomic(folder/'record.json', {'job': job, 'model': 'prior', 'selected': 'fixed_Laplace1',
        'selection_sha256': sha(folder/'selection.json'), 'metrics': metrics,
        'artifact_sha256': {'selected_predictions.csv': sha(folder/'selected_predictions.csv')}})
