"""Independent head refits, weighted metrics, raw sentinels and encoder replay."""
from pathlib import Path
import json
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_probe import (ASSETS, ROLES, STUDY, MODELS, C_VALUES,
    local_model, encode, preprocess, check_cache, check_features, load_panel,
    validate, roots, sha, atomic, stamp)
from gruxnet.material_controls import state_digest


def independent_metric(y, p):
    weights = compute_sample_weight('balanced', y)
    # Explicit same declared floor; sklearn otherwise clips to floating epsilon.
    loss = np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)), weights=weights)
    return {'n': len(y), 'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(1))),
            'balanced_log_loss': float(loss)}


def audit_head(root, output, job):
    identifier = f'{job["dataset"].lower()}_g{job["group"]}_{job["model"]}'
    folder = output/'fits'/identifier
    record = json.loads((folder/'record.json').read_text())
    if record['plan_sha256'] != sha(output/'plan.json') or record['job'] != job:
        raise ValueError('Wrong fit plan/job binding')
    cache = output/'inputs'/job['dataset'].lower()
    if record['feature_sha256'] != sha(cache/f'{job["model"]}.npy'):
        raise ValueError('Changed fit features')
    for name, checksum in record['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Changed saved fit artifact')
    table, idx, x = load_panel(root, output, **job)
    y = table.label.to_numpy(dtype=int)
    scored = []; maximum = 0.; probability_maximum = 0.
    for k, C in enumerate(C_VALUES):
        item = record['candidates'][k]
        if item['id'] != k or item['C'] != C or item['parameters_sha256'] != sha(folder/f'candidate{k}.npz'):
            raise ValueError('Candidate order/binding mismatch')
        saved = np.load(folder/f'candidate{k}.npz', allow_pickle=False)
        scaler = StandardScaler().fit(x[idx['train']])
        z = scaler.transform(x)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model = LogisticRegression(C=C, class_weight='balanced', solver='lbfgs',
                max_iter=4000, tol=1e-6, random_state=42).fit(z[idx['train']], y[idx['train']])
        for name, value in (('mean', scaler.mean_), ('scale', scaler.scale_), ('coef', model.coef_),
                            ('intercept', model.intercept_), ('classes', model.classes_)):
            np.testing.assert_array_equal(saved[name], value)
        measures = {}
        for role, rows in idx.items():
            # Reproduce estimator's FP32 scaler output before FP64 linear link.
            logits = z[rows] @ saved['coef'].T + saved['intercept']
            p = (np.column_stack((1-expit(logits[:, 0]), expit(logits[:, 0])))
                 if len(saved['classes']) == 2 else softmax(logits, axis=1))
            np.testing.assert_allclose(p, saved[role], atol=2e-12, rtol=0)
            probability_maximum = max(probability_maximum, float(np.max(np.abs(p-saved[role]))))
            measures[role] = independent_metric(y[rows], p)
            for name in ('balanced_accuracy', 'balanced_log_loss'):
                delta = abs(measures[role][name]-item['metrics'][role][name])
                maximum = max(maximum, delta)
                if delta > 2e-11:
                    raise ValueError('Independent metric mismatch')
            if k == record['selected_id']:
                csv = pd.read_csv(folder/f'{role}.csv')
                pd.testing.assert_frame_equal(csv[['trial_id', 'subject_id', 'material_key', 'label']],
                    table.iloc[rows][['trial_id', 'subject_id', 'material_key', 'label']].reset_index(drop=True))
                np.testing.assert_allclose(csv[[f'p{c}' for c in range(p.shape[1])]], p, atol=2e-12, rtol=0)
        primary = np.mean([measures[r]['balanced_log_loss'] for r in ROLES[1:]])
        secondary = -np.mean([measures[r]['balanced_accuracy'] for r in ROLES[1:]])
        scored.append((primary, secondary, k))
    if min(scored)[2] != record['selected_id'] or record['selected_C'] != C_VALUES[record['selected_id']]:
        raise ValueError('Independent selection mismatch')
    result = {'complete': True, 'record_sha256': sha(folder/'record.json'),
        'candidate_refits': 4, 'metric_sets': 12, 'max_metric_abs': maximum,
        'max_probability_link_abs': probability_maximum, 'created_utc': stamp()}
    atomic(folder/'verification.json', result)
    return result


def audit_inputs(root, output, data_root):
    locations = roots(data_root)
    lineage = {r['trial_id']: r for r in json.loads((root/'cache_common14/lineage.json').read_text())}
    records = []
    for dataset in ('DEAP', 'SEEDIV'):
        cache = output/'inputs'/dataset.lower()
        check_cache(cache); features = check_features(cache)
        table = pd.read_csv(cache/'trials.csv')
        indexes = [0, 1, len(table)-2, len(table)-1]
        raw = np.load(cache/'native200.npy', mmap_mode='r', allow_pickle=False)
        for i in indexes:
            row = table.iloc[i]
            if dataset == 'DEAP':
                source = lineage[row.trial_id]
                path = Path(source['source'])
                if sha(path) != source['source_sha256']:
                    raise ValueError('Changed raw sentinel')
                with path.open('rb') as stream:
                    values = pickle.load(stream, encoding='latin1')
                trial = int(row.trial_id.rsplit('T', 1)[1])-1
                expected = preprocess(values['data'][trial, :32, 384:], dataset)
            else:
                path = locations[dataset]/row.source_file
                if sha(path) != row.source_sha256:
                    raise ValueError('Changed raw sentinel')
                expected = preprocess(loadmat(path, variable_names=[row.source_key])[row.source_key], dataset)
            np.testing.assert_array_equal(expected, raw[i])
        for name, pretrained in (('pretrained', True), ('random42', False)):
            model = local_model(root/ASSETS, pretrained).to('cuda')
            state = state_digest(model.state_dict())
            if state != features['encoder_states'][name]['state_sha256']:
                raise ValueError('Wrong replay encoder state')
            replay = encode(model, np.array(raw[indexes]), batch=1)
            for readout in ('average', 'flatten'):
                original = np.load(cache/f'{name}_{readout}.npy', mmap_mode='r', allow_pickle=False)[indexes]
                delta = float(np.max(np.abs(replay[readout]-original)))
                np.testing.assert_allclose(replay[readout], original, atol=2e-5, rtol=0)
                records.append({'dataset': dataset, 'encoder': name, 'readout': readout,
                                'raw_sentinels': 4, 'batch1_replay_trials': 4, 'max_abs': delta})
            if state_digest(model.state_dict()) != state:
                raise ValueError('Encoder changed during replay')
            del model
    return records


def finish(root, output, data_root):
    plan = validate(root, output)
    jobs = plan['jobs']
    audits = []
    for job in jobs:
        folder = output/'fits'/f'{job["dataset"].lower()}_g{job["group"]}_{job["model"]}'
        verification = json.loads((folder/'verification.json').read_text())
        if not verification['complete'] or verification['record_sha256'] != sha(folder/'record.json'):
            raise ValueError('Missing/stale verification')
        audits.append(verification)
    input_replay = audit_inputs(root, output, data_root)
    rows = []
    for job in jobs:
        folder = output/'fits'/f'{job["dataset"].lower()}_g{job["group"]}_{job["model"]}'
        record = json.loads((folder/'record.json').read_text())
        rows.append({**job, 'selected_C': record['selected_C'], 'metrics': record['metrics']})
    atomic(output/'summary.json', {'development_only': True, 'outer_test_inferences': 0, 'rows': rows})
    proof = {'complete': True, 'heads': len(jobs), 'fresh_candidate_refits': sum(a['candidate_refits'] for a in audits),
             'independent_metric_sets': sum(a['metric_sets'] for a in audits),
             'max_metric_abs': max(a['max_metric_abs'] for a in audits),
             'max_probability_link_abs': max(a['max_probability_link_abs'] for a in audits),
             'raw_and_encoder_replay': input_replay, 'plan_sha256': sha(output/'plan.json'),
             'summary_sha256': sha(output/'summary.json'), 'created_utc': stamp()}
    atomic(output/'verification.json', proof)
    return proof
