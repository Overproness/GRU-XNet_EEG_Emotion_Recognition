"""Replay new source-only candidates and independently refit classical heads."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.preprocessing_diagnostic import (ROLES, atomic, sha, load_panel,
    transform, windows, predict, selection_key, classical_features)
from gruxnet.eegnet_control import EEGNetControl
from gruxnet.material_controls import state_digest
from gruxnet.learning_controls import draw_stream
from gruxnet.data import digest
from gruxnet.train import seed_everything
from scripts.source_bn_diagnostic import calibrate


def independent(rows):
    columns = sorted([c for c in rows if c.startswith('p') and c[1:].isdigit()], key=lambda c: int(c[1:]))
    p = rows[columns].to_numpy(dtype=float)
    y = rows.label.to_numpy(dtype=int)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-6):
        raise ValueError('Invalid public probabilities')
    loss = np.array([-np.log(max(p[i, label], 1e-12)) for i,label in enumerate(y)])
    return {'n': len(rows), 'balanced_accuracy': float(balanced_accuracy_score(y, np.argmax(p, axis=1))),
            'balanced_log_loss': float(np.mean([loss[y == c].mean() for c in range(p.shape[1])]))}


def compare(actual, expected):
    if actual['n'] != expected['n']:
        raise ValueError('Changed population')
    for field in ('balanced_accuracy', 'balanced_log_loss'):
        np.testing.assert_allclose(actual[field], expected[field], atol=1e-6, rtol=0)


def saved_rows(folder, label, table, idx):
    result = {}
    for role in ROLES:
        rows = pd.read_csv(folder/f'{label}_{role}.csv')
        metadata = ['trial_id', 'subject_id', 'material_key', 'label']
        pd.testing.assert_frame_equal(rows[metadata], table.iloc[idx[role]][metadata].reset_index(drop=True))
        result[role] = rows
    return result


def replay_neural(output, folder):
    record = json.loads((folder/'record.json').read_text())
    if record['plan_sha256'] != sha(output/'plan.json'):
        raise ValueError('Wrong plan binding')
    for name, expected in record['artifact_sha256'].items():
        if sha(folder/name) != expected:
            raise ValueError('Changed neural artifact')
    job = record['job']
    table, idx, raw, baseline, info = load_panel(output, job['dataset'], job['group'])
    if record['input_cache_sha256'] != sha(output/'inputs'/job['dataset'].lower()/'prepared.json'):
        raise ValueError('Wrong input binding')
    values, mean, scale = transform(raw, baseline, info['channels'], job['recipe'], idx['train'])
    normalizer = np.load(folder/'normalizer.npz', allow_pickle=False)
    np.testing.assert_array_equal(mean, normalizer['mean'])
    np.testing.assert_array_equal(scale, normalizer['scale'])
    short = job['recipe']['seconds'] == 4
    data = torch.from_numpy(np.ascontiguousarray(windows(values) if short else values)).cuda()
    classes = 2 if job['dataset'] == 'DEAP' else 3
    draws, signature = draw_stream(table, idx['train'], job['initialization'], classes, 600)
    windows_signature = digest(np.random.default_rng(job['initialization']+2000000).integers(0, 10, draws.shape).tolist())
    if signature != record['training_draw_digest'] or windows_signature != record['window_draw_digest']:
        raise ValueError('Changed training draws')
    seed_everything(job['initialization'])
    model = EEGNetControl(classes, channels=values.shape[1], samples=job['recipe']['seconds']*128).cuda()
    if state_digest(model.state_dict()) != record['initial_state_digest']:
        raise ValueError('Changed initialization')
    history = json.loads((folder/'history.json').read_text())
    for item in history:
        for role, rows in saved_rows(folder, f'curve{item["step"]}', table, idx).items():
            compare(independent(rows), item['metrics'][role])
    if record['selected'] != min(record['candidates'], key=selection_key)['id']:
        raise ValueError('Wrong source selection')
    error = 0.
    for candidate in record['candidates']:
        state = torch.load(folder/f'{candidate["id"]}.pt', map_location='cpu', weights_only=True)['state']
        if state_digest(state) != candidate['state_digest']:
            raise ValueError('Changed candidate state')
        model.load_state_dict(state)
        if candidate['normalization'] == 'source_population':
            ema = torch.load(folder/f'step{candidate["step"]}_ema.pt', map_location='cpu', weights_only=True)['state']
            model.load_state_dict(ema)
            training = data[idx['train']]
            calibrate(model, training.flatten(0, 1) if short else training)
            for name, expected in state.items():
                if not torch.equal(model.state_dict()[name].cpu(), expected):
                    raise ValueError('Source-only population BN reconstruction failed')
        for role, rows in saved_rows(folder, candidate['id'], table, idx).items():
            compare(independent(rows), candidate['metrics'][role])
            p = predict(model, data, idx[role], short)
            expected = rows[[f'p{c}' for c in range(classes)]].to_numpy()
            error = max(error, float(np.max(np.abs(p-expected))))
            np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
    del model, data
    torch.cuda.empty_cache()
    result = {'passed': True, 'record_sha256': sha(folder/'record.json'),
              'candidate_states_replayed': 4, 'curve_metric_sets_recomputed': len(history)*3,
              'maximum_probability_error': error, 'normalizer_exact': True,
              'draws_initialization_and_selection_checked': True,
              'source_population_states_reconstructed_exact': True,
              'scope': 'Fresh-model replay of all candidate states and exact source-population reconstruction using the previously audited calibrator, not a full optimization rerun or a different independent BN algorithm.'}
    atomic(folder/'verification.json', result)
    return result


def replay_classical(output, folder):
    from gruxnet.full_context_controls_v2 import prior
    record = json.loads((folder/'record.json').read_text())
    if record['plan_sha256'] != sha(output/'plan.json'):
        raise ValueError('Wrong plan binding')
    for name, expected in record['artifact_sha256'].items():
        if sha(folder/name) != expected:
            raise ValueError('Changed classical artifact')
    table, idx, raw, baseline, info = load_panel(output, record['dataset'], record['group'])
    classes = 2 if record['dataset'] == 'DEAP' else 3
    error, checked = 0., 0
    for recipe in record['recipes']:
        x = classical_features(raw, baseline, info['channels'], recipe['montage'], recipe['representation'])
        scaler = StandardScaler().fit(x[idx['train']])
        z = scaler.transform(x)
        if recipe['selected'] != min(recipe['candidates'], key=selection_key)['id']:
            raise ValueError('Wrong classical selection')
        for candidate in recipe['candidates']:
            label = candidate['id']
            model = LogisticRegression(C=candidate['C'], class_weight='balanced', max_iter=4000, tol=1e-6, random_state=42)
            model.fit(z[idx['train']], table.iloc[idx['train']].label)
            params = np.load(folder/f'{label}.npz', allow_pickle=False)
            for actual, field in ((scaler.mean_, 'mean'), (scaler.scale_, 'scale'),
                                  (model.coef_, 'coef'), (model.intercept_, 'intercept')):
                np.testing.assert_array_equal(actual, params[field])
            for role, rows in saved_rows(folder, label, table, idx).items():
                compare(independent(rows), candidate['metrics'][role])
                p = model.predict_proba(z[idx[role]])
                expected = rows[[f'p{c}' for c in range(classes)]].to_numpy()
                error = max(error, float(np.max(np.abs(p-expected))))
                np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
            checked += 1
    for role, rows in saved_rows(folder, 'context_prior', table, idx).items():
        p = np.exp(prior(table, idx['train'], idx[role], classes, crossfit=role == 'train'))
        np.testing.assert_allclose(p, rows[[f'p{c}' for c in range(classes)]].to_numpy(), atol=1e-6, rtol=0)
        compare(independent(rows), record['context'][role])
    result = {'passed': True, 'record_sha256': sha(folder/'record.json'),
              'classical_candidates_refitted': checked, 'coefficients_and_scalers_exact': True,
              'maximum_probability_error': error}
    atomic(folder/'verification.json', result)
    return result
