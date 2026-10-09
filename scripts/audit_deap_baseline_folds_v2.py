"""Independent prior, coefficient-refit and saved-probability audit."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.deap_baseline_folds_v2 import (MODELS, REPRESENTATIONS, SOURCE_ROLES, ROLES,
    C_VALUES, cell_id, full_job, inputs_for, estimator, direct_probability, fast_prior)
from gruxnet.preprocessing_diagnostic_v2 import atomic, sha, stamp, selection_key
from gruxnet.heldout_tuning import partitions, context_id
from gruxnet.full_context_controls_v2 import prior as reference_prior


def independent_metrics(rows):
    p = rows[['p0', 'p1']].to_numpy(); y = rows.label.to_numpy(dtype=int)
    if set(y) != {0, 1} or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid binary outcomes')
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-12, rtol=0)
    sample_weight = np.where(y == 0, .5/np.count_nonzero(y == 0), .5/np.count_nonzero(y == 1))
    loss = log_loss(y, np.clip(p, 1e-12, 1.), sample_weight=sample_weight, labels=[0, 1])
    return {'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(1))),
        'balanced_log_loss': float(loss), 'accuracy': float(np.mean(p.argmax(1) == y)), 'n': len(y)}


def compare_metrics(rows, expected):
    actual = independent_metrics(rows)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=0)


def assert_metadata(rows, table, indexes, job, model, role):
    expected = table.iloc[indexes][['trial_id', 'subject_id', 'material_key', 'label', 'original_label']].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[expected.columns].reset_index(drop=True), expected, check_exact=True)
    for column, value in (('group', job['group']), ('material_rotation', job['rotation']),
        ('source_session', 1), ('fold', job['fold']), ('arm', job['arm']), ('model', model), ('role', role)):
        if not rows[column].eq(value).all(): raise ValueError('Changed prediction context')


def certificate_valid(folder, output):
    certificate = json.loads((folder/'verification.json').read_text())
    if not certificate['passed'] or certificate['plan_sha256'] != sha(output/'plan.json'):
        raise ValueError('Wrong verified declaration')
    for name, checksum in certificate['artifact_sha256'].items():
        if sha(folder/name) != checksum: raise ValueError(f'Changed verified artifact: {name}')
    return certificate


def audit_model(output, root, base, features, job, model):
    folder = output/'cells'/cell_id(job)/model
    if (folder/'verification.json').exists(): return certificate_valid(folder, output)
    table, idx = partitions(base, full_job(job))
    record = json.loads((folder/'record.json').read_text())
    selection = json.loads((folder/'selection.json').read_text())
    if record['job'] != job or record['model'] != model or record['selection_sha256'] != sha(folder/'selection.json'):
        raise ValueError('Changed sealed outcome')
    if selection['config_sha256'] != sha(output/'config.json'):
        raise ValueError('Changed sealed configuration')
    if selection['split_trials'] != {r: table.iloc[idx[r]].trial_id.tolist() for r in ROLES}:
        raise ValueError('Changed source/test boundaries')
    for name, checksum in record['artifact_sha256'].items():
        if sha(folder/name) != checksum: raise ValueError('Changed prediction file')
    selected_rows = pd.read_csv(folder/'selected_predictions.csv', float_precision='round_trip')
    if set(selected_rows.role) != set(ROLES): raise ValueError('Missing prediction role')
    error = 0.; metric_sets = 0; prefixes = 0; refits = 0
    if model == 'prior':
        for role in ROLES:
            rows = selected_rows[selected_rows.role.eq(role)]
            assert_metadata(rows, table, idx[role], job, model, role)
            expected = reference_prior(table, idx['train'], idx[role], 2, crossfit=role == 'train')
            np.testing.assert_array_equal(expected, fast_prior(table, idx['train'], idx[role], crossfit=role == 'train'))
            np.testing.assert_allclose(rows[['p0', 'p1']], expected, atol=1e-12, rtol=0)
            compare_metrics(rows, record['metrics'][role]); metric_sets += 1
    else:
        inputs = inputs_for(features, table, idx, job, model, ROLES)
        scaler = StandardScaler().fit(inputs['train'])
        params = np.load(folder/'parameters.npz', allow_pickle=False)
        if sha(folder/'parameters.npz') != selection['parameters_sha256']:
            raise ValueError('Changed local coefficients')
        np.testing.assert_array_equal(scaler.mean_, params['mean'])
        np.testing.assert_array_equal(scaler.scale_, params['scale'])
        candidates = pd.read_csv(folder/'candidates_source.csv', float_precision='round_trip')
        if set(candidates.role) != set(SOURCE_ROLES) or set(candidates.C) != set(C_VALUES):
            raise ValueError('Missing candidate or leaked test candidate')
        if len(selection['candidates']) != 4 or selection['selected'] != min(selection['candidates'], key=selection_key)['id']:
            raise ValueError('Source-only selection rule changed')
        for position, candidate in enumerate(selection['candidates']):
            c = candidate['C']
            if c != C_VALUES[position]: raise ValueError('Candidate order changed')
            other = estimator(c).fit(scaler.transform(inputs['train']), table.iloc[idx['train']].label)
            if np.any(other.n_iter_ >= 4000): raise ValueError('Refit did not converge')
            np.testing.assert_array_equal(other.coef_[0], params['coef'][position])
            np.testing.assert_array_equal(other.intercept_[0], params['intercept'][position]); refits += 1
            for role in SOURCE_ROLES:
                rows = candidates[candidates.C.eq(c) & candidates.role.eq(role)]
                assert_metadata(rows, table, idx[role], job, model, role)
                p = direct_probability(inputs[role], params, position)
                error = max(error, float(np.max(np.abs(p-rows[['p0', 'p1']].to_numpy()))))
                np.testing.assert_allclose(p, rows[['p0', 'p1']], atol=1e-12, rtol=0)
                np.testing.assert_allclose(other.predict_proba(scaler.transform(inputs[role])), p, atol=1e-12, rtol=0)
                compare_metrics(rows, candidate['metrics'][role]); metric_sets += 1
            # Exact source-control sentinels against the earlier declared panel.
            if job['rotation'] == 0 and job['fold'] == 0 and job['arm'] == 'unexposed' and model in REPRESENTATIONS:
                mapping = {'stimulus_absolute': 'absolute', 'stimulus_relative': 'relative', 'baseline_relative': 'baseline'}
                if model == 'baseline_only':
                    old_path = root/'prestimulus_control_2026-10-09/fits'/f'deap_g{job["group"]}_native'/f'C{c}.npz'
                else:
                    old_path = root/'preprocessing_diagnostic_v2_2026-10-09/classical'/f'deap_g{job["group"]}'/f'native_{mapping[model]}_C{c}.npz'
                old = np.load(old_path, allow_pickle=False)
                for field in ('mean', 'scale'): np.testing.assert_array_equal(old[field], params[field])
                np.testing.assert_array_equal(old['coef'].reshape(-1), params['coef'][position])
                np.testing.assert_array_equal(old['intercept'].reshape(-1), [params['intercept'][position]])
                prefixes += 1
        chosen = next(i for i, c in enumerate(selection['candidates']) if c['id'] == selection['selected'])
        for role in ROLES:
            rows = selected_rows[selected_rows.role.eq(role)]
            assert_metadata(rows, table, idx[role], job, model, role)
            p = direct_probability(inputs[role], params, chosen)
            error = max(error, float(np.max(np.abs(p-rows[['p0', 'p1']].to_numpy()))))
            np.testing.assert_allclose(p, rows[['p0', 'p1']], atol=1e-12, rtol=0)
            compare_metrics(rows, record['metrics'][role]); metric_sets += 1
    context_match = False
    if model in ('prior', 'context_logistic'):
        old = pd.read_csv(root/'heldout_tuning_2026-10-06/contexts'/context_id(job)/f'predictions_{model}_test.csv', float_precision='round_trip')
        rows = selected_rows[selected_rows.role.eq('test')]
        if rows.trial_id.tolist() != old.trial_id.tolist(): raise ValueError('Old contextual trial population changed')
        np.testing.assert_allclose(rows[['p0', 'p1']], old[['p0', 'p1']], atol=1e-12, rtol=0)
        context_match = True
    names = [*record['artifact_sha256'], 'record.json', 'selection.json']
    if model != 'prior': names.append('parameters.npz')
    certificate = {'passed': True, 'created_utc': stamp(), 'job': job, 'model': model,
        'plan_sha256': sha(output/'plan.json'), 'candidate_refits': refits, 'metric_sets': metric_sets,
        'exact_older_candidate_prefixes': prefixes, 'older_context_test_match': context_match,
        'maximum_probability_error': error, 'artifact_sha256': {name: sha(folder/name) for name in names},
        'scope': 'Fresh training-only scaler/model refits, exact coefficients, direct logistic-link predictions, independent sklearn balanced metrics, exact source/test rows and source selection. Reused cohorts; not first-party signal authentication or a new independent sample.'}
    atomic(folder/'verification.json', certificate)
    return certificate


def audit_complete(output):
    plan = json.loads((output/'plan.json').read_text()); certificates = []
    for job in plan['jobs']:
        for model in MODELS:
            certificates.append(certificate_valid(output/'cells'/cell_id(job)/model, output))
    result = {'passed': True, 'complete': True, 'plan_sha256': sha(output/'plan.json'),
        'verified_cells': len(plan['jobs']), 'verified_models_and_priors': len(certificates),
        'candidate_fits_independently_refitted': sum(c['candidate_refits'] for c in certificates),
        'metric_sets_independently_recomputed': sum(c['metric_sets'] for c in certificates),
        'exact_older_candidate_prefixes': sum(c['exact_older_candidate_prefixes'] for c in certificates),
        'older_context_test_matches': sum(c['older_context_test_match'] for c in certificates),
        'maximum_probability_error': max(c['maximum_probability_error'] for c in certificates)}
    assert result['candidate_fits_independently_refitted'] == 5760
    assert result['verified_models_and_priors'] == 1600
    assert result['exact_older_candidate_prefixes'] == 32
    assert result['older_context_test_matches'] == 320
    atomic(output/'verification.json', result)
    return result
