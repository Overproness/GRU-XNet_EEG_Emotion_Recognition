"""Guard source-only scaling, readout information and grouped selection semantics."""
import numpy as np
import pandas as pd
import pytest
from gruxnet.cbramod_probe import patches, pool, band_features, preprocess, fit_estimator
from gruxnet.preprocessing_diagnostic_v2 import selection_key
from scripts.audit_cbramod_source_probe import independent_metric


def test_native_patch_order_and_physical_amplitude_scaling():
    x = np.zeros((2, 3, 8000), np.float32)
    for t in range(2):
        for c in range(3):
            for w in range(4):
                x[t, c, w*2000:(w+1)*2000] = 100*(100*t+10*c+w)
    actual = patches(x)
    assert actual.shape == (8, 3, 10, 200)
    for t in range(2):
        for w in range(4):
            for c in range(3):
                np.testing.assert_array_equal(actual[4*t+w, c], 100*t+10*c+w)
    with pytest.raises(ValueError): patches(x[:, :, :-1])


def test_flatten_preserves_channel_and_temporal_information_average_discards():
    a = np.arange(8*3*10*200, dtype=np.float32).reshape(8, 3, 10, 200)
    first = pool(a); second = pool(a[:, ::-1, ::-1])
    np.testing.assert_array_equal(first['average'], second['average'])
    assert not np.array_equal(first['flatten'], second['flatten'])
    assert first['flatten'].shape == (2, 6000)
    np.testing.assert_array_equal(first['flatten'], a.reshape(2, 4, 3, 10, 200).mean(1).reshape(2, -1))


def test_band_controls_share_native_input_and_amplitude_response():
    rng = np.random.default_rng(61)
    raw = rng.normal(size=(2, 3, 8000)).astype(np.float32)
    a = band_features(raw); b = band_features(raw*2)
    assert all(v.shape == (2, 12) for v in a.values())
    np.testing.assert_allclose(b['band_absolute']-a['band_absolute'], np.log(4), atol=4e-6)
    np.testing.assert_allclose(b['band_relative'], a['band_relative'], atol=4e-6)


def test_deap_resampling_preserves_native_amplitude_and_duration():
    raw = np.ones((3, 7680), dtype=np.float32)*14
    transformed = preprocess(raw, 'DEAP')
    assert transformed.shape == (3, 8000)
    np.testing.assert_allclose(transformed, 14., atol=1e-5)


def test_changing_validation_covariates_or_labels_cannot_change_fitted_scaler_or_head():
    rng = np.random.default_rng(50)
    x = rng.normal(size=(18, 5)).astype(np.float32)
    table = pd.DataFrame({'label': [0, 1]*9})
    idx = {'train': np.arange(12), 'validation_unseen': np.arange(12, 14),
           'validation_familiar': np.arange(14, 18)}
    a = fit_estimator(x, table, idx, .01)
    changed = x.copy(); changed[12:] += 300
    other = table.copy(); other.loc[12:, 'label'] = 1-other.loc[12:, 'label']
    b = fit_estimator(changed, other, idx, .01)
    np.testing.assert_array_equal(a[0].mean_, b[0].mean_)
    np.testing.assert_array_equal(a[1].coef_, b[1].coef_)
    np.testing.assert_array_equal(a[1].intercept_, b[1].intercept_)
    for role, values in a[2].items():
        independent = independent_metric(table.label.to_numpy()[idx[role]], values)
        for key, value in independent.items():
            assert abs(value-a[3][role][key]) < 1e-12


def test_selection_balances_panels_and_keeps_declared_tie_order():
    def candidate(k, losses):
        return {'id': k, 'metrics': {r: {'balanced_log_loss': l, 'balanced_accuracy': .5}
            for r, l in zip(('validation_unseen', 'validation_familiar'), losses)}}
    values = [candidate(0, [.1, 1.1]), candidate(1, [.5, .5]), candidate(2, [.5, .5])]
    assert min(values, key=selection_key)['id'] == 1


@pytest.mark.parametrize('classes', (2, 3))
def test_complete_fit_selection_csv_binding_and_independent_refit(classes, monkeypatch):
    from pathlib import Path
    from uuid import uuid4
    import gruxnet.cbramod_probe as implementation
    import scripts.audit_cbramod_source_probe as audit
    output = Path(__file__).resolve().parents[2]/'publication_runs'/f'cbramod_test_{uuid4().hex}'
    output.mkdir(parents=True)
    (output/'plan.json').write_text('{}\n', encoding='utf-8')
    rng = np.random.default_rng(48)
    x = rng.normal(size=(36, 5)).astype(np.float32)
    table = pd.DataFrame({'trial_id': [f'T{i}' for i in range(36)],
        'subject_id': [f'S{i//6}' for i in range(36)],
        'material_key': [f'V{i%6}' for i in range(36)],
        'original_label': np.arange(36) % classes, 'label': np.arange(36) % classes})
    idx = {'train': np.arange(24), 'validation_unseen': np.arange(24, 30),
           'validation_familiar': np.arange(30, 36)}
    cache = output/'inputs/deap'; cache.mkdir(parents=True)
    np.save(cache/'pretrained_average.npy', x)
    def panel(*args, **kwargs): return table, idx, x
    monkeypatch.setattr(implementation, 'load_panel', panel)
    monkeypatch.setattr(audit, 'load_panel', panel)
    job = {'dataset': 'DEAP', 'group': 1, 'model': 'pretrained_average'}
    implementation.fit_job(output.parent, output, job)
    proof = audit.audit_head(output.parent, output, job)
    assert proof['complete'] and proof['candidate_refits'] == 4
    assert proof['max_metric_abs'] < 2e-11
