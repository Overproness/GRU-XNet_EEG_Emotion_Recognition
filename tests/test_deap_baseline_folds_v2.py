"""Source boundary, person-exclusion, power and repeated-score semantics."""
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from gruxnet.deap_baseline_folds_v2 import (fast_prior, make_features, metric, inputs_for,
    fit_model, cell_id, REPRESENTATIONS)
from gruxnet.full_context_controls_v2 import prior
from gruxnet.preprocessing_diagnostic_v2 import selection_key


def test_rating_roundoff_is_accepted_without_accepting_real_rating_or_class_changes():
    import pytest
    from gruxnet.deap_baseline_folds_v2 import check_rating
    check_rating(2.51, 2.5100000000000002, 0)
    with pytest.raises(ValueError): check_rating(2.51, 2.52, 0)
    with pytest.raises(ValueError): check_rating(2.51, 2.5100000000000002, 1)


def toy():
    table = pd.DataFrame({'trial_id': [f'T{i}' for i in range(12)],
        'subject_id': np.repeat(list('ABCDEF'), 2),
        'material_key': ['a', 'b', 'a', 'b', 'u', 'v', 'a', 'b', 'x', 'y', 'x', 'y'],
        'label': [0, 1]*6, 'original_label': [2., 8.]*6})
    idx = {'train': np.arange(4), 'validation_unseen': np.arange(4, 6),
           'validation_familiar': np.arange(6, 8), 'test': np.arange(8, 12)}
    return table, idx


def test_whole_participant_prior_and_unseen_global_fallback():
    table, idx = toy()
    for role, rows in idx.items():
        actual = fast_prior(table, idx['train'], rows, crossfit=role == 'train')
        np.testing.assert_array_equal(actual, prior(table, idx['train'], rows, 2, crossfit=role == 'train'))
    # Receiving person A has only negatives, while B has only positives.
    table.loc[[0, 1], 'label'] = 0; table.loc[[2, 3], 'label'] = 1
    np.testing.assert_array_equal(fast_prior(table, idx['train'], np.array([0, 1]), True), [[1/3, 2/3]]*2)
    np.testing.assert_array_equal(fast_prior(table, idx['train'], idx['test']), [[.5, .5]]*4)


def test_heldout_labels_cannot_change_source_context():
    table, idx = toy(); changed = table.copy(); changed.loc[4:, 'label'] = 1-changed.loc[4:, 'label']
    for role, rows in idx.items():
        np.testing.assert_array_equal(fast_prior(table, idx['train'], rows, role == 'train'),
                                      fast_prior(changed, idx['train'], rows, role == 'train'))


def test_identical_stimulus_and_baseline_periodic_signal_has_zero_log_ratio():
    t = np.arange(5120)/128; b = np.arange(384)/128
    signal = np.stack([np.sin(2*np.pi*10*t)+.5*np.cos(2*np.pi*20*t)]*32).astype(np.float32)
    baseline = np.stack([np.sin(2*np.pi*10*b)+.5*np.cos(2*np.pi*20*b)]*32).astype(np.float32)
    features = make_features(signal, baseline)
    assert all(x.shape == (128,) and np.isfinite(x).all() for x in features.values())
    np.testing.assert_allclose(features['baseline_relative'], 0., atol=5e-6, rtol=0)


def test_power_ratio_preserves_channel_band_order_and_amplitude_response():
    rng = np.random.default_rng(31)
    signal = rng.normal(size=(32, 5120)).astype(np.float32)
    baseline = rng.normal(size=(32, 384)).astype(np.float32)
    a = make_features(signal, baseline); b = make_features(2*signal, baseline)
    np.testing.assert_allclose(b['baseline_relative']-a['baseline_relative'], np.log(4), atol=4e-6, rtol=0)
    np.testing.assert_allclose(a['stimulus_relative'], b['stimulus_relative'], atol=4e-6, rtol=0)
    np.testing.assert_array_equal(a['baseline_only'], b['baseline_only'])


def test_selection_uses_equal_panel_loss_and_stable_first_tie():
    def row(c, losses, accuracy):
        return {'id': str(c), 'C': c, 'metrics': {role: {'balanced_log_loss': loss, 'balanced_accuracy': accuracy}
            for role, loss in zip(('validation_unseen', 'validation_familiar'), losses)}}
    candidates = [row(.01, [.1, 1.1], 1.), row(.1, [.5, .5], .1), row(1., [.5, .5], .1)]
    assert min(candidates, key=selection_key)['C'] == .1


def test_source_only_retrieval_does_not_touch_test_features():
    table, idx = toy(); job = {'group': 1, 'rotation': 2, 'fold': 1, 'arm': 'unexposed'}
    class Guard:
        def __getitem__(self, indexes):
            assert not set(indexes) & set(idx['test'])
            return np.ones((len(indexes), 128))
    values = inputs_for({'baseline_relative': Guard()}, table, idx, job, 'baseline_relative_context',
                       ('train', 'validation_unseen', 'validation_familiar'))
    assert values['train'].shape == (4, 130)


def test_selection_seal_precedes_test_query_and_scaler_uses_only_train(monkeypatch):
    import gruxnet.deap_baseline_folds_v2 as experiment
    import scripts.audit_deap_baseline_folds_v2 as audit
    table, idx = toy(); job = {'dataset': 'DEAP', 'session': 1, 'group': 1, 'rotation': 2, 'fold': 1, 'arm': 'unexposed'}
    root = Path(__file__).resolve().parents[2]/'publication_runs'/f'deap_folds_test_{uuid4().hex}'
    root.mkdir(); (root/'plan.json').write_text('{}'); (root/'config.json').write_text('{}')
    features = {name: np.random.default_rng(4).normal(size=(12, 4)) for name in REPRESENTATIONS}
    features['stimulus_absolute'][idx['test']] += 1e6
    folder = root/'cells'/cell_id(job)/'stimulus_absolute'
    original = experiment.inputs_for
    def guarded(*args):
        if 'test' in args[-1]: assert (folder/'selection.json').exists()
        return original(*args)
    monkeypatch.setattr(experiment, 'partitions', lambda *args: (table, idx))
    monkeypatch.setattr(audit, 'partitions', lambda *args: (table, idx))
    monkeypatch.setattr(experiment, 'inputs_for', guarded)
    fit_model(root, table, features, job, 'stimulus_absolute')
    params = np.load(folder/'parameters.npz', allow_pickle=False)
    np.testing.assert_array_equal(params['mean'], features['stimulus_absolute'][idx['train']].mean(0))
    result = audit.audit_model(root, root, table, features, job, 'stimulus_absolute')
    assert result['candidate_refits'] == 4 and result['metric_sets'] == 16
    assert audit.certificate_valid(folder, root)['passed']


def test_balanced_metrics_average_classes_and_do_not_ensemble_repeats():
    y = np.array([0, 0, 0, 1]); p = np.array([[.6, .4], [.6, .4], [.6, .4], [.6, .4]])
    assert metric(y, p)['balanced_accuracy'] == .5
    # Repeated grouping scores must average correctness, not classify mean p.
    a = np.array([[.99, .01], [.51, .49]]); b = np.array([[.49, .51], [.01, .99]])
    repeated = (metric([0, 1], a)['balanced_accuracy']+metric([0, 1], b)['balanced_accuracy'])/2
    assert repeated == .5 and metric([0, 1], (a+b)/2)['balanced_accuracy'] == 1.
