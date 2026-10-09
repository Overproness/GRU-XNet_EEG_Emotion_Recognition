import numpy as np
import torch
from gruxnet.preprocessing_diagnostic import (source_panel, transform, predict,
    recipes, classical_features, selection_key)
from gruxnet.data import COMMON_CHANNELS
from test_session_controls import cohort
from test_grouped_material_controls import deap_cohort


def test_new_source_panels_exclude_outer_people_and_trials():
    for dataset, base in (('SEEDIV', cohort()), ('DEAP', deap_cohort())):
        for group in (1, 2):
            table, idx, excluded = source_panel(base, dataset, group)
            assert not set(table.subject_id) & set(excluded)
            assert not set(idx['train']) & set(idx['validation_unseen'])
            assert not set(idx['validation_familiar']) & set(idx['validation_unseen'])
            assert not set(table.iloc[idx['train']].subject_id) & set(table.iloc[idx['validation_familiar']].subject_id)
            assert set(table.iloc[idx['validation_familiar']].material_key) <= set(table.iloc[idx['train']].material_key)
            assert not set(table.iloc[idx['validation_unseen']].material_key) & set(table.iloc[idx['train']].material_key)


def test_training_scaler_never_uses_validation_signal():
    raw = np.random.default_rng(1).normal(size=(4, 14, 5120)).astype(np.float32)
    recipe = dict(montage='native', baseline=False, normalization='source_channel')
    a, mean_a, scale_a = transform(raw, None, COMMON_CHANNELS, recipe, [0, 1])
    modified = raw.copy(); modified[2:] = modified[2:]*200+300
    b, mean_b, scale_b = transform(modified, None, COMMON_CHANNELS, recipe, [0, 1])
    np.testing.assert_array_equal(mean_a, mean_b)
    np.testing.assert_array_equal(scale_a, scale_b)
    np.testing.assert_array_equal(a[:2], b[:2])


def test_trial_normalization_does_not_pool_other_observations():
    raw = np.random.default_rng(2).normal(size=(4, 14, 5120)).astype(np.float32)
    recipe = dict(montage='native', baseline=False, normalization='trial_zscore')
    a, _, _ = transform(raw, None, COMMON_CHANNELS, recipe, [0, 1])
    modified = raw.copy(); modified[:3] *= 900
    b, _, _ = transform(modified, None, COMMON_CHANNELS, recipe, [0, 1])
    np.testing.assert_array_equal(a[3], b[3])


def test_baseline_uses_measured_waveform_and_is_subtracted_before_scaling():
    pattern = np.random.default_rng(3).normal(size=(2, 14, 128)).astype(np.float32)
    raw = np.tile(pattern, (1, 1, 40))
    baseline = np.tile(pattern, (1, 1, 3))
    recipe = dict(montage='native', baseline=True, normalization='source_channel')
    normalized, _, _ = transform(raw, baseline, COMMON_CHANNELS, recipe, [0])
    np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
    f = classical_features(raw, baseline, COMMON_CHANNELS, 'native', 'baseline')
    np.testing.assert_allclose(f, 0., atol=2e-6, rtol=0)


def test_short_inference_averages_logits_before_trial_softmax():
    class Example(torch.nn.Module):
        def forward(self, x):
            value = x[:, 0, 0]
            return torch.stack([value, -value], -1)
    values = torch.zeros(2, 10, 14, 512)
    values[:, :, 0, 0] = torch.arange(10)
    actual = predict(Example(), values, np.array([0, 1]), True)
    expected = torch.softmax(torch.tensor([[4.5, -4.5], [4.5, -4.5]]), -1).numpy()
    np.testing.assert_array_equal(actual, expected)
    mean_probability = torch.softmax(torch.stack([torch.arange(10.), -torch.arange(10.)], -1), -1).mean(0).numpy()
    assert not np.allclose(actual[0], mean_probability)


def test_all_recipes_keep_both_durations_and_baseline_is_deap_only():
    assert len(recipes('DEAP')) == 10 and len(recipes('SEEDIV')) == 8
    assert all(not r['baseline'] for r in recipes('SEEDIV'))
    assert all(r['seconds'] == 4 and r['normalization'] == 'source_channel' for r in recipes('DEAP') if r['baseline'])


def test_source_selection_uses_equal_panel_loss_before_accuracy():
    def candidate(a, b, accuracy):
        return {'metrics': {role: {'balanced_log_loss': loss, 'balanced_accuracy': accuracy}
                            for role, loss in (('validation_unseen', a), ('validation_familiar', b))}}
    a = candidate(.1, 1.1, 1.)
    b = candidate(.5, .5, .1)
    assert selection_key(b) < selection_key(a)
