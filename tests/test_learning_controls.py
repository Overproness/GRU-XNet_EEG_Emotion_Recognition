import numpy as np
import pandas as pd
import pytest
import torch
from gruxnet.learning_controls import draw_stream, normalize_waveform, tiny_indices
from gruxnet.full_context_controls_v2 import batches
from gruxnet.eegnet_control import EEGNetControl


def source_table():
    return pd.DataFrame({'subject_id': [f'p{i//8}' for i in range(24)],
                         'training_class': [i % 2 for i in range(24)],
                         'label': [i % 2 for i in range(24)],
                         'trial_id': [f't{i}' for i in range(24)]})


def test_longer_stream_preserves_entire_original_200_update_prefix():
    table = source_table(); train = np.arange(len(table))
    short, _ = batches(table, train, 42, 2)
    long, _ = draw_stream(table, train, 42, 2, 1200)
    np.testing.assert_array_equal(short, long[:200])
    assert all(np.bincount(table.iloc[row].label, minlength=2).tolist() == [6, 6] for row in long)


def test_waveform_scaler_ignores_heldout_values():
    rng = np.random.default_rng(19); original = rng.normal(size=(5, 14, 128)).astype('float32')
    train = np.array([0, 1, 2]); changed = original.copy(); changed[3:] += 10000
    normalized, mean, scale = normalize_waveform(original, train)
    other, other_mean, other_scale = normalize_waveform(changed, train)
    np.testing.assert_array_equal(mean, other_mean); np.testing.assert_array_equal(scale, other_scale)
    np.testing.assert_array_equal(normalized[train], other[train])
    np.testing.assert_allclose(normalized[train].mean((0, 2)), 0, atol=1e-6)


def test_tiny_batch_has_distinct_training_trials_and_balanced_labels():
    table = source_table(); eligible = np.arange(2, 22)
    selected = tiny_indices(table, eligible, 2)
    assert len(set(selected)) == 12 and set(selected) <= set(eligible)
    assert np.bincount(table.iloc[selected].label).tolist() == [6, 6]
    np.testing.assert_array_equal(selected, tiny_indices(table, eligible[::-1], 2))


def test_author_constraints_are_applied_to_correct_axes_after_update():
    model = EEGNetControl(3, samples=128)
    with torch.no_grad():
        model.spatial.weight.fill_(2.)
        model.head.weight.fill_(2.)
    model.constrain()
    assert torch.linalg.vector_norm(model.spatial.weight, dim=2).max() <= 1.000001
    assert torch.linalg.vector_norm(model.head.weight, dim=1).max() <= .250001
    loss = torch.nn.functional.cross_entropy(model(torch.randn(6, 14, 128)), torch.arange(6) % 3)
    loss.backward()
    for module in ('temporal', 'spatial', 'depthwise', 'pointwise', 'head'):
        weight = getattr(model, module).weight
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0


def test_eegnet_context_requires_a_matching_source_prior():
    model = EEGNetControl(2, samples=128, dropout=0., context=True).eval()
    x = torch.randn(3, 14, 128)
    with pytest.raises(ValueError): model(x)
    q = torch.log(torch.tensor([[.2, .8]]).repeat(3, 1))
    actual = model(x, q)
    model.context = False
    torch.testing.assert_close(actual, model(x)+q)
