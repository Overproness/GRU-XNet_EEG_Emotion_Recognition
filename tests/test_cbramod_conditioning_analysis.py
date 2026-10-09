"""Check factorial effects against a closed-form interaction, including selection."""
import json
import pandas as pd
import pytest
from scripts.analyze_cbramod_conditioning import contrasts


def fixture_rows():
    rows = []
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            for pretrained in (True, False):
                for trainable in (False, True):
                    for scaled in (False, True):
                        for dropout in (True, False):
                            for step in (0, 200, 600, 1200):
                                for role in ('train', 'validation_unseen', 'validation_familiar'):
                                    off = not dropout
                                    # The four cells independently imply effects 2,7,3,8,5.
                                    value = group + 2*scaled + 3*off + 5*scaled*off
                                    rows.append(dict(dataset=dataset, group=group, pretrained=pretrained,
                                        trainable=trainable, scaled=scaled, dropout=dropout, step=step, role=role,
                                        balanced_log_loss=value, balanced_accuracy=value/100))
    return pd.DataFrame(rows)


def test_closed_form_factorial_and_different_selected_durations():
    rows = fixture_rows()
    selected = rows[((~rows.scaled)&(rows.step == 200)) | (rows.scaled&(rows.step == 600))]
    points = contrasts(rows, selected)
    expected = {'normalization_dropout_on': 2, 'normalization_dropout_off': 7,
                'dropout_off_raw': 3, 'dropout_off_standardized': 8, 'interaction': 5}
    assert len(points) == 2400
    assert json.loads(json.dumps(points, allow_nan=False)) == points
    for point in points:
        scale = 1 if point['metric'] == 'balanced_log_loss' else .01
        assert point['delta'] == pytest.approx(expected[point['contrast']]*scale)
        if point['scope'] == 'fixed_step' and point['contrast'] != 'interaction':
            assert point['changed_step'] == point['reference_step'] == point['step']
        if point['scope'] == 'selected_secondary' and point['contrast'].startswith('normalization'):
            assert (point['changed_step'], point['reference_step']) == (600, 200)
    assert sum(p['scope'] == 'fixed_step' and p['step'] > 0 and p['role'] != 'train' for p in points) == 960


def test_missing_factorial_cell_is_rejected():
    rows = fixture_rows()
    selected = rows[rows.step == 200]
    missing = rows.drop(rows.index[0])
    with pytest.raises(ValueError, match='Missing matched factorial cell'):
        contrasts(missing, selected)
