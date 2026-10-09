"""Independent arithmetic for frozen readout contrasts and balanced metrics."""
import json
import numpy as np
import pandas as pd
import pytest
from scripts.analyze_cbramod_readout import contrasts, measures, HEADS


def test_head_and_pretraining_contrasts_preserve_exact_pairs_and_json():
    rows = []
    for pretrained in (True, False):
        for head, value in zip(HEADS, (2, 5, 11)):
            rows.append(dict(dataset='DEAP', group=np.int64(1), pretrained=pretrained, head=head, step=np.int64(200),
                role='validation_unseen', balanced_log_loss=float(value+(7 if pretrained else 0)), balanced_accuracy=value/20))
    points = contrasts(pd.DataFrame(rows), 'fixed_step')
    assert len(points) == 18
    assert json.loads(json.dumps(points)) == points
    head = [p for p in points if p['kind'] == 'head' and p['metric'] == 'balanced_log_loss']
    assert [p['delta'] for p in head] == [3, 9, 6, 3, 9, 6]
    pretraining = [p for p in points if p['kind'] == 'pretraining' and p['metric'] == 'balanced_log_loss']
    assert [p['delta'] for p in pretraining] == [7, 7, 7]
    with pytest.raises(ValueError, match='Missing head'):
        contrasts(pd.DataFrame(rows[:-1]), 'fixed_step')


def test_balanced_loss_and_accuracy_do_not_follow_class_frequency():
    y = np.array([0, 0, 0, 1]); p = np.array([[.8, .2], [.8, .2], [.8, .2], [.7, .3]])
    actual = measures(y, p)
    assert actual['balanced_accuracy'] == .5
    assert abs(actual['balanced_log_loss']-(-np.log(.8)-np.log(.3))/2) < 1e-15
    with pytest.raises(ValueError, match='Missing task class'):
        measures(y[:3], p[:3])
