"""Independent dyadic uncertainty check and legacy-schema recovery regression."""
import numpy as np
import pandas as pd
import pytest
from gruxnet.deap_baseline_folds_v2 import MODELS, ARMS
from scripts.analyze_deap_baseline_folds_v2 import alignment
from scripts.within_video_alignment import pairs, scores, aggregate
from scripts.audit_deap_baseline_schema_recovery import legacy_probability
from scripts.analyze_deap_baseline_schema_recovery import statistic, independent_regular_check


def test_legacy_probability_columns_are_read_explicitly():
    rows = pd.DataFrame({'p_0': [.3, .6], 'p_1': [.7, .4]})
    np.testing.assert_array_equal(legacy_probability(rows), [[.3, .7], [.6, .4]])
    with pytest.raises(KeyError): legacy_probability(rows.rename(columns={'p_0': 'wrong'}))


def test_regular_points_and_bootstrap_endpoints_recompute_with_explicit_schema_adapter():
    rows = pd.DataFrame({'subject_id': np.repeat(['A', 'B', 'C'], 2), 'material_key': ['a', 'b']*3,
        'trial_id': [f'T{i}' for i in range(6)], 'label': [0, 1, 1, 0, 0, 1], 'original_label': [2, 8, 8, 2, 2, 8],
        'group': 1, 'p0': [.7, .1, .1, .8, .6, .3], 'p1': [.3, .9, .9, .2, .4, .7]})
    second = rows.copy(); second['group'] = 2
    full = pd.concat([rows, second], ignore_index=True)
    frames = {('baseline_relative', 'unexposed'): full}
    subjects = ['A', 'B', 'C']; materials = ['a', 'b']
    sw = np.array([[1., 1., 1.], [2., 1., 0.], [0., 2., 1.]])
    mw = np.array([[1., 1.], [2., 0.], [0., 2.]])
    result = {'models': {}, 'contrasts': []}
    for scope, group in (('combined', None), ('group1', 1), ('group2', 2)):
        frame = full if group is None else full[full.group.eq(group)]
        report = {}
        for which in ('BA', 'logloss'):
            point, crossed = statistic(frame, 'DEAP', 'binary', subjects, materials, sw, mw, which)
            _, person = statistic(frame, 'DEAP', 'binary', subjects, materials, sw, np.ones_like(mw), which)
            report[which] = {'point': point, 'crossed_percentile_95': np.quantile(crossed, [.025, .975]),
                'participant_percentile_95': np.quantile(person, [.025, .975])}
        result['models'][scope] = {'baseline_relative': {'unexposed': report}}
    assert independent_regular_check(frames, result, subjects, materials, sw, mw) < 1e-12


def test_vectorized_dyadic_intervals_match_previously_audited_pairwise_implementation():
    first = pd.DataFrame({'subject_id': np.repeat(['A', 'B', 'C'], 2), 'material_key': ['a', 'b']*3,
        'label': [0, 1, 1, 0, 0, 1], 'source_session': 1, 'material_rotation': 0, 'fold': 0,
        'trial_id': [f'T{i}' for i in range(6)], 'group': 1})
    second = first.copy(); second['group'] = 2
    reference = pd.concat([first, second], ignore_index=True); frames = {}
    for model in MODELS:
        for arm in ARMS:
            rows = reference.copy()
            probability = np.tile([.3, .6, .9, .2, .4, .7], 2)
            if model in ('prior', 'context_logistic'): probability = np.tile([.4, .7], 6)
            rows['p0'] = 1-probability; rows['p1'] = probability; frames[(model, arm)] = rows
    subjects = ['A', 'B', 'C']; materials = ['a', 'b']
    sw = np.array([[1., 1., 1.], [2., 1., 0.], [0., 2., 1.]])
    mw = np.array([[1., 1.], [2., 0.], [0., 2.]])
    result = alignment(frames, subjects, materials, sw, mw)
    recipient = []; donor = []; base = []
    for _, part in reference.groupby('group'):
        r, d, b, _ = pairs(part.reset_index(drop=True))
        recipient.extend(part.index.to_numpy()[r]); donor.extend(part.index.to_numpy()[d]); base.extend(b)
    recipient = np.asarray(recipient); donor = np.asarray(donor); base = np.asarray(base)
    ri = reference.iloc[recipient].subject_id.map(dict(zip(subjects, range(3)))).to_numpy()
    di = reference.iloc[donor].subject_id.map(dict(zip(subjects, range(3)))).to_numpy()
    vi = reference.iloc[recipient].material_key.map(dict(zip(materials, range(2)))).to_numpy()
    for model in ('baseline_relative_context', 'prior', 'context_logistic'):
        for arm in ARMS:
            legacy = frames[(model, arm)].rename(columns={'p0': 'p_0', 'p1': 'p_1'})
            target, included, values, classes = scores(legacy, 'binary', 'DEAP', recipient, donor)
            for name in ('aligned', 'exchanged'):
                for which in ('BA', 'logloss'):
                    saved = result['models'][model][arm][name][which]
                    for scope, weights in (('crossed', mw), ('participant', np.ones_like(mw))):
                        point, draws, valid = aggregate(values[name][which], target, included, classes, base, ri, di, vi, sw, weights)
                        np.testing.assert_allclose(saved['point'], point, atol=1e-12, rtol=0)
                        np.testing.assert_allclose(saved[f'{scope}_percentile_95'], np.quantile(draws[valid], [.025, .975]), atol=1e-12, rtol=0)
    assert len(result['contrasts']) == 40
