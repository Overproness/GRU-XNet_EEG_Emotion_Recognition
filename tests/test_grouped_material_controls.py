import numpy as np
import pandas as pd
import pytest
from gruxnet.grouped_material_controls import (annotate, bootstrap, cells, feasibility, folds, matched_deap, paired_batches, material_prior, new_model)
from test_session_controls import cohort


def deap_cohort():
    rows = []
    for p in range(1, 33):
        for t in range(1, 41):
            if p <= 16 and t == p: continue  # Sixteen discarded midpoint ratings.
            label = (p+t) % 2
            rows.append({"subject_id": f"DEAP:S{p:02}", "trial_id": f"DEAP:S{p:02}:T{t:02}",
                         "original_label": 7. if label else 3., "label": label, "session": 1})
    return pd.DataFrame(rows)


def test_new_seed_groupings_are_different_and_complete_with_original_sizes():
    original = cohort(); assignments = []; participant_groups = []
    for g in (1, 2):
        table = annotate(original, "SEEDIV", g)
        records = feasibility(original, "SEEDIV")
        assert all([r["train"], r["validation"], r["test"]] == [108, 12, 24] for r in records)
        assignments.append(table.material_rank.tolist())
        participant_groups.append(cells(table, "SEEDIV", g)[0])
    assert assignments[0] != assignments[1]
    assert participant_groups[0] != participant_groups[1]


def test_deap_individual_labels_matching_and_oof_with_missing_midpoints():
    original = deap_cohort(); assert original.trial_id.nunique() == 1264
    records = feasibility(original, "DEAP"); assert len(records) == 80
    table = annotate(original, "DEAP", 1); assert table.groupby("material_key").label.nunique().eq(2).all()
    fs, conditions = cells(table, "DEAP", 1); tested = []
    for _, r, f in conditions:
        pair = matched_deap(table, f, r, 1)
        for role in ("validation", "test"): np.testing.assert_array_equal(pair["exposed"][role], pair["unexposed"][role])
        a = table.iloc[pair["exposed"]["train"]]; b = table.iloc[pair["unexposed"]["train"]]
        pd.testing.assert_series_equal(a.groupby(["subject_id", "label"]).size(), b.groupby(["subject_id", "label"]).size())
        sigs = [paired_batches(table, pair[arm]["train"], 42, 2, 5)[1] for arm in ("exposed", "unexposed")]
        assert sigs[0] == sigs[1]
        tested.extend(table.iloc[pair["exposed"]["test"]].trial_id)
    assert len(tested) == len(set(tested)) == 1264
    assert [len(f["train"]) for f in fs] == [24]*8


def test_training_match_is_stable_to_table_order_and_rejects_missing_class():
    original = deap_cohort(); table = annotate(original, "DEAP", 1); f = cells(table, "DEAP", 1)[0][0]
    first = matched_deap(table, f, 0, 1)
    shuffled = annotate(original.sample(frac=1, random_state=10).reset_index(drop=True), "DEAP", 1)
    second = matched_deap(shuffled, f, 0, 1)
    for arm in first:
        assert set(table.iloc[first[arm]["train"]].trial_id) == set(shuffled.iloc[second[arm]["train"]].trial_id)
    broken = table.copy(); broken["label"] = 0
    with pytest.raises(ValueError, match="class coverage"): matched_deap(broken, f, 0, 1)


def test_crossed_bootstrap_respects_individual_classes_and_missing_cells():
    rows = pd.DataFrame({"subject_id": ["A", "A", "B"], "material_key": ["V1", "V2", "V1"],
                         "label": [0, 1, 1], "p_0": [.8, .6, .2], "p_1": [.2, .4, .8]})
    sw = np.array([[1, 1], [2, 1]], dtype=float); mw = np.array([[1, 1], [3, 1]], dtype=float)
    point, scores = bootstrap(rows, "DEAP", "binary", ["A", "B"], ["V1", "V2"], sw, mw)
    assert point == .75
    # Draw2: negative A/V1 correct; positive numerator3(B/V1),denominator3+2(A/V2).
    np.testing.assert_allclose(scores, [.75, .8])
    repeated = pd.concat([rows, rows], ignore_index=True)
    p2, s2 = bootstrap(repeated, "DEAP", "binary", ["A", "B"], ["V1", "V2"], sw, mw)
    assert p2 == point; np.testing.assert_array_equal(s2, scores)


def test_material_prior_only_uses_training_labels_and_unseen_global_fallback():
    rows = pd.DataFrame({"material_key": ["V1", "V1", "V2", "V1", "V9"], "label": [1, 1, 0, 0, 1]})
    first = material_prior(rows, [0, 1, 2], [3, 4]); rows.loc[[3, 4], "label"] = [1, 0]
    np.testing.assert_array_equal(first, material_prior(rows, [0, 1, 2], [3, 4]))
    np.testing.assert_allclose(first[:, 1], [.75, .6])


def test_binary_heads_are_real_two_class_models_with_matched_capacity():
    models = [new_model(n, "DEAP") for n in ("mean_mlp", "transformer")]
    counts = [sum(p.numel() for p in m.parameters()) for m in models]
    assert counts == [18992, 19042]
    assert [m.head.out_features for m in models] == [2, 2]
