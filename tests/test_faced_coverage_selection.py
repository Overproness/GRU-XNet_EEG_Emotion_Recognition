"""Synthetic role, support, quality, matching and selection qualification."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import gruxnet.faced_coverage_selection as module
from gruxnet.faced_coverage_selection import (
    ALPHAS, MODELS, SourceSelectionGuard, crossed_source_cells,
    distributed_features, fit_full_source, nominate_coverage, qualify_quality,
    run_source_cv, select_common_alpha, source_plan, source_roster,
)


def fixture():
    people = [f"source-{i:03d}" for i in range(70)]
    category_clips = {"Amusement": (2, 3), "Anger": (4, 6),
                      "Disgust": (11, 12), "Fear": (9,), "Inspiration": (20,),
                      "Joy": (17, 18), "Sadness": (23, 25), "Tenderness": (27, 28)}
    category_map = {v: category for category, clips in category_clips.items() for v in clips}
    pairs = [(p, v) for p in people for v in sorted(category_map)]
    metadata = {pair: {"role": "source", "assigned_category": category_map[pair[1]],
                       "family": f"film-{pair[1]}", "assigned_valence": "positive" if pair[1] % 2 else "negative",
                       "acquisition_group": ("legacy_1000", "legacy_250", "modern_1000")[i % 3],
                       "catalogue_duration_seconds": 90.0 + pair[1], "trial": pair[1]}
                for i, pair in enumerate(pairs)}
    folds = [people[i * 14:(i + 1) * 14] for i in range(5)]
    quality = [{"subject": p, "vid": v, "role": "source", "region": region,
                "technical_passed": True, "review_flags": []}
               for p, v in pairs for region in ("early", "middle", "late")]
    return pairs, metadata, folds, quality


def test_crossed_cells_exclude_people_and_material_while_preserving_category_support():
    pairs, metadata, folds, _ = fixture()
    cells = crossed_source_cells(pairs, metadata, folds)
    assert len(cells) == 10
    for cell in cells:
        assert len(cell["training"]) == 448 and len(cell["held"]) == 84
        assert not {p[0] for p in cell["training"]} & {p[0] for p in cell["held"]}
        assert not {metadata[p]["family"] for p in cell["training"]} & {metadata[p]["family"] for p in cell["held"]}
        assert {metadata[p]["assigned_category"] for p in cell["training"]} == set(module.CATEGORIES)
        assert {9, 20} <= set(cell["training_clips"])
        assert not {9, 20} & set(cell["held_clips"])
    assert Counter(p for cell in cells for p in cell["held"]) == Counter(p for p in pairs if p[1] not in (9, 20))
    assert cells[0]["held_clips"] == [2, 4, 11, 17, 23, 27]
    assert cells[1]["held_clips"] == [3, 6, 12, 18, 25, 28]


@pytest.mark.parametrize("change", ["duplicate", "protected", "missing", "shared_family", "singleton"])
def test_source_roster_rejects_role_geometry_and_material_drift(change):
    pairs, metadata, _, _ = fixture()
    if change == "duplicate":
        pairs[-1] = pairs[0]
    elif change == "protected":
        metadata[pairs[0]]["role"] = "confirmation"
    elif change == "missing":
        metadata.pop(pairs[0])
    elif change == "shared_family":
        for pair in pairs:
            metadata[pair]["family"] = "all-one-film"
    else:
        for pair in pairs:
            if pair[1] == 9:
                metadata[pair]["assigned_category"] = "Joy"
    with pytest.raises(PermissionError):
        source_roster(pairs, metadata)


def test_changed_person_folds_are_rejected():
    pairs, metadata, folds, _ = fixture()
    folds[0][0] = folds[1][0]
    with pytest.raises(PermissionError):
        crossed_source_cells(pairs, metadata, folds)


def bomb(*args):
    raise AssertionError("Disallowed or premature callback was invoked")


def test_identity_guards_reject_protected_before_feature_or_target_callbacks():
    pairs, metadata, _, _ = fixture()
    guard = SourceSelectionGuard(pairs, metadata)
    with pytest.raises(PermissionError):
        guard.feature(("protected", 2), "late", bomb)
    with pytest.raises(PermissionError):
        guard.target(("validation", 2), bomb)
    with pytest.raises(PermissionError):
        guard.feature(pairs[0], "full_video", bomb)
    with pytest.raises(PermissionError):
        guard.target((pairs[0][0], True), bomb)


@pytest.mark.parametrize("change", ["missing", "duplicate", "flagged", "failed", "protected"])
def test_quality_holds_all_callbacks_when_any_region_is_unqualified(change):
    pairs, metadata, folds, quality = fixture()
    if change == "missing":
        quality.pop()
    elif change == "duplicate":
        quality[-1] = quality[0]
    elif change == "flagged":
        quality[-1]["review_flags"] = ["railed_channel"]
    elif change == "failed":
        quality[-1]["technical_passed"] = False
    else:
        quality[-1]["subject"] = "protected"
    with pytest.raises(PermissionError):
        run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                      quality_rows=quality, feature_loader=bomb, target_loader=bomb)


def test_all_source_feature_finiteness_is_checked_before_first_target_callback():
    pairs, metadata, folds, quality = fixture()
    def broken(pair, region):
        return np.full(150, np.nan if pair == pairs[-1] and region == "late" else 1.)
    with pytest.raises(ValueError, match="finite"):
        run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                      quality_rows=quality, feature_loader=broken, target_loader=bomb)


def test_distributed_mean_is_of_log_features_with_equal_region_weight():
    early = np.full(150, -3.)
    middle = np.full(150, 0.)
    late = np.full(150, 3.)
    assert np.array_equal(distributed_features(early, middle, late), np.zeros(150))
    with pytest.raises(ValueError):
        distributed_features(early, middle, np.full(149, 3.))
    with pytest.raises(ValueError):
        distributed_features(early, middle, np.full(150, np.inf))


def candidates():
    return [{"model": model, "alpha": alpha, "participant_title_mae": 1., "trial_mse": 2.}
            for model in MODELS for alpha in ALPHAS]


def test_common_alpha_uses_context_and_both_coverages_rather_than_best_joint():
    rows = candidates()
    for row in rows:
        if row["alpha"] == .1:
            row["participant_title_mae"] = .1 if row["model"] == "late_joint" else 4.
        elif row["alpha"] == 10:
            row["participant_title_mae"] = 2.
    selected = select_common_alpha(rows)
    assert selected["selected_alpha"] == 100.
    assert nominate_coverage(rows, 100.)["model"] == "late_joint"
    assert nominate_coverage(rows, 100.)["alpha"] == 100.


def test_common_alpha_tie_uses_mse_then_fixed_alpha_priority():
    rows = candidates()
    assert select_common_alpha(rows)["selected_alpha"] == .1
    for row in rows:
        if row["alpha"] == 10:
            row["trial_mse"] = 1.
    assert select_common_alpha(rows)["selected_alpha"] == 10.
    with pytest.raises(PermissionError):
        select_common_alpha(rows[:-1])
    rows[0]["participant_title_mae"] = np.nan
    with pytest.raises(ValueError):
        select_common_alpha(rows)


def test_coverage_nomination_cannot_take_other_alpha_and_retains_adverse_competitor():
    rows = candidates()
    for row in rows:
        if row["model"] == "distributed_joint" and row["alpha"] == 10:
            row["trial_mse"] = .9
        if row["alpha"] == .1:
            row["participant_title_mae"] = 0.
    assert nominate_coverage(rows, 10.)["model"] == "distributed_joint"
    assert nominate_coverage(rows, 10.)["alpha"] == 10.
    with pytest.raises(PermissionError):
        nominate_coverage(rows, 1000.)


def mock_fit(monkeypatch):
    calls = []
    def fitted(pairs, features, targets, alpha):
        x = np.stack([features[p] for p in pairs])
        y = np.asarray([targets[p] for p in pairs])
        scaler = StandardScaler().fit(x)
        state = {"scaler": scaler, "estimator": SimpleNamespace(intercept_=float(np.mean(y))),
                 "source_identity_sha256": f"synthetic-{len(pairs)}", "source_pairs": list(pairs)}
        calls.append((list(pairs), x.copy(), y.copy(), alpha))
        return state
    monkeypatch.setattr(module, "fit_source_ridge", fitted)
    monkeypatch.setattr(module, "predict_ridge", lambda state, pairs, features: np.full(len(pairs), state["estimator"].intercept_))
    return calls


def test_bounded_cv_retains_every_candidate_and_matched_training_scaler(monkeypatch):
    pairs, metadata, folds, quality = fixture()
    fits = mock_fit(monkeypatch)
    scaler_checks = []
    original_check = module._assert_matched_scalers
    def observed_check(states):
        original_check(states)
        scaler_checks.append(tuple(states))
    monkeypatch.setattr(module, "_assert_matched_scalers", observed_check)
    access = {"feature": [], "target": []}
    def feature(pair, region):
        access["feature"].append((pair, region))
        return np.full(150, ("early", "middle", "late").index(region) + pair[1] / 100.)
    def target(pair):
        access["target"].append(pair)
        return float(pair[1] % 8)
    result = run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                           quality_rows=quality, feature_loader=feature, target_loader=target)
    assert len(fits) == 90 and len(result["cv_predictions"]) == 7560
    assert len(scaler_checks) == 30 and all(set(keys) == set(MODELS) for keys in scaler_checks)
    assert len(result["category_predictions"]) == 840 and len(result["candidate_metrics"]) == 9
    assert len(result["cell_metrics"]) == 100
    assert len(access["feature"]) == 2940 and len(access["target"]) == 980
    assert len(result["contrasts"]) == 3
    assert sum(r["primary_fixed_alpha"] for r in result["contrasts"]) == 1
    assert result["source_nomination"]["alpha"] == result["common_alpha_selection"]["selected_alpha"]
    for index in range(0, 90, 3):
        context, late, distributed = fits[index:index + 3]
        assert context[0] == late[0] == distributed[0]
        assert np.array_equal(context[1], late[1][:, :13])
        assert np.array_equal(context[1], distributed[1][:, :13])
        assert np.array_equal(context[2], late[2]) and np.array_equal(context[2], distributed[2])
        assert context[3] == late[3] == distributed[3]
        assert context[1].shape == (448, 13) and late[1].shape == (448, 163)


def test_shared_context_moments_must_match_exactly():
    state = lambda n: {"scaler": SimpleNamespace(mean_=np.zeros(n), var_=np.ones(n), scale_=np.ones(n))}
    fitted = {"context_only": state(13), "late_joint": state(163), "distributed_joint": state(163)}
    module._assert_matched_scalers(fitted)
    fitted["late_joint"]["scaler"].mean_[0] = 1e-15
    with pytest.raises(PermissionError, match="moments"):
        module._assert_matched_scalers(fitted)


def test_real_synthetic_svd_fits_preserve_shared_context_moments():
    pairs, metadata, folds, _ = fixture()
    cell = crossed_source_cells(pairs, metadata, folds)[0]
    contexts = {p: module.context_features(metadata[p]) for p in pairs}
    rng = np.random.default_rng(13)
    late = {p: np.concatenate((contexts[p], rng.normal(size=150))) for p in pairs}
    distributed = {p: np.concatenate((contexts[p], rng.normal(size=150))) for p in pairs}
    targets = {p: float(p[1] % 8) for p in pairs}
    maps = {"context_only": contexts, "late_joint": late, "distributed_joint": distributed}
    with threadpool_limits(limits=2, user_api="blas"):
        fitted = {model: module.fit_source_ridge(cell["training"], features, targets, 100.)
                  for model, features in maps.items()}
    module._assert_matched_scalers(fitted)
    for model in MODELS:
        predicted = module.predict_ridge(fitted[model], cell["held"], maps[model])
        assert predicted.shape == (84,) and np.isfinite(predicted).all()
        assert np.all((predicted >= 0) & (predicted <= 7))


def test_budget_expiry_stops_before_first_estimator_fit(monkeypatch):
    pairs, metadata, folds, quality = fixture()
    calls = mock_fit(monkeypatch)
    def expired():
        raise TimeoutError("Declared fitting budget expired")
    with pytest.raises(TimeoutError):
        run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                      quality_rows=quality, feature_loader=lambda p, r: np.zeros(150),
                      target_loader=lambda p: 3.5, check_budget=expired)
    assert not calls
    with pytest.raises(TimeoutError):
        fit_full_source(source_pairs=pairs, metadata=metadata, quality_rows=quality,
                        feature_loader=lambda p, r: np.zeros(150), target_loader=lambda p: 3.5,
                        candidate_metrics=candidates(), check_budget=expired)
    assert not calls


def test_full_source_state_helper_has_exactly_three_fits_and_no_held_scoring(monkeypatch):
    pairs, metadata, _, quality = fixture()
    calls = mock_fit(monkeypatch)
    scaler_checks = []
    original_check = module._assert_matched_scalers
    def observed_check(states):
        original_check(states)
        scaler_checks.append(tuple(states))
    monkeypatch.setattr(module, "_assert_matched_scalers", observed_check)
    result = fit_full_source(source_pairs=pairs, metadata=metadata, quality_rows=quality,
                             feature_loader=lambda p, r: np.full(150, p[1]),
                             target_loader=lambda p: float(p[1] % 8), candidate_metrics=candidates())
    assert len(calls) == result["full_source_fit_count"] == 3
    assert len(scaler_checks) == 1
    assert result["full_source_prior_count"] == 1
    assert all(len(call[0]) == 980 for call in calls)
    assert all(row["validation_rows_scored"] == row["protected_rows_scored"] == 0 for row in result["fits"])
    assert not result["validation_access"] and not result["protected_access"]


def test_metadata_plan_fixes_primary_alpha_and_secondary_fair_selection():
    pairs, metadata, folds, _ = fixture()
    plan = source_plan(pairs, metadata, folds)
    assert plan["primary_fixed_alpha"] == 100.
    assert plan["fit_cap"] == 90
    assert plan["total_ridge_fit_cap_including_optional_state_qualification"] == 93
    assert plan["ridge_cv_prediction_rows"] == 7560
    assert len(plan["cells_sha256"]) == 64
    assert not plan["validation_access"] and not plan["protected_access"]
