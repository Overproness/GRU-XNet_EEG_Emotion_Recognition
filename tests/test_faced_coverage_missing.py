"""Manufactured qualification of shared missing EEG, without real data I/O."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import gruxnet.faced_coverage_missing as module
from gruxnet.faced_coverage_selection import ALPHAS, MODELS, crossed_source_cells
from gruxnet.faced_eegnet_bridge import array_sha256


def fixture():
    people = [f"source-{i:03d}" for i in range(70)]
    categories = {"Amusement": (2, 3), "Anger": (4, 6), "Disgust": (11, 12),
                  "Fear": (9,), "Inspiration": (20,), "Joy": (17, 18),
                  "Sadness": (23, 25), "Tenderness": (27, 28)}
    mapping = {v: c for c, titles in categories.items() for v in titles}
    pairs = [(p, v) for p in people for v in sorted(mapping)]
    metadata = {p: {"role": "source", "assigned_category": mapping[p[1]], "family": f"film-{p[1]}",
                    "assigned_valence": "positive" if p[1] % 2 else "negative",
                    "acquisition_group": ("legacy_1000", "legacy_250", "modern_1000")[i % 3],
                    "catalogue_duration_seconds": 90.0 + p[1], "trial": p[1]}
                for i, p in enumerate(pairs)}
    folds = [people[i * 14:(i + 1) * 14] for i in range(5)]
    quality = [{"subject": p, "vid": v, "role": "source", "region": region,
                "technical_passed": True, "review_flags": []}
               for p, v in pairs for region in ("early", "middle", "late")]
    return pairs, metadata, folds, quality


def mask(quality, pair, region="middle"):
    row = next(r for r in quality if (r["subject"], r["vid"], r["region"]) == (*pair, region))
    row["review_flags"] = [{"channel": module.COMMON_CHANNELS[0], "reason": "high_processed_excursion"}]


def bomb(*args):
    raise AssertionError("Premature, disallowed or missing-EEG callback invoked")


def candidate_metrics():
    return [{"model": m, "alpha": a, "participant_title_mae": 1., "trial_mse": 2.}
            for a in ALPHAS for m in MODELS]


def mock_fit(monkeypatch):
    calls = []
    def fitted(pairs, features, targets, alpha):
        x, y = np.stack([features[p] for p in pairs]), np.asarray([targets[p] for p in pairs])
        scaler = StandardScaler().fit(x)
        calls.append((list(pairs), x.copy(), y.copy(), alpha))
        return {"scaler": scaler, "estimator": SimpleNamespace(intercept_=float(np.mean(y))),
                "source_pairs": list(pairs), "source_identity_sha256": hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()}
    monkeypatch.setattr(module, "fit_source_ridge", fitted)
    monkeypatch.setattr(module, "predict_ridge", lambda fitted, pairs, features: np.full(len(pairs), fitted["estimator"].intercept_))
    return calls


def test_one_flagged_middle_region_masks_the_entire_trial_for_both_coverages():
    pairs, metadata, _, quality = fixture()
    missing = pairs[20]
    mask(quality, missing, "middle")
    available, summary = module.shared_availability(pairs, metadata, quality)
    assert not available[missing]
    assert summary["unavailable_pairs"] == [missing]
    assert summary["available_trials"] == 979 and summary["failed_or_flagged_region_count"] == 1
    assert not summary["availability_uses_targets"] and not summary["qc_thresholds_changed"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "protected", "wrong_region", "wrong_role", "technical_failure", "bad_bool", "bad_flags"])
def test_quality_incompleteness_and_technical_failure_stop_before_all_callbacks(change):
    pairs, metadata, folds, quality = fixture()
    if change == "missing":
        quality.pop()
    elif change == "duplicate":
        quality[-1] = quality[0]
    elif change == "protected":
        quality[-1]["subject"] = "protected"
    elif change == "wrong_region":
        quality[-1]["region"] = "full_video"
    elif change == "wrong_role":
        quality[-1]["role"] = "validation"
    elif change == "technical_failure":
        quality[-1]["technical_passed"] = False
    elif change == "bad_bool":
        quality[-1]["technical_passed"] = 1
    else:
        quality[-1]["review_flags"] = "flag"
    with pytest.raises((PermissionError, ValueError)):
        module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                             quality_rows=quality, feature_loader=bomb, target_loader=bomb)


def test_metadata_protected_role_is_rejected_before_any_feature_or_target_callback():
    pairs, metadata, folds, quality = fixture()
    metadata[pairs[-1]]["role"] = "confirmation"
    with pytest.raises(PermissionError):
        module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                             quality_rows=quality, feature_loader=bomb, target_loader=bomb)


def test_last_available_feature_failure_is_not_imputed_and_precedes_first_target():
    pairs, metadata, folds, quality = fixture()
    def invalid(pair, region):
        return np.full(150, np.nan if pair == pairs[-1] and region == "late" else 1.)
    with pytest.raises(ValueError, match="finite"):
        module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                             quality_rows=quality, feature_loader=invalid, target_loader=bomb)


def test_source_cell_without_available_training_donors_stops_before_features_or_targets():
    pairs, metadata, folds, quality = fixture()
    cell = crossed_source_cells(pairs, metadata, folds)[0]
    for pair in cell["training"]:
        mask(quality, pair)
    with pytest.raises(PermissionError, match="donors"):
        module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                             quality_rows=quality, feature_loader=bomb, target_loader=bomb)


def test_full_source_without_any_available_donors_stops_before_callbacks():
    pairs, metadata, _, quality = fixture()
    for row in quality:
        if row["region"] == "middle":
            row["review_flags"] = [{"channel": module.COMMON_CHANNELS[0], "reason": "high_processed_excursion"}]
    with pytest.raises(PermissionError, match="donors"):
        module.fit_full_source(source_pairs=pairs, metadata=metadata, quality_rows=quality,
                               feature_loader=bomb, target_loader=bomb,
                               candidate_metrics=candidate_metrics())


def test_training_mean_excludes_held_people_and_titles_even_if_their_features_are_extreme():
    pairs, metadata, folds, _ = fixture()
    cell = crossed_source_cells(pairs, metadata, folds)[0]
    training_set = set(cell["training"])
    missing = cell["training"][0]
    availability = {p: p != missing for p in pairs}
    contexts = {p: module.context_features(metadata[p]) for p in pairs}
    eeg = {p: np.full(150, 2. if p in training_set else 1e6) for p in pairs if availability[p]}
    features, mean, info = module.training_imputation(cell["training"], contexts, eeg, availability)
    assert np.array_equal(mean, np.full(150, 2.))
    assert info["donor_rows"] == 447 and info["missing_training_pairs"] == [missing]
    assert set(info["donor_pairs"]) <= training_set
    assert not {p[0] for p in info["donor_pairs"]} & {p[0] for p in cell["held"]}
    assert not {metadata[p]["family"] for p in info["donor_pairs"]} & {metadata[p]["family"] for p in cell["held"]}
    assert np.array_equal(features[missing][13:], mean)
    assert np.array_equal(features[missing][:13], contexts[missing])
    assert info["mean_sha256"] == array_sha256(mean)
    assert info["donor_identity_sha256"] == hashlib.sha256(json.dumps(info["donor_pairs"], separators=(",", ":")).encode()).hexdigest()


def test_imputation_with_no_donors_is_fatal():
    pairs, metadata, _, _ = fixture()
    training = pairs[:5]
    with pytest.raises(PermissionError, match="donors"):
        module.training_imputation(training, {p: module.context_features(metadata[p]) for p in training}, {}, {p: False for p in training})


def test_matched_cv_retains_missing_labels_skips_missing_features_and_shares_held_fallback(monkeypatch):
    pairs, metadata, folds, quality = fixture()
    source_only_missing = ("source-042", 20)
    held_missing = ("source-000", 2)
    missing = {source_only_missing, held_missing}
    for pair in missing:
        mask(quality, pair)
    calls = mock_fit(monkeypatch)
    observed_features, observed_targets, observed_states = [], [], []
    scaler_checks = []
    original_check = module._assert_matched_scalers
    def checked(states):
        original_check(states)
        scaler_checks.append(tuple(states))
    monkeypatch.setattr(module, "_assert_matched_scalers", checked)
    def feature(pair, region):
        if pair in missing:
            bomb()
        observed_features.append((pair, region))
        return np.full(150, pair[1] / 100. + ("early", "middle", "late").index(region))
    def target(pair):
        observed_targets.append(pair)
        return 7. if pair in missing else float(pair[1] % 7)
    result = module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                                  quality_rows=quality, feature_loader=feature, target_loader=target,
                                  on_fit=lambda info, state: observed_states.append((info, state)))
    assert len(calls) == 90 and len(result["cv_predictions"]) == 7560
    assert len(result["category_predictions"]) == 840 and len(result["candidate_metrics"]) == 9
    assert len(observed_features) == 2934 and set(observed_targets) == set(pairs)
    assert len(scaler_checks) == 30
    for index in range(0, 90, 3):
        context, late, distributed = calls[index:index + 3]
        assert context[0] == late[0] == distributed[0]
        assert np.array_equal(context[1], late[1][:, :13]) and np.array_equal(context[1], distributed[1][:, :13])
        assert np.array_equal(context[2], late[2]) and np.array_equal(context[2], distributed[2])
        for pair in missing & set(context[0]):
            assert context[2][context[0].index(pair)] == 7.
    for info, state in observed_states:
        if info["model"] == "context_only":
            assert state["eeg_imputation_mean"] is None and info["imputed_training_rows"] == 0
        else:
            assert state["eeg_imputation_mean"].shape == (150,)
            assert info["imputation"]["mean_sha256"] == array_sha256(state["eeg_imputation_mean"])
            assert set(info["imputed_training_pairs"]) == missing & set(info["training_pairs"])
            assert set(info["eligible_train_donor_pairs"]) == set(info["training_pairs"]) - missing
    prior = next(r for r in result["category_predictions"] if (r["subject"], r["vid"]) == held_missing)
    held_rows = [r for r in result["cv_predictions"] if (r["subject"], r["vid"]) == held_missing]
    assert len(held_rows) == 9
    assert all(r["prediction"] == prior["prediction"] and r["common_category_fallback"] and not r["shared_eeg_available"] for r in held_rows)
    assert result["source_nomination"]["alpha"] == result["common_alpha_selection"]["selected_alpha"]
    assert next(r for r in result["contrasts"] if r["primary_fixed_alpha"])["alpha"] == 100.


def test_full_source_state_fits_preserve_all980_labels_and_publish_means(monkeypatch):
    pairs, metadata, _, quality = fixture()
    missing = pairs[0]
    mask(quality, missing)
    calls = mock_fit(monkeypatch)
    def feature(pair, region):
        if pair == missing:
            bomb()
        return np.full(150, pair[1])
    result = module.fit_full_source(source_pairs=pairs, metadata=metadata, quality_rows=quality,
                                   feature_loader=feature, target_loader=lambda p: 7. if p == missing else 3.5,
                                   candidate_metrics=candidate_metrics())
    assert len(calls) == 3 and all(len(call[0]) == 980 for call in calls)
    assert all(call[2][call[0].index(missing)] == 7. for call in calls)
    assert result["full_source_prior_count"] == 1
    for model in MODELS[1:]:
        state = result["private_fitted_models"][model]
        info = next(r for r in result["fits"] if r["model"] == model)
        assert info["imputation"]["donor_rows"] == 979
        assert info["imputation"]["mean_sha256"] == array_sha256(state["eeg_imputation_mean"])
        assert info["imputed_training_pairs"] == [missing]
    assert not result["validation_access"] and not result["protected_access"]


def test_budget_expiry_precedes_any_cv_or_full_source_fit(monkeypatch):
    pairs, metadata, folds, quality = fixture()
    calls = mock_fit(monkeypatch)
    def expired():
        raise TimeoutError("Declared budget expired")
    with pytest.raises(TimeoutError):
        module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                             quality_rows=quality, feature_loader=lambda p, r: np.zeros(150),
                             target_loader=lambda p: 3.5, check_budget=expired)
    with pytest.raises(TimeoutError):
        module.fit_full_source(source_pairs=pairs, metadata=metadata, quality_rows=quality,
                               feature_loader=lambda p, r: np.zeros(150), target_loader=lambda p: 3.5,
                               candidate_metrics=candidate_metrics(), check_budget=expired)
    assert not calls


@pytest.mark.parametrize("flag", [
    {"channel": "all", "reason": "nonfinite_processed_values"},
    {"channel": "all", "reason": "numerically_unavailable_eeg"},
    {"channel": "all", "reason": "new_unqualified_rule"},
    {"channel": "all", "reason": "high_processed_excursion"},
    {"channel": "not_a_channel", "reason": "adc_rail"},
    {"channel": module.COMMON_CHANNELS[0], "reason": "repair_capacity_exceeded"},
    {"channel": module.COMMON_CHANNELS[0], "reason": "adc_rail", "new": True},
    {"channel": module.COMMON_CHANNELS[0], "reason": []},
    "high_processed_excursion",
])
def test_unknown_numerical_or_malformed_flags_are_fatal_before_callbacks(flag):
    pairs, metadata, folds, quality = fixture()
    quality[-1]["review_flags"] = [flag]
    with pytest.raises((PermissionError, ValueError)):
        module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                             quality_rows=quality, feature_loader=bomb, target_loader=bomb)


def test_allowed_repair_capacity_flag_masks_whole_trial_without_threshold_changes():
    pairs, metadata, _, quality = fixture()
    quality[-1]["review_flags"] = [{"channel": "all", "reason": "repair_capacity_exceeded"}]
    summary, availability = module.qualify_quality(pairs, quality)
    assert summary["unavailable_pairs"] == [pairs[-1]]
    assert not availability[pairs[-1]] and not summary["qc_thresholds_changed"]


def test_region_availability_disagreement_with_flags_is_fatal():
    pairs, metadata, folds, quality = fixture()
    mask(quality, pairs[-1], "late")
    quality[-1]["eeg_available"] = True
    with pytest.raises(PermissionError, match="disagrees"):
        module.run_source_cv(source_pairs=pairs, metadata=metadata, person_folds=folds,
                             quality_rows=quality, feature_loader=bomb, target_loader=bomb)


def test_real_synthetic_imputed_svd_preserves_matched_context_scaling():
    pairs, metadata, folds, _ = fixture()
    cell = crossed_source_cells(pairs, metadata, folds)[0]
    missing = cell["training"][0]
    availability = {p: p != missing for p in pairs}
    contexts = {p: module.context_features(metadata[p]) for p in pairs}
    rng = np.random.default_rng(13)
    eeg = {p: rng.normal(size=150) for p in pairs if availability[p]}
    joint, mean, info = module.training_imputation(cell["training"], contexts, eeg, availability)
    maps = {"context_only": contexts, "late_joint": joint, "distributed_joint": joint}
    targets = {p: float(p[1] % 8) for p in pairs}
    with threadpool_limits(limits=2, user_api="blas"):
        fitted = {m: module.fit_source_ridge(cell["training"], features, targets, 100.) for m, features in maps.items()}
    module._assert_matched_scalers(fitted)
    assert info["training_rows_retained"] == 448 and info["donor_rows"] == 447
    assert np.array_equal(joint[missing][13:], mean)
