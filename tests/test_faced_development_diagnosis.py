"""Synthetic identity, fold, specificity-control and aggregation regressions."""
from pathlib import Path

import numpy as np
import pytest

from gruxnet.faced_cached_targets import cached_original_target
from gruxnet.faced_development_diagnosis import (
    ARMS, FIT_CAP, OLD_NEURAL, REPO, SCHEMES, average_raw_correction,
    balanced_decomposition, crossed_loss_interval, crossed_source_folds, donor_assignments,
    mixed_joint_features, persist_partial_trace, require_replay_gate, residual_swap_predictions,
    repeat_loss_summary, run,
)


def source_fixture():
    people = [f"source-{i:03d}" for i in range(70)]
    clips = list(range(1, 9))
    pairs = [(p, v) for p in people for v in clips]
    metadata = {p: {"role": "source", "family": f"film-{p[1]}"} for p in pairs}
    folds = [people[i * 14:(i + 1) * 14] for i in range(5)]
    return pairs, metadata, folds


def validation_fixture():
    pairs = [(p, v) for p in ("A", "B", "C") for v in (1, 2, 3, 4)]
    metadata = {p: {"role": "validation", "family": f"film-{p[1]}",
                    "acquisition_group": "common" if p[0] != "C" else "singleton",
                    "assigned_valence": "positive" if p[1] < 3 else "negative"} for p in pairs}
    decisions = {p: {"eeg_available": True} for p in pairs}
    return pairs, metadata, decisions


def test_crossed_folds_exclude_both_axes_and_other_quadrants():
    pairs, metadata, people_folds = source_fixture()
    result = crossed_source_folds(pairs, metadata, people_folds)
    assert len(result) == 20
    all_held = []
    for fold in result:
        held_people = {p[0] for p in fold["held"]}
        held_films = {metadata[p]["family"] for p in fold["held"]}
        assert len(fold["training"]) == 336 and len(fold["held"]) == 28
        assert not held_people & {p[0] for p in fold["training"]}
        assert not held_films & {metadata[p]["family"] for p in fold["training"]}
        assert all(p[0] not in held_people and metadata[p]["family"] not in held_films for p in fold["training"])
        all_held.extend(fold["held"])
    assert sorted(all_held) == sorted(pairs)
    assert result == crossed_source_folds(pairs, metadata, people_folds)


def test_crossed_folds_reject_protected_and_duplicate_person_fold():
    pairs, metadata, folds = source_fixture()
    protected = dict(metadata)
    protected[pairs[0]] = {**protected[pairs[0]], "role": "confirmation"}
    with pytest.raises(PermissionError):
        crossed_source_folds(pairs, protected, folds)
    folds[1][0] = folds[0][0]
    with pytest.raises(PermissionError):
        crossed_source_folds(pairs, metadata, folds)


def test_balanced_variance_additive_fixture_and_interaction():
    pairs = [(p, v) for p in ("A", "B") for v in (1, 2)]
    metadata = {p: {"family": f"film-{p[1]}"} for p in pairs}
    values = {("A", 1): 1., ("A", 2): 3., ("B", 1): 3., ("B", 2): 5.}
    result = balanced_decomposition(pairs, values, metadata)
    assert result["total_ss"] == 8.
    assert result["fractions"] == {"person": .5, "title": .5, "additive_residual": 0.}
    interaction = {("A", 1): 1., ("A", 2): 5., ("B", 1): 5., ("B", 2): 1.}
    assert balanced_decomposition(pairs, interaction, metadata)["fractions"]["additive_residual"] == 1.
    assert "not measurement noise" in result["interpretation"]
    with pytest.raises(PermissionError):
        balanced_decomposition(pairs[:-1], values, metadata)


def test_person_swaps_strict_derangement_singletons_and_unavailable():
    pairs, metadata, decisions = validation_fixture()
    decisions["A", 4]["eeg_available"] = False
    a = donor_assignments(pairs, metadata, decisions, SCHEMES[0], 44)
    assert a == donor_assignments(pairs, metadata, decisions, SCHEMES[0], 44)
    for p, row in a.items():
        donor = row["donor"]
        assert donor in pairs
        if row["eligible"]:
            assert donor != p and donor[1] == p[1]
            assert metadata[donor]["acquisition_group"] == metadata[p]["acquisition_group"]
            assert decisions[donor]["eeg_available"]
        else:
            assert donor == p
    assert all(not a["C", v]["eligible"] for v in range(1, 5))
    assert a["A", 4]["reason"] == "unavailable_eeg"
    assert a["B", 4]["reason"] == "singleton_stratum"


def test_film_swaps_preserve_person_and_coarse_valence_only():
    pairs, metadata, decisions = validation_fixture()
    assignments = donor_assignments(pairs, metadata, decisions, SCHEMES[1], 55)
    assert all(r["eligible"] for r in assignments.values())
    for p, row in assignments.items():
        donor = row["donor"]
        assert donor[0] == p[0] and donor[1] != p[1]
        assert metadata[donor]["assigned_valence"] == metadata[p]["assigned_valence"]


def test_pairing_rejects_source_or_protected_donor_scope():
    pairs, metadata, decisions = validation_fixture()
    metadata[pairs[0]]["role"] = "source"
    with pytest.raises(PermissionError):
        donor_assignments(pairs, metadata, decisions, SCHEMES[0], 1)


def test_joint_swap_uses_recipient_context_and_donor_eeg():
    pairs = [("A", 1), ("B", 1)]
    decisions = {p: {"eeg_available": True} for p in pairs}
    context = {pairs[0]: np.array([10., 1.]), pairs[1]: np.array([99., 2.])}
    spectral = {pairs[0]: np.array([3., 4.]), pairs[1]: np.array([5., 6.])}
    donors = {pairs[0]: {"donor": pairs[1]}, pairs[1]: {"donor": pairs[0]}}
    mixed = mixed_joint_features(pairs, donors, context, spectral, decisions)
    np.testing.assert_array_equal(mixed[pairs[0]], [10., 1., 5., 6.])
    np.testing.assert_array_equal(mixed[pairs[1]], [99., 2., 3., 4.])
    decisions[pairs[1]]["eeg_available"] = False
    with pytest.raises(PermissionError):
        mixed_joint_features(pairs, donors, context, spectral, decisions)


def test_neural_swap_raw_ensemble_before_clipping_and_fixed_fallback():
    pairs = [("A", 1), ("B", 1), ("F", 1)]
    decisions = {p: {"eeg_available": p[0] != "F"} for p in pairs}
    offsets = {pairs[0]: 1., pairs[1]: 2., pairs[2]: 6.}
    # Mean raw prediction is four, its original offset two: correction two.
    correction = average_raw_correction([-1., 9.], 2.)
    assert correction == 2.
    corrections = {pairs[0]: -4., pairs[1]: correction}
    donors = {pairs[0]: {"donor": pairs[1]}, pairs[1]: {"donor": pairs[0]}, pairs[2]: {"donor": pairs[0]}}
    result = residual_swap_predictions(pairs, donors, offsets, corrections, decisions)
    assert result == {pairs[0]: 3., pairs[1]: 0., pairs[2]: 6.}
    assert result[pairs[0]] != np.mean(np.clip([-1., 9.], 0, 7)) - 2. + 1.


def test_repeat_losses_are_not_loss_of_averaged_predictions():
    original = [{"subject": "A", "vid": 1, "family": "F", "target": 3.5, "prediction": 3.5}]
    rows = [[{**original[0], "prediction": 0.}], [{**original[0], "prediction": 7.}]]
    report = repeat_loss_summary(original, rows, {("A", 1): True})
    assert report["mean_full_delta"] == 3.5
    assert report["mean_eligible_delta"] == 3.5
    assert abs(original[0]["target"] - np.mean([0., 7.])) == 0.
    assert "No averaged-prediction" in report["loss_aggregation"]


class FloatBomb:
    def __float__(self):
        raise AssertionError("Protected target was numerically interpreted")


def test_identity_guard_before_target_conversion():
    metadata = {("allowed", 1): {"family": "F", "acquisition_group": "A"}}
    row = {"subject": "protected", "vid": "1", "family": "F", "target": FloatBomb()}
    with pytest.raises(PermissionError):
        cached_original_target(row, metadata, {}, "quality")


def test_gate_failure_before_any_numeric_cache_or_target_open(monkeypatch):
    import gruxnet.faced_development_diagnosis as module
    def forbidden():
        raise AssertionError("Numeric cache loader was reached")
    monkeypatch.setattr(module, "metadata_bundle", forbidden)
    activation = {"bindings": {}, "new_fit_cap": FIT_CAP, "time_cap_seconds": 1800,
                  "confirmation_access": False, "research_question_changed": False,
                  "new_original_rating_decoding": 0, "original_data_access": False}
    tmp_path = Path(__file__).resolve().parent / "synthetic_gate_must_not_create_directory"
    with pytest.raises(PermissionError, match="source-fixture"):
        run(tmp_path / "public", tmp_path / "private", activation, None)
    assert not (tmp_path / "private").exists()


def test_full_panel_crossed_loss_interval_is_mean_repeat_loss():
    original = [{"subject": s, "vid": v, "family": f"film-{v}", "target": 3.5, "prediction": 3.5} for s in ("A", "B") for v in (1, 2)]
    repeats = [[{**r, "prediction": value} for r in original] for value in (0., 7.)]
    result = crossed_loss_interval(original, repeats, draws=20)
    assert result["mean_loss_delta"] == 3.5
    assert result["descriptive_95_percent_interval"] == [3.5, 3.5]
    incomplete = original[:-1]
    with pytest.raises(PermissionError):
        crossed_loss_interval(incomplete, [[{**r, "prediction": 0.} for r in incomplete]])


def test_gate_binds_exact_checkpoint_csv():
    name = (OLD_NEURAL / "checkpoint_predictions.csv").relative_to(REPO).as_posix()
    activation = {"bindings": {name: "expected"}}
    gate = {"passed": True, "source_fixture_replay_passed": True,
            "validation_forward_replay_performed": False, "confirmation_access": False,
            "checkpoint_csv_sha256": "expected"}
    assert require_replay_gate(gate, activation) == gate
    with pytest.raises(PermissionError):
        require_replay_gate({**gate, "checkpoint_csv_sha256": "other"}, activation)


def test_failed_attempt_retains_partial_fit_and_row_trace_without_resume(monkeypatch):
    from types import SimpleNamespace
    import gruxnet.faced_development_diagnosis as module
    captured = {}
    monkeypatch.setattr(module, "save", lambda path, value: captured.update({Path(path).name: value}))
    monkeypatch.setattr(module, "_write_csv", lambda path, rows: captured.update({Path(path).name: list(rows)}))
    ledger = SimpleNamespace(records=[{"index": 0, "source_pairs": [("source", 1)], "private_state_name": "model_000.npz"}])
    row = {"subject": "source", "vid": 1}
    persist_partial_trace(Path("synthetic_failed_attempt"), ledger, [row], [], [], [row])
    assert set(captured) == {"partial_model_manifest.json", "partial_cv_predictions.csv", "partial_swapped_predictions.csv"}
    assert captured["partial_model_manifest.json"]["fits"] == ledger.records
    assert captured["partial_model_manifest.json"]["resumption_authorized"] is False
