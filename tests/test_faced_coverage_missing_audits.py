"""Manufactured training-only mean replay and donor-scope audit fixtures."""
import numpy as np
import pytest

from scripts.audit_faced_coverage_missing_qualification_2026_10_13 import (
    array_sha, donor_means, fixture_check, identity_sha,
)
from scripts.audit_faced_coverage_missing_results_2026_10_13 import imputation_record


def fixture():
    pairs = [("a", 2), ("b", 2), ("held", 3)]
    vectors = np.stack([np.stack([np.full(150, x + r) for r in (0., 3., 6.)])
                        for x in (1., 5., 1e12)])
    training = [("a", 2), ("missing", 20), ("b", 2)]
    return vectors, pairs, training


def test_available_training_only_means_ignore_extreme_held_and_missing():
    vectors, pairs, training = fixture()
    donors, missing, means = donor_means(vectors, pairs, training)
    assert donors == [("a", 2), ("b", 2)] and missing == [("missing", 20)]
    assert np.array_equal(means["late_joint"], np.full(150, 9.))
    assert np.array_equal(means["distributed_joint"], np.full(150, 6.))


def test_independent_fixture_hashes_bind_mean_and_donor_order():
    vectors, pairs, training = fixture()
    donors, missing, means = donor_means(vectors, pairs, training)
    declared = {"cell": 0, "eligible_train_donor_pairs": donors, "donor_count": 2,
                "eligible_train_donor_identity_sha256": identity_sha(donors),
                "imputed_training_pairs": missing,
                "mean_sha256": {m: array_sha(v) for m, v in means.items()}}
    assert fixture_check(declared, vectors, pairs, training, 0)["imputed_rows"] == 1
    declared["mean_sha256"]["late_joint"] = "wrong"
    with pytest.raises(ValueError, match="numerical"):
        fixture_check(declared, vectors, pairs, training, 0)


def test_missing_all_donors_stops_before_any_mean():
    vectors, pairs, _ = fixture()
    with pytest.raises(PermissionError):
        donor_means(vectors, pairs, [("missing", 20)])


def test_duplicate_training_and_bad_source_geometry_rejected():
    vectors, pairs, _ = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        donor_means(vectors, pairs, [pairs[0], pairs[0]])
    with pytest.raises(ValueError, match="vectors"):
        donor_means(vectors[:, :, :149], pairs, [pairs[0]])


def test_numerical_replay_matches_declared_numpy_means_on_manufactured_data():
    rng = np.random.default_rng(813001)
    vectors = rng.normal(size=(19, 3, 150))
    pairs = [("source", i) for i in range(19)]
    training = pairs[:15]
    _, _, means = donor_means(vectors, pairs, training)
    assert array_sha(means["late_joint"]) == array_sha(vectors[:15, 2, :].mean(axis=0))
    assert array_sha(means["distributed_joint"]) == array_sha(vectors.mean(axis=1)[:15].mean(axis=0))


@pytest.mark.parametrize("donor_count", [447, 448, 979])
def test_actual_geometry_hashes_match_root_fixture_without_rounding(donor_count):
    from scripts.run_faced_coverage_missing_2026_10_13 import mean_fixtures
    rng = np.random.default_rng(813001 + donor_count)
    vectors = rng.normal(size=(donor_count + 7, 3, 150)).astype(np.float64)
    pairs = [("source", i) for i in range(donor_count + 7)]
    training = pairs[:donor_count] + [("missing", 20)]
    fixtures = mean_fixtures(vectors, pairs, [{"cell": 0, "training": training}])
    replay = fixture_check(fixtures["cells"][0], vectors, pairs, training, 0)
    assert replay["donor_rows"] == donor_count
    assert replay["mean_hashes_independently_recomputed"] == fixtures["cells"][0]["mean_sha256"]
    assert fixture_check(fixtures["full_source"], vectors, pairs, sorted(training), None)["donor_rows"] == donor_count


def test_result_mean_manifest_preserves_missing_rows_and_excludes_held_donors():
    vectors, pairs, training = fixture()
    donors, missing, means = donor_means(vectors, pairs, training)
    availability = {p: p in donors for p in training}
    mean = {"donor_pairs": donors, "donor_rows": 2, "missing_training_pairs": missing,
            "missing_training_rows": 1, "training_rows_retained": 3, "eeg_dimension": 150,
            "donor_identity_sha256": identity_sha(donors), "mean_sha256": array_sha(means["late_joint"]),
            "mean_uses_only_available_training_rows": True, "held_rows_used_for_imputation": 0,
            "target_values_used_for_imputation": 0}
    record = {"model": "late_joint", "eligible_train_donor_pairs": donors,
              "eligible_train_donor_identity_sha256": identity_sha(donors),
              "shared_available_training_rows": 2, "shared_missing_training_pairs": missing,
              "imputation": mean, "imputed_training_pairs": missing, "imputed_training_rows": 1}
    declared = {"mean_sha256": {m: array_sha(v) for m, v in means.items()}}
    assert imputation_record(record, training, availability, declared) == 1
    mean["held_rows_used_for_imputation"] = 1
    with pytest.raises(ValueError, match="imputation"):
        imputation_record(record, training, availability, declared)


def test_context_receipt_cannot_claim_eeg_mean_imputation():
    training = [("a", 2), ("missing", 20)]
    availability = {training[0]: True, training[1]: False}
    record = {"model": "context_only", "eligible_train_donor_pairs": training[:1],
              "eligible_train_donor_identity_sha256": identity_sha(training[:1]),
              "shared_available_training_rows": 1, "shared_missing_training_pairs": training[1:],
              "imputation": None, "imputed_training_pairs": [], "imputed_training_rows": 0}
    assert imputation_record(record, training, availability, {}) == 0
    record["imputed_training_rows"] = 1
    with pytest.raises(ValueError, match="Context"):
        imputation_record(record, training, availability, {})
