"""Independent loss-weighting audit with manufactured outcomes only."""
import pytest

from scripts.audit_faced_coverage_results_2026_10_13 import close, identity_hash, loss_metrics


def row(person, clip, family, error):
    return {"subject": person, "vid": clip, "family": family,
            "target": 0., "prediction": float(error)}


def test_family_and_participant_weighting_differs_from_naive_clip_mean():
    rows = [row("a", 1, "x", 1), row("a", 2, "x", 3), row("a", 3, "y", 6),
            row("b", 1, "x", 1), row("b", 3, "y", 1)]
    metrics = loss_metrics(rows)
    assert metrics["participant_title_mae"] == 2.5
    assert metrics["trial_mse"] == 9.6
    assert metrics["prediction_rows"] == 5


def test_duplicate_source_trial_cannot_change_weighting():
    with pytest.raises(ValueError, match="Duplicate"):
        loss_metrics([row("a", 1, "x", 1), row("a", 1, "x", 2)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., 7.1])
def test_invalid_emitted_prediction_holds_audit(bad):
    with pytest.raises(ValueError):
        loss_metrics([row("a", 1, "x", bad)])


def test_numeric_audit_rejects_more_than_serialization_tolerance():
    close(1., 1. + 5e-13)
    with pytest.raises(ValueError):
        close(1., 1. + 1e-8)


def test_identity_hash_depends_on_training_order_and_membership():
    a, b = [("a", 1), ("b", 2)], [("b", 2), ("a", 1)]
    assert identity_hash(a) != identity_hash(b)
    assert identity_hash(a) == identity_hash([list(p) for p in a])
