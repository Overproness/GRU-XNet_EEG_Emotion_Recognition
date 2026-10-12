"""Synthetic outcome tests; actual reads are limited to public reservation metadata."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gruxnet.faced_confirmation_contract import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, PREDICTION_MODELS, canonical_sha256, guard_source_records,
    guard_trial, guarded_read, metadata_builder, paired_loss_analysis,
    prediction_with_fallback, require_confirmation_activation, require_source_integrity,
    require_prediction_lock, prediction_only_projection,
)
from gruxnet.faced_inference_quality import POLICY_ID

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen():
    return metadata_builder(REPO)


def synthetic_authorization(contract, *, with_prediction_lock=True, prediction_rows=None):
    """Manufactured passing receipts; never a real author approval or data gate."""
    digest = canonical_sha256(contract)
    receipts = {
        "synthetic/precision.json": {"passed": True, "contract_semantic_sha256": digest,
                                     "confirmation_outcomes_accessed": False},
        "synthetic/source.json": {
            "passed": True, "source_identity_sha256": contract["source_identity_sha256"],
            "source_trials": 980, "source_failures": 0, "policy_id": POLICY_ID,
            "complete_source_feature_cache_qualification_passed": True,
            "validation_people_used_for_fitting": False, "confirmation_access": False,
        },
        "synthetic/continuation.json": {"passed": True, "contract_semantic_sha256": digest,
                                        "empirical_continuation_requirement_met": True,
                                        "confirmation_outcomes_accessed": False},
        "synthetic/state.json": {
            "passed": True, "contract_semantic_sha256": digest,
            "source_identity_sha256": contract["source_identity_sha256"],
            "joint_and_matched_context_states_qualified": True,
            "source_only_scaling_and_priors": True, "confirmation_outcomes_accessed": False,
            "ridge_alpha": 100.0, "matched_context_dimension": 13, "joint_dimension": 163,
            "no_validation_targets_in_fit": True,
        },
    }
    blobs = {path: (json.dumps(value, sort_keys=True) + "\n").encode()
             for path, value in receipts.items()}
    gate = lambda path: {"passed": True, "receipt_path": path,
                         "receipt_sha256": hashlib.sha256(blobs[path]).hexdigest()}
    activation = {
        "status": "active_confirmation", "contract_semantic_sha256": digest,
        "author_question_approval": True, "confirmation_access_approval": True,
        "precision_gate": gate("synthetic/precision.json"),
        "source_integrity": gate("synthetic/source.json"),
        "original_no_lead_gate": gate("synthetic/continuation.json"),
        "final_state_qualification": gate("synthetic/state.json"),
    }
    if with_prediction_lock:
        prediction_identities = {arm: contract["arms"][arm]["pairs"]
                                 for arm in ("confirmation_unseen", "confirmation_familiar")}
        files = []
        for model in PREDICTION_MODELS:
            path = "synthetic/" + model + ".prediction-bytes"
            blobs[path] = ("Manufactured opaque prediction fixture for " + model).encode()
            files.append({"model": model, "path": path,
                          "sha256": hashlib.sha256(blobs[path]).hexdigest(),
                          "identity_sha256": canonical_sha256(prediction_identities),
                          "primary_trials": 297, "familiar_trials": 462})
        lock = {
            "passed": True, "contract_semantic_sha256": digest,
            "source_identity_sha256": contract["source_identity_sha256"],
            "final_state_receipt_sha256": activation["final_state_qualification"]["receipt_sha256"],
            "predictions_locked": True, "protected_rating_values_decoded_before_lock": 0,
            "prediction_identities": prediction_identities, "models": list(PREDICTION_MODELS),
            "prediction_files": files,
            "prediction_only_projection_sha256": {
                arm: canonical_sha256(prediction_only_projection(
                    prediction_rows if prediction_rows is not None and prediction_rows[0]["arm"] == arm
                    else numerical_rows(contract, arm), arm, contract))
                for arm in prediction_identities
            },
        }
        blobs["synthetic/lock.json"] = (json.dumps(lock, sort_keys=True) + "\n").encode()
        activation["prediction_lock"] = gate("synthetic/lock.json")
    return activation, blobs


def identities(contract, arm):
    title = {row["vid"]: row["family"] for row in contract["catalogue"]}
    role = contract["arms"][arm]["role"]
    return [{**pair, "arm": arm, "role": role, "family": title[pair["vid"]]}
            for pair in contract["arms"][arm]["pairs"]]


class ExplodingNumber:
    def __float__(self):
        raise AssertionError("Forbidden numeric outcome was converted before identity/access guard")


def test_metadata_freezes_exact_roles_and_honest_material_scope(frozen):
    assert frozen["status"] == "frozen_not_activated"
    assert frozen["research_question_adopted"] is False
    assert frozen["confirmation_access"] is False
    assert frozen["original_recordings_opened"] == frozen["original_ratings_decoded"] == 0
    assert {arm: len(packet["pairs"]) for arm, packet in frozen["arms"].items()} == {
        "source_fit": 980, "confirmation_unseen": 297, "confirmation_familiar": 462}
    assert all(value is False for value in frozen["currently_passed_confirmation_gates"].values())
    assert frozen["material_scope"]["original_footage_overlap_authenticated"] is False
    assert frozen["material_scope"]["source_titles_per_category"]["Fear"] == 1
    assert frozen["material_scope"]["source_titles_per_category"]["Inspiration"] == 1
    assert frozen["material_scope"]["source_material_cv_all_categories_supported"] is False
    assert len(frozen["policy_code_bindings"]) == 2


def test_inactive_protected_read_refuses_without_invoking_reader(frozen):
    row = identities(frozen, "confirmation_unseen")[0]
    row["target"] = ExplodingNumber()
    with pytest.raises(PermissionError, match="inactive"):
        guarded_read(row, "confirmation_unseen", frozen,
                     lambda _: pytest.fail("Inactive decoder was invoked"))


def test_full_generic_approval_still_cannot_read_ratings_before_prediction_lock(frozen):
    activation, blobs = synthetic_authorization(frozen, with_prediction_lock=False)
    row = identities(frozen, "confirmation_unseen")[0]
    with pytest.raises(PermissionError, match="prediction lock"):
        guarded_read(row, "confirmation_unseen", frozen,
                     lambda _: pytest.fail("Unlocked rating decoder was invoked"),
                     activation=activation, bound_receipts=blobs,
                     read_kind="confirmation_rating")
    with pytest.raises(PermissionError, match="access phase"):
        guarded_read(row, "confirmation_unseen", frozen,
                     lambda _: pytest.fail("Unspecified protected backend was invoked"),
                     activation=activation, bound_receipts=blobs)


def test_signal_phase_does_not_require_rating_lock_and_valid_locked_rating_phase_proceeds(frozen):
    activation, blobs = synthetic_authorization(frozen, with_prediction_lock=False)
    row = identities(frozen, "confirmation_unseen")[0]
    pair = guarded_read(row, "confirmation_unseen", frozen, lambda pair: pair,
                        activation=activation, bound_receipts=blobs,
                        read_kind="confirmation_signal")
    assert pair == (row["subject"], row["vid"])
    activation, blobs = synthetic_authorization(frozen)
    assert guarded_read(row, "confirmation_unseen", frozen, lambda _: "synthetic rating",
                        activation=activation, bound_receipts=blobs,
                        read_kind="confirmation_rating") == "synthetic rating"


def test_prediction_bytes_and_complete_identities_are_locked_before_outcome_conversion(frozen):
    activation, blobs = synthetic_authorization(frozen)
    blobs["synthetic/spectral_joint_ridge.prediction-bytes"] += b"tampered"
    with pytest.raises(PermissionError, match="prediction bytes"):
        require_prediction_lock(frozen, activation, blobs)
    activation, blobs = synthetic_authorization(frozen)
    lock = json.loads(blobs["synthetic/lock.json"])
    lock["prediction_identities"]["confirmation_unseen"].pop()
    blobs["synthetic/lock.json"] = json.dumps(lock).encode()
    activation["prediction_lock"]["receipt_sha256"] = hashlib.sha256(blobs["synthetic/lock.json"]).hexdigest()
    rows = numerical_rows(frozen)
    rows[0]["target"] = ExplodingNumber()
    with pytest.raises(PermissionError, match="prediction identities"):
        paired_loss_analysis(rows, "confirmation_unseen", frozen,
                             activation=activation, bound_receipts=blobs)


def test_changing_caller_prediction_after_lock_fails_before_target_conversion(frozen):
    activation, blobs = synthetic_authorization(frozen)
    rows = numerical_rows(frozen)
    rows[0]["joint_prediction"] = 0.25
    rows[0]["target"] = ExplodingNumber()
    with pytest.raises(PermissionError, match="prediction-only projection"):
        paired_loss_analysis(rows, "confirmation_unseen", frozen,
                             activation=activation, bound_receipts=blobs)


def test_prediction_projection_excludes_target_and_is_order_invariant(frozen):
    rows = numerical_rows(frozen)
    first = canonical_sha256(prediction_only_projection(rows, "confirmation_unseen", frozen))
    for row in rows:
        row["target"] = ExplodingNumber()
    assert canonical_sha256(prediction_only_projection(rows[::-1], "confirmation_unseen", frozen)) == first


@pytest.mark.parametrize("field,value", [("role", "source"), ("family", "wrong-title"),
                                         ("arm", "confirmation_familiar"), ("vid", 13),
                                         ("subject", "sub-not-reserved"), ("vid", True)])
def test_invalid_identity_arm_or_role_guard_precedes_numeric_access(frozen, field, value):
    row = identities(frozen, "confirmation_unseen")[0]
    row.update({field: value, "target": ExplodingNumber()})
    activation, blobs = synthetic_authorization(frozen)
    with pytest.raises(PermissionError):
        guarded_read(row, "confirmation_unseen", frozen,
                     lambda _: float(row["target"]), activation=activation, bound_receipts=blobs)


@pytest.mark.parametrize("gate", ["author_question_approval", "confirmation_access_approval",
                                  "precision_gate", "source_integrity", "original_no_lead_gate",
                                  "final_state_qualification"])
def test_each_independent_approval_or_gate_is_mandatory(frozen, gate):
    activation, blobs = synthetic_authorization(frozen)
    activation[gate] = False if gate.endswith("approval") else {"passed": False}
    with pytest.raises(PermissionError):
        require_confirmation_activation(frozen, activation, blobs)


def test_passing_precision_cannot_override_original_empirical_hold(frozen):
    activation, blobs = synthetic_authorization(frozen)
    activation["original_no_lead_gate"]["passed"] = False
    with pytest.raises(PermissionError, match="Original empirical"):
        require_confirmation_activation(frozen, activation, blobs)


def test_receipt_bytes_and_contract_must_be_bound(frozen):
    activation, blobs = synthetic_authorization(frozen)
    blobs["synthetic/precision.json"] += b" "
    with pytest.raises(PermissionError, match="receipt bytes"):
        require_confirmation_activation(frozen, activation, blobs)
    activation, blobs = synthetic_authorization(frozen)
    activation["contract_semantic_sha256"] = "0" * 64
    with pytest.raises(PermissionError):
        require_confirmation_activation(frozen, activation, blobs)


def test_source_requires_exact_complete_identities_and_clean_cache_qc(frozen):
    activation, blobs = synthetic_authorization(frozen)
    rows = identities(frozen, "source_fit")
    assert len(guard_source_records(rows, frozen, activation["source_integrity"], blobs)) == 980
    rows[0]["target"] = ExplodingNumber()
    with pytest.raises(PermissionError, match="980"):
        guard_source_records(rows[:-1], frozen, activation["source_integrity"], blobs)
    with pytest.raises(PermissionError, match="980"):
        guard_source_records(rows[:-1] + [rows[1]], frozen, activation["source_integrity"], blobs)
    activation["source_integrity"]["passed"] = False
    with pytest.raises(PermissionError):
        guarded_read(rows[0], "source_fit", frozen, lambda _: float(rows[0]["target"]),
                     source_gate=activation["source_integrity"], bound_receipts=blobs)


def test_source_gate_will_not_substitute_fixture_replay_for_complete_cache_qualification(frozen):
    activation, blobs = synthetic_authorization(frozen)
    receipt = json.loads(blobs["synthetic/source.json"])
    receipt.pop("complete_source_feature_cache_qualification_passed")
    receipt["source_fixture_replay_passed"] = True
    blobs["synthetic/source.json"] = json.dumps(receipt).encode()
    activation["source_integrity"]["receipt_sha256"] = hashlib.sha256(blobs["synthetic/source.json"]).hexdigest()
    with pytest.raises(PermissionError, match="cache"):
        require_source_integrity(frozen, activation["source_integrity"], blobs)


def test_unavailable_eeg_keeps_category_for_both_models_without_backend_calls(frozen):
    activation, blobs = synthetic_authorization(frozen)
    row = identities(frozen, "confirmation_unseen")[0]
    decision = {"policy_id": POLICY_ID, "eeg_available": False,
                "reasons": [{"channel": "all", "reason": "stationary_counts"}]}
    result = prediction_with_fallback(
        row, decision, lambda: 3.25,
        lambda: pytest.fail("Unavailable trial invoked matched-context model"),
        lambda: pytest.fail("Unavailable trial invoked EEG model"), frozen,
        activation=activation, bound_receipts=blobs)
    assert result["category_prediction"] == result["joint_prediction"] == result["matched_context_prediction"] == 3.25
    assert result["fallback"] is True
    assert result["subject"] == row["subject"] and result["vid"] == row["vid"]


def test_protected_inference_helper_rejects_source_before_any_callback(frozen):
    activation, blobs = synthetic_authorization(frozen)
    row = identities(frozen, "source_fit")[0]
    decision = {"policy_id": POLICY_ID, "eeg_available": True, "reasons": []}
    callback = lambda: pytest.fail("Protected inference helper invoked a source callback")
    with pytest.raises(PermissionError, match="source rows"):
        prediction_with_fallback(row, decision, callback, callback, callback, frozen,
                                 activation=activation, bound_receipts=blobs)


def numerical_rows(frozen, arm="confirmation_unseen"):
    # Every supplied number is manufactured; no cached/original outcomes read.
    return [{**row, "target": 0.0, "category_prediction": 1.0,
             "joint_prediction": 0.0, "matched_context_prediction": 2.0,
             "eeg_available": True, "fallback": False}
            for row in identities(frozen, arm)]


def test_clip_losses_precede_title_averaging_and_titles_receive_equal_weight(frozen):
    activation, blobs = synthetic_authorization(frozen)
    rows = numerical_rows(frozen)
    # Two shared-title clips: averaging predictions first would produce a
    # different error. Their loss mean=2, seven other titles have loss=1.
    for row in rows:
        row["target"] = 3.0
        row["category_prediction"] = 4.0
        row["matched_context_prediction"] = 5.0
        row["joint_prediction"] = 3.0
        if row["vid"] in (7, 8):
            row["category_prediction"] = 1.0 if row["vid"] == 7 else 5.0
    activation, blobs = synthetic_authorization(frozen, prediction_rows=rows)
    result = paired_loss_analysis(rows, "confirmation_unseen", frozen,
                                  activation=activation, bound_receipts=blobs)
    assert result["category_mae"] == pytest.approx(9 / 8)
    assert result["paired_gain"] == pytest.approx(9 / 8)
    assert result["mandatory_matched_context_gain"] == pytest.approx(2)
    assert result["trials"] == 297 and result["catalogue_title_families"] == 8
    assert result["crossed_resamples"] == BOOTSTRAP_DRAWS == 4000
    assert result["seed"] == BOOTSTRAP_SEED == 425401


def test_complete_denominator_guard_before_any_target_conversion(frozen):
    activation, blobs = synthetic_authorization(frozen)
    rows = numerical_rows(frozen)
    rows[0]["target"] = ExplodingNumber()
    with pytest.raises(PermissionError, match="denominator"):
        paired_loss_analysis(rows[:-1], "confirmation_unseen", frozen,
                             activation=activation, bound_receipts=blobs)
    with pytest.raises(PermissionError, match="denominator"):
        paired_loss_analysis(rows[:-1] + [rows[1]], "confirmation_unseen", frozen,
                             activation=activation, bound_receipts=blobs)


def test_unavailable_rows_remain_in_full_denominator_and_have_zero_matched_gain(frozen):
    activation, blobs = synthetic_authorization(frozen)
    rows = numerical_rows(frozen)
    for row in rows:
        if row["vid"] in (7, 8):
            row.update({"joint_prediction": 1.0, "matched_context_prediction": 1.0,
                        "eeg_available": False, "fallback": True,
                        "fallback_policy": "source_category_median"})
    activation, blobs = synthetic_authorization(frozen, prediction_rows=rows)
    result = paired_loss_analysis(rows, "confirmation_unseen", frozen,
                                  activation=activation, bound_receipts=blobs)
    assert result["trials"] == 297 and result["unavailable_eeg_fallback_trials"] == 66
    assert result["paired_gain"] == pytest.approx(7 / 8)
    assert result["mandatory_matched_context_gain"] == pytest.approx(14 / 8)
    assert result["all_declared_rows_retained"] is True
    rows[0].update({"eeg_available": False, "fallback": True, "fallback_policy": "source_category_median"})
    with pytest.raises(PermissionError, match="Unavailable EEG"):
        paired_loss_analysis(rows, "confirmation_unseen", frozen,
                             activation=activation, bound_receipts=blobs)


@pytest.mark.parametrize("value", [np.nan, -0.1, 7.1, True])
def test_invalid_target_stops_without_outcome_derived_drop(frozen, value):
    activation, blobs = synthetic_authorization(frozen)
    rows = numerical_rows(frozen)
    rows[0]["target"] = value
    with pytest.raises(ValueError):
        paired_loss_analysis(rows, "confirmation_unseen", frozen,
                             activation=activation, bound_receipts=blobs)


def test_contract_never_claims_footage_independence_or_equivalence(frozen):
    changed = copy.deepcopy(frozen)
    changed["material_scope"]["original_footage_overlap_authenticated"] = True
    with pytest.raises(PermissionError, match="Footage"):
        guard_trial(identities(frozen, "source_fit")[0], "source_fit", changed)
    assert "No equivalence" in frozen["analysis"]["interpretation"]
    assert "fixed catalogue" in frozen["analysis"]["interval"]
