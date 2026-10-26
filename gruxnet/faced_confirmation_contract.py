"""Inactive FACED confirmation contract; no recording, target or model reader.

``metadata_builder`` reads four already-public metadata files. Every numerical
helper otherwise accepts caller-supplied, identity-guarded synthetic/future data.
This module neither adopts a manuscript question nor authorizes confirmation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .faced_inference_quality import POLICY_ID

DEVELOPMENT_CLIPS = (2, 3, 4, 6, 9, 11, 12, 17, 18, 20, 23, 25, 27, 28)
CONFIRMATION_CLIPS = (1, 5, 7, 8, 10, 19, 21, 24, 26)
ARMS = ("source_fit", "confirmation_unseen", "confirmation_familiar")
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 425401
PREDICTION_MODELS = ("category_median", "matched_context_ridge", "spectral_joint_ridge")
METADATA_PATHS = (
    "results/development/faced_development_protocol_2026-10-11/plan.json",
    "results/development/faced_backup_2026-10-10/participant_reservation.json",
    "results/development/faced_design_proposal_2026-10-11/recommended_proposal.json",
    "results/development/faced_codebooks_2026-10-10/audit.json",
)


def canonical_sha256(value):
    """Digest JSON meaning, independently of file formatting."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pairs(people, clips):
    return [{"subject": person, "vid": vid}
            for person in sorted(people) for vid in sorted(clips)]


def source_identity_sha256(contract):
    return canonical_sha256(contract["arms"]["source_fit"]["pairs"])


def metadata_builder(repo):
    """Freeze exact public identities/catalogue groups without opening outcomes.

    No source/validation prediction file, original header, rating or signal file
    is read. Title hashes identify catalogue titles, not authenticated footage.
    """
    repo = Path(repo)
    payloads = {name: (repo / name).read_bytes() for name in METADATA_PATHS}
    programme, roles, design, audit = [json.loads(payloads[name])
                                      for name in METADATA_PATHS]
    # Authenticate the historical metadata links carried by the old programme.
    for name in METADATA_PATHS[1:]:
        if programme["bindings"].get(name) != hashlib.sha256(payloads[name]).hexdigest():
            raise PermissionError("Historical participant/material metadata binding changed")
    source = roles["development_source"]
    validation = roles["development_validation"]
    confirmation = roles["confirmation"]
    if (len(source) != 70 or len(validation) != 20 or len(confirmation) != 33
            or len(set(source + validation + confirmation)) != 123):
        raise PermissionError("Participant reservation geometry changed")
    original_pairs = {(r["subject"], r["vid"])
                      for r in programme["input_access"]["future_development_pairs"]}
    if original_pairs != {(p, v) for p in source + validation for v in DEVELOPMENT_CLIPS}:
        raise PermissionError("Prior development rectangle changed")
    for rotation in design["rotations"]:
        clips = rotation["clips"]
        if (tuple(sorted(clips["confirmation"])) != CONFIRMATION_CLIPS
                or set(clips["development_source"] + clips["development_validation"])
                != set(DEVELOPMENT_CLIPS)):
            raise PermissionError("Material reservation changed across rotations")
    catalogue = {r["clip"]: r for r in audit["stimuli"]}
    needed = DEVELOPMENT_CLIPS + CONFIRMATION_CLIPS
    metadata = [{"vid": v, "family": catalogue[v]["source_film_sha256"],
                 "assigned_category": catalogue[v]["assigned_category"],
                 "assigned_valence": catalogue[v]["assigned_valence"],
                 "catalogue_duration_seconds": catalogue[v]["catalogue_duration_seconds"]}
                for v in sorted(needed)]
    title = {r["vid"]: r["family"] for r in metadata}
    if (len({title[v] for v in DEVELOPMENT_CLIPS}) != 14
            or len({title[v] for v in CONFIRMATION_CLIPS}) != 8
            or title[7] != title[8]
            or {title[v] for v in DEVELOPMENT_CLIPS}
            & {title[v] for v in CONFIRMATION_CLIPS}):
        raise PermissionError("Catalogue title separation/counts changed")
    result = {
        "contract_version": "faced-confirmation-information-v1",
        "status": "frozen_not_activated",
        "research_question_adopted": False,
        "confirmation_access": False,
        "original_recordings_opened": 0,
        "original_ratings_decoded": 0,
        "bindings": {p: hashlib.sha256(payloads[p]).hexdigest() for p in METADATA_PATHS},
        "policy_code_bindings": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest()
            for p in ("gruxnet/faced_inference_quality.py", "gruxnet/faced_artifact_control.py")
        },
        "participant_roles": {"source": sorted(source), "validation": sorted(validation),
                              "confirmation": sorted(confirmation)},
        "catalogue": metadata,
        "material_scope": {
            "development_clips": list(DEVELOPMENT_CLIPS),
            "confirmation_clips": list(CONFIRMATION_CLIPS),
            "development_catalogue_title_families": 14,
            "confirmation_catalogue_title_families": 8,
            "catalogue_title_separation_verified": True,
            "original_footage_overlap_authenticated": False,
            "interpretation": "Held-out catalogue title families; footage independence is unverified",
            "source_titles_per_category": {
                category: len({catalogue[v]["source_film_sha256"] for v in DEVELOPMENT_CLIPS
                               if catalogue[v]["assigned_category"] == category})
                for category in sorted({catalogue[v]["assigned_category"] for v in DEVELOPMENT_CLIPS})
            },
            "source_material_cv_all_categories_supported": False,
        },
        "arms": {
            "source_fit": {"role": "source", "pairs": _pairs(source, DEVELOPMENT_CLIPS),
                           "trials": 980, "participants": 70, "title_families": 14},
            "confirmation_unseen": {"role": "confirmation", "pairs": _pairs(confirmation, CONFIRMATION_CLIPS),
                                    "trials": 297, "participants": 33, "title_families": 8},
            "confirmation_familiar": {"role": "confirmation", "pairs": _pairs(confirmation, DEVELOPMENT_CLIPS),
                                      "trials": 462, "participants": 33, "title_families": 14},
        },
        "target": {"item_one_based": 10, "name": "retrospective_whole_clip_valence",
                   "scale": [0, 7], "invalid_rule": "Stop; no target imputation or dropped row"},
        "candidate_predictor": {
            "name": "context_plus_spectral_eeg", "context_features": 13,
            "spectral_features": 150, "alpha": 100.0, "ridge_solver": "svd",
            "intercept": True, "scaler_fit_scope": "Exact 980 source identities only",
            "prediction_clip": [0, 7], "training_trials": 980,
            "category_comparator": "Exact-source assigned-category median, global-source median fallback",
            "mandatory_secondary": "Matched-context Ridge13, alpha100, same exact 980 source identities and source-only scaling",
            "unavailable_eeg_fallback": "Source-only category median comparator; keep trial in every denominator",
            "matched_context_unavailable_eeg_fallback": "Also category median; zero matched EEG gain on unavailable trials",
            "validation_people_promoted_into_source": False,
        },
        "signal_contract": {
            "native_interval_seconds": 32, "common_scalp_channels": 30,
            "processed_sample_rate_hz": 128, "core_interval": "end-30 to end-2 seconds",
            "windows": 7, "window_samples": 512,
            "artifact_policy": "Existing fixed pre-CAR amplitude/interpolation control, at most three channels",
            "artifact_rules": {"pre_car_core_sd_above_uv": 100.0,
                               "pre_car_core_peak_to_peak_above_uv": 1000.0,
                               "maximum_interpolated_channels": 3},
            "availability_policy_id": POLICY_ID,
            "quality_scope": "Fixed engineering rules, not validated artifact labels",
        },
        "analysis": {
            "primary_arm": "confirmation_unseen",
            "secondary_arm": "confirmation_familiar",
            "primary_gain": "Category-median absolute error minus spectral-joint absolute error",
            "primary_information_scope": "Whole-recipe utility; category versus joint differs in context representation and EEG",
            "mandatory_matched_gain": "Matched-context Ridge13 absolute error minus spectral-joint absolute error",
            "weighting": "Mean clip losses within person/title, equal titles within person, equal people",
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "interval": "Paired crossed participant/title resampling, descriptive percentile 95%; fixed catalogue titles, not a random population sample",
            "multiplicity": "One primary comparison; matched-context comparison and familiar materials are descriptive secondary",
            "interpretation": "No equivalence claim; no general EEG-information null or causal claim",
            "full_denominator": True,
        },
        "activation_requirements": ["author_question_approval", "confirmation_access_approval",
                                    "bound_passed_precision_gate", "bound_passed_source_integrity",
                                    "bound_passed_original_no_lead_gate", "bound_passed_final_state_qualification"],
        "protected_target_access_requirement": "All three exact primary/familiar prediction sets byte-locked before decoding any protected rating",
        "future_backend_requirement": "Qualify export bytes to prediction-only projection before locking; no actual export/data backend is implemented here",
        "currently_passed_confirmation_gates": {
            "author_question_approval": False, "confirmation_access_approval": False,
            "precision_gate": False, "source_integrity": False,
            "original_no_lead_gate": False, "final_state_qualification": False,
            "prediction_lock": False,
        },
    }
    result["source_identity_sha256"] = source_identity_sha256(result)
    validate_contract(result)
    return result


def validate_contract(contract):
    """Validate identities using metadata only; never touch a numerical outcome."""
    if (contract.get("status") != "frozen_not_activated"
            or contract.get("research_question_adopted") is not False
            or contract.get("confirmation_access") is not False):
        raise PermissionError("The information contract must remain inactive and unadopted")
    roles = contract["participant_roles"]
    if ([len(roles[r]) for r in ("source", "validation", "confirmation")] != [70, 20, 33]
            or len(set(sum((roles[r] for r in ("source", "validation", "confirmation")), []))) != 123):
        raise PermissionError("Participant roles overlap or reservation counts changed")
    catalogue = {r["vid"]: r["family"] for r in contract["catalogue"]}
    if (len(catalogue) != 23 or set(catalogue) != set(DEVELOPMENT_CLIPS + CONFIRMATION_CLIPS)
            or catalogue[7] != catalogue[8]
            or len({catalogue[v] for v in DEVELOPMENT_CLIPS}) != 14
            or len({catalogue[v] for v in CONFIRMATION_CLIPS}) != 8
            or {catalogue[v] for v in DEVELOPMENT_CLIPS} & {catalogue[v] for v in CONFIRMATION_CLIPS}):
        raise PermissionError("Catalogue family contract changed")
    for arm, role, clips in (("source_fit", "source", DEVELOPMENT_CLIPS),
                             ("confirmation_unseen", "confirmation", CONFIRMATION_CLIPS),
                             ("confirmation_familiar", "confirmation", DEVELOPMENT_CLIPS)):
        packet = contract["arms"][arm]
        if packet["role"] != role or packet["pairs"] != _pairs(roles[role], clips):
            raise PermissionError("Exact declared trial rectangle changed")
    if contract.get("source_identity_sha256") != source_identity_sha256(contract):
        raise PermissionError("Source identity digest changed")
    if (contract["material_scope"].get("original_footage_overlap_authenticated") is not False
            or contract["analysis"].get("bootstrap_draws") != BOOTSTRAP_DRAWS
            or contract["analysis"].get("bootstrap_seed") != BOOTSTRAP_SEED):
        raise PermissionError("Footage scope or fixed inference recipe changed")
    recipe = contract["candidate_predictor"]
    if (recipe.get("context_features") != 13 or recipe.get("spectral_features") != 150
            or recipe.get("alpha") != 100.0 or recipe.get("training_trials") != 980
            or recipe.get("validation_people_promoted_into_source") is not False):
        raise PermissionError("Frozen source-only predictor recipe changed")


def _bound_receipt(gate, bound_receipts, name):
    if gate.get("passed") is not True:
        raise PermissionError(name + " gate has not passed")
    path = gate.get("receipt_path")
    digest = gate.get("receipt_sha256")
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise PermissionError(name + " requires a bound receipt")
    blob = bound_receipts.get(path)
    if not isinstance(blob, bytes) or hashlib.sha256(blob).hexdigest() != digest:
        raise PermissionError(name + " receipt bytes differ from activation binding")
    receipt = json.loads(blob)
    if receipt.get("passed") is not True:
        raise PermissionError(name + " bound receipt did not pass")
    return receipt


def require_source_integrity(contract, source_gate, bound_receipts):
    """A metadata receipt must bind every clean source identity before targets."""
    validate_contract(contract)
    receipt = _bound_receipt(source_gate, bound_receipts, "Source integrity")
    if (receipt.get("source_identity_sha256") != contract["source_identity_sha256"]
            or receipt.get("source_trials") != 980 or receipt.get("source_failures") != 0
            or receipt.get("policy_id") != POLICY_ID
            or receipt.get("complete_source_feature_cache_qualification_passed") is not True
            or receipt.get("validation_people_used_for_fitting") is not False
            or receipt.get("confirmation_access") is not False):
        raise PermissionError("Exact clean-source authenticated feature-cache/QC qualification is required")
    return receipt


def require_confirmation_activation(contract, activation, bound_receipts):
    """Require explicit separate approvals plus receipt-bound precision/source gates."""
    validate_contract(contract)
    if (activation.get("status") != "active_confirmation"
            or activation.get("contract_semantic_sha256") != canonical_sha256(contract)
            or activation.get("author_question_approval") is not True
            or activation.get("confirmation_access_approval") is not True):
        raise PermissionError("Protected confirmation is inactive; separate explicit author approvals required")
    precision = _bound_receipt(activation.get("precision_gate", {}), bound_receipts, "Precision")
    if (precision.get("contract_semantic_sha256") != canonical_sha256(contract)
            or precision.get("confirmation_outcomes_accessed") is not False):
        raise PermissionError("Outcome-free precision receipt must bind this exact contract")
    require_source_integrity(contract, activation.get("source_integrity", {}), bound_receipts)
    continuation = _bound_receipt(activation.get("original_no_lead_gate", {}), bound_receipts,
                                  "Original empirical continuation")
    if (continuation.get("contract_semantic_sha256") != canonical_sha256(contract)
            or continuation.get("empirical_continuation_requirement_met") is not True
            or continuation.get("confirmation_outcomes_accessed") is not False):
        raise PermissionError("The original empirical continuation hold cannot be cleared by precision simulation")
    state = _bound_receipt(activation.get("final_state_qualification", {}), bound_receipts,
                           "Final state qualification")
    if (state.get("contract_semantic_sha256") != canonical_sha256(contract)
            or state.get("source_identity_sha256") != contract["source_identity_sha256"]
            or state.get("joint_and_matched_context_states_qualified") is not True
            or state.get("source_only_scaling_and_priors") is not True
            or state.get("ridge_alpha") != 100.0
            or state.get("matched_context_dimension") != 13
            or state.get("joint_dimension") != 163
            or state.get("no_validation_targets_in_fit") is not True
            or state.get("confirmation_outcomes_accessed") is not False):
        raise PermissionError("Both fixed final states require source-only qualification")


def require_prediction_lock(contract, activation, bound_receipts):
    """Require a complete byte-bound prediction ledger before protected ratings.

    Prediction payloads remain opaque bytes: this helper hashes them without
    interpreting prediction numbers or targets. A future backend must produce
    the identity ledger from its guarded prediction export, before rating I/O.
    """
    require_confirmation_activation(contract, activation, bound_receipts)
    lock = _bound_receipt(activation.get("prediction_lock", {}), bound_receipts,
                          "Protected prediction lock")
    expected = {arm: contract["arms"][arm]["pairs"]
                for arm in ("confirmation_unseen", "confirmation_familiar")}
    if (lock.get("contract_semantic_sha256") != canonical_sha256(contract)
            or lock.get("source_identity_sha256") != contract["source_identity_sha256"]
            or lock.get("final_state_receipt_sha256") != activation["final_state_qualification"]["receipt_sha256"]
            or lock.get("predictions_locked") is not True
            or lock.get("protected_rating_values_decoded_before_lock") != 0
            or lock.get("prediction_identities") != expected
            or lock.get("models") != list(PREDICTION_MODELS)):
        raise PermissionError("Complete three-model prediction identities must be locked before protected ratings")
    files = lock.get("prediction_files")
    if (not isinstance(files, list) or len(files) != 3
            or [f.get("model") for f in files] != list(PREDICTION_MODELS)
            or len({f.get("path") for f in files}) != 3):
        raise PermissionError("Exactly three distinct complete prediction files required")
    identities_digest = canonical_sha256(expected)
    for record in files:
        payload = bound_receipts.get(record.get("path"))
        if (not isinstance(payload, bytes)
                or record.get("sha256") != hashlib.sha256(payload).hexdigest()
                or record.get("identity_sha256") != identities_digest
                or record.get("primary_trials") != 297 or record.get("familiar_trials") != 462):
            raise PermissionError("Locked prediction bytes or complete identity binding changed")
    projections = lock.get("prediction_only_projection_sha256")
    if (not isinstance(projections, dict) or set(projections) != set(expected)
            or any(not isinstance(v, str) or len(v) != 64 for v in projections.values())):
        raise PermissionError("Each exact arm requires a locked prediction-only projection digest")
    return lock


def guard_trial(row, arm, contract, *, activation=None, bound_receipts=None):
    """Guard person, role, arm, clip and title before any target/feature conversion."""
    validate_contract(contract)
    if arm not in ARMS:
        raise PermissionError("Unknown trial arm")
    packet = contract["arms"][arm]
    allowed = {(r["subject"], r["vid"]) for r in packet["pairs"]}
    catalogue = {r["vid"]: r["family"] for r in contract["catalogue"]}
    pair = _guard_identity(row, arm, packet["role"], allowed, catalogue)
    if packet["role"] == "confirmation":
        require_confirmation_activation(contract, activation or {}, bound_receipts or {})
    return pair


def _guard_identity(row, arm, role, allowed, catalogue):
    """Internal metadata-only check after caller validates the whole contract."""
    if row.get("arm") != arm:
        raise PermissionError("Unknown or changed trial arm")
    subject, vid = row.get("subject"), row.get("vid")
    if not isinstance(subject, str) or type(vid) is not int:
        raise PermissionError("Trial identities require a string person and integer clip")
    if ((subject, vid) not in allowed or row.get("role") != role
            or row.get("family") != catalogue.get(vid)):
        raise PermissionError("Participant, title, arm or role is outside the exact contract")
    return subject, vid


def guarded_read(row, arm, contract, reader, *, activation=None, bound_receipts=None,
                 source_gate=None, read_kind=None):
    """Invoke a future backend only after identities and relevant gates pass."""
    pair = guard_trial(row, arm, contract, activation=activation, bound_receipts=bound_receipts)
    if arm == "source_fit":
        require_source_integrity(contract, source_gate or {}, bound_receipts or {})
    else:
        if read_kind not in ("confirmation_signal", "confirmation_rating"):
            raise PermissionError("Protected backend requires an explicit signal or rating access phase")
        if read_kind == "confirmation_rating":
            require_prediction_lock(contract, activation or {}, bound_receipts or {})
    return reader(pair)


def guard_source_records(records, contract, source_gate, bound_receipts):
    """Require the whole clean source rectangle before a future fit/target join."""
    validate_contract(contract)
    expected = {(r["subject"], r["vid"]) for r in contract["arms"]["source_fit"]["pairs"]}
    catalogue = {r["vid"]: r["family"] for r in contract["catalogue"]}
    pairs = [_guard_identity(row, "source_fit", "source", expected, catalogue) for row in records]
    if len(pairs) != len(expected) or len(set(pairs)) != len(pairs) or set(pairs) != expected:
        raise PermissionError("Exact 980 source identities required before target conversion/fitting")
    require_source_integrity(contract, source_gate, bound_receipts)
    return pairs


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean is not a " + name)
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 7:
        raise ValueError("Invalid " + name + "; stop without dropping or imputing its row")
    return value


def prediction_with_fallback(row, decision, comparator, matched_context_predictor, eeg_predictor, contract,
                             *, activation, bound_receipts):
    """For unavailable EEG, retain the row and never call the EEG backend."""
    if row.get("arm") not in ("confirmation_unseen", "confirmation_familiar"):
        raise PermissionError("Protected inference helper cannot read or fit source rows")
    guard_trial(row, row.get("arm"), contract, activation=activation, bound_receipts=bound_receipts)
    if (decision.get("policy_id") != POLICY_ID or type(decision.get("eeg_available")) is not bool
            or not isinstance(decision.get("reasons"), list)
            or decision["eeg_available"] == bool(decision["reasons"])):
        raise PermissionError("Availability differs from the frozen signal-only policy")
    baseline = _number(comparator(), "category comparator prediction")
    available = decision["eeg_available"]
    # Clip an available Ridge output before score validation. Preserve failures.
    if available:
        raw = float(eeg_predictor())
        if not np.isfinite(raw):
            raise ValueError("Nonfinite EEG output; stop rather than outcome-derived exclusion")
        prediction = float(np.clip(raw, 0, 7))
        matched_raw = float(matched_context_predictor())
        if not np.isfinite(matched_raw):
            raise ValueError("Nonfinite matched-context output; stop without dropping its row")
        matched = float(np.clip(matched_raw, 0, 7))
    else:
        prediction = baseline
        matched = baseline
    return {**row, "category_prediction": baseline, "joint_prediction": prediction,
            "matched_context_prediction": matched,
            "eeg_available": available, "fallback": not available,
            "fallback_policy": "source_category_median" if not available else None}


def prediction_only_projection(rows, arm, contract):
    """Canonical guarded prediction export projection; targets are never accessed.

    Caller must separately qualify actual locked export bytes against this
    projection. This pure helper provides no recording/export reader.
    """
    validate_contract(contract)
    if arm not in ("confirmation_unseen", "confirmation_familiar"):
        raise PermissionError("Prediction projection requires a declared confirmation arm")
    expected = {(r["subject"], r["vid"]) for r in contract["arms"][arm]["pairs"]}
    catalogue = {r["vid"]: r["family"] for r in contract["catalogue"]}
    pairs = [_guard_identity(row, arm, "confirmation", expected, catalogue) for row in rows]
    if len(pairs) != len(expected) or len(set(pairs)) != len(pairs) or set(pairs) != expected:
        raise PermissionError("Complete declared denominator required before converting outcomes")
    result = []
    for row in rows:
        category = _number(row["category_prediction"], "category prediction")
        joint = _number(row["joint_prediction"], "joint prediction")
        matched = _number(row["matched_context_prediction"], "matched-context prediction")
        if type(row.get("eeg_available")) is not bool or type(row.get("fallback")) is not bool:
            raise PermissionError("Every trial requires fixed availability/fallback metadata")
        if row["fallback"] != (not row["eeg_available"]):
            raise PermissionError("Fallback availability disagrees")
        if row["fallback"] and (joint != category or matched != category
                                or row.get("fallback_policy") != "source_category_median"):
            raise PermissionError("Unavailable EEG must retain source-category comparator prediction")
        result.append({"subject": row["subject"], "vid": row["vid"], "family": row["family"],
                       "role": row["role"], "arm": arm,
                       "category_prediction": category, "joint_prediction": joint,
                       "matched_context_prediction": matched,
                       "eeg_available": row["eeg_available"], "fallback": row["fallback"],
                       "fallback_policy": row.get("fallback_policy")})
    return sorted(result, key=lambda r: (r["subject"], r["vid"]))


def paired_loss_analysis(rows, arm, contract, *, activation, bound_receipts):
    """Fixed complete-denominator analysis of a future authorized prediction set.

    Clip-level losses, including unavailable-EEG fallback rows, are averaged
    within person/catalogue-title before equal title/person weighting.
    """
    if arm not in ("confirmation_unseen", "confirmation_familiar"):
        raise PermissionError("Confirmation analysis requires a declared confirmation arm")
    lock = require_prediction_lock(contract, activation, bound_receipts)
    projection = prediction_only_projection(rows, arm, contract)
    if canonical_sha256(projection) != lock["prediction_only_projection_sha256"][arm]:
        raise PermissionError("Caller prediction-only projection differs from locked predictions before target conversion")
    grouped = {}
    fallback = 0
    for row in rows:
        target = _number(row["target"], "item10 valence")
        category = _number(row["category_prediction"], "category prediction")
        joint = _number(row["joint_prediction"], "joint prediction")
        matched = _number(row["matched_context_prediction"], "matched-context prediction")
        if type(row.get("eeg_available")) is not bool or type(row.get("fallback")) is not bool:
            raise PermissionError("Every trial requires fixed EEG availability and fallback metadata")
        if row["fallback"] != (not row["eeg_available"]):
            raise PermissionError("Fallback availability disagrees")
        if row["fallback"] and (joint != category or matched != category
                                or row.get("fallback_policy") != "source_category_median"):
            raise PermissionError("Unavailable EEG must retain source-category comparator prediction")
        fallback += int(row["fallback"])
        grouped.setdefault((row["subject"], row["family"]), []).append(
            (abs(target - category), abs(target - joint), abs(target - matched)))
    people = sorted({p for p, _ in grouped})
    families = sorted({f for _, f in grouped})
    if len(grouped) != len(people) * len(families):
        raise PermissionError("Complete person/catalogue-title rectangle required")
    losses = np.asarray([[np.mean(grouped[p, f], axis=0) for f in families] for p in people])
    gain = losses[..., 0] - losses[..., 1]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    person_draw = rng.integers(len(people), size=(BOOTSTRAP_DRAWS, len(people)))
    title_draw = rng.integers(len(families), size=(BOOTSTRAP_DRAWS, len(families)))
    sampled = gain[person_draw[:, :, None], title_draw[:, None, :]].mean(axis=(1, 2))
    matched_gain = losses[..., 2] - losses[..., 1]
    matched_sampled = matched_gain[person_draw[:, :, None], title_draw[:, None, :]].mean(axis=(1, 2))
    return {
        "arm": arm, "trials": len(rows), "participants": len(people),
        "catalogue_title_families": len(families), "unavailable_eeg_fallback_trials": fallback,
        "category_mae": float(losses[..., 0].mean()), "joint_mae": float(losses[..., 1].mean()),
        "matched_context_mae": float(losses[..., 2].mean()),
        "paired_gain": float(gain.mean()),
        "primary_information_scope": "Whole-recipe utility; not an isolated EEG contrast",
        "mandatory_matched_context_gain": float(matched_gain.mean()),
        "matched_context_descriptive_95_percent_interval": np.quantile(matched_sampled, [.025, .975]).tolist(),
        "crossed_resamples": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
        "descriptive_95_percent_interval": np.quantile(sampled, [.025, .975]).tolist(),
        "leave_one_catalogue_title_out_gain": {f: float(np.delete(gain, i, axis=1).mean())
                                                for i, f in enumerate(families)},
        "all_declared_rows_retained": True,
        "original_footage_overlap_authenticated": False,
        "interpretation": "Specific fixed recipes/task; interval including zero does not establish equivalence or absence of EEG information",
    }
