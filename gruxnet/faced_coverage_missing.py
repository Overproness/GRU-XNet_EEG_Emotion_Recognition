"""Pure source-only coverage comparison with a shared label-blind EEG mask.

Acquisition thresholds stay unchanged. A failed region makes the entire trial's
EEG unavailable for both coverage representations. Training rows and labels are
retained; EEG means use only available training donors. Held unavailable trials
use the same category prior in all three models. No filesystem or raw-data I/O.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .faced_classical_contract import context_features, fit_source_ridge, predict_ridge
from .faced_classical_runner import estimate_priors, prior_predictions
from .faced_coverage_selection import (
    ALPHAS, MODELS, JOINT_MODELS, PRIMARY_ALPHA, FIT_CAP, EXPECTED_PREDICTIONS,
    SourceSelectionGuard, source_roster, crossed_source_cells,
    distributed_features, select_common_alpha, nominate_coverage,
    _identity, _prediction_rows, _metrics, _assert_matched_scalers,
)
from .faced_temporal_coverage import REGIONS
from .faced_proposal_preprocessing import COMMON_CHANNELS

ENGINEERED_MISSING_REASONS = frozenset((
    "adc_rail", "stationary_counts", "low_processed_variation",
    "high_processed_variation", "high_processed_excursion", "residual_50hz_power",
    "repair_capacity_exceeded",
))


def array_sha256(array):
    values = np.ascontiguousarray(array)
    return hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode() + values.tobytes()).hexdigest()


def shared_availability(source_pairs, metadata, quality_rows):
    """Authenticate every QC identity before deriving a target-independent mask."""
    pairs, _ = source_roster(source_pairs, metadata)
    quality, available = qualify_quality(pairs, quality_rows)
    return available, quality


def qualify_quality(source_pairs, quality_rows):
    """Return (quality gate, shared availability) from exact source QC records.

    Only unchanged engineered review flags activate the missing-EEG route.
    Technical decoding/numerical failures remain fatal, never imputable.
    The runner must also bind metadata before invoking any access callbacks.
    """
    pairs = tuple(_identity(p) for p in source_pairs)
    people = {p[0] for p in pairs}
    if len(pairs) != 980 or len(set(pairs)) != 980 or len(people) != 70:
        raise PermissionError("Require 980 distinct source identities from 70 participants")
    from .faced_temporal_coverage import DEVELOPMENT_CLIPS
    if set(pairs) != {(p, v) for p in people for v in DEVELOPMENT_CLIPS}:
        raise PermissionError("Require the exact source-only development rectangle")
    allowed = frozenset(pairs)
    statuses = {}
    failed = []
    for row in quality_rows:
        pair = _identity((row.get("subject"), row.get("vid")))
        region = row.get("region")
        if pair not in allowed or row.get("role") != "source" or region not in REGIONS:
            raise PermissionError("Only exact source-region quality identities permitted")
        key = (*pair, region)
        if key in statuses:
            raise PermissionError("Duplicate source-region quality identity")
        passed, flags = row.get("technical_passed"), row.get("review_flags")
        if not isinstance(passed, bool) or not isinstance(flags, list):
            raise ValueError("Require unchanged Boolean technical status and list of QC flags")
        if not passed:
            raise PermissionError("Technical QC failure is fatal and cannot be converted to missing EEG")
        for flag in flags:
            if not isinstance(flag, dict) or set(flag) != {"channel", "reason"}:
                raise ValueError("Require exact channel/reason QC dictionaries")
            if not isinstance(flag["channel"], str) or not isinstance(flag["reason"], str):
                raise ValueError("Require literal channel and reason names")
            if flag["reason"] not in ENGINEERED_MISSING_REASONS:
                raise PermissionError("Unknown or numerical QC reason cannot be converted to missing EEG")
            expected_channels = {"all"} if flag["reason"] == "repair_capacity_exceeded" else set(COMMON_CHANNELS)
            if flag["channel"] not in expected_channels:
                raise PermissionError("QC flag channel differs from the unchanged policy")
        if "eeg_available" in row and (not isinstance(row["eeg_available"], bool)
                                       or row["eeg_available"] != (not flags)):
            raise PermissionError("Authenticated regional availability disagrees with QC flags")
        statuses[key] = passed and not flags
        if not statuses[key]:
            failed.append({"subject": pair[0], "vid": pair[1], "region": region,
                           "technical_passed": passed, "review_flags": list(flags)})
    if set(statuses) != {(*p, region) for p in pairs for region in REGIONS} or len(statuses) != 2940:
        raise PermissionError("All 2,940 declared source-region QC records are required")
    available = {p: all(statuses[(*p, region)] for region in REGIONS) for p in pairs}
    unavailable = [p for p in pairs if not available[p]]
    return {"source_trials": 980, "source_regions": 2940,
                       "available_trials": 980 - len(unavailable),
                       "unavailable_trials": len(unavailable),
                       "unavailable_pairs": unavailable,
                       "failed_or_flagged_region_count": len(failed),
                       "failed_or_flagged_regions": failed,
                       "availability_rule": "All three regions must pass existing technical QC with no review flags",
                       "availability_uses_targets": False, "qc_thresholds_changed": False,
                       "training_labels_retained": True,
                       "technical_failures_imputable": False,
                       "passed_for_shared_missing_policy": True}, available


def _qualify_donors(cells, availability):
    for cell in cells:
        if not any(availability[p] for p in cell["training"]):
            raise PermissionError("No jointly available training EEG donors; fitting held")


def _source_maps(pairs, metadata, availability, feature_loader, target_loader):
    guard = SourceSelectionGuard(pairs, metadata)
    contexts = {p: context_features(metadata[p]) for p in pairs}
    eeg = {"late_joint": {}, "distributed_joint": {}}
    for pair in pairs:
        if not availability[pair]:
            continue
        regions = [guard.feature(pair, region, feature_loader) for region in REGIONS]
        eeg["late_joint"][pair] = regions[2]
        eeg["distributed_joint"][pair] = distributed_features(*regions)
    # Every permitted feature is qualified before the first target callback.
    targets = {p: guard.target(p, target_loader) for p in pairs}
    return contexts, eeg, targets


def training_imputation(training_pairs, contexts, eeg_features, availability):
    """Compute EEG means from available training rows; retain all training rows."""
    training = [_identity(p) for p in training_pairs]
    if not training or len(training) != len(set(training)) or any(p not in availability for p in training):
        raise PermissionError("Imputation requires distinct declared training identities")
    donors = [p for p in training if availability[p]]
    if not donors:
        raise PermissionError("No jointly available training EEG donors; fitting held")
    values = np.stack([eeg_features[p] for p in donors]).astype(np.float64)
    if values.shape != (len(donors), 150) or not np.isfinite(values).all():
        raise ValueError("Invalid available-training EEG donor vectors")
    mean = values.mean(axis=0)
    if mean.shape != (150,) or not np.isfinite(mean).all():
        raise ValueError("Invalid available-training EEG mean")
    features = {p: np.concatenate((contexts[p], eeg_features[p] if availability[p] else mean)) for p in training}
    if any(x.shape != (163,) or not np.isfinite(x).all() for x in features.values()):
        raise ValueError("Invalid imputed training context/EEG vectors")
    mean_sha = array_sha256(mean)
    donor_sha = hashlib.sha256(json.dumps(donors, separators=(",", ":")).encode()).hexdigest()
    manifest = {"donor_pairs": donors, "donor_rows": len(donors),
                "missing_training_pairs": [p for p in training if not availability[p]],
                "missing_training_rows": sum(not availability[p] for p in training),
                "training_rows_retained": len(training), "eeg_dimension": 150,
                "mean_sha256": mean_sha, "mean_hash_rule": "Existing array_sha256: dtype text, shape text, contiguous float64 bytes",
                "donor_identity_sha256": donor_sha,
                "mean_uses_only_available_training_rows": True,
                "held_rows_used_for_imputation": 0, "target_values_used_for_imputation": 0}
    return features, mean, manifest


def _held_values(fitted, cell, features, availability, category_values):
    available_held = [p for p in cell["held"] if availability[p]]
    predicted = predict_ridge(fitted, available_held, features) if available_held else np.empty(0)
    lookup = dict(zip(available_held, predicted))
    return np.asarray([lookup[p] if availability[p] else prior
                       for p, prior in zip(cell["held"], category_values)], dtype=np.float64)


def _rows(cell, model, alpha, values, targets, metadata, availability):
    rows = _prediction_rows(cell, model, alpha, values, targets, metadata)
    for row in rows:
        pair = row["subject"], row["vid"]
        row["shared_eeg_available"] = availability[pair]
        row["common_category_fallback"] = model != "category_median" and not availability[pair]
    return rows


def _summaries(predictions, priors):
    candidates = [{"model": model, "alpha": alpha,
                   **_metrics([r for r in predictions if r["model"] == model and r["alpha"] == alpha])}
                  for alpha in ALPHAS for model in MODELS]
    category_metrics = _metrics(priors)
    contrasts = []
    for alpha in ALPHAS:
        values = {m: next(r["participant_title_mae"] for r in candidates if r["model"] == m and r["alpha"] == alpha)
                  for m in MODELS}
        contrasts.append({"alpha": alpha, "primary_fixed_alpha": alpha == PRIMARY_ALPHA,
                          "distributed_minus_late_mae": values["distributed_joint"] - values["late_joint"],
                          "distributed_coverage_gain": values["late_joint"] - values["distributed_joint"],
                          "matched_context_minus_late_mae": values["context_only"] - values["late_joint"],
                          "matched_context_minus_distributed_mae": values["context_only"] - values["distributed_joint"],
                          "category_median_minus_late_mae": category_metrics["participant_title_mae"] - values["late_joint"],
                          "category_median_minus_distributed_mae": category_metrics["participant_title_mae"] - values["distributed_joint"]})
    alpha_selection = select_common_alpha(candidates)
    nomination = nominate_coverage(candidates, alpha_selection["selected_alpha"])
    nomination.update({"selection_scope": "Adaptive source-only crossed person/material CV with shared label-blind missing-EEG policy",
                       "matched_context_alpha": nomination["alpha"],
                       "primary_comparison_reselected": False, "independent_confirmation": False})
    return {"candidate_metrics": candidates, "category_median_metrics": category_metrics,
            "contrasts": contrasts, "common_alpha_selection": alpha_selection,
            "source_nomination": nomination}


def run_source_cv(*, source_pairs, metadata, person_folds, quality_rows,
                  feature_loader, target_loader, on_fit=None, check_budget=None):
    """Exactly 90 source CV fits, retaining all 448 training/84 held cell rows."""
    pairs, _ = source_roster(source_pairs, metadata)
    cells = crossed_source_cells(pairs, metadata, person_folds)
    availability, quality = shared_availability(pairs, metadata, quality_rows)
    _qualify_donors(cells, availability)
    contexts, eeg, targets = _source_maps(pairs, metadata, availability, feature_loader, target_loader)
    predictions, priors, fits, cell_metrics = [], [], [], []
    for cell in cells:
        prior = estimate_priors(cell["training"], targets, metadata)
        category_values = prior_predictions(prior, cell["held"], metadata)["category_median"]
        prior_rows = _rows(cell, "category_median", None, category_values, targets, metadata, availability)
        priors.extend(prior_rows)
        cell_metrics.append({"cell": cell["cell"], "model": "category_median", "alpha": None, **_metrics(prior_rows)})
        features = {"context_only": contexts}
        means, imputations = {}, {}
        for model in JOINT_MODELS:
            features[model], means[model], imputations[model] = training_imputation(cell["training"], contexts, eeg[model], availability)
            for p in cell["held"]:
                if availability[p]:
                    features[model][p] = np.concatenate((contexts[p], eeg[model][p]))
        for alpha in ALPHAS:
            fitted_models = {}
            for model in MODELS:
                if check_budget is not None:
                    check_budget()
                if len(fits) >= FIT_CAP:
                    raise PermissionError("Declared 90-fit source cap exceeded")
                fitted = fit_source_ridge(cell["training"], features[model], targets, alpha)
                fitted_models[model] = fitted
                fitted["eeg_imputation_mean"] = means[model].copy() if model in JOINT_MODELS else None
                manifest = {"fit": len(fits), "cell": cell["cell"], "model": model,
                            "alpha": alpha, "feature_dimension": 13 if model == "context_only" else 163,
                            "training_pairs": cell["training"], "held_pairs": cell["held"],
                            "source_identity_sha256": fitted["source_identity_sha256"],
                            "scaler_training_rows": 448, "target_training_rows": 448,
                            "imputation": imputations.get(model),
                            "eligible_train_donor_pairs": [p for p in cell["training"] if availability[p]],
                            "eligible_train_donor_identity_sha256": hashlib.sha256(json.dumps([p for p in cell["training"] if availability[p]], separators=(",", ":")).encode()).hexdigest(),
                            "imputed_training_pairs": [p for p in cell["training"] if not availability[p]] if model in JOINT_MODELS else [],
                            "shared_available_training_rows": sum(availability[p] for p in cell["training"]),
                            "imputed_training_rows": sum(not availability[p] for p in cell["training"]) if model in JOINT_MODELS else 0,
                            "shared_missing_training_pairs": [p for p in cell["training"] if not availability[p]],
                            "shared_missing_held_pairs": [p for p in cell["held"] if not availability[p]]}
                fits.append(manifest)
                if on_fit is not None:
                    on_fit(manifest, fitted)
                values = _held_values(fitted, cell, features[model], availability, category_values)
                rows = _rows(cell, model, alpha, values, targets, metadata, availability)
                predictions.extend(rows)
                cell_metrics.append({"cell": cell["cell"], "model": model, "alpha": alpha, **_metrics(rows)})
            _assert_matched_scalers(fitted_models)
            if check_budget is not None:
                check_budget()
    if len(fits) != FIT_CAP or len(predictions) != EXPECTED_PREDICTIONS or len(priors) != 840:
        raise PermissionError("Incomplete shared-policy source coverage comparison")
    return {"fits": fits, "cv_predictions": predictions, "category_predictions": priors,
            "cell_metrics": cell_metrics, **_summaries(predictions, priors),
            "quality_gate": quality, "shared_availability": availability,
            "cells": cells, "source_targets_used": 980,
            "source_feature_callbacks": 3 * quality["available_trials"], "source_target_callbacks": 980,
            "validation_access": False, "protected_access": False,
            "independent_confirmation": False, "final_source_fits": 0,
            "uncertainty_claim": "No independent confidence interval from adaptive source CV"}


def fit_full_source(*, source_pairs, metadata, quality_rows, feature_loader,
                    target_loader, candidate_metrics, on_fit=None, check_budget=None):
    """Exactly three source980 state fits with jointly available donor means."""
    pairs, _ = source_roster(source_pairs, metadata)
    availability, quality = shared_availability(pairs, metadata, quality_rows)
    if not any(availability.values()):
        raise PermissionError("No jointly available full-source EEG donors; fitting held")
    selected = select_common_alpha(candidate_metrics)
    alpha = selected["selected_alpha"]
    contexts, eeg, targets = _source_maps(pairs, metadata, availability, feature_loader, target_loader)
    features = {"context_only": contexts}
    means, imputations = {}, {}
    for model in JOINT_MODELS:
        features[model], means[model], imputations[model] = training_imputation(pairs, contexts, eeg[model], availability)
    fitted_models, manifests = {}, []
    for model in MODELS:
        if check_budget is not None:
            check_budget()
        fitted = fit_source_ridge(pairs, features[model], targets, alpha)
        fitted_models[model] = fitted
        fitted["eeg_imputation_mean"] = means[model].copy() if model in JOINT_MODELS else None
        manifest = {"fit": len(manifests), "stage": "full_source_state_qualification",
                    "model": model, "alpha": alpha, "feature_dimension": 13 if model == "context_only" else 163,
                    "training_pairs": list(pairs), "training_rows": 980,
                    "source_identity_sha256": fitted["source_identity_sha256"],
                    "imputation": imputations.get(model),
                    "eligible_train_donor_pairs": [p for p in pairs if availability[p]],
                    "eligible_train_donor_identity_sha256": hashlib.sha256(json.dumps([p for p in pairs if availability[p]], separators=(",", ":")).encode()).hexdigest(),
                    "imputed_training_pairs": quality["unavailable_pairs"] if model in JOINT_MODELS else [],
                    "shared_available_training_rows": quality["available_trials"],
                    "imputed_training_rows": quality["unavailable_trials"] if model in JOINT_MODELS else 0,
                    "shared_missing_training_pairs": quality["unavailable_pairs"],
                    "validation_rows_scored": 0, "protected_rows_scored": 0}
        manifests.append(manifest)
        if on_fit is not None:
            on_fit(manifest, fitted)
    _assert_matched_scalers(fitted_models)
    if check_budget is not None:
        check_budget()
    return {"fits": manifests, "private_fitted_models": fitted_models,
            "category_prior": estimate_priors(pairs, targets, metadata),
            "common_alpha_selection": selected, "quality_gate": quality,
            "shared_availability": availability, "full_source_fit_count": 3,
            "full_source_prior_count": 1, "validation_access": False, "protected_access": False}
