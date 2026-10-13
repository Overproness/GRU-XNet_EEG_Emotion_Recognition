"""Pure, bounded source-only late/distributed spectral coverage comparison.

The caller supplies already authenticated metadata, quality records and guarded
callbacks. This module has no filesystem, recording or original-rating reader.
All ten cells and all declared candidates are retained. Source CV is adaptive
development evidence, not an independent confirmation experiment.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .faced_classical_contract import (
    CATEGORIES, context_features, fit_source_ridge, participant_film_mae,
    predict_ridge,
)
from .faced_classical_runner import estimate_priors, prior_predictions
from .faced_temporal_coverage import DEVELOPMENT_CLIPS, REGIONS

ALPHAS = (0.1, 10.0, 100.0)
MODELS = ("context_only", "late_joint", "distributed_joint")
JOINT_MODELS = MODELS[1:]
PRIMARY_ALPHA = 100.0
FIT_CAP = 90
EXPECTED_PREDICTIONS = 7560
SINGLETON_CATEGORIES = {"Fear": 9, "Inspiration": 20}


def _identity(pair):
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or not isinstance(pair[0], str) or not pair[0]
            or isinstance(pair[1], bool) or not isinstance(pair[1], int)):
        raise PermissionError("Malformed source trial identity")
    return tuple(pair)


def source_roster(source_pairs, metadata):
    """Qualify the exact rectangle before any feature or target callback."""
    pairs = tuple(_identity(p) for p in source_pairs)
    if len(pairs) != 980 or len(set(pairs)) != 980:
        raise PermissionError("Require 980 distinct source identities")
    people = {p[0] for p in pairs}
    if len(people) != 70 or set(pairs) != {(s, v) for s in people for v in DEVELOPMENT_CLIPS}:
        raise PermissionError("Require the fixed 70 by 14 source rectangle")
    if set(metadata) != set(pairs):
        raise PermissionError("Metadata must contain only the declared source rectangle")
    titles = {}
    for pair in pairs:
        row = metadata[pair]
        if row.get("role") != "source":
            raise PermissionError("Validation or protected role rejected before callbacks")
        category, family = row.get("assigned_category"), row.get("family")
        if category not in CATEGORIES or not isinstance(family, str) or not family:
            raise PermissionError("Missing source category or material family")
        definition = category, family
        if pair[1] in titles and titles[pair[1]] != definition:
            raise PermissionError("Source title category/family varies across people")
        titles[pair[1]] = definition
    if len({family for _, family in titles.values()}) != 14:
        raise PermissionError("Source material families are not distinct")
    categories = {c: sorted(v for v, (category, _) in titles.items() if category == c)
                  for c in CATEGORIES}
    for category, clips in categories.items():
        expected = [SINGLETON_CATEGORIES[category]] if category in SINGLETON_CATEGORIES else None
        if (expected is not None and clips != expected) or (expected is None and len(clips) != 2):
            raise PermissionError("Changed six paired categories/two source-only singleton titles")
    return tuple(sorted(pairs)), categories


def crossed_source_cells(source_pairs, metadata, person_folds):
    """Five fixed person folds crossed with two supported-category title arms."""
    pairs, categories = source_roster(source_pairs, metadata)
    people = {p[0] for p in pairs}
    folds = tuple(tuple(fold) for fold in person_folds)
    flat = [p for fold in folds for p in fold]
    if (len(folds) != 5 or any(len(fold) != 14 for fold in folds)
            or len(flat) != len(set(flat)) or set(flat) != people):
        raise PermissionError("Five frozen 14-person source folds required")
    singleton_clips = sorted(SINGLETON_CATEGORIES.values())
    cells = []
    for person_fold, held_people in enumerate(folds):
        for material_assignment in range(2):
            held_clips = sorted(clips[material_assignment] for category, clips in categories.items()
                                if category not in SINGLETON_CATEGORIES)
            training_clips = sorted(set(DEVELOPMENT_CLIPS) - set(held_clips))
            training = [p for p in pairs if p[0] not in held_people and p[1] in training_clips]
            held = [p for p in pairs if p[0] in held_people and p[1] in held_clips]
            if len(training) != 448 or len(held) != 84:
                raise PermissionError("Changed 448-training/84-held source cell")
            if ({p[0] for p in training} & {p[0] for p in held}
                    or {metadata[p]["family"] for p in training} & {metadata[p]["family"] for p in held}):
                raise PermissionError("Source person/material exclusion failed")
            if {metadata[p]["assigned_category"] for p in training} != set(CATEGORIES):
                raise PermissionError("Category support missing from source training")
            if {metadata[p]["assigned_category"] for p in held} - {metadata[p]["assigned_category"] for p in training}:
                raise PermissionError("Held category unsupported")
            cells.append({"cell": len(cells), "person_fold": person_fold,
                          "material_assignment": material_assignment,
                          "training": training, "held": held,
                          "training_clips": training_clips, "held_clips": held_clips,
                          "source_only_singleton_clips": singleton_clips})
    expected = Counter(p for p in pairs if p[1] not in singleton_clips)
    if Counter(p for cell in cells for p in cell["held"]) != expected:
        raise PermissionError("Every paired-category source trial must be held exactly once")
    return cells


def qualify_quality(source_pairs, quality_rows):
    """Require all 2,940 regions with no failed or flagged source acquisition."""
    allowed = {_identity(p) for p in source_pairs}
    observed = set()
    for row in quality_rows:
        pair = _identity((row.get("subject"), row.get("vid")))
        region = row.get("region")
        if pair not in allowed or row.get("role") != "source" or region not in REGIONS:
            raise PermissionError("Nonsource or undeclared quality identity")
        key = (*pair, region)
        if key in observed:
            raise PermissionError("Duplicate source-region quality record")
        observed.add(key)
        if row.get("technical_passed") is not True or row.get("review_flags") != []:
            raise PermissionError("Failed or flagged source region holds features, targets and fits")
    if observed != {(*p, region) for p in allowed for region in REGIONS} or len(observed) != 2940:
        raise PermissionError("Complete 2,940-region source quality is required")
    return {"source_trials": 980, "source_regions": 2940, "passed": True}


class SourceSelectionGuard:
    """Guard exact source identities before invoking any supplied callback."""
    def __init__(self, source_pairs, metadata):
        pairs, _ = source_roster(source_pairs, metadata)
        self.allowed = frozenset(pairs)
        self.roles = MappingProxyType({p: metadata[p]["role"] for p in pairs})

    def _allow(self, pair):
        pair = _identity(pair)
        if pair not in self.allowed or self.roles.get(pair) != "source":
            raise PermissionError("Only frozen source identities are permitted")
        return pair

    def feature(self, pair, region, loader):
        pair = self._allow(pair)
        if region not in REGIONS:
            raise PermissionError("Only frozen early/middle/late regions are permitted")
        values = np.asarray(loader(pair, region), dtype=np.float64)
        if values.shape != (150,) or not np.isfinite(values).all():
            raise ValueError("Require one finite 150-dimensional source spectral vector")
        return values.copy()

    def target(self, pair, loader):
        pair = self._allow(pair)
        value = float(loader(pair))
        if not np.isfinite(value) or not 0 <= value <= 7:
            raise ValueError("Require finite original-scale source valence in [0, 7]")
        return value


def distributed_features(early, middle, late):
    """Mean already logged region vectors; retain 150 EEG columns."""
    values = np.stack((early, middle, late)).astype(np.float64)
    if values.shape != (3, 150) or not np.isfinite(values).all():
        raise ValueError("Distributed coverage requires all three finite 150-vectors")
    return values.mean(axis=0)


def select_common_alpha(candidates):
    """Fair common alpha: equal model weight, source MAE/MSE/alpha priority."""
    expected = {(m, a) for m in MODELS for a in ALPHAS}
    observed = [(r["model"], r["alpha"]) for r in candidates]
    if len(observed) != 9 or set(observed) != expected:
        raise PermissionError("Require all nine declared source candidates")
    if any(not np.isfinite([r["participant_title_mae"], r["trial_mse"]]).all() for r in candidates):
        raise ValueError("Invalid source-only candidate score")
    scores = [{"alpha": alpha,
               "equal_model_mean_mae": float(np.mean([r["participant_title_mae"] for r in candidates if r["alpha"] == alpha])),
               "equal_model_mean_mse": float(np.mean([r["trial_mse"] for r in candidates if r["alpha"] == alpha]))}
              for alpha in ALPHAS]
    selected = min(scores, key=lambda r: (r["equal_model_mean_mae"], r["equal_model_mean_mse"], ALPHAS.index(r["alpha"])))
    return {"selected_alpha": selected["alpha"], "alpha_scores": scores,
            "selection_scope": "Adaptive source-only crossed person/material CV",
            "rule": "Equal model weight across context/late/distributed; mean MAE, mean MSE, alpha order"}


def nominate_coverage(candidates, common_alpha):
    """Select only coverage at the already selected common alpha."""
    if common_alpha not in ALPHAS:
        raise PermissionError("Undeclared common source alpha")
    rows = [r for r in candidates if r["alpha"] == common_alpha and r["model"] in JOINT_MODELS]
    if len(rows) != 2 or {r["model"] for r in rows} != set(JOINT_MODELS):
        raise PermissionError("Both joint coverages at the common alpha required")
    if any(not np.isfinite([r["participant_title_mae"], r["trial_mse"]]).all() for r in rows):
        raise ValueError("Invalid source coverage nomination score")
    return dict(min(rows, key=lambda r: (r["participant_title_mae"], r["trial_mse"], JOINT_MODELS.index(r["model"]))))


def _source_maps(pairs, metadata, feature_loader, target_loader):
    guard = SourceSelectionGuard(pairs, metadata)
    contexts = {p: context_features(metadata[p]) for p in pairs}
    late, distributed = {}, {}
    for pair in pairs:
        regions = [guard.feature(pair, region, feature_loader) for region in REGIONS]
        late[pair] = np.concatenate((contexts[pair], regions[2]))
        distributed[pair] = np.concatenate((contexts[pair], distributed_features(*regions)))
    targets = {p: guard.target(p, target_loader) for p in pairs}
    return {"context_only": contexts, "late_joint": late, "distributed_joint": distributed}, targets


def _assert_matched_scalers(fitted_models):
    """Shared context columns must have byte-identical training moments."""
    context = fitted_models["context_only"]["scaler"]
    for model in JOINT_MODELS:
        scaler = fitted_models[model]["scaler"]
        for name in ("mean_", "var_", "scale_"):
            if not np.array_equal(getattr(context, name), getattr(scaler, name)[:13]):
                raise PermissionError("Matched context scaler moments differ")


def _prediction_rows(cell, model, alpha, values, targets, metadata):
    if len(values) != 84:
        raise PermissionError("Every source held cell must retain all 84 predictions")
    return [{"subject": p[0], "vid": p[1], "family": metadata[p]["family"],
             "acquisition_group": metadata[p]["acquisition_group"],
             "assigned_category": metadata[p]["assigned_category"],
             "cell": cell["cell"], "person_fold": cell["person_fold"],
             "material_assignment": cell["material_assignment"], "role": "source",
             "model": model, "alpha": alpha, "target": targets[p],
             "prediction": float(value)} for p, value in zip(cell["held"], values)]


def _metrics(rows):
    errors = np.asarray([r["prediction"] - r["target"] for r in rows])
    return {"participant_title_mae": participant_film_mae(rows),
            "trial_mse": float(np.mean(errors ** 2)), "prediction_rows": len(rows)}


def run_source_cv(*, source_pairs, metadata, person_folds, quality_rows,
                  feature_loader, target_loader, on_fit=None, check_budget=None):
    """Run the declared 90 pure source fits after complete quality qualification.

    All source vectors are checked before the first target callback. Features
    and targets are fetched once from source-only callbacks, then every cell's
    scalers and regressors are fitted only on its 448 training rows.
    """
    pairs, _ = source_roster(source_pairs, metadata)
    cells = crossed_source_cells(pairs, metadata, person_folds)
    quality_gate = qualify_quality(pairs, quality_rows)
    features, targets = _source_maps(pairs, metadata, feature_loader, target_loader)
    predictions, priors, fits, cell_metrics = [], [], [], []
    for cell in cells:
        prior = estimate_priors(cell["training"], targets, metadata)
        category_values = prior_predictions(prior, cell["held"], metadata)["category_median"]
        rows = _prediction_rows(cell, "category_median", None, category_values, targets, metadata)
        priors.extend(rows)
        cell_metrics.append({"cell": cell["cell"], "model": "category_median", "alpha": None, **_metrics(rows)})
        for alpha in ALPHAS:
            fitted_models = {}
            for model in MODELS:
                if check_budget is not None:
                    check_budget()
                if len(fits) >= FIT_CAP:
                    raise PermissionError("Declared 90-fit source cap exceeded")
                fitted = fit_source_ridge(cell["training"], features[model], targets, alpha)
                fitted_models[model] = fitted
                manifest = {"fit": len(fits), "cell": cell["cell"], "model": model,
                            "alpha": alpha, "feature_dimension": 13 if model == "context_only" else 163,
                            "training_pairs": cell["training"], "held_pairs": cell["held"],
                            "source_identity_sha256": fitted["source_identity_sha256"],
                            "scaler_training_rows": 448, "target_training_rows": 448}
                fits.append(manifest)
                if on_fit is not None:
                    on_fit(manifest, fitted)
                values = predict_ridge(fitted, cell["held"], features[model])
                rows = _prediction_rows(cell, model, alpha, values, targets, metadata)
                predictions.extend(rows)
                cell_metrics.append({"cell": cell["cell"], "model": model, "alpha": alpha, **_metrics(rows)})
            _assert_matched_scalers(fitted_models)
            if check_budget is not None:
                check_budget()
    if len(fits) != FIT_CAP or len(predictions) != EXPECTED_PREDICTIONS or len(priors) != 840:
        raise PermissionError("Incomplete bounded source coverage comparison")
    candidate_metrics = []
    for alpha in ALPHAS:
        for model in MODELS:
            rows = [r for r in predictions if r["model"] == model and r["alpha"] == alpha]
            candidate_metrics.append({"model": model, "alpha": alpha, **_metrics(rows)})
    prior_metrics = _metrics(priors)
    contrasts = []
    for alpha in ALPHAS:
        by_model = {m: next(r for r in candidate_metrics if r["model"] == m and r["alpha"] == alpha)
                    for m in MODELS}
        contrasts.append({"alpha": alpha, "primary_fixed_alpha": alpha == PRIMARY_ALPHA,
                          "distributed_minus_late_mae": by_model["distributed_joint"]["participant_title_mae"] - by_model["late_joint"]["participant_title_mae"],
                          "distributed_coverage_gain": by_model["late_joint"]["participant_title_mae"] - by_model["distributed_joint"]["participant_title_mae"],
                          "matched_context_minus_late_mae": by_model["context_only"]["participant_title_mae"] - by_model["late_joint"]["participant_title_mae"],
                          "matched_context_minus_distributed_mae": by_model["context_only"]["participant_title_mae"] - by_model["distributed_joint"]["participant_title_mae"],
                          "category_median_minus_late_mae": prior_metrics["participant_title_mae"] - by_model["late_joint"]["participant_title_mae"],
                          "category_median_minus_distributed_mae": prior_metrics["participant_title_mae"] - by_model["distributed_joint"]["participant_title_mae"]})
    alpha_selection = select_common_alpha(candidate_metrics)
    nomination = nominate_coverage(candidate_metrics, alpha_selection["selected_alpha"])
    nomination.update({"selection_scope": "Adaptive source-only crossed person/material CV",
                       "matched_context_alpha": nomination["alpha"],
                       "primary_comparison_reselected": False,
                       "independent_confirmation": False})
    return {"fits": fits, "cv_predictions": predictions, "category_predictions": priors,
            "cell_metrics": cell_metrics, "candidate_metrics": candidate_metrics,
            "category_median_metrics": prior_metrics, "contrasts": contrasts,
            "common_alpha_selection": alpha_selection,
            "source_nomination": nomination, "quality_gate": quality_gate,
            "cells": cells, "source_targets_used": 980,
            "source_feature_callbacks": 2940, "source_target_callbacks": 980,
            "validation_access": False, "protected_access": False,
            "independent_confirmation": False, "final_source_fits": 0,
            "uncertainty_claim": "No independent confidence interval from adaptive source CV"}


def fit_full_source(*, source_pairs, metadata, quality_rows, feature_loader,
                    target_loader, candidate_metrics, on_fit=None, check_budget=None):
    """Exactly three state-qualification fits at the common source-only alpha.

    This does not score any validation or protected rows. Its private fitted
    objects are for the caller's authenticated state export only. No source
    quality failures are discarded and no failed source fit is resumed here.
    """
    pairs, _ = source_roster(source_pairs, metadata)
    quality_gate = qualify_quality(pairs, quality_rows)
    selected = select_common_alpha(candidate_metrics)
    alpha = selected["selected_alpha"]
    features, targets = _source_maps(pairs, metadata, feature_loader, target_loader)
    fitted_models, manifests = {}, []
    for model in MODELS:
        if check_budget is not None:
            check_budget()
        fitted = fit_source_ridge(pairs, features[model], targets, alpha)
        fitted_models[model] = fitted
        manifest = {"fit": len(manifests), "stage": "full_source_state_qualification",
                    "model": model, "alpha": alpha,
                    "feature_dimension": 13 if model == "context_only" else 163,
                    "training_pairs": list(pairs), "training_rows": 980,
                    "source_identity_sha256": fitted["source_identity_sha256"],
                    "validation_rows_scored": 0, "protected_rows_scored": 0}
        manifests.append(manifest)
        if on_fit is not None:
            on_fit(manifest, fitted)
    _assert_matched_scalers(fitted_models)
    if check_budget is not None:
        check_budget()
    category_prior = estimate_priors(pairs, targets, metadata)
    return {"fits": manifests, "private_fitted_models": fitted_models,
            "category_prior": category_prior, "common_alpha_selection": selected,
            "quality_gate": quality_gate, "full_source_fit_count": 3,
            "full_source_prior_count": 1, "validation_access": False,
            "protected_access": False}


def source_plan(source_pairs, metadata, person_folds):
    """Metadata-only declaration suitable for freezing before source fitting."""
    cells = crossed_source_cells(source_pairs, metadata, person_folds)
    canonical = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
    return {"date": "2026-10-13", "status": "prepared_source_only_not_activated",
            "source_trials": 980, "source_regions_required": 2940,
            "cells": cells, "cells_sha256": hashlib.sha256(canonical).hexdigest(),
            "alphas": list(ALPHAS), "models": list(MODELS), "fit_cap": FIT_CAP,
            "ridge_cv_prediction_rows": EXPECTED_PREDICTIONS,
            "category_prior_prediction_rows": 840, "primary_fixed_alpha": PRIMARY_ALPHA,
            "common_alpha_rule": "Equal model mean pooled held MAE, equal model mean MSE, alpha order .1/10/100",
            "nomination_rule": "Joint coverage MAE at common alpha, then MSE, then prefer late",
            "nomination_scope": "Adaptive source-only development; not independent confirmation",
            "material_rule": "Lower numeric title held in assignment 0; upper in 1; Fear 9 and Inspiration 20 train-only",
            "distributed_rule": "Mean early/middle/late 150-dimensional log-band-power vectors",
            "category_median_mandatory": True, "matched_alpha_mandatory": True,
            "feature_scaling": "StandardScaler fit separately on the same 448 training rows per cell; SVD Ridge with intercept",
            "validation_access": False, "protected_access": False,
            "final_source_state_qualification_fit_cap": 3,
            "total_ridge_fit_cap_including_optional_state_qualification": 93,
            "source_prior_constructions_including_optional_full_source": 11}
