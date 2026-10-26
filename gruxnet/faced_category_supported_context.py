"""Bounded context qualification with every assigned category supported.

Source-person-only tuning answers a different development question from the
preceding crossed person/title tuning. It does not independently validate title
generalization. This module has no EEG, waveform or original-rating reader.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .faced_classical_contract import CATEGORIES, ALPHAS, context_features, predict_ridge, participant_film_mae
from .faced_classical_runner import estimate_priors, prior_predictions
from .faced_eegnet_bridge import array_sha256, guard_cached_identity
from .faced_development_diagnosis import (
    REPO, BASE, PROGRAMME, OLD_ARTIFACT, OLD_QUALITY, ARMS, PRIOR_NAMES,
    COMPARATOR_PRIORITY, FitLedger, metadata_bundle, load, save, sha256,
    _row, _write_csv, loss_metrics, crossed_loss_interval, cached_bool,
)

PREVIOUS = BASE / "faced_target_context_pairing_2026-10-12"
FIT_CAP = 156
TIME_CAP = 900
ANCHORS = ("joint_spectral", "eegnet_residual")


def required_binding_paths():
    paths = [PROGRAMME, BASE / "faced_backup_2026-10-10/participant_reservation.json",
             BASE / "faced_codebooks_2026-10-10/audit.json", OLD_ARTIFACT / "source_records.json",
             OLD_ARTIFACT / "validation_records.json", OLD_QUALITY / "signal_audit.json",
             PREVIOUS / "verification.json", PREVIOUS / "predictions.csv",
             PREVIOUS / "cv_predictions.csv", PREVIOUS / "execution.json"]
    paths += [REPO / x for x in ("gruxnet/faced_category_supported_context.py", "gruxnet/faced_development_diagnosis.py",
               "gruxnet/faced_classical_contract.py", "gruxnet/faced_classical_runner.py", "gruxnet/faced_eegnet_bridge.py",
               "gruxnet/faced_eegnet_runner.py", "gruxnet/faced_cached_targets.py")]
    return [p.relative_to(REPO).as_posix() for p in paths]


def person_folds(source_pairs, metadata, folds):
    """Five source-person folds, preserving all eight training categories."""
    source = list(source_pairs)
    if len(source) != len(set(source)) or any(p not in metadata or metadata[p]["role"] != "source" for p in source):
        raise PermissionError("Source-person fold contains duplicate or protected/nonsource identity")
    people, clips = {p[0] for p in source}, {p[1] for p in source}
    if len(people) != 70 or len(clips) != 8 or set(source) != {(p, v) for p in people for v in clips}:
        raise PermissionError("Require exactly the 70 by 8 source rectangle")
    flat = [p for fold in folds for p in fold]
    if len(folds) != 5 or any(len(fold) != 14 for fold in folds) or len(flat) != len(set(flat)) or set(flat) != people:
        raise PermissionError("Source-person folds changed")
    required = set(CATEGORIES)
    if {metadata[p]["assigned_category"] for p in source} != required:
        raise PermissionError("Source roster does not support all eight categories")
    result = []
    for index, held_people in enumerate(folds):
        training = [p for p in source if p[0] not in held_people]
        held = [p for p in source if p[0] in held_people]
        if len(training) != 448 or len(held) != 112 or {p[0] for p in training} & {p[0] for p in held}:
            raise PermissionError("Source-person fold size or participant exclusion failed")
        if {metadata[p]["assigned_category"] for p in training} != required or {p[1] for p in training} != clips:
            raise PermissionError("Held participants removed training category/title support")
        result.append({"person_fold": index, "training": training, "held": held,
                       "training_categories": list(CATEGORIES)})
    if Counter(p for f in result for p in f["held"]) != Counter(source):
        raise PermissionError("Source-person held coverage differs")
    return result


def select_alpha(candidates):
    if len(candidates) != len(ALPHAS) or [r["alpha"] for r in candidates] != ALPHAS:
        raise PermissionError("Changed declared context alpha grid")
    if any(not np.isfinite(r["participant_title_mae"]) for r in candidates):
        raise ValueError("Invalid source-person selection score")
    return min(candidates, key=lambda r: (r["participant_title_mae"], r["alpha"]))["alpha"]


def strongest_comparator(scores):
    if set(scores) != set(COMPARATOR_PRIORITY) or not np.isfinite(list(scores.values())).all():
        raise PermissionError("Missing or invalid source context comparator")
    return min(COMPARATOR_PRIORITY, key=lambda name: (scores[name], COMPARATOR_PRIORITY.index(name)))


class ContextFitLedger(FitLedger):
    def fit(self, *args, **kwargs):
        if len(self.records) >= FIT_CAP:
            raise PermissionError("Category-supported qualification 156-fit cap exceeded")
        return super().fit(*args, **kwargs)


def cached_development_inputs(programme, metadata, decisions):
    """Join two completed CSVs, guarding identity before numeric conversion."""
    targets, anchors, seen = {}, {}, set()
    allowed_models = set(COMPARATOR_PRIORITY) | set(ANCHORS)
    for name, source in (("source_cv", PREVIOUS / "cv_predictions.csv"), ("validation", PREVIOUS / "predictions.csv")):
        with source.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = guard_cached_identity(row, metadata)
                if row["acquisition_group"] != metadata[p]["acquisition_group"]:
                    raise PermissionError("Completed cache acquisition changed")
                rotation = int(row["rotation"])
                if rotation not in (0, 1, 2) or row["model"] not in allowed_models:
                    raise PermissionError("Changed cache rotation/model")
                manifest = programme["rotation_manifests"][rotation]
                arm = "source_fit" if name == "source_cv" else row["arm"]
                if arm not in ("source_fit",) + ARMS:
                    raise PermissionError("Unknown completed prediction arm")
                intended = {(r["subject"], r["vid"]) for r in manifest["arms"][arm]}
                expected_role = "source" if name == "source_cv" else "validation"
                if p not in intended or metadata[p]["role"] != expected_role:
                    raise PermissionError("Completed cache leaves declared role/rotation arm")
                target = float(row["target"])
                if not np.isfinite(target) or not 0 <= target <= 7:
                    raise ValueError("Invalid cached original-scale rating")
                if p in targets and targets[p] != target:
                    raise ValueError("Repeated original targets in completed cache disagree")
                targets[p] = target
                if name == "validation" and row["model"] in ANCHORS:
                    if cached_bool(row["eeg_available"]) != decisions[p]["eeg_available"] or cached_bool(row["fallback"]) != (not decisions[p]["eeg_available"]):
                        raise PermissionError("Copied EEG anchor availability/fallback changed")
                    key = rotation, arm, row["model"], p
                    if key in seen:
                        raise PermissionError("Duplicate copied EEG anchor")
                    seen.add(key)
                    prediction = float(row["prediction"])
                    if not np.isfinite(prediction) or not 0 <= prediction <= 7:
                        raise ValueError("Invalid copied EEG anchor prediction")
                    anchors[key] = prediction
    if set(targets) != set(metadata) or len(seen) != 1680:
        raise PermissionError("Completed source/validation target or EEG anchor coverage incomplete")
    return targets, anchors


def preserve_partial(public_dir, ledger, cv_rows, predictions, priors):
    save(Path(public_dir) / "partial_model_manifest.json", {"fits": ledger.records, "new_ridge_fits": len(ledger.records),
         "status": "partial_unqualified_attempt", "resumption_authorized": False})
    if cv_rows:
        _write_csv(Path(public_dir) / "partial_cv_predictions.csv", cv_rows)
    if predictions:
        _write_csv(Path(public_dir) / "partial_predictions.csv", predictions)
    if priors:
        save(Path(public_dir) / "partial_prior_manifest.json", {"constructions": priors})


def run(public_dir, private_dir, activation):
    """Run a separately frozen source-person context qualification exactly once."""
    public_dir, private_dir = Path(public_dir), Path(private_dir)
    if activation.get("confirmation_access") is not False or activation.get("research_question_changed") is not False or activation.get("original_data_access") is not False or activation.get("new_original_rating_decoding") != 0:
        raise PermissionError("Protected/original access or research-question change not allowed")
    if activation.get("new_fit_cap") != FIT_CAP or not 0 < activation.get("time_cap_seconds", 0) <= TIME_CAP:
        raise PermissionError("Changed bounded context qualification fit/time cap")
    if not set(required_binding_paths()) <= set(activation.get("bindings", {})):
        raise PermissionError("Missing category-supported input/implementation bindings")
    for name, expected in activation["bindings"].items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()) or sha256(path) != expected:
            raise PermissionError("Frozen qualification input changed")
    if load(PREVIOUS / "execution.json")["status"] != "completed_development_diagnosis":
        raise PermissionError("Previous diagnosis must be complete and bound")
    if any((public_dir / n).exists() for n in ("execution.json", "cv_predictions.csv", "model_manifest.json")) or private_dir.exists():
        raise PermissionError("Preserve existing/partial context qualification; no resume")
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=False)
    ledger = ContextFitLedger(private_dir, time_cap=activation["time_cap_seconds"])
    cv_rows, predictions, priors, selections = [], [], [], []
    try:
        programme, record_map, metadata, decisions = metadata_bundle()
        frozen = {str(r["rotation"]): person_folds([(q["subject"], q["vid"]) for q in r["arms"]["source_fit"]], metadata, programme["source_participant_oof_folds"]) for r in programme["rotation_manifests"]}
        if json.loads(json.dumps(frozen)) != activation.get("source_person_folds"):
            raise PermissionError("Source-person fold plan differs before outcomes")
        contexts = {p: context_features(m) for p, m in metadata.items()}
        if any(array_sha256(x) != record_map[p]["context_feature_sha256"] for p, x in contexts.items()):
            raise PermissionError("Metadata-reconstructed context bytes changed")
        save(public_dir / "context_reconstruction.json", {"context_vectors_verified": len(contexts), "private_arrays_opened": 0,
             "original_files_opened": 0, "protected_outcomes_opened": False, "source_person_folds_verified_before_targets": True})
        targets, anchors = cached_development_inputs(programme, metadata, decisions)
        save(public_dir / "target_reuse.json", {"cached_original_targets": len(targets), "sources": {p.relative_to(REPO).as_posix(): sha256(p) for p in (PREVIOUS / "cv_predictions.csv", PREVIOUS / "predictions.csv")},
             "new_original_rating_decoding": 0, "confirmation_access": False})
        maps = {"category_only": {p: x[:8].copy() for p, x in contexts.items()}, "full_context": contexts}
        strongest = {}
        with threadpool_limits(limits=2, user_api="blas"):
            for rotation in programme["rotation_manifests"]:
                k = rotation["rotation"]
                arms = {a: [(r["subject"], r["vid"]) for r in rows] for a, rows in rotation["arms"].items()}
                source = arms["source_fit"]
                folds = frozen[str(k)]
                scores = {}
                prior_cv = {name: [] for name in PRIOR_NAMES}
                for fold in folds:
                    fitted = estimate_priors(fold["training"], targets, metadata)
                    priors.append({"rotation": k, "stage": "source_person_cv", "person_fold": fold["person_fold"], "source_pairs": fold["training"], "fitted": fitted})
                    values = prior_predictions(fitted, fold["held"], metadata)
                    for name in PRIOR_NAMES:
                        rows = [_row(p, v, targets, metadata, rotation=k, model=name, alpha=None, person_fold=fold["person_fold"]) for p, v in zip(fold["held"], values[name])]
                        cv_rows.extend(rows)
                        prior_cv[name].extend(rows)
                full_priors = estimate_priors(source, targets, metadata)
                priors.append({"rotation": k, "stage": "full_source", "source_pairs": source, "fitted": full_priors})
                for name in PRIOR_NAMES:
                    scores[name] = participant_film_mae(prior_cv[name])
                for arm in ARMS:
                    values = prior_predictions(full_priors, arms[arm], metadata)
                    for name in PRIOR_NAMES:
                        predictions.extend(_row(p, v, targets, metadata, rotation=k, model=name, arm=arm, alpha=None, eeg_available=decisions[p]["eeg_available"], fallback=False) for p, v in zip(arms[arm], values[name]))
                for name, features in maps.items():
                    candidates = []
                    for alpha in ALPHAS:
                        held_rows = []
                        for fold in folds:
                            fitted = ledger.fit(fold["training"], features, targets, alpha, rotation=k, model=name, stage="source_person_cv", person_fold=fold["person_fold"])
                            values = predict_ridge(fitted, fold["held"], features)
                            rows = [_row(p, v, targets, metadata, rotation=k, model=name, alpha=alpha, person_fold=fold["person_fold"]) for p, v in zip(fold["held"], values)]
                            held_rows.extend(rows)
                            cv_rows.extend(rows)
                        candidates.append({"alpha": alpha, "participant_title_mae": participant_film_mae(held_rows), "trial_mse": float(np.mean([(r["target"] - r["prediction"]) ** 2 for r in held_rows]))})
                    alpha = select_alpha(candidates)
                    scores[name] = next(r["participant_title_mae"] for r in candidates if r["alpha"] == alpha)
                    fitted = ledger.fit(source, features, targets, alpha, rotation=k, model=name, stage="selected_full_source")
                    selections.append({"rotation": k, "model": name, "selected_alpha": alpha, "candidates": candidates,
                                       "selection_scope": "Source-person-only CV with all assigned categories supported", "validation_selection": False})
                    for arm in ARMS:
                        values = predict_ridge(fitted, arms[arm], features)
                        predictions.extend(_row(p, v, targets, metadata, rotation=k, model=name, arm=arm, alpha=alpha, eeg_available=decisions[p]["eeg_available"], fallback=False) for p, v in zip(arms[arm], values))
                strongest[k] = {"model": strongest_comparator(scores), "source_person_cv_mae": scores, "tie_priority": list(COMPARATOR_PRIORITY)}
                for arm in ARMS:
                    for name in ANCHORS:
                        predictions.extend(_row(p, anchors[k, arm, name, p], targets, metadata, rotation=k, model=name, arm=arm, alpha=None, eeg_available=decisions[p]["eeg_available"], fallback=not decisions[p]["eeg_available"]) for p in arms[arm])
                print({"phase": "category_supported_context_rotation", "rotation": k, "fits": len(ledger.records), "elapsed_seconds": round(time.monotonic() - ledger.started, 1)}, flush=True)
        if len(ledger.records) != FIT_CAP or len(priors) != 18 or len(cv_rows) != 21840 or len(predictions) != 5880:
            raise PermissionError("Incomplete category-supported context coverage")
        comparisons = []
        for k in range(3):
            for arm in ARMS:
                panel = {name: [r for r in predictions if r["rotation"] == k and r["arm"] == arm and r["model"] == name] for name in COMPARATOR_PRIORITY + ANCHORS}
                comparator = strongest[k]["model"]
                intervals = {}
                for anchor in ANCHORS:
                    result = crossed_loss_interval(panel[comparator], [panel[anchor]], draws=4000, seed=425401)
                    intervals[anchor] = {"gain": -result["mean_loss_delta"], "descriptive_95_percent_gain_interval": [-result["descriptive_95_percent_interval"][1], -result["descriptive_95_percent_interval"][0]], "draws": 4000, "seed": 425401,
                                         "limits": "Reused development with source-person-only context selection; fixed previously selected EEG anchors; descriptive uncertainty, no independent confirmation."}
                comparisons.append({"rotation": k, "arm": arm, "source_selected_comparator": comparator,
                                    "metrics": {name: loss_metrics(rows) for name, rows in panel.items()}, "eeg_vs_source_selected_context": intervals})
        ledger.check()
        _write_csv(public_dir / "cv_predictions.csv", cv_rows)
        _write_csv(public_dir / "predictions.csv", predictions)
        save(public_dir / "source_selection.json", {"source_person_folds": frozen, "selections": selections, "strongest_comparators": strongest,
             "validation_selection": False, "alpha_tie_rule": "MAE, then smaller alpha", "independent_title_validation": False})
        save(public_dir / "model_manifest.json", {"fits": ledger.records, "new_ridge_fits": FIT_CAP, "new_eeg_fits": 0})
        save(public_dir / "prior_manifest.json", {"constructions": priors, "source_prior_constructions": len(priors), "prediction_rules": list(PRIOR_NAMES)})
        save(public_dir / "comparison.json", {"panels": comparisons, "new_ridge_fits": FIT_CAP, "source_prior_constructions": 18,
             "validation_selection": False, "source_selection_scope": "Person-held source CV; all eight categories supported; familiar source titles within selection", "independent_title_validation": False,
             "eeg_anchor_recipe_changed": False, "new_eeg_feature_or_model_access": False, "adaptively_reused_development": True,
             "independent_confirmation": False, "confirmation_access": False, "research_question_changed": False})
        save(public_dir / "execution.json", {"status": "completed_category_supported_context", "new_ridge_fits": len(ledger.records), "source_prior_constructions": len(priors),
             "elapsed_seconds": time.monotonic() - ledger.started, "cv_prediction_rows": len(cv_rows), "prediction_rows": len(predictions),
             "new_eeg_fits": 0, "private_eeg_arrays_opened": 0, "original_files_opened": 0, "new_original_rating_decoding": 0,
             "confirmation_access": False, "research_question_changed": False})
        return load(public_dir / "comparison.json")
    except BaseException as exc:
        preserve_partial(public_dir, ledger, cv_rows, predictions, priors)
        if not (public_dir / "execution.json").exists():
            save(public_dir / "execution.json", {"status": "held_or_failed", "error_type": type(exc).__name__, "error": str(exc),
                 "new_ridge_fits": len(ledger.records), "source_prior_constructions": len(priors), "new_eeg_fits": 0,
                 "elapsed_seconds": time.monotonic() - ledger.started, "original_files_opened": 0, "new_original_rating_decoding": 0,
                 "confirmation_access": False, "research_question_changed": False})
        raise
