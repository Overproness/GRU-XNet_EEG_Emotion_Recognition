"""Independent public-only audit of the completed category-supported stage.

No import-time I/O, original-data reader, private-state access or model fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_faced_development_diagnosis_2026_10_12 import (
    REPO, BASE, QUALITY, ARMS, PRIORS, CONTEXT, ANCHORS, PRIORITY, ALPHAS,
    load, digest, rows, identity, close, boolean, score, check_metric,
    source_prior, prior_value,
)

PUBLIC = BASE / "faced_category_supported_context_2026-10-12"
PREVIOUS = BASE / "faced_target_context_pairing_2026-10-12"
CATEGORIES = {"Amusement", "Anger", "Disgust", "Fear", "Inspiration", "Joy", "Sadness", "Tenderness"}


def audit(*, write=True):
    execution = load(PUBLIC / "execution.json")
    if execution.get("status") != "completed_category_supported_context":
        raise PermissionError("Completed separate context qualification required before numeric audit")
    destination = PUBLIC / "independent_numerical_audit.json"
    if write and destination.exists():
        raise PermissionError("Preserve existing audit; no overwrite")
    activation = load(PUBLIC / "activation.json")
    expected_scope = {"new_fit_cap": 156, "source_prior_constructions_cap": 18,
                      "new_neural_fit_cap": 0, "confirmation_access": False,
                      "research_question_changed": False, "original_data_access": False}
    if any(activation.get(k) != v for k, v in expected_scope.items()):
        raise PermissionError("Changed qualification scope/count declaration")
    for bindings in (activation["bindings"], activation["prior_packet_bindings"]):
        for relative, expected in bindings.items():
            p = (REPO / relative).resolve()
            if not p.is_relative_to(REPO) or digest(p) != expected:
                raise AssertionError("Immutable input or previous evidence changed: " + relative)
    if digest(REPO.parent / "report.tex") != activation["original_report_sha256"]:
        raise AssertionError("Original report changed")
    prior_activation = load(PREVIOUS / "activation.json")
    canonical_path = QUALITY / "predictions.csv"
    if digest(canonical_path) != prior_activation["bindings"][canonical_path.relative_to(REPO).as_posix()]:
        raise AssertionError("Canonical historical target projection changed")
    programme = load(BASE / "faced_development_protocol_2026-10-11/plan.json")
    reservation = load(BASE / "faced_backup_2026-10-10/participant_reservation.json")
    catalogue = {r["clip"]: r for r in load(BASE / "faced_codebooks_2026-10-10/audit.json")["stimuli"]}
    records = load(BASE / "faced_artifact_control_2026-10-11/source_records.json")["records"] + load(BASE / "faced_artifact_control_2026-10-11/validation_records.json")["records"]
    metadata = {(r["subject"], r["vid"]): {**catalogue[r["vid"]], **r,
                 "family": catalogue[r["vid"]]["source_film_sha256"]} for r in records}
    if len(metadata) != 1260 or len(records) != 1260:
        raise AssertionError("Changed source/validation metadata rectangle")
    for p, m in metadata.items():
        intended = "source" if p[0] in reservation["development_source"] else "validation"
        if p[0] in reservation["confirmation"] or m["role"] != intended:
            raise PermissionError("Protected or misassigned metadata identity")
    targets = {}
    for r in rows(QUALITY / "predictions.csv"):
        p = identity(r, metadata)
        y = float(r["target"])
        if not np.isfinite(y) or not 0 <= y <= 7 or p in targets and targets[p] != y:
            raise AssertionError("Invalid/inconsistent canonical original-scale rating")
        targets[p] = y
    if set(targets) != set(metadata):
        raise AssertionError("Incomplete canonical development targets")
    arms = {(r["rotation"], name): {(q["subject"], q["vid"]) for q in values}
            for r in programme["rotation_manifests"] for name, values in r["arms"].items()}
    selection = load(PUBLIC / "source_selection.json")
    if selection["source_person_folds"] != activation["source_person_folds"] or selection["validation_selection"] is not False or selection["independent_title_validation"] is not False:
        raise AssertionError("Source selection scope or fold identity changed")
    folds = {}
    for k in range(3):
        source = arms[k, "source_fit"]
        source_titles = {p[1] for p in source}
        title_categories = Counter(catalogue[v]["assigned_category"] for v in source_titles)
        if title_categories != Counter({name: 1 for name in CATEGORIES}):
            raise AssertionError("One-title-per-category source support changed")
        all_held = []
        for f in selection["source_person_folds"][str(k)]:
            index = f["person_fold"]
            training, held = {tuple(p) for p in f["training"]}, {tuple(p) for p in f["held"]}
            hp = set(programme["source_participant_oof_folds"][index])
            if training != {p for p in source if p[0] not in hp} or held != {p for p in source if p[0] in hp} or len(training) != 448 or len(held) != 112:
                raise AssertionError("Source-person exclusion/held rectangle changed")
            if {metadata[p]["assigned_category"] for p in training} != CATEGORIES or {p[1] for p in training} != source_titles or {p[1] for p in held} != source_titles:
                raise AssertionError("Source category or familiar-title support missing")
            folds[k, index] = training, held
            all_held.extend(held)
        if Counter(all_held) != Counter(source):
            raise AssertionError("Source person-held coverage differs")
    manifest = load(PUBLIC / "model_manifest.json")["fits"]
    expected_fit_keys = {(k, name, "source_person_cv", alpha, fold) for k in range(3) for name in CONTEXT for alpha in ALPHAS for fold in range(5)}
    expected_fit_keys |= {(k, name, "selected_full_source", None, None) for k in range(3) for name in CONTEXT}
    observed_fit_keys = set()
    for r in manifest:
        k, stage = r["rotation"], r["stage"]
        key = (k, r["model"], stage, r["alpha"], r["person_fold"]) if stage == "source_person_cv" else (k, r["model"], stage, None, None)
        if key in observed_fit_keys or key not in expected_fit_keys:
            raise AssertionError("Duplicate or undeclared fit condition")
        observed_fit_keys.add(key)
        expected = folds[k, r["person_fold"]][0] if stage == "source_person_cv" else arms[k, "source_fit"]
        pairs = [tuple(p) for p in r["source_pairs"]]
        if set(pairs) != expected or len(pairs) != len(expected) or r["source_rows"] != len(expected):
            raise AssertionError("Context fit training identities changed")
        if hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest() != r["source_identity_sha256"]:
            raise AssertionError("Context source-identity hash differs")
    if len(manifest) != 156 or observed_fit_keys != expected_fit_keys:
        raise AssertionError("Incomplete 156-fit grid")
    states = {}
    prior_records = load(PUBLIC / "prior_manifest.json")["constructions"]
    for r in prior_records:
        k = r["rotation"]
        index = r["person_fold"] if r["stage"] == "source_person_cv" else None
        training = folds[k, index][0] if index is not None else arms[k, "source_fit"]
        if (k, index) in states or {tuple(p) for p in r["source_pairs"]} != training:
            raise AssertionError("Prior source identities or duplicate construction changed")
        state = source_prior(training, targets, metadata)
        for field, value in state.items():
            if field == "global":
                close(r["fitted"][field], value, "Source global median")
            else:
                if r["fitted"][field].keys() != value.keys():
                    raise AssertionError("Source prior group support changed")
                for group, prediction in value.items():
                    close(r["fitted"][field][group], prediction, "Source grouped median")
        states[k, index] = state
    if len(prior_records) != 18 or set(states) != {(k, index) for k in range(3) for index in (None, 0, 1, 2, 3, 4)}:
        raise AssertionError("Incomplete 18-prior construction grid")
    cv, cv_seen = defaultdict(list), set()
    for r in rows(PUBLIC / "cv_predictions.csv"):
        p = identity(r, metadata)
        k, fold, name = int(r["rotation"]), int(r["person_fold"]), r["model"]
        alpha = None if not r["alpha"] else float(r["alpha"])
        key = k, name, alpha, p
        if key in cv_seen or p not in folds[k, fold][1] or name not in PRIORITY or name in CONTEXT and alpha not in ALPHAS or name in PRIORS and alpha is not None:
            raise AssertionError("Duplicate or wrong person-held CV row")
        cv_seen.add(key)
        y, value = float(r["target"]), float(r["prediction"])
        close(y, targets[p], "CV original target", 0)
        if not np.isfinite(value) or not 0 <= value <= 7:
            raise AssertionError("Invalid CV prediction")
        if name in PRIORS:
            close(value, prior_value(name, states[k, fold], p, metadata), "Person-held source prior")
        cv[k, name, alpha].append({**r, "vid": p[1], "target": y, "prediction": value})
    if len(cv_seen) != 21840:
        raise AssertionError("Incomplete category-supported CV rows")
    selected, scores = {}, defaultdict(dict)
    for r in selection["selections"]:
        k, name = r["rotation"], r["model"]
        if [c["alpha"] for c in r["candidates"]] != list(ALPHAS):
            raise AssertionError("Candidate alpha sequence changed")
        for c in r["candidates"]:
            panel = cv[k, name, c["alpha"]]
            if {(q["subject"], q["vid"]) for q in panel} != arms[k, "source_fit"] or len(panel) != 560:
                raise AssertionError("Incomplete held source candidate rectangle")
            close(c["participant_title_mae"], score(panel), "Source-person CV MAE")
            close(c["trial_mse"], np.mean([(q["target"] - q["prediction"]) ** 2 for q in panel]), "Source-person CV MSE")
        winner = min(r["candidates"], key=lambda c: (c["participant_title_mae"], c["alpha"]))
        if winner["alpha"] != r["selected_alpha"]:
            raise AssertionError("Wrong source-only alpha selection")
        selected[k, name] = winner["alpha"]
        scores[k][name] = winner["participant_title_mae"]
    for k in range(3):
        for name in PRIORS:
            panel = cv[k, name, None]
            if {(r["subject"], r["vid"]) for r in panel} != arms[k, "source_fit"] or len(panel) != 560:
                raise AssertionError("Incomplete held source prior rectangle")
            scores[k][name] = score(panel)
        winner = min(PRIORITY, key=lambda name: (scores[k][name], PRIORITY.index(name)))
        actual = selection["strongest_comparators"][str(k)]
        if actual["model"] != winner or actual["tie_priority"] != list(PRIORITY):
            raise AssertionError("Wrong source-selected comparator/tie priority")
        for name in PRIORITY:
            close(actual["source_person_cv_mae"][name], scores[k][name], "Strongest source score")
    for r in manifest:
        if r["stage"] == "selected_full_source" and r["alpha"] != selected[r["rotation"], r["model"]]:
            raise AssertionError("Full-source refit alpha differs from source selection")
    old_anchors = {}
    for r in rows(PREVIOUS / "predictions.csv"):
        if r["model"] not in ANCHORS:
            continue
        p = identity(r, metadata)
        key = int(r["rotation"]), r["arm"], r["model"], p
        if key in old_anchors:
            raise AssertionError("Duplicate original anchor")
        old_anchors[key] = r
    panels, seen, anchor_checks = defaultdict(list), set(), 0
    for r in rows(PUBLIC / "predictions.csv"):
        p = identity(r, metadata)
        k, arm, name = int(r["rotation"]), r["arm"], r["model"]
        key = k, arm, name, p
        if key in seen or p not in arms[k, arm] or name not in PRIORITY + ANCHORS:
            raise AssertionError("Prediction duplicate, wrong role or wrong arm")
        seen.add(key)
        y, value = float(r["target"]), float(r["prediction"])
        close(y, targets[p], "Validation original target", 0)
        if not np.isfinite(value) or not 0 <= value <= 7:
            raise AssertionError("Invalid validation prediction")
        if name in PRIORS:
            close(value, prior_value(name, states[k, None], p, metadata), "Full-source prior")
        if name in CONTEXT and float(r["alpha"]) != selected[k, name]:
            raise AssertionError("Validation predictor alpha changed")
        if name in ANCHORS:
            previous = old_anchors[key]
            for field in ("prediction", "target", "family", "acquisition_group", "eeg_available", "fallback"):
                if r[field] != previous[field]:
                    raise AssertionError("Copied EEG anchor field is not byte-identical: " + field)
            close(value, float(previous["prediction"]), "Copied EEG anchor", 0)
            anchor_checks += 1
        panels[k, arm, name].append({**r, "vid": p[1], "target": y, "prediction": value})
    if len(seen) != 5880 or anchor_checks != 1680:
        raise AssertionError("Incomplete validation/anchor coverage")
    comparison = load(PUBLIC / "comparison.json")
    interval_checks = 0
    comparison_keys = set()
    for c in comparison["panels"]:
        k, arm = c["rotation"], c["arm"]
        if (k, arm) in comparison_keys:
            raise AssertionError("Duplicate paired comparison panel")
        comparison_keys.add((k, arm))
        winner = selection["strongest_comparators"][str(k)]["model"]
        if c["source_selected_comparator"] != winner:
            raise AssertionError("Comparator substituted using validation outcomes")
        for name in PRIORITY + ANCHORS:
            panel = panels[k, arm, name]
            if {(r["subject"], r["vid"]) for r in panel} != arms[k, arm] or len(panel) != len(arms[k, arm]):
                raise AssertionError("Incomplete validation predictor rectangle")
            check_metric(c["metrics"][name], panel, "Validation context/anchor metric")
        context = {(r["subject"], r["vid"]): r for r in panels[k, arm, winner]}
        for anchor in ANCHORS:
            cells = {}
            for r in panels[k, arm, anchor]:
                p = r["subject"], r["vid"]
                original = context[p]
                cells[p[0], r["family"]] = abs(original["target"] - original["prediction"]) - abs(r["target"] - r["prediction"])
            people, films = sorted({p for p, f in cells}), sorted({f for p, f in cells})
            if len(cells) != len(people) * len(films):
                raise AssertionError("Paired gain not a full person/title rectangle")
            matrix = np.array([[cells[p, f] for f in films] for p in people])
            rng = np.random.default_rng(425401)
            pd = rng.integers(len(people), size=(4000, len(people)))
            fd = rng.integers(len(films), size=(4000, len(films)))
            interval = np.quantile(matrix[pd[:, :, None], fd[:, None, :]].mean(axis=(1, 2)), [.025, .975])
            actual = c["eeg_vs_source_selected_context"][anchor]
            if actual["draws"] != 4000 or actual["seed"] != 425401:
                raise AssertionError("Paired interval draw/seed differs")
            close(actual["gain"], matrix.mean(), "EEG minus context improvement sign")
            for value, expected in zip(actual["descriptive_95_percent_gain_interval"], interval):
                close(value, expected, "Descriptive crossed paired gain interval")
            interval_checks += 1
    if interval_checks != 12 or comparison_keys != {(k, arm) for k in range(3) for arm in ARMS}:
        raise AssertionError("Incomplete paired gain comparison coverage")
    verification = load(PUBLIC / "verification.json")
    if verification.get("passed") is not True or verification["activation_sha256"] != digest(PUBLIC / "activation.json"):
        raise AssertionError("Passing verification bound to this activation required")
    for relative, expected in verification["bindings"].items():
        if digest(REPO / relative) != expected:
            raise AssertionError("Completed qualification output changed")
    result = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "frozen_bindings_checked": len(activation["bindings"]),
              "prior_packet_bindings_preserved": len(activation["prior_packet_bindings"]),
              "source_person_folds_checked": 15, "training_categories_per_fold": 8,
              "ridge_fit_identities_checked": 156, "source_prior_constructions_checked": 18,
              "canonical_targets_checked": 1260, "cv_prediction_rows_checked": 21840,
              "validation_prediction_rows_checked": 5880, "unchanged_anchor_rows_checked": 1680,
              "paired_crossed_gain_intervals_recomputed": interval_checks,
              "private_arrays_or_states_opened": 0, "original_data_files_opened": 0, "new_fits": 0,
              "confirmation_access": False, "research_question_changed": False,
              "limits": "Public prediction/identity/selection/median/interval audit; no independent Ridge refit or private-state forward replay. Person-held source tuning retains familiar source titles and is not unseen-title confirmation. Validation targets are adaptively reused.",
              "input_bindings": {p.relative_to(REPO).as_posix(): digest(p) for p in PUBLIC.iterdir() if p.is_file() and p != destination},
              "audit_script_sha256": digest(Path(__file__))}
    if write:
        with destination.open("x", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="Write first receipt; refuses overwrite")
    action.add_argument("--check", action="store_true", help="Public-only check without writing")
    args = parser.parse_args()
    print(json.dumps(audit(write=args.run)), flush=True)
