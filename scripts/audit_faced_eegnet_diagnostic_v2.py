"""Versioned independent audit: true-label joins exclude known permutation controls.

No waveform/rating decoder, producing runner or training module is imported.
Identities and complete axes are checked before numeric target conversion.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

READOUTS = ("eeg_only", "context_residual")
RATES = (.0003, .001, .003)
SEEDS = (20261101, 20261102)
EPOCHS = (30, 60, 120)
ARMS = ("source_fit", "source_inference", "validation_unseen", "validation_familiar")
CONTROLS = ("global_median", "coarse_valence_median", "category_median", "context_only", "context_kernel")
PERMUTED_CONTROLS = ("capacity_spectral_permuted", "capacity_covariance_permuted")
CATEGORIES = ("Amusement", "Anger", "Disgust", "Fear", "Inspiration", "Joy", "Sadness", "Tenderness")
ACQUISITIONS = ("legacy_1000", "legacy_250", "modern_1000")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = "results/development/faced_eegnet_diagnostic_v2_2026-10-11"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(value):
    if value is True or value in ("true", "True", "1"):
        return True
    if value is False or value in ("false", "False", "0"):
        return False
    raise ValueError("Non-boolean availability/fallback")


def finite(value, *, bounded=False):
    number = float(value)
    if not math.isfinite(number) or (bounded and not 0 <= number <= 7):
        raise ValueError("Invalid finite original-scale value")
    return number


def close(a, b, label, tolerance=2e-9):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a)-float(b)) > tolerance:
        raise ValueError(label+" disagreement")


def pair(row):
    return row["subject"], int(row["vid"])


@dataclass
class Axes:
    arms: dict
    metadata: dict
    decisions: dict
    folds: list

    @property
    def allowed(self):
        return set(self.metadata)


def derive_axes(programme, roles, catalogue, technical, decisions):
    """Join only public role, material, acquisition and fixed QC metadata."""
    source_people = set(roles["development_source"])
    validation_people = set(roles["development_validation"])
    protected = set(roles["confirmation"])
    if len(source_people) != 70 or len(validation_people) != 20 or len(protected) != 33 or source_people & validation_people or protected & (source_people | validation_people):
        raise PermissionError("Changed reserved participant roles")
    declared = [pair(row) for row in programme["input_access"]["future_development_pairs"]]
    clips = {v for _, v in declared}
    expected = {(p, v) for p in source_people | validation_people for v in clips}
    if len(clips) != 14 or len(declared) != 1260 or len(set(declared)) != 1260 or set(declared) != expected:
        raise PermissionError("Changed complete development bundle")
    catalog = {int(r["clip"]): r for r in catalogue["records"]}
    tech = {pair(r): r for r in technical}
    quality = {pair(r): r for r in decisions}
    if len(tech) != len(technical) or len(quality) != len(decisions) or set(tech) != expected or set(quality) != expected:
        raise PermissionError("Technical or QC identity disagreement")
    metadata = {}
    for p in expected:
        role = "source" if p[0] in source_people else "validation"
        t, q, c = tech[p], quality[p], catalog[p[1]]
        if t["role"] != role or q["role"] != role or q["policy_id"] != "faced-quality-context-fallback-v1" or type(q["eeg_available"]) is not bool:
            raise PermissionError("Fixed quality/role disagreement")
        if q["eeg_available"] == bool(q["reasons"]) or (role == "source" and not q["eeg_available"]):
            raise PermissionError("Contradictory or failed source QC")
        if t["acquisition_group"] not in ACQUISITIONS or c["assigned_emotion"] not in CATEGORIES or not 1 <= int(t["trial"]) <= 28 or not math.isfinite(float(c["duration_seconds"])) or float(c["duration_seconds"]) <= 0:
            raise PermissionError("Changed declared context geometry")
        metadata[p] = {"family": c["source_film_sha256"], "acquisition_group": t["acquisition_group"],
                       "category": c["assigned_emotion"], "coarse": c["assigned_valence"],
                       "duration": float(c["duration_seconds"]), "trial": int(t["trial"]), "role": role,
                       "processed_window_sha256": t["processed_window_sha256"]}
    arms = {}
    if [r["rotation"] for r in programme["rotation_manifests"]] != [0, 1, 2]:
        raise PermissionError("Changed rotation order")
    for rotation in programme["rotation_manifests"]:
        r = int(rotation["rotation"])
        panels = {arm: {pair(x) for x in rows} for arm, rows in rotation["arms"].items()}
        if set(panels) != {"source_fit", "validation_unseen", "validation_familiar"}:
            raise PermissionError("Changed diagnostic arms")
        for arm, count, people, nclips in (("source_fit", 560, source_people, 8), ("validation_unseen", 120, validation_people, 6), ("validation_familiar", 160, validation_people, 8)):
            pairs = panels[arm]; materials = {v for _, v in pairs}
            if len(pairs) != count or len(rotation["arms"][arm]) != count or len(materials) != nclips or pairs != {(p, v) for p in people for v in materials}:
                raise PermissionError("Changed or incomplete original-trial rectangle")
        if {v for _, v in panels["source_fit"]} != {v for _, v in panels["validation_familiar"]}:
            raise PermissionError("Changed familiar materials")
        if {metadata[p]["family"] for p in panels["source_fit"]} & {metadata[p]["family"] for p in panels["validation_unseen"]}:
            raise PermissionError("Title-family leakage")
        panels["source_inference"] = panels["source_fit"]
        arms[r] = panels
    folded = [p for fold in programme["source_participant_oof_folds"] for p in fold]
    if len(folded) != 70 or len(set(folded)) != 70 or set(folded) != source_people or len(programme["source_participant_oof_folds"]) != 5:
        raise PermissionError("Changed source OOF folds")
    if sum(not q["eeg_available"] for q in quality.values()) != 15:
        raise PermissionError("Changed fixed validation fallback coverage")
    return Axes(arms, metadata, quality, programme["source_participant_oof_folds"])


def identity_guard(rows, axes, *, controls=False):
    """Never convert target/prediction fields until every identity passes."""
    for row in rows:
        p = pair(row)
        if p not in axes.allowed or row["family"] != axes.metadata[p]["family"]:
            raise PermissionError("Protected or changed trial/material identity")
        if not controls and row.get("acquisition_group") != axes.metadata[p]["acquisition_group"]:
            raise PermissionError("Acquisition identity disagreement")
        rotation = int(row["rotation"])
        arm = "validation_unseen" if controls else row["arm"]
        if rotation not in axes.arms or arm not in axes.arms[rotation] or p not in axes.arms[rotation][arm]:
            raise PermissionError("Trial outside declared rotation/arm")
        if controls:
            if row["model"] not in CONTROLS:
                raise PermissionError("Undeclared no-EEG control")
        else:
            if row["readout"] not in READOUTS or boolean(row["eeg_available"]) != axes.decisions[p]["eeg_available"] or boolean(row["fallback"]) == axes.decisions[p]["eeg_available"]:
                raise PermissionError("Readout or availability/fallback disagreement")


def balanced_mae(pairs, predictions, targets, metadata):
    cells = {}
    for p in pairs:
        cells.setdefault((p[0], metadata[p]["family"]), []).append(abs(targets[p]-predictions[p]))
    people = {}
    for (person, _), errors in cells.items():
        people.setdefault(person, []).append(math.fsum(errors)/len(errors))
    return math.fsum(math.fsum(values)/len(values) for values in people.values())/len(people)


def learning_flags(baseline, at60, at120, seed_errors):
    values = [baseline, *at60, *at120, *seed_errors]
    if len(at60) != 2 or len(at120) != 2 or len(seed_errors) != 2 or any(not math.isfinite(v) or v < 0 for v in values):
        return ["invalid_or_nonfinite_learning_metrics"]
    flags = []
    if any(v > .8*baseline for v in at120): flags.append("insufficient_source_learning")
    if any(last < .9*mid for mid, last in zip(at60, at120)): flags.append("budget_limited_convergence")
    if abs(seed_errors[0]-seed_errors[1]) > .15: flags.append("validation_seed_divergence")
    if max(at120) > 2*max(min(at120), 1e-8): flags.append("source_seed_divergence")
    return flags


def context_vector(metadata):
    return np.array([float(metadata["category"] == c) for c in CATEGORIES]+[float(metadata["acquisition_group"] == a) for a in ACQUISITIONS]+[metadata["duration"], metadata["trial"]])


def direct_ridge(training, evaluation, targets, metadata, alpha):
    """Independent population standardization + centered SVD Ridge solve."""
    x = np.stack([context_vector(metadata[p]) for p in training])
    y = np.array([targets[p] for p in training])
    mean = x.mean(axis=0); sd = x.std(axis=0); sd[sd == 0] = 1
    standardized = (x-mean)/sd; intercept = y.mean()
    # Center again to match an intercept fitted independently of standardization.
    xm = standardized.mean(axis=0)
    u, s, vt = np.linalg.svd(standardized-xm, full_matrices=False)
    coefficient = vt.T @ ((s/(s*s+alpha))*(u.T @ (y-intercept)))
    held = (np.stack([context_vector(metadata[p]) for p in evaluation])-mean)/sd
    return {p: float(v) for p, v in zip(evaluation, np.clip((held-xm)@coefficient+intercept, 0, 7))}


def source_offsets(axes, targets):
    outputs = {}; comparators = {}
    for rotation, arms in axes.arms.items():
        source = sorted(arms["source_fit"]); unseen = sorted(arms["validation_unseen"])
        all_pairs = sorted(set().union(*arms.values())); alpha = (.1, 100., 100.)[rotation]
        full = direct_ridge(source, all_pairs, targets, axes.metadata, alpha)
        oof = {}
        for people in axes.folds:
            training = [p for p in source if p[0] not in people]; held = [p for p in source if p[0] in people]
            oof.update(direct_ridge(training, held, targets, axes.metadata, alpha))
        median = float(np.median([targets[p] for p in source]))
        global_offsets = {p: median for p in all_pairs}
        residual = dict(full); residual.update(oof)
        outputs[rotation] = {"eeg_only": {"fit": global_offsets, "inference": global_offsets}, "context_residual": {"fit": residual, "inference": full}}
        coarse = {}; categories = {}
        for key, destination in (("coarse", coarse), ("category", categories)):
            groups = {axes.metadata[p][key] for p in source}
            medians = {g: float(np.median([targets[p] for p in source if axes.metadata[p][key] == g])) for g in groups}
            destination.update({p: medians.get(axes.metadata[p][key], median) for p in unseen})
        comparators[rotation] = {"global_median": {p: median for p in unseen}, "coarse_valence_median": coarse, "category_median": categories, "context_only": {p: full[p] for p in unseen}}
    return outputs, comparators


def chosen_schedule(checkpoints, axes, targets, readout):
    pairs = axes.arms[0]["validation_unseen"]
    candidates = []
    for rate in RATES:
        for epoch in EPOCHS:
            values = {p: min(7., max(0., math.fsum(checkpoints[0, readout, rate, seed, epoch, "validation_unseen"][p] for seed in SEEDS)/2)) for p in pairs}
            candidates.append({"learning_rate": rate, "epoch": epoch, "mae": balanced_mae(pairs, values, targets, axes.metadata)})
    best = min(c["mae"] for c in candidates)
    return min((c for c in candidates if c["mae"] <= best+1e-8), key=lambda c: (c["epoch"], c["learning_rate"])), candidates


def audit_results(axes, checkpoint_rows, ensemble_rows, fit_records, control_rows, execution):
    # All tables are guarded in full, including unused alternatives, before a target is converted.
    for rows, controls in ((checkpoint_rows, False), (ensemble_rows, False), (control_rows, True)):
        identity_guard(rows, axes, controls=controls)
    targets = {}; checkpoints = {}; ensembles = {}; controls = {}
    def target(row):
        p = pair(row); value = finite(row["target"], bounded=True)
        if p in targets and targets[p] != value: raise ValueError("Target disagreement across rows")
        targets[p] = value
    for row in checkpoint_rows:
        p = pair(row); target(row)
        key = (int(row["rotation"]), row["readout"], float(row["learning_rate"]), int(row["seed"]), int(row["epoch"]), row["arm"])
        panel = checkpoints.setdefault(key, {})
        if p in panel: raise ValueError("Duplicate checkpoint trial")
        panel[p] = finite(row["prediction_raw"])
    for row in ensemble_rows:
        p = pair(row); target(row)
        key = (int(row["rotation"]), row["readout"], row["arm"])
        panel = ensembles.setdefault(key, {})
        if p in panel: raise ValueError("Duplicate ensemble trial")
        panel[p] = finite(row["prediction"], bounded=True)
    for row in control_rows:
        p = pair(row); target(row); key = (int(row["rotation"]), row["model"])
        panel = controls.setdefault(key, {})
        if p in panel: raise ValueError("Duplicate control trial")
        panel[p] = finite(row["prediction"], bounded=True)
    if set(targets) != axes.allowed: raise ValueError("Incomplete target coverage")
    expected0 = {(0, head, rate, seed) for head in READOUTS for rate in RATES for seed in SEEDS}
    for key in expected0:
        for epoch in EPOCHS:
            for arm in ARMS:
                if set(checkpoints.get((*key, epoch, arm), {})) != axes.arms[0][arm]: raise ValueError("Incomplete first selection panel")
    selections = {head: chosen_schedule(checkpoints, axes, targets, head)[0] for head in READOUTS}
    expected_fits = expected0 | {(r, head, selections[head]["learning_rate"], seed) for r in (1, 2) for head in READOUTS for seed in SEEDS}
    expected_checkpoints = {(*key, epoch, arm) for key in expected_fits for epoch in EPOCHS for arm in ARMS}
    if set(checkpoints) != expected_checkpoints: raise ValueError("Extra or missing checkpoint schedule")
    for key, values in checkpoints.items():
        if set(values) != axes.arms[key[0]][key[-1]]: raise ValueError("Incomplete checkpoint rectangle")
    expected_ensembles = {(r, head, arm) for r in (0, 1, 2) for head in READOUTS for arm in ARMS}
    if set(ensembles) != expected_ensembles: raise ValueError("Extra or missing ensemble panel")
    for key, values in ensembles.items():
        r, head, arm = key; schedule = selections[head]
        if set(values) != axes.arms[r][arm]: raise ValueError("Incomplete ensemble rectangle")
        for p in values:
            raw = math.fsum(checkpoints[r, head, schedule["learning_rate"], seed, schedule["epoch"], arm][p] for seed in SEEDS)/2
            close(values[p], min(7., max(0., raw)), "Post-ensemble clipping")
    for row in ensemble_rows:
        chosen = selections[row["readout"]]
        close(float(row.get("selected_learning_rate", row.get("learning_rate"))), chosen["learning_rate"], "Transferred ensemble rate")
        if int(row.get("selected_epoch", row.get("epoch"))) != chosen["epoch"]: raise ValueError("Transferred ensemble epoch disagreement")
    expected_controls = {(r, name) for r in (0, 1, 2) for name in CONTROLS}
    if set(controls) != expected_controls or any(set(panel) != axes.arms[r]["validation_unseen"] for (r, _), panel in controls.items()): raise ValueError("Incomplete stronger comparator rectangles")
    offsets, independent_controls = source_offsets(axes, targets)
    for r, names in independent_controls.items():
        for name, values in names.items():
            for p, value in values.items(): close(controls[r, name][p], value, "Independent source context "+name, 2e-8)
    records = fit_records.get("fits", fit_records) if isinstance(fit_records, dict) else fit_records
    keyed = {(int(f["rotation"]), f["readout"], float(f["learning_rate"]), int(f["seed"])): f for f in records}
    if len(records) != 20 or set(keyed) != expected_fits: raise ValueError("Incomplete or changed 20-fit budget")
    mse = {}; baselines = {}
    for key, record in keyed.items():
        r, head, rate, seed = key; source = axes.arms[r]["source_fit"]
        if record.get("status", "completed") != "completed": raise ValueError("Failed fit cannot qualify diagnostic")
        history = record["history"]
        if [int(h["epoch"]) for h in history] != list(range(1, 121)): raise ValueError("Incomplete fixed training history")
        for h in history:
            if finite(h["normalized_training_mse"]) < 0: raise ValueError("Negative training loss")
        baseline = math.fsum((targets[p]-offsets[r][head]["fit"][p])**2 for p in source)/len(source)
        close(record["starting_control_mse"], baseline, "Independent starting-control MSE", 2e-8)
        baselines[key] = baseline
        stats = record["checkpoints"]
        if {int(e) for e in stats} != set(EPOCHS): raise ValueError("Incomplete checkpoint stats")
        for epoch in EPOCHS:
            fit_values = checkpoints[*key, epoch, "source_fit"]
            inf_values = checkpoints[*key, epoch, "source_inference"]
            mse[key, epoch] = math.fsum((targets[p]-fit_values[p])**2 for p in source)/len(source)
            stat = stats.get(str(epoch), stats.get(epoch))
            close(stat["training_target_source_mse"], mse[key, epoch], "Unbounded training-target MSE", 2e-8)
            inference_mse = math.fsum((targets[p]-inf_values[p])**2 for p in source)/len(source)
            close(stat["source_inference_mse"], inference_mse, "Unbounded source inference MSE", 2e-8)
            state_sha = stat.get("state_dict_sha256", stat.get("state_sha256"))
            if not isinstance(state_sha, str) or len(state_sha) != 64 or any(c not in "0123456789abcdef" for c in state_sha): raise ValueError("Missing private state hash")
            for p in source:
                close(inf_values[p]-fit_values[p], offsets[r][head]["inference"][p]-offsets[r][head]["fit"][p], "OOF/full-context source offset", 2e-7)
            for arm in ("validation_unseen", "validation_familiar"):
                for p, value in checkpoints[*key, epoch, arm].items():
                    if not axes.decisions[p]["eeg_available"]: close(value, offsets[r][head]["inference"][p], "Unavailable EEG fallback", 2e-6)
    outputs = []
    for r in (0, 1, 2):
        unseen = axes.arms[r]["validation_unseen"]
        control_maes = {name: balanced_mae(unseen, controls[r, name], targets, axes.metadata) for name in CONTROLS}
        strongest = min(CONTROLS, key=lambda name: (control_maes[name], CONTROLS.index(name)))
        for head in READOUTS:
            schedule = selections[head]; keys = [(r, head, schedule["learning_rate"], seed) for seed in SEEDS]
            seed_errors = [balanced_mae(unseen, {p: min(7., max(0., v)) for p, v in checkpoints[*key, schedule["epoch"], "validation_unseen"].items()}, targets, axes.metadata) for key in keys]
            flags = learning_flags(baselines[keys[0]], [mse[key, 60] for key in keys], [mse[key, 120] for key in keys], seed_errors)
            error = balanced_mae(unseen, ensembles[r, head, "validation_unseen"], targets, axes.metadata)
            ensemble = ensembles[r, head, "validation_unseen"]
            reference = controls[r, strongest]
            family_omission = {}
            for family in sorted({axes.metadata[p]["family"] for p in unseen}):
                subset = {p for p in unseen if axes.metadata[p]["family"] != family}
                family_omission[family] = balanced_mae(subset, reference, targets, axes.metadata)-balanced_mae(subset, ensemble, targets, axes.metadata)
            acquisitions = {}
            for acquisition in sorted({axes.metadata[p]["acquisition_group"] for p in unseen}):
                subset = {p for p in unseen if axes.metadata[p]["acquisition_group"] == acquisition}
                acquisitions[acquisition] = {"participants": len({p[0] for p in subset}), "trials": len(subset), "gain": balanced_mae(subset, reference, targets, axes.metadata)-balanced_mae(subset, ensemble, targets, axes.metadata)}
            auxiliary = {arm: {"participant_family_mae": balanced_mae(axes.arms[r][arm], ensembles[r, head, arm], targets, axes.metadata), "trials": len(axes.arms[r][arm]), "fallback_trials": sum(not axes.decisions[p]["eeg_available"] for p in axes.arms[r][arm])} for arm in ARMS}
            outputs.append({"rotation": r, "readout": head, "schedule": schedule, "learning_flags": flags, "strongest_context": strongest, "control_maes": control_maes, "gain": control_maes[strongest]-error, "joint_mae": error,
                            "descriptive_leave_one_family_out_gain": family_omission, "descriptive_acquisition_gain": acquisitions, "arm_metrics": auxiliary})
    leads = {head: all(not row["learning_flags"] and row["gain"] > 0 for row in outputs if row["readout"] == head) and math.fsum(row["gain"] for row in outputs if row["readout"] == head)/3 >= .15 for head in READOUTS}
    if execution["fits"] != 20 or execution.get("confirmation_access") is not False or execution.get("research_question_changed") is not False: raise ValueError("Execution budget/confirmation contract disagreement")
    for head in READOUTS:
        selected = execution["selection"][head]
        for key in ("learning_rate", "epoch", "mae"): close(selected[key], selections[head][key], "Rotation-zero schedule")
    indexed = {(r["rotation"], r["readout"]): r for r in execution["rotations"]}
    if len(execution["rotations"]) != 6 or set(indexed) != {(r, head) for r in (0, 1, 2) for head in READOUTS}: raise ValueError("Incomplete reported output panels")
    for row in outputs:
        observed = indexed[row["rotation"], row["readout"]]
        if observed["learning_flags"] != row["learning_flags"] or observed["strongest_context"] != row["strongest_context"]: raise ValueError("Learning/comparator selection disagreement")
        for key in ("joint_mae", "gain"): close(observed[key], row[key], "Reported primary metric")
        for name in CONTROLS: close(observed["control_maes"][name], row["control_maes"][name], "Reported control metric")
    if execution["engineering_leads"] != leads: raise ValueError("Engineering lead disagreement")
    charged = execution.get("charged_fit_seconds")
    if charged is not None and (finite(charged) <= 0 or float(charged) >= 21600): raise ValueError("Fixed GPU-time budget disagreement")
    return {"passed": True, "fits_audited": 20, "checkpoint_rows": len(checkpoint_rows), "ensemble_rows": len(ensemble_rows), "control_rows": len(control_rows), "development_targets": len(targets), "selection": selections, "rotations": outputs, "engineering_leads": leads,
            "independent_source_context_svd": True, "original_rating_or_waveform_files_opened": False, "protected_outcomes_opened": False, "new_fits": 0}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_cached_projection(axes, old_quality, old_representation, new_controls, new_checkpoints):
    """Post-completion only: exact projection from authorized public caches.

    Both entire old tables are identity-guarded before reading numeric outcomes,
    including alternatives not retained in the diagnostic's controls table.
    """
    for rows, origin in ((old_quality, "quality"), (old_representation, "representation")):
        for row in rows:
            p = pair(row)
            if p not in axes.allowed or row["family"] != axes.metadata[p]["family"]:
                raise PermissionError("Protected or changed old cached outcome identity")
            if "acquisition_group" in row and row["acquisition_group"] != axes.metadata[p]["acquisition_group"]:
                raise PermissionError("Old cached acquisition disagreement")
            if int(row["rotation"]) not in (0, 1, 2):
                raise PermissionError("Changed cached rotation")
            if row["model"] in PERMUTED_CONTROLS and (origin != "representation" or row["arm"] != "capacity_source_fit" or p not in axes.arms[int(row["rotation"])]["source_fit"]):
                raise PermissionError("Known synthetic-control model outside exact declared source arm")
    identity_guard(new_controls, axes, controls=True)
    identity_guard(new_checkpoints, axes)
    targets = {}
    for row in old_quality:
        p = pair(row); value = finite(row["target"], bounded=True)
        if p in targets and targets[p] != value:
            raise ValueError("Authoritative quality-cache original targets inconsistent")
        targets[p] = value
    if set(targets) != axes.allowed:
        raise ValueError("Incomplete authoritative original target cache")
    for row in new_checkpoints:
        if finite(row["target"], bounded=True) != targets[pair(row)]:
            raise ValueError("New checkpoint original target differs from authoritative quality cache")
    expected = {}
    skipped = set()
    expected_skipped = {(r, model, p) for r in (0, 1, 2) for model in PERMUTED_CONTROLS for p in axes.arms[r]["source_fit"]}
    for rows, origin in ((old_quality, "quality"), (old_representation, "representation")):
        for row in rows:
            p = pair(row)
            rotation = int(row["rotation"]); model = row["model"]
            if origin == "representation" and model in PERMUTED_CONTROLS and row["arm"] == "capacity_source_fit":
                key = rotation, model, p
                if key in skipped:
                    raise ValueError("Duplicate known synthetic-control row")
                skipped.add(key)
                # Deliberately never decode this synthetic target as an original label.
                continue
            if finite(row["target"], bounded=True) != targets[p]:
                raise ValueError("Original cached development target changed")
            if row["arm"] != "validation_unseen":
                continue
            if origin == "quality" and model in CONTROLS[:4]:
                if model == "context_only" and float(row["alpha"]) != (.1, 100., 100.)[rotation]:
                    continue
            elif origin == "representation" and model == "context_kernel":
                # Kernel alpha is fixed in the bound source plan, absent from this CSV.
                pass
            else:
                continue
            key = rotation, model, p
            if key in expected:
                raise ValueError("Duplicate selected cached comparator row")
            expected[key] = (finite(row["target"], bounded=True), finite(row["prediction"], bounded=True))
    if skipped != expected_skipped:
        raise ValueError("Changed/missing exact synthetic-control coverage")
    projected = {(int(row["rotation"]), row["model"], pair(row)): (finite(row["target"], bounded=True), finite(row["prediction"], bounded=True)) for row in new_controls}
    if expected != projected:
        raise ValueError("Selected cached controls were not exactly reused")
    return {"rows": len(projected), "old_quality_rows_identity_checked": len(old_quality), "old_representation_rows_identity_checked": len(old_representation), "known_synthetic_control_rows_not_parsed_as_original_targets": len(skipped), "original_targets_source": "Authoritative quality cache only", "kernel_alpha_source": "Bound representation plan, fixed context-only RBF alpha0.1; no CSV alpha column", "exact_targets_and_all_five_control_predictions_reused": True}


def audit_packet(repo=REPO, packet=None):
    repo = Path(repo).resolve(); packet = Path(packet or repo/DEFAULT_PACKET).resolve()
    base = repo/"results/development"
    activation = load(packet/"activation.json")
    if activation.get("status") != "active_development_diagnostic" or activation.get("confirmation_access") is not False or activation.get("research_question_changed") is not False: raise PermissionError("Invalid frozen activation")
    if activation.get("preflight_plan_sha256") != sha(base/"faced_eegnet_preflight_2026-10-11/plan.json"): raise PermissionError("Preflight plan hash mismatch")
    count = 0
    for name in ("bindings", "input_bindings", "code_bindings"):
        for rel, binding in activation.get(name, {}).items():
            value = binding.get("sha256") if isinstance(binding, dict) else binding
            if sha(repo/rel) != value: raise PermissionError("Activation bound input changed: "+rel)
            count += 1
    if not activation.get("code_bindings"): raise PermissionError("Missing activation code bindings")
    preflight = load(base/"faced_eegnet_preflight_2026-10-11/plan.json")
    for rel, value in preflight["bindings"].items():
        if sha(repo/rel) != value: raise PermissionError("Preflight immutable scientific input changed: "+rel)
    if preflight["fit_cap"] != 20 or preflight["gpu_fit_seconds_cap"] != 21600 or preflight["cuda_reserved_bytes_cap"] != 5500000000: raise PermissionError("Changed bounded diagnostic resources")
    technical = load(base/"faced_artifact_control_2026-10-11/source_records.json")["records"]+load(base/"faced_artifact_control_2026-10-11/validation_records.json")["records"]
    axes = derive_axes(load(base/"faced_development_protocol_2026-10-11/plan.json"), load(base/"faced_backup_2026-10-10/participant_reservation.json"), load(base/"faced_backup_2026-10-10/stimulus_metadata.json"), technical, load(base/"faced_inference_quality_2026-10-11/signal_audit.json")["trial_decisions"])
    gate = load(packet/"source_gate.json")
    if gate.get("passed") is not True or gate.get("source_trials") != 980 or gate.get("source_failures") != 0 or gate.get("targets_joined") is not False: raise PermissionError("Source gate did not precede target join")
    reuse = load(packet/"targets_reuse.json")
    if reuse.get("source_gate_sha256") != sha(packet/"source_gate.json") or reuse.get("original_rating_files_opened") is not False or reuse.get("protected_outcomes_opened") is not False: raise PermissionError("Cached target reuse provenance mismatch")
    if datetime.fromisoformat(gate["completed_utc"].replace("Z", "+00:00")) > datetime.fromisoformat(reuse["started_utc"].replace("Z", "+00:00")): raise PermissionError("Targets joined before source gate completion")
    features = load(packet/"feature_manifest.json")
    records = features.get("records", [])
    if len(records) != 1260 or {pair(row) for row in records} != axes.allowed: raise PermissionError("Changed feature replay manifest")
    for row in records:
        p = pair(row)
        if row["role"] != axes.metadata[p]["role"] or boolean(row["eeg_available"]) != axes.decisions[p]["eeg_available"]: raise PermissionError("Feature replay role/QC mismatch")
        if boolean(row["eeg_available"]) and row["processed_window_sha256"] != axes.metadata[p]["processed_window_sha256"]: raise PermissionError("Processed window replay hash mismatch")
    control_reuse = load(packet/"control_reuse.json")
    if control_reuse.get("controls_csv_sha256") != sha(packet/"controls.csv") or not control_reuse.get("bindings") or not control_reuse.get("filters"):
        raise PermissionError("Missing frozen cached-control projection provenance")
    for rel, binding in control_reuse["bindings"].items():
        value = binding.get("sha256") if isinstance(binding, dict) else binding
        if sha(repo/rel) != value: raise PermissionError("Cached control source file changed")
    checkpoint_rows = read_csv(packet/"checkpoint_predictions.csv")
    control_rows = read_csv(packet/"controls.csv")
    result = audit_results(axes, checkpoint_rows, read_csv(packet/"ensemble_predictions.csv"), load(packet/"fit_records.json"), control_rows, load(packet/"execution.json"))
    # Deliberately after completed execution and source gate/reuse qualification.
    old_quality_path = base/"faced_inference_quality_2026-10-11/predictions.csv"
    old_representation_path = base/"faced_representation_v2_2026-10-11/predictions.csv"
    required = {str(path.relative_to(repo)).replace("\\", "/") for path in (old_quality_path, old_representation_path)}
    if not required <= set(control_reuse["bindings"]): raise PermissionError("Cached source projections not both frozen")
    kernel_plan_rel = "results/development/faced_representation_v2_2026-10-11/plan.json"
    if activation.get("bindings", {}).get(kernel_plan_rel) != sha(repo/kernel_plan_rel) or load(repo/kernel_plan_rel)["conditions"].get("context_kernel") != ["rbf", ["context"], .1]:
        raise PermissionError("Missing/changed fixed kernel plan binding")
    result["cached_projection_audit"] = verify_cached_projection(axes, read_csv(old_quality_path), read_csv(old_representation_path), control_rows, checkpoint_rows)
    result["activation_bindings_verified"] = count
    result["source_gate_verified"] = True
    result["replay_manifest_trials"] = len(records)
    result["replay_available_trials"] = sum(axes.decisions[p]["eeg_available"] for p in axes.allowed)
    result["cached_control_source_bindings_verified"] = len(control_reuse["bindings"])
    result["kernel_projection_scope"] = "All original cache identities guarded, then exact selected-row comparison; first four controls independently reconstructed by source-only priors/SVD"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(); packet = args.packet or args.repo/DEFAULT_PACKET
    result = audit_packet(args.repo, packet); destination = packet/"independent_audit.json"
    if args.check:
        if load(destination) != result: raise ValueError("Independent audit receipt changed")
    else:
        destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
