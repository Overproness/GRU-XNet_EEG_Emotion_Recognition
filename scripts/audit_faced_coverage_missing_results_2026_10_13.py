"""Independent arithmetic/donor audit of shared-missing source predictions.

No private model/feature/waveform array, original recording or original rating
archive is read. This follows the complete-source QC audit and completed fits.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from math import fsum, isfinite
from pathlib import Path
from statistics import median
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_faced_coverage_source_qc_2026_10_13 import preservation

BASE = REPO / "results/development"
PUBLIC = BASE / "faced_coverage_missing_2026-10-13"
ALPHAS = (.1, 10., 100.)
MODELS = ("context_only", "late_joint", "distributed_joint")


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    if not isfinite(actual) or not isfinite(expected) or abs(actual - expected) > 1e-12:
        raise ValueError("Independent source result arithmetic differs")


def loss_metrics(rows):
    """Average clip losses within family, families within person, then people."""
    groups, seen, squares = defaultdict(list), set(), []
    for row in rows:
        identity = row["subject"], row["vid"]
        if identity in seen:
            raise ValueError("Duplicate source prediction identity")
        seen.add(identity)
        target, predicted = row["target"], row["prediction"]
        if (isinstance(target, bool) or isinstance(predicted, bool)
                or not isfinite(target) or not isfinite(predicted)
                or not 0 <= target <= 7 or not 0 <= predicted <= 7):
            raise ValueError("Nonfinite/outside-scale source prediction")
        error = predicted - target
        groups[row["subject"], row["family"]].append(abs(error))
        squares.append(error * error)
    people = defaultdict(list)
    for (subject, _), errors in groups.items():
        people[subject].append(fsum(errors) / len(errors))
    if not people:
        raise ValueError("Empty source predictions")
    return {"participant_title_mae": fsum(fsum(v) / len(v) for v in people.values()) / len(people),
            "trial_mse": fsum(squares) / len(squares), "prediction_rows": len(rows)}


def identity_hash(pairs):
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def imputation_record(record, training_pairs, available, fixture):
    training = [tuple(p) for p in training_pairs]
    donors = [p for p in training if available[p]]
    missing = [p for p in training if not available[p]]
    if (list(map(tuple, record["eligible_train_donor_pairs"])) != donors
            or record["eligible_train_donor_identity_sha256"] != identity_hash(donors)
            or record["shared_available_training_rows"] != len(donors)
            or list(map(tuple, record["shared_missing_training_pairs"])) != missing):
        raise ValueError("Source imputation donor identity scope differs")
    if record["model"] == "context_only":
        if record["imputation"] is not None or record["imputed_training_rows"] != 0 or record["imputed_training_pairs"] != []:
            raise ValueError("Context control must retain rows without EEG imputation")
        return 0
    mean = record["imputation"]
    if (list(map(tuple, mean["donor_pairs"])) != donors or mean["donor_rows"] != len(donors)
            or list(map(tuple, mean["missing_training_pairs"])) != missing
            or mean["missing_training_rows"] != len(missing) or mean["training_rows_retained"] != len(training)
            or mean["eeg_dimension"] != 150 or mean["donor_identity_sha256"] != identity_hash(donors)
            or mean["mean_sha256"] != fixture["mean_sha256"][record["model"]]
            or mean["mean_uses_only_available_training_rows"] is not True
            or mean["held_rows_used_for_imputation"] != 0 or mean["target_values_used_for_imputation"] != 0
            or list(map(tuple, record["imputed_training_pairs"])) != missing
            or record["imputed_training_rows"] != len(missing)):
        raise ValueError("Joint imputation mean/donor manifest differs")
    return 1


def audit():
    destination = PUBLIC / "independent_audit.json"
    if destination.exists():
        raise FileExistsError("Preserve existing final audit")
    qc = load(PUBLIC / "qualification_independent_audit.json")
    summary = load(PUBLIC / "comparison.json")
    if (qc["passed"] is not True or qc["masked_input_eligibility_passed"] is not True
            or summary["status"] != "completed_source_coverage_missing_selection"
            or summary["cv_fit_count"] != 90 or summary["full_source_fit_count"] != 3
            or summary["validation_access"] is not False or summary["protected_access"] is not False
            or summary["independent_confirmation"] is not False
            or (PUBLIC / "fitting_failure.json").exists()):
        raise PermissionError("Complete qualified source-only fits required")
    if (sha(PUBLIC / "cv_predictions.json") != summary["cv_predictions_sha256"]
            or sha(PUBLIC / "cached_source_targets_receipt.json") != summary["cached_targets_receipt_sha256"]):
        raise ValueError("Completed source result binding changed")
    targets_receipt = load(PUBLIC / "cached_source_targets_receipt.json")
    if targets_receipt["distinct_source_targets"] != 980 or targets_receipt["original_rating_decoding"] != 0:
        raise PermissionError("Cached target scope differs")
    source_plan = load(BASE / "faced_temporal_coverage_preflight_2026-10-12/plan.json")
    plan = load(BASE / "faced_coverage_selection_preflight_2026-10-13/plan.json")
    metadata = {(r["subject"], r["vid"]): r for r in source_plan["source_records"]}
    source_pairs = sorted(metadata)
    prior_rows = [json.loads(x) for x in (BASE / "faced_coverage_selection_2026-10-13/source_regions.jsonl").read_text().splitlines()]
    statuses = {(r["subject"], r["vid"], r["region"]): r["eeg_available"] for r in prior_rows}
    available = {p: all(statuses[(*p, r)] for r in ("early", "middle", "late")) for p in source_pairs}
    if [p for p in source_pairs if not available[p]] != [("sub-087", 20)] or summary["old_strict_gate_passed"] is not False:
        raise PermissionError("Preserved source mask/gate changed")
    if summary["quality_thresholds_changed"] is not False:
        raise PermissionError("QC threshold changed")
    fixtures = load(PUBLIC / "imputation_fixtures.json")
    if qc["imputation_fixtures_sha256"] != sha(PUBLIC / "imputation_fixtures.json"):
        raise ValueError("Pretarget mean fixtures changed")
    reported_availability = {(r["subject"], r["vid"]): r["eeg_available"] for r in summary["shared_availability"]}
    if reported_availability != available or len(summary["shared_availability"]) != 980:
        raise ValueError("Reported shared source mask differs")
    cells = {c["cell"]: c for c in plan["cells"]}
    all_predictions = load(PUBLIC / "cv_predictions.json")
    ridge, category = all_predictions["ridge"], all_predictions["category"]
    if len(ridge) != 7560 or len(category) != 840:
        raise ValueError("Incomplete source prediction denominator")
    panels, categories, observed, targets = defaultdict(list), defaultdict(list), set(), {}
    for row in ridge + category:
        pair = row["subject"], row["vid"]
        if pair not in metadata or row["role"] != "source" or row["cell"] not in cells:
            raise PermissionError("Wrong source prediction role/identity")
        original, cell = metadata[pair], cells[row["cell"]]
        if (list(pair) not in cell["held"] or row["person_fold"] != cell["person_fold"]
                or row["material_assignment"] != cell["material_assignment"]
                or row["family"] != original["family"]
                or row["assigned_category"] != original["assigned_category"]
                or row["acquisition_group"] != original["acquisition_group"]):
            raise PermissionError("Prediction source cell/material/context differs")
        if (row["shared_eeg_available"] is not available[pair]
                or row["common_category_fallback"] is not (row["model"] != "category_median" and not available[pair])):
            raise ValueError("Source fallback/availability policy differs")
        if row["model"] == "category_median":
            if row["alpha"] is not None:
                raise ValueError("Category comparator has no Ridge alpha")
            categories[row["cell"]].append(row)
        else:
            if row["model"] not in MODELS or row["alpha"] not in ALPHAS:
                raise ValueError("Undeclared source candidate")
            panels[row["model"], row["alpha"]].append(row)
        key = row["cell"], row["model"], row["alpha"], pair
        if key in observed:
            raise ValueError("Duplicate candidate/source held prediction")
        observed.add(key)
        target = row["target"]
        if pair in targets and targets[pair] != target:
            raise ValueError("Repeated source targets disagree")
        targets[pair] = target
    expected_held = {tuple(p) for c in cells.values() for p in c["held"]}
    if set(targets) != expected_held or len(targets) != 840:
        raise ValueError("Source held-target coverage differs")
    candidate_metrics = {(r["model"], r["alpha"]): r for r in summary["candidate_metrics"]}
    if set(candidate_metrics) != {(m, a) for m in MODELS for a in ALPHAS}:
        raise ValueError("All nine source candidates are mandatory")
    for key, rows in panels.items():
        if len(rows) != 840 or {(r["subject"], r["vid"]) for r in rows} != expected_held:
            raise ValueError("Candidate missing held source rows")
        calculated = loss_metrics(rows)
        for name in ("participant_title_mae", "trial_mse"):
            close(candidate_metrics[key][name], calculated[name])
        if candidate_metrics[key]["prediction_rows"] != 840:
            raise ValueError("Reported candidate denominator differs")
    # Every evaluated category is paired: its fold median can be recomputed
    # from emitted source labels without opening the train-only singleton labels.
    category_checks = 0
    for index, cell in cells.items():
        if len(categories[index]) != 84:
            raise ValueError("Category comparator cell incomplete")
        for row in categories[index]:
            training = [tuple(p) for p in cell["training"]
                        if metadata[tuple(p)]["assigned_category"] == row["assigned_category"]]
            if len(training) != 56 or any(p not in targets for p in training):
                raise ValueError("Category-supported source median scope differs")
            close(row["prediction"], median(targets[p] for p in training)); category_checks += 1
    category_metric = loss_metrics(category)
    for name in ("participant_title_mae", "trial_mse"):
        close(summary["category_median_metrics"][name], category_metric[name])
    cell_metrics = {(r["cell"], r["model"], r["alpha"]): r for r in summary["cell_metrics"]}
    if len(cell_metrics) != 100:
        raise ValueError("All 100 cell metrics must be retained")
    for index in cells:
        for model, alpha in [("category_median", None)] + [(m, a) for a in ALPHAS for m in MODELS]:
            panel = categories[index] if model == "category_median" else [r for r in panels[model, alpha] if r["cell"] == index]
            if len(panel) != 84:
                raise ValueError("Source held cell candidate missing rows")
            metrics = loss_metrics(panel)
            for name in ("participant_title_mae", "trial_mse"):
                close(cell_metrics[index, model, alpha][name], metrics[name])
    alpha_scores = [{"alpha": a,
                     "equal_model_mean_mae": sum(candidate_metrics[m, a]["participant_title_mae"] for m in MODELS) / 3,
                     "equal_model_mean_mse": sum(candidate_metrics[m, a]["trial_mse"] for m in MODELS) / 3} for a in ALPHAS]
    selected = min(alpha_scores, key=lambda r: (r["equal_model_mean_mae"], r["equal_model_mean_mse"], ALPHAS.index(r["alpha"])))
    for reported, calculated in zip(summary["common_alpha_selection"]["alpha_scores"], alpha_scores):
        if reported["alpha"] != calculated["alpha"]:
            raise ValueError("Shared-alpha priority differs")
        close(reported["equal_model_mean_mae"], calculated["equal_model_mean_mae"])
        close(reported["equal_model_mean_mse"], calculated["equal_model_mean_mse"])
    alpha = selected["alpha"]
    if summary["common_alpha_selection"]["selected_alpha"] != alpha:
        raise ValueError("Shared alpha selection differs")
    nominee = min(MODELS[1:], key=lambda m: (candidate_metrics[m, alpha]["participant_title_mae"],
                                           candidate_metrics[m, alpha]["trial_mse"], MODELS.index(m)))
    if summary["source_nomination"]["model"] != nominee or summary["source_nomination"]["alpha"] != alpha:
        raise ValueError("Source coverage nomination differs")
    for contrast in summary["contrasts"]:
        a = contrast["alpha"]; c, late, dist = [candidate_metrics[m, a]["participant_title_mae"] for m in MODELS]
        expected = {"distributed_minus_late_mae": dist - late, "distributed_coverage_gain": late - dist,
                    "matched_context_minus_late_mae": c - late, "matched_context_minus_distributed_mae": c - dist,
                    "category_median_minus_late_mae": category_metric["participant_title_mae"] - late,
                    "category_median_minus_distributed_mae": category_metric["participant_title_mae"] - dist}
        for name, value in expected.items():
            close(contrast[name], value)
        if contrast["primary_fixed_alpha"] is not (a == 100.):
            raise ValueError("Fixed primary diagnostic changed")
    fits = [json.loads(line) for line in (PUBLIC / "fit_records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(fits) != 93:
        raise ValueError("Actual source fit manifest count differs")
    expected_keys = [(cell, a, m) for cell in cells for a in ALPHAS for m in MODELS]
    imputation_checks = 0
    for i, record in enumerate(fits[:90]):
        key = record["cell"], record["alpha"], record["model"]
        cell = cells[record["cell"]]
        if (key != expected_keys[i] or record["fit"] != i
                or record["training_pairs"] != cell["training"] or record["held_pairs"] != cell["held"]
                or record["scaler_training_rows"] != 448 or record["target_training_rows"] != 448
                or record["feature_dimension"] != (13 if record["model"] == "context_only" else 163)
                or record["source_identity_sha256"] != identity_hash(cell["training"])):
            raise ValueError("CV fit recipe/source manifest differs")
        imputation_checks += imputation_record(record, cell["training"], available, fixtures["cells"][record["cell"]])
    for i, record in enumerate(fits[90:]):
        if (record["stage"] != "full_source_state_qualification" or record["model"] != MODELS[i]
                or record["fit"] != i or record["alpha"] != alpha
                or list(map(tuple, record["training_pairs"])) != source_pairs or record["training_rows"] != 980
                or record["source_identity_sha256"] != identity_hash(source_pairs)
                or record["validation_rows_scored"] != 0 or record["protected_rows_scored"] != 0):
            raise ValueError("Final source state recipe manifest differs")
        imputation_checks += imputation_record(record, source_pairs, available, fixtures["full_source"])
    scaler_checks = 0
    for triplet in [fits[i:i+3] for i in range(0, 90, 3)] + [fits[90:]]:
        moments = [r["context_scaler_moment_hashes"] for r in triplet]
        if moments[0] != moments[1] or moments[0] != moments[2]:
            raise ValueError("Shared context scaling moment hashes differ")
        scaler_checks += 1
    state_receipt = load(PUBLIC / "final_source_state_qualification.json")
    if (state_receipt["passed"] is not True or state_receipt["source_rows"] != 980
            or state_receipt["protected_exports_qualified"] is not False
            or state_receipt["validation_scored"] is not False
            or {r["model"] for r in state_receipt["models"]} != set(MODELS)):
        raise ValueError("Final source state receipt scope differs")
    for row in state_receipt["models"]:
        manifest = next(r for r in fits[90:] if r["model"] == row["model"])
        if (row["rows"] != 980 or row["alpha"] != alpha
                or row["source_identity_sha256"] != identity_hash(source_pairs)
                or row["private_state_sha256"] != manifest["private_state_sha256"]
                or not 0 <= row["max_prediction_discrepancy"] <= 1e-10):
            raise ValueError("Source saved-state replay receipt differs")
        expected_mean = None if row["model"] == "context_only" else fixtures["full_source"]["mean_sha256"][row["model"]]
        if row["imputation_mean_sha256"] != expected_mean:
            raise ValueError("Saved-state mean differs from pretarget numerical replay")
    activation = load(PUBLIC / "activation.json")
    if (not 0 <= summary["elapsed_seconds"] <= activation["fitting_wall_seconds_cap"]
            or not 0 <= summary["cpu_seconds"] <= activation["fitting_cpu_seconds_cap"]):
        raise ValueError("Source fit resource receipt differs")
    names = ("activation.json", "qualification.json", "imputation_fixtures.json", "qualification_independent_audit.json",
             "cv_predictions.json", "comparison.json", "fit_records.jsonl", "cached_source_targets_receipt.json",
             "final_source_state_qualification.json", "fitting_started.json")
    receipt = {"date": "2026-10-13", "passed": True, "old_strict_quality_gate_passed": False, "masked_input_eligibility_passed": True,
               "audit_scope": "Completed public source CV prediction arithmetic, source recipes and saved-state receipt agreement",
               "ridge_cv_prediction_rows": 7560, "category_prediction_rows": 840,
               "distinct_public_source_target_values": 840, "source_candidate_metrics": 9,
               "source_cell_metrics": 100, "source_category_predictions_recomputed": category_checks,
               "cv_fit_manifests": 90, "full_source_fit_manifests": 3,
               "joint_mean_donor_manifests_verified": imputation_checks,
               "matched_context_scaler_triplets_verified": scaler_checks,
               "prior_fold_numerical_means_independently_recomputed": qc["fold_numerical_means_independently_recomputed"],
               "prior_full_numerical_means_independently_recomputed": qc["full_source_numerical_means_independently_recomputed"],
               "shared_alpha_independently_selected": alpha, "coverage_independently_nominated": nominee,
               "fixed_alpha100_contrasts": next(r for r in summary["contrasts"] if r["primary_fixed_alpha"]),
               "bindings": {n: sha(PUBLIC / n) for n in names}, "auditor_code_sha256": sha(Path(__file__)),
               "preservation": preservation(), "private_files_opened": 0,
               "validation_numeric_access": False, "confirmation_access": False,
               "private_state_prediction_replay_independently_executed": False,
               "limits": "Independent emitted-source arithmetic and recipe-manifest audit. No independent refit or private fitted-state replay; previous source data are adaptive development, not confirmation."}
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    audit()
