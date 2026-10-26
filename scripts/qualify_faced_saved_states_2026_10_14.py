"""Independent source-only saved Ridge-state qualification; never fit a model.

Declare binds opaque private files before numerical access. Qualify is allowed
only after the root freezes the declaration. No producer functions are imported.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
OLD = BASE / "faced_coverage_missing_2026-10-13"
PRIVATE = REPO.parent / "publication_runs/faced_coverage_missing_2026-10-13"
PUBLIC = BASE / "faced_saved_state_qualification_2026-10-14"
PLAN_V1 = PUBLIC / "plan.json"
PLAN = PUBLIC / "plan_v2.json"
PROTOCOL = REPO / "docs/publication/FACED_Saved_State_Qualification_Protocol_2026-10-14.md"
SOURCE_PLAN = BASE / "faced_temporal_coverage_preflight_2026-10-12/plan.json"
TARGETS = BASE / "faced_inference_quality_2026-10-11/predictions.csv"
CATEGORIES = ("Amusement", "Anger", "Disgust", "Fear", "Inspiration", "Joy", "Sadness", "Tenderness")
ACQUISITIONS = ("legacy_1000", "legacy_250", "modern_1000")
CLIPS = frozenset((2, 3, 4, 6, 9, 11, 12, 17, 18, 20, 23, 25, 27, 28))
MODELS = ("context_only", "late_joint", "distributed_joint")
REGIONS = ("early", "middle", "late")
CHANNELS = ("Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2")
BANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 45))
SCALE_RTOL = 1e-12
SCALE_ATOL = 1e-12
EQUATION_RTOL = 1e-9
PREDICTION_ATOL = 1e-10
CAP_SECONDS = 600


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def array_digest(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode() + values.tobytes()).hexdigest()


def identity_digest(pairs):
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rehash(bindings):
    for name, expected in bindings.items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()) or digest(path) != expected:
            raise PermissionError("Frozen public binding changed: " + name)


def preservation():
    paper = load(REPO / "paper/exploratory_evaluation_2026-10-12/build_receipt.json")
    rehash(paper["bindings"])
    old = load(BASE / "faced_confirmation_freeze_2026-10-12/activation.json")
    count = 0
    for key in ("bindings", "prior_completed_evidence_bindings"):
        rehash(old[key]); count += len(old[key])
    archive = REPO / "docs/paper_archive/2026-10-12-before-improvement"
    files = load(archive / "manifest.json")["files"]
    for row in files:
        if digest(REPO / row["source"]) != row["sha256"] or digest(archive / row["archive"]) != row["sha256"]:
            raise PermissionError("Preserved manuscript changed")
    proposals = []
    for name in ("faced_coverage_confirmation_proposal_2026-10-13", "faced_coverage_missing_confirmation_proposal_2026-10-13"):
        packet = load(BASE / name / "proposal.json")
        rehash(packet["bindings"])
        proposals.append({"packet": name, "bindings": len(packet["bindings"])})
    activation = load(OLD / "activation.json")
    rehash(activation["bindings"])
    return {"paper_bindings": len(paper["bindings"]), "archive_pairs": len(files),
            "old_activation_bindings": count, "coverage_proposals": proposals,
            "source_activation_bindings": len(activation["bindings"])}


def context_vector(row):
    category, acquisition = row["assigned_category"], row["acquisition_group"]
    duration, trial = float(row["catalogue_duration_seconds"]), row["trial"]
    if category not in CATEGORIES or acquisition not in ACQUISITIONS or not np.isfinite(duration) or duration <= 0 or type(trial) is not int or not 1 <= trial <= 28:
        raise PermissionError("Invalid context or order")
    return np.array([float(category == c) for c in CATEGORIES] +
                    [float(acquisition == a) for a in ACQUISITIONS] + [duration, float(trial)], dtype=np.float64)


def metadata():
    source = load(SOURCE_PLAN)
    people = source["source_people"]
    denied = set(source["validation_people_denied"]) | set(source["protected_people_denied"])
    if len(people) != 70 or len(set(people)) != 70 or set(people) & denied:
        raise PermissionError("Source participant reservation changed")
    catalogue = {r["clip"]: r for r in load(BASE / "faced_codebooks_2026-10-10/audit.json")["stimuli"]}
    history = {(r["subject"], r["vid"]): r for r in load(BASE / "faced_artifact_control_2026-10-11/source_records.json")["records"]}
    result = {}
    for row in source["source_records"]:
        pair = row["subject"], row["vid"]
        if pair in result or pair[0] not in people or pair[1] not in CLIPS or row.get("role") != "source" or history[pair].get("role") != "source":
            raise PermissionError("Undeclared source context identity")
        result[pair] = {**catalogue[pair[1]], **row, "trial": history[pair]["trial"]}
        if array_digest(context_vector(result[pair])) != history[pair]["context_feature_sha256"]:
            raise PermissionError("Historical context prefix bytes changed")
    pairs = sorted(result)
    if len(pairs) != 980 or set(pairs) != {(s, v) for s in people for v in CLIPS}:
        raise PermissionError("Require complete 70 by 14 source rectangle")
    return result, pairs


def source_records():
    records = [json.loads(line) for line in (OLD / "fit_records.jsonl").read_text().splitlines()]
    selected = [r for r in records if r.get("stage") == "full_source_state_qualification"]
    if len(records) != 93 or len(selected) != 3 or [r["model"] for r in selected] != list(MODELS):
        raise PermissionError("Require original 93 records and exactly three final states")
    return selected


def source_targets(rows, meta, expected_counts=None):
    """Complete role/join pass precedes first numeric target conversion."""
    permitted = []
    seen = set()
    ignored = 0
    people = {p[0] for p in meta}
    for row in rows:
        if row.get("subject") not in people or row.get("arm") != "source_oof":
            ignored += 1
            continue
        pair = row["subject"], int(row["vid"])
        if pair not in meta or meta[pair].get("role") != "source" or row.get("family") != meta[pair]["family"] or row.get("acquisition_group") != meta[pair]["acquisition_group"]:
            raise PermissionError("Source target join changed before numerical interpretation")
        permitted.append((pair, row))
        seen.add(pair)
    if seen != set(meta):
        raise PermissionError("Source target coverage incomplete")
    if expected_counts is not None and expected_counts != {
            "distinct": len(seen), "numeric_rows": len(permitted), "ignored_rows": ignored}:
        raise PermissionError("Source CSV exact row conversion budget differs before conversion")
    values = {}
    for pair, row in permitted:
        target = float(row["target"])
        if not np.isfinite(target) or not 0 <= target <= 7 or pair in values and values[pair] != target:
            raise ValueError("Invalid/conflicting cached source target")
        values[pair] = target
    return values, {"distinct_source_targets": len(values), "numeric_source_rows": len(permitted), "rows_ignored_before_conversion": ignored}


def population_moments(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    variance = ((x - mean) ** 2).mean(axis=0)
    eps = np.finfo(np.float64).eps
    constant = variance <= len(x) * eps * variance + (len(x) * mean * eps) ** 2
    scale = np.sqrt(variance)
    scale[constant] = 1.0
    return mean, variance, scale


def validate_state_arrays(state, dimension):
    expected = {"coefficient", "intercept", "mean", "scale", "variance", "alpha"}
    if dimension == 163:
        expected.add("eeg_imputation_mean")
    if set(state) != expected:
        raise ValueError("Saved archive keys differ")
    for name, value in state.items():
        shape = (150,) if name == "eeg_imputation_mean" else ((dimension,) if name in ("coefficient", "mean", "scale", "variance") else ())
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("Invalid saved state shape/dtype/finite values: " + name)
    if float(state["alpha"]) != 100.0 or np.any(state["scale"] <= 0) or np.any(state["variance"] < 0):
        raise ValueError("Invalid saved alpha/scaler")


def inspect_equations(x, y, state):
    """Check stationarity of saved coefficients, without solving or fitting."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    validate_state_arrays(state, x.shape[1])
    mean, variance, scale = population_moments(x)
    discrepancies = {}
    for name, reference in (("mean", mean), ("variance", variance), ("scale", scale)):
        value = state[name]
        discrepancies[name] = float(np.max(np.abs(value - reference)))
        if not np.allclose(value, reference, rtol=SCALE_RTOL, atol=SCALE_ATOL):
            raise ValueError("Saved scaler moment mismatch: " + name)
    z = (x - state["mean"]) / state["scale"]
    zmean = z.mean(axis=0)
    yc = y - y.mean()
    zc = z - zmean
    coef = state["coefficient"]
    lhs = (zc.T @ zc) @ coef + float(state["alpha"]) * coef
    rhs = zc.T @ yc
    residual = float(np.max(np.abs(lhs - rhs)))
    relative = residual / max(1.0, float(np.max(np.abs(lhs))), float(np.max(np.abs(rhs))))
    intercept = float(y.mean() - zmean @ coef)
    intercept_delta = abs(intercept - float(state["intercept"]))
    if relative > EQUATION_RTOL or intercept_delta > PREDICTION_ATOL:
        raise ValueError("Saved Ridge stationarity/intercept mismatch")
    matrix_raw = z @ coef + float(state["intercept"])
    loop_raw = np.asarray([sum(float(a) * float(b) for a, b in zip(row, coef)) + float(state["intercept"]) for row in z])
    prediction_delta = float(np.max(np.abs(matrix_raw - loop_raw)))
    if prediction_delta > PREDICTION_ATOL:
        raise ValueError("Independent row-sum prediction replay differs")
    return {"scaler_max_absolute_discrepancies": discrepancies,
            "normal_equation_max_absolute_residual": residual,
            "normal_equation_relative_residual": relative,
            "intercept_max_absolute_discrepancy": intercept_delta,
            "matrix_vs_row_sum_max_prediction_discrepancy": prediction_delta,
            "raw_prediction_sha256": array_digest(matrix_raw),
            "clipped_prediction_sha256": array_digest(np.clip(matrix_raw, 0, 7))}, np.clip(matrix_raw, 0, 7)


def independent_priors(pairs, targets, meta):
    y = np.asarray([targets[p] for p in pairs], dtype=np.float64)
    result = {"global": float(np.median(y)), "coarse": {}, "category": {}, "video": {}}
    for field, key in (("assigned_valence", "coarse"), ("assigned_category", "category"), ("vid", "video")):
        groups = {}
        for pair, value in zip(pairs, y):
            group = str(pair[1]) if field == "vid" else meta[pair][field]
            groups.setdefault(group, []).append(value)
        result[key] = {group: float(np.median(values)) for group, values in groups.items()}
    return result


def declare():
    _, pairs = metadata()
    preserved = preservation()
    completion = load(OLD / "completion.json")
    for name, expected in completion["public_bindings"].items():
        if digest(OLD / name) != expected:
            raise PermissionError("Completed source receipt changed")
    records = source_records()
    if not PLAN_V1.exists():
        raise PermissionError("Preserve the original unexecuted declaration")
    target_receipt = load(OLD / "cached_source_targets_receipt.json")
    if target_receipt["distinct_source_targets"] != 980 or target_receipt["source_rows_agreeing"] != 11760 or target_receipt["rows_ignored_before_target_conversion"] != 15960:
        raise PermissionError("Previously qualified source target scope changed")
    qualified = load(OLD / "qualification.json")
    private_files = []
    for row in records:
        if row["training_pairs"] != [list(p) for p in pairs] or row["alpha"] != 100.0 or row["source_identity_sha256"] != identity_digest(pairs):
            raise PermissionError("Full-source state identity/alpha changed")
        file = PRIVATE / row["private_state_file"]
        if file.parent.resolve() != PRIVATE.resolve() or digest(file) != row["private_state_sha256"]:
            raise PermissionError("Opaque saved state hash differs")
        private_files.append({"path": file.relative_to(REPO.parent).as_posix(), "sha256": digest(file), "kind": row["model"]})
    for name, expected, kind in (("available_source_vectors.npz", qualified["source_snapshot_file_sha256"], "available_source_snapshot"),
                                 ("full_source_category_prior.json", load(OLD / "final_source_state_qualification.json")["private_category_prior_sha256"], "source_prior")):
        file = PRIVATE / name
        if digest(file) != expected:
            raise PermissionError("Opaque source snapshot/prior hash changed")
        private_files.append({"path": file.relative_to(REPO.parent).as_posix(), "sha256": expected, "kind": kind})
    public_files = [SOURCE_PLAN, TARGETS, BASE / "faced_codebooks_2026-10-10/audit.json",
                    BASE / "faced_artifact_control_2026-10-11/source_records.json", PROTOCOL,
                    Path(__file__), REPO / "tests/test_faced_saved_states_2026_10_14.py"]
    public_files += [OLD / name for name in ("completion.json", "fit_records.jsonl", "qualification.json", "imputation_fixtures.json", "final_source_state_qualification.json", "source_policy_plan.json", "cached_source_targets_receipt.json")]
    # Feature definitions are bound for ordering/provenance, never imported here.
    public_files += [REPO / "gruxnet/faced_classical_contract.py", REPO / "gruxnet/faced_proposal_preprocessing.py", REPO / "gruxnet/faced_coverage_missing.py"]
    plan = {"date": "2026-10-14", "status": "declared_source_only_saved_state_qualification_not_executed",
            "source_people": 70, "source_trials": 980, "source_identity_sha256": identity_digest(pairs),
            "source_snapshot_shape": [979, 3, 150], "source_feature_scalar_cap": 440550,
            "source_distinct_target_cap": 980, "source_numeric_row_conversion_cap": 11760,
            "source_ignored_rows": 15960, "full_source_state_files": 3, "source_prior_files": 1,
            "preserved_unexecuted_v1_plan_sha256": digest(PLAN_V1),
            "new_fits": 0, "new_raw_sample_decodes": 0, "original_rating_decodes": 0,
            "validation_numeric_access": False, "confirmation_access": False,
            "source_metadata_only": True, "all_metadata_checked_before_source_target_conversion": True,
            "mean_and_scaler_relative_tolerance": SCALE_RTOL, "mean_and_scaler_absolute_tolerance": SCALE_ATOL,
            "normal_equation_relative_tolerance": EQUATION_RTOL, "prediction_absolute_tolerance": PREDICTION_ATOL,
            "alpha": 100.0, "automatic_retry": False, "wall_seconds_cap": CAP_SECONDS, "cpu_seconds_cap": CAP_SECONDS,
            "context_feature_order": list(CATEGORIES) + list(ACQUISITIONS) + ["catalogue_duration_seconds", "trial"],
            "eeg_feature_order": [f"{channel}/{band[0]}" for channel in CHANNELS for band in BANDS],
            "regions": list(REGIONS), "private_opaque_inputs": private_files,
            "public_bindings": {p.relative_to(REPO).as_posix(): digest(p) for p in public_files},
            "preservation": preserved, "protected_exports_qualified": False,
            "limits": "Independent stationarity/reconstruction of existing full-source states, not a refit or predictive evaluation. Spectral formula/raw signal backend qualification is separate."}
    write(PLAN, plan)
    print(json.dumps({"declared": True, "private_input_files": len(private_files), "numeric_arrays_decoded": 0, "target_values_converted": 0}))


def qualify():
    plan = load(PLAN)
    rehash(plan["public_bindings"])
    if digest(PLAN_V1) != plan["preserved_unexecuted_v1_plan_sha256"]:
        raise PermissionError("Preserved unexecuted v1 changed")
    if plan["status"] != "declared_source_only_saved_state_qualification_not_executed" or plan["confirmation_access"] is not False or plan["validation_numeric_access"] is not False:
        raise PermissionError("Exact inactive source-only declaration required")
    if (PUBLIC / "started.json").exists():
        raise PermissionError("Preserve first attempt; no retry")
    meta, pairs = metadata()
    qualified = load(OLD / "qualification.json")
    available = [tuple(p) for p in qualified["available_pairs"]]
    policy = load(OLD / "source_policy_plan.json")
    missing = [tuple(p) for p in policy["shared_missing_pairs"]]
    if available != [p for p in pairs if p not in missing] or missing != [("sub-087", 20)] or identity_digest(available) != qualified["available_identity_sha256"]:
        raise PermissionError("Available/missing source order changed before private decode")
    inputs = {}
    for record in plan["private_opaque_inputs"]:
        path = (REPO.parent / record["path"]).resolve()
        if path.parent != PRIVATE.resolve() or digest(path) != record["sha256"]:
            raise PermissionError("Source private allowlist/hash differs")
        inputs[record["kind"]] = path
    start, cpu = time.monotonic(), time.process_time()
    def budget():
        if time.monotonic() - start > CAP_SECONDS or time.process_time() - cpu > CAP_SECONDS:
            raise TimeoutError("Source saved-state qualification budget exceeded")
    write(PUBLIC / "started.json", {"plan_sha256": digest(PLAN), "utc": datetime.now(timezone.utc).isoformat(),
                                   "confirmation_access": False, "new_fits": 0})
    try:
        with np.load(inputs["available_source_snapshot"], allow_pickle=False) as bundle:
            if bundle.files != ["vectors"]:
                raise ValueError("Snapshot archive keys differ")
            vectors = bundle["vectors"].copy()
        if vectors.shape != (979, 3, 150) or vectors.dtype != np.float64 or not np.isfinite(vectors).all() or array_digest(vectors) != qualified["source_snapshot_array_sha256"]:
            raise ValueError("Snapshot array geometry/identity differs")
        contexts = np.stack([context_vector(meta[p]) for p in pairs])
        indices = {p: i for i, p in enumerate(available)}
        branches = {"late_joint": vectors[:, 2, :], "distributed_joint": vectors.mean(axis=1)}
        features = {"context_only": contexts}
        states, mean_results = {}, {}
        fixtures = load(OLD / "imputation_fixtures.json")["full_source"]
        for model in MODELS:
            budget()
            with np.load(inputs[model], allow_pickle=False) as archive:
                states[model] = {name: archive[name].copy() for name in archive.files}
            validate_state_arrays(states[model], 13 if model == "context_only" else 163)
            if model == "context_only":
                continue
            mean = branches[model].mean(axis=0)
            if array_digest(mean) != fixtures["mean_sha256"][model] or array_digest(states[model]["eeg_imputation_mean"]) != array_digest(mean):
                raise ValueError("Available-only imputation mean differs")
            eeg = np.stack([branches[model][indices[p]] if p in indices else mean for p in pairs])
            features[model] = np.concatenate((contexts, eeg), axis=1)
            mean_results[model] = {"donors": 979, "imputed_training_rows": 1, "mean_sha256": array_digest(mean),
                                   "donor_identity_sha256": identity_digest(available)}
        # All source feature/state geometry validated before source numeric labels.
        with TARGETS.open(newline="", encoding="utf-8") as stream:
            targets, target_receipt = source_targets(list(csv.DictReader(stream)), meta,
                {"distinct": plan["source_distinct_target_cap"],
                 "numeric_rows": plan["source_numeric_row_conversion_cap"],
                 "ignored_rows": plan["source_ignored_rows"]})
        y = np.asarray([targets[p] for p in pairs], dtype=np.float64)
        metrics, predictions = {}, {}
        with threadpool_limits(limits=2, user_api="blas"):
            for model in MODELS:
                budget()
                metrics[model], predictions[model] = inspect_equations(features[model], y, states[model])
        for model in MODELS[1:]:
            for name in ("mean", "scale", "variance"):
                if not np.array_equal(states[model][name][:13], states["context_only"][name]):
                    raise ValueError("Context scaler prefix differs across final states")
        saved_prior = load(inputs["source_prior"])
        expected_prior = independent_priors(pairs, targets, meta)
        if saved_prior != expected_prior:
            raise ValueError("Saved global/coarse/category/video source medians differ")
        # Source inference fixtures retain missing trials via common category routing.
        rows = []
        for i, pair in enumerate(pairs):
            category = float(saved_prior["category"].get(meta[pair]["assigned_category"], saved_prior["global"]))
            available_trial = pair in indices
            rows.append({"subject": pair[0], "vid": pair[1], "role": "source", "family": meta[pair]["family"],
                         "shared_eeg_available": available_trial, "category_median": category,
                         "training_state_predictions": {m: float(predictions[m][i]) for m in MODELS},
                         "inference_predictions": {m: float(predictions[m][i]) if available_trial else category for m in MODELS}})
        write(PUBLIC / "source_prediction_fixtures.json", {"scope": "980 source predictions; no source scoring or protected outcome", "rows": rows})
        rehash(plan["public_bindings"])
        for item in plan["private_opaque_inputs"]:
            if digest(REPO.parent / item["path"]) != item["sha256"]:
                raise PermissionError("Private input changed during qualification")
        preserved = preservation()
        budget()
        result = {"date": "2026-10-14", "passed": True, "plan_sha256": digest(PLAN), "models": metrics,
                  "imputation_checks": mean_results, "source_target_receipt": target_receipt,
                  "source_rows": 980, "source_feature_scalar_values": int(vectors.size), "source_state_count": 3,
                  "context_scaler_prefix_exact_match": True, "saved_prior_exact_match": True,
                  "source_training_prediction_rows": 2940, "source_inference_prediction_rows": 3920,
                  "source_missing_inference_cases": 1, "normal_equations_checked_without_solving": True,
                  "new_fits": 0, "new_raw_sample_decodes": 0, "original_rating_decodes": 0,
                  "validation_numeric_access": False, "confirmation_access": False,
                  "protected_exports_qualified": False, "preservation": preserved,
                  "source_fixture_sha256": digest(PUBLIC / "source_prediction_fixtures.json"),
                  "wall_seconds": time.monotonic() - start, "cpu_seconds": time.process_time() - cpu,
                  "limits": plan["limits"]}
        write(PUBLIC / "assessment.json", result)
        write(PUBLIC / "completion.json", {"date": "2026-10-14", "status": "source_saved_state_qualification_passed",
              "passed": True, "new_fits": 0, "confirmation_access": False, "public_bindings": {
                  name: digest(PUBLIC / name) for name in ("plan.json", "plan_v2.json", "started.json", "source_prediction_fixtures.json", "assessment.json")}})
        print(json.dumps({"passed": True, "states": 3, "source_rows": 980, "new_fits": 0, "confirmation_access": False}))
    except Exception as error:
        write(PUBLIC / "failure.json", {"status": "held_or_failed", "error": str(error), "automatic_retry": False,
                                        "new_fits": 0, "confirmation_access": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "qualify"))
    action = parser.parse_args().action
    {"declare": declare, "qualify": qualify}[action]()
