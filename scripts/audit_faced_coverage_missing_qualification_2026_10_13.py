"""Independent pre-label mean replay for the declared available source snapshot.

Decode only the authenticated 979-by-3-by-150 source feature snapshot after
complete public role/mask/order checks. No failed-trial array, raw recording,
target, prediction or fitted state is opened.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_faced_coverage_source_qc_2026_10_13 import preservation, qualify_public_rows

BASE = REPO / "results/development"
PUBLIC = BASE / "faced_coverage_missing_2026-10-13"
PRIOR = BASE / "faced_coverage_selection_2026-10-13"
PRIVATE = REPO.parent / "publication_runs/faced_coverage_missing_2026-10-13"
REGIONS = ("early", "middle", "late")
JOINTS = ("late_joint", "distributed_joint")


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_sha(pairs):
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def array_sha(values):
    array = np.ascontiguousarray(values)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def donor_means(vectors, available_pairs, training_pairs):
    """Replay training-only regional representation means independently."""
    index = {tuple(p): i for i, p in enumerate(available_pairs)}
    training = [tuple(p) for p in training_pairs]
    if len(training) != len(set(training)) or len(index) != len(available_pairs):
        raise ValueError("Duplicate donor/source order")
    donors = [p for p in training if p in index]
    if not donors:
        raise PermissionError("No available training donors")
    selected = np.stack([vectors[index[p]] for p in donors])
    if selected.shape != (len(donors), 3, 150) or not np.isfinite(selected).all():
        raise ValueError("Invalid declared source donor vectors")
    late = np.add.reduce(selected[:, 2, :], axis=0) / len(donors)
    distributed_rows = np.add.reduce(selected, axis=1) / 3
    distributed = np.add.reduce(distributed_rows, axis=0) / len(donors)
    return donors, [p for p in training if p not in index], {"late_joint": late, "distributed_joint": distributed}


def fixture_check(fixture, vectors, available_pairs, training_pairs, expected_cell):
    donors, missing, means = donor_means(vectors, available_pairs, training_pairs)
    if (fixture["cell"] != expected_cell
            or list(map(tuple, fixture["eligible_train_donor_pairs"])) != donors
            or fixture["donor_count"] != len(donors)
            or fixture["eligible_train_donor_identity_sha256"] != identity_sha(donors)
            or list(map(tuple, fixture["imputed_training_pairs"])) != missing
            or set(fixture["mean_sha256"]) != set(JOINTS)):
        raise ValueError("Declared imputation donor/source scope differs")
    for model, mean in means.items():
        if fixture["mean_sha256"][model] != array_sha(mean):
            raise ValueError("Independent numerical donor mean differs")
    return {"cell": expected_cell, "donor_rows": len(donors), "imputed_rows": len(missing),
            "mean_hashes_independently_recomputed": {m: array_sha(v) for m, v in means.items()}}


def audit():
    output = PUBLIC / "qualification_independent_audit.json"
    if output.exists():
        raise FileExistsError("Preserve previous independent qualification")
    if not (PUBLIC / "qualification.json").exists() or (PUBLIC / "qualification_failure.json").exists():
        raise PermissionError("Completed source-cache qualification required")
    if any((PUBLIC / name).exists() for name in ("fitting_started.json", "cached_source_targets_receipt.json",
                                                "fit_records.jsonl", "comparison.json")):
        raise PermissionError("Independent qualification must precede labels/fitting")
    activation = load(PUBLIC / "activation.json")
    for name, expected in activation["bindings"].items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()) or sha(path) != expected:
            raise ValueError("Frozen qualification binding changed: " + name)
    source_plan = load(BASE / "faced_temporal_coverage_preflight_2026-10-12/plan.json")
    rows = [json.loads(x) for x in (PRIOR / "source_regions.jsonl").read_text().splitlines()]
    meta = qualify_public_rows(rows, source_plan)
    pairs = sorted(meta)
    row_map = {(r["subject"], r["vid"], r["region"]): r for r in rows}
    available = [p for p in pairs if all(row_map[(*p, r)]["eeg_available"] for r in REGIONS)]
    missing = [p for p in pairs if p not in available]
    if len(available) != 979 or missing != [("sub-087", 20)]:
        raise PermissionError("Exact declared source mask required before numerical arrays")
    qualification = load(PUBLIC / "qualification.json")
    q_hash = sha(PUBLIC / "qualification.json")
    fixtures_hash = sha(PUBLIC / "imputation_fixtures.json")
    if (qualification["passed"] is not True or qualification["old_strict_gate_passed"] is not False
            or qualification["activation_sha256"] != sha(PUBLIC / "activation.json")
            or qualification["prior_source_regions_sha256"] != sha(PRIOR / "source_regions.jsonl")
            or list(map(tuple, qualification["available_pairs"])) != available
            or list(map(tuple, qualification["shared_missing_pairs"])) != missing
            or qualification["available_identity_sha256"] != identity_sha(available)
            or qualification["available_trials"] != 979 or qualification["regional_vectors_decoded"] != 2937
            or qualification["feature_scalar_values"] != 440550
            or qualification["unavailable_trial_regional_arrays_decoded"] != 0
            or qualification["unavailable_regional_arrays_skipped"] != 3
            or qualification["imputation_fixtures_sha256"] != fixtures_hash
            or qualification["targets_converted"] != 0 or qualification["fits"] != 0
            or qualification["raw_samples_decoded"] != 0 or qualification["original_rating_decodes"] != 0
            or qualification["validation_numeric_access"] is not False
            or qualification["confirmation_access"] is not False):
        raise PermissionError("Qualified source snapshot/role/value scope changed")
    before = preservation()
    snapshot = PRIVATE / "available_source_vectors.npz"
    if sha(snapshot) != qualification["source_snapshot_file_sha256"]:
        raise ValueError("Source-only snapshot bytes changed")
    # The full source role/mask/order is checked before the sole array decode.
    with np.load(snapshot, allow_pickle=False) as bundle:
        if bundle.files != ["vectors"]:
            raise ValueError("Unexpected source-only snapshot payload")
        vectors = bundle["vectors"].copy()
    if (vectors.shape != (979, 3, 150) or vectors.dtype != np.float64
            or not np.isfinite(vectors).all() or array_sha(vectors) != qualification["source_snapshot_array_sha256"]):
        raise ValueError("Source snapshot geometry/array identity differs")
    checked = 0
    for index, pair in enumerate(available):
        for region_index, region in enumerate(REGIONS):
            if array_sha(vectors[index, region_index]) != row_map[(*pair, region)]["spectral_feature_sha256"]:
                raise ValueError("Snapshot regional vector differs from prior source lineage")
            checked += 1
    cells = load(BASE / "faced_coverage_selection_preflight_2026-10-13/plan.json")["cells"]
    fixtures = load(PUBLIC / "imputation_fixtures.json")
    if len(fixtures["cells"]) != 10:
        raise ValueError("All ten cell mean fixtures are required")
    replays = [fixture_check(f, vectors, available, c["training"], c["cell"])
               for f, c in zip(fixtures["cells"], cells)]
    full = fixture_check(fixtures["full_source"], vectors, available, pairs, None)
    if (sum(r["imputed_rows"] == 1 for r in replays) != 8
            or sum(r["imputed_rows"] == 0 for r in replays) != 2 or full["donor_rows"] != 979
            or full["imputed_rows"] != 1):
        raise ValueError("Declared cell/full-source mean-imputation counts differ")
    if (sha(PUBLIC / "qualification.json") != q_hash
            or sha(PUBLIC / "imputation_fixtures.json") != fixtures_hash
            or sha(snapshot) != qualification["source_snapshot_file_sha256"]):
        raise ValueError("Audited qualification changed during numerical replay")
    after = preservation()
    if before != after:
        raise ValueError("Saved paper/confirmation evidence changed")
    for name, expected in activation["bindings"].items():
        if sha(REPO / name) != expected:
            raise ValueError("Frozen input changed during replay")
    receipt = {"date": "2026-10-13", "passed": True,
               "qualification_sha256": q_hash, "imputation_fixtures_sha256": fixtures_hash,
               "activation_sha256": sha(PUBLIC / "activation.json"),
               "auditor_code_sha256": sha(Path(__file__)),
               "source_snapshot_file_sha256": qualification["source_snapshot_file_sha256"],
               "available_source_trials": 979, "unavailable_source_trials": 1,
               "snapshot_arrays_numerically_opened": 1, "source_regional_array_hashes_verified": checked,
               "source_feature_scalars_read": int(vectors.size),
               "fold_numerical_means_independently_recomputed": 20,
               "full_source_numerical_means_independently_recomputed": 2,
               "fold_replays": replays, "full_source_replay": full,
               "old_strict_quality_gate_passed": False, "masked_input_eligibility_passed": True,
               "failed_trial_arrays_opened": 0, "raw_recording_files_opened": 0,
               "targets_or_predictions_numerically_read": 0, "fitted_states_opened": 0,
               "validation_numeric_access": False, "confirmation_access": False,
               "preservation": after,
               "limits": "Independent available-source vector identity and numerical training-only donor-mean replay. No failed-trial recovery, new raw processing, supervised fit or protected export qualification."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    audit()
