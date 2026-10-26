"""Read-only scientific/ledger audit of the preserved zero-fit held attempt.

Only public technical evidence and already authorized development caches are
read. All cached identities pass before any numeric cached target conversion.
"""
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from scripts.audit_faced_eegnet_diagnostic import derive_axes, load, pair, sha

REPO = Path(__file__).resolve().parents[1]
BASE = REPO/"results/development"
PACKET = BASE/"faced_eegnet_diagnostic_2026-10-11"
PERMUTED = {"capacity_spectral_permuted", "capacity_covariance_permuted"}


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def guarded_cache_audit(axes, programme, representation_plan, quality, representation):
    for collection in (quality, representation):
        for row in collection:
            p = pair(row)
            if p not in axes.allowed or row["family"] != axes.metadata[p]["family"]:
                raise PermissionError("Protected/changed cache identity before numeric conversion")
            if row.get("acquisition_group", axes.metadata[p]["acquisition_group"]) != axes.metadata[p]["acquisition_group"] or int(row["rotation"]) not in (0, 1, 2):
                raise PermissionError("Changed cached acquisition/rotation")
    canonical = {}
    for row in quality:
        p = pair(row); value = float(row["target"])
        if not math.isfinite(value) or not 0 <= value <= 7 or (p in canonical and value != canonical[p]):
            raise ValueError("Authoritative quality-cache target inconsistency")
        canonical[p] = value
    if set(canonical) != axes.allowed:
        raise ValueError("Incomplete authoritative development target cache")
    expected_permuted = {}; expected_identities = set()
    for rotation in programme["rotation_manifests"]:
        r = int(rotation["rotation"])
        full = [tuple(p) for p in representation_plan["source_sets"][f"r{r}/full"]]
        if set(full) != axes.arms[r]["source_fit"]:
            raise ValueError("Changed historical permutation source set")
        permutation = np.random.default_rng(representation_plan["capacity_permutation_seed_base"]+r).permutation(len(full))
        values = [canonical[p] for p in full]
        expected_permuted[r] = dict(zip(full, [values[int(i)] for i in permutation]))
        expected_identities.update((r, model, p) for model in PERMUTED for p in full)
    observed_synthetic = set(); ordinary_rows = 0; synthetic_rows = 0; synthetic_differences = 0; first_conflict = None
    for row in representation:
        p = pair(row); r = int(row["rotation"]); model = row["model"]; value = float(row["target"])
        if not math.isfinite(value) or not 0 <= value <= 7:
            raise ValueError("Nonfinite/out-of-range cached target")
        if model in PERMUTED and row["arm"] == "capacity_source_fit":
            key = r, model, p
            if key in observed_synthetic or key not in expected_identities or value != expected_permuted[r][p]:
                raise ValueError("Historical seeded permutation cache differs")
            observed_synthetic.add(key); synthetic_rows += 1
            if value != canonical[p]:
                synthetic_differences += 1
                if first_conflict is None:
                    first_conflict = {"rotation": r, "model": model, "arm": row["arm"], "subject": p[0], "vid": p[1]}
        else:
            ordinary_rows += 1
            if value != canonical[p]:
                raise ValueError("Nonpermuted true-target cache inconsistency")
    if observed_synthetic != expected_identities or synthetic_rows != 3360:
        raise ValueError("Changed declared synthetic-control coverage")
    return {"quality_rows_identity_checked": len(quality), "representation_rows_identity_checked": len(representation), "canonical_development_targets": len(canonical), "canonical_cache_consistent": True,
            "ordinary_representation_targets_exactly_equal_canonical": ordinary_rows, "synthetic_permuted_target_rows": synthetic_rows, "synthetic_rows_different_from_original_pair_target": synthetic_differences,
            "synthetic_rows_exactly_match_frozen_seeded_permutations": True, "first_indiscriminate_join_conflict": first_conflict,
            "true_labels_corrupt_or_nonfinite": False, "root_cause": "Bridge joined intentional source-capacity permutation-control labels as if they were original trial labels", "individual_target_values_published_by_this_audit": False}


def audit():
    execution = load(PACKET/"execution.json"); activation = load(PACKET/"activation.json")
    if execution["status"] != "held_or_failed" or execution["failure"] != "ValueError: Cached target nonfinite/inconsistent" or execution["activation_sha256"] != sha(PACKET/"activation.json"):
        raise ValueError("Wrong preserved held attempt")
    binding_count = 0
    for collection in (activation["bindings"], activation["code_bindings"]):
        for rel, expected in collection.items():
            if sha(REPO/rel) != expected: raise ValueError("Held activation binding changed: "+rel)
            binding_count += 1
    programme = load(BASE/"faced_development_protocol_2026-10-11/plan.json")
    technical = load(BASE/"faced_artifact_control_2026-10-11/source_records.json")["records"]+load(BASE/"faced_artifact_control_2026-10-11/validation_records.json")["records"]
    axes = derive_axes(programme, load(BASE/"faced_backup_2026-10-10/participant_reservation.json"), load(BASE/"faced_backup_2026-10-10/stimulus_metadata.json"), technical, load(BASE/"faced_inference_quality_2026-10-11/signal_audit.json")["trial_decisions"])
    original = {pair(r): r for r in technical}; manifest = load(PACKET/"feature_manifest.json"); records = manifest["records"]
    if len(records) != 1260 or len({pair(r) for r in records}) != 1260 or {pair(r) for r in records} != axes.allowed:
        raise ValueError("Changed held replay coverage")
    samples = 0; roles = {"source": 0, "validation": 0}; fallback = 0
    for record in records:
        p = pair(record); old = original[p]; available = axes.decisions[p]["eeg_available"]
        if record["role"] != old["role"] or record["eeg_available"] is not available or record["fallback"] is available:
            raise ValueError("Held role/QC/fallback changed")
        if available:
            for field in ("count_array_sha256", "range_ledger_sha256", "processed_window_sha256", "quality_file_sha256"):
                if record[field] != old[field]: raise ValueError("Held replay technical hash mismatch")
            if not all(record[k] is True for k in ("original_signal_decoded", "count_hash_matches", "qc_numeric_bytes_equal", "processed_window_hash_matches", "quality_replayed")):
                raise ValueError("Incomplete held exact-replay claim")
            start, stop = old["sample_interval"]
            if record["samples_decoded"] != (stop-start)*len(old["channel_indices"]): raise ValueError("Selected count geometry disagreement")
            samples += record["samples_decoded"]; roles[record["role"]] += 1
        else:
            if record["samples_decoded"] != 0 or record["original_signal_decoded"] is not False: raise ValueError("Unavailable EEG opened")
            fallback += 1
    if roles != {"source": 980, "validation": 265} or fallback != 15 or samples != execution["selected_sample_values_decoded"]:
        raise ValueError("Held selective-access ledger disagreement")
    gate = load(PACKET/"source_gate.json")
    if gate != {"passed": True, "source_trials": 980, "source_failures": 0, "targets_joined": False, "validation_signals_decoded": 0, "fits": 0, "window_bytes_in_memory": 421478400, "completed_utc": gate["completed_utc"]}:
        raise ValueError("Source-first gate changed")
    events = [json.loads(line) for line in (PACKET/"events.jsonl").read_text().splitlines()]
    if [r["type"] for r in events] != ["source_replay_started", "source_gate_passed", "execution_failed"]:
        raise ValueError("Unexpected outcome/fitting event in held attempt")
    if datetime.fromisoformat(gate["completed_utc"]) >= datetime.fromisoformat(execution["completed_utc"]): raise ValueError("Source gate chronology failed")
    if load(PACKET/"fit_records.json")["fits"] or rows(PACKET/"checkpoint_predictions.csv") or rows(PACKET/"ensemble_predictions.csv"):
        raise ValueError("Held attempt unexpectedly includes fit outcomes")
    zero_fields = ("new_neural_fits", "completed_neural_fits", "charged_fit_seconds", "checkpoint_rows", "ensemble_rows", "new_original_rating_decoding", "auxiliary_neighbor_or_confirmation_samples_decoded", "peak_cuda_reserved_bytes")
    if any(execution[k] != 0 for k in zero_fields) or any(execution[k] is not False for k in ("original_rating_files_opened", "raw_waveforms_persisted", "confirmation_access", "research_question_changed", "valid_contribution_selection")):
        raise ValueError("Held no-fit/no-protected-access contract failed")
    if any((PACKET/name).exists() for name in ("controls.csv", "targets_reuse.json", "control_reuse.json", "independent_audit.json")):
        raise ValueError("Failed target join was represented as complete")
    quality_path = BASE/"faced_inference_quality_2026-10-11/predictions.csv"
    representation_path = BASE/"faced_representation_v2_2026-10-11/predictions.csv"
    caches = guarded_cache_audit(axes, programme, load(BASE/"faced_representation_v2_2026-10-11/plan.json"), rows(quality_path), rows(representation_path))
    names = ("activation.json", "source_gate.json", "feature_manifest.json", "execution.json", "fit_records.json", "events.jsonl", "checkpoint_predictions.csv", "ensemble_predictions.csv")
    return {"passed": True, "scope": "Preserved held-attempt public provenance/count/failure audit, not a completed model diagnostic", "activation_bindings_verified": binding_count, "held_packet_bindings": {name: sha(PACKET/name) for name in names}, "source_gate_passed_before_cached_join": True,
            "metadata_trials": 1260, "source_signal_trials_replayed": 980, "validation_signal_trials_replayed": 265, "validation_fallback_trials_without_EEG_access": 15, "selected_sample_values_ledger": samples,
            "cache_diagnosis": caches, "new_neural_fits": 0, "completed_neural_fits": 0, "original_rating_or_waveform_files_opened_by_auditor": False, "protected_outcomes_opened": False,
            "continuation": {"new_activation_required": True, "preserve_held_attempt_and_frozen_v1": True, "change_scope": "Separately version target/cache join and independent audit; authoritative true targets from quality cache; known synthetic permutation rows never become original labels", "same_original_target_roles_quality_model_grid_and_question": True, "rescue_fits_or_expanded_grid_authorized": False, "no_learning_or_incremental_prediction_result_available": True}}


if __name__ == "__main__":
    result = audit()
    (PACKET/"held_attempt_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))
