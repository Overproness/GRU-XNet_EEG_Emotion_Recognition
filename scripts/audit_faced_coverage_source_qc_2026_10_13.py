"""Independent source-only QC arithmetic and byte-ledger audit.

Reads only declared private source QC/access/artifact JSON after complete public
identity qualification. Feature archives are hashed, never decoded. No original
recording, waveform/feature array, rating, prediction or fitted state is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
PUBLIC = BASE / "faced_coverage_selection_2026-10-13"
PRIVATE = REPO.parent / "publication_runs/faced_coverage_selection_2026-10-13"
SOURCE_PLAN = BASE / "faced_temporal_coverage_preflight_2026-10-12/plan.json"
CHANNELS = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
            "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4",
            "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2"]
REGIONS = ("early", "middle", "late")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def number(value, *, upper=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0 or upper is not None and value > upper):
        raise ValueError("Invalid authenticated source QC number")
    return value


def artifact_decision(artifact):
    """Reconstruct pre-CAR repair scope directly from engineering metrics."""
    sd, excursion = artifact["pre_car_sd_uv"], artifact["pre_car_peak_to_peak_uv"]
    if len(sd) != 30 or len(excursion) != 30:
        raise ValueError("Artifact metrics changed channel geometry")
    bad = [i for i in range(30) if number(sd[i]) > 100 or number(excursion[i]) > 1000]
    exceeded = len(bad) > 3
    expected = "unchanged" if not bad else ("held_beyond_repair_capacity" if exceeded
                                             else "fixed_spherical_interpolation_before_CAR")
    repaired = [] if not bad or exceeded else [CHANNELS[i] for i in bad]
    if (artifact["detected_indices"] != bad or artifact["detected_channels"] != [CHANNELS[i] for i in bad]
            or artifact["repair_capacity_exceeded"] is not exceeded
            or artifact["decision"] != expected or artifact["repaired_channels"] != repaired
            or artifact["donors_unchanged"] is not True or artifact["trials_or_windows_dropped"] != 0
            or number(artifact["interpolation_weight_max_discrepancy"]) > 1e-9):
        raise ValueError("Artifact scope/repair decision differs from metrics")
    return expected, exceeded, len(repaired)


def availability(quality, repair_capacity_exceeded):
    """Independent fixed-rule calculation; do not import producer's checker."""
    channels = quality["channels"]
    if ([c["channel"] for c in channels] != CHANNELS or quality["valid_windows"] != 7
            or quality["windows_dropped"] != 0):
        raise ValueError("Authenticated source QC geometry changed")
    reasons = []
    if quality["finite_processed_values"] is not True:
        reasons.append({"channel": "all", "reason": "nonfinite_processed_values"})
    if repair_capacity_exceeded:
        reasons.append({"channel": "all", "reason": "repair_capacity_exceeded"})
    flagged = []
    for channel in channels:
        tests = [
            (number(channel["raw_rail_fraction"], upper=1) > 0.0001, "adc_rail"),
            (number(channel["raw_identical_adjacent_fraction"], upper=1) > 0.95, "stationary_counts"),
            (number(channel["processed_sd_uv"]) < 0.1, "low_processed_variation"),
            (number(channel["processed_sd_uv"]) > 100, "high_processed_variation"),
            (number(channel["processed_peak_to_peak_uv"]) > 1000, "high_processed_excursion"),
            (number(channel["processed_49_51_to_1_45_power_ratio"]) > 0.1, "residual_50hz_power"),
        ]
        ordinary = [name for active, name in tests[:5] if active]
        if channel["review_flags"] != ordinary:
            raise ValueError("Saved ordinary QC flags differ from metrics")
        if ordinary:
            flagged.append(channel["channel"])
        reasons.extend({"channel": channel["channel"], "reason": name} for active, name in tests if active)
    if quality["flagged_channels"] != flagged:
        raise ValueError("Saved flagged-channel list differs")
    return not reasons, reasons


def native_ranges(header, interval, selected_channels):
    """Reconstruct the physical BDF byte geometry without sample transport."""
    sizes = [int(x) for x in header["columns"]["samples_per_record"]]
    if len(set(sizes)) != 1 or not sizes or sizes[0] <= 0 or header["sample_width"] != 3:
        raise ValueError("Outside qualified native BDF geometry")
    begin, end = interval
    per_record, record_bytes = sizes[0], sum(sizes) * 3
    if not 0 <= begin < end <= int(header["record_count"]) * per_record:
        raise ValueError("Native source interval outside recording")
    if len(selected_channels) != 30 or len(set(selected_channels)) != 30:
        raise ValueError("Native source channel geometry changed")
    prefixes = [sum(sizes[:c]) * 3 for c in selected_channels]
    rows = []
    for record in range(begin // per_record, (end - 1) // per_record + 1):
        lo, hi = max(begin, record * per_record), min(end, (record + 1) * per_record)
        for channel, prefix in zip(selected_channels, prefixes):
            rows.append({"channel_index": channel, "start_sample": lo, "stop_sample": hi,
                         "offset": int(header["header_bytes"]) + record * record_bytes + prefix
                                   + 3 * (lo - record * per_record), "bytes": (hi - lo) * 3})
    return rows


def qualify_public_rows(rows, source_plan):
    authoritative = {(r["subject"], r["vid"]): r for r in source_plan["source_records"]}
    if (len(authoritative) != 980 or len(source_plan["source_people"]) != 70
            or {p for p, _ in authoritative} != set(source_plan["source_people"])
            or set(source_plan["source_people"]) & (set(source_plan["validation_people_denied"])
                                                       | set(source_plan["protected_people_denied"]))):
        raise PermissionError("Exact source reservation required before private reads")
    expected = [(r["subject"], r["vid"], region) for r in source_plan["source_records"] for region in REGIONS]
    identities = []
    for row in rows:
        key = row["subject"], row["vid"], row["region"]
        if (row["role"] != "source" or key[:2] not in authoritative or key[2] not in REGIONS
                or type(row["eeg_available"]) is not bool or not isinstance(row["reasons"], list)):
            raise PermissionError("Undeclared source QC identity or decision")
        old = authoritative[key[:2]]
        if (row["sample_interval"] != old["segments"][key[2]]
                or row["native_rate_hz"] != old["native_rate_hz"]
                or row["samples_decoded"] != 30 * 32 * old["native_rate_hz"]
                or row["extra_boundary_samples_decoded"] != 0
                or row["late_exact_replay_passed"] is not (key[2] == "late")):
            raise PermissionError("Source range, geometry or replay scope changed")
        identities.append(key)
    if identities != expected or len(set(identities)) != 2940:
        raise PermissionError("Require complete frozen source region order/rectangle")
    return authoritative


def rehash(values):
    for name, expected in values.items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()) or digest(path) != expected:
            raise ValueError("Preserved binding changed: " + name)


def preservation():
    paper = load(REPO / "paper/exploratory_evaluation_2026-10-12/build_receipt.json")
    confirmation = load(BASE / "faced_confirmation_freeze_2026-10-12/activation.json")
    rehash(paper["bindings"])
    count = 0
    for key in ("bindings", "prior_completed_evidence_bindings"):
        rehash(confirmation[key]); count += len(confirmation[key])
    archive = REPO / "docs/paper_archive/2026-10-12-before-improvement"
    files = load(archive / "manifest.json")["files"]
    for row in files:
        if digest(REPO / row["source"]) != row["sha256"] or digest(archive / row["archive"]) != row["sha256"]:
            raise ValueError("Saved paper archive changed")
    return {"paper_bindings": len(paper["bindings"]), "archive_source_copy_pairs": len(files),
            "inactive_confirmation_bindings": count}


def public_git_bindings(commit, activation):
    full = subprocess.run(["git", "rev-parse", commit + "^{commit}"], cwd=REPO,
                          capture_output=True, check=True).stdout.decode().strip()
    for name, expected in activation["bindings"].items():
        blob = subprocess.run(["git", "show", full + ":" + name], cwd=REPO,
                              capture_output=True, check=True).stdout
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("Binding differs from pre-access Git bytes: " + name)
    return full


def audit(freeze_commit):
    started = time.monotonic()
    if not (PUBLIC / "source_gate.json").exists() or (PUBLIC / "preparation_failure.json").exists():
        raise PermissionError("Complete source preparation required before private QC audit")
    if (PUBLIC / "fitting_started.json").exists():
        raise PermissionError("Source QC audit must precede every fitting input attempt")
    destination = PUBLIC / "source_qc_independent_audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing audit receipt")
    activation_path = PUBLIC / "activation_v2.json"
    gate_path = PUBLIC / "source_gate.json"
    gate_hash = digest(gate_path)
    activation, gate, plan = load(activation_path), load(gate_path), load(SOURCE_PLAN)
    rehash(activation["bindings"])
    commit = public_git_bindings(freeze_commit, activation)
    if (activation["confirmation_access"] is not False or activation["validation_signal_access"] is not False
            or activation["original_rating_access"] is not False
            or gate["activation_sha256"] != digest(activation_path)
            or gate["cached_targets_converted"] != 0 or gate["new_fits"] != 0
            or gate["original_rating_access"] is not False or gate["confirmation_access"] is not False
            or gate["validation_signal_access"] is not False):
        raise PermissionError("Source QC access scope changed")
    ledger_path = PUBLIC / "source_regions.jsonl"
    ledger_hash = digest(ledger_path)
    if ledger_hash != gate["source_regions_sha256"]:
        raise ValueError("Source QC public ledger changed")
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    authoritative = qualify_public_rows(rows, plan)
    original = {(r["subject"], r["vid"]): r for r in load(BASE / "faced_artifact_control_2026-10-11/source_records.json")["records"]}
    headers = {r["subject"]: r["data.bdf"] for r in load(BASE / "faced_raw_2026-10-11/headers.json")["records"]
               if r["subject"] in plan["source_people"]}
    before = preservation()
    range_count = channel_count = late_replays = late_quality_replays = 0
    counts = 0; artifacts = {}; failures = []; repairs = Counter(); reasons_count = Counter()
    root = PRIVATE.resolve()
    for index, row in enumerate(rows):
        pair = row["subject"], row["vid"]
        # Every identity/range was already qualified for the entire rectangle.
        stem = f"{pair[0]}_clip{pair[1]:02d}_{row['region']}"
        paths = {name: (root / (stem + suffix)).resolve() for name, suffix in
                 (("quality", "_quality.json"), ("access", "_access.json"),
                  ("artifact", "_artifact.json"), ("feature", "_features.npz"))}
        if any(not path.is_relative_to(root) for path in paths.values()):
            raise PermissionError("Private source QC path escaped its declared directory")
        authenticated = {}
        for name, field in (("quality", "quality_file_sha256"), ("access", "access_file_sha256")):
            payload = paths[name].read_bytes()
            if hashlib.sha256(payload).hexdigest() != row[field]:
                raise ValueError("Declared source QC/cache hash differs: " + stem + "/" + name)
            authenticated[name] = json.loads(payload)
        if digest(paths["feature"]) != row["feature_file_sha256"]:
            raise ValueError("Declared source feature-file hash differs")
        artifact_bytes = paths["artifact"].read_bytes()
        artifacts[stem] = hashlib.sha256(artifact_bytes).hexdigest()
        quality, access, artifact = authenticated["quality"], authenticated["access"], json.loads(artifact_bytes)
        decision, capacity, repaired = artifact_decision(artifact)
        if decision != row["artifact_decision"] or repaired != row["repair_count"]:
            raise ValueError("Public artifact/repair record differs")
        available, reasons = availability(quality, capacity)
        if available is not row["eeg_available"] or reasons != row["reasons"]:
            raise ValueError("Independent availability differs from public decision")
        if not available:
            failures.append({k: row[k] for k in ("subject", "vid", "region", "reasons")})
            reasons_count.update(r["reason"] for r in reasons)
        repairs[(row["region"], repaired)] += 1
        channel_count += len(quality["channels"])
        expected_ranges = native_ranges(headers[pair[0]], row["sample_interval"], authoritative[pair]["channel_indices"])
        expected_hash = hashlib.sha256(json.dumps(expected_ranges, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (access["ranges"] != expected_ranges or access["range_ledger_sha256"] != expected_hash
                or row["range_ledger_sha256"] != expected_hash
                or access["sample_values_decoded"] != row["samples_decoded"]
                or access["signal_bytes_decoded"] != row["samples_decoded"] * 3
                or access["extra_boundary_samples_decoded"] != 0
                or access["all_counts_scalar_reference_checked"] is not True):
            raise ValueError("Independent native byte geometry differs")
        range_count += len(expected_ranges); counts += row["samples_decoded"]
        if row["region"] == "late":
            if (row["count_array_sha256"] != authoritative[pair]["prior_late_count_array_sha256"]
                    or row["processed_window_sha256"] != authoritative[pair]["prior_late_processed_window_sha256"]
                    or row["spectral_feature_sha256"] != original[pair]["spectral_feature_sha256"]):
                raise ValueError("Historical late output identity differs")
            late_replays += 1
            encoded = json.dumps(quality, indent=2, allow_nan=False) + "\n"
            alternatives = {hashlib.sha256(encoded.encode()).hexdigest(),
                            hashlib.sha256(encoded.replace("\n", "\r\n").encode()).hexdigest()}
            if authoritative[pair]["prior_late_quality_file_sha256"] not in alternatives:
                raise ValueError("Historical full late QC dictionary differs")
            late_quality_replays += 1
        if (index + 1) % 420 == 0:
            print(json.dumps({"audit_regions": index + 1, "total": 2940,
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    passed_gate = not failures
    by_region = {region: {"regions": 980, "unavailable": sum(not r["eeg_available"] for r in rows if r["region"] == region)}
                 for region in REGIONS}
    if (gate["passed"] is not passed_gate or gate["failures"] != failures or gate["by_region"] != by_region
            or gate["source_trials"] != 980 or gate["source_regions"] != 2940 or gate["source_people"] != 70
            or gate["unavailable_regions"] != len(failures)
            or gate["unavailable_trials"] != len({(r["subject"], r["vid"]) for r in failures})
            or counts != 1915200000 or gate["counts_sample_values_decoded"] != counts
            or counts != activation["counts_sample_values_cap"] or late_replays != 980
            or late_quality_replays != 980 or gate["late_exact_replays"] != 980
            or number(gate["elapsed_seconds"]) > activation["preparation_wall_seconds_cap"]
            or number(gate["cpu_seconds"]) > activation["preparation_cpu_seconds_cap"]):
        raise ValueError("Complete source gate/counters differ from independent calculation")
    rehash(activation["bindings"])
    after = preservation()
    if before != after or digest(ledger_path) != ledger_hash or digest(gate_path) != gate_hash:
        raise ValueError("Bound source ledger or preserved papers changed during audit")
    receipt = {"date": "2026-10-13", "passed": True, "source_quality_gate_passed": passed_gate,
               "status": "passed_independent_source_qc_arithmetic", "freeze_commit": commit,
               "source_regions_sha256": ledger_hash, "source_gate_sha256": gate_hash,
               "activation_sha256": digest(activation_path), "auditor_code_sha256": digest(Path(__file__)),
               "activation_bindings": len(activation["bindings"]), "frozen_git_bindings": len(activation["bindings"]),
               "source_trials": 980, "source_people": 70, "source_regions": 2940,
               "source_channel_qc_records": channel_count, "native_byte_range_records": range_count,
               "native_scalar_counts": counts, "native_decoded_sample_bytes": counts * 3,
               "private_source_quality_files": 2940, "private_source_access_files": 2940,
               "private_source_artifact_files": 2940, "source_feature_files_hashed_without_decoding": 2940,
               "artifact_binding_scope": "Current produced artifact bytes independently bound here; no prior public per-region artifact digest existed",
               "current_artifact_bindings": artifacts, "late_output_identity_replays": late_replays,
               "full_late_quality_dictionary_hash_replays": late_quality_replays,
               "unavailable_regions": len(failures),
               "unavailable_trials": len({(r['subject'], r['vid']) for r in failures}),
               "by_region": by_region, "failure_reason_records": dict(sorted(reasons_count.items())),
               "repair_counts_by_region": {region: {str(k): v for (r, k), v in sorted(repairs.items()) if r == region}
                                          for region in REGIONS},
               "preservation": after, "elapsed_seconds": time.monotonic() - started,
               "original_recording_files_opened": 0, "private_waveform_feature_model_arrays_decoded": 0,
               "outcome_numeric_values_interpreted": 0, "target_prediction_csv_parsed": False,
               "target_prediction_csv_integrity_transport_only": True, "private_fitted_states_opened": 0,
               "validation_numeric_access": False, "confirmation_access": False,
               "limits": "Independent source QC/range arithmetic and historical output/hash agreement only. No second raw waveform/filter/spectral execution; source QC success is not improved accuracy, neural specificity or independent confirmation."}
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "current_artifact_bindings"}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    audit(args.freeze_commit)
