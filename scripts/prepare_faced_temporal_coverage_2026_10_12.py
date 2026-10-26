"""Prepare metadata-only source temporal coverage and a frozen pilot allowlist.

Reads public metadata/code only. No original recording, private array, rating
table, fit, validation signal or protected signal is opened by this script.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_temporal_coverage import (
    CATEGORIES, DEVELOPMENT_CLIPS, PILOT_SALT, REGIONS,
    SourceOnlySegmentGuard, choose_source_pilot, segment_bounds, union_sample_count,
)

BASE = REPO / "results/development"
PUBLIC = BASE / "faced_temporal_coverage_preflight_2026-10-12"
INPUTS = {
    "programme": BASE / "faced_development_protocol_2026-10-11/plan.json",
    "roles": BASE / "faced_backup_2026-10-10/participant_reservation.json",
    "catalogue": BASE / "faced_codebooks_2026-10-10/audit.json",
    "joins": BASE / "faced_raw_2026-10-11/joins_numeric_projection.json",
    "headers": BASE / "faced_raw_2026-10-11/headers.json",
    "source_records": BASE / "faced_artifact_control_2026-10-11/source_records.json",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build():
    values = {name: json.loads(path.read_bytes()) for name, path in INPUTS.items()}
    roles = values["roles"]
    people = frozenset(roles["development_source"])
    validation = frozenset(roles["development_validation"])
    protected = frozenset(roles["confirmation"])
    if len(people) != 70 or len(validation) != 20 or len(protected) != 33:
        raise PermissionError("Frozen participant role counts changed")
    if people & validation or people & protected or validation & protected:
        raise PermissionError("Frozen participant roles overlap")
    source_pairs = {(row["subject"], row["vid"])
                    for rotation in values["programme"]["rotation_manifests"]
                    for row in rotation["arms"]["source_fit"]}
    if len(source_pairs) != 980 or source_pairs != {(p, v) for p in people for v in DEVELOPMENT_CLIPS}:
        raise PermissionError("Source rectangle differs from the declared 70-by-14 roster")
    originals = values["source_records"]["records"]
    if len(originals) != 980 or {(r["subject"], r["vid"]) for r in originals} != source_pairs:
        raise PermissionError("Existing source lineage is incomplete or duplicated")
    catalogue = {r["clip"]: r for r in values["catalogue"]["stimuli"] if r["clip"] in DEVELOPMENT_CLIPS}
    headers = {r["subject"]: r["data.bdf"] for r in values["headers"]["records"] if r["subject"] in people}
    spans = {(r["subject"], span["vid"]): span for r in values["joins"]["records"] if r["subject"] in people
             for span in r["spans"] if (r["subject"], span["vid"]) in source_pairs}
    if set(spans) != source_pairs or set(headers) != people:
        raise PermissionError("Source marker/header joins are incomplete")
    records = []
    for old in sorted(originals, key=lambda r: (r["subject"], r["vid"])):
        pair = old["subject"], old["vid"]
        # Reject any changed role before interpreting timing/range metadata.
        if pair not in source_pairs or old["role"] != "source" or not old["technical_passed"]:
            raise PermissionError("Non-source or technically unqualified lineage")
        span = spans[pair]; header = headers[pair[0]]; rate = old["native_rate_hz"]
        literals = header["columns"]["samples_per_record"]
        if any(isinstance(v, bool) or not (isinstance(v, int) or
                   isinstance(v, str) and v.isascii() and v.isdecimal()) for v in literals):
            raise ValueError("Header sample counts are not exact integer literals")
        channel_sizes = [int(v) for v in literals]
        if len(set(channel_sizes)) != 1 or set(header["sample_rates_hz"]) != {float(rate)}:
            raise ValueError("Recording rate/sample geometry changed")
        samples = header["record_count"] * channel_sizes[0]
        intervals = segment_bounds(span["start_seconds"], span["end_seconds"], rate, samples)
        if intervals["late"] != old["sample_interval"]:
            raise ValueError("Late coverage segment differs from the prior exact native range")
        records.append({
            "subject": pair[0], "vid": pair[1], "role": "source",
            "assigned_category": catalogue[pair[1]]["assigned_category"],
            "family": catalogue[pair[1]]["source_film_sha256"],
            "acquisition_group": old["acquisition_group"], "native_rate_hz": rate,
            "marker_start_seconds": span["start_seconds"], "marker_end_seconds": span["end_seconds"],
            "recording_samples": samples, "segments": intervals,
            "covered_native_sample_count": union_sample_count(intervals.values()),
            "late_range_exactly_matches_prior": True,
            "channel_indices": old["channel_indices"],
            "original_data_sha256": old["original_data_sha256"],
            "original_header_sha256": old["original_header_sha256"],
            "prior_late_count_array_sha256": old["count_array_sha256"],
            "prior_late_processed_window_sha256": old["processed_window_sha256"],
            "prior_late_quality_file_sha256": old["quality_file_sha256"],
        })
    selected = choose_source_pilot(records)
    pilot_rows = [{"subject": r["subject"], "vid": r["vid"], "role": "source", "region": region,
                   "native_rate_hz": r["native_rate_hz"], "sample_interval": r["segments"][region]}
                  for r in selected for region in REGIONS]
    guard = SourceOnlySegmentGuard(source_people=people, validation_people=validation,
                                  protected_people=protected, source_pairs=sorted(source_pairs), pilot_rows=pilot_rows)
    if len(guard.pilot_ranges) != 24 or {r["assigned_category"] for r in selected} != set(CATEGORIES):
        raise ValueError("Pilot category/segment count changed")
    plan = {
        "date": "2026-10-12", "status": "prepared_source_only_not_activated",
        "scope": "Metadata-only distributed temporal coverage preflight; no EEG, private arrays, ratings or fits",
        "source_trial_count": 980, "source_segment_count": 2940,
        "source_people": sorted(people), "validation_people_denied": sorted(validation),
        "protected_people_denied": sorted(protected), "allowed_clips": sorted(DEVELOPMENT_CLIPS),
        "source_records": records,
        "region_rule": {"early": "ceil(marker_start * native_rate)",
                        "middle": "floor((ceil(marker_start * native_rate) + floor(marker_end * native_rate) - 32 * native_rate) / 2)",
                        "late": "floor(marker_end * native_rate) - 32 * native_rate",
                        "stop": "native start + 32 * native rate", "processor_core": "local seconds 2..30; seven 4-second windows"},
        "coverage_interpretation": "Equal-duration distributed early/middle/late observations. Regions can overlap substantially; neither exhaustive full-video coverage nor independent windows.",
        "pilot": {"salt": PILOT_SALT, "trial_count": 8, "segment_count": 24,
                  "trials": [{k: r[k] for k in ("subject", "vid", "assigned_category", "acquisition_group")} for r in selected],
                  "segments": pilot_rows,
                  "maximum_processed_float32_bytes_if_all_retained": 24 * 7 * 30 * 512 * 4,
                  "counts_sample_values_planned": sum(30 * 32 * r["native_rate_hz"] for r in pilot_rows),
                  "outcome_blind_selection": True},
        "input_bindings": {path.relative_to(REPO).as_posix(): digest(path) for path in INPUTS.values()},
        "code_bindings": {name: digest(REPO / name) for name in (
            "gruxnet/faced_temporal_coverage.py", "scripts/prepare_faced_temporal_coverage_2026_10_12.py",
            "tests/test_faced_temporal_coverage.py")},
        "signal_access_activated": False, "validation_signal_access": False,
        "confirmation_access": False, "original_rating_access": False,
        "new_fits": 0, "research_question_changed": False,
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    exclusive_json(PUBLIC / "plan.json", plan)
    exclusive_json(PUBLIC / "receipt.json", {"plan_sha256": digest(PUBLIC / "plan.json"),
        "source_range_checks": 2940, "late_exact_identity_checks": 980, "pilot_rows": 24,
        "raw_recording_files_opened": 0, "private_array_files_opened": 0, "outcome_files_opened": 0,
        "waveform_samples_decoded": 0, "fits": 0, "protected_or_validation_signal_access": False})
    return {"packet": PUBLIC.relative_to(REPO).as_posix(), "source_trials": 980, "pilot_segments": 24,
            "plan_sha256": digest(PUBLIC / "plan.json"), "signal_access_activated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True,
                        help="Write a metadata-only inactive plan; never opens EEG or targets")
    parser.parse_args()
    print(json.dumps(build(), indent=2))
