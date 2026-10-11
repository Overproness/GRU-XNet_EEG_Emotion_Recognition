"""Authenticate DENS TSV bodies; selectively decode material keys and numeric timing.

The exact access boundary is declared in DENS_Metadata_Timing_Pass_2026-10-12.md.
Complete bodies are transiently transported for Git blob authentication. Rating
and event semantic bytes are not decoded, printed, retained, or used numerically.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import urllib.request


REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "results/development/dens_metadata_qualification_2026-10-12"
INVENTORY = PACKET / "inventory.json"
DESTINATION = PACKET / "metadata_annotations.json"
PROTOCOL = REPO / "docs/publication/DENS_Metadata_Timing_Pass_2026-10-12.md"
COMMIT = "5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c"
RAW = f"https://raw.githubusercontent.com/OpenNeuroDatasets/ds003751/{COMMIT}/"
EVENT_SUBJECTS = ("sub-mit003", "sub-mit081", "sub-mit108")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def ordered_digest(values):
    return sha256(json.dumps(values, separators=(",", ":"), allow_nan=False).encode("ascii"))


def get_allowed_body(entry):
    path = entry["path"]
    if not path.endswith("_beh.tsv") and not (path.endswith("_events.tsv") and path.split("/")[0] in EVENT_SUBJECTS):
        raise PermissionError("Annotation outside the declared pass: " + path)
    request = urllib.request.Request(RAW + path, headers={"User-Agent": "GRU-XNet-DENS-selective-metadata", "Accept-Encoding": "identity"})
    checked = datetime.now(timezone.utc).isoformat()
    with urllib.request.urlopen(request, timeout=30) as response:
        # Full bodies are allowed only for authentication; no full data-row
        # decoder is called. The transient remainder includes outcome bytes.
        body = response.read(entry["size"] + 1)
        access = {"url": RAW + path, "checked_utc": checked, "status": response.status, "final_url": response.geturl(), "content_length": response.headers.get("Content-Length"), "body_bytes_returned_for_authentication": len(body), "body_sha256": sha256(body), "body_github_blob_sha1": blob_sha(body), "body_transport_includes_undecoded_outcome_bytes": True, "complete_annotation_file_retained": False}
    if len(body) != entry["size"] or blob_sha(body) != entry["sha"]:
        raise ValueError("Pinned annotation body authentication failed: " + path)
    access["pinned_complete_blob_authenticated"] = True
    return body, access


def numeric_field(field):
    field = field.strip()
    if field in (b"", b"n/a", b"NA", b"NaN", b"nan"):
        return None
    try:
        value = float(field.decode("ascii"))
    except (UnicodeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def process(entry, sidecar):
    body, access = get_allowed_body(entry)
    rows = body.splitlines()
    if not rows:
        raise ValueError("Empty authenticated annotation file")
    result = {"path": entry["path"], "participant": entry["path"].split("/")[0], "git_blob_sha1": entry["sha"], "git_blob_bytes": entry["size"], "access": access}
    if entry["path"].endswith("_beh.tsv"):
        # No remainder of a behavior data row is decoded.
        first_header = rows[0].split(b"\t", 1)[0].decode("utf-8-sig")
        if first_header != "stimuliName":
            raise ValueError("Changed first-column schema")
        for line in rows[1:]:
            if line and line.count(b"\t") != 10:
                raise ValueError("Behavior byte-level TSV framing changed: " + entry["path"])
        keys = [line.split(b"\t", 1)[0].decode("utf-8") for line in rows[1:] if line]
        unique = sorted(set(keys))
        result.update({"kind": "material_keys", "first_column": first_header, "data_row_count": len(keys), "ordered_material_keys": keys, "unique_material_keys": unique, "unique_material_key_count": len(unique), "duplicate_material_key_count": len(keys) - len(unique), "empty_material_key_count": sum(not key for key in keys), "logical_first_column_utf8_bytes": sum(len(key.encode("utf-8")) for key in keys), "behavior_rating_or_click_fields_decoded": 0})
    else:
        header_parts = rows[0].split(b"\t", 2)
        if header_parts[:2] != [b"onset", b"duration"]:
            raise ValueError("Changed event numeric-column schema")
        onsets, durations, malformed = [], [], 0
        for line in rows[1:]:
            if not line:
                continue
            fields = line.split(b"\t", 2)
            if len(fields) != 3:
                malformed += 1
                continue
            onsets.append(numeric_field(fields[0]))
            durations.append(numeric_field(fields[1]))
            # fields[2] may contain ratings; never decode or inspect it.
        numeric_onsets = [value for value in onsets if value is not None]
        numeric_durations = [value for value in durations if value is not None]
        duration_counts = {}
        for value in numeric_durations:
            duration_counts[str(value)] = duration_counts.get(str(value), 0) + 1
        recording_duration = float(sidecar["RecordingDuration"])
        result.update({
            "kind": "numeric_event_timing", "data_row_count": len(onsets) + malformed,
            "parsed_numeric_pair_rows": len(onsets), "malformed_numeric_prefix_rows": malformed,
            "onset_numeric_count": len(numeric_onsets), "onset_missing_or_nonnumeric_count": len(onsets) - len(numeric_onsets),
            "duration_numeric_count": len(numeric_durations), "duration_missing_or_nonnumeric_count": len(durations) - len(numeric_durations),
            "onset_min": min(numeric_onsets) if numeric_onsets else None, "onset_max": max(numeric_onsets) if numeric_onsets else None,
            "duration_min": min(numeric_durations) if numeric_durations else None, "duration_max": max(numeric_durations) if numeric_durations else None,
            "duration_value_counts": duration_counts,
            "negative_onset_count": sum(value < 0 for value in numeric_onsets), "negative_duration_count": sum(value < 0 for value in numeric_durations),
            "zero_duration_count": sum(value == 0 for value in numeric_durations),
            "duplicate_numeric_onsets": len(numeric_onsets) - len(set(numeric_onsets)),
            "onsets_nondecreasing_among_numeric_rows": all(left <= right for left, right in zip(numeric_onsets, numeric_onsets[1:])),
            "ordered_onset_fields_sha256": ordered_digest(onsets), "ordered_duration_fields_sha256": ordered_digest(durations),
            "numeric_digest_encoding": "JSON array in original row order, float or null, separators comma/colon, ASCII, no trailing newline",
            "sidecar_recording_duration": recording_duration, "sidecar_sampling_hz": sidecar["SamplingFrequency"],
            "tsv_numeric_field_units": "unqualified; field names alone do not establish seconds",
            "conditional_on_seconds_onsets_within_sidecar_recording": all(0 <= value <= recording_duration for value in numeric_onsets),
            "conditional_on_seconds_onset_plus_duration_within_sidecar_recording": all(0 <= onset + duration <= recording_duration for onset, duration in zip(onsets, durations) if onset is not None and duration is not None),
            "event_semantic_or_outcome_fields_decoded": 0,
        })
    return result


def aggregate(receipt):
    materials = receipt["materials"]
    unique = sorted({key for row in materials for key in row["unique_material_keys"]})
    count_by_key = {key: sum(key in row["unique_material_keys"] for row in materials) for key in unique}
    return {
        "behavior_files_authenticated": len(materials),
        "behavior_rows_first_column_decoded": sum(row["data_row_count"] for row in materials),
        "distinct_material_key_strings": len(unique),
        "unique_material_keys": unique, "participants_by_material_key": count_by_key,
        "participants_with_duplicate_material_keys": [row["participant"] for row in materials if row["duplicate_material_key_count"]],
        "event_files_authenticated": len(receipt["timing"]),
        "numeric_event_prefix_rows_processed": sum(row["parsed_numeric_pair_rows"] for row in receipt["timing"]),
        "complete_annotation_body_bytes_returned_for_authentication": sum(row["access"]["body_bytes_returned_for_authentication"] for row in materials + receipt["timing"]),
        "body_transport_includes_undecoded_outcome_bytes": True,
        "usable_trial_count": None, "trial_join_authenticated": False,
    }


def build():
    if not PROTOCOL.exists():
        raise PermissionError("Write the access declaration before network retrieval")
    inventory = json.loads(INVENTORY.read_bytes())
    if inventory["commit"] != COMMIT or inventory["summary"]["potential_eeg_behavior_joins"] != 34:
        raise ValueError("Pinned inventory changed")
    sidecars = {row["path"].split("/")[0]: row["metadata"] for row in inventory["metadata"] if row["path"].endswith("_eeg.json")}
    tasks = [row for row in inventory["tree_entries"] if row["type"] == "blob" and (row["path"].endswith("_beh.tsv") or (row["path"].endswith("_events.tsv") and row["path"].split("/")[0] in EVENT_SUBJECTS))]
    if len(tasks) != 37:
        raise ValueError("Declared annotation roster changed")
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda entry: process(entry, sidecars[entry["path"].split("/")[0]]), tasks))
    result = {
        "schema": "dens_material_numeric_timing_qualification_v1", "date": "2026-10-12", "commit": COMMIT,
        "protocol": str(PROTOCOL.relative_to(REPO)).replace("\\", "/"), "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "inventory_sha256": sha256(INVENTORY.read_bytes()), "script_sha256": sha256(Path(__file__).read_bytes()),
        "materials": [row for row in records if row["kind"] == "material_keys"],
        "timing": [row for row in records if row["kind"] == "numeric_event_timing"],
        "unresolved": [
            "Filename-level material keys are not verified film-title identity or physical stimulus independence.",
            "A behavior row is not yet a verified recorded, complete, target-available trial.",
            "No individual rating scale/observed rating values are inspected.",
            "Numeric event field units require independent SET-latency/codebook qualification.",
            "No event semantic labels are decoded, so trial endpoints remain unqualified.",
            "Full annotation body bytes are transferred transiently for authentication; outcome bytes are not decoded, retained or used.",
        ],
        "original_rating_or_click_values_decoded": 0, "event_semantic_fields_decoded": 0,
        "eeg_waveform_body_requests": 0, "new_fits": 0, "new_predictions": 0,
        "confirmation_outcomes_opened": 0, "research_question_changed": False,
    }
    result["summary"] = aggregate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Offline aggregate and binding replay")
    args = parser.parse_args()
    if args.check:
        result = json.loads(DESTINATION.read_bytes())
        for key, path in (("protocol_sha256", PROTOCOL), ("inventory_sha256", INVENTORY), ("script_sha256", Path(__file__))):
            if result[key] != sha256(path.read_bytes()):
                raise ValueError("Metadata access binding changed: " + key)
        if result["commit"] != COMMIT or aggregate(result) != result["summary"]:
            raise ValueError("Annotation inventory changed")
        for key in ("original_rating_or_click_values_decoded", "event_semantic_fields_decoded", "eeg_waveform_body_requests", "new_fits", "new_predictions", "confirmation_outcomes_opened", "research_question_changed"):
            if result[key] not in (0, False):
                raise PermissionError("Access or experiment boundary changed")
    else:
        if DESTINATION.exists():
            raise FileExistsError("Preserve the existing selective metadata receipt")
        result = build()
        with DESTINATION.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "offline_check": args.check, "summary": result["summary"], "ratings_decoded": 0, "semantic_event_fields_decoded": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
