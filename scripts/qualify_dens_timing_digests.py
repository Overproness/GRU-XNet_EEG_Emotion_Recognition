"""Repeat three permitted DENS numeric-prefix reads for float64 digest comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

try:
    from scripts import qualify_dens_annotations as preceding
except ImportError:
    import qualify_dens_annotations as preceding


REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "results/development/dens_metadata_qualification_2026-10-12"
DECLARATION = REPO / "docs/publication/DENS_Timing_Digest_Addendum_2026-10-12.md"
ANNOTATIONS = PACKET / "metadata_annotations.json"
DESTINATION = PACKET / "timing_digests.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(values):
    return sha(b"".join(struct.pack("<d", value) for value in values))


def build():
    if not DECLARATION.exists():
        raise PermissionError("Write the numeric digest declaration before access")
    inventory = json.loads(preceding.INVENTORY.read_bytes())
    annotations = json.loads(ANNOTATIONS.read_bytes())
    expected = {row["participant"]: row for row in annotations["timing"]}
    rows = []
    for subject in preceding.EVENT_SUBJECTS:
        path = subject + "/eeg/" + subject + "_task-emotion_events.tsv"
        entry = next(row for row in inventory["tree_entries"] if row["path"] == path)
        body, access = preceding.get_allowed_body(entry)
        if access["body_sha256"] != expected[subject]["access"]["body_sha256"]:
            raise ValueError("Authenticated event body differs from first pass")
        lines = body.splitlines()
        if lines[0].split(b"\t", 2)[:2] != [b"onset", b"duration"]:
            raise ValueError("Numeric-prefix schema changed")
        onsets, durations = [], []
        for line in lines[1:]:
            if not line:
                continue
            fields = line.split(b"\t", 2)
            if len(fields) != 3:
                raise ValueError("Numeric-prefix framing changed")
            onset, duration = preceding.numeric_field(fields[0]), preceding.numeric_field(fields[1])
            if onset is None or duration is None:
                raise ValueError("Exact numeric digest needs finite timing values")
            onsets.append(onset)
            durations.append(duration)
            # No event text or values from the remaining byte string are decoded.
        if len(onsets) != expected[subject]["parsed_numeric_pair_rows"]:
            raise ValueError("Event row count changed")
        rows.append({
            "participant": subject, "path": path, "access": access,
            "events": len(onsets), "onset_digest_little_endian_float64": digest(onsets),
            "duration_digest_little_endian_float64": digest(durations),
            "onset_digest_raw_export_matches_rounded_6_decimal_export": digest(onsets) == digest([round(value, 6) for value in onsets]),
            "onset_ordered_json_digest_matches_initial_receipt": preceding.ordered_digest(onsets) == expected[subject]["ordered_onset_fields_sha256"],
            "duration_ordered_json_digest_matches_initial_receipt": preceding.ordered_digest(durations) == expected[subject]["ordered_duration_fields_sha256"],
            "event_semantic_fields_decoded": 0,
        })
    return {
        "schema": "dens_numeric_timing_float64_digest_addendum_v1", "date": "2026-10-12",
        "bindings": {str(path.relative_to(REPO)).replace("\\", "/"): sha(path.read_bytes()) for path in (DECLARATION, ANNOTATIONS, preceding.INVENTORY, Path(__file__), Path(preceding.__file__))},
        "records": rows,
        "encoding": "Ordered numeric values, IEEE754 little-endianfloat64 bytes, SHA256; no public raw timing sequences",
        "full_annotation_body_bytes_returned_for_authentication": sum(row["access"]["body_bytes_returned_for_authentication"] for row in rows),
        "undecoded_outcome_bytes_transported_for_authentication": True,
        "event_semantic_fields_decoded": 0, "individual_rating_values_decoded": 0,
        "waveform_body_requests": 0, "new_fits": 0, "new_predictions": 0,
        "confirmation_outcomes_opened": 0, "research_question_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Offline binding replay")
    args = parser.parse_args()
    if args.check:
        result = json.loads(DESTINATION.read_bytes())
        for rel, expected in result["bindings"].items():
            if sha((REPO / rel).read_bytes()) != expected:
                raise ValueError("Numeric digest binding changed: " + rel)
        for key in ("event_semantic_fields_decoded", "individual_rating_values_decoded", "waveform_body_requests", "new_fits", "new_predictions", "confirmation_outcomes_opened", "research_question_changed"):
            if result[key] not in (0, False):
                raise PermissionError("Numeric access boundary changed")
    else:
        if DESTINATION.exists():
            raise FileExistsError("Preserve the existing numeric-digest receipt")
        result = build()
        with DESTINATION.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "offline_check": args.check, "records": [{key: value for key, value in row.items() if key != "access"} for row in result["records"]], "event_semantic_fields_decoded": 0}))


if __name__ == "__main__":
    main()
