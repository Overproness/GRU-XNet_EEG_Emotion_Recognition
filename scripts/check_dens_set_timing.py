"""Bounded selective acquisition-timing metadata, never event responses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from inspect_dens_set_metadata import Matrix, SUBJECTS, inspect_bytes, tag


def ordered_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def event_timing(body):
    # The authenticated probe files are uncompressed MAT5; fail closed on changes.
    header = inspect_bytes(body)
    endian = header["mat_endian"]
    if header["inflated_container_bytes"]:
        raise ValueError("This separate timing pass permits only uncompressed probe files")
    typ, payload, next_pos = tag(memoryview(body), 128, endian)
    if typ != 14 or next_pos != len(body):
        raise ValueError("Unexpected SET geometry")
    eeg = Matrix(payload, endian)
    events = None
    for _, field, child in eeg.members():
        if field == "event":
            events = Matrix(child, endian)
    if events is None or events.cls != 2:
        raise ValueError("Missing event metadata")
    values = {"latency": [], "duration": []}
    for row, field, child in events.members():
        if field in values:
            values[field].append(Matrix(child, endian).scalar())
    if any(len(v) != events.count for v in values.values()):
        raise ValueError("Timing field coverage mismatch")
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite event timing")
    return header, values


def qualify(work_dir, packet, output):
    bound = json.loads((packet / "set_metadata.json").read_text())
    annotations = json.loads((packet / "metadata_annotations.json").read_text()) if (packet / "metadata_annotations.json").exists() else None
    records = []
    for record in bound["records"]:
        subject = record["subject"]
        if subject not in SUBJECTS:
            raise ValueError("Undeclared subject")
        body = (work_dir / (subject + ".set")).read_bytes()
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise ValueError("SET changed after binding")
        header, values = event_timing(body)
        pnts = header["acquisition"]["pnts"]
        rate = header["acquisition"]["srate"]
        lat = np.asarray(values["latency"], dtype=float)
        dur = np.asarray(values["duration"], dtype=float)
        candidates = {"raw_latency": lat, "latency_minus_one": lat - 1,
                      "latency_minus_one_divided_by_srate": (lat - 1) / rate,
                      "latency_divided_by_srate": lat / rate}
        summary = {"subject": subject, "events": len(lat), "srate": rate, "pnts": pnts,
                   "latency_min": float(lat.min()), "latency_max": float(lat.max()),
                   "duration_min": float(dur.min()), "duration_max": float(dur.max()),
                   "latency_within_one_based_record": bool(((lat >= 1) & (lat <= pnts)).all()),
                   "latencies_monotonic": bool((np.diff(lat) >= 0).all()),
                   "candidate_onset_digests": {k: ordered_digest(v) for k, v in candidates.items()},
                   "duration_digest_raw": ordered_digest(dur),
                   "duration_digest_divided_by_srate": ordered_digest(dur / rate),
                   "source_set_sha256": record["sha256"],
                   "event_fields_decoded": ["latency", "duration"],
                   "event_response_fields_decoded": 0, "individual_target_values_decoded": 0}
        records.append(summary)
    receipt = {"scope": "Timing-only event pass; no event labels/categories/responses", "records": records,
               "set_file_downloads": 0, "fdt_body_requests": 0,
               "event_timing_values_decoded": sum(r["events"] * 2 for r in records),
               "individual_target_values_decoded": 0, "new_fits": 0, "new_predictions": 0,
               "note": "Ordered digests support exact export conversion checks; endpoints/phase semantics are not inferred"}
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"subjects": len(records), "timing_values": receipt["event_timing_values_decoded"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.work_dir, args.packet, args.output)
