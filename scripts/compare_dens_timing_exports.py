"""Compare declared, non-outcome timing metadata without waveform access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_dens_set_timing import event_timing


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(packet, work_dir):
    headers = json.loads((packet / "set_metadata.json").read_text())
    digests = json.loads((packet / "timing_digests.json").read_text())
    indexed = {r["participant"]: r for r in digests["records"]}
    records = []
    for rec in headers["records"]:
        subject = rec["subject"]
        path = work_dir / (subject + ".set")
        if sha(path) != rec["sha256"]:
            raise ValueError("Bound SET changed")
        header, values = event_timing(path.read_bytes())
        rounded = np.asarray([float(format(v, ".6f")) for v in values["latency"]], dtype="<f8")
        onset_digest = hashlib.sha256(rounded.tobytes()).hexdigest()
        duration_digest = hashlib.sha256(np.asarray(values["duration"], dtype="<f8").tobytes()).hexdigest()
        tsv = indexed[subject]
        onset_match = onset_digest == tsv["onset_digest_little_endian_float64"]
        duration_match = duration_digest == tsv["duration_digest_little_endian_float64"]
        count_match = len(rounded) == tsv["events"]
        if not all((onset_match, duration_match, count_match)):
            raise ValueError("Declared sample-latency export hypothesis failed")
        rate = header["acquisition"]["srate"]
        seconds = (rounded - 1) / rate
        records.append({"participant": subject, "event_count": len(rounded),
                        "set_latency_formatted_6_decimals_equals_tsv_onset": onset_match,
                        "raw_set_duration_equals_tsv_duration": duration_match,
                        "event_count_agrees": count_match,
                        "set_latency_rounded_6_decimal_digest": onset_digest,
                        "set_duration_digest": duration_digest,
                        "conventional_eeglab_seconds_min": float(seconds.min()),
                        "conventional_eeglab_seconds_max": float(seconds.max()),
                        "recording_seconds": header["acquisition"]["xmax"],
                        "conventional_seconds_within_recording": bool(((seconds >= 0) & (seconds <= header["acquisition"]["xmax"])).all()),
                        "clock_conclusion": "TSV onset is rounded raw EEGLAB latency, not BIDS seconds in this probe",
                        "duration_units": "Unqualified despite exact raw-value export match",
                        "trial_endpoints_qualified": False})
    result = {"date": "2026-10-12", "snapshot_commit": headers["snapshot_commit"],
              "bindings": {n: sha(packet / n) for n in ("set_metadata.json", "set_timing.json", "timing_digests.json", "metadata_annotations.json")},
              "script_sha256": sha(Path(__file__)), "records": records,
              "all_three_roundtrip_matches": True,
              "scope": "Three predeclared metadata probes; do not generalize export conversion to all40 recordings without replay",
              "numerical_waveform_values_decoded": 0, "individual_rating_values_decoded": 0,
              "event_labels_or_responses_decoded": 0, "new_fits": 0, "new_predictions": 0,
              "confirmation_opened": False, "research_question_changed": False}
    (packet / "timing_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"probes": len(records), "raw_sample_onset_export_confirmed": True, "trial_endpoints": "hold"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.packet, args.work_dir)
