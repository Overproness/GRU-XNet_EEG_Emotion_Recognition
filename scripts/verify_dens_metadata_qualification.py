"""Offline integrity and independent count/digest checks for DENS qualification."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "results/development/dens_metadata_qualification_2026-10-12"
COMMIT = "5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((PACKET / name).read_text(encoding="utf-8"))


def check():
    inv, ann, sets = load("inventory.json"), load("metadata_annotations.json"), load("set_metadata.json")
    tsv, comparison, catalogue = load("timing_digests.json"), load("timing_comparison.json"), load("material_catalogue.json")
    if inv["commit"] != COMMIT or ann["commit"] != COMMIT or sets["snapshot_commit"] != COMMIT:
        raise ValueError("Release changed")
    paths = [r["path"] for r in inv["tree_entries"] if r["type"] == "blob"]
    eeg = {p.split("/")[0] for p in paths if p.endswith(".set")}
    behavior = {p.split("/")[0] for p in paths if p.endswith("_beh.tsv")}
    events = {p.split("/")[0] for p in paths if p.endswith("_events.tsv")}
    if (len(eeg), len(behavior), len(events), len(eeg & behavior & events)) != (40, 34, 40, 34):
        raise ValueError("Participant inventory changed")
    if {r["participant"] for r in ann["materials"]} != behavior:
        raise ValueError("Behaviour coverage differs from pinned inventory")
    key_set = set()
    rows = 0
    for rec in ann["materials"]:
        keys = rec["ordered_material_keys"]
        if len(keys) != rec["data_row_count"] or len(keys) != len(set(keys)):
            raise ValueError("Material row/key integrity changed")
        key_set.update(keys)
        rows += len(keys)
    if rows != 367 or len(key_set) != 20:
        raise ValueError("Material counts changed")
    set_index = {r["subject"]: r for r in sets["records"]}
    tsv_index = {r["participant"]: r for r in tsv["records"]}
    if set(set_index) != {"sub-mit003", "sub-mit081", "sub-mit108"}:
        raise ValueError("Undeclared SET probe")
    for rec in comparison["records"]:
        subject = rec["participant"]
        if rec["set_latency_rounded_6_decimal_digest"] != tsv_index[subject]["onset_digest_little_endian_float64"]:
            raise ValueError("Onset digest mismatch")
        if rec["set_duration_digest"] != tsv_index[subject]["duration_digest_little_endian_float64"]:
            raise ValueError("Duration digest mismatch")
        if rec["event_count"] != set_index[subject]["metadata"]["schema"]["event"]["entry_count"]:
            raise ValueError("Event geometry mismatch")
        if rec["trial_endpoints_qualified"] is not False:
            raise ValueError("Unqualified trial endpoints released")
    for name, expected in comparison["bindings"].items():
        if sha(PACKET / name) != expected:
            raise ValueError("Timing binding changed")
    if len(catalogue["films"]) != 69 or ann["summary"]["usable_trial_count"] is not None:
        raise ValueError("Material scope or trial-status changed")
    working = REPO.parent / "report.tex"
    archived = REPO / "docs/paper_archive/2026-10-12-before-current-progress-draft/report.tex"
    expected_source = "06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0"
    if sha(archived) != expected_source or (working.exists() and sha(working) != expected_source):
        raise ValueError("Original manuscript changed")
    return {"passed": True, "possible_participant_joins": 34, "material_rows": rows,
            "filename_keys": len(key_set), "timing_probes": 3, "movies_in_catalogue": 69,
            "usable_trials": None, "supervised_experiments": "hold",
            "ratings_or_waveform_values_read_by_verifier": 0, "new_fits": 0,
            "confirmation_opened": False, "original_question_adopted_or_changed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true", help="Bind only completed public qualification artifacts")
    args = parser.parse_args()
    result = check()
    manifest = PACKET / "completion/verification.json"
    if args.seal:
        if manifest.exists():
            raise FileExistsError("Preserve the existing completion receipt")
        paths = list(PACKET.glob("*.json")) + [PACKET / "README.md"]
        paths += list((REPO / "docs/publication").glob("DENS*2026-10-12.md"))
        paths += list((REPO / "scripts").glob("*dens*.py"))
        paths += [REPO / "tests/test_dens_selective_metadata.py"]
        result["bindings"] = {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in paths}
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        saved = json.loads(manifest.read_text())
        for path, expected in saved["bindings"].items():
            if sha(REPO / path) != expected:
                raise ValueError("DENS completion binding changed: " + path)
        for key, value in result.items():
            if saved[key] != value:
                raise ValueError("DENS conclusion changed")
        result["bindings_checked"] = len(saved["bindings"])
    print(json.dumps({k: v for k, v in result.items() if k != "bindings"}))


if __name__ == "__main__":
    main()
