"""Freeze source coverage CV from public metadata only; no arrays or outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_coverage_selection import source_plan

PREVIOUS = REPO / "results/development/faced_temporal_coverage_preflight_2026-10-12/plan.json"
PROGRAMME = REPO / "results/development/faced_development_protocol_2026-10-11/plan.json"
PUBLIC = REPO / "results/development/faced_coverage_selection_preflight_2026-10-13"


def build():
    previous = json.loads(PREVIOUS.read_bytes())
    programme = json.loads(PROGRAMME.read_bytes())
    records = previous["source_records"]
    pairs = [(r["subject"], r["vid"]) for r in records]
    metadata = {pair: row for pair, row in zip(pairs, records)}
    plan = source_plan(pairs, metadata, programme["source_participant_oof_folds"])
    bindings = [PREVIOUS, PROGRAMME, REPO / "gruxnet/faced_coverage_selection.py",
                REPO / "tests/test_faced_coverage_selection.py", Path(__file__).resolve(),
                REPO / "gruxnet/faced_classical_contract.py", REPO / "gruxnet/faced_classical_runner.py",
                REPO / "gruxnet/faced_temporal_coverage.py"]
    plan["bindings"] = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in bindings}
    plan["private_arrays_opened"] = 0
    plan["original_files_opened"] = 0
    plan["target_values_opened"] = 0
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the metadata-only plan (the only operation).")
    parser.add_argument("--out", type=Path, default=PUBLIC)
    args = parser.parse_args()
    plan = build()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "plan.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"metadata_plan": str(path.relative_to(REPO)), "cells": 10,
                      "cv_fit_cap": 90, "optional_full_source_fits": 3,
                      "outcomes_opened": 0, "arrays_opened": 0}))


if __name__ == "__main__":
    main()
