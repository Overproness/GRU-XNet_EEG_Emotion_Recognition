"""Describe frozen source coverage timing metadata, without outcomes or EEG."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_temporal_coverage import (
    CATEGORIES, DEVELOPMENT_CLIPS, REGIONS, segment_bounds, union_sample_count,
)

PLAN = REPO / "results/development/faced_temporal_coverage_preflight_2026-10-12/plan.json"
OUTPUT = REPO / "results/development/faced_coverage_selection_2026-10-13/source_geometry_diagnostics.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    values = sorted(float(v) for v in values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite or empty source geometry summary")
    def percentile(q):
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (position - low) * (values[high] - values[low])
    return {"count": len(values), "minimum": values[0], "p05": percentile(.05),
            "p25": percentile(.25), "median": statistics.median(values),
            "mean": statistics.mean(values), "p75": percentile(.75),
            "p95": percentile(.95), "maximum": values[-1]}


def summarize(records):
    fields = ("marker_span_seconds", "unique_nominal_core_seconds",
              "repeated_nominal_core_seconds", "unique_core_fraction_of_marker_span",
              "unique_core_gain_over_late_seconds")
    return {"trials": len(records),
            "trials_with_any_core_overlap": sum(r["any_positive_core_overlap"] for r in records),
            "trials_with_all_cores_disjoint": sum(not r["any_positive_core_overlap"] for r in records),
            "trials_with_triple_core_overlap": sum(r["triple_core_overlap_seconds"] > 0 for r in records),
            "descriptive_statistics": {field: describe(r[field] for r in records) for field in fields},
            "overlap_seconds_by_region_pair": {
                label: describe(r["pairwise_core_overlap_seconds"][label] for r in records)
                for label in ("early_middle", "early_late", "middle_late")}}


def build():
    before = sha256(PLAN)
    plan = json.loads(PLAN.read_bytes())
    originals = plan["source_records"]
    people = frozenset(plan["source_people"])
    denied = frozenset(plan["validation_people_denied"]) | frozenset(plan["protected_people_denied"])
    pairs = [(r["subject"], r["vid"]) for r in originals]
    if (len(people) != 70 or len(denied) != 53 or people & denied
            or len(pairs) != 980 or len(set(pairs)) != 980
            or set(pairs) != {(p, v) for p in people for v in DEVELOPMENT_CLIPS}):
        raise PermissionError("Source-only 70-by-14 geometry roster changed")
    results = []
    for row in originals:
        if row["subject"] not in people or row["role"] != "source":
            raise PermissionError("Nonsource identity rejected before timing interpretation")
        rate = row["native_rate_hz"]
        if isinstance(rate, bool) or not isinstance(rate, int) or rate not in (250, 1000):
            raise ValueError("Source native rate changed")
        spans = segment_bounds(row["marker_start_seconds"], row["marker_end_seconds"],
                               rate, row["recording_samples"])
        if spans != row["segments"] or not row["late_range_exactly_matches_prior"]:
            raise ValueError("Source regions differ from exact frozen native ranges")
        cores = {region: [spans[region][0] + 2 * rate, spans[region][1] - 2 * rate]
                 for region in REGIONS}
        if any(end - begin != 28 * rate for begin, end in cores.values()):
            raise ValueError("Nominal retained core geometry changed")
        span_seconds = row["marker_end_seconds"] - row["marker_start_seconds"]
        union_seconds = union_sample_count(cores.values()) / rate
        overlaps = {}
        for first, second in (("early", "middle"), ("early", "late"), ("middle", "late")):
            overlaps[first + "_" + second] = max(0, min(cores[first][1], cores[second][1])
                                                - max(cores[first][0], cores[second][0])) / rate
        triple = max(0, min(v[1] for v in cores.values()) - max(v[0] for v in cores.values())) / rate
        if not 28 <= union_seconds <= 84 or not 0 < union_seconds <= span_seconds:
            raise ValueError("Source nominal core union falls outside marker span")
        results.append({"subject": row["subject"], "vid": row["vid"], "role": "source",
                        "assigned_category": row["assigned_category"], "family": row["family"],
                        "native_rate_hz": rate, "marker_span_seconds": span_seconds,
                        "nominal_core_native_intervals": cores,
                        "sum_nominal_core_seconds": 84,
                        "unique_nominal_core_seconds": union_seconds,
                        "repeated_nominal_core_seconds": 84 - union_seconds,
                        "unique_core_gain_over_late_seconds": union_seconds - 28,
                        "unique_core_fraction_of_marker_span": union_seconds / span_seconds,
                        "pairwise_core_overlap_seconds": overlaps,
                        "triple_core_overlap_seconds": triple,
                        "any_positive_core_overlap": any(v > 0 for v in overlaps.values())})
    if sha256(PLAN) != before:
        raise PermissionError("Frozen geometry input changed during metadata audit")
    if set(r["assigned_category"] for r in results) != set(CATEGORIES):
        raise PermissionError("Source category support changed")
    grouped = {"category": defaultdict(list), "title": defaultdict(list), "participant": defaultdict(list)}
    for row in results:
        grouped["category"][row["assigned_category"]].append(row)
        grouped["title"][str(row["vid"])].append(row)
        grouped["participant"][row["subject"]].append(row)
    return {"date": "2026-10-13", "status": "completed_source_metadata_geometry_audit",
            "scope": "Descriptive reporting only; no selection or model changes",
            "input_bindings": {PLAN.relative_to(REPO).as_posix(): before},
            "code_bindings": {Path(__file__).resolve().relative_to(REPO).as_posix(): sha256(Path(__file__)),
                              "gruxnet/faced_temporal_coverage.py": sha256(REPO / "gruxnet/faced_temporal_coverage.py")},
            "source_trials": 980, "source_regions": 2940,
            "nominal_region_seconds": 32, "nominal_retained_core_seconds": 28,
            "sum_nominal_core_seconds_per_trial": 84,
            "interpretation": "Three processed 28-second cores contribute 84 nominal seconds before overlap accounting. Their physical native intervals can overlap. Core bounds describe nominal retained intervals; filtering and resampling can depend on surrounding samples. These regions are neither independent observations nor exhaustive full-video coverage.",
            "summary": summarize(results),
            "by_category": {k: summarize(v) for k, v in sorted(grouped["category"].items())},
            "by_title": {k: summarize(v) for k, v in sorted(grouped["title"].items(), key=lambda item: int(item[0]))},
            "by_participant": {k: summarize(v) for k, v in sorted(grouped["participant"].items())},
            "native_rate_trial_counts": dict(sorted(Counter(r["native_rate_hz"] for r in results).items())),
            "trials": results, "original_recordings_opened": 0,
            "private_arrays_opened": 0, "feature_values_opened": 0,
            "target_values_opened": 0, "model_fits": 0,
            "validation_access": False, "protected_access": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out.relative_to(REPO)), "summary": result["summary"],
                      "source_trials": 980, "outcomes_or_arrays_opened": 0}))


if __name__ == "__main__":
    main()
