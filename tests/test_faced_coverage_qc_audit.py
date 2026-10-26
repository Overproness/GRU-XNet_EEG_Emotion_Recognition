"""Manufactured QC boundary, repair-scope and physical-offset audit fixtures."""
from copy import deepcopy

import pytest

from scripts.audit_faced_coverage_source_qc_2026_10_13 import (
    CHANNELS, artifact_decision, availability, native_ranges, qualify_public_rows,
)


def quality():
    return {"channels": [{"channel": c, "raw_rail_fraction": 0.,
                           "raw_identical_adjacent_fraction": .2, "processed_sd_uv": 5.,
                           "processed_peak_to_peak_uv": 25., "processed_49_51_to_1_45_power_ratio": .01,
                           "review_flags": []} for c in CHANNELS],
            "valid_windows": 7, "windows_dropped": 0, "finite_processed_values": True,
            "flagged_channels": []}


def test_exact_quality_limits_remain_available():
    q = quality()
    q["channels"][0].update(raw_rail_fraction=.0001, raw_identical_adjacent_fraction=.95,
                             processed_sd_uv=.1, processed_peak_to_peak_uv=1000.,
                             processed_49_51_to_1_45_power_ratio=.1)
    q["channels"][1]["processed_sd_uv"] = 100.
    assert availability(q, False) == (True, [])


@pytest.mark.parametrize("metric,value,reason", [
    ("raw_rail_fraction", .000100001, "adc_rail"),
    ("raw_identical_adjacent_fraction", .95000001, "stationary_counts"),
    ("processed_sd_uv", .09999, "low_processed_variation"),
    ("processed_sd_uv", 100.00001, "high_processed_variation"),
    ("processed_peak_to_peak_uv", 1000.00001, "high_processed_excursion"),
    ("processed_49_51_to_1_45_power_ratio", .10000001, "residual_50hz_power"),
])
def test_each_acquisition_flag_is_reconstructed_without_saved_decision(metric, value, reason):
    q = quality(); q["channels"][0][metric] = value
    if reason != "residual_50hz_power":
        q["channels"][0]["review_flags"] = [reason]; q["flagged_channels"] = [CHANNELS[0]]
    assert availability(q, False) == (False, [{"channel": CHANNELS[0], "reason": reason}])


def test_saved_text_cannot_conceal_quality_failure():
    q = quality(); q["channels"][0]["raw_rail_fraction"] = .2
    with pytest.raises(ValueError, match="flags"):
        availability(q, False)
    q = quality(); q["channels"][0]["processed_sd_uv"] = float("nan")
    with pytest.raises(ValueError, match="number"):
        availability(q, False)


def artifact(bad):
    capacity = len(bad) > 3
    decision = "unchanged" if not bad else ("held_beyond_repair_capacity" if capacity
                                             else "fixed_spherical_interpolation_before_CAR")
    return {"pre_car_sd_uv": [101. if i in bad else 10. for i in range(30)],
            "pre_car_peak_to_peak_uv": [25.] * 30, "detected_indices": bad,
            "detected_channels": [CHANNELS[i] for i in bad], "repair_capacity_exceeded": capacity,
            "decision": decision, "repaired_channels": [] if capacity else [CHANNELS[i] for i in bad],
            "donors_unchanged": True, "trials_or_windows_dropped": 0,
            "interpolation_weight_max_discrepancy": 0.}


@pytest.mark.parametrize("bad", [[], [0], [0, 1, 2], [0, 1, 2, 3]])
def test_repair_capacity_is_computed_from_pre_car_metrics(bad):
    a = artifact(bad)
    decision, capacity, count = artifact_decision(a)
    assert capacity is (len(bad) > 3)
    assert count == (0 if len(bad) > 3 else len(bad))
    assert decision == a["decision"]


def test_capacity_text_cannot_promote_four_bad_channels_to_repairable():
    a = artifact([0, 1, 2, 3]); a["repair_capacity_exceeded"] = False
    with pytest.raises(ValueError, match="repair"):
        artifact_decision(a)
    assert availability(quality(), True) == (False, [{"channel": "all", "reason": "repair_capacity_exceeded"}])


def test_native_offsets_include_only_requested_partial_records():
    header = {"columns": {"samples_per_record": ["4"] * 32},
              "sample_width": 3, "header_bytes": 128, "record_count": 4}
    ranges = native_ranges(header, [3, 9], list(range(30)))
    assert len(ranges) == 90
    assert ranges[0] == {"channel_index": 0, "start_sample": 3, "stop_sample": 4, "offset": 137, "bytes": 3}
    assert ranges[30] == {"channel_index": 0, "start_sample": 4, "stop_sample": 8, "offset": 512, "bytes": 12}
    assert ranges[60] == {"channel_index": 0, "start_sample": 8, "stop_sample": 9, "offset": 896, "bytes": 3}
    assert sum(r["bytes"] for r in ranges) == 30 * 6 * 3
    assert ranges[29]["offset"] == 137 + 29 * 12


def public_fixture():
    people = [f"source-{i:03d}" for i in range(70)]
    records = [{"subject": p, "vid": v, "native_rate_hz": 250,
                "segments": {r: [10000, 18000] for r in ("early", "middle", "late")}}
               for p in people for v in range(14)]
    plan = {"source_people": people, "source_records": records,
            "validation_people_denied": ["validation"], "protected_people_denied": ["protected"]}
    rows = [{"subject": r["subject"], "vid": r["vid"], "region": region, "role": "source",
             "eeg_available": True, "reasons": [], "sample_interval": [10000, 18000],
             "native_rate_hz": 250, "samples_decoded": 240000, "extra_boundary_samples_decoded": 0,
             "late_exact_replay_passed": region == "late"}
            for r in records for region in ("early", "middle", "late")]
    return rows, plan


@pytest.mark.parametrize("drift", ["protected", "duplicate", "missing", "wrong_range", "wrong_role"])
def test_public_identity_gate_rejects_drift_before_any_private_json(drift):
    rows, plan = public_fixture()
    if drift == "protected": rows[0]["subject"] = "protected"
    elif drift == "duplicate": rows[-1] = deepcopy(rows[0])
    elif drift == "missing": rows.pop()
    elif drift == "wrong_range": rows[0]["sample_interval"][0] += 1
    else: rows[0]["role"] = "validation"
    with pytest.raises(PermissionError):
        qualify_public_rows(rows, plan)


def test_complete_manufactured_source_rectangle_qualifies():
    rows, plan = public_fixture()
    assert len(qualify_public_rows(rows, plan)) == 980
