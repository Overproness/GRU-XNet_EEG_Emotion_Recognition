"""Synthetic guards and marker-boundary checks; no original FACED data reads."""
import math
import pytest

from gruxnet.faced_temporal_coverage import (
    CATEGORIES, SourceOnlySegmentGuard, choose_source_pilot, segment_bounds, union_sample_count,
)


@pytest.mark.parametrize("rate", [250, 1000])
def test_ranges_contained_and_late_exact(rate):
    start, end = 1.1231, 65.4567
    ranges = segment_bounds(start, end, rate, 100 * rate)
    first, last = math.ceil(start * rate), math.floor(end * rate)
    assert ranges["early"] == [first, first + 32 * rate]
    assert ranges["late"] == [last - 32 * rate, last]
    assert ranges["middle"][0] == (first + last - 32 * rate) // 2
    assert all(first <= a < b <= last and b-a == 32*rate for a, b in ranges.values())


def test_short_video_has_overlapping_not_independent_observations():
    ranges = segment_bounds(0., 34., 250, 10000)
    assert union_sample_count(ranges.values()) == 34*250
    assert sum(b-a for a,b in ranges.values()) == 96*250


@pytest.mark.parametrize("args", [
    (0., 31.999, 250, 10000), (-1., 50., 250, 20000), (0., 50., 250, 10000),
    (float("nan"), 50., 250, 20000), (0., float("inf"), 250, 20000),
    (False, 50., 250, 20000), (0., 50., 128, 20000), (0., 50., 250., 20000),
    (0., 50., 250, True),
])
def test_boundary_or_geometry_failures(args):
    with pytest.raises(ValueError):
        segment_bounds(*args)


def guard():
    return SourceOnlySegmentGuard(source_people=["source"], validation_people=["validation"],
        protected_people=["protected"], source_pairs=[("source", 2), ("source", 3)],
        pilot_rows=[{"subject": "source", "vid": 2, "role": "source", "region": "late",
                     "native_rate_hz": 250, "sample_interval": [0,8000]}])


@pytest.mark.parametrize("change", [
    {"subject": "protected"}, {"subject": "validation"}, {"subject": "unknown"},
    {"vid": 1}, {"vid": 3}, {"region": "early"}, {"sample_interval": [0,8001]},
    {"native_rate_hz": 1000}, {"vid": True}, {"sample_interval": [False,8000]},
])
def test_reject_before_signal_callback(change):
    calls = []
    request = dict(subject="source", vid=2, region="late", sample_interval=[0,8000], native_rate_hz=250)
    request.update(change)
    with pytest.raises((PermissionError, ValueError)):
        guard().read(**request, reader=lambda *values: calls.append(values))
    assert calls == []


def test_exact_pilot_callback_and_immutable_range():
    current = guard(); calls = []
    assert current.read(subject="source", vid=2, region="late", sample_interval=[0,8000], native_rate_hz=250,
                        reader=lambda *v: calls.append(v) or "synthetic") == "synthetic"
    assert len(calls) == 1
    with pytest.raises(TypeError):
        current.pilot_ranges[("source",2,"late")] = (250,0,16000)


def test_role_overlap_and_undeclared_source_prohibited():
    with pytest.raises(PermissionError):
        SourceOnlySegmentGuard(source_people=["p"], validation_people=["p"], protected_people=[],
                               source_pairs=[("p",2)], pilot_rows=[])
    with pytest.raises(PermissionError):
        SourceOnlySegmentGuard(source_people=["p"], validation_people=[], protected_people=[],
                               source_pairs=[("p",7)], pilot_rows=[])


def test_pilot_metadata_only_deterministic_category_and_group_coverage():
    rows = [{"subject": f"synthetic-{j}", "vid": i+2, "assigned_category": category,
             "acquisition_group": group}
            for i,category in enumerate(CATEGORIES)
            for j,group in enumerate(("legacy_1000", "legacy_250", "modern_1000"))]
    chosen = choose_source_pilot(rows)
    assert chosen == choose_source_pilot(list(reversed(rows)))
    assert {r["assigned_category"] for r in chosen} == set(CATEGORIES)
    assert {r["acquisition_group"] for r in chosen} == {"legacy_1000", "legacy_250", "modern_1000"}


def test_pilot_missing_category_fails():
    with pytest.raises(ValueError):
        choose_source_pilot([])
