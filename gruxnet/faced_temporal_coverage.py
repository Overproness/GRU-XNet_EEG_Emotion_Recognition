"""Pure source-only temporal-coverage planning; no recording or target reader.

The three regions retain the qualified 32-second input/seven-window processor.
They are distributed observations, not independent or exhaustive video samples.
"""
from __future__ import annotations

import hashlib
import math
from types import MappingProxyType

REGIONS = ("early", "middle", "late")
CATEGORIES = ("Amusement", "Anger", "Disgust", "Fear", "Inspiration", "Joy", "Sadness", "Tenderness")
DEVELOPMENT_CLIPS = frozenset((2, 3, 4, 6, 9, 11, 12, 17, 18, 20, 23, 25, 27, 28))
PILOT_SALT = "FACED_SOURCE_TEMPORAL_COVERAGE_PILOT_2026-10-12/v1"


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Require exact integer " + label)
    return value


def segment_bounds(start_seconds, end_seconds, rate, recording_samples):
    """Exact native ranges, each contained in the marker-defined video span.

    Early uses ceil(marker start), late uses floor(marker end), and middle is
    centred in the resulting integer native interval, rounded toward its start.
    """
    _integer(rate, "rate"); _integer(recording_samples, "recording sample count")
    if rate not in (250, 1000) or recording_samples <= 0:
        raise ValueError("Outside qualified recording geometry")
    for value in (start_seconds, end_seconds):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Require finite marker times")
    first = math.ceil(start_seconds * rate)
    last = math.floor(end_seconds * rate)
    size = 32 * rate
    if first < 0 or last > recording_samples or last - first < size:
        raise ValueError("Video span cannot contain the fixed processor")
    starts = (first, (first + last - size) // 2, last - size)
    return {region: [begin, begin + size] for region, begin in zip(REGIONS, starts)}


def union_sample_count(ranges):
    """Count physical coverage once, including overlapping source regions."""
    ordered = sorted((int(r[0]), int(r[1])) for r in ranges)
    count = 0; stop = -1
    for begin, end in ordered:
        if begin < 0 or end <= begin:
            raise ValueError("Invalid sample range")
        count += max(0, end - max(begin, stop))
        stop = max(stop, end)
    return count


def _rank(row):
    identity = f"{PILOT_SALT}/{row['subject']}/{row['vid']}".encode()
    return hashlib.sha256(identity).hexdigest()


def choose_source_pilot(records):
    """One trial per category, covering every observed acquisition group.

    Allocation uses only declared category/group/identity metadata. It never
    considers signal quality, target values or model performance.
    """
    grouped = {category: sorted((r for r in records if r["assigned_category"] == category), key=_rank)
               for category in CATEGORIES}
    if any(not rows for rows in grouped.values()):
        raise ValueError("Pilot lacks a declared category")
    groups = sorted({r["acquisition_group"] for r in records})
    if len(groups) != 3:
        raise ValueError("Source acquisition-group coverage changed")
    # Cover the groups using the first deterministic category assignment that
    # permits it; rank trial identities only after that assignment is fixed.
    def allocate(position, chosen):
        if position == len(groups):
            return chosen
        group = groups[position]
        for category in CATEGORIES:
            if category in chosen:
                continue
            candidates = [r for r in grouped[category] if r["acquisition_group"] == group]
            if candidates:
                result = allocate(position + 1, {**chosen, category: candidates[0]})
                if result is not None:
                    return result
        return None
    chosen = allocate(0, {})
    if chosen is None:
        raise ValueError("Cannot cover acquisition groups with distinct categories")
    selected = [chosen.get(category, grouped[category][0]) for category in CATEGORIES]
    if len({(r["subject"], r["vid"]) for r in selected}) != 8:
        raise ValueError("Repeated source pilot trial")
    return selected


class SourceOnlySegmentGuard:
    """Authenticate role and frozen pilot range before invoking any reader.

    This guard does not activate access. A future execution backend must also
    authenticate its declaration, original-file hashes, channel/calibration
    bindings and quality policy before supplying a reader here.
    """
    def __init__(self, *, source_people, validation_people, protected_people, source_pairs, pilot_rows):
        self.source_people = frozenset(source_people)
        self.validation_people = frozenset(validation_people)
        self.protected_people = frozenset(protected_people)
        if (self.source_people & self.validation_people or self.source_people & self.protected_people
                or self.validation_people & self.protected_people):
            raise PermissionError("Participant roles overlap")
        pairs = tuple(source_pairs)
        self.source_pairs = frozenset(pairs)
        if len(pairs) != len(self.source_pairs) or not pairs:
            raise PermissionError("Duplicate or empty source identities")
        if any(p not in self.source_people or v not in DEVELOPMENT_CLIPS for p, v in pairs):
            raise PermissionError("Source pair lies outside the fixed development role")
        rows = tuple(pilot_rows)
        frozen = {}
        for row in rows:
            pair = (row["subject"], row["vid"])
            if pair not in self.source_pairs or row["region"] not in REGIONS or row["role"] != "source":
                raise PermissionError("Pilot includes an undeclared source segment")
            rate = _integer(row["native_rate_hz"], "pilot native rate")
            begin, end = row["sample_interval"]
            _integer(begin, "pilot beginning"); _integer(end, "pilot endpoint")
            if rate not in (250, 1000) or begin < 0 or end - begin != 32 * rate:
                raise ValueError("Pilot segment geometry changed")
            key = (*pair, row["region"])
            if key in frozen:
                raise PermissionError("Duplicate pilot segment")
            # Immutable tuples prevent a caller from widening scope in place.
            frozen[key] = (rate, begin, end)
        self.pilot_ranges = MappingProxyType(frozen)

    def read(self, *, subject, vid, region, sample_interval, native_rate_hz, reader):
        # Role rejection intentionally precedes all signal callbacks.
        if subject in self.protected_people or subject in self.validation_people or subject not in self.source_people:
            raise PermissionError("Only frozen source participants are permitted")
        _integer(vid, "video identity")
        key = (subject, vid, region)
        if (subject, vid) not in self.source_pairs or key not in self.pilot_ranges:
            raise PermissionError("Segment is outside the frozen eight-trial pilot")
        _integer(native_rate_hz, "native rate")
        if len(sample_interval) != 2:
            raise ValueError("Invalid sample interval")
        begin, end = sample_interval
        _integer(begin, "sample beginning"); _integer(end, "sample endpoint")
        if (native_rate_hz, begin, end) != self.pilot_ranges[key]:
            raise PermissionError("Requested segment differs from the frozen pilot")
        return reader(subject, vid, region, [begin, end], native_rate_hz)
