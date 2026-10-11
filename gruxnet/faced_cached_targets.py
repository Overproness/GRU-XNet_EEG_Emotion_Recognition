"""Distinguish original cached ratings from declared shuffled capacity targets.

No I/O or fitting. Guard every identity before interpreting numeric outcomes.
"""
from __future__ import annotations

import math

from gruxnet.faced_eegnet_bridge import guard_cached_identity

PERMUTED_MODELS = ("capacity_spectral_permuted", "capacity_covariance_permuted")
REAL_CAPACITY_MODELS = ("capacity_spectral_real", "capacity_covariance_real")


def cached_original_target(row, metadata, programme, origin):
    if origin not in ("quality", "representation"):
        raise PermissionError("Unknown bound cache origin")
    pair = guard_cached_identity(row, metadata)
    rotation = int(row["rotation"])
    if rotation not in (0, 1, 2):
        raise PermissionError("Changed cached rotation")
    if row.get("acquisition_group") != metadata[pair]["acquisition_group"]:
        raise PermissionError("Cached acquisition changed")
    model, arm = row["model"], row["arm"]
    if model in PERMUTED_MODELS + REAL_CAPACITY_MODELS or arm == "capacity_source_fit":
        if origin != "representation" or model not in PERMUTED_MODELS + REAL_CAPACITY_MODELS or arm != "capacity_source_fit":
            raise PermissionError("Unknown or misassigned capacity diagnostic")
        source = {(r["subject"], r["vid"]) for r in programme["rotation_manifests"][rotation]["arms"]["source_fit"]}
        if pair not in source:
            raise PermissionError("Capacity row leaves original source set")
        if model in PERMUTED_MODELS:
            # These are deliberately reassigned experimental targets, not ratings.
            # Do not convert either target or prediction on excluded rows.
            return pair, None
    target = float(row["target"])
    if not math.isfinite(target) or not 0 <= target <= 7:
        raise ValueError("Cached original rating nonfinite/outside scale")
    return pair, target
