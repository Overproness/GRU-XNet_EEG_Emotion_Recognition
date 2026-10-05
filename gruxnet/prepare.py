"""Stateless preprocessing and disk-backed windows with explicit lineage."""
from __future__ import annotations

import json
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from .data import COMMON_CHANNELS, POLICY, aligned_signal, audit, digest, roots, seed_channels, sha256, source_trials, write_json

PREPROCESS_VERSION = 1


def preprocess_signal(signal, fs, source_channels, target_channels):
    # Filtering/resampling within a trial cannot mix held-out subjects or trials.
    aligned, mask = aligned_signal(signal, source_channels, target_channels)
    filtered = sosfiltfilt(butter(4, (4, 40), fs=fs, btype="bandpass", output="sos"), aligned, axis=-1)
    if fs != 128:
        common = gcd(fs, 128)
        filtered = resample_poly(filtered, 128 // common, fs // common, axis=-1)
    return np.asarray(filtered, dtype=np.float32), mask


def prepare(root: Path, output: Path, montage="common14", seconds=4, provenance: Path | None = None) -> dict:
    if seconds < 2 or int(seconds) != seconds:
        raise ValueError("Use an integer window duration of at least 2 seconds")
    if (output / "manifest.csv").exists():
        raise FileExistsError(f"Prepared cache already exists: {output}. Use a fresh directory to change preprocessing.")
    if montage not in ("common14", "canonical62"):
        raise ValueError(montage)
    report = audit(root, output / "audit", provenance=provenance)
    ratings = json.loads((output / "audit/gameemo_sam_ratings.json").read_text())
    channels = COMMON_CHANNELS if montage == "common14" else seed_channels(roots(root)["SEEDIV"])
    config = {"preprocess_version": PREPROCESS_VERSION, "sampling_rate": 128, "window_seconds": seconds,
              "stride_seconds": seconds, "montage": montage, "channels": channels, "label_policy": POLICY,
              "filter": "4th-order Butterworth 4..40 Hz, zero-phase per source trial, before resampling",
              "resampling": "scipy resample_poly, 200->128 (16/25), no time interpolation",
              "baseline": "DEAP first 384 samples excluded; no baseline feature subtraction",
              "normalization": "per-window per-observed-channel z-score at model input; no fitted population statistics",
              "stft": {"n_fft": 128, "hop_length": 32, "window": "periodic Hann", "center": False,
                       "frequencies_hz": [4, 40], "representation": "log1p magnitude / window sum"}}
    rows, lineage, excluded = [], [], []
    window_size = int(seconds * 128)
    (output / "windows").mkdir(parents=True, exist_ok=True)
    for meta, signal, source_channels, fs in source_trials(root, ratings):
        if meta["label"] is None:
            excluded.append({k: v for k, v in meta.items() if k != "label"})
            continue
        filtered, mask = preprocess_signal(signal, fs, source_channels, channels)
        starts = list(range(0, filtered.shape[1] - window_size + 1, window_size))
        if not starts:
            raise ValueError(f"Trial shorter than requested window: {meta['trial_id']}")
        windows = np.stack([filtered[:, start:start + window_size] for start in starts])
        filename = digest(meta["trial_id"])[:24] + ".npy"
        cache_file = output / "windows" / filename
        np.save(cache_file, windows, allow_pickle=False)
        mask_string = "".join("1" if m else "0" for m in mask)
        for index, start in enumerate(starts):
            rows.append({"sample_id": f"{meta['trial_id']}:W{start:06d}", "dataset": meta["dataset"],
                         "subject_id": meta["subject_id"], "session": meta["session"], "trial_id": meta["trial_id"],
                         "label": meta["label"], "original_label": meta["original_label"],
                         "start_sample": start, "stop_sample": start + window_size, "sampling_rate": 128,
                         "cache_file": f"windows/{filename}", "cache_index": index, "channel_mask": mask_string})
        lineage.append({**meta, "original_sampling_rate": fs, "source_samples_after_baseline": signal.shape[1],
                        "resampled_samples": filtered.shape[1], "windows": len(starts), "source_channels": source_channels,
                        "discarded_tail_samples": filtered.shape[1] - starts[-1] - window_size,
                        "cache_sha256": sha256(cache_file), "cache_file": f"windows/{filename}"})
        if len(lineage) % 100 == 0:
            print(f"Prepared {len(lineage)} trials / {len(rows)} windows ({meta['dataset']})", flush=True)
    frame = pd.DataFrame(rows)
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate samples in manifest")
    frame.to_csv(output / "manifest.csv", index=False)
    write_json(output / "lineage.json", lineage)
    write_json(output / "excluded_trials.json", excluded)
    summary = {"config": config, "source_audit": report, "eligible_trials": len(lineage), "excluded_trials": len(excluded),
               "windows": len(rows), "dataset_window_counts": dict(Counter(frame.dataset)),
               "manifest_sha256": sha256(output / "manifest.csv"), "lineage_sha256": sha256(output / "lineage.json")}
    summary["cache_fingerprint"] = digest(summary)
    write_json(output / "prepared.json", summary)
    print(f"Preparation complete: {len(lineage)} trials, {len(rows)} windows, {summary['dataset_window_counts']}", flush=True)
    return summary


def load_prepared(cache: Path, verify_files=False):
    info = json.loads((cache / "prepared.json").read_text())
    if sha256(cache / "manifest.csv") != info["manifest_sha256"] or sha256(cache / "lineage.json") != info["lineage_sha256"]:
        raise ValueError("Prepared metadata changed since creation")
    check = dict(info)
    fingerprint = check.pop("cache_fingerprint")
    if digest(check) != fingerprint:
        raise ValueError("Prepared configuration fingerprint mismatch")
    if verify_files:
        for trial in json.loads((cache / "lineage.json").read_text()):
            if sha256(cache / trial["cache_file"]) != trial["cache_sha256"]:
                raise ValueError(f"Cached EEG changed: {trial['trial_id']}")
    return pd.read_csv(cache / "manifest.csv", dtype={"channel_mask": str, "session": str}), info
