"""Split ORIGINAL subjects, retaining every trial/window in its subject partition."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_split(frame: pd.DataFrame) -> dict:
    if frame.sample_id.duplicated().any() or frame.split.isna().any():
        raise ValueError("Duplicate/unassigned samples")
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("All three non-empty partitions are required")
    for key in ["subject_id", "trial_id"]:
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Leakage: {key} occurs in multiple splits")
    if any(not s.startswith(f"{d}:") for s, d in zip(frame.subject_id, frame.dataset)):
        raise ValueError("Subject IDs must be dataset-qualified")
    summary = {}
    for split, rows in frame.groupby("split"):
        summary[split] = {}
        for dataset, group in rows.groupby("dataset"):
            if set(group.label) != {0, 1}:
                raise ValueError(f"Both valence classes required: {split}/{dataset}; choose another predeclared fold")
            summary[split][dataset] = {"subjects": sorted(group.subject_id.unique().tolist()),
                                       "trials": int(group.trial_id.nunique()), "windows": len(group),
                                       "trial_classes": group.drop_duplicates("trial_id").label.value_counts().sort_index().to_dict()}
    return summary


def make_split(frame, protocol="subject", seed=42, target=None, test_subject=None):
    rng = np.random.default_rng(seed)
    assignments = {}
    datasets = sorted(frame.dataset.unique())
    if protocol not in ("subject", "lodo", "loso"):
        raise ValueError(protocol)
    if protocol == "lodo" and (target not in datasets or len(datasets) < 2):
        raise ValueError("LODO requires a target dataset and at least one source dataset")
    if protocol == "loso" and test_subject not in set(frame.subject_id):
        raise ValueError("LOSO requires a dataset-qualified --test-subject, e.g. DEAP:S01")
    for dataset in datasets:
        subjects = sorted(frame.loc[frame.dataset == dataset, "subject_id"].unique())
        rng.shuffle(subjects)
        if protocol == "lodo" and dataset == target:
            assignments.update({s: "test" for s in subjects})
            continue
        if protocol == "loso":
            subjects = [s for s in subjects if s != test_subject]
            assignments[test_subject] = "test"
        n_test = max(1, round(.15 * len(subjects))) if protocol == "subject" else 0
        n_val = max(1, round(.15 * len(subjects)))
        if len(subjects) - n_test - n_val < 1:
            raise ValueError(f"Not enough {dataset} subjects for independent splits")
        assignments.update({s: "test" for s in subjects[:n_test]})
        assignments.update({s: "validation" for s in subjects[n_test:n_test+n_val]})
        assignments.update({s: "train" for s in subjects[n_test+n_val:]})
    result = frame.copy()
    result["split"] = result.subject_id.map(assignments)
    summary = validate_split(result)
    if protocol == "lodo":
        if set(result.loc[result.split == "test", "dataset"]) != {target} or target in set(result.loc[result.split != "test", "dataset"]):
            raise ValueError("Target dataset entered LODO training/validation")
    return result, summary


def cap_windows(frame, maximum=None):
    if maximum is None:
        return frame.copy()
    if maximum < 1:
        raise ValueError("Window cap must be positive")
    groups = []
    for _, group in frame.groupby("trial_id", sort=True):
        group = group.sort_values("start_sample")
        indices = np.linspace(0, len(group)-1, min(maximum, len(group)), dtype=int)
        groups.append(group.iloc[indices])
    return pd.concat(groups, ignore_index=True)


def sampling_weights(frame):
    # Equal dataset/class contributions; equal trials within each dataset/class.
    # Window counts cannot give a longer GAMEEMO trial disproportionate weight.
    n_datasets = frame.dataset.nunique()
    trials = frame.drop_duplicates("trial_id")
    n_classes = trials.groupby("dataset").label.nunique().to_dict()
    n_trials = trials.groupby(["dataset", "label"]).size().to_dict()
    n_windows = frame.groupby("trial_id").size().to_dict()
    return np.array([1 / (n_datasets * n_classes[r.dataset] * n_trials[(r.dataset, r.label)] * n_windows[r.trial_id])
                     for r in frame.itertuples()], dtype=np.float64)
