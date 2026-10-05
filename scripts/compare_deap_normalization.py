"""Compare two predeclared normalization runs without fitting or selecting on test."""
from argparse import ArgumentParser
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.data import sha256, write_json


def compare(reference, alternative):
    configs = [json.loads((p / "config.json").read_text()) for p in [reference, alternative]]
    fixed = ["cache_fingerprint", "model", "parameters", "seed", "epochs", "minimum_epochs", "patience",
             "batch_size", "optimizer", "scheduler", "augment", "amp", "selection", "early_stopping",
             "protocol", "sampler", "selected_split_sha256", "prepared_config"]
    for key in fixed:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Unmatched setting: {key}")
    for name in ["eegnet.py", "deap_control.py", "train.py", "splits.py"]:
        if configs[0]["source_sha256"][name] != configs[1]["source_sha256"][name]:
            raise ValueError(f"Different source: {name}")
    assert [c["normalization"] for c in configs] == ["train-channel", "window"]
    frames = [pd.read_csv(p / "test_trial_predictions.csv").sort_values("trial_id").reset_index(drop=True)
              for p in [reference, alternative]]
    for field in ["trial_id", "subject_id", "label"]:
        np.testing.assert_array_equal(frames[0][field], frames[1][field])
    subjects = sorted(frames[0].subject_id.unique())
    groups = {s: np.flatnonzero(frames[0].subject_id == s) for s in subjects}
    # Five subjects allow enumerating all 5**5 equally likely bootstrap draws.
    if len(subjects) != 5:
        raise ValueError("Expected the predeclared five test participants")
    differences = []
    for draw in itertools.product(subjects, repeat=len(subjects)):
        indices = np.concatenate([groups[s] for s in draw])
        values = [balanced_accuracy_score(f.label.iloc[indices], (f.positive_probability.iloc[indices] >= .5).astype(int))
                  for f in frames]
        differences.append(values[1]-values[0])
    runs = []
    for path in [reference, alternative]:
        result = json.loads((path / "test_metrics.json").read_text())
        validation = json.loads((path / "best_validation_metrics.json").read_text())
        training = json.loads((path / "training_metrics.json").read_text())
        per_subject = pd.read_csv(path / "test_subject_metrics.csv")
        runs.append({"run": path.name, "normalization": result["normalization"],
                     "best_epoch": result["best_epoch"], "epochs_completed": result["epochs_completed"],
                     "train_trial": training["trial"], "validation_trial": validation["trial"],
                     "test_trial": result["trial"], "test_subject_constant_predictions": int(per_subject.constant_prediction.sum()),
                     "test_subjects": per_subject.to_dict("records"), "test_metrics_sha256": sha256(path / "test_metrics.json")})
    observed = runs[1]["test_trial"]["balanced_accuracy"] - runs[0]["test_trial"]["balanced_accuracy"]
    report = {"development_only": True, "research_question_change_approved": False,
              "fixed_settings_verified": fixed, "evaluation_source_hashes_equal": True,
              "single_configured_difference": "train-channel versus per-window per-electrode normalization",
              "realized_budget": "Identical stopping/selection policies, different realized stopping epochs and selected checkpoints",
              "runs": runs, "paired_test_balanced_accuracy_difference": observed,
              "paired_subject_bootstrap_percentile_95": np.quantile(differences, [.025,.975]).tolist(),
              "bootstrap_draws": len(differences), "bootstrap_method": "Exact enumeration of equally weighted five-subject resampling draws; pooled trial BA in each draw",
              "limitations": "Only one training seed and five test subjects. Previously inspected development test cohort. No test-based calibration. Does not establish equivalence, statistical superiority, cause of all failures, or publication novelty.",
              "interpretation": "Neither normalization has useful held-out discrimination; changing normalization alone did not resolve generalization on this development cohort"}
    write_json(alternative / "normalization_comparison.json", report)
    print(json.dumps({"observed_BA_difference": observed, "paired_95_interval": report["paired_subject_bootstrap_percentile_95"]}))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    args = parser.parse_args()
    compare(args.reference, args.alternative)
