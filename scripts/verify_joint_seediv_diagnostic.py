"""Reproduce the pooling control's candidate selection and every held-out probability."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.data import sha256, write_json
from scripts.joint_seediv_diagnostic import balanced_weights
from scripts.native_seediv_diagnostic import make_folds, summarize


def verify(output):
    config = json.loads((output / "config.json").read_text())
    original = json.loads((output / "verification.json").read_text())
    if config["source_sha256"] != sha256(Path(__file__).with_name("joint_seediv_diagnostic.py")):
        raise ValueError("Changed pooling source")
    for filename,key in [("features.npz","features_sha256"), ("trial_predictions.csv","predictions_sha256")]:
        if sha256(output / filename) != original[key]:
            raise ValueError("Changed derived feature or prediction file")
    with np.load(output / "features.npz", allow_pickle=False) as values:
        target_x, other_x = values["target"], values["others"]
    target_table = pd.read_csv(output / "target_trials.csv")
    other_table = pd.read_csv(output / "source_trials.csv")
    folds = json.loads((output / "folds.json").read_text())
    assert folds == make_folds(target_table.subject_id.tolist())
    for dataset, info in config["external_sources"].items():
        rows = other_table[other_table.dataset == dataset]
        assert sorted(rows.subject_id.unique()) == info["subjects"] and len(rows) == info["trials"]
    assert not set(target_table.trial_id) & set(other_table.trial_id)
    saved = pd.read_csv(output / "trial_predictions.csv")
    reports = json.loads((output / "comparison.json").read_text())
    candidates, predictions = 0, 0
    for condition, report in reports.items():
        rows = saved[saved.condition == condition]
        assert set(rows.trial_id) == set(target_table.trial_id) and not rows.trial_id.duplicated().any()
        assert summarize(rows.label, rows.prediction, [0,1]) == report["test_out_of_fold"]
        for fold in folds:
            parts = {p: target_table.subject_id.isin(fold[p]).to_numpy() for p in ["train","validation","test"]}
            training = target_table[parts["train"]].reset_index(drop=True)
            raw = target_x[parts["train"]]
            if condition != "single":
                training = pd.concat([training,other_table], ignore_index=True)
                raw = np.concatenate([raw,other_x])
            if condition == "joint_by_dataset":
                transformed = np.empty_like(raw)
                for dataset in training.dataset.unique():
                    mask = (training.dataset == dataset).to_numpy()
                    fitted = StandardScaler().fit(raw[mask])
                    transformed[mask] = fitted.transform(raw[mask])
                    if dataset == "SEEDIV":
                        scaler = fitted
                raw = transformed
            else:
                scaler = StandardScaler().fit(raw)
                raw = scaler.transform(raw)
            validation_x, test_x = [scaler.transform(target_x[parts[p]]) for p in ["validation","test"]]
            record = report["folds"][fold["fold"]]
            best, score, best_c = None, -1., None
            for candidate in record["validation_candidates"]:
                estimator = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=2000, tol=1e-6, random_state=42)
                estimator.fit(raw, training.label, sample_weight=balanced_weights(training))
                actual = summarize(target_table.label[parts["validation"]], estimator.predict(validation_x), [0,1])
                assert actual == {k:v for k,v in candidate.items() if k != "C"}
                if actual["balanced_accuracy"] > score:
                    best, score, best_c = estimator, actual["balanced_accuracy"], candidate["C"]
                candidates += 1
            assert best_c == record["selected_C"]
            ids = target_table[parts["test"]].trial_id.tolist()
            actual = rows[rows.fold == fold["fold"]].set_index("trial_id").loc[ids]
            np.testing.assert_array_equal(target_table.label[parts["test"]],actual.label)
            np.testing.assert_array_equal(best.predict(test_x),actual.prediction)
            np.testing.assert_allclose(best.predict_proba(test_x)[:,1],actual.positive_probability,atol=1e-10,rtol=0)
            predictions += len(actual)
    record = {"passed": True, "validation_candidates_reproduced": candidates, "selected_models_reproduced": 15,
              "test_predictions_reproduced": predictions, "scope": "Refit scalers/classifiers from retained derived training features; reproduce all validation candidates/selections and all held-out probabilities. Features themselves are not regenerated in this check."}
    write_json(output / "reproduction.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
