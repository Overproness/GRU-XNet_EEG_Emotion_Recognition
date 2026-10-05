"""Refit all selected controls and compare shared physical features with the older cache."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.data import COMMON_CHANNELS, sha256, write_json
from gruxnet.prepare import load_prepared
from scripts.native_seediv_diagnostic import load, make_folds, model, trial_features, verify, view


def reproduce(cache, common_cache, output):
    verify(cache, output)
    x, table, metadata = load(cache)
    common, prepared = load_prepared(common_cache, verify_files=True)
    if prepared["config"]["channels"] != COMMON_CHANNELS:
        raise ValueError("Expected the previously prepared common-14 physical montage")
    original_lineage = json.loads((common_cache / "lineage.json").read_text())
    trial_index = {row.trial_id: i for i,row in table.iterrows()}
    selected = [metadata["channels"].index(c) for c in COMMON_CHANNELS]
    maximum_difference, trials_checked = 0., 0
    for row in original_lineage:
        if row["dataset"] != "SEEDIV":
            continue
        index = trial_index[row["trial_id"]]
        source = table.iloc[index]
        if row["source_sha256"] != source.source_sha256 or row["source_key"] != source.source_key:
            raise ValueError("Different original SEED-IV signal source")
        if row["original_label"] != source.original_label or row["windows"] != source.windows:
            raise ValueError("Different label/window lineage")
        windows = np.load(common_cache / row["cache_file"], allow_pickle=False)
        feature, n = trial_features(windows.transpose(1,0,2).reshape(14,-1))
        expected = x[index].reshape(62,4)[selected].reshape(-1)
        difference = float(np.max(np.abs(feature-expected)))
        maximum_difference = max(maximum_difference, difference)
        np.testing.assert_allclose(feature, expected, atol=1e-6, rtol=0)
        if n != row["windows"]:
            raise ValueError("Different feature window count")
        trials_checked += 1
    if trials_checked != 810:
        raise ValueError("Incomplete native/common feature consistency check")

    reports = json.loads((output / "comparison.json").read_text())
    saved = pd.read_csv(output / "trial_predictions.csv")
    folds = make_folds(table.subject_id.tolist())
    refitted, candidate_fits, reconstructed = 0, 0, 0
    for name, report in reports.items():
        inputs, labels, trials = view(x, table, metadata["channels"], report["montage"], report["task"])
        if set(saved[saved.view == name].trial_id) != set(trials.trial_id):
            raise ValueError("Incomplete original trial coverage")
        for fold in folds:
            masks = {part: trials.subject_id.isin(fold[part]).to_numpy() for part in ["train", "validation", "test"]}
            record = report["folds"][fold["fold"]]
            # Rebuild every validation candidate to verify selection from the original training rows.
            best, score, selected_c = None, -1., None
            for candidate in record["validation_candidates"]:
                fitted = model(candidate["C"])
                fitted.fit(inputs[masks["train"]], labels[masks["train"]])
                validation_ba = balanced_accuracy_score(labels[masks["validation"]], fitted.predict(inputs[masks["validation"]]))
                if validation_ba != candidate["balanced_accuracy"]:
                    raise ValueError("Validation candidate does not reproduce")
                if validation_ba > score:
                    best, score, selected_c = fitted, validation_ba, candidate["C"]
                candidate_fits += 1
            if selected_c != record["selected_C"]:
                raise ValueError("Validation-only model selection differs")
            fitted = best
            np.testing.assert_allclose(fitted.named_steps["standardscaler"].mean_, inputs[masks["train"]].mean(0, dtype=np.float64), atol=1e-10, rtol=0)
            prediction = fitted.predict(inputs[masks["test"]])
            probability = fitted.predict_proba(inputs[masks["test"]])
            ids = trials[masks["test"]].trial_id.tolist()
            actual = saved[(saved.view == name) & (saved.fold == fold["fold"])].set_index("trial_id").loc[ids]
            np.testing.assert_array_equal(labels[masks["test"]], actual.label)
            np.testing.assert_array_equal(prediction, actual.prediction)
            for j, c in enumerate(fitted.classes_):
                np.testing.assert_allclose(probability[:,j], actual[f"p_{c}"], atol=1e-10, rtol=0)
            refitted += 1
            reconstructed += len(prediction)
    record = {"passed": True, "validation_candidate_fits_reproduced": candidate_fits,
              "selected_models_reproduced": refitted, "test_predictions_reproduced": reconstructed,
              "same_source_common14_trials": trials_checked, "maximum_shared_feature_difference": maximum_difference,
              "common_cache_fingerprint": prepared["cache_fingerprint"], "native_cache_fingerprint": metadata["fingerprint"],
              "predictions_sha256": sha256(output / "trial_predictions.csv"),
              "scope": "All validation fits/selections and test labels, predictions and probabilities; training-only scaler means; shared-electrode features for all 810 nonneutral trials agree with independently retained common14 cache. Native-only and neutral source features are not regenerated by this check."}
    write_json(output / "reproduction.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--common-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.cache, args.common_cache, args.output)
