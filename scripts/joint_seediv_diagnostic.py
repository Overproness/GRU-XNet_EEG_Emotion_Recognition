"""Matched exploratory controls for adding other datasets to SEED-IV training."""
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
from gruxnet.prepare import load_prepared
from gruxnet.splits import make_split
from scripts.native_seediv_diagnostic import C_VALUES, load, make_folds, summarize, trial_features, view


def balanced_weights(table):
    weights = np.zeros(len(table))
    for dataset, group in table.groupby("dataset"):
        for label in [0,1]:
            indices = group.index[group.label == label]
            if len(indices) == 0:
                raise ValueError("Missing training class")
            weights[indices] = 1 / len(indices)
    return weights * len(table) / weights.sum()


def run(native_cache, common_cache, output):
    if output.exists():
        raise FileExistsError("Use a fresh pooling diagnostic directory")
    output.mkdir(parents=True)
    native, native_table, metadata = load(native_cache)
    target_x, target_y, target_table = view(native, native_table, metadata["channels"], "common14", "binary")
    target_table = target_table[["subject_id","trial_id"]].copy()
    target_table["dataset"], target_table["label"] = "SEEDIV", target_y
    manifest, prepared = load_prepared(common_cache, verify_files=True)
    split, _ = make_split(manifest, "subject", 42)
    others = split[(split.dataset != "SEEDIV") & (split.split == "train")]
    other_rows, other_x = [], []
    for trial_id, rows in others.groupby("trial_id", sort=True):
        first = rows.iloc[0]
        if rows.cache_file.nunique() != 1 or rows.subject_id.nunique() != 1 or rows.label.nunique() != 1:
            raise ValueError("Mixed original trial")
        windows = np.load(common_cache / first.cache_file, allow_pickle=False)
        if len(rows) != len(windows):
            raise ValueError("Incomplete original training trial")
        feature, _ = trial_features(windows.transpose(1,0,2).reshape(14,-1))
        other_x.append(feature)
        other_rows.append({"dataset": first.dataset, "subject_id": first.subject_id, "trial_id": trial_id, "label": int(first.label)})
    other_x, other_table = np.stack(other_x), pd.DataFrame(other_rows)
    config = {"development_only": True, "research_question_change_approved": False,
              "source_sha256": sha256(Path(__file__)), "native_cache_fingerprint": metadata["fingerprint"],
              "common_cache_fingerprint": prepared["cache_fingerprint"], "C_values": C_VALUES,
              "target_validation": "SEED-IV validation subjects only; ties choose smallest C",
              "external_sources": {d: {"subjects": sorted(t.subject_id.unique()), "trials": len(t)} for d,t in other_table.groupby("dataset")},
              "weights": "Equal datasets, then equal classes, then equal original trials; total sample weights normalized to number of training trials",
              "scaler": "Unweighted original training trials only; global or separate per dataset; validation/test never fitted",
              "caveats": "Pooling changes training count/domain and label source together; not a causal test of label semantics or neural-model transfer"}
    write_json(output / "config.json", config)
    folds = make_folds(target_table.subject_id.tolist())
    write_json(output / "folds.json", folds)
    reports, all_predictions = {}, []
    for condition in ["single", "joint_global", "joint_by_dataset"]:
        fold_results = []
        for fold in folds:
            train = target_table.subject_id.isin(fold["train"]).to_numpy()
            validation = target_table.subject_id.isin(fold["validation"]).to_numpy()
            test = target_table.subject_id.isin(fold["test"]).to_numpy()
            train_table = target_table[train].reset_index(drop=True)
            training_x = target_x[train]
            if condition != "single":
                train_table = pd.concat([train_table, other_table], ignore_index=True)
                training_x = np.concatenate([training_x, other_x])
            if condition == "joint_by_dataset":
                transformed = np.empty_like(training_x)
                for dataset in train_table.dataset.unique():
                    indices = (train_table.dataset == dataset).to_numpy()
                    scaler = StandardScaler().fit(training_x[indices])
                    transformed[indices] = scaler.transform(training_x[indices])
                    if dataset == "SEEDIV":
                        target_scaler = scaler
                training_x = transformed
            else:
                target_scaler = StandardScaler().fit(training_x)
                training_x = target_scaler.transform(training_x)
            validation_x = target_scaler.transform(target_x[validation])
            test_x = target_scaler.transform(target_x[test])
            weights = balanced_weights(train_table)
            best, best_score, c_best, candidates = None, -1., None, []
            for c in C_VALUES:
                model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-6, random_state=42)
                model.fit(training_x, train_table.label, sample_weight=weights)
                result = summarize(target_y[validation], model.predict(validation_x), [0,1])
                candidates.append({"C": c, **result})
                if result["balanced_accuracy"] > best_score:
                    best, best_score, c_best = model, result["balanced_accuracy"], c
            prediction, probability = best.predict(test_x), best.predict_proba(test_x)[:,1]
            result = summarize(target_y[test], prediction, [0,1])
            fold_results.append({"fold": fold["fold"], "selected_C": c_best, "validation_candidates": candidates,
                                 "test": result, "training_trials": len(train_table)})
            selected_rows = target_table[test].reset_index(drop=True)
            for i, row in selected_rows.iterrows():
                all_predictions.append({"condition": condition, "fold": fold["fold"], "trial_id": row.trial_id,
                                        "subject_id": row.subject_id, "label": int(row.label), "prediction": int(prediction[i]),
                                        "positive_probability": probability[i]})
            print(f"{condition} fold={fold['fold']} C={c_best} validation_BA={best_score:.4f} test_BA={result['balanced_accuracy']:.4f}", flush=True)
        combined = pd.DataFrame([r for r in all_predictions if r["condition"] == condition])
        if combined.trial_id.duplicated().any() or set(combined.trial_id) != set(target_table.trial_id):
            raise ValueError("Incorrect original test trial coverage")
        reports[condition] = {"folds": fold_results, "test_out_of_fold": summarize(combined.label, combined.prediction, [0,1])}
    predictions = pd.DataFrame(all_predictions)
    predictions.to_csv(output / "trial_predictions.csv", index=False)
    write_json(output / "comparison.json", reports)
    # Save only derived features and lineage locally for repeatable replay, outside Git exports.
    np.savez(output / "features.npz", target=target_x, others=other_x)
    target_table.to_csv(output / "target_trials.csv", index=False)
    other_table.to_csv(output / "source_trials.csv", index=False)
    record = {"passed": True, "scope": "Check cached fingerprints, subject partitions, original-trial coverage and reconstruct all aggregate metrics from saved predictions; models not independently refitted by this record",
              "predictions_sha256": sha256(output / "trial_predictions.csv"), "prediction_rows": len(predictions),
              "features_sha256": sha256(output / "features.npz")}
    for condition, report in reports.items():
        rows = predictions[predictions.condition == condition]
        assert summarize(rows.label, rows.prediction, [0,1]) == report["test_out_of_fold"]
        for fold in folds:
            assert set(rows[rows.fold == fold["fold"]].subject_id) == set(fold["test"])
    write_json(output / "verification.json", record)
    print(json.dumps({k: r["test_out_of_fold"] for k,r in reports.items()}))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--common-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.native_cache, args.common_cache, args.output)
