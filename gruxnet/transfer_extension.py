"""Cross-target extension reusing the frozen, verified feature-MLP trainer."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch

from . import transfer_controls as core
from .data import sha256, write_json
from .prepare import load_prepared
from .splits import make_split
from .train import scores


@contextmanager
def target_mapping(target_dataset):
    """Isolate dataset routing in this sequential process; never edit the old trainer."""
    if target_dataset not in {"DEAP", "GAMEEMO", "SEEDIV"}:
        raise ValueError("Unknown target dataset")
    datasets = [target_dataset] + sorted({"DEAP", "GAMEEMO", "SEEDIV"} - {target_dataset})
    old_conditions, old_indices = dict(core.CONDITIONS), dict(core.DATASET_INDEX)
    core.CONDITIONS.clear()
    core.CONDITIONS.update(single=[target_dataset], joint=datasets, joint_heads=datasets)
    core.DATASET_INDEX.clear()
    core.DATASET_INDEX.update({name: index for index, name in enumerate(datasets)})
    try:
        yield
    finally:
        core.CONDITIONS.clear()
        core.CONDITIONS.update(old_conditions)
        core.DATASET_INDEX.clear()
        core.DATASET_INDEX.update(old_indices)


def participant_folds(subjects, seed=42):
    subjects = sorted(set(subjects))
    if len(subjects) not in {15, 28, 32}:
        raise ValueError("Expected the complete eligible participant cohort")
    groups = [group.tolist() for group in np.array_split(np.random.default_rng(seed).permutation(subjects), 5)]
    folds = []
    for index in range(5):
        test, validation = groups[index], groups[(index + 1) % 5]
        train = sorted(set(subjects) - set(test) - set(validation))
        folds.append({"fold": index, "train": train, "validation": validation, "test": test})
    return folds


def prepare_features(common_cache, previous_pack, output):
    from scripts.native_seediv_diagnostic import trial_features
    if output.exists():
        raise FileExistsError("Use a fresh feature directory")
    manifest, prepared = load_prepared(common_cache, verify_files=True)
    rows, features = [], []
    for trial_id, trial in manifest.groupby("trial_id", sort=True):
        first = trial.iloc[0]
        windows = np.load(common_cache / first.cache_file, allow_pickle=False)
        if len(trial) != len(windows) or windows.shape[1:] != (14, 512):
            raise ValueError("Incomplete/mismatched original trial windows")
        feature, count = trial_features(windows.transpose(1, 0, 2).reshape(14, -1))
        features.append(feature)
        rows.append({"dataset": first.dataset, "subject_id": first.subject_id,
                     "trial_id": trial_id, "label": int(first.label), "windows": count})
    features, table = np.stack(features), pd.DataFrame(rows)
    # Independently re-extract every old feature and demand exact agreement.
    x, target, z, sources, binding = core.load_inputs(previous_pack, common_cache)
    lookup = dict(zip(table.trial_id, range(len(table))))
    for values, old in [(x, target), (z, sources)]:
        np.testing.assert_array_equal(features[[lookup[key] for key in old.trial_id]], values)
    if features.shape != (2167, 56) or not np.isfinite(features).all():
        raise ValueError("Unexpected complete feature inventory")
    output.mkdir(parents=True)
    np.savez(output / "features.npz", features=features)
    table.to_csv(output / "trials.csv", index=False)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "common_cache_fingerprint": prepared["cache_fingerprint"],
              "features_sha256": sha256(output / "features.npz"),
              "trials_sha256": sha256(output / "trials.csv"),
              "source_sha256": sha256(Path(__file__)),
              "previous_binding": binding, "previous_features_exactly_reproduced": len(target) + len(sources),
              "scope": "All 2167 common-14 original trials, no fitted statistics, identical four bandpower ranges and aggregation as prior controls"}
    write_json(output / "prepared.json", record)
    print(json.dumps(record), flush=True)


def load_extension_inputs(config):
    pack, cache = Path(config["feature_cache"]), Path(config["common_cache"])
    record = json.loads((pack / "prepared.json").read_text())
    assert sha256(pack / "features.npz") == record["features_sha256"]
    assert sha256(pack / "trials.csv") == record["trials_sha256"]
    with np.load(pack / "features.npz", allow_pickle=False) as values:
        features = values["features"].copy()
    table = pd.read_csv(pack / "trials.csv")
    manifest, prepared = load_prepared(cache, verify_files=True)
    assert record["common_cache_fingerprint"] == prepared["cache_fingerprint"]
    assert features.shape == (len(table), 56) and np.isfinite(features).all()
    assert not table.trial_id.duplicated().any() and set(table.trial_id) == set(manifest.trial_id)
    original = manifest.groupby("trial_id", sort=False).first().loc[table.trial_id]
    for field in ["dataset", "subject_id", "label"]:
        np.testing.assert_array_equal(table[field], original[field])
    split, _ = make_split(manifest, "subject", 42)
    external = set(split[(split.dataset != config["target_dataset"]) & (split.split == "train")].trial_id)
    target_mask = (table.dataset == config["target_dataset"]).to_numpy()
    source_mask = table.trial_id.isin(external).to_numpy()
    fields = ["subject_id", "trial_id", "dataset", "label"]
    target, sources = table.loc[target_mask, fields].reset_index(drop=True), table.loc[source_mask, fields].reset_index(drop=True)
    assert set(sources.trial_id) == external and not (set(target.subject_id) & set(sources.subject_id))
    binding = {"pack_sha256": {name: sha256(pack / name) for name in ["features.npz", "trials.csv", "prepared.json"]},
               "common_cache_fingerprint": prepared["cache_fingerprint"],
               "external_training_subjects": sorted(sources.subject_id.unique())}
    return features[target_mask], target, features[source_mask], sources, binding


def linear_pair(x, target, z, sources, folds, output):
    reports, predictions = {}, []
    for condition in ["single", "joint"]:
        results = []
        for fold in folds:
            masks, table, features, scaler = core.frame_for(x, target, z, sources, fold, condition)
            weights = np.zeros(len(table))
            for dataset in core.CONDITIONS[condition]:
                for label in [0, 1]:
                    rows = (table.dataset == dataset) & (table.label == label)
                    if not rows.any():
                        raise ValueError("Missing training class")
                    weights[rows] = 1 / rows.sum()
            weights *= int(masks["train"].sum()) / weights.sum()
            training_x = scaler.transform(features)
            validation_x, test_x = [scaler.transform(x[masks[part]]) for part in ["validation", "test"]]
            candidates, best, best_score, best_c = [], None, -1., None
            for c in [.01, .1, 1., 10.]:
                model = LogisticRegression(C=c, solver="lbfgs", tol=1e-6, max_iter=2000, random_state=42)
                model.fit(training_x, table.label, sample_weight=weights)
                result = scores(target.label[masks["validation"]], model.predict_proba(validation_x)[:, 1])
                candidates.append({"C": c, **result})
                if result["balanced_accuracy"] > best_score:
                    best, best_score, best_c = model, result["balanced_accuracy"], c
            probabilities = best.predict_proba(test_x)[:, 1]
            test = target[masks["test"]].reset_index(drop=True)
            for index, row in test.iterrows():
                predictions.append({"condition": condition, "fold": fold["fold"], "subject_id": row.subject_id,
                                    "trial_id": row.trial_id, "label": int(row.label), "positive_probability": float(probabilities[index])})
            results.append({"fold": fold["fold"], "selected_C": best_c, "validation_candidates": candidates,
                            "test": scores(test.label, probabilities), "effective_training_weight": float(weights.sum())})
        heldout = pd.DataFrame([r for r in predictions if r["condition"] == condition])
        reports[condition] = {"folds": results, "test_out_of_fold": scores(heldout.label, heldout.positive_probability)}
        print(f"Linear {condition}: BA={reports[condition]['test_out_of_fold']['balanced_accuracy']:.4f}", flush=True)
    pd.DataFrame(predictions).to_csv(output / "linear_trial_predictions.csv", index=False)
    write_json(output / "linear_comparison.json", reports)
    return reports


def run(feature_cache, common_cache, target_dataset, plan, output):
    if output.exists():
        raise FileExistsError("Use a fresh target experiment directory")
    declaration = json.loads(plan.read_text())
    if target_dataset not in declaration["targets"] or declaration["seeds"] != [42, 43, 44]:
        raise ValueError("Target/seeds differ from declared extension")
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "development_only": True,
              "research_question_change_approved": False, "target_dataset": target_dataset,
              "feature_cache": str(feature_cache.resolve()), "common_cache": str(common_cache.resolve()),
              "plan_sha256": sha256(plan), "hardware": {"device": "cuda", "gpu": torch.cuda.get_device_name(0)},
              "scope": "Matched feature-MLP target extension, not GRU-XNet, native four-class supervision, or unseen-dataset transfer"}
    x, target, z, sources, binding = load_extension_inputs(config)
    config["input_binding"] = binding
    folds = participant_folds(target.subject_id.tolist())
    for fold in folds:
        for part in ["train", "validation", "test"]:
            if set(target[target.subject_id.isin(fold[part])].label) != {0, 1}:
                raise ValueError("Class-deficient predeclared partition; do not change folds based on results")
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes(plan.read_bytes())
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    config["source_sha256"] = {}
    for name in ["transfer_extension.py", "transfer_controls.py", "train.py", "deap_control.py", "splits.py"]:
        source = Path(__file__).with_name(name)
        (snapshot / name).write_bytes(source.read_bytes())
        config["source_sha256"][name] = sha256(source)
    torch.set_num_threads(2)
    with target_mapping(target_dataset):
        config["conditions"], config["dataset_index"] = dict(core.CONDITIONS), dict(core.DATASET_INDEX)
        write_json(output / "config.json", config)
        write_json(output / "folds.json", folds)
        linear_pair(x, target, z, sources, folds, output)
        frames, models = [], []
        for seed in declaration["seeds"]:
            for fold in folds:
                for condition in core.CONDITIONS:
                    name = f"{condition}_seed{seed}_fold{fold['fold']}"
                    predictions, result = core.train_one(x, target, z, sources, fold, condition, seed,
                                                         output / "models" / name, torch.device("cuda"))
                    frames.append(predictions)
                    models.append(result)
                    pd.concat(frames, ignore_index=True).to_csv(output / "trial_predictions.csv", index=False)
                    write_json(output / "model_metrics.json", models)
        report = {}
        for (condition, budget), rows in pd.concat(frames, ignore_index=True).groupby(["condition", "budget"]):
            per_seed = {str(seed): scores(group.label, group.positive_probability) for seed, group in rows.groupby("seed")}
            values = [r["balanced_accuracy"] for r in per_seed.values()]
            report[f"{condition}:{budget}"] = {"per_seed": per_seed, "mean_seed_BA": float(np.mean(values)),
                                                "std_seed_BA": float(np.std(values, ddof=1))}
        write_json(output / "neural_comparison.json", report)
        print(json.dumps({key: value["mean_seed_BA"] for key, value in report.items()}), flush=True)
