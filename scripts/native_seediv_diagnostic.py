"""Exploratory native-label/montage controls; this does not adopt a new paper question."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gruxnet.data import COMMON_CHANNELS, SEED_LABELS, digest, numeric_eeg_keys, roots, seed_channels, sha256, write_json
from gruxnet.prepare import preprocess_signal

BANDS = [(4, 8), (8, 14), (14, 31), (31, 40)]
C_VALUES = [.01, .1, 1., 10.]


def trial_features(filtered):
    """Mean window log power, channel-major and band-minor, without fitted statistics."""
    if filtered.ndim != 2 or filtered.shape[1] < 512 or not np.isfinite(filtered).all():
        raise ValueError("Expected finite channel-by-time signal, at least four seconds")
    n = filtered.shape[1] // 512
    windows = filtered[:, :n*512].reshape(filtered.shape[0], n, 512).transpose(1, 0, 2)
    frequency, density = welch(windows, fs=128, window="hann", nperseg=256,
                               noverlap=128, nfft=256, detrend="constant", scaling="density", axis=-1)
    power = np.stack([density[..., (frequency >= low) & (frequency < high)].sum(-1) * .5
                      for low, high in BANDS], axis=-1)
    return np.log(np.maximum(power, 1e-12)).mean(0).reshape(-1).astype(np.float32), n


def make_folds(subjects, seed=42):
    subjects = sorted(set(subjects))
    if len(subjects) != 15:
        raise ValueError("This predeclared control requires all 15 SEED-IV participants")
    permutation = np.random.default_rng(seed).permutation(subjects).tolist()
    groups = [permutation[i:i+3] for i in range(0, 15, 3)]
    folds = []
    for i in range(5):
        test, validation = groups[i], groups[(i+1) % 5]
        train = sorted(set(subjects) - set(test) - set(validation))
        if set(train) & set(test) or set(train) & set(validation) or set(test) & set(validation):
            raise ValueError("Participant leakage")
        folds.append({"fold": i, "train": train, "validation": validation, "test": test})
    return folds


def view(features, table, channels, montage, task):
    if features.shape != (len(table), 62*4) or len(channels) != 62:
        raise ValueError("Native feature dimensions differ")
    if montage not in ("native62", "common14") or task not in ("native4", "binary"):
        raise ValueError("Unknown predeclared view")
    selected = channels if montage == "native62" else COMMON_CHANNELS
    indices = [channels.index(name) for name in selected]
    x = features.reshape(len(features), 62, 4)[:, indices].reshape(len(features), -1)
    eligible = np.ones(len(table), dtype=bool) if task == "native4" else (table.original_label != 0).to_numpy()
    y = table.original_label.to_numpy(dtype=int)
    if task == "binary":
        y = (y == 3).astype(int)
    return x[eligible], y[eligible], table[eligible].reset_index(drop=True)


def prepare(data_root, cache):
    if cache.exists():
        raise FileExistsError("Use a fresh native feature cache")
    root = roots(data_root)["SEEDIV"]
    channels = seed_channels(root)
    cache.mkdir(parents=True)
    rows, features, sources = [], [], []
    start = time.perf_counter()
    for session in range(1, 4):
        files = sorted((root / "eeg_raw_data" / str(session)).glob("*.mat"),
                       key=lambda path: int(path.stem.split("_")[0]))
        if [int(p.stem.split("_")[0]) for p in files] != list(range(1, 16)):
            raise ValueError("Missing native SEED-IV participant")
        for path in files:
            values = loadmat(path)
            source = {"session": session, "file": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            sources.append(source)
            subject = f"SEEDIV:S{int(path.stem.split('_')[0]):02d}"
            for number, key in enumerate(numeric_eeg_keys(values), start=1):
                signal = values[key]
                filtered, mask = preprocess_signal(signal, 200, channels, channels)
                if not mask.all():
                    raise ValueError("Missing native electrode")
                feature, count = trial_features(filtered)
                features.append(feature)
                rows.append({"trial_id": f"{subject}:R{session}:T{number:02d}", "subject_id": subject,
                             "session": session, "original_label": SEED_LABELS[session][number-1],
                             "source_file": source["file"], "source_sha256": source["sha256"], "source_key": key,
                             "original_samples": signal.shape[1], "resampled_samples": filtered.shape[1],
                             "windows": count, "discarded_tail_samples": filtered.shape[1] - count*512})
            del values
            print(f"Native features: session {session}, {subject}, {len(rows)} trials", flush=True)
    table, x = pd.DataFrame(rows), np.stack(features)
    if len(table) != 1080 or table.trial_id.duplicated().any() or not (table.groupby('original_label').size() == 270).all():
        raise ValueError("Incomplete native trials or incorrect labels")
    table.to_csv(cache / "trials.csv", index=False)
    np.save(cache / "features.npy", x, allow_pickle=False)
    metadata = {"schema": 1, "development_only": True, "research_question_change_approved": False,
                "channels": channels, "bands_hz_half_open": BANDS, "source_files": sources,
                "metadata_sources": {name: sha256(root / name) for name in ["ReadMe.txt", "Channel Order.xlsx"]},
                "filter": "4th-order 4..40 Hz zero-phase within each trial; resample 200 to 128 Hz",
                "features": "Welch density, Hann 256, overlap 128, FFT 256; bin sum * 0.5 Hz; natural log floor 1e-12; mean over non-overlapping 4-second windows within trial",
                "features_sha256": sha256(cache / "features.npy"), "trials_sha256": sha256(cache / "trials.csv"),
                "shape": list(x.shape), "eligible_trials": len(table), "elapsed_seconds": time.perf_counter()-start}
    metadata["fingerprint"] = digest(metadata)
    write_json(cache / "prepared.json", metadata)
    return metadata


def load(cache):
    metadata = json.loads((cache / "prepared.json").read_text())
    expected = dict(metadata)
    fingerprint = expected.pop("fingerprint")
    if digest(expected) != fingerprint:
        raise ValueError("Changed native feature metadata")
    for name, key in [("features.npy", "features_sha256"), ("trials.csv", "trials_sha256")]:
        if sha256(cache / name) != metadata[key]:
            raise ValueError(f"Changed native cache: {name}")
    return np.load(cache / "features.npy", allow_pickle=False), pd.read_csv(cache / "trials.csv"), metadata


def model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                         max_iter=2000, tol=1e-6, random_state=42))


def summarize(y, prediction, classes):
    return {"trials": len(y), "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=classes).tolist()}


def run(cache, output):
    if output.exists():
        raise FileExistsError("Use a fresh native diagnostic result directory")
    x, table, metadata = load(cache)
    output.mkdir(parents=True)
    folds = make_folds(table.subject_id.tolist())
    write_json(output / "folds.json", folds)
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "development_only": True,
              "research_question_change_approved": False, "cache_fingerprint": metadata["fingerprint"],
              "seed": 42, "models": "Trial-level log-bandpower multinomial logistic regression",
              "C_values": C_VALUES, "folds": "Five fixed rotations: 9 train / 3 validation / 3 test participants",
              "selection": "Maximum validation balanced accuracy; ties select first (smallest) C",
              "scaler": "Training trial features only; validation/test never refitted",
              "source_sha256": sha256(Path(__file__)), "classes": {"native4": [0,1,2,3], "binary": [0,1]},
              "native_class_order": ["neutral", "sad", "fear", "happy"], "evaluation_unit": "original trial",
              "caveats": "Different targets have different chance levels and trial counts; this diagnostic is not neural baseline reproduction, transfer evidence, or a new contribution"}
    write_json(output / "config.json", config)
    reports, predictions = {}, []
    for task in ["native4", "binary"]:
        for montage in ["native62", "common14"]:
            name = f"{task}_{montage}"
            inputs, labels, trials = view(x, table, metadata["channels"], montage, task)
            fold_reports = []
            for fold in folds:
                masks = {part: trials.subject_id.isin(fold[part]).to_numpy() for part in ["train", "validation", "test"]}
                best, best_score, selected, candidates = None, -1., None, []
                for c in C_VALUES:
                    estimator = model(c)
                    estimator.fit(inputs[masks["train"]], labels[masks["train"]])
                    prediction = estimator.predict(inputs[masks["validation"]])
                    result = summarize(labels[masks["validation"]], prediction, config["classes"][task])
                    candidates.append({"C": c, **result})
                    if result["balanced_accuracy"] > best_score:
                        best, best_score, selected = estimator, result["balanced_accuracy"], c
                prediction = best.predict(inputs[masks["test"]])
                probability = best.predict_proba(inputs[masks["test"]])
                tested = trials[masks["test"]].reset_index(drop=True)
                result = summarize(labels[masks["test"]], prediction, config["classes"][task])
                fold_reports.append({"fold": fold["fold"], "selected_C": selected,
                                     "validation_candidates": candidates, "test": result,
                                     "train": summarize(labels[masks["train"]], best.predict(inputs[masks["train"]]), config["classes"][task])})
                for i, row in tested.iterrows():
                    predictions.append({"view": name, "fold": fold["fold"], "subject_id": row.subject_id,
                                        "trial_id": row.trial_id, "label": int(labels[masks["test"]][i]),
                                        "prediction": int(prediction[i]), **{f"p_{c}": probability[i,j] for j,c in enumerate(best.classes_)}})
                print(f"{name} fold={fold['fold']} C={selected} val_BA={best_score:.4f} test_BA={result['balanced_accuracy']:.4f}", flush=True)
            predicted = pd.DataFrame([p for p in predictions if p["view"] == name])
            if predicted.trial_id.duplicated().any() or len(predicted) != len(trials):
                raise ValueError("Each eligible trial must be tested exactly once")
            reports[name] = {"task": task, "montage": montage, "chance_balanced_accuracy": 1/len(config["classes"][task]),
                             "test_out_of_fold": summarize(predicted.label, predicted.prediction, config["classes"][task]),
                             "folds": fold_reports,
                             "subject_balanced_accuracy": {s: float(balanced_accuracy_score(t.label, t.prediction))
                                                           for s,t in predicted.groupby("subject_id")}}
    pd.DataFrame(predictions).to_csv(output / "trial_predictions.csv", index=False)
    write_json(output / "comparison.json", reports)
    return reports


def verify(cache, output):
    _, table, metadata = load(cache)
    config = json.loads((output / "config.json").read_text())
    if config["cache_fingerprint"] != metadata["fingerprint"] or config["source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Changed native cache or diagnostic source")
    folds = json.loads((output / "folds.json").read_text())
    if folds != make_folds(table.subject_id.tolist()):
        raise ValueError("Changed participant folds")
    reports = json.loads((output / "comparison.json").read_text())
    predictions = pd.read_csv(output / "trial_predictions.csv")
    for name, report in reports.items():
        rows = predictions[predictions.view == name]
        if rows.trial_id.duplicated().any():
            raise ValueError("Repeated test trial")
        assert summarize(rows.label, rows.prediction, config["classes"][report["task"]]) == report["test_out_of_fold"]
        for fold in folds:
            selected = rows[rows.fold == fold["fold"]]
            assert set(selected.subject_id) == set(fold["test"])
            assert not set(selected.subject_id) & (set(fold["train"]) | set(fold["validation"]))
            saved = report["folds"][fold["fold"]]
            assert summarize(selected.label, selected.prediction, config["classes"][report["task"]]) == saved["test"]
            best = max(saved["validation_candidates"], key=lambda c: c["balanced_accuracy"])
            assert saved["selected_C"] == best["C"]
    record = {"passed": True, "scope": "Cache/source hashes, subject folds, validation-only C selection, all saved test summary metrics. Does not regenerate source features or refit estimators.",
              "prediction_rows": len(predictions), "predictions_sha256": sha256(output / "trial_predictions.csv"),
              "cache_fingerprint": metadata["fingerprint"]}
    write_json(output / "verification.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "verify"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.data_root is None:
            parser.error("prepare requires --data-root")
        prepare(args.data_root, args.cache)
    else:
        if args.output is None:
            parser.error("run/verify require --output")
        (run if args.command == "run" else verify)(args.cache, args.output)
