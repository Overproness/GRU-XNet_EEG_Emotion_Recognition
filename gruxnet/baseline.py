"""A training-fitted log-bandpower baseline on the exact neural subject split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import sha256, write_json
from .prepare import load_prepared
from .splits import sampling_weights, validate_split
from .train import metrics

BANDS = [(4, 8), (8, 14), (14, 31), (31, 41)]


def log_bandpower(windows):
    frequencies, power = welch(windows, fs=128, nperseg=256, noverlap=128, axis=-1)
    width = frequencies[1] - frequencies[0]
    bands = [power[..., (frequencies >= low) & (frequencies < high)].sum(axis=-1) * width for low, high in BANDS]
    return np.log(np.maximum(np.stack(bands, axis=-1), 1e-12)).reshape(len(windows), -1)


def baseline(cache: Path, split: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Use a fresh baseline directory: {output}")
    manifest, prepared = load_prepared(cache, verify_files=True)
    frame = pd.read_csv(split, dtype={"channel_mask": str, "session": str})
    validate_split(frame)
    indexed = manifest.set_index("sample_id")
    reference = indexed.loc[frame.sample_id].reset_index()
    fields = ["dataset", "subject_id", "trial_id", "label", "cache_file", "cache_index", "channel_mask"]
    if not reference[fields].equals(frame[fields].reset_index(drop=True)):
        raise ValueError("Baseline split does not match the prepared manifest")
    if prepared["config"]["montage"] != "common14":
        raise ValueError("This reference baseline uses matching observed electrodes (common14)")
    feature_blocks = []
    for filename, rows in frame.groupby("cache_file", sort=False):
        windows = np.load(cache / filename, mmap_mode="r", allow_pickle=False)
        selected = np.array(windows[rows.cache_index.astype(int)], dtype=np.float32)
        feature_blocks.append(pd.DataFrame(log_bandpower(selected), index=rows.index))
    features = pd.concat(feature_blocks).sort_index().to_numpy()
    training = frame.split.eq("train").to_numpy()
    validation = frame.split.eq("validation").to_numpy()
    testing = frame.split.eq("test").to_numpy()
    weights = sampling_weights(frame[training]) * int(training.sum())
    scaler = StandardScaler().fit(features[training], sample_weight=weights)
    normalized = scaler.transform(features)
    best, best_score, history = None, -1., []
    for strength in [.01, .1, 1., 10.]:
        candidate = LogisticRegression(C=strength, max_iter=1000, random_state=42)
        candidate.fit(normalized[training], frame.loc[training, "label"], sample_weight=weights)
        validation_predictions = frame.loc[validation, ["sample_id", "dataset", "subject_id", "trial_id", "label"]].copy()
        validation_predictions["positive_probability"] = candidate.predict_proba(normalized[validation])[:, 1]
        values, _ = metrics(validation_predictions)
        score = values["macro_dataset_trial_balanced_accuracy"]
        history.append({"C": strength, "validation_macro_dataset_trial_balanced_accuracy": score})
        if score > best_score:
            best, best_score = candidate, score
        print(f"Log-bandpower validation C={strength}: BA={score:.4f}", flush=True)
    output.mkdir(parents=True)
    predictions = frame.loc[testing, ["sample_id", "dataset", "subject_id", "trial_id", "label"]].copy()
    predictions["positive_probability"] = best.predict_proba(normalized[testing])[:, 1]
    result, trials = metrics(predictions, bootstrap=1000, seed=42)
    result.update(model="Log-bandpower + logistic regression", selected_C=best.C, validation_score=best_score,
                  cache_fingerprint=prepared["cache_fingerprint"], split_sha256=sha256(split), bands_hz=BANDS,
                  preprocessing="Raw filtered window power; StandardScaler fitted to weighted training windows only",
                  split_path=str(split.resolve()), development_result=True)
    np.savez(output / "model.npz", mean=scaler.mean_, scale=scaler.scale_, coefficient=best.coef_, intercept=best.intercept_)
    predictions.to_csv(output / "test_window_predictions.csv", index=False)
    trials.to_csv(output / "test_trial_predictions.csv", index=False)
    # Independent reconstruction from exported coefficients, without sklearn state.
    from scipy.special import expit
    exported = np.load(output / "model.npz", allow_pickle=False)
    reconstructed = expit(((features[testing]-exported["mean"])/exported["scale"]) @ exported["coefficient"][0] + exported["intercept"][0])
    if not np.allclose(reconstructed, predictions.positive_probability, atol=1e-12):
        raise ValueError("Exported baseline does not reproduce test probabilities")
    result["coefficient_reconstruction_verified"] = True
    result["model_sha256"] = sha256(output / "model.npz")
    result["predictions_sha256"] = sha256(output / "test_window_predictions.csv")
    write_json(output / "validation_selection.json", history)
    write_json(output / "test_metrics.json", result)
    print(f"Baseline completed: test macro trial BA={result['macro_dataset_trial_balanced_accuracy']:.4f}", flush=True)
    return result
