"""DEAP-only raw-EEG control with independent participants and frozen scaling."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import (DEAP_CHANNELS, POLICY, deap_reference, digest, load_provenance,
                   roots, sha256, valence_label, verify_deap_labels, write_json)
from .eegnet import EEGNetControl, TrainingChannelScaler, normalized_waveforms
from .prepare import load_prepared, preprocess_signal
from .splits import cap_windows, make_split, validate_split
from .train import aggregate_trials, loader_for, metrics, plot_run, seed_everything


def prepare_deap(data_root: Path, output: Path, provenance: Path | None = None):
    if output.exists():
        raise FileExistsError(f"Use a fresh DEAP cache: {output}")
    root = roots(data_root)["DEAP"]
    references, rating_info = deap_reference(root)
    config = {"preprocess_version": "deap-control-1", "datasets": ["DEAP"],
              "sampling_rate": 128, "window_seconds": 4, "stride_seconds": 4,
              "montage": "deap32", "channels": DEAP_CHANNELS,
              "label_policy": {"DEAP": POLICY["DEAP"]},
              "filter": "4th-order Butterworth 4..40 Hz, zero-phase per source trial",
              "baseline": "First 384 samples discarded; no baseline waveform/feature subtraction",
              "representation": "Raw float32 filtered windows; no STFT",
              "normalization": "Deferred to each run; train-channel statistics fitted after participant splitting"}
    audit = {"rating_reference": rating_info, "declared_sources": load_provenance(provenance) if provenance else None,
             "recovered_rating_changes": [0, 0, 0, 0], "first_party_authenticated": False}
    files = sorted((root / "data_preprocessed_python").glob("s*.dat"))
    if [p.stem for p in files] != [f"s{i:02d}" for i in range(1, 33)]:
        raise ValueError("Expected exactly s01.dat through s32.dat")
    (output / "windows").mkdir(parents=True)
    rows, lineage, excluded = [], [], []
    changes = np.zeros(4, dtype=int)
    for path in files:
        with path.open("rb") as stream:
            values = pickle.load(stream, encoding="latin1")
        if values["data"].shape != (40, 40, 8064):
            raise ValueError(f"Unexpected DEAP signal shape: {path}")
        reference = references[int(path.stem[1:])]
        changes += verify_deap_labels(values["labels"], reference)
        source_checksum, subject = sha256(path), f"DEAP:{path.stem.upper()}"
        for index, (signal, rating) in enumerate(zip(values["data"], reference), start=1):
            trial_id = f"{subject}:T{index:02d}"
            meta = {"dataset": "DEAP", "subject_id": subject, "session": "1", "trial_id": trial_id,
                    "source": str(path.resolve()), "source_sha256": source_checksum,
                    "source_key": str(index), "original_label": float(rating[0]),
                    "pickle_valence": float(values["labels"][index - 1, 0]),
                    "rating_metadata_sha256": rating_info["sha256"], "baseline_removed_samples": 384}
            label = valence_label(float(rating[0]), "DEAP")
            if label is None:
                excluded.append(meta)
                continue
            filtered, mask = preprocess_signal(signal[:32, 384:], 128, DEAP_CHANNELS, DEAP_CHANNELS)
            if not mask.all() or filtered.shape != (32, 7680):
                raise ValueError("DEAP montage/duration mismatch")
            windows = np.stack([filtered[:, start:start + 512] for start in range(0, 7680, 512)])
            filename = f"windows/{digest(trial_id)[:24]}.npy"
            np.save(output / filename, windows, allow_pickle=False)
            lineage.append({**meta, "label": label, "original_sampling_rate": 128,
                            "source_samples_after_baseline": 7680, "resampled_samples": 7680,
                            "source_channels": DEAP_CHANNELS, "discarded_tail_samples": 0,
                            "windows": 15, "cache_file": filename, "cache_sha256": sha256(output / filename)})
            for window in range(15):
                rows.append({"sample_id": f"{trial_id}:W{window*512:06d}", "dataset": "DEAP",
                             "subject_id": subject, "session": "1", "trial_id": trial_id,
                             "label": label, "original_label": float(rating[0]),
                             "start_sample": window*512, "stop_sample": (window+1)*512,
                             "sampling_rate": 128, "cache_file": filename, "cache_index": window,
                             "channel_mask": "1"*32})
        print(f"Prepared {subject}: {len(lineage)} eligible trials / {len(rows)} windows", flush=True)
        del values
    frame = pd.DataFrame(rows)
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate DEAP windows")
    audit["recovered_rating_changes"] = changes.tolist()
    frame.to_csv(output / "manifest.csv", index=False)
    write_json(output / "lineage.json", lineage)
    write_json(output / "excluded_trials.json", excluded)
    write_json(output / "deap_rating_audit.json", audit)
    summary = {"config": config, "source_audit": audit, "eligible_trials": len(lineage),
               "excluded_trials": len(excluded), "windows": len(rows),
               "dataset_window_counts": dict(Counter(frame.dataset)),
               "manifest_sha256": sha256(output / "manifest.csv"), "lineage_sha256": sha256(output / "lineage.json")}
    summary["cache_fingerprint"] = digest(summary)
    write_json(output / "prepared.json", summary)
    return summary


def trial_diagnostics(predictions):
    trials = aggregate_trials(predictions)
    probabilities = trials.positive_probability.to_numpy()
    labels = trials.label.to_numpy()
    p = np.clip(probabilities, 1e-7, 1-1e-7)
    losses = -(labels * np.log(p) + (1-labels) * np.log(1-p))
    return {"trial_predicted_class_counts": np.bincount((probabilities >= .5).astype(int), minlength=2).tolist(),
            "trial_probability_mean": float(probabilities.mean()),
            "trial_probability_std": float(probabilities.std()),
            "trial_probability_min": float(probabilities.min()),
            "trial_probability_max": float(probabilities.max()),
            "balanced_trial_log_loss": float(np.mean([losses[labels == c].mean() for c in [0, 1]]))}


def candidate_is_better(score, loss, best_score, best_loss):
    return score > best_score + 1e-12 or (abs(score-best_score) <= 1e-12 and loss < best_loss-1e-8)


@torch.no_grad()
def evaluate_control(model, loader, device, statistics, normalization):
    model.eval()
    records, total_loss, n = [], 0., 0
    for waveforms, mask, labels, indices in loader:
        if not mask.all():
            raise ValueError("This EEGNet control requires all declared electrodes to be observed")
        x, labels = waveforms.to(device), labels.to(device)
        logits = model(normalized_waveforms(x, statistics, normalization))
        total_loss += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
        n += len(labels)
        for index, probability in zip(indices.tolist(), logits.softmax(-1)[:, 1].cpu().tolist()):
            row = loader.dataset.rows.iloc[index]
            records.append({"sample_id": row.sample_id, "dataset": row.dataset, "subject_id": row.subject_id,
                            "trial_id": row.trial_id, "label": int(row.label), "positive_probability": probability})
    return pd.DataFrame(records), total_loss / n


def train_control(cache: Path, output: Path, seed=42, epochs=100, minimum_epochs=25, patience=15,
                  batch_size=32, learning_rate=.001, normalization="train-channel", device_name="auto"):
    if min(epochs, minimum_epochs, patience, batch_size) < 1 or minimum_epochs > epochs or learning_rate <= 0:
        raise ValueError("Invalid positive training budget")
    if normalization not in ("train-channel", "window"):
        raise ValueError("Unknown normalization")
    if output.exists():
        raise FileExistsError(f"Use a fresh control run: {output}")
    frame, prepared = load_prepared(cache, verify_files=True)
    if set(frame.dataset) != {"DEAP"} or prepared["config"]["channels"] != DEAP_CHANNELS:
        raise ValueError("Use the canonical 32-channel DEAP-only cache")
    split, split_audit = make_split(frame, "subject", seed)
    partitions = {name: split[split.split == name].copy() for name in ["train", "validation", "test"]}
    scaler = TrainingChannelScaler.fit(partitions["train"], cache)
    seed_everything(seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    model = EEGNetControl(channels=32, samples=512, dropout=.5).to(device)
    output.mkdir(parents=True)
    split.to_csv(output / "selected_split.csv", index=False)
    scaler.save(output / "normalizer.npz")
    fit_info = {"fit_partition": "train", "fit_subjects": sorted(partitions["train"].subject_id.unique().tolist()),
                "fit_windows": len(partitions["train"]), "values_per_electrode": scaler.count,
                "fit_sample_ids_sha256": digest(partitions["train"].sample_id.tolist()),
                "mean": scaler.mean.tolist(), "scale": scaler.scale.tolist(),
                "normalizer_sha256": sha256(output / "normalizer.npz")}
    write_json(output / "normalizer_fit.json", fit_info)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
        source_hashes[path.name] = sha256(path)
    config = {"run_id": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
              "cache": str(cache.resolve()), "cache_fingerprint": prepared["cache_fingerprint"],
              "model": {"name": "EEGNet-8,2-PyTorch-Control", "channels": DEAP_CHANNELS, "samples": 512,
                        "F1": 8, "D": 2, "F2": 16, "temporal_kernels": [64, 16], "pooling": [4, 8],
                        "dropout": .5, "spatial_max_norm": 1., "classifier_max_norm": .25,
                        "reference": "https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py"},
              "parameters": sum(p.numel() for p in model.parameters()), "seed": seed,
              "epochs": epochs, "minimum_epochs": minimum_epochs, "patience": patience, "batch_size": batch_size,
              "optimizer": {"name": "Adam", "lr": learning_rate, "weight_decay": 0., "eps": 1e-8},
              "scheduler": None, "augment": False, "normalization": normalization, "amp": False,
              "selection": "validation trial balanced accuracy; ties use lower class-balanced trial log loss",
              "early_stopping": "patience on selected validation score/tie loss; never stop before minimum_epochs",
              "protocol": "subject", "development_result": True, "pilot": False,
              "development_reason": "Same DEAP test subjects as earlier inspected development split; not confirmatory",
              "sampler": "Equal training classes/trials; inverse windows per trial; replacement",
              "training_probe": "Two fixed windows per training trial; descriptive diagnostics only",
              "selected_split_sha256": sha256(output / "selected_split.csv"),
              "normalizer_sha256": fit_info["normalizer_sha256"], "source_sha256": source_hashes,
              "prepared_config": prepared["config"],
              "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "hardware": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    write_json(output / "config.json", config)
    write_json(output / "split_audit.json", split_audit)
    statistics = scaler.tensors(device)
    training = loader_for(partitions["train"], cache, batch_size, device, seed, training=True)
    validation = loader_for(partitions["validation"], cache, batch_size, device, seed)
    probe = loader_for(cap_windows(partitions["train"], 2), cache, batch_size, device, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_score, best_loss, stale, history = -1., float("inf"), 0, []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"DEAP control: {config['parameters']:,} parameters, {device}, "
          f"{ {k: len(v) for k,v in partitions.items()} }, normalization={normalization}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, seen, gradient_norm = 0., 0, 0, 0.
        epoch_started = time.perf_counter()
        for step, (waveforms, mask, labels, _) in enumerate(training, start=1):
            if not mask.all():
                raise ValueError("Missing electrode in DEAP control")
            x, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalized_waveforms(x, statistics, normalization))
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("Non-finite gradients")
            gradient_norm += float(norm)
            optimizer.step()
            model.apply_constraints()
            running_loss += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
            if step % 200 == 0:
                print(f"epoch={epoch} batch={step}/{len(training)} loss={running_loss/seen:.4f}", flush=True)
        val_predictions, val_loss = evaluate_control(model, validation, device, statistics, normalization)
        val_metrics, _ = metrics(val_predictions)
        val_diag = trial_diagnostics(val_predictions)
        probe_predictions, probe_loss = evaluate_control(model, probe, device, statistics, normalization)
        probe_metrics, _ = metrics(probe_predictions)
        score, tie_loss = val_metrics["macro_dataset_trial_balanced_accuracy"], val_diag["balanced_trial_log_loss"]
        record = {"epoch": epoch, "train_loss": running_loss/seen, "train_online_window_accuracy": correct/seen,
                  "train_probe_loss": probe_loss, "train_probe_trial_balanced_accuracy": probe_metrics["trial"]["balanced_accuracy"],
                  "train_probe_diagnostics": trial_diagnostics(probe_predictions),
                  "validation_loss": val_loss, "validation_macro_trial_balanced_accuracy": score,
                  "validation_trial_accuracy": val_metrics["trial"]["accuracy"],
                  "validation_trial_auroc": val_metrics["trial"]["auroc"], "validation_diagnostics": val_diag,
                  "gradient_norm_mean_before_clip": gradient_norm/len(training),
                  "lr": learning_rate, "seconds": time.perf_counter()-epoch_started}
        history.append(record)
        improved = candidate_is_better(score, tie_loss, best_score, best_loss)
        if improved:
            best_score, best_loss, stale = score, tie_loss, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "config": config,
                        "validation_score": score, "validation_tie_loss": tie_loss}, output / "best.pt")
            val_predictions.to_csv(output / "best_validation_predictions.csv", index=False)
            write_json(output / "best_validation_metrics.json", {**val_metrics, "diagnostics": val_diag})
        else:
            stale += 1
        record.update(checkpoint_improved=improved, stale_epochs=stale)
        write_json(output / "history.json", history)
        print(f"epoch={epoch} train_loss={record['train_loss']:.4f} train_probe_BA={record['train_probe_trial_balanced_accuracy']:.4f} "
              f"validation_BA={score:.4f} predictions={val_diag['trial_predicted_class_counts']} "
              f"best={improved} stale={stale} seconds={record['seconds']:.1f}", flush=True)
        if epoch >= minimum_epochs and stale >= patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    # Test is loaded for inference only after the final checkpoint is selected.
    test = loader_for(partitions["test"], cache, batch_size, device, seed)
    predictions, test_loss = evaluate_control(model, test, device, statistics, normalization)
    result, trials = metrics(predictions, bootstrap=1000, seed=seed)
    result.update(run_id=output.name, model=config["model"]["name"], protocol="subject", pilot=False,
                  development_result=True, checkpoint="best.pt", checkpoint_sha256=sha256(output / "best.pt"),
                  best_epoch=checkpoint["epoch"], epochs_completed=len(history), test_loss=test_loss,
                  test_diagnostics=trial_diagnostics(predictions), normalization=normalization,
                  selected_split_sha256=config["selected_split_sha256"], cache_fingerprint=config["cache_fingerprint"],
                  normalizer_sha256=config["normalizer_sha256"], elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
                  peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type == "cuda" else 0)
    predictions.to_csv(output / "test_window_predictions.csv", index=False)
    trials.to_csv(output / "test_trial_predictions.csv", index=False)
    result["predictions_sha256"] = sha256(output / "test_window_predictions.csv")
    write_json(output / "test_metrics.json", result)
    plot_run(output, history, result)
    print(json.dumps({key: result[key] for key in ["run_id", "best_epoch", "epochs_completed", "macro_dataset_trial_balanced_accuracy", "elapsed_seconds", "peak_cuda_allocated_mib"]}), flush=True)
    return result


def verify_control(run: Path, cache: Path | None = None):
    config = json.loads((run / "config.json").read_text())
    cache = cache or Path(config["cache"])
    manifest, prepared = load_prepared(cache, verify_files=True)
    if prepared["cache_fingerprint"] != config["cache_fingerprint"]:
        raise ValueError("Different DEAP cache")
    for filename, expected in [("selected_split.csv", config["selected_split_sha256"]),
                               ("normalizer.npz", config["normalizer_sha256"])]:
        if sha256(run / filename) != expected:
            raise ValueError(f"Changed run artifact: {filename}")
    split = pd.read_csv(run / "selected_split.csv", dtype={"channel_mask": str, "session": str})
    validate_split(split)
    expected_split, _ = make_split(manifest, "subject", config["seed"])
    pd.testing.assert_frame_equal(split, expected_split)
    fitted = TrainingChannelScaler.fit(split[split.split == "train"], cache)
    saved_scaler = TrainingChannelScaler.load(run / "normalizer.npz")
    np.testing.assert_array_equal(fitted.mean, saved_scaler.mean)
    np.testing.assert_array_equal(fitted.scale, saved_scaler.scale)
    if fitted.count != saved_scaler.count:
        raise ValueError("Scaler sample count differs")
    for name in ["eegnet.py", "deap_control.py", "train.py", "splits.py"]:
        if sha256(Path(__file__).parent / name) != config["source_sha256"][name]:
            raise ValueError(f"Live evaluation source differs; use saved snapshot: {name}")
    seed_everything(config["seed"])
    device = torch.device(config["hardware"]["device"])
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model = EEGNetControl().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = loader_for(split[split.split == "test"], cache, config["batch_size"], device, config["seed"])
    predictions, loss = evaluate_control(model, loader, device, saved_scaler.tensors(device), config["normalization"])
    saved_predictions = pd.read_csv(run / "test_window_predictions.csv")
    result = json.loads((run / "test_metrics.json").read_text())
    for filename, expected in [("best.pt", result["checkpoint_sha256"]),
                               ("test_window_predictions.csv", result["predictions_sha256"])]:
        if sha256(run / filename) != expected:
            raise ValueError(f"Changed result artifact: {filename}")
    if predictions.sample_id.tolist() != saved_predictions.sample_id.tolist():
        raise ValueError("Prediction rows differ")
    np.testing.assert_allclose(predictions.positive_probability, saved_predictions.positive_probability, atol=1e-7, rtol=0)
    actual, _ = metrics(predictions, bootstrap=1000, seed=config["seed"])
    for key in ["window", "trial", "datasets", "macro_dataset_trial_balanced_accuracy"]:
        if actual[key] != result[key]:
            raise ValueError(f"Metric mismatch: {key}")
    if abs(loss-result["test_loss"]) > 1e-7:
        raise ValueError("Test loss differs")
    history = json.loads((run / "history.json").read_text())
    best_score, best_loss, best_epoch = -1., float("inf"), None
    for record in history:
        score = record["validation_macro_trial_balanced_accuracy"]
        tie_loss = record["validation_diagnostics"]["balanced_trial_log_loss"]
        if candidate_is_better(score, tie_loss, best_score, best_loss):
            best_score, best_loss, best_epoch = score, tie_loss, record["epoch"]
    if checkpoint["epoch"] != best_epoch or checkpoint["validation_score"] != best_score or checkpoint["validation_tie_loss"] != best_loss:
        raise ValueError("Checkpoint does not follow declared validation selection")
    verification = {"verified_utc": datetime.now(timezone.utc).isoformat(), "run_id": run.name,
                    "cache_rehashed": True, "split_recreated": True, "train_only_scaler_refitted": True,
                    "test_predictions_reproduced": len(predictions), "all_metrics_reproduced": True,
                    "validation_selection_reproduced": True}
    write_json(run / "verification.json", verification)
    print(json.dumps(verification), flush=True)
    return verification
