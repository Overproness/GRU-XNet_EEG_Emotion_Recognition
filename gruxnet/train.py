"""Subject-independent training, isolated validation, and checkpoint-consistent metrics."""
from __future__ import annotations

import json
import os
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import sha256, write_json
from .model import CompactGRUXNet, time_frequency
from .prepare import load_prepared
from .splits import cap_windows, make_split, sampling_weights, validate_split


class WindowDataset(Dataset):
    def __init__(self, frame, cache):
        self.rows = frame.reset_index(drop=True)
        self.cache = cache
        self.open_files = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        filename = row.cache_file
        if filename not in self.open_files:
            self.open_files[filename] = np.load(self.cache / filename, mmap_mode="r", allow_pickle=False)
            if len(self.open_files) > 8:
                self.open_files.popitem(last=False)
        self.open_files.move_to_end(filename)
        waveform = torch.from_numpy(np.array(self.open_files[filename][int(row.cache_index)], dtype=np.float32, copy=True))
        mask = torch.tensor([c == "1" for c in row.channel_mask], dtype=torch.bool)
        return waveform, mask, int(row.label), index


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def scores(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = (p >= .5).astype(int)
    return {"n": len(y), "class_counts": np.bincount(y, minlength=2).tolist(),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if len(np.unique(y)) == 2 else None,
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def aggregate_trials(predictions):
    if (predictions.groupby("trial_id").label.nunique() != 1).any():
        raise ValueError("A trial has conflicting labels")
    return predictions.groupby(["dataset", "subject_id", "trial_id"], as_index=False).agg(
        label=("label", "first"), positive_probability=("positive_probability", "mean"), windows=("label", "size"))


def metrics(predictions, bootstrap=0, seed=42):
    trials = aggregate_trials(predictions)
    result = {"window": scores(predictions.label, predictions.positive_probability),
              "trial": scores(trials.label, trials.positive_probability), "datasets": {}}
    rng = np.random.default_rng(seed)
    for dataset, group in trials.groupby("dataset"):
        item = scores(group.label, group.positive_probability)
        item["subjects"] = group.subject_id.nunique()
        subjects = sorted(group.subject_id.unique())
        if bootstrap and len(subjects) >= 2:
            estimates = []
            blocks = {s: group[group.subject_id == s] for s in subjects}
            for _ in range(bootstrap):
                sampled = pd.concat([blocks[s] for s in rng.choice(subjects, len(subjects), replace=True)])
                if sampled.label.nunique() == 2:
                    estimates.append(balanced_accuracy_score(sampled.label, sampled.positive_probability >= .5))
            item["subject_cluster_bootstrap_balanced_accuracy_95ci"] = np.quantile(estimates, [.025, .975]).tolist() if estimates else None
            item["bootstrap_valid_replicates"] = len(estimates)
            item["bootstrap_requested_replicates"] = bootstrap
        result["datasets"][dataset] = item
    values = [s["balanced_accuracy"] for s in result["datasets"].values() if s["balanced_accuracy"] is not None]
    result["macro_dataset_trial_balanced_accuracy"] = float(np.mean(values))
    return result, trials


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    records, total_loss, n = [], 0., 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for waveforms, mask, labels, indices in loader:
        waveforms, mask, labels = waveforms.to(device), mask.to(device), labels.to(device)
        features, mask = time_frequency(waveforms, mask, augment=False)
        # Full precision evaluation gives identical metric reconstruction across CLI calls.
        logits = model(features, mask)
        total_loss += float(criterion(logits, labels))
        n += len(labels)
        probabilities = logits.softmax(dim=-1)[:, 1].cpu().numpy()
        for i, p in zip(indices.tolist(), probabilities):
            row = loader.dataset.rows.iloc[i]
            records.append({"sample_id": row.sample_id, "dataset": row.dataset, "subject_id": row.subject_id,
                            "trial_id": row.trial_id, "label": int(row.label), "positive_probability": float(p)})
    return pd.DataFrame(records), total_loss / n


def loader_for(frame, cache, batch_size, device, seed, training=False):
    dataset = WindowDataset(frame, cache)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.from_numpy(sampling_weights(frame)), len(frame), replacement=True,
                                    generator=generator) if training else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0,
                      pin_memory=device.type == "cuda", generator=generator)


def train(cache: Path, output: Path, protocol="subject", target=None, test_subject=None, seed=42,
          epochs=30, batch_size=16, accumulation=4, patience=8, max_windows_per_trial=None,
          recurrent="gru", attention=True, augment=True, device_name="auto", verify_cache=False, frequency_pooling="flatten"):
    if min(epochs, batch_size, accumulation, patience) < 1:
        raise ValueError("Epochs, batch size, accumulation, and patience must be positive")
    if output.exists():
        raise FileExistsError(f"Use a new run directory: {output}")
    frame, prepared = load_prepared(cache, verify_files=verify_cache)
    split, full_summary = make_split(frame, protocol, seed, target, test_subject)
    selected = cap_windows(split, max_windows_per_trial)
    summary = validate_split(selected)
    seed_everything(seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    model = CompactGRUXNet(len(prepared["config"]["channels"]), recurrent, attention, frequency_pooling).to(device)
    output.mkdir(parents=True)
    split.to_csv(output / "full_split.csv", index=False)
    selected.to_csv(output / "selected_split.csv", index=False)
    code = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        (snapshot / p.name).write_bytes(p.read_bytes())
    config = {"run_id": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
              "cache": str(cache.resolve()), "cache_fingerprint": prepared["cache_fingerprint"],
              "protocol": protocol, "target": target, "test_subject": test_subject, "seed": seed,
              "epochs": epochs, "batch_size": batch_size, "accumulation": accumulation, "patience": patience,
              "max_windows_per_trial": max_windows_per_trial, "pilot": max_windows_per_trial is not None,
              "model": {"name": "GRU-XNet-Compact-v2" if frequency_pooling == "flatten" else "GRU-XNet-Compact-v1", "channels": len(prepared["config"]["channels"]),
                        "recurrent": recurrent, "attention": attention, "frequency_pooling": frequency_pooling},
              "augment": augment, "augmentation": ["Gaussian noise SD .03", "gain .9..1.1", "channel dropout p=.05"] if augment else [],
              "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001},
              "selection_metric": "validation macro dataset trial balanced accuracy", "amp": device.type == "cuda",
              "sampler": "equal datasets/classes/trials; inverse windows per trial; training only",
              "source_sha256": code, "selected_split_sha256": sha256(output / "selected_split.csv"),
              "verify_cache_at_start": verify_cache,
              "full_split_sha256": sha256(output / "full_split.csv"), "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "hardware": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "parameters": sum(p.numel() for p in model.parameters()), "prepared_config": prepared["config"]}
    write_json(output / "config.json", config)
    write_json(output / "split_audit.json", {"full": full_summary, "selected": summary})
    partitions = {s: selected[selected.split == s].copy() for s in ["train", "validation", "test"]}
    train_loader = loader_for(partitions["train"], cache, batch_size, device, seed, training=True)
    val_loader = loader_for(partitions["validation"], cache, batch_size, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    best, stale, history = -1., 0, []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Training {config['parameters']:,} parameters on {device}; splits="
          f"{ {s: len(f) for s,f in partitions.items()} }; pilot={config['pilot']}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, samples, group_samples = 0., 0, 0
        epoch_started = time.perf_counter()
        for step, (waveforms, mask, labels, _) in enumerate(train_loader, start=1):
            waveforms, mask, labels = waveforms.to(device), mask.to(device), labels.to(device)
            features, mask = time_frequency(waveforms, mask, augment=augment)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_sum = criterion(model(features, mask), labels)
            scaler.scale(loss_sum).backward()
            running += float(loss_sum.detach())
            samples += len(labels)
            group_samples += len(labels)
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                # Correct normalization even for the final, short accumulation group.
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(group_samples)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
            if step % 100 == 0:
                print(f"epoch={epoch} batch={step}/{len(train_loader)} loss={running/samples:.4f}", flush=True)
        validation, val_loss = evaluate(model, val_loader, device)
        val_metrics, _ = metrics(validation)
        score = val_metrics["macro_dataset_trial_balanced_accuracy"]
        record = {"epoch": epoch, "train_loss": running/samples, "validation_loss": val_loss,
                  "validation_macro_trial_balanced_accuracy": score, "lr": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter()-epoch_started}
        history.append(record)
        scheduler.step()
        print(f"epoch={epoch} train_loss={record['train_loss']:.4f} validation_BA={score:.4f} seconds={record['seconds']:.1f}", flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "config": config, "validation_score": score}, output / "best.pt")
            validation.to_csv(output / "best_validation_predictions.csv", index=False)
            write_json(output / "best_validation_metrics.json", val_metrics)
        else:
            stale += 1
        write_json(output / "history.json", history)
        if stale >= patience:
            break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loader = loader_for(partitions["test"], cache, batch_size, device, seed)
    predictions, test_loss = evaluate(model, test_loader, device)
    result, trials = metrics(predictions, bootstrap=1000, seed=seed)
    result.update(run_id=output.name, checkpoint="best.pt", checkpoint_sha256=sha256(output / "best.pt"),
                  best_epoch=checkpoint["epoch"], test_loss=test_loss, protocol=protocol, pilot=config["pilot"],
                  max_windows_per_trial=max_windows_per_trial, selected_split_sha256=config["selected_split_sha256"],
                  cache_fingerprint=config["cache_fingerprint"], elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
                  peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type == "cuda" else 0)
    predictions.to_csv(output / "test_window_predictions.csv", index=False)
    trials.to_csv(output / "test_trial_predictions.csv", index=False)
    result["predictions_sha256"] = sha256(output / "test_window_predictions.csv")
    write_json(output / "test_metrics.json", result)
    plot_run(output, history, result)
    print(json.dumps({"run": output.name, "best_epoch": result["best_epoch"], "test_trial_macro_BA": result["macro_dataset_trial_balanced_accuracy"],
                      "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def plot_run(output, history, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Training")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[1].plot(epochs, [r["validation_macro_trial_balanced_accuracy"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation balanced accuracy", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(result["datasets"]), figsize=(4*len(result["datasets"]), 3.6), squeeze=False)
    for ax, (dataset, values) in zip(axes[0], result["datasets"].items()):
        ConfusionMatrixDisplay(np.array(values["confusion_matrix"]), display_labels=["Negative", "Positive"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{dataset}: held-out trials")
    fig.suptitle("Pilot — limited windows per trial" if result["pilot"] else "Independent test evaluation")
    fig.tight_layout()
    fig.savefig(output / "test_confusion.png", dpi=160)
    plt.close(fig)


def verify_run(run: Path, cache: Path | None = None):
    config = json.loads((run / "config.json").read_text())
    cache = cache or Path(config["cache"])
    _, prepared = load_prepared(cache, verify_files=True)
    if prepared["cache_fingerprint"] != config["cache_fingerprint"]:
        raise ValueError("Run uses a different prepared cache")
    if sha256(run / "selected_split.csv") != config["selected_split_sha256"]:
        raise ValueError("Split manifest changed")
    split = pd.read_csv(run / "selected_split.csv", dtype={"channel_mask": str, "session": str})
    validate_split(split)
    seed_everything(config["seed"])
    device = torch.device(config["hardware"]["device"])
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    settings = config["model"]
    model = CompactGRUXNet(settings["channels"], settings["recurrent"], settings["attention"], settings.get("frequency_pooling", "mean")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = loader_for(split[split.split == "test"], cache, config["batch_size"], device, config["seed"])
    predictions, loss = evaluate(model, loader, device)
    saved = pd.read_csv(run / "test_window_predictions.csv")
    result = json.loads((run / "test_metrics.json").read_text())
    if sha256(run / "best.pt") != result["checkpoint_sha256"] or sha256(run / "test_window_predictions.csv") != result["predictions_sha256"]:
        raise ValueError("Checkpoint or saved predictions changed")
    if predictions.sample_id.tolist() != saved.sample_id.tolist() or not np.allclose(predictions.positive_probability, saved.positive_probability, atol=1e-7, rtol=0):
        raise ValueError("Checkpoint does not reproduce saved predictions")
    actual, _ = metrics(predictions, bootstrap=1000, seed=config["seed"])
    for key in ("window", "trial", "datasets", "macro_dataset_trial_balanced_accuracy"):
        if actual[key] != result[key]:
            raise ValueError(f"Saved metric mismatch: {key}")
    if abs(loss - result["test_loss"]) > 1e-7:
        raise ValueError("Saved test loss does not reproduce")
    print(f"Verified cache, split, checkpoint, all {len(saved)} predictions, and reported metrics: {run.name}", flush=True)
