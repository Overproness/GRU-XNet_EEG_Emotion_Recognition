"""Describe the selected control checkpoint; does not train or select models."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "GRU-XNet_EEG_Emotion_Recognition"))

import numpy as np
import pandas as pd
import torch

from gruxnet.data import sha256, write_json
from gruxnet.deap_control import evaluate_control, trial_diagnostics
from gruxnet.eegnet import EEGNetControl, TrainingChannelScaler
from gruxnet.train import aggregate_trials, loader_for, metrics, scores, seed_everything


def describe(run):
    run = run.resolve()
    config = json.loads((run / "config.json").read_text())
    result = json.loads((run / "test_metrics.json").read_text())
    assert sha256(run / "best.pt") == result["checkpoint_sha256"]
    assert sha256(run / "normalizer.npz") == config["normalizer_sha256"]
    if (run / "training_metrics.json").exists():
        training = json.loads((run / "training_metrics.json").read_text())
        assert training["checkpoint_sha256"] == result["checkpoint_sha256"]
        assert training["normalization"] == config["normalization"]
        predictions = pd.read_csv(run / "training_window_predictions.csv")
        reconstructed, _ = metrics(predictions)
        for key in ["window", "trial", "datasets", "macro_dataset_trial_balanced_accuracy"]:
            assert reconstructed[key] == training[key]
    else:
        seed_everything(config["seed"])
        device = torch.device(config["hardware"]["device"])
        model = EEGNetControl().to(device)
        checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        scale = TrainingChannelScaler.load(run / "normalizer.npz")
        split = pd.read_csv(run / "selected_split.csv", dtype={"channel_mask": str, "session": str})
        loader = loader_for(split[split.split == "train"], Path(config["cache"]), config["batch_size"], device, config["seed"])
        predictions, loss = evaluate_control(model, loader, device, scale.tensors(device), config["normalization"])
        training, trials = metrics(predictions)
        training.update(purpose="Post-selection descriptive training performance; not used to select checkpoints",
                        best_epoch=result["best_epoch"], loss=loss, diagnostics=trial_diagnostics(predictions),
                        checkpoint_sha256=result["checkpoint_sha256"], normalization=config["normalization"])
        predictions.to_csv(run / "training_window_predictions.csv", index=False)
        trials.to_csv(run / "training_trial_predictions.csv", index=False)
        write_json(run / "training_metrics.json", training)

    # Use existing saved test probabilities; no additional test inference/tuning.
    test_predictions = pd.read_csv(run / "test_window_predictions.csv")
    subjects = []
    for subject, group in aggregate_trials(test_predictions).groupby("subject_id"):
        values = scores(group.label, group.positive_probability)
        predicted = (group.positive_probability.to_numpy() >= .5).astype(int)
        subjects.append({"subject_id": subject, "trials": values["n"], "negative_trials": values["class_counts"][0],
                         "positive_trials": values["class_counts"][1], "accuracy": values["accuracy"],
                         "balanced_accuracy": values["balanced_accuracy"], "auroc": values["auroc"],
                         "predicted_negative_trials": int((predicted == 0).sum()),
                         "predicted_positive_trials": int((predicted == 1).sum()),
                         "constant_prediction": bool(len(np.unique(predicted)) == 1)})
    pd.DataFrame(subjects).to_csv(run / "test_subject_metrics.csv", index=False)

    comparisons = []
    paths = [("Joint compact GRU-XNet v1 / common14 STFT", WORKSPACE / "publication_runs/full_subject_seed42/test_metrics.json"),
             ("Joint log-bandpower logistic regression / common14", WORKSPACE / "publication_runs/bandpower_subject_seed42/test_metrics.json"),
             ("DEAP-only EEGNet / canonical32 raw", run / "test_metrics.json")]
    for name, path in paths:
        values = json.loads(path.read_text())["datasets"]["DEAP"]
        confusion = np.array(values["confusion_matrix"])
        comparisons.append({"description": name, "metrics_source": str(path.relative_to(WORKSPACE)),
                            "metrics_sha256": sha256(path), **values,
                            "predicted_class_counts": confusion.sum(axis=0).tolist()})
    comparison = {"scope": "Descriptive DEAP scores on the same previously inspected test participants",
                  "limitations": "Different training sources, electrodes, representations, regularization and budgets; not a matched architecture or normalization ablation",
                  "training": training, "test_comparison": comparisons,
                  "validation": json.loads((run / "best_validation_metrics.json").read_text())}
    write_json(run / "development_comparison.json", comparison)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = json.loads((run / "history.json").read_text())
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Training, balanced sampler")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation, observed classes")
    axes[0].set(xlabel="Epoch", ylabel="Window cross entropy")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [r["train_probe_trial_balanced_accuracy"] for r in history], label="Training probe, 2 windows/trial")
    axes[1].plot(epochs, [r["validation_macro_trial_balanced_accuracy"] for r in history], label="Validation, 15 windows/trial")
    axes[1].axhline(.5, color="gray", linestyle=":", label="Chance")
    axes[1].axvline(result["best_epoch"], color="black", linestyle="--", alpha=.5)
    axes[1].set(xlabel="Epoch", ylabel="Trial balanced accuracy", ylim=(.35, .85))
    axes[1].legend(fontsize=8)
    fraction = [r["validation_diagnostics"]["trial_predicted_class_counts"][1] /
                sum(r["validation_diagnostics"]["trial_predicted_class_counts"]) for r in history]
    axes[2].plot(epochs, fraction, color="tab:green", label="Predicted positive fraction")
    axes[2].set(xlabel="Epoch", ylabel="Validation trials predicted positive", ylim=(0, 1))
    fig.suptitle("DEAP EEGNet learning control — development split", fontsize=12)
    fig.tight_layout()
    fig.savefig(run / "development_diagnostics.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"training_trial_metrics": training["trial"], "test_comparison": comparisons}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    describe(parser.parse_args().run)
