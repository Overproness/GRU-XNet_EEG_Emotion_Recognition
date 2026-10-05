"""Compare all targets using paired participant resampling of pooled trial counts."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.data import sha256, write_json
from gruxnet.train import scores


def pooled_ba(counts):
    """Last axis is TN, FP, FN, TP; absent classes give NaN, never fabricated BA."""
    denominators = np.stack([counts[..., 0] + counts[..., 1], counts[..., 2] + counts[..., 3]], -1)
    correct = np.stack([counts[..., 0], counts[..., 3]], -1)
    recalls = np.full(correct.shape, np.nan, dtype=float)
    np.divide(correct, denominators, out=recalls, where=denominators > 0)
    return recalls.mean(-1)


def trial_counts(rows, subjects, seeds):
    counts = np.empty((len(seeds), len(subjects), 4), dtype=int)
    for i, seed in enumerate(seeds):
        one = rows[rows.seed == seed]
        assert not one.trial_id.duplicated().any() and set(one.subject_id) == set(subjects)
        for j, subject in enumerate(subjects):
            group = one[one.subject_id == subject]
            counts[i, j] = confusion_matrix(group.label, group.positive_probability >= .5, labels=[0, 1]).ravel()
        np.testing.assert_allclose(pooled_ba(counts[i].sum(0)), scores(one.label, one.positive_probability)["balanced_accuracy"])
    return counts


def comparison(first, reference, draws):
    first_boot = pooled_ba(first[:, draws, :].sum(2))
    reference_boot = pooled_ba(reference[:, draws, :].sum(2))
    valid = np.isfinite(first_boot).all(0) & np.isfinite(reference_boot).all(0)
    difference = (first_boot[:, valid] - reference_boot[:, valid]).mean(0)
    values = pooled_ba(first.sum(1)) - pooled_ba(reference.sum(1))
    return {"mean_BA_difference": float(values.mean()), "per_seed_BA_difference": values.tolist(),
            "paired_subject_percentile_95": np.quantile(difference, [.025, .975]).tolist(),
            "bootstrap_draws": len(draws), "valid_draws": int(valid.sum()),
            "class_deficient_draws_excluded": int((~valid).sum())}


def analyze(root, output):
    if output.exists():
        expected = {"comparison.json", "analysis_provenance.json", "multitarget_transfer_comparison.png"}
        if not {path.name for path in output.iterdir()} <= expected:
            raise FileExistsError("Analysis directory contains unrelated files")
    output.mkdir(exist_ok=True, parents=True)
    results = {}
    bindings = {}
    for name in ["SEEDIV", "DEAP", "GAMEEMO"]:
        run = root / ("negative_transfer_neural_seediv" if name == "SEEDIV" else f"negative_transfer_neural_{name.lower()}")
        for verification in ["verification.json", "linear_verification.json"]:
            checked = json.loads((run / verification).read_text())
            assert checked["passed"]
            prediction_file = "trial_predictions.csv" if verification == "verification.json" else "linear_trial_predictions.csv"
            assert sha256(run / prediction_file) == checked["predictions_sha256"]
        rows = pd.read_csv(run / "trial_predictions.csv")
        report = json.loads((run / "neural_comparison.json").read_text())
        linear_report = json.loads((run / "linear_comparison.json").read_text())
        subjects, seeds = sorted(rows.subject_id.unique()), sorted(int(s) for s in rows.seed.unique())
        draws = np.random.default_rng(42).integers(0, len(subjects), size=(10000, len(subjects)))
        arrays, identity = {}, None
        for key, group in rows.groupby(["condition", "budget"]):
            identifier = group[group.seed == seeds[0]][["trial_id", "subject_id", "label"]].sort_values("trial_id").reset_index(drop=True)
            if identity is None:
                identity = identifier
            pd.testing.assert_frame_equal(identifier, identity)
            arrays[f"{key[0]}:{key[1]}"] = trial_counts(group, subjects, seeds)
            np.testing.assert_allclose(pooled_ba(arrays[f"{key[0]}:{key[1]}"].sum(1)).mean(), report[f"{key[0]}:{key[1]}"]["mean_seed_BA"])
        differences = {}
        for key in ["joint:primary", "joint:exposure", "joint_heads:primary", "joint_heads:exposure"]:
            differences[key + " minus single:primary"] = comparison(arrays[key], arrays["single:primary"], draws)
        for budget in ["primary", "exposure"]:
            differences[f"joint_heads:{budget} minus joint:{budget}"] = comparison(arrays[f"joint_heads:{budget}"], arrays[f"joint:{budget}"], draws)
        linear = pd.read_csv(run / "linear_trial_predictions.csv")
        linear["seed"] = 42
        first, reference = [trial_counts(linear[linear.condition == key], subjects, [42]) for key in ["joint", "single"]]
        results[name] = {"target_subjects": len(subjects), "target_trials": len(identity), "seeds": seeds,
                         "neural": {key: report[key] for key in ["single:primary", "joint:primary", "joint:exposure", "joint_heads:primary", "joint_heads:exposure"]},
                         "paired_comparisons": differences,
                         "linear_BA": {key: linear_report[key]["test_out_of_fold"]["balanced_accuracy"] for key in ["single", "joint"]},
                         "linear_joint_minus_single": comparison(first, reference, draws)}
        bindings[name] = {path: sha256(run / path) for path in ["config.json", "trial_predictions.csv", "neural_comparison.json", "linear_comparison.json", "linear_trial_predictions.csv", "verification.json", "linear_verification.json"]}
    write_json(output / "comparison.json", results)
    write_json(output / "analysis_provenance.json", {"development_only": True, "source_sha256": sha256(Path(__file__)), "input_sha256": bindings,
        "bootstrap": "Same 10000 participant draws for paired conditions; pool TN/FP/FN/TP across sampled participants for each initialization, calculate trial BA, then average differences over three initializations. Linear uses its one deterministic fit set. Discard/report class-deficient draws.",
        "limitations": "One fold grouping per target, overlapping training sets, previously inspected/adaptive development cohorts. Intervals capture participant variation conditional on selected models, not all training or selection uncertainty. No confirmatory significance, causal or novelty claim."})
    keys = ["single:primary", "joint:primary", "joint:exposure", "joint_heads:primary", "joint_heads:exposure"]
    labels = ["Target only", "Joint: 600 updates", "Joint: target exposure", "Heads: 600 updates", "Heads: target exposure"]
    colors = ["#466e9a", "#c68a55", "#dfb18a", "#4c917d", "#95bfaf"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, name in zip(axes, results):
        entries = results[name]["neural"]
        values = [100 * entries[key]["mean_seed_BA"] for key in keys]
        deviations = [100 * entries[key]["std_seed_BA"] for key in keys]
        for index, (value, deviation, color, label) in enumerate(zip(values, deviations, colors, labels)):
            ax.bar(index, value, yerr=deviation, capsize=4, color=color, label=label)
            ax.text(index, 18, f"{value:.1f}", ha="center", color="white", fontweight="bold", fontsize=10)
        ax.axhline(50, linestyle="--", color="gray", linewidth=1)
        ax.set_xticks(range(5), ["Only", "Joint\n600", "Joint\nexposure", "Heads\n600", "Heads\nexposure"], fontsize=9)
        ax.set_title(f"{name}: {results[name]['target_subjects']} participants, {results[name]['target_trials']} trials")
        ax.set_ylim(0, 90)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Held-out original-trial balanced accuracy (%)")
    fig.suptitle("Matched feature-MLP development controls: mean ± SD over three initializations")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="lower center", ncol=5, fontsize=9)
    fig.tight_layout(rect=[0, .08, 1, 1])
    fig.savefig(output / "multitarget_transfer_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({name: {key: value["mean_seed_BA"] for key, value in result["neural"].items()} for name, result in results.items()}))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.runs_root, args.output)
