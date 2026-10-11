"""Publication figures from completed public summaries, no fitting/raw access."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_faced_eegnet_diagnostic import qualified_inputs

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "current_progress_2026-10-12"
BASE = REPO / "results" / "development"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    inputs = {
        "spectral": BASE / "faced_inference_quality_2026-10-11" / "comparison.json",
        "covariance": BASE / "faced_representation_v2_2026-10-11" / "comparison.json",
        "neural_audit": BASE / "faced_eegnet_diagnostic_v2_2026-10-11" / "independent_audit.json",
        "neural_records": BASE / "faced_eegnet_diagnostic_v2_2026-10-11" / "fit_records.json",
    }
    audit, records, indexed = qualified_inputs(inputs["neural_audit"].parent)
    spectral, covariance = load(inputs["spectral"]), load(inputs["covariance"])
    table = {}
    for r in range(3):
        for name, val in indexed[r, "eeg_only"]["control_maes"].items():
            table.setdefault(name, {})[r] = float(val)
        for name in ("spectral_eeg_only", "context_plus_spectral_eeg"):
            table.setdefault(name, {})[r] = spectral["rotations"][r]["selected"][name]["validation_unseen"]["participant_film_mae"]
        for name, model in covariance["rotations"][r]["models"].items():
            val = model["metrics"]["validation_unseen"]["participant_film_mae"]
            if name == "context_kernel":
                if abs(table[name][r] - val) > 1e-9:
                    raise ValueError("Context-kernel comparator differs across packets")
                continue
            table.setdefault(name, {})[r] = val
        for name in ("eeg_only", "context_residual"):
            table.setdefault(name, {})[r] = indexed[r, name]["joint_mae"]
    for vals in table.values():
        if set(vals) != {0, 1, 2} or not np.isfinite(list(vals.values())).all():
            raise ValueError("Incomplete paper table")
    out = {name: {"rotations": [vals[r] for r in range(3)], "mean": float(np.mean(list(vals.values())))}
           for name, vals in table.items()}
    out["retrospective_context_envelope"] = {
        "rotations": [min(indexed[r, "eeg_only"]["control_maes"].values()) for r in range(3)],
        "note": "Post-hoc per-panel minimum; not a deployable source-selected model"
    }
    out["retrospective_context_envelope"]["mean"] = float(np.mean(out["retrospective_context_envelope"]["rotations"]))
    PAPER.mkdir(parents=True, exist_ok=True)
    figure_dir = PAPER / "figures"
    figure_dir.mkdir(exist_ok=True)
    (PAPER / "results_table.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = [("context_only", "Context Ridge", "#555555", "o"),
              ("category_median", "Category median", "#999999", "s"),
              ("context_plus_spectral_eeg", "Context + spectral", "#276CA4", "D"),
              ("context_covariance_kernel", "Context + covariance RBF", "#14846D", "^"),
              ("eeg_only", "EEGNet only", "#BF6622", "v"),
              ("context_residual", "Context + EEGNet residual", "#6F4D96", "P")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15), sharey=True)
    y = np.arange(len(labels))[::-1]
    for r, ax in enumerate(axes):
        for pos, (key, label, color, marker) in zip(y, labels):
            ax.scatter(table[key][r], pos, color=color, marker=marker, s=28, zorder=3)
        ax.set_title(f"Panel {r}")
        ax.set_xlim(1.12, 1.76)
        ax.set_xticks([1.2, 1.4, 1.6])
        ax.set_xlabel("MAE (valence units)")
        ax.set_yticks(y)
        ax.set_yticklabels([l[1] for l in labels])
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=.6, w_pad=.8)
    fig.savefig(figure_dir / "mae_comparison.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "mae_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    keyed = {(int(row["rotation"]), row["readout"], float(row["learning_rate"]), int(row["seed"])): row for row in records}
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.5), sharex=True, sharey="row")
    epochs = (30, 60, 120)
    learning = []
    for h, head in enumerate(("eeg_only", "context_residual")):
        chosen = audit["selection"][head]
        for r in range(3):
            ax = axes[h, r]
            for seed, color, marker in ((20261101, "#276CA4", "o"), (20261102, "#BF6622", "s")):
                record = keyed[r, head, float(chosen["learning_rate"]), seed]
                values = [record["checkpoints"][str(e)]["training_target_source_mse"] for e in epochs]
                ax.plot(epochs, values, color=color, marker=marker, markersize=3.3, linewidth=1, label=f"Seed {seed % 100}")
                learning.append({"readout": head, "rotation": r, "seed": seed, "mse": values})
            ax.axhline(record["starting_control_mse"], color=".5", linestyle="--", linewidth=.8, label="Offset only")
            ax.axvline(chosen["epoch"], color=".6", linestyle=":", linewidth=.9, label="Chosen epoch")
            ax.set_title(("EEGNet only" if h == 0 else "Context residual") + f", panel {r}")
            ax.set_xticks(epochs)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_ylabel("Source MSE")
            if h == 1:
                ax.set_xlabel("Training epoch")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .055, 1, 1), pad=.7, h_pad=1.2)
    fig.savefig(figure_dir / "source_learning.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "source_learning.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    receipt = {"scope": "Completed public aggregate results only; zero fits, new predictions or original ratings/waveforms",
               "inputs": {str(v.relative_to(REPO)).replace("\\", "/"): sha(v) for v in inputs.values()},
               "script_sha256": sha(Path(__file__)),
               "outputs": {str(v.relative_to(PAPER)).replace("\\", "/"): sha(v) for v in list(figure_dir.glob("*")) + [PAPER / "results_table.json"]},
               "learning": learning}
    (PAPER / "figure_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(table), "figures": 4, "destination": str(PAPER)}))


if __name__ == "__main__":
    build()
