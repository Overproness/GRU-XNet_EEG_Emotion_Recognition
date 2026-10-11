"""Standalone plots for the completed and independently audited diagnostic.

This module has no raw-data, target-decoding or training imports. Its CLI refuses
to plot a running/incomplete stage or one without a successful independent audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = "results/development/faced_eegnet_diagnostic_v2_2026-10-11"
READOUTS = ("eeg_only", "context_residual")
SEEDS = (20261101, 20261102)
EPOCHS = (30, 60, 120)
LABELS = {"eeg_only": "EEGNet only", "context_residual": "Context + EEGNet residual"}
COLORS = {"eeg_only": "#276CA4", "context_residual": "#14846D"}
CONTROL_LABELS = {"global_median": "global median", "coarse_valence_median": "coarse-valence median", "category_median": "category median", "context_only": "context Ridge", "context_kernel": "context kernel"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_finite(value, *, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError("Invalid finite plot metric")
    return value


def qualified_inputs(packet):
    """Require completed/audited receipts before opening fit-result records."""
    packet = Path(packet)
    if not (packet/"independent_audit.json").is_file():
        raise PermissionError("A completed independent audit is required before plotting")
    execution = load(packet/"execution.json")
    if execution.get("status") != "completed" or execution.get("fits") != 20 or execution.get("confirmation_access") is not False or execution.get("research_question_changed") is not False:
        raise PermissionError("Only the completed fixed development diagnostic may be plotted")
    audit = load(packet/"independent_audit.json")
    if audit.get("passed") is not True or audit.get("fits_audited") != 20 or audit.get("protected_outcomes_opened") is not False:
        raise PermissionError("Successful independent development audit required")
    indexed = {(int(row["rotation"]), row["readout"]): row for row in audit["rotations"]}
    if len(audit["rotations"]) != 6 or set(indexed) != {(r, head) for r in (0, 1, 2) for head in READOUTS}:
        raise ValueError("Incomplete independently audited comparison panels")
    reported = {(int(row["rotation"]), row["readout"]): row for row in execution["rotations"]}
    if set(reported) != set(indexed) or set(execution["selection"]) != set(READOUTS) or set(audit["selection"]) != set(READOUTS):
        raise ValueError("Selection changed after independent audit")
    for head in READOUTS:
        selected = execution["selection"][head]; verified = audit["selection"][head]
        if set(selected) != set(verified) or set(selected) != {"learning_rate", "epoch", "mae"} or selected["learning_rate"] != verified["learning_rate"] or selected["epoch"] != verified["epoch"]:
            raise ValueError("Selected schedule keys/rate/epoch changed after independent audit")
        if not math.isfinite(float(selected["mae"])) or not math.isfinite(float(verified["mae"])) or abs(float(selected["mae"])-float(verified["mae"])) > 2e-9:
            raise ValueError("Selected MAE changed beyond independent arithmetic tolerance")
    for key, row in indexed.items():
        observed = reported[key]
        if row["learning_flags"] != observed["learning_flags"] or row["strongest_context"] != observed["strongest_context"]:
            raise ValueError("Diagnostic flags/comparator changed after audit")
        for metric in ("gain", "joint_mae"):
            if abs(float(row[metric])-float(observed[metric])) > 2e-9:
                raise ValueError("Primary metric changed after audit")
    records = load(packet/"fit_records.json")
    records = records["fits"] if isinstance(records, dict) else records
    if len(records) != 20:
        raise ValueError("Fixed fit record count changed")
    return audit, records, indexed


def save_figure(figure, destination):
    figure.savefig(destination, dpi=190, facecolor="white", bbox_inches="tight")


def comparison_figure(audit, indexed, directory):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    positions = np.arange(3); width = .23
    strongest = []
    for r in range(3):
        first = indexed[r, READOUTS[0]]; second = indexed[r, READOUTS[1]]
        if first["strongest_context"] != second["strongest_context"] or first["control_maes"] != second["control_maes"]:
            raise ValueError("Readouts do not share the same stronger control panel")
        strongest.append(first["control_maes"][first["strongest_context"]])
    bars = [(axes[0].bar(positions-width, strongest, width, color="#747B83", label="Strongest no-EEG control"), strongest)]
    for offset, head in ((0, "eeg_only"), (width, "context_residual")):
        values = [positive_finite(indexed[r, head]["joint_mae"], allow_zero=True) for r in range(3)]
        bars.append((axes[0].bar(positions+offset, values, width, color=COLORS[head], label=LABELS[head]), values))
    for collection, values in bars:
        for rect, value in zip(collection, values):
            axes[0].annotate(f"{value:.3f}", (rect.get_x()+rect.get_width()/2, rect.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(positions, ["Rotation 0", "Rotation 1", "Rotation 2"])
    axes[0].set_ylabel("Participant/family-equal MAE (valence units)")
    axes[0].set_title("Prediction error; lower is better", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    axes[0].set_ylim(0, max(max(values) for _, values in bars)*1.22+.04)
    axes[0].grid(axis="y", alpha=.2); axes[0].set_axisbelow(True)
    for head in READOUTS:
        values = [float(indexed[r, head]["gain"]) for r in range(3)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite gain")
        mean = math.fsum(values)/3
        axes[1].plot(positions, values, marker="o", markersize=6, linewidth=1.8, color=COLORS[head], label=f"{LABELS[head]} (mean {mean:+.3f})")
    axes[1].axhline(0, color="#69717A", linewidth=1)
    axes[1].axhline(.15, color="#A46827", linestyle="--", linewidth=1.3, label="Engineering mean-gain margin: 0.15")
    axes[1].set_xticks(positions, ["Rotation 0", "Rotation 1", "Rotation 2"])
    axes[1].set_ylabel("MAE gain over strongest no-EEG control\n(valence units; positive favors EEG)")
    axes[1].set_title("Exploratory incremental prediction", fontsize=11)
    axes[1].legend(fontsize=8, loc="best", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    axes[1].grid(alpha=.2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    comparator_text = "; ".join(f"rotation {r}: {CONTROL_LABELS[indexed[r, READOUTS[0]]['strongest_context']]}" for r in range(3))
    figure.suptitle("FACED development diagnostic: context and EEG", fontsize=14, y=1.01)
    figure.text(.5, .96, "Exploratory: reused validation participants and material panels; no confirmation outcomes", ha="center", fontsize=9, color="#444A50")
    figure.text(.5, .015, "Continuation also requires positive gain in every rotation and no learning flags. The margin is an engineering rule.", ha="center", fontsize=8)
    figure.text(.5, -.018, "Strongest controls — "+comparator_text, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .07, 1, .94), w_pad=2.8)
    paths = [directory/"diagnostic_comparison.png", directory/"diagnostic_comparison.svg"]
    for path in paths: save_figure(figure, path)
    plt.close(figure)
    return paths


def learning_figure(audit, records, indexed, directory):
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True)
    keyed = {(int(row["rotation"]), row["readout"], float(row["learning_rate"]), int(row["seed"])): row for row in records}
    if len(keyed) != 20:
        raise ValueError("Duplicate fit records")
    source_metrics = []
    for h, head in enumerate(READOUTS):
        selected = audit["selection"][head]; rate = float(selected["learning_rate"])
        for r in range(3):
            axis = axes[h, r]; baseline = None
            for seed, color, marker in ((SEEDS[0], "#276CA4", "o"), (SEEDS[1], "#D0792B", "s")):
                record = keyed[r, head, rate, seed]
                if record.get("status", "completed") != "completed":
                    raise ValueError("Incomplete fit cannot supply a learning curve")
                start = positive_finite(record["starting_control_mse"], allow_zero=True)
                if baseline is not None and abs(start-baseline) > 2e-9:
                    raise ValueError("Seed baseline mismatch")
                baseline = start
                values = [positive_finite(record["checkpoints"][str(epoch)]["training_target_source_mse"], allow_zero=True) for epoch in EPOCHS]
                axis.plot(EPOCHS, values, marker=marker, color=color, linewidth=1.6, label=f"Seed {seed}")
                source_metrics.append({"rotation": r, "readout": head, "learning_rate": rate, "seed": seed, "baseline_mse": baseline, "checkpoint_mse": dict(zip(EPOCHS, values))})
            axis.axhline(baseline, color="#737B84", linestyle="--", linewidth=1.1, label="Offset-only starting control")
            axis.axvline(int(selected["epoch"]), color="#14846D", linestyle=":", linewidth=1.1, label="Chosen checkpoint")
            flags = indexed[r, head]["learning_flags"]
            axis.set_title(f"{LABELS[head]} · rotation {r}\nlr={rate:g}; learning flags: {'present' if flags else 'none'}", fontsize=10)
            axis.set_xticks(EPOCHS)
            axis.set_xlim(23, 127); axis.set_ylim(bottom=0)
            axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)
            if h == 1: axis.set_xlabel("Training epoch")
            if r == 0: axis.set_ylabel("Source eval MSE\n(unbounded valence units²)")
            if h == 0 and r == 2: axis.legend(fontsize=7, frameon=False, loc="best")
    figure.suptitle("Selected EEGNet schedules: two-seed source learning", fontsize=14, y=1.01)
    figure.text(.5, .965, "Checkpoint eval-mode outputs with training offsets; continuous source targets; no clipping", ha="center", fontsize=9, color="#444A50")
    figure.text(.5, .015, "EEG-only offset: source median. Residual offset: participant-OOF context. Curves are not online dropout training loss.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, .95), h_pad=2, w_pad=1.5)
    path = directory/"selected_learning.png"; save_figure(figure, path); plt.close(figure)
    return [path], source_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=REPO/DEFAULT_PACKET)
    args = parser.parse_args(); packet = args.packet.resolve()
    audit, records, indexed = qualified_inputs(packet)
    directory = packet/"figures"; directory.mkdir(exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
        images = comparison_figure(audit, indexed, directory)
        learning, source_metrics = learning_figure(audit, records, indexed, directory)
    receipt = {"status": "completed_post_audit_plots", "inputs": {name: sha(packet/name) for name in ("execution.json", "independent_audit.json", "fit_records.json")}, "code_sha256": sha(Path(__file__)), "images": {str(path.relative_to(packet)).replace("\\", "/"): sha(path) for path in images+learning}, "selected_source_learning": source_metrics,
               "source_mse": "Unbounded original-scale checkpoint eval MSE with regression training offsets; not online dropout loss", "interpretation": "Reused development-panel consistency; engineering margin only; no confirmation evidence", "original_files_opened": False, "new_fits": 0}
    (directory/"plot_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "images": list(receipt["images"])}))


if __name__ == "__main__":
    main()
