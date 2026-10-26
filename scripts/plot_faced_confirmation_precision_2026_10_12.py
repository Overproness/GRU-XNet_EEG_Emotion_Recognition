"""Plot the completed declared synthetic precision summary, without simulation.

Reads only precision_plan.json, precision_summary.json and precision_receipt.json.
No empirical EEG, rating, cached prediction, original metadata or model is read.
The fixed gain 0.045 is a sensitivity assumption, not empirical power calibration.
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
from matplotlib.ticker import PercentFormatter
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = REPO / "results/development/faced_confirmation_freeze_2026-10-12"
ASSUMED_GAIN = 0.045
DIMENSIONS = ((33, 8), (66, 8), (33, 16), (66, 16))
SCENARIOS = ("low", "participant_dominant", "title_dominant", "balanced")
COLORS = ("#203C66", "#4F8CBF", "#E2A342", "#5A9A7A")
DESIGN_LABELS = (
    "33 people × 8 families: reserved dimensions",
    "66 people × 8 families: hypothetical",
    "33 people × 16 families: hypothetical",
    "66 people × 16 families: hypothetical",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_summary(packet):
    """Authenticate the declared summary and select exactly 16 synthetic rows."""
    plan_path = packet / "precision_plan.json"
    summary_path = packet / "precision_summary.json"
    receipt_path = packet / "precision_receipt.json"
    plan, summary, receipt = [json.loads(p.read_text(encoding="utf-8"))
                              for p in (plan_path, summary_path, receipt_path)]
    if (receipt.get("status") != "completed_synthetic_only"
            or receipt.get("plan_sha256") != sha256(plan_path)
            or receipt["output_bindings"].get("precision_summary.json") != sha256(summary_path)
            or receipt.get("confirmation_access") is not False
            or receipt.get("cached_target_or_prediction_access") is not False
            or receipt.get("new_empirical_fits") != 0):
        raise ValueError("Require the authenticated completed synthetic-only packet")
    metadata = plan["metadata_projection"]
    if (metadata["protected_participants"] != 33 or metadata["protected_title_families"] != 8
            or metadata["protected_nominal_trials"] != 297
            or metadata["protected_person_title_cells"] != 264
            or metadata["empirical_value_access"] is not False):
        raise ValueError("Reserved metadata dimensions changed")
    conditions = plan["conditions"]
    if (tuple((d["subjects"], d["titles"]) for d in conditions["dimensions"]) != DIMENSIONS
            or tuple(s["name"] for s in conditions["variance_scenarios"]) != SCENARIOS
            or conditions["true_gains"] != [0.0, ASSUMED_GAIN, 0.15]
            or conditions["replicates_per_scenario_dimension"] != 256
            or conditions["crossed_bootstrap_draws_per_replicate"] != 512):
        raise ValueError("Synthetic plot conditions differ from the declaration")
    rows = summary["rows"]
    identities = [(r["scenario"], r["subjects"], r["titles"], r["true_gain"]) for r in rows]
    expected = {(scenario, n, f, gain) for scenario in SCENARIOS
                for n, f in DIMENSIONS for gain in conditions["true_gains"]}
    if (summary.get("schema") != "synthetic_crossed_precision_summary_v1"
            or len(rows) != 48 or len(set(identities)) != 48 or set(identities) != expected):
        raise ValueError("Incomplete or duplicated synthetic condition summary")
    selected = {(r["scenario"], r["subjects"], r["titles"]): r
                for r in rows if r["true_gain"] == ASSUMED_GAIN}
    if len(selected) != 16:
        raise ValueError("Exactly 16 conditions required for the assumed gain")
    for (_, n, f), row in selected.items():
        count, reps = row["positive_lower_endpoint_count"], row["replicates"]
        if (reps != 256 or type(count) is not int or not 0 <= count <= reps
                or not math.isfinite(row["mean_interval_width"])
                or row["mean_interval_width"] <= 0
                or ((n, f) == DIMENSIONS[0]
                    and row["dimension_status"] != "actual_reserved_dimensions_only")
                or ((n, f) != DIMENSIONS[0] and not row["dimension_status"].startswith("hypothetical"))):
            raise ValueError("Invalid declared synthetic plot row")
    return plan, selected


def render(packet, output, *, replace=False):
    plan, selected = load_summary(packet)
    output.mkdir(parents=True, exist_ok=True)
    paths = {suffix: output / ("precision_sensitivity." + suffix) for suffix in ("png", "svg")}
    receipt_path = output / "precision_plot_receipt.json"
    if not replace and any(path.exists() for path in [*paths.values(), receipt_path]):
        raise FileExistsError("Preserve existing figures; replace only provisional outputs before freeze")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.1))
    fig.subplots_adjust(left=.065, right=.985, top=.715, bottom=.235, wspace=.19)
    x = np.arange(len(SCENARIOS))
    bar_width = .18
    max_width = 0.0
    plot_rows = []
    for design_index, (people, titles) in enumerate(DIMENSIONS):
        rows = [selected[(scenario, people, titles)] for scenario in SCENARIOS]
        widths = np.asarray([r["mean_interval_width"] for r in rows])
        fractions = np.asarray([r["positive_lower_endpoint_count"] / r["replicates"] for r in rows])
        # The reported zero-exclusion MCSE includes adverse negative intervals.
        # Recompute only the positive-count binomial MCSE from summary counts.
        mcse = np.sqrt(fractions * (1 - fractions) / 256)
        position = x + (design_index - 1.5) * bar_width
        axes[0].bar(position, widths, width=bar_width, color=COLORS[design_index],
                    label=DESIGN_LABELS[design_index], linewidth=.5, edgecolor="white")
        axes[1].bar(position, fractions, width=bar_width, color=COLORS[design_index],
                    yerr=mcse, capsize=2.5, error_kw={"elinewidth": .8, "ecolor": "#263442"},
                    linewidth=.5, edgecolor="white")
        max_width = max(max_width, float(widths.max()))
        plot_rows.extend({"scenario": r["scenario"], "subjects": people, "titles": titles,
                          "assumed_gain": ASSUMED_GAIN, "replicates": 256,
                          "mean_interval_width": float(width), "positive_lower_endpoint_count": r["positive_lower_endpoint_count"],
                          "positive_lower_endpoint_fraction": float(fraction),
                          "positive_count_binomial_monte_carlo_se": float(error)}
                         for r, width, fraction, error in zip(rows, widths, fractions, mcse))
    scenario_names = ("Low", "Participant dominant", "Title dominant", "Balanced")
    scenario_ticks = []
    for name, declaration in zip(scenario_names, plan["conditions"]["variance_scenarios"]):
        sds = tuple(declaration[key] for key in ("subject_sd", "title_sd", "interaction_sd"))
        scenario_ticks.append(name + "\n(" + ", ".join(f"{v:.2f}" for v in sds) + ")")
    for axis in axes:
        axis.set_xticks(x, scenario_ticks)
        axis.tick_params(axis="x", labelsize=10, length=0, pad=10)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D8DEE5", linewidth=.7)
        axis.set_xlim(-.55, len(SCENARIOS) - .45)
        axis.set_xlabel("Assumed variance scenario", labelpad=12)
    axes[0].set_title("A  Mean synthetic 95% interval width", loc="left", pad=15)
    axes[0].set_ylabel("Width in paired MAE gain (rating points)")
    axes[0].set_ylim(0, max_width * 1.14)
    axes[1].set_title("B  Synthetic intervals with lower endpoint > 0", loc="left", pad=15)
    axes[1].set_ylabel("Fraction of synthetic replicates")
    axes[1].set_ylim(0, 1.0)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle("Synthetic sensitivity, assumed variance; no empirical FACED power",
                 fontsize=15, fontweight="bold", x=.5, y=.96)
    fig.text(.5, .90, "Assumed paired gain: 0.045 rating points  |  256 replicates per condition  |  512 crossed resamples per replicate",
             ha="center", fontsize=10.5, color="#3A4653")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .865),
               ncol=2, frameon=False, fontsize=10.5, columnspacing=2.8, handlelength=1.5)
    fig.text(.065, .10, "Scenario numbers are assumed SDs: person, catalogue title, interaction (rating points).",
             fontsize=10, color="#3A4653")
    fig.text(.065, .063, "Gaussian loss-difference model; variance is not fitted to FACED. Eight catalogue families are a fixed roster; larger designs are hypothetical.",
             fontsize=10, color="#3A4653")
    fig.text(.065, .026, "Error bars in B: ±1 binomial Monte Carlo standard error for positive lower endpoints. These results neither establish empirical power nor unlock confirmation.",
             fontsize=9.5, color="#3A4653")
    fig.savefig(paths["png"], dpi=180, facecolor="white")
    fig.savefig(paths["svg"], facecolor="white", metadata={"Date": "2026-10-12",
                "Description": "Declared synthetic sensitivity only; no empirical FACED power"})
    plt.close(fig)
    receipt = {
        "status": "rendered_declared_synthetic_summary_only",
        "input_bindings": {name: sha256(packet / name) for name in
                           ("precision_plan.json", "precision_summary.json", "precision_receipt.json")},
        "script_sha256": sha256(Path(__file__)),
        "output_bindings": {path.name: sha256(path) for path in paths.values()},
        "source_summary_rows": 48, "plotted_condition_rows": 16,
        "assumed_gain": ASSUMED_GAIN, "plotted_rows": plot_rows,
        "new_simulations": 0, "new_empirical_fits": 0,
        "empirical_values_read": False, "confirmation_access": False,
        "limits": "Sensitivity under analyst-chosen Gaussian variance only; not calibrated FACED power. Actual dimensions and hypothetical designs distinguished.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"plots_created": True, "plotted_synthetic_conditions": 16,
                      "new_simulations": 0, "confirmation_access": False,
                      "png": str(paths["png"]), "svg": str(paths["svg"])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true",
                        help="Replace only this script's provisional outputs before root freezes them")
    args = parser.parse_args()
    render(args.packet, args.output or args.packet / "figures", replace=args.replace)
