"""Render a presentation-only full-width derivative of frozen pairing points.

Reads only the new paper's existing evidence JSON. No scoring, interval
estimation, model fitting, original-data access or protected outcomes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper/exploratory_evaluation_2026-10-12"
FIGURES = PAPER / "figures"
EVIDENCE = PAPER / "evidence.json"
OUTPUTS = [FIGURES / "pairing_publication.pdf", FIGURES / "pairing_publication.png"]
RECEIPT = FIGURES / "figure_layout_receipt.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if any(path.exists() for path in [*OUTPUTS, RECEIPT]):
        raise PermissionError("Preserve existing presentation derivative; no overwrite")
    dependencies = [EVIDENCE, PAPER / "evidence_receipt.json",
                    FIGURES / "pairing_unfamiliar.pdf", FIGURES / "pairing_unfamiliar.png",
                    Path(__file__).resolve()]
    bindings = {p.relative_to(REPO).as_posix(): sha(p) for p in dependencies}
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if evidence.get("status") != "completed_exploratory_public_evidence" or evidence.get("confirmation_access") is not False:
        raise PermissionError("Only completed public exploratory evidence may be displayed")
    anchors = ("joint_spectral", "eegnet_residual")
    schemes = ("person_same_film_acquisition", "film_same_person_coarse_valence")
    selected = [p for p in evidence["pairing_panels"] if p["arm"] == "validation_unseen"]
    keys = [(p["anchor"], p["scheme"], p["rotation"]) for p in selected]
    expected = {(a, s, r) for a in anchors for s in schemes for r in range(3)}
    if len(keys) != 12 or set(keys) != expected:
        raise ValueError("All 12 frozen unfamiliar-film points must be retained")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.25), sharex=True, sharey=True)
    colors = ["#2878a5", "#b06b22", "#526359"]
    points = []
    for ai, anchor in enumerate(anchors):
        for si, scheme in enumerate(schemes):
            ax = axes[ai, si]
            title = ("Spectral + context" if ai == 0 else "EEGNet + context residual")
            relation = ("Other person; same film/acquisition" if si == 0
                        else "Other film; same person/coarse valence")
            ax.set_title(title + "\n" + relation, fontsize=8.1, pad=7, linespacing=1.3)
            ax.axvline(0, color="0.50", lw=.8, ls="--", zorder=1)
            for rotation in range(3):
                panel = next(p for p in selected if
                             (p["anchor"], p["scheme"], p["rotation"]) == (anchor, scheme, rotation))
                estimate = panel["mean_full_delta"]
                interval = panel["full_panel_crossed_interval"]["descriptive_95_percent_interval"]
                lo, hi = interval
                if not lo <= estimate <= hi or not -.27 <= lo <= hi <= .375:
                    raise ValueError("Fixed layout must not truncate a frozen interval")
                ax.errorbar(estimate, 2-rotation, xerr=[[estimate-lo], [hi-estimate]],
                            fmt="o", color=colors[rotation], capsize=2.5, ms=4,
                            lw=1.3, zorder=3)
                points.append({"anchor": anchor, "scheme": scheme, "rotation": rotation,
                               "arm": panel["arm"], "mean_full_delta": estimate,
                               "descriptive_95_percent_interval": interval,
                               "source_path": "pairing_panels", "color": colors[rotation]})
            ax.set_yticks([2, 1, 0], ["R0", "R1", "R2"], fontsize=8.5)
            ax.tick_params(axis="y", length=0, pad=5)
            ax.tick_params(axis="x", labelsize=8.5, length=3)
            ax.set_xlim(-.27, .375)
            ax.set_ylim(-.5, 2.5)
            ax.set_xticks([-.2, -.1, 0, .1, .2, .3])
            ax.grid(axis="x", alpha=.15)
            if ai == 1:
                ax.set_xlabel("Swap loss increase (rating-scale MAE)", fontsize=8.2, labelpad=5)
    fig.subplots_adjust(left=.08, right=.985, bottom=.22, top=.875, hspace=.65, wspace=.24)
    fig.text(.5, .08, "Unfamiliar films; mean of 32 repeat losses", ha="center", fontsize=8.1, color=".25")
    fig.text(.5, .035, "Bars: descriptive 95% crossed person/title intervals", ha="center", fontsize=8.1, color=".25")
    # Preserve exact native dimensions, ensuring full-width text is at least 8 pt.
    fig.savefig(OUTPUTS[0])
    fig.savefig(OUTPUTS[1], dpi=300)
    plt.close(fig)
    after = {p.relative_to(REPO).as_posix(): sha(p) for p in dependencies}
    if after != bindings:
        raise AssertionError("Frozen dependencies changed while rendering")
    receipt = {
        "status": "completed_presentation_only_derivative",
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "input_bindings": bindings,
        "output_bindings": {p.relative_to(REPO).as_posix(): sha(p) for p in OUTPUTS},
        "figure_width_inches": 6.5, "figure_height_inches": 4.25,
        "minimum_text_font_pt": 8.1, "recommended_include_width": "6.5in or unscaled native width",
        "display_points": points, "display_point_count": 12,
        "x_limits": [-.27, .375], "zero_reference": 0,
        "empirical_scores_recomputed": 0, "intervals_recomputed": 0,
        "new_fits": 0, "original_data_access": False, "confirmation_access": False,
        "limits": "Presentation-only copy of existing audited unfamiliar-film point estimates and intervals. Original frozen producer, evidence, figures and receipts remain unchanged."
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "display_points": len(points),
                      "minimum_font_pt": receipt["minimum_text_font_pt"]}))


if __name__ == "__main__":
    main()
