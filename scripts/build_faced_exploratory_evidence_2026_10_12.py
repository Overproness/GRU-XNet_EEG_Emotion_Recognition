"""Build an exploratory paper's public, audited evidence without fitting models.

The one new numerical summary is a frozen retrospective alpha-100 sensitivity.
It is gated by a separately committed declaration; import and --input-bindings
never calculate that comparison or read numeric prediction/target values.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_classical_runner import metrics, paired_diagnostics

BASE = REPO / "results/development"
CATEGORY = BASE / "faced_category_supported_context_2026-10-12"
PAIRING = BASE / "faced_target_context_pairing_2026-10-12"
QUALITY = BASE / "faced_inference_quality_2026-10-11"
PAPER = REPO / "paper/exploratory_evaluation_2026-10-12"
ARMS = ("validation_unseen", "validation_familiar")
DRAW_COUNT, DRAW_SEED, MATCHED_ALPHA = 4000, 425401, 100.0
INPUTS = (
    CATEGORY / "comparison.json", CATEGORY / "source_selection.json",
    CATEGORY / "independent_numerical_audit.json", CATEGORY / "predictions.csv",
    PAIRING / "comparison.json", PAIRING / "target_decomposition.json",
    PAIRING / "independent_numerical_audit.json", PAIRING / "checkpoint_replay.json",
    QUALITY / "predictions.csv", QUALITY / "signal_audit.json",
    BASE / "faced_eegnet_diagnostic_v2_2026-10-11/findings_receipt.json",
    BASE / "faced_backup_2026-10-10/participant_reservation.json",
    BASE / "faced_codebooks_2026-10-10/audit.json",
    BASE / "faced_artifact_control_2026-10-11/source_records.json",
    BASE / "faced_artifact_control_2026-10-11/validation_records.json",
    BASE / "faced_development_protocol_2026-10-11/plan.json",
    REPO / "paper/submission_note_2026-10-12/results_table.json",
    REPO / "paper/submission_note_2026-10-12/results_table.csv",
    REPO / "gruxnet/faced_classical_runner.py",
    REPO / "gruxnet/faced_classical_contract.py",
    Path(__file__).resolve(),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_bindings():
    """Hash-only declaration input; no numeric decoding or outcome summary."""
    return {p.relative_to(REPO).as_posix(): sha(p) for p in INPUTS}


def guard_activation(path):
    d = load(path)
    requirements = {
        "status": "active_exploratory_evidence_build",
        "author_framing_approved": True,
        "confirmation_access": False,
        "new_fits": 0,
        "original_data_access": False,
        "private_array_access": False,
        "matched_alpha": MATCHED_ALPHA,
        "bootstrap_draws": DRAW_COUNT,
        "bootstrap_seed": DRAW_SEED,
        "shared_historical_fallback": True,
        "all_development_panels_retained": True,
    }
    if any(d.get(k) != v for k, v in requirements.items()):
        raise PermissionError("Evidence build requires an explicit matching frozen activation")
    actual = input_bindings()
    if any(d.get("input_bindings", {}).get(k) != v for k, v in actual.items()):
        raise PermissionError("Declaration lacks or differs from an immutable public input binding")
    for audit in (CATEGORY / "independent_numerical_audit.json", PAIRING / "independent_numerical_audit.json"):
        if not load(audit).get("passed"):
            raise PermissionError("Previously audited evidence must pass before reuse")
    return d, actual


def strict_bool(value):
    if str(value).lower() not in {"true", "false"}:
        raise ValueError("Unexpected Boolean encoding")
    return str(value).lower() == "true"


def public_metadata():
    reservation = load(BASE / "faced_backup_2026-10-10/participant_reservation.json")
    source, validation, protected = (set(reservation[k]) for k in
                                    ("development_source", "development_validation", "confirmation"))
    if len(source) != 70 or len(validation) != 20 or len(protected) != 33:
        raise PermissionError("Changed person reservation")
    if source & validation or (source | validation) & protected:
        raise PermissionError("Overlapping person roles")
    catalogue = {x["clip"]: x for x in load(BASE / "faced_codebooks_2026-10-10/audit.json")["stimuli"]}
    records = load(BASE / "faced_artifact_control_2026-10-11/source_records.json")["records"]
    records += load(BASE / "faced_artifact_control_2026-10-11/validation_records.json")["records"]
    metadata = {}
    for record in records:
        pair = record["subject"], int(record["vid"])
        role = "source" if pair[0] in source else "validation" if pair[0] in validation else None
        if pair[0] in protected or role is None or record["role"] != role or pair in metadata:
            raise PermissionError("Protected, duplicate or misassigned metadata identity")
        metadata[pair] = {**catalogue[pair[1]], **record,
                          "family": catalogue[pair[1]]["source_film_sha256"]}
    videos = {p[1] for p in metadata}
    if len(metadata) != 1260 or len(videos) != 14 or set(metadata) != {(s, v) for s in source | validation for v in videos}:
        raise PermissionError("Changed complete development rectangle")
    return load(BASE / "faced_development_protocol_2026-10-11/plan.json"), metadata, validation


def guard_row(row, metadata, validation, panel_pairs):
    """Check role, material, arm and identity before converting any target."""
    pair = row["subject"], int(row["vid"])
    if pair not in metadata or pair[0] not in validation:
        raise PermissionError("Protected or out-of-scope prediction row")
    record = metadata[pair]
    if row["family"] != record["family"] or row["acquisition_group"] != record["acquisition_group"]:
        raise PermissionError("Changed title or acquisition identity")
    if row["arm"] not in ARMS or int(row["rotation"]) not in range(3):
        raise PermissionError("Changed validation panel")
    if pair not in panel_pairs.get((int(row["rotation"]), row["arm"]), set()):
        raise PermissionError("Wrong-arm participant/material identity before numeric conversion")
    return pair


def guarded_numeric(row, metadata, validation, panel_pairs):
    guard_row(row, metadata, validation, panel_pairs)
    target, prediction = float(row["target"]), float(row["prediction"])
    if not np.isfinite([target, prediction]).all() or not 0 <= target <= 7 or not 0 <= prediction <= 7:
        raise ValueError("Invalid original-scale target or clipped prediction")
    return {**row, "vid": int(row["vid"]), "rotation": int(row["rotation"]),
            "target": target, "prediction": prediction,
            "eeg_available": strict_bool(row["eeg_available"])}


def expected_panel_pairs(programme, rotation, arm, validation):
    # The frozen plan explicitly carries the intended person/material arms.
    plan = programme["rotation_manifests"][rotation]
    if plan["rotation"] != rotation:
        raise PermissionError("Frozen rotation identity differs")
    result = {(r["subject"], int(r["vid"])) for r in plan["arms"][arm]}
    if len(result) != (120 if arm == "validation_unseen" else 160) or {p for p, _ in result} != validation:
        raise PermissionError("Incomplete intended participant/material panel")
    return result


def effective_matched_context(context, joint):
    """Keep the recipient metadata/target fixed and share acquisition fallback."""
    identity_fields = ("subject", "vid", "rotation", "arm", "target", "family", "acquisition_group", "eeg_available")
    if any(context[k] != joint[k] for k in identity_fields):
        raise ValueError("Matched prediction target or metadata differs")
    if joint["eeg_available"]:
        if not strict_bool(joint["eeg_used"]) or strict_bool(joint["fallback"]):
            raise ValueError("Available joint prediction must use EEG without fallback")
        return context["prediction"]
    if not strict_bool(joint["fallback"]) or strict_bool(joint["eeg_used"]):
        raise ValueError("Unavailable EEG lacks frozen fallback")
    return joint["prediction"]


def matched_context_sensitivity(programme, metadata, validation):
    panel_pairs = {(rotation, arm): expected_panel_pairs(programme, rotation, arm, validation)
                   for rotation in range(3) for arm in ARMS}
    selected = defaultdict(dict)
    with (QUALITY / "predictions.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["arm"] not in ARMS or row["model"] not in {"context_only", "context_plus_spectral_eeg"}:
                continue
            guard_row(row, metadata, validation, panel_pairs)
            if float(row["alpha"]) != MATCHED_ALPHA:
                continue
            key = int(row["rotation"]), row["arm"], row["model"]
            pair = row["subject"], int(row["vid"])
            if pair in selected[key]:
                raise ValueError("Duplicate matched prediction")
            selected[key][pair] = guarded_numeric(row, metadata, validation, panel_pairs)

    panels, public_rows = [], []
    for rotation in range(3):
        rotation_unavailable = set()
        for arm in ARMS:
            expected = panel_pairs[rotation, arm]
            c = selected[rotation, arm, "context_only"]
            j = selected[rotation, arm, "context_plus_spectral_eeg"]
            if set(c) != expected or set(j) != expected:
                raise ValueError("Matched sensitivity requires every intended validation outcome")
            matched_context = []
            for pair in sorted(expected):
                context, joint = c[pair], j[pair]
                available = joint["eeg_available"]
                # Preserve the historical joint fallback; never rewrite the EEG anchor.
                effective = effective_matched_context(context, joint)
                if not available:
                    rotation_unavailable.add(pair)
                matched_context.append({**context, "prediction": effective})
                public_rows.append({"rotation": rotation, "arm": arm, "subject": pair[0], "vid": pair[1],
                    "family": joint["family"], "acquisition_group": joint["acquisition_group"],
                    "target": joint["target"], "eeg_available": available,
                    "cached_context100_prediction": context["prediction"],
                    "matched_context_prediction": effective, "joint100_prediction": joint["prediction"],
                    "shared_fallback": not available,
                    "paired_gain": abs(joint["target"] - effective) - abs(joint["target"] - joint["prediction"])})
                if not available and public_rows[-1]["paired_gain"] != 0.0:
                    raise AssertionError("Acquisition failures must have zero matched EEG gain")
            diagnostics = paired_diagnostics(matched_context, list(j.values()), DRAW_COUNT, DRAW_SEED)
            panels.append({"rotation": rotation, "arm": arm,
                "matched_context_metrics": metrics(matched_context), "joint_metrics": metrics(list(j.values())),
                "shared_fallback_trials": sum(not r["eeg_available"] for r in j.values()),
                "paired_diagnostics": diagnostics})
        if len(rotation_unavailable) != 15:
            raise AssertionError("Every rotation must retain all 15 unavailable EEG trials")
    if len(public_rows) != 840 or len(panels) != 6:
        raise AssertionError("Retrospective sensitivity count differs from declaration")
    return {"selection_status": "Retrospective fixed-alpha sensitivity; no new fit or selection",
            "alpha": MATCHED_ALPHA, "panels": panels, "paired_prediction_units": public_rows,
            "paired_prediction_units_count": len(public_rows), "method_prediction_rows_count": 2*len(public_rows),
            "limits": "Reused development; alpha 100 fixed after inspecting earlier selections. Available EEG contrasts share contextual inputs, source rows, scaler rule and alpha, but scaling is column-wise in their respective models. Unavailable EEG retains the ORIGINAL selected-context joint fallback in BOTH methods, unlike the prospective category-fallback contract. This is not independent confirmation, a retuned optimum, or a causal EEG-specific effect."}


def fmt(x):
    return f"{x:.3f}"


def interval_text(estimate, interval):
    return f"{estimate:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]"


def table(caption, label, columns, headings, lines, note=""):
    return "\n".join([r"\begin{table}[tbp]", r"\centering", r"\small",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\begin{tabular}{" + columns + "}", r"\toprule",
        " & ".join(headings) + r" \\", r"\midrule", *lines,
        r"\bottomrule", r"\end{tabular}",
        *( [r"\par\medskip\begin{minipage}{0.97\linewidth}\footnotesize " + note + r"\end{minipage}"] if note else []),
        r"\end{table}", ""])


def tables(evidence):
    panels = [p for p in evidence["category_supported_comparison"]["panels"] if p["arm"] == "validation_unseen"]
    matched = [p for p in evidence["matched_context_sensitivity"]["panels"] if p["arm"] == "validation_unseen"]
    names = [("Category median (source selected)", "category_median"),
             ("Category Ridge (source selected)", "category_only"),
             ("Full-context Ridge (source selected)", "full_context"),
             (r"Matched context, $\alpha=100$ (sensitivity)", "matched_context"),
             ("Spectral + context Ridge (frozen)", "joint_spectral"),
             ("EEGNet + context residual (frozen)", "eegnet_residual")]
    lines = []
    for name, key in names:
        values = [p["matched_context_metrics"]["participant_film_mae"] for p in matched] if key == "matched_context" else [p["metrics"][key]["participant_title_mae"] for p in panels]
        lines.append(" & ".join([name] + [fmt(x) for x in values] + [fmt(np.mean(values))]) + r" \\")
    result = {"primary_mae.tex": table(
        "Unfamiliar-film validation MAE on the original 0--7 rating scale (lower is better).",
        "tab:primary", "lrrrr", ["Predictor", "R0", "R1", "R2", "Mean"], lines,
        "Each panel contains 20 held people and six title families; all outcomes, including unavailable EEG, are retained. The mean is a descriptive average of dependent rotations. Category/full-context selection is source-person-only and familiar-title CV; frozen EEG recipes were previously development selected. Matched context is a separately declared retrospective sensitivity with the historical joint fallback shared by both methods on acquisition failures.")}
    lines = []
    for role in ("source", "validation"):
        d=evidence["target_decomposition"][role]
        lines.append(" & ".join([role.capitalize(), str(d["participants"]), str(d["title_families"])] + [f"{100*d['fractions'][k]:.2f}\\%" for k in ("person", "title", "additive_residual")]) + r" \\")
    result["target_decomposition.tex"] = table("Descriptive additive squared-deviation components of retrospective valence.",
        "tab:variation", "lrrrrr", ["Role", "People", "Titles", "Person", "Title", "Residual"], lines,
        "Complete person-by-title rectangles contain one rating per cell. The residual includes person--title interaction and unexplained variation; these fractions do not identify measurement noise, response style, reliability or a prediction floor.")
    lines=[]
    for row in evidence["original_all_conditions"]:
        label = row["condition"].replace("_", r"\_")
        values=[float(row[f"rotation_{i}_mae"]) for i in range(3)]
        lines.append(" & ".join([label]+[fmt(x) for x in values]+[fmt(float(row["mean_mae"]))]) + r" \\")
    result["all_original_conditions.tex"] = table("All historical unfamiliar-film conditions, including adverse results.",
        "tab:allconditions", "lrrrr", ["Historical condition", "R0", "R1", "R2", "Mean"], lines,
        "The first 15 rows retain every declared historical main condition. The final retrospective per-panel context envelope is an outcome-selected diagnostic and cannot be deployed as a source-selected predictor. These recipes are distinct from the later category-supported qualification.")
    lines=[]
    for panel in evidence["category_supported_comparison"]["panels"]:
        for anchor,label in (("joint_spectral","Spectral"),("eegnet_residual","EEGNet")):
            p=panel["eeg_vs_source_selected_context"][anchor]
            lines.append(" & ".join([str(panel["rotation"]), "Unfamiliar" if panel["arm"]=="validation_unseen" else "Familiar",label,
                fmt(panel["metrics"]["category_median"]["participant_title_mae"]), fmt(panel["metrics"][anchor]["participant_title_mae"]),
                interval_text(p["gain"],p["descriptive_95_percent_gain_interval"])]) + r" \\")
    result["category_gain_intervals.tex"] = table("All category-control contrasts and descriptive crossed intervals.",
        "tab:categorygains", "rllrrl", ["R", "Titles", "EEG", "Category MAE", "EEG MAE", "Gain [95\\% interval]"],lines,
        "Gain is category-control MAE minus EEG-recipe MAE. It compares whole prediction recipes with different metadata/readout forms. Intervals use 4,000 crossed person/title resamples (seed 425401), reused development and no selection correction. An interval including zero does not establish equivalence or lack of EEG information.")
    lines=[]
    for panel in evidence["matched_context_sensitivity"]["panels"]:
        d=panel["paired_diagnostics"]
        lines.append(" & ".join([str(panel["rotation"]), "Unfamiliar" if panel["arm"]=="validation_unseen" else "Familiar",
            fmt(panel["matched_context_metrics"]["participant_film_mae"]),fmt(panel["joint_metrics"]["participant_film_mae"]),
            str(panel["shared_fallback_trials"]), interval_text(d["gain"], d["descriptive_95_percent_interval"])]) + r" \\")
    result["matched_context_sensitivity.tex"] = table("Retrospective fixed-alpha matched-context sensitivity; no new fits.",
        "tab:matched", "rlrrrl", ["R","Titles","Context MAE","Joint MAE","Fallbacks","Gain [95\\% interval]"],lines,
        "Available trials use cached context-only and joint spectral Ridge predictions at alpha 100, the same source identities, 13 contextual columns and source scaling rule. On every unavailable EEG trial, both methods use the unchanged historical joint fallback; paired gain is exactly zero. The declared comparison includes every familiar and unfamiliar outcome. Development reuse and retrospective choice of alpha remain limitations.")
    lines=[]
    for p in evidence["pairing_panels"]:
        d=p["full_panel_crossed_interval"]
        lines.append(" & ".join([str(p["rotation"]),"Unfamiliar" if p["arm"]=="validation_unseen" else "Familiar",
            "Spectral" if p["anchor"]=="joint_spectral" else "EEGNet",
            "Other person" if p["scheme"]=="person_same_film_acquisition" else "Other film",
            f"{p['eligible_trials']}/{p['original_full']['trials']}",interval_text(p["mean_full_delta"],d["descriptive_95_percent_interval"])]) + r" \\")
    result["all_pairing_intervals.tex"] = table("All pairing controls, including familiar-film panels.",
        "tab:allpairing", "rlllll", ["R","Titles","EEG","Donor","Eligible","Loss increase [95\\% interval]"],lines,
        "Recipient context and target stay fixed. Other-person donors share film/acquisition; other-film donors share participant/coarse valence. The 32 repeat losses are averaged before scoring, with unavailable recipients and singleton strata unperturbed. Intervals use the full rectangle, not the nonrectangular eligible subset. All 24 intervals contain zero; no causal or formal permutation-null interpretation is justified.")
    lines=[]
    selection=evidence["source_selection"]
    for model in ("global_median","coarse_valence_median","category_median","category_only","full_context"):
        values=[selection["strongest_comparators"][str(i)]["source_person_cv_mae"][model] for i in range(3)]
        lines.append(" & ".join([model.replace("_",r"\_")]+[fmt(x) for x in values]) + r" \\")
    result["source_selection.tex"] = table("Category-supported source-person CV used for contextual qualification.",
        "tab:sourcecv","lrrr",["Context candidate","R0","R1","R2"],lines,
        "Five fixed folds train on 56 source people and assess 14, preserving all eight source titles/categories. Category median wins in every rotation; category and full-context Ridge select alpha 0.1. These folds assess held people on familiar source titles and do not independently qualify unseen-title transfer. Earlier crossed folds omitted the held titles' categories and triggered unsupported-category fallback in every held cell.")
    return result


def pairing_figure(panels, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.spines.top":False, "axes.spines.right":False, "pdf.fonttype":42})
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.2),sharex=True,sharey=True)
    colors=["#2878a5","#b06b22","#526359"]
    chosen=[p for p in panels if p["arm"]=="validation_unseen"]
    bounds=[x for p in chosen for x in p["full_panel_crossed_interval"]["descriptive_95_percent_interval"]]
    padding=0.04; limits=(min(-.04,min(bounds)-padding),max(.04,max(bounds)+padding))
    for ai,anchor in enumerate(("joint_spectral","eegnet_residual")):
        for si,scheme in enumerate(("person_same_film_acquisition","film_same_person_coarse_valence")):
            ax=axes[ai,si]
            title=("Spectral + context" if ai==0 else "EEGNet + context residual")+"\n"+("Other person; same film / acquisition" if si==0 else "Other film; same person / coarse valence")
            ax.set_title(title,fontsize=10.5,pad=10)
            ax.axvline(0,color="0.50",lw=1,ls="--",zorder=1)
            for ri in range(3):
                p=next(p for p in chosen if p["anchor"]==anchor and p["scheme"]==scheme and p["rotation"]==ri)
                gain=p["mean_full_delta"];lo,hi=p["full_panel_crossed_interval"]["descriptive_95_percent_interval"]
                ax.errorbar(gain,2-ri,xerr=[[gain-lo],[hi-gain]],fmt="o",color=colors[ri],capsize=3,ms=5,lw=1.6,zorder=3)
            ax.set_yticks([2,1,0],["Rotation 0","Rotation 1","Rotation 2"])
            ax.tick_params(axis="y",length=0,pad=7)
            ax.set_xlim(*limits);ax.set_ylim(-.5,2.5);ax.grid(axis="x",alpha=.15)
            if ai==1:ax.set_xlabel("Swap loss increase (rating-scale MAE)")
    fig.subplots_adjust(left=.115,right=.98,bottom=.15,top=.91,hspace=.48,wspace=.2)
    fig.text(.5,.035,"Unfamiliar films · mean of 32 repeat losses · descriptive 95% crossed-person/title intervals",ha="center",fontsize=9,color=".25")
    fig.savefig(directory/"pairing_unfamiliar.pdf",bbox_inches="tight")
    fig.savefig(directory/"pairing_unfamiliar.png",dpi=240,bbox_inches="tight")
    plt.close(fig)


def build(activation_path):
    activation,bindings=guard_activation(activation_path)
    if (PAPER/"evidence.json").exists() or (PAPER/"evidence_receipt.json").exists():
        raise PermissionError("Preserve existing evidence; no overwrite of a completed attempt")
    programme,metadata,validation=public_metadata()
    matched=matched_context_sensitivity(programme,metadata,validation)
    comparison=load(CATEGORY/"comparison.json")
    pairing=load(PAIRING/"comparison.json")
    # Reuse existing exact audited summaries, not new opportunistic subset scores.
    with (REPO/"paper/submission_note_2026-10-12/results_table.csv").open(newline="",encoding="utf-8") as f:
        original=list(csv.DictReader(f))
    if len(original)!=16:
        raise AssertionError("All 15 historical conditions and the post-hoc envelope must remain")
    evidence={"status":"completed_exploratory_public_evidence",
        "built_utc":datetime.now(timezone.utc).isoformat(),
        "framing":"Context controls and EEG pairing in retrospective FACED valence prediction: an exploratory evaluation",
        "confirmation_access":False,"new_fits":0,"original_data_access":False,"private_array_access":False,
        "category_supported_comparison":comparison,
        "source_selection":{k:v for k,v in load(CATEGORY/"source_selection.json").items() if k!="source_person_folds"},
        "target_decomposition":{role:{k:v for k,v in d.items() if k not in {"person_means","title_means"}} for role,d in load(PAIRING/"target_decomposition.json").items()},
        "pairing_panels":[{k:v for k,v in p.items() if k not in {"repeats","per_family"}} for p in pairing["pairing_panels"]],
        "original_all_conditions":original,
        "neural_learning":load(BASE/"faced_eegnet_diagnostic_v2_2026-10-11/findings_receipt.json"),
        "source_checkpoint_replay":load(PAIRING/"checkpoint_replay.json"),
        "signal_quality":load(QUALITY/"signal_audit.json"),
        "matched_context_sensitivity":matched,
        "activation_sha256":sha(activation_path),"input_bindings":bindings,
        "interpretation_limits":[
            "Exploratory adaptive reuse of development outcomes; no independent confirmation or correction for model selection.",
            "Three rotations share people and material; their arithmetic mean is descriptive, without a pooled independent confidence interval.",
            "Known stimulus-category information contract; one source title per category confounds title, category and duration.",
            "Retrospective item-10 valence, late 32-second native interval/28-second core; no full-video or momentary emotion inference.",
            "Single rating per person/title cannot identify reliability, measurement noise, response style or an irreducible prediction floor.",
            "Neural checkpoint-to-window qualification covers source fixtures; validation corrections are cached, not independently forward-replayed.",
            "An interval including zero does not establish equivalence or absence of useful EEG information; pairing does not identify causal emotion mechanisms."]}
    (PAPER/"tables").mkdir(parents=True,exist_ok=True)
    (PAPER/"figures").mkdir(parents=True,exist_ok=True)
    outputs=[]
    p=PAPER/"evidence.json";p.write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n",encoding="utf-8");outputs.append(p)
    for name,content in tables(evidence).items():
        p=PAPER/"tables"/name;p.write_text(content,encoding="utf-8");outputs.append(p)
    pairing_figure(evidence["pairing_panels"],PAPER/"figures")
    outputs += [PAPER/"figures/pairing_unfamiliar.pdf",PAPER/"figures/pairing_unfamiliar.png"]
    # Rehash immutable inputs after all processing and plotting; detect in-flight changes.
    if input_bindings()!=bindings:
        raise AssertionError("Public inputs changed during evidence build")
    receipt={"status":"completed_exploratory_public_evidence","built_utc":evidence["built_utc"],
        "activation_sha256":sha(activation_path),"input_bindings":bindings,
        "output_bindings":{p.relative_to(REPO).as_posix():sha(p) for p in outputs},
        "matched_panels":6,"paired_prediction_units":840,"method_prediction_rows":1680,
        "new_fits":0,"confirmation_access":False,"original_data_access":False,"private_array_access":False,
        "independent_numeric_audit_pending":True,
        "limits":"Producer receipt and reused prior audits; new matched sensitivity requires a separate independent numerical audit."}
    (PAPER/"evidence_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":receipt["status"],"output_files":len(outputs),"matched_panels":6,"new_fits":0}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-bindings",action="store_true")
    mode.add_argument("--build",action="store_true")
    parser.add_argument("--activation",type=Path)
    args=parser.parse_args()
    if args.input_bindings:
        print(json.dumps(input_bindings(),indent=2))
    elif args.activation is None:
        parser.error("--build requires --activation")
    else:
        build(args.activation.resolve())


if __name__=="__main__":
    main()
