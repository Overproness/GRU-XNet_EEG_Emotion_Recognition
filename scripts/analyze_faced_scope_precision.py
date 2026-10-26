"""Review public marker metadata and existing predictions; never load data or fit.

See the committed analysis_declaration.md. Planning calculations are descriptive,
conditional on unverified variance assumptions, and not validated study power.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, t

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
PACKET = BASE / "faced_scope_precision_2026-10-11"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bindings() -> None:
    bindings = load(PACKET / "input_bindings.json")
    for rel, expected in bindings["files"].items():
        path = (REPO / rel).resolve()
        if not path.is_relative_to(BASE.resolve()):
            raise ValueError("Input escaped public development directory")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Frozen input changed: {rel}")
    if hashlib.sha256((PACKET / "analysis_declaration.md").read_bytes()).hexdigest() != bindings["declaration_sha256"]:
        raise ValueError("Analysis declaration changed")


def guard_row(row: dict, allowed_people: set, catalogue: dict, allowed_clips: set) -> None:
    """Check identities before touching either outcome string, even for skipped arms."""
    if row["subject"] not in allowed_people:
        raise ValueError("Protected or unknown participant in public prediction input")
    vid = int(row["vid"])
    if vid not in allowed_clips:
        raise ValueError("Protected or excluded material in public prediction input")
    if row["family"] != catalogue[vid]["source_film_sha256"]:
        raise ValueError("Clip/family identity mismatch")


def read_rows(path: Path, allowed_people: set, catalogue: dict, allowed_clips: set) -> list:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        guard_row(row, allowed_people, catalogue, allowed_clips)
    return rows


def selected(rows: list, rotation: int, model: str, alpha: float | None = None) -> dict:
    result = {}
    for row in rows:
        if int(row["rotation"]) != rotation or row["model"] != model or row["arm"] != "validation_unseen":
            continue
        if alpha is not None and float(row["alpha"]) != alpha:
            continue
        key = (row["subject"], int(row["vid"]))
        if key in result:
            raise ValueError("Duplicate selected trial")
        target, prediction = float(row["target"]), float(row["prediction"])
        if not math.isfinite(target) or not 0 <= target <= 7 or not math.isfinite(prediction):
            raise ValueError("Invalid target or prediction")
        result[key] = (row["family"], target, prediction, row["fallback"])
    return result


def paired_matrix(context: dict, joint: dict, people: list, families: list) -> tuple:
    if set(context) != set(joint) or len(context) != 120:
        raise ValueError("Paired denominator mismatch")
    cells = {(person, family): [] for person in people for family in families}
    context_errors, joint_errors = [], []
    fallbacks = 0
    for key, (family, target, pred, _) in context.items():
        other_family, other_target, other_pred, fallback = joint[key]
        if family != other_family or abs(target - other_target) > 1e-12:
            raise ValueError("Paired target/family mismatch")
        cell = (key[0], family)
        if cell not in cells:
            raise ValueError("Trial outside fixed panel")
        ce, je = abs(target - pred), abs(target - other_pred)
        cells[cell].append(ce - je)
        context_errors.append(ce)
        joint_errors.append(je)
        fallbacks += fallback.lower() in {"true", "1"}
    if any(len(values) != 1 for values in cells.values()):
        raise ValueError("Existing panel is not complete 20 by six one-clip families")
    matrix = np.array([[np.mean(cells[p, f]) for f in families] for p in people])
    return matrix, float(np.mean(context_errors)), float(np.mean(joint_errors)), fallbacks


def variance_components(matrix: np.ndarray) -> dict:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Variance analysis requires a finite balanced matrix")
    ni, nj = x.shape
    grand, row, col = x.mean(), x.mean(axis=1), x.mean(axis=0)
    residual = x - row[:, None] - col[None, :] + grand
    msp = nj * np.square(row - grand).sum() / (ni - 1)
    msf = ni * np.square(col - grand).sum() / (nj - 1)
    mse = np.square(residual).sum() / ((ni - 1) * (nj - 1))
    raw = {"participant": float((msp - mse) / nj), "film": float((msf - mse) / ni), "residual": float(mse)}
    return {
        "mean_gain": float(grand), "shape": [ni, nj],
        "mean_squares": {"participant": float(msp), "film": float(msf), "residual": float(mse)},
        "raw_components": raw,
        "planning_components": {k: max(0.0, v) for k, v in raw.items()},
        "participant_mean_sample_sd": float(row.std(ddof=1)),
        "leave_one_film_gain": [float(np.delete(x, j, axis=1).mean()) for j in range(nj)],
    }


def planning_se(components: dict, participants: int, films: int, inflation: float = 1) -> float:
    if participants <= 0 or films <= 0 or inflation <= 0:
        raise ValueError("Invalid planning geometry")
    return math.sqrt(inflation * (components["participant"] / participants + components["film"] / films + components["residual"] / (participants * films)))


def planning(components: dict, participant_sd: float) -> dict:
    multiplier = float(norm.ppf(.975) + norm.ppf(.8))
    se = planning_se(components, 33, 8)
    margin_rows = []
    for margin in [.05, .10, .15, .25]:
        required_var = (margin / multiplier) ** 2
        remaining = required_var - components["film"] / 8
        numerator = components["participant"] + components["residual"] / 8
        n = None if remaining <= 0 else max(2, math.ceil(numerator / remaining))
        margin_rows.append({"margin": margin, "nominal_required_participants_at_eight_films": n,
                            "film_floor_prevents_finite_n": remaining <= 0})
    return {
        "status": "illustrative_unvalidated_not_actual_power",
        "participants": 33, "films": 8, "crossed_se": se,
        "normal_95_half_width": float(norm.ppf(.975) * se),
        "normal_80_percent_detectable_effect": multiplier * se,
        "infinite_participant_se_at_eight_films": math.sqrt(components["film"] / 8),
        "infinite_participant_detectable_effect_at_eight_films": multiplier * math.sqrt(components["film"] / 8),
        "fixed_roster_participant_se": participant_sd / math.sqrt(33),
        "fixed_roster_t32_95_half_width": float(t.ppf(.975, 32) * participant_sd / math.sqrt(33)),
        "margin_sensitivity": margin_rows,
        "participants_sensitivity": [{"participants": n, "films": 8, "detectable_effect": multiplier * planning_se(components, n, 8)} for n in [20, 33, 60, 120, 1000]],
        "films_sensitivity": [{"participants": 33, "films": j, "detectable_effect": multiplier * planning_se(components, 33, j)} for j in [8, 16, 32, 64]],
        "variance_inflation": [{"inflation": f, "crossed_se": planning_se(components, 33, 8, f), "detectable_effect": multiplier * planning_se(components, 33, 8, f)} for f in [1, 2, 4]],
    }


def scope_audit(catalogue: dict, design: dict, roles: dict) -> dict:
    joins = load(BASE / "faced_raw_2026-10-11/joins_numeric_projection.json")["records"]
    if len(joins) != 123 or len({r["subject"] for r in joins}) != 123:
        raise ValueError("Marker participant geometry changed")
    all_people = set().union(*(set(roles[k]) for k in ["development_source", "development_validation", "confirmation"]))
    if {r["subject"] for r in joins} != all_people:
        raise ValueError("Marker identities do not match reservation")
    excluded = {13, 14, 15, 16, 22}
    deltas, lengths, raw_margins, core_margins, clip22 = [], [], [], [], []
    for r in joins:
        if not r["passed"] or not r["all_endpoints_in_data_geometry"] or {s["vid"] for s in r["spans"]} != set(range(1,29)):
            raise ValueError("Numeric marker audit failed")
        for s in r["spans"]:
            length = s["end_seconds"] - s["start_seconds"]
            if s["vid"] == 22:
                clip22.append(length)
            if s["vid"] in excluded:
                continue
            lengths.append(length)
            deltas.append(length - catalogue[s["vid"]]["duration_seconds"])
            raw_margins.append(length - 32)
            core_margins.append(length - 30)
    if min(raw_margins) <= 0 or len(lengths) != 123*23:
        raise ValueError("Input interval is not contained in included marker spans")
    confirm = design["rotations"][0]["families"]["confirmation"]
    if len(confirm) != 8 or sum(len(f["clips"]) for f in confirm) != 9:
        raise ValueError("Confirmation material geometry changed")
    if any(rotation["families"]["confirmation"] != confirm for rotation in design["rotations"]):
        raise ValueError("Protected material roles changed across rotations")
    raw_summary = load(BASE / "faced_raw_2026-10-11/summary.json")
    native_floor_bound = 1 / min(float(fs) for fs in raw_summary["sampling_frequency_participants"])
    return {
        "participants": 123, "numeric_marker_spans": 3444,
        "included_marker_spans": len(lengths), "minimum_included_marker_duration_seconds": min(lengths),
        "marker_minus_catalogue_seconds_range_included": [min(deltas),max(deltas)],
        "minimum_32s_input_start_margin_from_start_marker_seconds": min(raw_margins),
        "minimum_28s_core_start_margin_from_start_marker_seconds": min(core_margins),
        "native_integer_endpoint_floor_maximum_lag_bound_seconds": native_floor_bound,
        "minimum_32s_input_start_margin_after_endpoint_floor_lower_bound_seconds": min(raw_margins) - native_floor_bound,
        "minimum_core_start_margin_after_endpoint_floor_lower_bound_seconds": min(core_margins) - native_floor_bound,
        "core_end_margin_from_end_marker_seconds": 2,
        "clip22_observed_marker_duration_seconds_range": [min(clip22),max(clip22)],
        "excluded_clips": sorted(excluded),
        "reserved_participants": len(roles["confirmation"]), "reserved_title_families": len(confirm),
        "reserved_clips": sorted(v for f in confirm for v in f["clips"]),
        "clip_7_8_share_one_family": any(f["clips"] == [7,8] for f in confirm),
        "nominal_confirmation_new_film_trials": 33*9, "nominal_confirmation_participant_family_cells": 33*8,
        "participant_and_material_roles_changed": False,
        "marker_geometry_only": True, "physical_playback_latency_authenticated": False,
        "media_hashes_or_exact_footage_independence_authenticated": False,
        "grouping_scope": "Exact normalized catalogue source title, not physical content or semantic independence",
    }


def analyze() -> dict:
    validate_bindings()
    roles = load(BASE / "faced_backup_2026-10-10/participant_reservation.json")
    design = load(BASE / "faced_design_proposal_2026-10-11/recommended_proposal.json")
    catalogue = {r["clip"]: r for r in load(BASE / "faced_backup_2026-10-10/stimulus_metadata.json")["records"]}
    people = sorted(roles["development_validation"])
    allowed_people = set(roles["development_source"]) | set(people)
    allowed_clips = set(design["rotations"][0]["clips"]["development_source"]) | set(design["rotations"][0]["clips"]["development_validation"])
    quality = load(BASE / "faced_inference_quality_2026-10-11/comparison.json")
    representation = load(BASE / "faced_representation_v2_2026-10-11/comparison.json")
    qr = read_rows(BASE / "faced_inference_quality_2026-10-11/predictions.csv", allowed_people, catalogue, allowed_clips)
    rr = read_rows(BASE / "faced_representation_v2_2026-10-11/predictions.csv", allowed_people, catalogue, allowed_clips)
    results = []
    for rotation in range(3):
        old = quality["rotations"][rotation]["selected"]
        panels = {
            "ridge_context": selected(qr,rotation,"context_only",old["context_only"]["alpha"]),
            "spectral_joint": selected(qr,rotation,"context_plus_spectral_eeg",old["context_plus_spectral_eeg"]["alpha"]),
            "kernel_context": selected(rr,rotation,"context_kernel"),
            "covariance_joint": selected(rr,rotation,"context_covariance_kernel"),
        }
        families = sorted(f["family"] for f in design["rotations"][rotation]["families"]["development_validation"])
        for name, ctx, joint in [
            ("spectral_joint_vs_ridge_context","ridge_context","spectral_joint"),
            ("covariance_joint_vs_kernel_context","kernel_context","covariance_joint"),
            ("covariance_joint_vs_ridge_context","ridge_context","covariance_joint"),
        ]:
            matrix, cmae, jmae, fallback = paired_matrix(panels[ctx],panels[joint],people,families)
            for model, value in [(ctx,cmae),(joint,jmae)]:
                if model in {"ridge_context","spectral_joint"}:
                    key = "context_only" if model == "ridge_context" else "context_plus_spectral_eeg"
                    expected = old[key]["validation_unseen"]["participant_film_mae"]
                else:
                    key = "context_kernel" if model == "kernel_context" else "context_covariance_kernel"
                    expected = representation["rotations"][rotation]["models"][key]["metrics"]["validation_unseen"]["participant_film_mae"]
                if abs(value - expected) > 1e-12:
                    raise ValueError("Prior aggregate metric disagreement")
            components = variance_components(matrix)
            results.append({"contrast":name,"rotation":rotation,"participants":people,"families":families,
                            "context_mae":cmae,"joint_mae":jmae,"joint_fallback_trials":fallback,
                            **components,"planning":planning(components["planning_components"],components["participant_mean_sample_sd"])})
    return {
        "status":"exploratory_existing_prediction_planning_only",
        "input_rows_guarded":{"quality":len(qr),"representation":len(rr)},
        "contrasts":results,"scope":scope_audit(catalogue,design,roles),
        "limitations":["Three overlapping rotations are not independent repetitions", "Six development films give imprecise and potentially negative variance components", "Truncation is a planning convention, not a corrected estimator", "Exchangeable additive random effects are unverified", "Final-model training and selection uncertainty omitted", "Category coverage differs: development unseen panels omit Fear/Inspiration, reserved roster includes both", "No projected interval or power is validated on protected data"],
        "new_models_fitted":0,"fresh_original_ratings_decoded":0,"waveforms_read":0,
        "confirmation_opened":False,"research_question_changed":False,
    }


def plot(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,4.3),sharey=True)
    names = ["spectral_joint_vs_ridge_context","covariance_joint_vs_kernel_context","covariance_joint_vs_ridge_context"]
    titles = ["Spectral joint / Ridge context", "Covariance joint / kernel context", "Covariance joint / Ridge context"]
    counts = np.geomspace(20,2000,150)
    multiplier = float(norm.ppf(.975) + norm.ppf(.8))
    for ax,name,title in zip(axes,names,titles):
        for entry in result["contrasts"]:
            if entry["contrast"] != name:
                continue
            vals = [multiplier*planning_se(entry["planning_components"],float(n),8) for n in counts]
            ax.plot(counts,vals,label=f"Development rotation {entry['rotation']}",lw=1.8)
        ax.axhline(.15,color="black",ls="--",lw=1,label="0.15 engineering margin")
        ax.axvline(33,color="gray",ls=":",lw=1)
        ax.set_xscale("log");ax.set_xlabel("Hypothetical participants; eight films")
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
    axes[0].set_ylabel("Nominal 80% detectable gain (MAE units)")
    axes[2].legend(fontsize=7,loc="upper right")
    fig.suptitle("Illustrative variance projections; actual power is unvalidated",fontsize=12)
    fig.text(.5,.02,"Reuse of six development films; negative components clipped for planning. Final selection uncertainty omitted.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.93])
    fig.savefig(PACKET/"precision_scenarios.png",dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    result = analyze()
    destination = PACKET / "analysis.json"
    if args.check:
        if load(destination) != result:
            raise ValueError("Saved scope/precision analysis differs from reproduction")
    else:
        destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        plot(result)
    print(json.dumps({"passed":True,"contrasts":len(result["contrasts"]),"new_fits":0,"confirmation_opened":False}))


if __name__ == "__main__":
    main()
