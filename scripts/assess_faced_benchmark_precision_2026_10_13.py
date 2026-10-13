"""Predeclared, source-only precision planning for an inactive FACED benchmark.

No recordings, private arrays, original ratings, development validation outcomes,
or reserved numerical data are read. Declaration hashes completed public outputs
without interpreting their prediction/target values. Analysis requires that exact
declaration and preserves its first attempt rather than replacing it.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from math import fsum, isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import t

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
OUT = BASE / "faced_benchmark_precision_2026-10-13"
SOURCE = BASE / "faced_coverage_missing_2026-10-13"
MODELS = ("category_median", "context_only", "late_joint", "distributed_joint")
ALPHAS = (.1, 10., 100.)
CONTRASTS = {
    "primary_matched_eeg_gain": ("context_only", "distributed_joint"),
    "secondary_practical_gain": ("category_median", "distributed_joint"),
    "secondary_coverage_gain": ("late_joint", "distributed_joint"),
}
RESOLUTIONS = (.05, .10, .15)
MULTIPLIERS = (.5, 1., 1.5, 2.)
INPUTS = (
    "results/development/faced_coverage_missing_2026-10-13/cv_predictions.json",
    "results/development/faced_coverage_missing_2026-10-13/comparison.json",
    "results/development/faced_coverage_missing_2026-10-13/completion.json",
    "results/development/faced_coverage_missing_2026-10-13/independent_audit.json",
    "results/development/faced_temporal_coverage_preflight_2026-10-12/plan.json",
    "results/development/faced_coverage_selection_preflight_2026-10-13/plan.json",
    "results/development/faced_coverage_missing_confirmation_proposal_2026-10-13/proposal.json",
    "results/development/faced_codebooks_2026-10-10/audit.json",
    "scripts/assess_faced_benchmark_precision_2026_10_13.py",
    "tests/test_faced_benchmark_precision.py",
    "docs/publication/FACED_Benchmark_Precision_Plan_2026-10-13.md",
)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def declare():
    """Metadata and byte digests only; no prediction/target summaries."""
    destination = OUT / "plan_v2.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing precision predeclaration")
    if not (OUT / "plan.json").exists() or (OUT / "analysis_started.json").exists():
        raise PermissionError("V2 requires the preserved, unexecuted initial declaration")
    proposal = load(BASE / "faced_coverage_missing_confirmation_proposal_2026-10-13/proposal.json")
    source = load(BASE / "faced_temporal_coverage_preflight_2026-10-12/plan.json")
    cv = load(BASE / "faced_coverage_selection_preflight_2026-10-13/plan.json")
    if (proposal["primary_analysis"]["nominated_joint"] != "distributed_joint"
            or proposal["mandatory_secondary_analysis"]["fixed_common_source_selected_alpha"] != 100.
            or proposal["protected_signal_access_authorized"] is not False
            or proposal["protected_rating_access_authorized"] is not False
            or len(source["source_people"]) != 70 or len(source["source_records"]) != 980
            or len(cv["cells"]) != 10):
        raise PermissionError("Completed-source candidate and sealed roles must agree")
    plan = {
        "date": "2026-10-13", "status": "predeclared_source_only_precision_planning",
        "declaration_version": 2,
        "preserved_unexecuted_v1": {"path": "results/development/faced_benchmark_precision_2026-10-13/plan.json",
                                   "sha256": sha(OUT / "plan.json"),
                                   "reason": "Global metadata-before-numeric barrier qualified before execution; scientific settings unchanged"},
        "candidate": "distributed_joint", "frozen_alpha": 100.,
        "comparisons": {k: list(v) for k, v in CONTRASTS.items()},
        "source_scope": {"participants": 70, "held_pairs_per_method": 840,
                         "held_clips_per_person": 12, "held_categories": 6,
                         "cv_prediction_rows": 8400, "numeric_rows_used": 3360,
                         "training_only_categories_without_held_support": ["Fear", "Inspiration"]},
        "future_scope_metadata_only": {"participants": 33, "title_families": 8,
                                       "clips": 9, "df": 32},
        "weighting": "Absolute clip losses, then within-person/family mean, equal families, equal people",
        "planned_outputs": ["all four source method errors", "all three fixed contrasts",
                            "70 anonymous participant gains", "all 12 held-title gains",
                            "all six held category gains and target descriptive support",
                            "two training-only zero-support category entries",
                            "plug-in paired-t halfwidth for 33 people",
                            "SD multiplier and hypothetical resolution planning",
                            "unknown Fear/Inspiration SD/correlation sensitivity",
                            "bounded-gain worst-case halfwidth guard"],
        "engineering_halfwidths": list(RESOLUTIONS), "sd_multipliers": list(MULTIPLIERS),
        "hypothetical_n_search": [2, 10000],
        "unknown_category_sensitivity": {"supported_weight": .75, "unsupported_weight": .25,
                                         "unknown_mean_sd": [0., .5, 1., 2., 7.],
                                         "correlations": [-1., 0., 1.]},
        "interpretation": "Source descriptive summaries and transport assumptions only; no certified future precision, power, clinical margin, equivalence or independent source interval",
        "future_paired_t_assumptions": "Independent participant gains; approximate sampling distribution; conditional on fixed films/states; no recruitment-representativeness or arbitrary-film guarantee",
        "selection_and_state_mismatch": "Repeated adaptive source choice; 448-row fold fits and two material branches per person differ from one frozen 980-row future state",
        "new_models_or_fits": 0, "resampling_or_selection": 0,
        "validation_numeric_access": False, "protected_numeric_access": False,
        "original_recording_or_rating_access": False, "confirmation_activated": False,
        "bindings": {path: sha(REPO / path) for path in INPUTS},
    }
    write_new(destination, plan)
    print(json.dumps({"status": plan["status"], "plan_sha256": sha(destination)}))


def qualify_rows(rows, metadata, cells, expected_pairs, *, expected_total=8400):
    """Reject wrong role/identity/context before interpreting any numeric value."""
    selected, seen = {model: {} for model in MODELS}, set()
    if len(rows) != expected_total:
        raise ValueError("Incomplete bounded public source prediction packet")
    for row in rows:
        if row.get("role") != "source" or not isinstance(row.get("subject"), str):
            raise PermissionError("Only completed source-role predictions are eligible")
        if type(row.get("vid")) is not int:
            raise PermissionError("Source clip identity must already be an integer")
        pair = row["subject"], row["vid"]
        if pair not in metadata or row.get("cell") not in cells:
            raise PermissionError("Prediction identity/cell is outside the source contract")
        meta, cell = metadata[pair], cells[row["cell"]]
        if (pair not in {tuple(p) for p in cell["held"]}
                or row.get("family") != meta["family"]
                or row.get("assigned_category") != meta["assigned_category"]
                or row.get("acquisition_group") != meta["acquisition_group"]
                or row.get("person_fold") != cell["person_fold"]
                or row.get("material_assignment") != cell["material_assignment"]):
            raise PermissionError("Source prediction context/material binding differs")
        model, alpha = row.get("model"), row.get("alpha")
        if model not in MODELS or (model == "category_median" and alpha is not None):
            raise ValueError("Undeclared source comparator")
        if model != "category_median" and (isinstance(alpha, bool) or alpha not in ALPHAS):
            raise ValueError("Undeclared source alpha")
        identity = model, alpha, pair
        if identity in seen:
            raise ValueError("Duplicate source candidate prediction")
        seen.add(identity)
        if model != "category_median" and alpha != 100.:
            continue
        selected[model][pair] = row
    if any(set(panel) != set(expected_pairs) for panel in selected.values()):
        raise ValueError("Every fixed comparator must retain the complete source denominator")
    if len(seen) != 10 * len(expected_pairs):
        raise ValueError("All nine source Ridge candidates and category prior must be retained")
    # Only after the complete packet passes metadata/model/denominator guards may
    # any selected target or prediction be converted. A denied later row cannot
    # follow an earlier numerical conversion.
    for model, panel in selected.items():
        for pair, row in list(panel.items()):
            target, prediction = row["target"], row["prediction"]
            if isinstance(target, bool) or isinstance(prediction, bool):
                raise ValueError("Boolean numeric output is invalid")
            target, prediction = float(target), float(prediction)
            if not isfinite(target) or not isfinite(prediction) or not 0 <= target <= 7 or not 0 <= prediction <= 7:
                raise ValueError("Nonfinite or outside-scale source prediction")
            panel[pair] = {**row, "target": target, "prediction": prediction}
    for pair in expected_pairs:
        if len({selected[m][pair]["target"] for m in MODELS}) != 1:
            raise ValueError("Paired source targets differ across methods")
    return selected


def participant_errors(panel):
    families = defaultdict(list)
    for (subject, _), row in panel.items():
        families[subject, row["family"]].append(abs(row["target"] - row["prediction"]))
    people = defaultdict(list)
    for (subject, _), losses in families.items():
        people[subject].append(fsum(losses) / len(losses))
    return {subject: fsum(losses) / len(losses) for subject, losses in people.items()}


def describe(values):
    if len(values) < 2:
        raise ValueError("At least two independent identities are needed for an SD summary")
    return {"n": len(values), "mean": mean(values), "sample_sd": stdev(values),
            "minimum": min(values), "maximum": max(values)}


def halfwidth(sd, n):
    if not isfinite(sd) or sd < 0 or type(n) is not int or n < 2:
        raise ValueError("Invalid hypothetical SD or participant count")
    return float(t.ppf(.975, n - 1)) * sd / sqrt(n)


def hypothetical_required_n(sd, resolution, cap=10000):
    if not isfinite(sd) or sd < 0 or not isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution and assumed SD must be valid engineering values")
    ns = np.arange(2, cap + 1)
    eligible = ns[t.ppf(.975, ns - 1) * sd / np.sqrt(ns) <= resolution]
    return int(eligible[0]) if len(eligible) else None


def unknown_category_sd(supported_sd, unknown_sd, rho):
    if (not isfinite(supported_sd) or supported_sd < 0 or not isfinite(unknown_sd)
            or unknown_sd < 0 or not isfinite(rho) or not -1 <= rho <= 1):
        raise ValueError("Invalid transport sensitivity assumption")
    variance = (.75 * supported_sd) ** 2 + (.25 * unknown_sd) ** 2 + 2 * .75 * .25 * rho * supported_sd * unknown_sd
    return sqrt(max(0., variance))


def analyze():
    if (OUT / "analysis_started.json").exists():
        raise FileExistsError("Preserve the existing precision analysis attempt")
    plan = load(OUT / "plan_v2.json")
    if (plan["status"] != "predeclared_source_only_precision_planning"
            or plan["declaration_version"] != 2
            or plan["candidate"] != "distributed_joint" or plan["frozen_alpha"] != 100.
            or plan["preserved_unexecuted_v1"]["sha256"] != sha(OUT / "plan.json")
            or plan["bindings"] != {path: sha(REPO / path) for path in INPUTS}):
        raise PermissionError("Precision declaration/input/code binding changed")
    write_new(OUT / "analysis_started.json", {"plan_sha256": sha(OUT / "plan_v2.json"),
                                              "new_fits": 0, "reserved_access": False})
    try:
        original = load(BASE / "faced_temporal_coverage_preflight_2026-10-12/plan.json")
        cv = load(BASE / "faced_coverage_selection_preflight_2026-10-13/plan.json")
        audit = load(SOURCE / "independent_audit.json")
        comparison = load(SOURCE / "comparison.json")
        packet = load(SOURCE / "cv_predictions.json")
        if audit["passed"] is not True or comparison["validation_access"] is not False or comparison["protected_access"] is not False:
            raise PermissionError("Completed independent source arithmetic qualification required")
        metadata = {(r["subject"], r["vid"]): r for r in original["source_records"]}
        cells = {c["cell"]: c for c in cv["cells"]}
        pairs = {tuple(p) for c in cv["cells"] for p in c["held"]}
        panels = qualify_rows(packet["ridge"] + packet["category"], metadata, cells, pairs)
        if len(pairs) != 840 or len({p[0] for p in pairs}) != 70:
            raise PermissionError("Frozen source denominator changed")
        methods = {model: participant_errors(panel) for model, panel in panels.items()}
        people = sorted(methods["context_only"])
        person_records = []
        contrast_records = {}
        for subject in people:
            own = {model: methods[model][subject] for model in MODELS}
            person_records.append({"subject": subject, "role": "source", "held_clips": 12,
                                   "method_mae": own,
                                   "gains": {name: own[a] - own[b] for name, (a, b) in CONTRASTS.items()}})
        for name in CONTRASTS:
            summary = describe([r["gains"][name] for r in person_records])
            sd = summary["sample_sd"]
            contrast_records[name] = {**summary, "plug_in_n33_halfwidth": halfwidth(sd, 33),
                "sd_sensitivity": [{"sd_multiplier": multiplier, "assumed_sd": multiplier * sd,
                                    "n33_halfwidth": halfwidth(multiplier * sd, 33),
                                    "hypothetical_required_n": {str(res): hypothetical_required_n(multiplier * sd, res) for res in RESOLUTIONS}}
                                   for multiplier in MULTIPLIERS]}
        primary_sd = contrast_records["primary_matched_eeg_gain"]["sample_sd"]
        catalogue = {r["clip"]: r for r in load(BASE / "faced_codebooks_2026-10-10/audit.json")["stimuli"]}
        categories = ("Anger", "Disgust", "Fear", "Sadness", "Amusement", "Inspiration", "Joy", "Tenderness")
        category_records, title_records = [], []
        for category in categories:
            own_pairs = {p for p in pairs if metadata[p]["assigned_category"] == category}
            if not own_pairs:
                category_records.append({"assigned_category": category, "held_source_pairs": 0,
                                         "held_source_titles": 0, "transfer_support": "training_only_no_source_held_evidence"})
                continue
            local = {model: participant_errors({p: panels[model][p] for p in own_pairs}) for model in MODELS}
            means = {model: mean(list(local[model].values())) for model in MODELS}
            gains = {name: describe([local[a][s] - local[b][s] for s in people]) for name, (a, b) in CONTRASTS.items()}
            category_records.append({"assigned_category": category, "held_source_pairs": len(own_pairs),
                                     "held_source_titles": len({p[1] for p in own_pairs}),
                                     "target_descriptive_support": describe([panels["context_only"][p]["target"] for p in sorted(own_pairs)]),
                                     "method_mae": means, "gains": gains,
                                     "transfer_support": "two_source_titles_no_protected_rating_evidence"})
        for vid in sorted({p[1] for p in pairs}):
            own_pairs = {p for p in pairs if p[1] == vid}
            local = {model: participant_errors({p: panels[model][p] for p in own_pairs}) for model in MODELS}
            title_records.append({"vid": vid, "family": catalogue[vid]["source_film_sha256"],
                                  "assigned_category": catalogue[vid]["assigned_category"], "source_people": 70,
                                  "method_mae": {m: mean(list(local[m].values())) for m in MODELS},
                                  "gains": {name: describe([local[a][s] - local[b][s] for s in people]) for name, (a, b) in CONTRASTS.items()}})
        unknown = [{"assumed_supported_sd": primary_sd, "unknown_two_category_mean_sd": u,
                    "supported_unknown_correlation": rho,
                    "assumed_future_gain_sd": unknown_category_sd(primary_sd, u, rho),
                    "n33_halfwidth": halfwidth(unknown_category_sd(primary_sd, u, rho), 33)}
                   for u in plan["unknown_category_sensitivity"]["unknown_mean_sd"]
                   for rho in plan["unknown_category_sensitivity"]["correlations"]]
        summary = {
            "date": "2026-10-13", "status": "completed_source_only_precision_planning",
            "candidate": "distributed_joint", "alpha": 100., "source_pairs": 840,
            "source_people": 70, "numeric_rows_used": 3360, "input_rows_role_checked": 8400,
            "method_mae": {m: mean(list(methods[m].values())) for m in MODELS},
            "fixed_contrasts": contrast_records, "categories": category_records,
            "unobserved_category_sensitivity": unknown,
            "unknown_category_mean_limit": "Under unsupported equality of six future/source category means, future mean=.75*source_mean+.25*unknown_mean; unknown_mean may be anywhere in [-7,7]. This is algebraic sensitivity, not a validated bound on the actual future mean.",
            "bounded_gain_n33_sample_halfwidth_guard": float(t.ppf(.975, 32)) * 7 / sqrt(32),
            "engineering_resolutions_are_not": ["perceptual thresholds", "clinical margins", "equivalence margins", "hypothesis-test effects"],
            "precision_certified": False, "independent_confirmation": False,
            "new_fits": 0, "validation_numeric_access": False, "protected_numeric_access": False,
            "original_recording_or_rating_access": False, "confirmation_activated": False,
            "plan_sha256": sha(OUT / "plan_v2.json"), "input_bindings": plan["bindings"],
            "limitations": ["Adaptive reused source outcomes and candidate nomination",
                            "Shared CV training makes source participant gains dependent",
                            "Two material branches and 448-row CV fits differ from a fixed 980-row final state",
                            "Source averages twelve families in six categories; future averages eight families in eight categories",
                            "Fear and Inspiration have no source material-held outcome support",
                            "Even the six supported source titles do not qualify transfer to new protected titles",
                            "No neural-convergence, brain-causality or arbitrary-film inference"],
        }
        write_new(OUT / "participant_source_gains.json", person_records)
        write_new(OUT / "held_title_source_gains.json", title_records)
        write_new(OUT / "assessment.json", summary)
        write_new(OUT / "completion.json", {"passed": True, "status": summary["status"],
                                            "plan_sha256": summary["plan_sha256"],
                                            "assessment_sha256": sha(OUT / "assessment.json"),
                                            "participant_gains_sha256": sha(OUT / "participant_source_gains.json"),
                                            "held_title_gains_sha256": sha(OUT / "held_title_source_gains.json"),
                                            "source_only": True, "future_precision_certified": False,
                                            "confirmation_activated": False})
        print(json.dumps({"status": summary["status"], "primary": contrast_records["primary_matched_eeg_gain"],
                          "future_precision_certified": False}))
    except Exception as error:
        write_new(OUT / "failure.json", {"type": type(error).__name__, "message": str(error), "confirmation_activated": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("declare", "analyze"))
    action = parser.parse_args().action
    declare() if action == "declare" else analyze()
