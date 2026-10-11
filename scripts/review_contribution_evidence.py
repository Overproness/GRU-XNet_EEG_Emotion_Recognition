"""Aggregate existing public evidence without importing a loader or fitting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
OUTPUT = BASE / "contribution_decision_2026-10-11/evidence_inventory.json"
INPUTS = {
    "representation": "faced_representation_v2_2026-10-11/comparison.json",
    "completion": "faced_representation_v2_2026-10-11/completion/verification.json",
    "numerical_audit": "faced_representation_v2_2026-10-11/independent_analysis/summary.json",
    "quality": "faced_inference_quality_2026-10-11/comparison.json",
    "deap": "deap_baseline_folds_v2_2026-10-09/comparison.json",
    "participants": "faced_backup_2026-10-10/participant_reservation.json",
    "design": "faced_design_proposal_2026-10-11/recommended_proposal.json",
    "targets": "data_target_review_2026-10-10/metadata_findings.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-12:
        raise ValueError(f"Published aggregate disagreement: {actual}, {expected}")


def build() -> dict:
    docs = {name: json.loads((BASE / rel).read_text(encoding="utf-8"))
            for name, rel in INPUTS.items()}
    completion = docs["completion"]
    for rel, expected in completion["bindings"].items():
        if sha256(REPO / rel) != expected:
            raise ValueError(f"Earlier scientific evidence changed: {rel}")
    if not completion["independent_numerical_audit_passed"]:
        raise ValueError("Missing preceding numerical audit")
    if completion["confirmation_access"] or completion["research_question_changed"]:
        raise ValueError("Prior scope has changed")

    representation = docs["representation"]
    rotations = representation["rotations"]
    if [r["rotation"] for r in rotations] != [0, 1, 2]:
        raise ValueError("Unexpected development rotations")
    arms = ("source_fit", "source_oof", "validation_unseen", "validation_familiar")
    metrics = {
        model: {
            arm: mean(r["models"][model]["metrics"][arm]["participant_film_mae"]
                      for r in rotations)
            for arm in arms
        }
        for model in rotations[0]["models"]
    }
    quality = docs["quality"]
    previous = {
        model: mean(r["selected"][model]["validation_unseen"]["participant_film_mae"]
                    for r in quality["rotations"])
        for model in ("context_only", "spectral_eeg_only", "context_plus_spectral_eeg")
    }
    close(previous["context_only"] - previous["context_plus_spectral_eeg"],
          quality["mean_joint_gain_over_context"])
    screens = representation["lead_screens"]
    for model, screen in screens.items():
        baseline = "context_kernel" if model.endswith("kernel") else None
        comparator = metrics[baseline]["validation_unseen"] if baseline else previous["context_only"]
        close(comparator - metrics[model]["validation_unseen"],
              screen["mean_gain_over_matched_context"])
        close(previous["context_only"] - metrics[model]["validation_unseen"],
              screen["mean_gain_over_original_context_ridge"])
        if screen["numerical_screen_passed"]:
            raise ValueError("Decision must be reconsidered: a lead screen passed")

    roles = docs["participants"]
    role_names = ("development_source", "development_validation", "confirmation")
    role_sets = {key: set(roles[key]) for key in role_names}
    if [len(role_sets[key]) for key in role_names] != [70, 20, 33]:
        raise ValueError("Participant roles changed")
    if sum(map(len, role_sets.values())) != len(set.union(*role_sets.values())):
        raise ValueError("Participant overlap")
    design = docs["design"]
    material_sets = []
    for rotation in design["rotations"]:
        sets = {key: {x["family"] for x in rotation["families"][key]}
                for key in role_names}
        if sum(map(len, sets.values())) != len(set.union(*sets.values())):
            raise ValueError("Material-family overlap")
        material_sets.append(sets)
    if any(x["confirmation"] != material_sets[0]["confirmation"] for x in material_sets):
        raise ValueError("Confirmation families changed")
    counts = design["proposed_trial_counts"]
    for count in counts:
        if count["final_unseen_confirmation_trials"] != 33 * 9:
            raise ValueError("Unexpected protected-person/film geometry")
        if count["final_familiar_confirmation_trials"] != 33 * 14:
            raise ValueError("Unexpected familiar geometry")

    manuscript = REPO.parent / "report.tex"
    archive = REPO / "docs/paper_archive/2026-10-05-pre-exploration"
    expected_tex = "06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0"
    expected_pdf = "921f07f271bc4dadbabaf62ede384c543681b9fe0aec02c490cf7d76ef9de101"
    if manuscript.exists() and sha256(manuscript) != expected_tex:
        raise ValueError("Working manuscript source changed")
    if sha256(archive / "report.tex") != expected_tex:
        raise ValueError("Manuscript source changed")
    if sha256(archive / "DL-Report.pdf") != expected_pdf:
        raise ValueError("Archived PDF changed")

    deap = docs["deap"]["models"]["combined"]
    deap_points = {
        model: {
            arm: {metric: values["point"] for metric, values in row.items()}
            for arm, row in deap[model].items()
        }
        for model in ("baseline_relative", "baseline_relative_context", "context_logistic", "prior")
    }
    capacities = {
        model: mean(row["training_raw_mae"] for row in representation["capacity_diagnostics"]
                    if row["model"] == model)
        for model in ("capacity_spectral_real", "capacity_spectral_permuted",
                      "capacity_covariance_real", "capacity_covariance_permuted")
    }
    target_counts = {
        name: {key: corpus[key] for key in ("participants", "eligible_trials", "materials_or_conditions")}
        for name, corpus in docs["targets"]["datasets"].items()
    }
    return {
        "date": "2026-10-11",
        "scope": "Aggregation/checks of existing public evidence; not a fresh numerical experiment or confirmation",
        "inputs": {f"results/development/{rel}": sha256(BASE / rel) for rel in INPUTS.values()},
        "script_sha256": sha256(Path(__file__)),
        "preceding_scientific_bindings_verified": len(completion["bindings"]),
        "preceding_numerical_audit": docs["numerical_audit"],
        "faced_mean_mae_by_condition": metrics,
        "faced_preceding_selected_unseen_mae": previous,
        "faced_lead_screens": screens,
        "new_covariance_joint_minus_preceding_spectral_joint_mae":
            metrics["context_covariance_kernel"]["validation_unseen"] - previous["context_plus_spectral_eeg"],
        "faced_source_capacity_raw_mae": capacities,
        "deap_combined_points": deap_points,
        "earlier_target_inventory": target_counts,
        "faced_roles": {key: len(role_sets[key]) for key in role_names},
        "faced_confirmation_geometry": {
            "participants": 33, "exact_title_families": len(material_sets[0]["confirmation"]),
            "clips": len(design["rotations"][0]["clips"]["confirmation"]),
            "potential_unseen_trials": 297, "potential_familiar_trials": 462,
            "scope": "Metadata capacity only; targets, physical media independence, confirmation quality and power not inspected",
        },
        "manuscript_source_sha256": expected_tex,
        "archived_pdf_sha256": expected_pdf,
        "decision": {
            "candidate_dataset": "FACED",
            "candidate_target": "Individual item-10 continuous valence, 0-7, under the existing non-neutral population definition",
            "architecture_expansion": "hold",
            "evaluation_paper_adoption": "hold_pending_distinct_claim_and_author_approval",
            "original_joint_architecture_submission": "not_supported_by_current_evidence",
            "negative_transfer_paper_adoption": "not_supported_by_current_evidence",
            "confirmation_opening": "hold",
        },
        "new_fits": 0, "new_predictions": 0, "new_original_ratings_decoded": 0,
        "original_recordings_opened": 0, "confirmation_outcomes_opened": 0,
        "research_question_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the saved review without changing it")
    args = parser.parse_args()
    result = build()
    if args.check:
        if json.loads(OUTPUT.read_text(encoding="utf-8")) != result:
            raise ValueError("Saved decision evidence differs from reconstruction")
    else:
        if OUTPUT.exists():
            raise FileExistsError("Use --check; do not overwrite a completed review")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checked": args.check, "earlier_scientific_files_verified":
                      result["preceding_scientific_bindings_verified"], "new_fits": 0,
                      "working_manuscript_checked": (REPO.parent / "report.tex").exists(),
                      "confirmation_outcomes_opened": 0, "research_question_changed": False}))


if __name__ == "__main__":
    main()
