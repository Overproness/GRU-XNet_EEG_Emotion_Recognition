"""Outcome-free synthetic FACED precision sensitivity; never opens empirical values.

Only the two explicit, public reservation/catalogue metadata JSON files are read.
Synthetic simulations require a separately frozen root activation and plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/development/faced_confirmation_freeze_2026-10-12"
SCRIPT = "scripts/faced_confirmation_precision_2026_10_12.py"
TEST = "tests/test_faced_confirmation_precision.py"
DESIGN = "docs/publication/FACED_Confirmation_Precision_Design_2026-10-12.md"
METADATA = (
    "results/development/faced_backup_2026-10-10/participant_reservation.json",
    "results/development/faced_design_proposal_2026-10-11/recommended_proposal.json",
)
SEED = 2026101209
REPLICATES = 256
BOOTSTRAP_DRAWS = 512
CPU_SECONDS = 300
GAINS = (0.0, 0.045, 0.15)
SCENARIOS = (
    {"name": "low", "subject_sd": 0.10, "title_sd": 0.10, "interaction_sd": 0.20},
    {"name": "participant_dominant", "subject_sd": 0.30, "title_sd": 0.05, "interaction_sd": 0.20},
    {"name": "title_dominant", "subject_sd": 0.05, "title_sd": 0.30, "interaction_sd": 0.20},
    {"name": "balanced", "subject_sd": 0.20, "title_sd": 0.20, "interaction_sd": 0.40},
)
DIMENSIONS = (
    {"subjects": 33, "titles": 8, "status": "actual_reserved_dimensions_only"},
    {"subjects": 66, "titles": 8, "status": "hypothetical_more_people_not_available_confirmation"},
    {"subjects": 33, "titles": 16, "status": "hypothetical_more_titles_not_available_confirmation"},
    {"subjects": 66, "titles": 16, "status": "hypothetical_both_not_available_confirmation"},
)
ALLOWED_STATUSES = {
    "active_confirmation_precision_simulation",
    "active_development_confirmation_freeze",
    "active_development_only_confirmation_protocol",
    "active_development_only_confirmation_freeze",
    "prepared_development_only_confirmation_protocol",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def metadata_projection(root: Path = ROOT) -> dict:
    """Read identities/catalogue grouping only; reject any changed geometry."""
    reservation = load_json(root / METADATA[0])
    proposal = load_json(root / METADATA[1])
    roles = {name: list(reservation[name]) for name in
             ("development_source", "development_validation", "confirmation")}
    if {name: len(value) for name, value in roles.items()} != {
        "development_source": 70, "development_validation": 20, "confirmation": 33
    }:
        raise ValueError("Unexpected metadata participant-role dimensions")
    flat = [person for group in roles.values() for person in group]
    if len(flat) != len(set(flat)):
        raise ValueError("Duplicate or overlapping metadata participant roles")
    rotations = proposal["rotations"]
    if len(rotations) != 3:
        raise ValueError("Unexpected metadata material rotations")
    confirmation = rotations[0]["families"]["confirmation"]
    expected = sorted((r["family"], tuple(sorted(r["clips"]))) for r in confirmation)
    for rotation in rotations:
        current = sorted((r["family"], tuple(sorted(r["clips"])))
                         for r in rotation["families"]["confirmation"])
        if current != expected:
            raise ValueError("Protected catalogue roster differs between rotations")
    confirm_clips = sorted(v for _, clips in expected for v in clips)
    if len(expected) != 8 or len(set(f for f, _ in expected)) != 8:
        raise ValueError("Unexpected metadata protected title-family dimensions")
    if confirm_clips != [1, 5, 7, 8, 10, 19, 21, 24, 26]:
        raise ValueError("Unexpected metadata protected clip roster")
    if sum(tuple(clips) == (7, 8) for _, clips in expected) != 1:
        raise ValueError("The two same-title clips are not grouped together")
    development = {r["family"] for rotation in rotations
                   for arm in ("development_source", "development_validation")
                   for r in rotation["families"][arm]}
    if len(development) != 14 or development & {f for f, _ in expected}:
        raise ValueError("Unexpected development/protected title-family separation")
    return {
        "source_participants": 70,
        "validation_participants": 20,
        "protected_participants": 33,
        "development_title_families": 14,
        "protected_title_families": 8,
        "protected_clips": 9,
        "protected_nominal_trials": 297,
        "protected_person_title_cells": 264,
        "source_only_development_rectangle_trials": 980,
        "same_title_rule": "Average paired losses within clips 7/8 before family-equal averaging",
        "protected_family_clip_projection": [
            {"family": f, "clips": list(clips)} for f, clips in expected
        ],
        "empirical_value_access": False,
    }


def scientific_conditions() -> dict:
    return {
        "model": "D_ij = delta + A_i + B_j + E_ij; independent zero-mean Gaussian random effects",
        "effect_sign": "Positive paired absolute-loss difference means lower EEG-model loss",
        "variance_scenarios": list(SCENARIOS),
        "scenario_provenance": "Fixed analyst-chosen sensitivity assumptions; not empirical estimates",
        "gain_provenance": "The root supplied 0.045 after published development summaries; scenarios are not blinded to public aggregates, but no empirical variance calibration is used",
        "dimensions": list(DIMENSIONS),
        "true_gains": list(GAINS),
        "units": "Valence rating points in paired absolute-loss differences",
        "replicates_per_scenario_dimension": REPLICATES,
        "crossed_bootstrap_draws_per_replicate": BOOTSTRAP_DRAWS,
        "bootstrap_method": "Independently resample both complete person and title axes with replacement; percentile quantiles 0.025/0.975",
        "quantile_method": "NumPy linear",
        "rng": "numpy.random.default_rng, SeedSequence([2026101209, scenario_index, dimension_index]).spawn(2)",
        "seed": SEED,
        "translation_reuse": "The same zero-gain matrix and bootstrap samples are translated by each fixed gain",
        "report_rows": 48,
        "synthetic_matrices": 4096,
        "bootstrap_means": 2097152,
        "cpu_time_cap_seconds": CPU_SECONDS,
        "wall_time_cap_seconds": CPU_SECONDS,
        "new_empirical_fits": 0,
        "confirmation_access": False,
        "precision_gate_passed": False,
        "raw_eeg_access": False,
        "original_ratings_access": False,
        "cached_target_or_prediction_access": False,
        "limits": [
            "A Gaussian loss-difference approximation has unbounded support outside actual paired losses [-7,7]; it is not a rating generator or fitted EEG model",
            "Simulation coverage/exclusion rates apply only to this synthetic random-effects model",
            "Eight catalogue titles are an assigned roster, not a random representative sample of all films",
            "Training/selection uncertainty, non-Gaussian errors, acquisition heterogeneity and missingness are omitted",
            "The 0.15 gain is an engineering sensitivity value, not a clinically validated margin",
            "The 66-person/16-title cases are hypothetical design comparisons, not permission to reassign protected roles",
            "These calculations do not establish empirical power, promise significance or authorize confirmation access",
        ],
    }


def declare(output: Path, root: Path = ROOT) -> Path:
    projection = metadata_projection(root)
    plan = {
        "schema": "faced_synthetic_confirmation_precision_v1",
        "date": "2026-10-12 Asia/Karachi",
        "status": "declared_not_executed",
        "metadata_projection": projection,
        "metadata_bindings": {p: sha256(root / p) for p in METADATA},
        "code_bindings": {p: sha256(root / p) for p in (SCRIPT, TEST, DESIGN)},
        "conditions": scientific_conditions(),
    }
    path = output / "precision_plan.json"
    write_new(path, plan)
    return path


def validate_activation(plan_path: Path, activation_path: Path, root: Path = ROOT) -> dict:
    """Fail closed before any synthetic random draw if not root-authorized."""
    activation = load_json(activation_path)
    if activation.get("status") not in ALLOWED_STATUSES:
        raise PermissionError("Precision simulation activation status is not active")
    if activation.get("precision_simulation_authorized") is not True:
        raise PermissionError("Precision simulation is not separately authorized")
    if activation.get("confirmation_access") is not False:
        raise PermissionError("Confirmation must remain sealed")
    if activation.get("research_question_changed") is not False:
        raise PermissionError("This precision sensitivity cannot adopt a manuscript question")
    if activation.get("precision_plan_sha256") != sha256(plan_path):
        raise PermissionError("Precision plan is not bound by the root activation")
    plan = load_json(plan_path)
    if plan.get("conditions") != scientific_conditions():
        raise ValueError("Scientific conditions differ from frozen implementation")
    for group in ("metadata_bindings", "code_bindings"):
        for relative, expected in plan[group].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
                raise ValueError("A frozen precision input/code binding changed")
    if plan["metadata_projection"] != metadata_projection(root):
        raise ValueError("Metadata geometry differs from frozen projection")
    return plan


def crossed_samples(matrix: np.ndarray, person_counts: np.ndarray,
                    title_counts: np.ndarray) -> np.ndarray:
    """Equivalent to Cartesian resampling of both axes, without a 3-D expansion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    people = np.asarray(person_counts, dtype=np.float64)
    titles = np.asarray(title_counts, dtype=np.float64)
    if matrix.ndim != 2 or people.ndim != 2 or titles.ndim != 2:
        raise ValueError("Crossed bootstrap arrays must be two-dimensional")
    n, j = matrix.shape
    if people.shape[1] != n or titles.shape[1] != j or people.shape[0] != titles.shape[0]:
        raise ValueError("Crossed bootstrap weight dimensions differ")
    if (people < 0).any() or (titles < 0).any():
        raise ValueError("Crossed bootstrap counts cannot be negative")
    if not np.equal(people, np.rint(people)).all() or not np.equal(titles, np.rint(titles)).all():
        raise ValueError("Crossed bootstrap counts must be integers")
    if not np.all(people.sum(axis=1) == n) or not np.all(titles.sum(axis=1) == j):
        raise ValueError("Crossed bootstrap count totals differ from complete axes")
    if not np.isfinite(matrix).all() or not np.isfinite(people).all() or not np.isfinite(titles).all():
        raise ValueError("Nonfinite synthetic matrix/counts")
    return np.sum((people @ matrix) * titles, axis=1) / (n * j)


def analytic_variance(scenario: dict, subjects: int, titles: int) -> dict:
    subject = scenario["subject_sd"] ** 2 / subjects
    title = scenario["title_sd"] ** 2 / titles
    interaction = scenario["interaction_sd"] ** 2 / (subjects * titles)
    variance = subject + title + interaction
    return {
        "subject_mean_variance": subject,
        "title_mean_variance": title,
        "interaction_mean_variance": interaction,
        "mean_variance": variance,
        "oracle_standard_error": math.sqrt(variance),
        "oracle_gaussian_95_interval_width": 2 * 1.959963984540054 * math.sqrt(variance),
        "infinite_people_title_standard_error_floor": math.sqrt(title),
    }


def summarize(records: list[dict]) -> list[dict]:
    """Aggregate synthetic base intervals; translations preserve coverage/width."""
    result = []
    for sindex, scenario in enumerate(SCENARIOS):
        for dindex, dimensions in enumerate(DIMENSIONS):
            matching = [r for r in records if r["scenario_index"] == sindex and r["dimension_index"] == dindex]
            if len(matching) != REPLICATES or sorted(r["replicate"] for r in matching) != list(range(REPLICATES)):
                raise ValueError("Synthetic replicate coverage differs from the declared grid")
            lower = np.asarray([r["base_interval_lower"] for r in matching])
            upper = np.asarray([r["base_interval_upper"] for r in matching])
            means = np.asarray([r["base_mean"] for r in matching])
            covered = int(((lower <= 0) & (upper >= 0)).sum())
            for gain in GAINS:
                positive = int((lower + gain > 0).sum())
                negative = int((upper + gain < 0).sum())
                excluded = positive + negative
                coverage = covered / REPLICATES
                exclusion = excluded / REPLICATES
                result.append({
                    "scenario": scenario["name"],
                    "scenario_index": sindex,
                    "dimension_index": dindex,
                    "subjects": dimensions["subjects"],
                    "titles": dimensions["titles"],
                    "dimension_status": dimensions["status"],
                    "true_gain": gain,
                    "replicates": REPLICATES,
                    "coverage_count": covered,
                    "coverage_rate": coverage,
                    "coverage_monte_carlo_se": math.sqrt(coverage * (1 - coverage) / REPLICATES),
                    "positive_lower_endpoint_count": positive,
                    "negative_upper_endpoint_count": negative,
                    "zero_exclusion_count": excluded,
                    "zero_exclusion_rate": exclusion,
                    "zero_exclusion_monte_carlo_se": math.sqrt(exclusion * (1 - exclusion) / REPLICATES),
                    "mean_interval_width": float(np.mean(upper - lower)),
                    "median_interval_width": float(np.median(upper - lower)),
                    "mean_estimate_bias": float(np.mean(means)),
                    **analytic_variance(scenario, dimensions["subjects"], dimensions["titles"]),
                })
    return result


def simulate(check_budget=lambda: None) -> list[dict]:
    records = []
    for sindex, scenario in enumerate(SCENARIOS):
        for dindex, dimensions in enumerate(DIMENSIONS):
            n, j = dimensions["subjects"], dimensions["titles"]
            synthetic_seed, bootstrap_seed = np.random.SeedSequence([SEED, sindex, dindex]).spawn(2)
            synthetic_rng = np.random.default_rng(synthetic_seed)
            bootstrap_rng = np.random.default_rng(bootstrap_seed)
            for replicate in range(REPLICATES):
                check_budget()
                matrix = (
                    synthetic_rng.normal(0, scenario["subject_sd"], size=(n, 1))
                    + synthetic_rng.normal(0, scenario["title_sd"], size=(1, j))
                    + synthetic_rng.normal(0, scenario["interaction_sd"], size=(n, j))
                )
                pc = bootstrap_rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP_DRAWS)
                tc = bootstrap_rng.multinomial(j, np.full(j, 1 / j), size=BOOTSTRAP_DRAWS)
                lower, upper = np.quantile(crossed_samples(matrix, pc, tc), [0.025, 0.975], method="linear")
                records.append({
                    "scenario_index": sindex, "dimension_index": dindex, "replicate": replicate,
                    "base_mean": float(matrix.mean()), "base_interval_lower": float(lower),
                    "base_interval_upper": float(upper),
                })
    check_budget()
    return records


def run(output: Path, activation: Path, root: Path = ROOT) -> dict:
    plan_path = output / "precision_plan.json"
    validate_activation(plan_path, activation, root)
    names = ("precision_replicates.json", "precision_summary.json", "precision_receipt.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Preserve prior precision outputs; no overwriting/repeated producer")
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    def check_budget():
        if time.process_time() - cpu_start > CPU_SECONDS or time.perf_counter() - wall_start > CPU_SECONDS:
            raise TimeoutError("Fixed five-minute synthetic CPU/wall budget exceeded")
    records = simulate(check_budget)
    rows = summarize(records)
    write_new(output / names[0], {"schema": "synthetic_zero_gain_replicates_v1", "records": records})
    write_new(output / names[1], {
        "schema": "synthetic_crossed_precision_summary_v1", "rows": rows,
        "assumptions": scientific_conditions(), "metadata_dimensions": metadata_projection(root),
        "interpretation": "Hypothetical precision sensitivity only; no empirical variance, power, effect or protected outcomes inferred",
    })
    receipt = {
        "status": "completed_synthetic_only",
        "plan_sha256": sha256(plan_path), "activation_sha256": sha256(activation),
        "output_bindings": {name: sha256(output / name) for name in names[:2]},
        "synthetic_matrices": len(records), "bootstrap_means": len(records) * BOOTSTRAP_DRAWS,
        "report_rows": len(rows), "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "confirmation_access": False, "cached_target_or_prediction_access": False,
        "precision_gate_passed": False,
        "original_ratings_access": False, "original_eeg_access": False, "new_empirical_fits": 0,
        "research_question_changed": False,
    }
    write_new(output / names[2], receipt)
    return receipt


def verify(output: Path) -> dict:
    """Aggregate recount from synthetic records, not a new simulation."""
    receipt = load_json(output / "precision_receipt.json")
    for name, digest in receipt["output_bindings"].items():
        if name not in {"precision_replicates.json", "precision_summary.json"} or sha256(output / name) != digest:
            raise ValueError("Synthetic output binding differs")
    if sha256(output / "precision_plan.json") != receipt["plan_sha256"]:
        raise ValueError("Synthetic plan binding differs")
    records = load_json(output / "precision_replicates.json")["records"]
    actual = load_json(output / "precision_summary.json")["rows"]
    expected = summarize(records)
    if actual != expected:
        raise ValueError("Synthetic summary differs from independent recount")
    result = {"passed": True, "synthetic_matrices": len(records), "summary_rows": len(actual),
              "confirmation_access": False, "new_simulation": False}
    write_new(output / "precision_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--declare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--activation", type=Path)
    args = parser.parse_args()
    if args.declare:
        print(json.dumps({"precision_plan": str(declare(args.output))}))
    elif args.run:
        if args.activation is None:
            parser.error("--run requires --activation")
        print(json.dumps(run(args.output, args.activation)))
    else:
        print(json.dumps(verify(args.output)))


if __name__ == "__main__":
    main()
