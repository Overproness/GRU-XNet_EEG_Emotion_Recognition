"""Review already exposed target metadata; no waveform loading, fitting or inference."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/development/data_target_review_2026-10-10"
INPUTS = {
    "DEAP": "results/development/repeated_material_deap/predictions_transformer_unexposed_group1.csv",
    "SEEDIV": "results/development/temporal_native_seediv/predictions_transformer_absolute_native4.csv",
    "GAMEEMO": "results/development/negative_transfer_neural_gameemo/trial_predictions.csv",
    **{f"deap_folds_{g}": f"results/development/repeated_material_deap/folds_group{g}.json" for g in (1, 2)},
    **{f"deap_materials_{g}": f"results/development/repeated_material_deap/material_assignments_group{g}.csv" for g in (1, 2)},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metadata(rows, dataset):
    """Ignore probability columns and keep one checked label per original trial."""
    unique = {}
    for r in rows:
        subject, trial = r["subject_id"], r["trial_id"]
        if not trial.startswith(subject + ":") or not subject.startswith(dataset + ":"):
            raise ValueError("Non-qualified participant/trial identity")
        if dataset == "SEEDIV":
            label = int(r["original_label"])
            material = ":".join(trial.split(":")[2:])
        else:
            label = int(r["label"])
            material = trial.rsplit(":", 1)[-1]
        if label not in (range(4) if dataset == "SEEDIV" else (0, 1)):
            raise ValueError("Invalid native target")
        item = {"subject_id": subject, "trial_id": trial, "material": material, "label": label}
        if dataset == "DEAP":
            rating = float(r["original_label"])
            if not 1 <= rating <= 9 or rating == 5 or label != int(rating > 5):
                raise ValueError("DEAP is not the previously corrected valence target")
            item["rating"] = rating
        if trial in unique and unique[trial] != item:
            raise ValueError("Repeated predictions disagree on original metadata")
        unique[trial] = item
    return [unique[key] for key in sorted(unique)]


def select_source_rows(rows, allowed_subjects, allowed_materials):
    """Select identities first; only then inspect labels on allowed source rows.

    No alternate exposure arm or excluded-material labels set source sample counts.
    This is a preparation reference, not a newly adopted/fitted study protocol.
    """
    selected = []
    for row in rows:
        if row["subject_id"] not in allowed_subjects or row["material"] not in allowed_materials:
            continue
        item = copy.deepcopy(row)
        if item["label"] not in (0, 1):
            raise ValueError("Invalid source target")
        selected.append(item)
    if len({r["trial_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicated source trial")
    return sorted(selected, key=lambda r: r["trial_id"])


def counts(rows):
    return {str(k): v for k, v in sorted(Counter(r["label"] for r in rows).items())}


def describe(rows, dataset):
    materials = defaultdict(list)
    for r in rows:
        materials[r["material"]].append(r)
    per_material = []
    for key, part in sorted(materials.items()):
        c = counts(part)
        per_material.append({"material": key, "eligible_trials": len(part), "class_counts": c,
                             "cohort_minority_trials": len(part) - max(c.values())})
    result = {"participants": len({r["subject_id"] for r in rows}), "eligible_trials": len(rows),
              "materials_or_conditions": len(materials), "class_counts": counts(rows),
              "materials_with_multiple_labels": sum(len(r["class_counts"]) > 1 for r in per_material),
              "cohort_minority_trials": sum(r["cohort_minority_trials"] for r in per_material),
              "material_counts": per_material,
              "evaluated_participant_ids": len({r["subject_id"] for r in rows}),
              "evaluated_material_keys": len(materials),
              "unexposed_declared_participant_ids": 0, "unexposed_declared_material_keys": 0}
    if dataset == "DEAP":
        result.update({"original_trials": 1280, "midpoint_exclusions": 16,
                       "eligible_ratings_within_one_of_midpoint": sum(abs(r["rating"]-5) <= 1 for r in rows),
                       "scope": "Individual recovered valence. Cohort-majority disagreements are descriptive, not a fitted or predictive baseline; no threshold is selected."})
    elif dataset == "GAMEEMO":
        result.update({"original_trials": 112, "midpoint_exclusions": 19,
                       "scope": "Individual SAM valence. Game keys describe interactive conditions, not identical video trajectories."})
        for r in per_material:
            r["midpoint_exclusions"] = 28-r["eligible_trials"]
    else:
        result.update({"original_trials": 1080, "midpoint_exclusions": 0,
                       "scope": "Native four-class assigned labels. Individual emotion-label agreement cannot be estimated from these clip labels."})
    return result


def deap_preparation(rows):
    results, metamorphic = [], 0
    subjects = {r["subject_id"] for r in rows}
    for group in (1, 2):
        folds = json.loads((REPO / INPUTS[f"deap_folds_{group}"]).read_text(encoding="utf-8"))
        assignments = read_csv(REPO / INPUTS[f"deap_materials_{group}"])
        ranks = {r["material_key"]: int(r["material_rank"]) for r in assignments}
        if sorted(ranks.values()) != list(range(40)):
            raise ValueError("Changed material assignment")
        for rotation in range(5):
            materials = {role: {key for key, rank in ranks.items() if rank//8 in blocks}
                         for role, blocks in {"train": ((rotation+2)%5, (rotation+3)%5),
                                              "validation": ((rotation+1)%5,), "test": (rotation,)}.items()}
            if any(materials[a] & materials[b] for a, b in itertools.combinations(materials, 2)):
                raise ValueError("Material boundary failed")
            for fold in folds:
                people = {role: set(fold[role]) for role in materials}
                if set.union(*people.values()) != subjects or any(people[a] & people[b] for a,b in itertools.combinations(people,2)):
                    raise ValueError("Participant boundary failed")
                source = select_source_rows(rows, people["train"], materials["train"])
                altered = copy.deepcopy(rows)
                removed = []
                for r in altered:
                    if r["subject_id"] not in people["train"] or r["material"] not in materials["train"]:
                        # An impossible label/rating detects even accidental inspection.
                        r["label"], r["rating"] = "EXCLUDED", None
                    else:
                        removed.append(r)
                if source != select_source_rows(altered, people["train"], materials["train"]):
                    raise ValueError("Excluded labels influenced source population")
                if source != select_source_rows(removed, people["train"], materials["train"]):
                    raise ValueError("Excluded records influenced source population")
                metamorphic += 2
                role_rows = {"train": source, **{role: [r for r in rows if r["subject_id"] in people[role] and r["material"] in materials[role]] for role in ("validation", "test")}}
                results.append({"group": group, "rotation": rotation, "fold": fold["fold"],
                                "trials": {role: len(part) for role, part in role_rows.items()},
                                "class_counts": {role: counts(part) for role, part in role_rows.items()},
                                "both_classes_all_roles": all(set(counts(part)) == {"0","1"} for part in role_rows.values())})
    return {"fold_geometries": len(results), "feasible_with_both_classes_all_roles": sum(r["both_classes_all_roles"] for r in results),
            "excluded_label_or_record_invariance_checks": metamorphic, "cells": results,
            "scope": "Metadata-only source-construction alternative on previously exposed groupings. Keep all eligible source trials, with no cross-arm class matching. No models, outcome selection, retrospective correction of historical fits, or independent confirmation."}


def gameemo_preparation(rows):
    games = sorted({r["material"] for r in rows})
    subjects = sorted({r["subject_id"] for r in rows})
    # Identity-only alphabetical 7x4 grouping; it is not chosen from label coverage.
    blocks = [subjects[i:i+4] for i in range(0, len(subjects), 4)]
    assignments = []
    for validation, test in itertools.permutations(games, 2):
        allowed = {"train": set(games)-{validation,test}, "validation": {validation}, "test": {test}}
        cohort_feasible = all({r["label"] for r in rows if r["material"] in keys} == {0,1} for keys in allowed.values())
        infeasible = Counter()
        feasible = 0
        for index, block in enumerate(blocks):
            people = {"test": set(block), "validation": set(blocks[(index+1)%len(blocks)])}
            people["train"] = set(subjects)-people["test"]-people["validation"]
            role_counts = {role: counts([r for r in rows if r["material"] in allowed[role] and r["subject_id"] in people[role]]) for role in allowed}
            failures = [role for role,c in role_counts.items() if set(c) != {"0","1"}]
            feasible += not failures
            infeasible.update(failures)
        assignments.append({"training_games": sorted(allowed["train"]), "validation_game": validation, "test_game": test,
                            "cohort_all_roles_have_both_classes": cohort_feasible,
                            "participant_folds_with_both_classes_all_roles": feasible,
                            "infeasible_role_counts": dict(infeasible)})
    return {"assignments": assignments, "global_assignments_with_both_classes": sum(r["cohort_all_roles_have_both_classes"] for r in assignments),
            "assignments_feasible_on_every_participant_fold": sum(r["participant_folds_with_both_classes_all_roles"] == len(blocks) for r in assignments),
            "all_assignments": 12, "participant_folds_per_assignment": 7,
            "fold_cells_with_both_classes": sum(r["participant_folds_with_both_classes_all_roles"] for r in assignments),
            "total_fold_cells": 84,
            "scope": "All 12 game allocations and fixed alphabetical participant grouping, metadata only. Not a chosen train/test design or a score. Whole-cohort coverage does not guarantee fold-wise balanced accuracy is estimable."}


def compute():
    tables = {name: metadata(read_csv(REPO / INPUTS[name]), name) for name in ("DEAP","SEEDIV","GAMEEMO")}
    expected = {"DEAP": (1264,32,40), "SEEDIV": (1080,15,72), "GAMEEMO": (93,28,4)}
    descriptions = {name: describe(rows,name) for name,rows in tables.items()}
    for name,(n,s,m) in expected.items():
        d=descriptions[name]
        if (d["eligible_trials"],d["participants"],d["materials_or_conditions"]) != (n,s,m):
            raise ValueError("Incomplete prior evaluation coverage")
    deap=deap_preparation(tables["DEAP"])
    game=gameemo_preparation(tables["GAMEEMO"])
    return {"development_only": True,"models_fitted": 0,"new_inferences": 0,"research_question_change_approved": False,
            "datasets": descriptions,"strict_deap_source_preparation": deap,"gameemo_class_coverage": game,
            "scope": "Existing public label/identity metadata only; no waveforms, probabilities, optimizer states or new cohort labels used. Coverage means previously evaluated dataset-qualified IDs/material keys, not proof of biological identity across corpora or a claim that unseen EEG information is absent."}


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("build","verify"));args=parser.parse_args()
    result=compute()
    binding={path:sha(REPO/path) for path in INPUTS.values()}
    if args.mode == "build":
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"metadata_findings.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        proof={"passed":True,"script_sha256":sha(Path(__file__)),"input_sha256":binding,
               "findings_sha256":sha(OUT/"metadata_findings.json"),"eligible_original_trials_checked":2437,
               "deap_exclusion_invariance_checks":result["strict_deap_source_preparation"]["excluded_label_or_record_invariance_checks"],
               "scope":"Byte-bound deterministic metadata audit; not source authentication, physical calibration, model performance or confirmation."}
        (OUT/"verification.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    else:
        proof=json.loads((OUT/"verification.json").read_text(encoding="utf-8"))
        if proof["script_sha256"]!=sha(Path(__file__)) or proof["input_sha256"]!=binding or proof["findings_sha256"]!=sha(OUT/"metadata_findings.json"):
            raise ValueError("Changed audit source/input/output")
        if result!=json.loads((OUT/"metadata_findings.json").read_text(encoding="utf-8")):
            raise ValueError("Audit does not reproduce")
    print(json.dumps({"passed":True,"mode":args.mode,"eligible_trials":{k:v["eligible_trials"] for k,v in result["datasets"].items()},
                      "deap_source_cells":len(result["strict_deap_source_preparation"]["cells"]),
                      "gameemo_feasible_fold_cells":result["gameemo_class_coverage"]["fold_cells_with_both_classes"]}))


if __name__=="__main__":main()
