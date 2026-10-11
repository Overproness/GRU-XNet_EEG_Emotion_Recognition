"""Qualify original FACED development targets in the mixed historical cache.

The representation cache also contains explicitly shuffled capacity diagnostics.
Guard their identities, then exclude only the exact declared permutation rows
before reading numeric outcomes. No original data/rating decoder or model I/O.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[1]
BASE = "results/development"
FAILED_PACKET = "faced_eegnet_diagnostic_2026-10-11"
PERMUTED_MODELS = ("capacity_spectral_permuted", "capacity_covariance_permuted")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(row, metadata):
    pair = row["subject"], int(row["vid"])
    if pair not in metadata or row["family"] != metadata[pair]["family"]:
        raise PermissionError("Cached participant/material/family before outcome conversion")
    if "acquisition_group" in row and row["acquisition_group"] != metadata[pair]["acquisition_group"]:
        raise PermissionError("Cached acquisition changed before outcome conversion")
    rotation = int(row["rotation"])
    if rotation not in (0, 1, 2):
        raise PermissionError("Undeclared cached rotation")
    return pair, rotation


def original_target(row):
    target = float(row["target"])
    if not math.isfinite(target) or not 0 <= target <= 7:
        raise ValueError("Invalid original development target")
    return target


def qualify_rows(quality_rows, representation_rows, metadata, source_pairs_by_rotation):
    targets = {}; quality_count = 0; representation_count = 0; original_count = 0
    excluded = {(r, model): set() for r in (0, 1, 2) for model in PERMUTED_MODELS}
    first = None
    for row in quality_rows:
        pair, _ = identity(row, metadata)
        target = original_target(row)
        if pair in targets and targets[pair] != target:
            raise ValueError("Repeated canonical quality-cache targets differ")
        targets[pair] = target; quality_count += 1
    if set(targets) != set(metadata):
        raise PermissionError("Complete canonical development target rectangle required")
    for line, row in enumerate(representation_rows, 2):
        pair, rotation = identity(row, metadata); representation_count += 1
        model, arm = row["model"], row["arm"]
        if arm == "capacity_source_fit" and model in PERMUTED_MODELS:
            # These are intentionally changed labels, never original ratings.
            group = excluded[rotation, model]
            if pair not in source_pairs_by_rotation[rotation] or pair in group:
                raise PermissionError("Changed/duplicate declared permutation diagnostic")
            group.add(pair)
            if first is None:
                first = {"csv_line": line, "subject": pair[0], "vid": pair[1],
                         "rotation": rotation, "model": model, "arm": arm}
            continue  # Do not fetch target or prediction fields in this branch.
        target = original_target(row); original_count += 1
        if targets[pair] != target:
            raise ValueError("Ordinary representation-cache original target differs")
    for (rotation, model), pairs in excluded.items():
        if pairs != set(source_pairs_by_rotation[rotation]):
            raise PermissionError("Complete declared permutation diagnostic rectangle required")
    receipt = {"canonical_targets": len(targets), "quality_rows_identity_guarded": quality_count,
               "representation_rows_identity_guarded": representation_count,
               "ordinary_representation_targets_checked": original_count,
               "ordinary_original_target_conflicts": 0,
               "permutation_rows_excluded_before_numeric_fields": sum(len(p) for p in excluded.values()),
               "permutation_rows_by_model": {model: sum(len(excluded[r, model]) for r in (0, 1, 2)) for model in PERMUTED_MODELS},
               "first_declared_permutation_row": first,
               "quality_cache_is_sole_canonical_target_source": True,
               "all_ordinary_representation_targets_agree_exactly": True,
               "predictions_parsed": False, "original_ratings_or_signals_opened": False,
               "new_targets_decoded": 0, "model_fits": 0, "protected_outcomes_opened": False}
    return targets, receipt


def public_inputs(repo=REPO):
    repo = Path(repo).resolve(strict=True); base = repo / BASE
    activation = load(base / FAILED_PACKET / "activation.json")
    names = ("faced_development_protocol_2026-10-11/plan.json",
             "faced_backup_2026-10-10/participant_reservation.json",
             "faced_codebooks_2026-10-10/audit.json",
             "faced_artifact_control_2026-10-11/source_records.json",
             "faced_artifact_control_2026-10-11/validation_records.json",
             "faced_inference_quality_2026-10-11/predictions.csv",
             "faced_representation_v2_2026-10-11/predictions.csv")
    hashes = {}
    for name in names:
        path = (base / name).resolve(strict=True)
        if not path.is_relative_to(base.resolve()):
            raise PermissionError("Bounded public input path")
        rel = path.relative_to(repo).as_posix(); observed = sha(path)
        if activation["bindings"].get(rel) != observed:
            raise ValueError("Frozen cache/identity metadata bytes changed")
        hashes[rel] = observed
    programme = load(base / names[0]); reservation = load(base / names[1])
    source = set(reservation["development_source"]); validation = set(reservation["development_validation"])
    protected = set(reservation["confirmation"])
    if len(source) != 70 or len(validation) != 20 or len(protected) != 33 or source & validation or (source | validation) & protected:
        raise PermissionError("Changed authoritative participant roles")
    expected = {(r["subject"], r["vid"]) for r in programme["input_access"]["future_development_pairs"]}
    clips = {2, 3, 4, 6, 9, 11, 12, 17, 18, 20, 23, 25, 27, 28}
    if expected != {(p, v) for p in source | validation for v in clips}:
        raise PermissionError("Changed development participants/materials")
    films = {r["clip"]: r for r in load(base / names[2])["stimuli"]}
    records = load(base / names[3])["records"] + load(base / names[4])["records"]
    metadata = {(r["subject"], r["vid"]): {"family": films[r["vid"]]["source_film_sha256"], "acquisition_group": r["acquisition_group"]} for r in records}
    if len(records) != 1260 or set(metadata) != expected:
        raise PermissionError("Changed technical identity rectangle")
    source_pairs = {r["rotation"]: {(p["subject"], p["vid"]) for p in r["arms"]["source_fit"]} for r in programme["rotation_manifests"]}
    if set(source_pairs) != {0, 1, 2} or any(len(p) != 560 for p in source_pairs.values()):
        raise PermissionError("Changed source rotation rectangles")
    return repo, metadata, source_pairs, hashes


def diagnose(repo=REPO):
    repo, metadata, source_pairs, hashes = public_inputs(repo); base = repo / BASE
    with (base / "faced_inference_quality_2026-10-11/predictions.csv").open(encoding="utf-8", newline="") as quality:
        with (base / "faced_representation_v2_2026-10-11/predictions.csv").open(encoding="utf-8", newline="") as representation:
            _, receipt = qualify_rows(csv.DictReader(quality), csv.DictReader(representation), metadata, source_pairs)
    if (receipt["quality_rows_identity_guarded"], receipt["representation_rows_identity_guarded"], receipt["permutation_rows_excluded_before_numeric_fields"]) != (27720, 36120, 3360):
        raise ValueError("Historical cache row counts changed")
    if any(sha(repo / rel) != value for rel, value in hashes.items()):
        raise ValueError("Frozen cache changed during qualification")
    execution = load(base / FAILED_PACKET / "execution.json")
    if execution.get("status") != "held_or_failed" or execution.get("new_neural_fits") != 0:
        raise PermissionError("Diagnosis applies to preserved zero-fit held execution")
    return {"scope": "Public cached-target semantic qualification of zero-fit held execution",
            "failure": execution["failure"], "failed_execution_sha256": sha(base / FAILED_PACKET / "execution.json"),
            "input_hashes": hashes, **receipt,
            "root_cause": "Mixed representation cache includes explicitly target-permuted capacity_source_fit rows; the initial loader and projection auditor compared these diagnostic labels with original ratings before condition selection.",
            "safe_remedy": "In a separately versioned bridge/auditor, guard every identity first, reuse only quality-cache canonical targets, skip only the two exact declared permutation/model arms before numeric fields, and require all remaining real targets to agree exactly. Preserve frozen core, failed packet, original labels, population and twenty-fit settings.",
            "research_question_changed": False, "confirmation_access": False}


class SyntheticQualifierTests(unittest.TestCase):
    def fixture(self):
        metadata = {(f"s{r}", 2): {"family": "film2", "acquisition_group": "modern_1000"} for r in (0, 1, 2)}
        source = {r: {(f"s{r}", 2)} for r in (0, 1, 2)}
        quality = [{"subject": p[0], "vid": p[1], "family": "film2", "rotation": r, "target": "2"} for r, p in enumerate(metadata)]
        representation = [dict(row, model="context_kernel", arm="validation_unseen") for row in quality]
        for r in (0, 1, 2):
            for model in PERMUTED_MODELS:
                representation.append(dict(quality[r], model=model, arm="capacity_source_fit", target="DO NOT READ"))
        return quality, representation, metadata, source

    def test_skip_exact_permutation_before_numeric_conversion(self):
        quality, representation, metadata, source = self.fixture()
        targets, receipt = qualify_rows(quality, representation, metadata, source)
        self.assertEqual(len(targets), 3)
        self.assertEqual(receipt["permutation_rows_excluded_before_numeric_fields"], 6)

    def test_ordinary_original_target_conflict_still_stops(self):
        quality, representation, metadata, source = self.fixture(); representation[0]["target"] = "3"
        with self.assertRaises(ValueError):
            qualify_rows(quality, representation, metadata, source)

    def test_protected_identity_cannot_hide_as_permuted(self):
        quality, representation, metadata, source = self.fixture(); representation[-1]["subject"] = "protected"
        with self.assertRaises(PermissionError):
            qualify_rows(quality, representation, metadata, source)

    def test_wrong_arm_or_missing_permutation_is_not_ignored(self):
        quality, representation, metadata, source = self.fixture(); representation[-1]["arm"] = "source_fit"
        with self.assertRaises(ValueError):
            qualify_rows(quality, representation, metadata, source)
        quality, representation, metadata, source = self.fixture()
        with self.assertRaises(PermissionError):
            qualify_rows(quality, representation[:-1], metadata, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticQualifierTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    receipt = diagnose(args.repo)
    destination = (args.repo / BASE / FAILED_PACKET / "cached_target_diagnosis.json").resolve()
    if destination.parent != (args.repo / BASE / FAILED_PACKET).resolve() or not destination.is_relative_to(args.repo.resolve()):
        raise PermissionError("Diagnosis receipt leaves named public packet")
    if args.check:
        if load(destination) != receipt:
            raise ValueError("Cached-target diagnosis receipt changed")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("canonical_targets", "ordinary_original_target_conflicts", "permutation_rows_excluded_before_numeric_fields", "model_fits")}))


if __name__ == "__main__":
    main()
