"""Reproduce the completed FACED diagnostic audit from bounded public files.

No raw data, private caches, manuscript or private model states are copied.
The subprocess runs only the pinned independent numerical auditor, never the
producing runner, decoder, target reader or training engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
BASE = "results/development"
VERSIONS = {
    "faced_eegnet_diagnostic_v2_2026-10-11": "scripts/audit_faced_eegnet_diagnostic_v2.py",
    "faced_eegnet_diagnostic_2026-10-11": "scripts/audit_faced_eegnet_diagnostic.py",
}
HISTORICAL = (("faced_representation_v2_2026-10-11", 53),
              ("contribution_decision_2026-10-11", 7),
              ("faced_scope_precision_2026-10-11", 15),
              ("faced_eegnet_preflight_2026-10-11", 19))
PACKET_FILES = ("activation.json", "execution.json", "source_gate.json", "targets_reuse.json",
                "feature_manifest.json", "control_reuse.json", "controls.csv", "checkpoint_predictions.csv",
                "ensemble_predictions.csv", "fit_records.json", "selection.json", "independent_audit.json")
SUFFIXES = {".py", ".md", ".json", ".csv", ".png", ".svg"}
FORBIDDEN_PARTS = {"Data", "Code", "publication_runs", "paper_archive", ".git", ".codex", ".agents", ".aws"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def public_source(repo, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise PermissionError("Require a normalized relative public path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or ".." in parts or PurePosixPath(relative).as_posix() != relative or set(parts) & FORBIDDEN_PARTS:
        raise PermissionError("Private, archive or nonrelative copy path")
    allowed = (len(parts) >= 3 and parts[:2] in (("results", "development"), ("docs", "publication"))) or (len(parts) >= 2 and parts[0] in ("scripts", "gruxnet", "tests"))
    if not allowed or PurePosixPath(relative).suffix not in SUFFIXES:
        raise PermissionError("Nonpublic/unsupported file type")
    repo = Path(repo).resolve(strict=True); path = (repo / relative).resolve(strict=True)
    if not path.is_relative_to(repo) or not path.is_file() or path.stat().st_size > 50_000_000:
        raise PermissionError("Missing, oversized or escaped public artifact")
    return path


def completion_guard(execution):
    if execution.get("status") != "completed" or execution.get("failure") is not None:
        raise PermissionError("No portability/outcome audit before completed execution")
    if (execution.get("fits"), execution.get("new_neural_fits"), execution.get("completed_neural_fits")) != (20, 20, 20):
        raise PermissionError("Require complete twenty-fit diagnostic")
    if execution.get("confirmation_access") is not False or execution.get("research_question_changed") is not False:
        raise PermissionError("Protected/manuscript contract changed")


def packet_name(repo, selected=None):
    if selected is not None:
        if selected not in VERSIONS:
            raise PermissionError("Unknown versioned public packet")
        return selected
    for name in VERSIONS:
        if (repo / BASE / name / "activation.json").is_file():
            return name
    raise FileNotFoundError("No versioned diagnostic activation")


def prepare_manifest(repo, selected=None):
    repo = Path(repo).resolve(strict=True); name = packet_name(repo, selected)
    prefix = f"{BASE}/{name}"
    execution_path = public_source(repo, prefix + "/execution.json")
    execution = load(execution_path)
    # This gate precedes all outcome CSV reads, copies or numerical subprocesses.
    completion_guard(execution)
    activation_rel = prefix + "/activation.json"
    activation_path = public_source(repo, activation_rel); activation = load(activation_path)
    if execution.get("activation_sha256") != sha(activation_path) or activation.get("status") != "active_development_diagnostic":
        raise PermissionError("Execution activation changed")
    if not activation.get("bindings") or not activation.get("code_bindings"):
        raise PermissionError("Missing frozen input/code bindings")
    hashes = {}; counts = {}

    def bind(relative, expected):
        value = expected.get("sha256") if isinstance(expected, dict) else expected
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Malformed public SHA binding")
        path = public_source(repo, relative)
        if sha(path) != value or (relative in hashes and hashes[relative] != value):
            raise ValueError("Changed/conflicting bound public artifact: " + relative)
        hashes[relative] = value

    for category in ("bindings", "input_bindings", "code_bindings"):
        values = activation.get(category, {})
        counts[category] = len(values)
        for relative, expected in values.items():
            bind(relative, expected)
    preflight_rel = f"{BASE}/faced_eegnet_preflight_2026-10-11/plan.json"
    bind(preflight_rel, activation["preflight_plan_sha256"])
    preflight = load(public_source(repo, preflight_rel))
    for relative, expected in preflight["bindings"].items():
        bind(relative, expected)
    historical_count = 0
    for historical, expected_count in HISTORICAL:
        relative = f"{BASE}/{historical}/completion/verification.json"
        receipt_path = public_source(repo, relative); receipt = load(receipt_path)
        if len(receipt["bindings"]) != expected_count:
            raise ValueError("Preceding scientific receipt count changed")
        for bound, expected in receipt["bindings"].items():
            bind(bound, expected); historical_count += 1
        bind(relative, sha(receipt_path))
    if historical_count != 94:
        raise ValueError("All ninety-four preceding scientific bindings required")
    for basename in PACKET_FILES:
        relative = f"{prefix}/{basename}"
        bind(relative, sha(public_source(repo, relative)))
    independent = load(public_source(repo, prefix + "/independent_audit.json"))
    if independent.get("passed") is not True or independent.get("fits_audited") != 20:
        raise PermissionError("Completed original public independent audit required")
    auditor = VERSIONS[name]
    if auditor not in activation["code_bindings"]:
        raise PermissionError("Numerical auditor is absent from frozen activation")
    # Copy this helper too, independently of its later completion-receipt binding.
    helper = "scripts/verify_faced_eegnet_portability.py"
    bind(helper, sha(public_source(repo, helper)))
    return name, hashes, {"activation_binding_counts": counts,
                          "preflight_input_bindings_verified": len(preflight["bindings"]),
                          "preceding_scientific_bindings_verified": historical_count,
                          "original_independent_audit_sha256": sha(public_source(repo, prefix + "/independent_audit.json"))}


def verify_portability(repo=REPO, selected=None):
    repo = Path(repo).resolve(strict=True)
    name, hashes, evidence = prepare_manifest(repo, selected)
    target = repo.parent / ("_egdiag_portable_v2" if "_v2_" in name else "_egdiag_portable_v1")
    if target.resolve().parent != repo.parent:
        raise PermissionError("Portable target leaves named workspace directory")
    target.mkdir(exist_ok=True)
    if target.resolve().parent != repo.parent:
        raise PermissionError("Resolved portable target escaped workspace")
    copied = {}
    for relative, expected in sorted(hashes.items()):
        source = public_source(repo, relative); destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.resolve().is_relative_to(target.resolve()):
            raise PermissionError("Resolved copy destination escaped portable directory")
        if destination.exists():
            if sha(destination) != expected:
                raise PermissionError("Preserve an existing portable file with differing bytes")
        else:
            with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                while block := input_stream.read(1024 * 1024):
                    output_stream.write(block)
        if sha(destination) != expected or sha(source) != expected:
            raise ValueError("Public-copy byte integrity changed")
        copied[relative] = expected
    # Refuse pre-existing private/data/archive artifacts in this dedicated copy.
    for path in target.rglob("*"):
        if path.is_file() and path.relative_to(target).as_posix() not in copied:
            raise PermissionError("Unexpected file in dedicated public-only portable copy")
    arguments = [VERSIONS[name], "--repo", ".", "--check"]
    process = subprocess.run([sys.executable, *arguments], cwd=target, capture_output=True,
                             text=True, encoding="utf-8", timeout=240)
    if process.returncode:
        raise RuntimeError("Portable independent audit failed: " + process.stdout + process.stderr)
    result = json.loads(process.stdout)
    if result.get("passed") is not True or (result.get("checkpoint_rows"), result.get("ensemble_rows")) != (84000, 8400):
        raise ValueError("Portable numerical audit incomplete")
    for relative, expected in copied.items():
        if sha(public_source(repo, relative)) != expected or sha(target / relative) != expected:
            raise ValueError("Source/copy changed during portable numerical audit")
    return {"scope": "Exact-byte public-only standalone numerical audit after completed execution",
            "packet": name, "portable_directory": target.name, "copied_public_files": len(copied),
            "files_sha256": copied, **evidence,
            "subprocess": {"arguments": arguments, "exit_code": process.returncode,
                           "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
                           "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest()},
            "independent_audit_receipt_reproduced_exactly": True,
            "fits_audited": result["fits_audited"], "checkpoint_rows": result["checkpoint_rows"],
            "ensemble_rows": result["ensemble_rows"], "control_rows": result["control_rows"],
            "protected_outcomes_opened": False, "confirmation_access": False,
            "private_data_or_cache_copied": False, "manuscript_or_archive_copied": False,
            "original_waveforms_or_ratings_decoded": False, "training_executed": False,
            "private_state_files_copied": False, "research_question_changed": False}


class SyntheticPortabilityTests(unittest.TestCase):
    def test_incomplete_execution_rejected(self):
        for status in ("running", "held_or_failed"):
            with self.assertRaises(PermissionError):
                completion_guard({"status": status})

    def test_complete_execution_gate(self):
        completion_guard({"status": "completed", "failure": None, "fits": 20,
                          "new_neural_fits": 20, "completed_neural_fits": 20,
                          "confirmation_access": False, "research_question_changed": False})

    def test_private_archive_and_escape_paths_rejected(self):
        for relative in ("../Data/data.bdf", "Data/data.bdf", "publication_runs/a.json",
                         "docs/paper_archive/report.tex", "results/development/x/state.pt",
                         "/scripts/a.py", "scripts/../a.py", "C:secret.json"):
            with self.assertRaises(PermissionError):
                public_source(REPO, relative)

    def test_existing_public_code_path(self):
        self.assertEqual(public_source(REPO, "scripts/verify_faced_eegnet_portability.py"), Path(__file__).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--packet-name", choices=list(VERSIONS))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticPortabilityTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    receipt = verify_portability(args.repo, args.packet_name)
    destination = args.repo / BASE / receipt["packet"] / "portability_check.json"
    if args.check:
        if load(destination) != receipt:
            raise ValueError("Portable audit receipt changed")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("copied_public_files", "preceding_scientific_bindings_verified", "checkpoint_rows", "ensemble_rows", "private_data_or_cache_copied")}))


if __name__ == "__main__":
    main()
