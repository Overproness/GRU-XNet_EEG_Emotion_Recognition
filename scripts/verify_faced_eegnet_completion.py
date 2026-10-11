"""Freeze/check completed diagnostic evidence without reopening original data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "results/development/faced_eegnet_diagnostic_v2_2026-10-11"
PREVIOUS = ("faced_representation_v2_2026-10-11", "contribution_decision_2026-10-11", "faced_scope_precision_2026-10-11", "faced_eegnet_preflight_2026-10-11")


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_and_archive():
    count = 0
    for name in PREVIOUS:
        for rel, expected in load(REPO / "results/development" / name / "completion/verification.json")["bindings"].items():
            if sha(REPO / rel) != expected:
                raise ValueError("Preceding scientific bytes changed: " + rel)
            count += 1
    archive = REPO / "docs/paper_archive/2026-10-05-pre-exploration"
    for path, expected in ((REPO.parent / "report.tex", "06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0"), (archive / "report.tex", "06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0"), (archive / "DL-Report.pdf", "921f07f271bc4dadbabaf62ede384c543681b9fe0aec02c490cf7d76ef9de101")):
        if sha(path) != expected:
            raise ValueError("Working/archive manuscript changed")
    return count


def require_completed():
    activation = load(PACKET / "activation.json")
    for group in ("code_bindings", "bindings"):
        for rel, expected in activation[group].items():
            if sha(REPO / rel) != expected:
                raise ValueError("Activated scientific bytes changed: " + rel)
    execution = load(PACKET / "execution.json")
    audit = load(PACKET / "independent_audit.json")
    state = load(PACKET / "checkpoint_state_audit.json")
    portable = load(PACKET / "portability_check.json")
    if execution["status"] != "completed" or execution["new_neural_fits"] != 20 or execution["completed_neural_fits"] != 20:
        raise PermissionError("Complete fixed twenty-fit execution required")
    if audit["passed"] is not True or audit["fits_audited"] != 20 or state["checkpoints_authenticated"] != 60:
        raise PermissionError("Independent prediction and full-state checks required")
    for key, filename in (("execution_sha256", "execution.json"), ("fit_records_sha256", "fit_records.json"), ("selection_sha256", "selection.json"), ("activation_sha256", "activation.json")):
        if state.get(key) != sha(PACKET / filename):
            raise ValueError("State-authenticated public receipt changed: " + filename)
    if portable.get("independent_audit_receipt_reproduced_exactly") is not True or portable.get("fits_audited") != 20:
        raise PermissionError("Public-only audit replay required")
    if portable.get("original_independent_audit_sha256") != sha(PACKET / "independent_audit.json"):
        raise ValueError("Public-only replay audited receipt changed")
    if execution["research_question_changed"] or execution["confirmation_access"] or execution["new_original_rating_decoding"]:
        raise PermissionError("Question/access boundary changed")
    if execution["checkpoint_rows"] != 84000 or execution["ensemble_rows"] != 8400:
        raise ValueError("Incomplete prediction denominator")
    return len(activation["code_bindings"]) + len(activation["bindings"])


def paths():
    names = set()
    for stage in (PACKET, REPO / "results/development/faced_eegnet_diagnostic_2026-10-11"):
        names |= {p.relative_to(REPO).as_posix() for p in stage.rglob("*") if p.is_file() and "completion" not in p.parts and p.suffix in (".json", ".jsonl", ".csv", ".md", ".png", ".svg")}
    activation = load(PACKET / "activation.json")
    names |= set(activation["code_bindings"]) | set(activation["bindings"])
    names |= {"scripts/verify_faced_eegnet_states.py", "scripts/plot_faced_eegnet_diagnostic.py", "scripts/verify_faced_eegnet_portability.py", "scripts/verify_faced_eegnet_completion.py", "scripts/prepare_faced_eegnet_diagnostic_findings.py", "docs/publication/GRU-XNet_FACED_EEGNet_Diagnostic_Findings_2026-10-11.md"}
    return sorted(names)


def check_links(names):
    count = 0
    for rel in names:
        if Path(rel).suffix != ".md":
            continue
        document = REPO / rel
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if re.match(r"(?:https?://|mailto:|#)", raw):
                continue
            target = (document.parent / raw.split("#", 1)[0].strip("<>")).resolve()
            if not target.exists():
                raise ValueError("Broken completed-evidence link: " + rel + ": " + raw)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    destination = PACKET / "completion/verification.json"
    activated = require_completed()
    previous = prior_and_archive()
    if args.declare:
        names = paths()
        receipt = {"date": "2026-10-11", "verified_utc": datetime.now(timezone.utc).isoformat(), "scope": "Completed adaptive FACED EEGNet diagnostic, retained zero-fit hold, independent public metrics/state/portability audits; no adopted contribution or confirmation", "bindings": {rel: sha(REPO / rel) for rel in names}, "activated_scientific_bindings_verified": activated, "preceding_scientific_bindings_verified": previous, "historical_manuscript_source_pdf_unchanged": True, "local_links_checked": check_links(names), "confirmation_access": False, "research_question_changed": False}
        destination.parent.mkdir(exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    else:
        receipt = load(destination)
        for rel, expected in receipt["bindings"].items():
            if sha(REPO / rel) != expected:
                raise ValueError("Completed scientific bytes changed: " + rel)
        if check_links(receipt["bindings"]) != receipt["local_links_checked"]:
            raise ValueError("Completed evidence links changed")
    print(json.dumps({"passed": True, "completed_scientific_bindings": len(receipt["bindings"]), "activated_scientific_bindings": activated, "preceding_scientific_bindings": previous, "manuscript_archive_unchanged": True}))


if __name__ == "__main__":
    main()
