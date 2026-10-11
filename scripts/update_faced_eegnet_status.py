"""Prepare/apply gated factual updates to three mutable publication status files.

Default invocation previews the completed-stage paragraphs. --apply writes only
the named mutable files, preserving their preceding mixed-newline body bytes.
No producing code, frozen protocol, waveform, rating or outcome CSV is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKET = "results/development/faced_eegnet_diagnostic_v2_2026-10-11"
DEFAULT_REPORT = "GRU-XNet_FACED_EEGNet_Diagnostic_Findings_2026-10-11.md"
READOUTS = ("eeg_only", "context_residual")
NAMES = {"eeg_only": "EEG-only", "context_residual": "context-plus-EEG"}
FLAGS = {"insufficient_source_learning": "insufficient source learning",
         "budget_limited_convergence": "continued source improvement at epoch 120",
         "validation_seed_divergence": "validation seed disagreement",
         "source_seed_divergence": "source seed disagreement",
         "invalid_or_nonfinite_learning_metrics": "invalid learning metrics"}
OLD_LABEL = b"**Latest completed closest-method/EEGNet preflight,"
PRECEDING_LABEL = b"**Preceding completed closest-method/EEGNet preflight,"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet_path(repo, basename):
    packet = (repo / PACKET).resolve(strict=True)
    if not packet.is_relative_to(repo):
        raise PermissionError("Public diagnostic packet leaves repository")
    path = (packet / basename).resolve(strict=True)
    if path.parent != packet or not path.is_file():
        raise PermissionError("Required receipt leaves public packet")
    return path


def completed_evidence(repo, report):
    # Completion gate precedes all numerical-audit result access.
    execution_path = packet_path(repo, "execution.json"); execution = load(execution_path)
    if execution.get("status") != "completed" or execution.get("failure") is not None:
        raise PermissionError("No status update before completed diagnostic")
    if (execution.get("fits"), execution.get("new_neural_fits"), execution.get("completed_neural_fits")) != (20, 20, 20):
        raise PermissionError("Require all twenty completed neural fits")
    if execution.get("confirmation_access") is not False or execution.get("research_question_changed") is not False:
        raise PermissionError("Confirmation/manuscript contract changed")
    if not report.is_file() or report.parent != repo.parent or report.suffix != ".md" or report.stat().st_size < 100:
        raise PermissionError("Completed root findings report required")
    audit_path = packet_path(repo, "independent_audit.json"); audit = load(audit_path)
    state_path = packet_path(repo, "checkpoint_state_audit.json"); states = load(state_path)
    if audit.get("passed") is not True or (audit.get("fits_audited"), audit.get("checkpoint_rows"), audit.get("ensemble_rows"), audit.get("control_rows"), audit.get("development_targets")) != (20, 84000, 8400, 1800, 1260):
        raise PermissionError("Successful complete independent numerical audit required")
    if audit.get("source_gate_verified") is not True or (audit.get("replay_manifest_trials"), audit.get("replay_available_trials")) != (1260, 1245) or audit.get("protected_outcomes_opened") is not False:
        raise PermissionError("Complete source/inference/protected audit required")
    if (states.get("completed_neural_fits"), states.get("checkpoints_authenticated"), states.get("expected_trainable_parameters")) != (20, 60, 1841):
        raise PermissionError("Successful complete sixty-state audit required")
    if states.get("confirmation_access") is not False or states.get("research_question_changed") is not False:
        raise PermissionError("State-audit protected/manuscript scope changed")
    for field, basename in (("execution_sha256", "execution.json"), ("fit_records_sha256", "fit_records.json"),
                            ("selection_sha256", "selection.json"), ("activation_sha256", "activation.json")):
        if states.get(field) != sha(packet_path(repo, basename)):
            raise ValueError("State-audit public receipt binding changed")
    metadata = load(packet_path(repo, "feature_manifest.json"))
    if (metadata.get("source_trials"), metadata.get("validation_trials"), metadata.get("validation_eeg_available"), metadata.get("validation_eeg_unavailable"), metadata.get("windows_in_memory")) != (980, 280, 265, 15, 1245):
        raise PermissionError("Changed replay/fallback denominator")
    panels = {(r["rotation"], r["readout"]): r for r in audit["rotations"]}
    expected = {(r, h) for r in (0, 1, 2) for h in READOUTS}
    if len(audit["rotations"]) != 6 or set(panels) != expected:
        raise ValueError("Complete six-panel result audit required")
    original = {(r["rotation"], r["readout"]): r for r in execution["rotations"]}
    if set(original) != expected or len(execution["rotations"]) != 6:
        raise ValueError("Changed reported diagnostic panels")
    for key, row in panels.items():
        if not math.isfinite(float(row["gain"])) or any(f not in FLAGS for f in row["learning_flags"]):
            raise ValueError("Invalid gain or undocumented learning flag")
        if abs(float(row["gain"]) - float(original[key]["gain"])) > 2e-9 or row["learning_flags"] != original[key]["learning_flags"]:
            raise ValueError("Reported/audited gain or learning flags changed")
    if audit["engineering_leads"] != execution["engineering_leads"] or set(audit["engineering_leads"]) != set(READOUTS):
        raise ValueError("Changed engineering-screen conclusion")
    bindings = {str(path.relative_to(repo)): sha(path) for path in (execution_path, audit_path, state_path)}
    bindings[str(report.name)] = sha(report)
    return execution, audit, panels, bindings


def result_prose(panels, audit):
    results = []; warnings = []
    for head in READOUTS:
        values = [float(panels[r, head]["gain"]) for r in (0, 1, 2)]
        results.append(f"{NAMES[head]} gains over the strongest no-EEG comparator are " +
                       "/".join(f"{v:+.5f}" for v in values) + f" MAE across rotations 0/1/2 (mean {math.fsum(values)/3:+.5f})")
        flagged = {}
        for r in (0, 1, 2):
            for flag in panels[r, head]["learning_flags"]:
                flagged.setdefault(flag, []).append(r)
        if flagged:
            text = "; ".join(f"{FLAGS[flag]} (rotation{'s' if len(rotations)>1 else ''} " +
                             ", ".join(map(str, rotations)) + ")" for flag, rotations in flagged.items())
            warnings.append(f"{NAMES[head]} learning flags: {text}")
        else:
            warnings.append(f"{NAMES[head]} has no declared learning flag")
    leads = [NAMES[h] for h in READOUTS if audit["engineering_leads"][h]]
    if leads:
        conclusion = ("The fixed engineering screen qualifies " + " and ".join(leads) +
                      " for a findings-backed proposal. Next: present that proposal for author review before any larger programme.")
    else:
        conclusion = ("Neither readout meets the fixed engineering screen; encoder expansion and confirmation stay held. "
                      "Next: an evidence-backed contribution and dataset/target decision using these findings.")
    return ". ".join(results) + ". " + ". ".join(warnings) + ". " + conclusion


def prepend_preserving_history(original, paragraph):
    bom = b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    body = original[len(bom):]
    if not body.startswith(b"# ") or b"\n" not in body:
        raise ValueError("Expected maintained Markdown H1")
    end = body.index(b"\n") + 1; heading = body[:end]; remainder = body[end:]
    newline = b"\r\n" if heading.endswith(b"\r\n") else b"\n"
    while remainder.startswith((b"\r\n", b"\n")):
        step = 2 if remainder.startswith(b"\r\n") else 1
        heading += remainder[:step]; remainder = remainder[step:]
    encoded = paragraph.encode("utf-8") + newline + newline
    if remainder.startswith(encoded):
        return original
    marker = b"**Latest completed FACED EEGNet diagnostic,"
    if remainder.startswith(marker):
        raise ValueError("Preserve an existing differing current-stage status")
    if remainder.count(OLD_LABEL) > 1:
        raise ValueError("Multiple preceding Latest labels need explicit review")
    remainder = remainder.replace(OLD_LABEL, PRECEDING_LABEL, 1)
    return bom + heading + encoded + remainder


def proposed_updates(repo=REPO, report=None):
    repo = Path(repo).resolve(strict=True)
    report = Path(report or repo.parent / DEFAULT_REPORT)
    if not report.is_absolute():
        report = repo.parent / report
    report = report.resolve(strict=True)
    execution, audit, panels, evidence = completed_evidence(repo, report)
    completed = datetime.fromisoformat(execution["completed_utc"].replace("Z", "+00:00"))
    if completed.tzinfo is None:
        raise ValueError("Timezone-aware completion receipt required")
    date = completed.astimezone(timezone(timedelta(hours=5))).strftime("%d %B %Y").lstrip("0")
    packet = (repo / PACKET).resolve()
    index = "README.md" if (packet / "README.md").is_file() else "independent_audit.json"
    raw_packet = f"{PACKET}/{index}"
    report_name = report.name
    common = ("All 20 fixed neural fits completed; 1,245 original trial signals passed exact source/window/QC replay "
              "(980 source, 265 validation), while all fifteen unavailable validation trials retain fallback. "
              "Independent numerical and sixty-checkpoint CPU state audits pass. " + result_prose(panels, audit) +
              " This is adaptive reused development, without independent confirmation or established novelty. "
              "The zero-fit cache-join hold and its separately versioned correction remain preserved; original targets and scientific settings are unchanged. "
              "Manuscript and research question remain unchanged; protected outcomes remain sealed. Any paper pivot requires explicit author approval and a fresh manuscript archive.")
    specifications = (
        (repo.parent / "GRU-XNet_Publication_Readiness_2026-10-05.md", report_name, f"{repo.name}/{raw_packet}"),
        (repo / "PUBLICATION.md", f"docs/publication/{report_name}", raw_packet),
        (repo / "docs/publication/Backup_Qualification_Protocol_2026-10-10.md", report_name, "../../" + raw_packet),
    )
    updates = []
    for path, report_link, packet_link in specifications:
        path = path.resolve(strict=True)
        if not path.is_relative_to(repo.parent) or not path.is_file():
            raise PermissionError("Mutable status path leaves workspace")
        original = path.read_bytes()
        paragraph = f"**Latest completed FACED EEGNet diagnostic, {date}:** [findings]({report_link}) and [audited packet]({packet_link}). " + common
        updated = prepend_preserving_history(original, paragraph)
        updates.append({"path": path, "original": original, "updated": updated, "paragraph": paragraph})
    return updates, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply the three gated mutable-document updates")
    args = parser.parse_args(); updates, evidence = proposed_updates(args.repo, args.report)
    if args.apply:
        if any(item["path"].read_bytes() != item["original"] for item in updates):
            raise ValueError("Mutable document changed after update preparation")
        for item in updates:
            if item["updated"] != item["original"]:
                item["path"].write_bytes(item["updated"])
        print(json.dumps({"updated_files": sum(item["updated"] != item["original"] for item in updates),
                          "paths": [str(item["path"]) for item in updates], "evidence_sha256": evidence}))
    else:
        for item in updates:
            print(str(item["path"]) + "\n" + item["paragraph"] + "\n")


if __name__ == "__main__":
    main()
