"""Summarize already-public evidence for a decision; never load data or fit models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts import verify_faced_eegnet_completion as preceding


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
PACKET = BASE / "post_diagnostic_decision_2026-10-11"
INPUTS = {
    "earlier_decision": "contribution_decision_2026-10-11/evidence_inventory.json",
    "execution": "faced_eegnet_diagnostic_v2_2026-10-11/execution.json",
    "audit": "faced_eegnet_diagnostic_v2_2026-10-11/independent_audit.json",
    "completion": "faced_eegnet_diagnostic_v2_2026-10-11/completion/verification.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    activated = preceding.require_completed()
    docs = {key: json.loads((BASE / rel).read_bytes()) for key, rel in INPUTS.items()}
    prior = 0
    for name in preceding.PREVIOUS:
        receipt = json.loads((BASE / name / "completion/verification.json").read_bytes())
        for rel, expected in receipt["bindings"].items():
            if sha(REPO / rel) != expected:
                raise ValueError("Preceding scientific evidence changed: " + rel)
            prior += 1
    earlier = docs["earlier_decision"]
    archive = REPO / "docs/paper_archive/2026-10-05-pre-exploration"
    for path, expected in ((archive / "report.tex", earlier["manuscript_source_sha256"]), (archive / "DL-Report.pdf", earlier["archived_pdf_sha256"])):
        if sha(path) != expected:
            raise ValueError("Archived manuscript changed")
    working = REPO.parent / "report.tex"
    if working.exists() and sha(working) != earlier["manuscript_source_sha256"]:
        raise ValueError("Working manuscript changed")
    completed = docs["completion"]
    for rel, expected in completed["bindings"].items():
        if sha(REPO / rel) != expected:
            raise ValueError("Completed diagnostic evidence changed: " + rel)
    if completed["research_question_changed"] or completed["confirmation_access"]:
        raise PermissionError("Preceding access/question boundary changed")
    if earlier["research_question_changed"] or earlier["confirmation_outcomes_opened"]:
        raise PermissionError("Earlier decision has a different scope")
    audit = docs["audit"]
    if audit["passed"] is not True or audit["fits_audited"] != 20:
        raise PermissionError("Independent completed numerical audit required")
    panels = {(row["rotation"], row["readout"]): row for row in audit["rotations"]}
    readouts = ("eeg_only", "context_residual")
    if len(audit["rotations"]) != 6 or set(panels) != {(r, h) for r in (0, 1, 2) for h in readouts}:
        raise ValueError("Changed diagnostic panel identities")
    results = {}
    for head in readouts:
        rows = [panels[r, head] for r in (0, 1, 2)]
        gains = [float(row["gain"]) for row in rows]
        if not all(math.isfinite(g) for g in gains):
            raise ValueError("Nonfinite aggregate input")
        results[head] = {
            "gains_rotations_0_1_2": gains,
            "mean_gain": math.fsum(gains) / 3,
            "positive_rotations": sum(g > 0 for g in gains),
            "learning_flags_by_rotation": {str(r): rows[r]["learning_flags"] for r in (0, 1, 2)},
            "selected_schedule": audit["selection"][head],
            "engineering_lead": audit["engineering_leads"][head],
        }
    any_lead = any(row["engineering_lead"] for row in results.values())
    flagged = sum(bool(row["learning_flags"]) for row in panels.values())
    return {
        "date": "2026-10-11",
        "scope": "Post-diagnostic decision summary of public audited aggregates; no new numerical experiment",
        "inputs": {"results/development/" + rel: sha(BASE / rel) for rel in INPUTS.values()},
        "script_sha256": sha(Path(__file__)),
        "bindings_checked": {"activated": activated, "preceding": prior, "completed_diagnostic": len(completed["bindings"])},
        "eegnet": results,
        "flagged_panels": flagged,
        "rows_in_preceding_numerical_audit": {"checkpoint": audit["checkpoint_rows"], "ensemble": audit["ensemble_rows"], "context_controls": audit["control_rows"]},
        "earlier_faced_selected_unseen_mae": earlier["faced_preceding_selected_unseen_mae"],
        "earlier_covariance_lead_screens": earlier["faced_lead_screens"],
        "earlier_covariance_minus_spectral_joint_mae": earlier["new_covariance_joint_minus_preceding_spectral_joint_mae"],
        "participant_roles": earlier["faced_roles"],
        "reserved_geometry_metadata_only": earlier["faced_confirmation_geometry"],
        "decision": {
            "qualified_local_target": "FACED individual retrospective item-10 valence, continuous 0–7, existing non-neutral scope",
            "same_panel_encoder_expansion": "no_go_without_new_scientific_rationale" if not any_lead else "reconsider_with_a_supported_proposal",
            "supported_paper_pivot_from_this_grid": False,
            "confirmation_opening": "hold",
            "external_individual_rating_replication": "conditional_on_original_codebook_identity_endpoint_and_material_overlap_qualification",
            "original_joint_architecture_or_negative_transfer_claim": "not_supported_by_this_single_corpus_diagnostic",
            "global_neural_information_null": "not_supported",
        },
        "new_fits": 0, "new_predictions": 0, "original_recordings_opened": 0,
        "new_original_rating_decoding": 0, "confirmation_outcomes_opened": 0,
        "research_question_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build()
    destination = PACKET / "evidence_inventory.json"
    if args.check:
        if json.loads(destination.read_bytes()) != result:
            raise ValueError("Decision inventory changed")
    else:
        PACKET.mkdir(exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "check_only": args.check, "bindings_checked": result["bindings_checked"], "flagged_panels": result["flagged_panels"], "new_fits": 0, "confirmation_outcomes_opened": 0, "research_question_changed": False}))


if __name__ == "__main__":
    main()
