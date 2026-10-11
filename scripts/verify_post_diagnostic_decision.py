"""Freeze or check the completed public-only decision review and its local links."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.review_post_diagnostic_decision import REPO, PACKET, build, sha
from scripts.verify_faced_eegnet_completion import check_links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    reconstructed = build()
    if json.loads((PACKET / "evidence_inventory.json").read_bytes()) != reconstructed:
        raise ValueError("Public decision inventory differs from reconstruction")
    decision = json.loads((PACKET / "public_dataset_decision.json").read_bytes())
    if decision["author_constraint"] != "Use public datasets only" or decision["research_question_changed"] or decision["confirmation_outcomes_opened"] or decision["new_fits"] or decision["new_predictions"] or decision["original_waveform_or_rating_files_opened"] or decision["new_corpus_body_downloads"]:
        raise PermissionError("Public-only/no-fit/no-pivot boundary changed")
    sources = json.loads((PACKET / "source_review.json").read_bytes())
    dens = sources["dens_metadata_review"]
    if dens["fit_ready"] or dens["waveform_body_bytes_read"] or dens["behavior_or_event_rows_read"] or sources["research_question_changed"]:
        raise PermissionError("DENS metadata-only review boundary changed")
    destination = PACKET / "completion/verification.json"
    if args.declare:
        names = sorted({p.relative_to(REPO).as_posix() for p in PACKET.iterdir() if p.is_file()} | {
            "docs/publication/GRU-XNet_Post_Diagnostic_Contribution_Decision_2026-10-11.md",
            "docs/publication/Contribution_Options_After_EEGNet_2026-10-11.md",
            "scripts/review_post_diagnostic_decision.py",
            "scripts/verify_post_diagnostic_decision.py",
        })
        receipt = {"date": "2026-10-11", "verified_utc": datetime.now(timezone.utc).isoformat(),
                   "scope": "Decision review and primary-source metadata; no new fitting or adopted question",
                   "bindings": {rel: sha(REPO / rel) for rel in names},
                   "preceding_binding_counts": reconstructed["bindings_checked"],
                   "local_links_checked": check_links(names),
                   "public_datasets_only": True, "new_fits": 0,
                   "confirmation_outcomes_opened": 0, "research_question_changed": False}
        destination.parent.mkdir(exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    else:
        receipt = json.loads(destination.read_bytes())
        for rel, expected in receipt["bindings"].items():
            if sha(REPO / rel) != expected:
                raise ValueError("Completed decision bytes changed: " + rel)
        if check_links(receipt["bindings"]) != receipt["local_links_checked"]:
            raise ValueError("Completed decision links changed")
    print(json.dumps({"passed": True, "decision_bindings": len(receipt["bindings"]),
                      "preceding_binding_counts": reconstructed["bindings_checked"],
                      "local_links_checked": receipt["local_links_checked"],
                      "new_fits": 0, "confirmation_outcomes_opened": 0, "research_question_changed": False}))


if __name__ == "__main__":
    main()
