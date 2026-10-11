"""Prepare final findings only after completed execution and both audits.

No current outcomes are accessed by preparing/importing this module. Running
with --write is a distinct, explicit post-audit reporting step, with no fitting,
raw signal/rating I/O, manuscript changes or root status-page edits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.audit_faced_eegnet_diagnostic_v2 import (
    balanced_mae, derive_axes, identity_guard, load, pair,
)

REPO = Path(__file__).resolve().parents[1]
PACKET_NAME = "faced_eegnet_diagnostic_v2_2026-10-11"
REPORT_NAME = "GRU-XNet_FACED_EEGNet_Diagnostic_Findings_2026-10-11.md"
READOUTS = ("eeg_only", "context_residual")
SEEDS = (20261101, 20261102)
LABELS = {"eeg_only": "EEGNet only", "context_residual": "Context + EEGNet residual"}
PUBLIC_URL = "https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet_inputs(repo):
    base = repo/"results/development"; packet = base/PACKET_NAME
    # Audit receipts only exist after completion. Check existence before opening results.
    for name in ("independent_audit.json", "checkpoint_state_audit.json"):
        if not (packet/name).is_file():
            raise PermissionError("Both completed diagnostic/state audits are required before reporting")
    execution = load(packet/"execution.json")
    if execution.get("status") != "completed" or execution.get("failure") is not None or execution.get("fits") != 20 or execution.get("confirmation_access") is not False or execution.get("research_question_changed") is not False:
        raise PermissionError("Complete fixed-grid development evidence required")
    audit = load(packet/"independent_audit.json"); states = load(packet/"checkpoint_state_audit.json")
    if audit.get("passed") is not True or audit.get("fits_audited") != 20 or audit.get("protected_outcomes_opened") is not False or states.get("checkpoints_authenticated") != 60 or states.get("completed_neural_fits") != 20:
        raise PermissionError("Successful twenty-fit/sixty-state audits required")
    if states.get("execution_sha256") != sha(packet/"execution.json") or states.get("fit_records_sha256") != sha(packet/"fit_records.json") or states.get("activation_sha256") != sha(packet/"activation.json"):
        raise ValueError("State-audited input bytes changed before reporting")
    if states.get("confirmation_access") is not False or states.get("research_question_changed") is not False or audit["selection"] != execution["selection"]:
        raise ValueError("Audited selection/paper contract changed")
    technical = load(base/"faced_artifact_control_2026-10-11/source_records.json")["records"]+load(base/"faced_artifact_control_2026-10-11/validation_records.json")["records"]
    axes = derive_axes(load(base/"faced_development_protocol_2026-10-11/plan.json"), load(base/"faced_backup_2026-10-10/participant_reservation.json"), load(base/"faced_backup_2026-10-10/stimulus_metadata.json"), technical, load(base/"faced_inference_quality_2026-10-11/signal_audit.json")["trial_decisions"])
    with (packet/"checkpoint_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    identity_guard(rows, axes)
    fits = load(packet/"fit_records.json")["fits"]
    if len(fits) != 20 or len(rows) != 84000:
        raise ValueError("Completed audit denominators changed")
    return packet, execution, audit, states, axes, fits, rows


def selected_seed_metrics(audit, axes, fits, rows):
    records = {(r["rotation"], r["readout"], r["learning_rate"], r["seed"]): r for r in fits}
    panels = {}
    for row in rows:
        readout = row["readout"]; selected = audit["selection"][readout]
        if row["arm"] != "validation_unseen" or float(row["learning_rate"]) != selected["learning_rate"] or int(row["epoch"]) != selected["epoch"]:
            continue
        key = int(row["rotation"]), readout, int(row["seed"])
        p = pair(row); target = float(row["target"]); raw = float(row["prediction_raw"])
        if not math.isfinite(target) or not 0 <= target <= 7 or not math.isfinite(raw):
            raise ValueError("Invalid audited selected-checkpoint value")
        if p in panels.setdefault(key, {}): raise ValueError("Duplicate selected checkpoint trial")
        panels[key][p] = target, min(7., max(0., raw))
    learning = []
    for rotation in (0, 1, 2):
        for readout in READOUTS:
            rate = audit["selection"][readout]["learning_rate"]
            members = [records[rotation, readout, rate, seed] for seed in SEEDS]
            errors = []
            for seed in SEEDS:
                panel = panels[rotation, readout, seed]
                if set(panel) != axes.arms[rotation]["validation_unseen"]:
                    raise ValueError("Incomplete selected seed rectangle")
                errors.append(balanced_mae(set(panel), {p: v[1] for p, v in panel.items()}, {p: v[0] for p, v in panel.items()}, axes.metadata))
            learning.append({"rotation": rotation, "readout": readout, "baseline_mse": members[0]["starting_control_mse"], "mse60": [r["checkpoints"]["60"]["training_target_source_mse"] for r in members], "mse120": [r["checkpoints"]["120"]["training_target_source_mse"] for r in members], "selected_seed_unseen_mae": errors})
    return learning


def build_report(repo=REPO):
    repo = Path(repo).resolve(); packet, execution, audit, states, axes, fits, rows = packet_inputs(repo)
    learning = selected_seed_metrics(audit, axes, fits, rows)
    indexed = {(r["rotation"], r["readout"]): r for r in audit["rotations"]}
    means = {head: math.fsum(indexed[r, head]["gain"] for r in (0, 1, 2))/3 for head in READOUTS}
    qualifiers = [LABELS[head] for head in READOUTS if audit["engineering_leads"][head]]
    if qualifiers:
        decision = ", ".join(qualifiers)+" passed the fixed engineering continuation screen. This supports preparing a contribution proposal and focused confirmation; it does not establish conference readiness or authorize adopting a new paper question."
    else:
        decision = "Neither readout passed the fixed engineering continuation screen. This diagnostic provides no qualifying lead for encoder-grid expansion. Preserve the complete evidence and reassess the contribution/data/target decision before further fitting."
    url = PUBLIC_URL+"/tree/main/results/development/"+PACKET_NAME
    report = ["# FACED EEGNet diagnostic findings — 11 October 2026", "", decision, "",
              "All 20 declared neural fits completed: rotation 0 used three rates and two seeds for each readout (12 fits), followed by eight unchanged-schedule transfer fits across rotations 1–2. Each fit ran 120 epochs with checkpoints at 30, 60 and 120. The independent evidence audit passed, and all 60 private CPU-loaded checkpoint states passed finite geometry, schedule-counter and norm-constraint authentication.", "",
              "The manuscript question remains unchanged. No confirmation outcomes or original MAT ratings were opened. The 33 reserved participants and reserved materials remain sealed. This is adaptive development on the same 20 validation participants and overlapping material panels; the three rotations provide consistency checks rather than independent replications.", "",
              "| Readout | Selected learning rate | Selected epoch | Mean gain over strongest no-EEG control | Engineering lead |", "|---|---:|---:|---:|---|"]
    for head in READOUTS:
        schedule = audit["selection"][head]
        report.append(f"| {LABELS[head]} | {schedule['learning_rate']:g} | {schedule['epoch']} | {means[head]:+.4f} | {'Yes' if audit['engineering_leads'][head] else 'No'} |")
    report += ["", "Gains are participant/title-family-equal MAE differences in valence units; positive favors EEG. A qualified engineering lead requires positive gain in all three rotations, mean gain ≥0.15, and no learning flags. This threshold is a continuation rule rather than a psychometric minimum or a publishability criterion.", "",
               "| Rotation | Readout | Strongest no-EEG control | Control MAE | EEG readout MAE | Gain | Learning flags |", "|---:|---|---|---:|---:|---:|---|"]
    for r in (0, 1, 2):
        for head in READOUTS:
            entry = indexed[r, head]; strongest = entry["strongest_context"]
            report.append(f"| {r} | {LABELS[head]} | {strongest} | {entry['control_maes'][strongest]:.4f} | {entry['joint_mae']:.4f} | {entry['gain']:+.4f} | {', '.join(entry['learning_flags']) or 'None'} |")
    report += ["", "The stronger comparator is selected from all five declared no-EEG controls: source global/coarse/category medians, the original selected context Ridge, and the fixed context kernel. Rotation 0 alone selected each neural schedule; familiar validation and rotations 1–2 did not select it.", "",
               "| Rotation | Readout | Offset-only source MSE | Source MSE at 60, seeds 1/2 | Source MSE at 120, seeds 1/2 | Chosen-checkpoint unseen MAE, seeds 1/2 |", "|---:|---|---:|---|---|---|"]
    for entry in learning:
        report.append(f"| {entry['rotation']} | {LABELS[entry['readout']]} | {entry['baseline_mse']:.4f} | {' / '.join(f'{v:.4f}' for v in entry['mse60'])} | {' / '.join(f'{v:.4f}' for v in entry['mse120'])} | {' / '.join(f'{v:.4f}' for v in entry['selected_seed_unseen_mae'])} |")
    report += ["", "These source MSE values use unbounded eval-mode checkpoint outputs with the regression training offsets: source median for EEG-only and participant-OOF context for residual training. Full-source-context inference is reported separately. Online normalized/dropout training loss is not this learning diagnostic. Insufficient source learning means the 120-epoch source MSE exceeds 80% of its offset-only baseline; continuing >10% improvement from 60 to 120 flags the fixed budget; chosen-checkpoint seed-MAE difference >0.15 and source-MSE ratio >2 flag seed instability. These engineering flags do not establish biological signal absence or true negative transfer.", "",
               "| Rotation | Readout | Leave-one-family-out gain range | Acquisition-group gain range |", "|---:|---|---|---|"]
    for r in (0, 1, 2):
        for head in READOUTS:
            entry = indexed[r, head]; films = list(entry["descriptive_leave_one_family_out_gain"].values()); acquisitions = [v["gain"] for v in entry["descriptive_acquisition_gain"].values()]
            report.append(f"| {r} | {LABELS[head]} | {min(films):+.4f} to {max(films):+.4f} | {min(acquisitions):+.4f} to {max(acquisitions):+.4f} |")
    report += ["", "Sensitivity summaries are descriptive and retain the full-panel stronger comparator; they do not select another schedule or supply a tuning-corrected confidence interval. The complete group values and familiar/source metrics remain in the audit packet.", "",
               "There were 980 unique source and 280 unique validation trial identities. All 980 source trials and 265 available validation trials passed exact selected-window/QC replay; the 15 fixed unavailable validation trials retained their declared fallback without EEG access. Every rotation retained 560 source-fit trials, 120 unfamiliar-validation trials and 160 familiar-validation trials. The four checkpoint/ensemble arms additionally distinguish source training-offset predictions from source full-context inference. The completed CSV denominators are 84,000 checkpoint rows, 8,400 ensemble rows and 1,800 frozen context-control rows.", ""]
    preflight = load(repo/"results/development/faced_eegnet_preflight_2026-10-11/synthetic_preflight.json")
    hardware = preflight["runtime"]
    report.append(f"The authenticated hardware preflight recorded {hardware.get('gpu_name', 'CUDA GPU')} with {hardware.get('gpu_total_bytes', 0)/1024**3:.2f} GiB device memory. Actual diagnostic peak PyTorch allocator reservation was {execution['peak_cuda_reserved_bytes']/1024**2:.1f} MiB. Conservatively charged neural fitting/checkpoint time was {execution['charged_fit_seconds']/3600:.3f} hours; full execution wall time, including replay and cached joins, was {execution['elapsed_seconds']/3600:.3f} hours. The frozen fit cap was six hours and the reserved-memory cap was 5,500,000,000 bytes. Allocator reservation excludes driver/context overhead and other processes; the hardware name is the preflight receipt rather than a newly asserted device inventory.")
    report += ["", f"Actual runtime: Python {execution['runtime']['python']}, Torch {execution['runtime']['torch']}, CUDA {execution['runtime']['cuda']}; cuDNN deterministic={execution['runtime']['cudnn_deterministic']}, benchmark={execution['runtime']['cudnn_benchmark']}, Torch deterministic algorithms={execution['runtime']['torch_deterministic_algorithms']}.", "",
               "The first attempt remains preserved. It passed source/available-validation replay and stopped before any neural fit because its cache join treated 3,360 deliberately permuted source-capacity control labels as original trial labels. Independent audit reproduced those frozen permutations and found every ordinary cached target consistent. Version 2 uses the quality cache as the authoritative true-target source, skips only the two exact known permutation-control/source-arm combinations before target conversion, and authenticates the kernel's fixed alpha from its bound plan. This was a separately activated implementation correction with the same target, participants, quality policy, grid and question; it was not an outcome-based rescue fit.", "",
               f"Complete evidence: [version 2 packet]({url}), [independent audit]({url.replace('/tree/', '/blob/')}/independent_audit.json), [state authentication]({url.replace('/tree/', '/blob/')}/checkpoint_state_audit.json), and [preserved first hold]({PUBLIC_URL}/blob/main/results/development/faced_eegnet_diagnostic_2026-10-11/held_attempt_audit.json).", ""]
    if (packet/"figures/diagnostic_comparison.png").exists():
        report += [f"![Context and EEG comparison]({url.replace('/tree/', '/raw/')}/figures/diagnostic_comparison.png)", "", f"![Selected two-seed source learning]({url.replace('/tree/', '/raw/')}/figures/selected_learning.png)", ""]
    readme = ["# FACED fixed EEGNet diagnostic — version 2", "", decision, "", "Completed adaptive development; original manuscript question unchanged and confirmation outcomes sealed. Twenty neural fits, sixty checkpoints, 84,000 checkpoint rows and 8,400 ensemble rows; all fixed fallback trials retained.", "",
              "- [Frozen activation](activation.json), [execution](execution.json), [selected schedules](selection.json) and [complete fit histories](fit_records.json).", "- [Source gate](source_gate.json), [feature/window replay manifest](feature_manifest.json), [cached target reuse](targets_reuse.json) and [stronger-control reuse](control_reuse.json).", "- [Independent numerical/identity/control audit](independent_audit.json) and [CPU checkpoint-state authentication](checkpoint_state_audit.json).", "- [Checkpoint predictions](checkpoint_predictions.csv), [ensemble predictions](ensemble_predictions.csv) and [all five frozen controls](controls.csv).", "- [Comparison figure](figures/diagnostic_comparison.png), [selected source learning](figures/selected_learning.png) and [figure input hashes](figures/plot_receipt.json).", "- [Preserved first attempt](../faced_eegnet_diagnostic_2026-10-11/held_attempt_audit.json).", "",
              "Rotation-zero selection reused development outcomes; schedules transferred unchanged to the other two development rotations. Positive/all-rotation mean-gain and learning gates are engineering continuation requirements, not independent confirmation, algorithm novelty or conference readiness. The target is continuous individual retrospective item-10 valence. Known historical permutation-capacity labels remain synthetic controls and never become original labels.", ""]
    return "\n".join(report), "\n".join(readme), {"selection": audit["selection"], "mean_gains": means, "engineering_leads": audit["engineering_leads"], "selected_learning": learning, "inputs": {name: sha(packet/name) for name in ("execution.json", "independent_audit.json", "checkpoint_state_audit.json", "fit_records.json", "checkpoint_predictions.csv")}, "original_files_opened": False, "new_fits": 0, "confirmation_access": False, "research_question_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repo", type=Path, default=REPO); parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.write:
        print("Prepared reporter; no outcome access. Use --write only after explicit completed/audited reporting instruction.")
        return
    repo = args.repo.resolve(); packet = repo/"results/development"/PACKET_NAME
    destinations = (repo.parent/REPORT_NAME, packet/"README.md", packet/"findings_receipt.json")
    if any(path.exists() for path in destinations): raise FileExistsError("Preserve existing findings; no implicit overwrite")
    report, readme, receipt = build_report(repo)
    for path, content in zip(destinations, (report, readme, json.dumps(receipt, indent=2)+"\n")):
        with path.open("x", encoding="utf-8") as stream: stream.write(content)
    print(json.dumps({"status": "findings_written_after_both_audits", "files": [str(p) for p in destinations]}))


if __name__ == "__main__":
    main()
