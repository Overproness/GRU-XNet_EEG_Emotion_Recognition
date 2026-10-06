"""Export a bounded, portable review checkpoint; never copy raw EEG or credentials."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
DATE = "2026-10-05"
REPORTS = [f"GRU-XNet_{name}_{DATE}.md" for name in (
    "Publication_Review", "Implementation_Status", "Dataset_Provenance",
    "First_Party_DEAP_Check", "GitHub_Configuration_Review", "DEAP_Control",
    "Publication_Readiness", "Exploration_Findings", "Neural_Transfer_Investigation", "Multitarget_Transfer_Findings")]
REPORTS += ["GRU-XNet_Transformer_Native_Label_Findings_2026-10-06.md",
            "GRU-XNet_Transformer_Research_Update_2026-10-06.md",
            "GRU-XNet_Session_Pretraining_Findings_2026-10-06.md",
            "GRU-XNet_Within_Session_Material_Findings_2026-10-06.md",
            "GRU-XNet_Material_Generalization_Research_Update_2026-10-06.md"]
REPORTS += ["GRU-XNet_Repeated_Material_Findings_2026-10-06.md"]
RUN_FILES = {
    "config.json", "history.json", "split_audit.json", "test_metrics.json",
    "best_validation_metrics.json", "test_trial_predictions.csv", "verification.json",
    "validation_selection.json", "normalizer_fit.json", "training_metrics.json",
    "test_subject_metrics.csv", "development_comparison.json", "normalization_comparison.json",
    "training.png", "test_confusion.png", "development_diagnostics.png",
}
EXTRA_FILES = [
    "repeated_material_plan_2026-10-06.json", "cache_temporal_deap/prepared.json", "cache_temporal_deap/raw_reproduction.json",
    "repeated_material_seediv_feasibility_2026-10-06.json", "repeated_material_deap_feasibility_2026-10-06.json",
    *[f"repeated_material_{dataset}/{name}" for dataset in ("seediv", "deap") for name in ("config.json", "plan.json", "feasibility.json", "model_index.json", "comparison.json", "verification.json", "analysis_verification.json", "label_heterogeneity.json", "environment.json", "comparison.png", "paired_comparison.png", "training_diagnostics.csv", "all_cells.csv", "all_groups.csv", "README.md")],
    *[f"repeated_material_{dataset}/{kind}_group{group}.{ext}" for dataset in ("seediv", "deap") for group in (1, 2) for kind, ext in (("folds", "json"), ("material_assignments", "csv"))],
    *[f"repeated_material_{dataset}/model_metrics_{model}_group{group}.json" for dataset in ("seediv", "deap") for model in ("mean_mlp", "transformer") for group in (1, 2)],
    *[f"repeated_material_{dataset}/predictions_{model}_{arm}_group{group}.csv" for dataset, models in (("seediv", ("mean_mlp", "transformer", "bandpower", "duration", "reve_pretrained", "reve_random42")), ("deap", ("mean_mlp", "transformer", "bandpower", "material_prior"))) for model in models for arm in ("exposed", "unexposed") for group in (1, 2)],
    *[f"repeated_material_{dataset}/linear_{model}_group{group}_session{session}_rotation{rotation}.json" for dataset, models, sessions, rotations in (("seediv", ("bandpower", "duration", "reve_pretrained", "reve_random42"), (1, 2, 3), (0, 1, 2)), ("deap", ("bandpower", "material_prior"), (1,), (0, 1, 2, 3, 4))) for model in models for group in (1, 2) for session in sessions for rotation in rotations],
    "material_prior_source_audit_2026-10-06/manifest.json",
    "material_population_audit/feasibility.json", "material_population_audit/material_label_counts.csv", "material_population_audit/verification.json",
    "within_session_material_plan_2026-10-06.json",
    *[f"within_session_material_seediv/{name}" for name in ("config.json","plan.json","folds.json","model_index.json","material_assignments.csv","comparison.json","verification.json","environment.json","comparison.png","paired_comparison.png","training_diagnostics.csv","all_cells.csv","all_seeds.csv","README.md")],
    *[f"within_session_material_seediv/model_metrics_{model}_session{session}.json" for model in ("mean_mlp","transformer") for session in (1,2,3)],
    *[f"within_session_material_seediv/predictions_{model}_{arm}.csv" for model in ("mean_mlp","transformer","bandpower","duration","reve_pretrained","reve_random42") for arm in ("exposed","unexposed")],
    *[f"within_session_material_seediv/linear_{model}_session{session}_rotation{rotation}.json" for model in ("bandpower","duration","reve_pretrained","reve_random42") for session in (1,2,3) for rotation in (0,1,2)],
    "session_material_sensitivity_plan_2026-10-06.json",
    *[f"session_material_sensitivity/{name}" for name in ("comparison.json","verification.json","comparison.png","session_cells.png")],
    "reve_probe_plan_2026-10-06.json", "cache_reve_input_seediv/prepared.json",
    *[f"reve_audit_2026-10-06/{name}" for name in ("download_manifest.json", "code_review.json", "feasibility.json")],
    *[f"reve_frozen_seediv/{name}" for name in ("plan.json", "features.json", "folds.json", "comparison.json", "verification.json", "verification_diagnostic.json", "environment.json", "comparison.png")],
    *[f"reve_frozen_seediv/linear_{model}_source{source}.json"
      for model in ("reve_pretrained","reve_random42") for source in (0,1,2,3)],
    *[f"reve_frozen_seediv/predictions_{model}.csv" for model in ("reve_pretrained","reve_random42")],
    "session_stimulus_plan_2026-10-06.json",
    *[f"session_stimulus_seediv/{name}" for name in (
        "config.json", "plan.json", "folds.json", "model_metrics.json", "comparison.json", "verification.json", "environment.json", "comparison.png")],
    *[f"session_stimulus_seediv/predictions_{model}_source{source}.csv"
      for model in ("mean_mlp","transformer") for source in (1,2,3)],
    *[f"session_stimulus_seediv/linear_{model}_source{source}.json"
      for model in ("bandpower","duration") for source in (1,2,3)],
    *[f"session_stimulus_seediv/predictions_{model}.csv" for model in ("bandpower","duration")],
    "temporal_native_plan_2026-10-06.json", "cache_temporal_native_seediv/prepared.json",
    *[f"temporal_native_seediv/{name}" for name in (
        "config.json", "plan.json", "folds.json", "model_metrics.json", "comparison.json",
        "linear_predictions.csv", "linear_models.json", "verification.json", "environment.json", "comparison.png", "paired_comparison.png")],
    *[f"temporal_native_seediv/predictions_{architecture}_{representation}_{objective}.csv"
      for architecture in ("mean_mlp", "transformer") for representation in ("absolute", "relative")
      for objective in ("coarse3", "native4")],
    "deap_control_plan_2026-10-05.json", "exploration_plan_2026-10-05.json",
    "joint_seediv_control_plan_2026-10-05.json",
    "neural_negative_transfer_plan_2026-10-05.json",
    "gradient_conflict_plan_2026-10-05.json",
    "multitarget_transfer_plan_2026-10-05.json", "cache_common14_trial_features/prepared.json",
    "deap32_cache_consistency.json", "cache_common14/prepared.json", "cache_deap32/prepared.json",
    "provenance/upstream_full_integrity_comparison.json", "provenance/upstream_sample_comparison.json",
    "provenance/deap_change_summary.json", "provenance/existing_cache_binding.json",
    "provenance/first_party_deap/first_party_attempt.json",
    "provenance/first_party_deap/official_documentation_check.json",
    "provenance/first_party_deap/verified_tls_routes/route_checks.json",
    "repository_review_2026-10-05/source_file_manifest.json",
    "native_seediv_diagnostic/comparison.json", "native_seediv_diagnostic/folds.json",
    "native_seediv_diagnostic/config.json", "native_seediv_diagnostic/trial_predictions.csv",
    "native_seediv_diagnostic/verification.json",
    "native_seediv_diagnostic/reproduction.json", "native_seediv_diagnostic/comparison.png",
    "cache_native_seediv_features/prepared.json",
    "joint_seediv_diagnostic/config.json", "joint_seediv_diagnostic/comparison.json",
    "joint_seediv_diagnostic/folds.json", "joint_seediv_diagnostic/trial_predictions.csv",
    "joint_seediv_diagnostic/verification.json", "joint_seediv_diagnostic/reproduction.json",
    "joint_seediv_diagnostic/paired_comparison.json",
    *[f"negative_transfer_neural_seediv/{name}" for name in (
        "config.json", "plan.json", "folds.json", "linear_comparison.json",
        "linear_trial_predictions.csv", "model_metrics.json", "trial_predictions.csv",
        "neural_comparison.json", "paired_comparison.json", "training_behavior.json",
        "verification.json", "linear_verification.json", "neural_transfer_comparison.png",
        "gradient_diagnostics.json", "gradient_summary.json", "gradient_plan.json")],
    *[f"negative_transfer_neural_{target}/{name}" for target in ("deap", "gameemo") for name in (
        "config.json", "plan.json", "folds.json", "linear_comparison.json",
        "linear_trial_predictions.csv", "model_metrics.json", "trial_predictions.csv",
        "neural_comparison.json", "verification.json", "linear_verification.json")],
    "multitarget_transfer_analysis/comparison.json", "multitarget_transfer_analysis/analysis_provenance.json",
    "multitarget_transfer_analysis/multitarget_transfer_comparison.png",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_out(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def portable_text(text):
    # Keep evidence hashes unchanged, but replace this machine's absolute path.
    for prefix in (str(WORKSPACE), WORKSPACE.as_posix()):
        text = text.replace(prefix, "<workspace>")
    # Signed download URLs are transient credentials, not publication evidence.
    text = re.sub(r"(https?://[^\s\"<>]+)\?[^\s\"<>]*(?:X-Amz|X-Goog|Signature|token)[^\s\"<>]*",
                  r"\1?redacted-transient-query", text, flags=re.I)
    return text


def portable_value(value):
    if isinstance(value, str):
        return portable_text(value)
    if isinstance(value, list):
        return [portable_value(v) for v in value]
    if isinstance(value, dict):
        return {k: portable_value(v) for k, v in value.items()}
    return value


def archive_paper():
    archive = REPO / "docs/paper_archive/2026-10-05-pre-exploration"
    source = WORKSPACE / "report.tex"
    target = archive / "report.tex"
    archive.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(target) != sha(source):
        raise ValueError("Preservation snapshot exists and differs; create a new dated archive")
    target.write_bytes(source.read_bytes())
    uncommented = re.sub(r"(?m)(?<!\\)%.*$", "", source.read_text(encoding="utf-8"))
    assets = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", uncommented)
    missing, files = [], [{"source": "report.tex", "archive": "report.tex", "sha256": sha(target)}]
    for name in assets:
        asset = WORKSPACE / name
        if asset.is_file():
            dest = archive / Path(name).name
            dest.write_bytes(asset.read_bytes())
            files.append({"source": name, "archive": dest.name, "sha256": sha(dest)})
        else:
            missing.append(name)
    # An existing historical PDF is retained; it is not asserted to compile from this source.
    pdf = REPO / "paper/DL-Report.pdf"
    if pdf.is_file():
        (archive / pdf.name).write_bytes(pdf.read_bytes())
        files.append({"source": "paper/DL-Report.pdf", "archive": pdf.name, "sha256": sha(pdf)})
    json_out(archive / "manifest.json", {"purpose": "Preservation before exploratory work, not an approved pivot",
             "files": files, "missing_referenced_assets": missing,
             "compile_status": "Not compiled; historical PDF may not correspond exactly to report.tex"})
    (archive / "README.md").write_text(
        "# Manuscript preservation checkpoint\n\n"
        "The [original source](report.tex) is a byte-for-byte copy of the working manuscript on 5 October 2026. "
        "Its SHA-256 and the existing [historical PDF](DL-Report.pdf) are recorded in [manifest.json](manifest.json). "
        "The PDF is retained as an existing artifact; it was not regenerated from this source.\n\n"
        "Missing source image files: " + ", ".join(f"`{x}`" for x in missing) + ". "
        "This snapshot therefore does not yet provide a self-contained compilable paper.\n\n"
        "No research-question change has been approved. Archive the then-current paper again immediately before "
        "any approved change. The historical claims are subject to the publication review.\n", encoding="utf-8")


def export():
    archive_paper()
    docs, results = REPO / "docs/publication", REPO / "results/development"
    docs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    paths, records = {}, []

    def copy(src, dst):
        if not src.is_file():
            return
        if src.stat().st_size > 2_000_000:
            raise ValueError(f"Unexpected large review artifact: {src.name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".json":
            json_out(dst, portable_value(json.loads(src.read_text(encoding="utf-8"))))
        else:
            dst.write_bytes(src.read_bytes())
        paths[src.resolve()] = dst
        records.append({"workspace_source": src.relative_to(WORKSPACE).as_posix(),
                        "repository_export": dst.relative_to(REPO).as_posix(),
                        "source_sha256": sha(src), "export_sha256": sha(dst)})

    for name in REPORTS:
        src = WORKSPACE / name
        if src.is_file():
            paths[src.resolve()] = docs / name
    paths[(WORKSPACE / "report.tex").resolve()] = REPO / "docs/paper_archive/2026-10-05-pre-exploration/report.tex"
    runs = WORKSPACE / "publication_runs"
    for run in sorted(runs.iterdir()):
        if run.is_dir() and (run / "test_metrics.json").is_file():
            for name in sorted(RUN_FILES):
                copy(run / name, results / run.name / name)
    for name in EXTRA_FILES:
        copy(runs / name, results / name)

    local_only = set()
    for name in REPORTS:
        src = WORKSPACE / name
        if not src.is_file():
            continue
        dest = docs / name

        def link(match):
            label, raw = match.groups()
            if re.match(r"(?:https?://|mailto:|#)", raw):
                return match.group(0)
            location, sep, fragment = raw.partition("#")
            candidate = (WORKSPACE / location.replace("\\", "/")).resolve()
            mapped = paths.get(candidate)
            if mapped is None and candidate.is_relative_to(REPO) and candidate.exists():
                mapped = candidate
            if mapped:
                relative = Path(os.path.relpath(mapped, dest.parent)).as_posix()
                return f"[{label}]({relative}{sep}{fragment})"
            local_only.add(location)
            return f"{label} (local workspace evidence: `{location}`)"

        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, src.read_text(encoding="utf-8"))
        dest.write_text(portable_text(text), encoding="utf-8")
        records.append({"workspace_source": name, "repository_export": dest.relative_to(REPO).as_posix(),
                        "source_sha256": sha(src), "export_sha256": sha(dest)})
    json_out(results / "export_manifest.json", {"scope": "Selected derived development evidence, not raw recordings or full run bundles",
             "transformations": "JSON absolute workspace paths redacted; review links made portable. Source/export hashes are distinct where transformed.",
             "excluded": "Raw EEG, window caches, checkpoints, local dependencies, credentials, full third-party source copies",
             "files": records, "local_only_references": sorted(local_only)})
    (results / "README.md").write_text(
        "# Development evidence\n\n"
        "Selected metrics, configurations, trial-level predictions, diagnostic figures, provenance checks and plans "
        "are exported from the local research workspace. All runs here are exploratory; several are explicitly capped pilots. "
        "Previously inspected test cohorts must not be presented as untouched confirmatory evidence.\n\n"
        "[Export manifest](export_manifest.json) records source and exported hashes. Local paths and report links are "
        "made portable. Checkpoints, raw EEG, full caches, and downloaded third-party sources are excluded. "
        "Local verification records document completed checks; exact checkpoint replay also requires the retained local "
        "run bundle. Reproduce training with the maintained CLI and separately obtained datasets.\n", encoding="utf-8")
    print(json.dumps({"exported_files": len(records), "local_only_references": len(local_only)}))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.parse_args()
    export()
