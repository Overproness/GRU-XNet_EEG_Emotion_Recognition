"""Create complete, nonselective tables/figures after numerical verification."""
from argparse import ArgumentParser
import json
from pathlib import Path
import platform
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from gruxnet.grouped_material_controls import read_predictions, probabilities
from gruxnet.temporal_controls import common_metrics, metric
from gruxnet.data import write_json

LABELS = {"mean_mlp": "Mean bandpower MLP", "transformer": "Temporal transformer", "bandpower": "Bandpower logistic",
          "duration": "Duration only (no EEG)", "reve_pretrained": "Frozen REVE pretrained", "reve_random42": "Frozen REVE random42", "material_prior": "Video prior (no EEG)"}


def percent(v): return f"{100*v:.2f}%"
def interval(v): return f"[{100*v[0]:+.2f}, {100*v[1]:+.2f}]"


def process(dataset, output, previous):
    verification = json.loads((output/"verification.json").read_text()); analysis = json.loads((output/"analysis_verification.json").read_text())
    if not verification["passed"] or not analysis["passed"]: raise ValueError("Verified complete evidence required")
    report = json.loads((output/"comparison.json").read_text()); frames = read_predictions(output, previous, dataset)
    population = next(iter(frames.values()))["exposed"].drop_duplicates("trial_id")
    label_counts = population.groupby(["material_key", "label"]).size().unstack(fill_value=0)
    label_metadata = {"dataset": dataset, "retained_trials": len(population), "materials": len(label_counts),
                      "materials_with_multiple_target_classes": int((label_counts.gt(0).sum(1)>1).sum()),
                      "observed_people_per_material_range": [int(label_counts.sum(1).min()), int(label_counts.sum(1).max())],
                      "class_counts_by_material": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in label_counts.iterrows()},
                      "scope": "Descriptive original-cohort label heterogeneity; no fitted model, all-cohort prior evaluation or additional independent observations"}
    write_json(output/"label_heterogeneity.json", label_metadata)
    all_cells = []; groups = []; training = []
    for name, arms in frames.items():
        for arm, rows in arms.items():
            for (group, session, rotation, fold), part in rows.groupby(["group", "source_session", "material_rotation", "fold"]):
                metrics = common_metrics(part.original_label.to_numpy(dtype=int), probabilities(part), "coarse3") if dataset == "SEEDIV" else {"binary": metric(part.label.to_numpy(dtype=int), probabilities(part))}
                for task, values in metrics.items():
                    all_cells.append({"dataset": dataset, "model": name, "arm": arm, "group": group, "session": session, "rotation": rotation, "fold": fold, "task": task,
                                      **{k: v for k, v in values.items() if k != "confusion_matrix"}, "confusion_matrix": json.dumps(values["confusion_matrix"])})
    for scope, models in report["models"].items():
        for name, arms in models.items():
            for arm, tasks in arms.items():
                for task, values in tasks.items():
                    if task == "per_group_metrics": continue
                    groups.append({"dataset": dataset, "scope": scope, "model": name, "arm": arm, "task": task, "balanced_accuracy": values["mean_BA"],
                                   "crossed_low": values["crossed_percentile_95"][0], "crossed_high": values["crossed_percentile_95"][1],
                                   "participant_low": values["participant_percentile_95"][0], "participant_high": values["participant_percentile_95"][1]})
    primary = "coarse3" if dataset == "SEEDIV" else "binary"
    for item in json.loads((output/"model_index.json").read_text()):
        folder = output/"models"/item["id"]; r = json.loads((folder/"metrics.json").read_text()); history = json.loads((folder/"history.json").read_text())
        selected = next(h for h in history if h["step"] == r["selected_step"])
        training.append({"dataset": dataset, **{k: r[k] for k in ("id", "model", "arm", "group", "session", "rotation", "fold", "selected_step", "parameters", "elapsed_seconds", "peak_allocated_cuda_bytes")},
                         "train_BA": r["training"][primary]["balanced_accuracy"], "validation_BA": selected["validation"][primary]["balanced_accuracy"],
                         "test_BA": r["test"][primary]["balanced_accuracy"], "train_n": r["training"][primary]["n"],
                         "validation_n": selected["validation"][primary]["n"], "test_n": r["test"][primary]["n"]})
    for filename, rows in (("all_cells.csv", all_cells), ("all_groups.csv", groups), ("training_diagnostics.csv", training)):
        pd.DataFrame(rows).to_csv(output/filename, index=False)
    environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                   "scipy": scipy.__version__, "sklearn": sklearn.__version__, "CUDA": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "GPU": torch.cuda.get_device_name(), "pretrained_encoders": "SEED existing frozen features only; no new DEAP encoder extraction"}
    write_json(output/"environment.json", environment)
    models = report["models"]["combined"]; names = list(models); y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5.2)); colors = {"exposed": "#1f77b4", "unexposed": "#d55e00"}
    for arm, offset, label in (("exposed", -.12, "Shared test videos"), ("unexposed", .12, "Unseen test videos")):
        points = np.array([models[n][arm][primary]["mean_BA"]*100 for n in names]); bounds = np.array([models[n][arm][primary]["crossed_percentile_95"] for n in names])*100
        ax.errorbar(points, y+offset, xerr=np.stack([points-bounds[:, 0], bounds[:, 1]-points]), fmt="o", color=colors[arm], capsize=3, label=label)
    ax.axvline(100/(3 if dataset == "SEEDIV" else 2), color="grey", lw=1, ls="--", label="Chance BA")
    ax.set_yticks(y, [LABELS[n] for n in names]); ax.invert_yaxis(); ax.set_xlabel("Balanced accuracy (%)"); ax.grid(axis="x", alpha=.2)
    ax.set_title(f"{dataset}: combined grouping sensitivity\nPaired person/video bootstrap; fixed fits and one initialization", fontsize=11)
    ax.legend(loc="best", fontsize=9); fig.tight_layout(); fig.savefig(output/"comparison.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    scopes = [s for s in report["models"] if s != "combined"]
    contrasts = [c for c in report["contrasts"] if c["scope"] == "combined" and c["task"] == primary and c["model_a"] == c["model_b"]]
    for i, c in enumerate(contrasts):
        point = c["BA_difference"]*100; lo, hi = np.array(c["crossed_percentile_95"])*100
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", color="#222222", capsize=4, label="Combined crossed interval" if i == 0 else None)
        for j, scope in enumerate(scopes):
            group = next(k for k in report["contrasts"] if k["scope"] == scope and k["task"] == primary and k["model_a"] == k["model_b"] == c["model_a"])
            ax.scatter(group["BA_difference"]*100, i+(j-(len(scopes)-1)/2)*.10, s=28, marker=("s", "^", "D")[j], color=("#0072b2", "#e69f00", "#009e73")[j], label=f"Grouping {scope}" if i == 0 else None)
    ax.axvline(0, color="grey", lw=1, ls="--"); ax.set_yticks(range(len(contrasts)), [LABELS[c["model_a"]] for c in contrasts]); ax.invert_yaxis()
    ax.set_xlabel("Unseen minus shared test-video BA (percentage points)"); ax.grid(axis="x", alpha=.2)
    ax.set_title(f"{dataset}: all exposure contrasts and grouping estimates", fontsize=11); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(output/"paired_comparison.png", dpi=160); plt.close(fig)
    (output/"README.md").write_text(f"# {dataset} repeated material controls\n\nCompleted new neural fits: {verification['neural_fits_replayed']}. Independently refitted linear candidates: {verification['candidates_independently_refitted']}. All groupings and declared contrasts are retained. SEED grouping0 reuses the previous verified initialization42; groupings1/2 are new. DEAP has two new groupings. See comparison.json, verification.json, analysis_verification.json, all_groups.csv, all_cells.csv and training_diagnostics.csv. Raw data, embeddings and checkpoints remain local. This is exploratory sensitivity on reused cohorts, with one initialization; the manuscript and research question are unchanged.\n", encoding="utf-8")
    return report, pd.DataFrame(training), frames


def create_report(workspace):
    runs = workspace/"publication_runs"; previous = runs/"within_session_material_seediv"; evidence = {}
    for dataset, name in (("SEEDIV", "seediv"), ("DEAP", "deap")): evidence[dataset] = process(dataset, runs/f"repeated_material_{name}", previous)
    seed, seed_train, _ = evidence["SEEDIV"]; deap, deap_train, _ = evidence["DEAP"]
    raw = json.loads((runs/"cache_temporal_deap/raw_reproduction.json").read_text())
    if not raw["passed"] or raw["prefix_sequences_exactly_reproduced"] != 1264: raise ValueError("Complete DEAP source replay required")
    selected = pd.concat([seed_train, deap_train], ignore_index=True)
    lines = ["# Repeated participant/video grouping findings — 6 October 2026", "",
             "**Completed and numerically verified:** 680 new neural fits, 880 selected linear heads, all 3,520 independently refitted linear candidates and 160 training-label-only video-prior diagnostics. New fits produce 46,144 test probability rows; SEED analyses additionally reuse 12,960 previously verified initialization-42 rows. **The manuscript and research question remain unchanged. No pivot is approved.**", "",
             "**Research assessment:** SEED transformer three-class BA is 43.27% with shared videos and 39.07% with unseen videos, difference -4.20 pp with crossed interval [-7.39,-1.07]. Its exposure differences have the same negative sign in all three groupings. Frozen pretrained REVE also falls 4.40 pp, interval [-8.48,-0.49]. Repeated groupings strengthen the conditional SEED sensitivity finding, while reusing the same people/videos and one initialization.", "",
             "DEAP transformer binary BA is 52.28% shared and 50.66% unseen, difference -1.61 pp with interval [-4.97,+1.73]. All three DEAP EEG-model exposure intervals include zero; these data allow modest effects and do not prove equivalence. There is no clear unseen-video transformer advantage over the MLP in either corpus. A broad EEG material-collapse or novel transformer-superiority claim is not supported.", "",
             "The strongest DEAP diagnostic is the source-training-label video prior: 77.54% shared-video BA without EEG, versus 49.67% unseen, difference -27.87 pp with crossed interval [-33.86,-20.53]. This establishes strong contextual predictability of individual labels across people viewing known stimuli. It does not identify the EEG networks' mechanism or prove the provenance of historical/published high scores. All forty DEAP videos contain both rating classes, so video identity is informative without determining each person's answer.", "",
             "Frozen pretrained-minus-random REVE on unseen SEED materials is +7.53 pp in three-class BA with interval [+2.92,+12.16], a clearer feature-pretraining advantage under these groupings. Its complete private pretraining-corpus exclusion is still uncertified. This is an existing representation control, not a new method. These studies do not test negative transfer caused by jointly training on three corpora.", "",
             "The [protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/Repeated_Material_Control_Protocol_2026-10-06.md), source, cache and split records were [pushed before fitting](https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition/commit/f5bbdfa). Two additional SEED-IV participant/video groupings and two compatible DEAP groupings were fixed without searching test results. This tests grouping sensitivity on the existing cohorts; repeated partitions are not independent replications with new subjects or videos.", "", "## Combined model results", "",
             "SEED combines the original grouping at initialization42 with two new groupings at initialization42. The previous [three-initialization result](GRU-XNet_Within_Session_Material_Findings_2026-10-06.md) remains separate. DEAP combines two groupings at initialization42. Combined scores average correctness per observed person/video cell, then class recalls; they do not average unequal fold BAs or choose a preferred grouping.", "",
             "The shared-video arm estimates generalization to new people viewing known materials; the unseen-video arm estimates transfer to new people and new materials. Both are valid defined settings. A score difference alone cannot diagnose recording/augmentation leakage in historical experiments.", ""]
    for dataset, (report, train, frames) in evidence.items():
        task = "coarse3" if dataset == "SEEDIV" else "binary"
        lines += [f"### {dataset}", "", "| Model | Shared videos: BA | Unseen videos: BA | Difference (pp) | Crossed 95% interval |", "|---|---:|---:|---:|---:|"]
        for name, arms in report["models"]["combined"].items():
            c = next(c for c in report["contrasts"] if c["scope"] == "combined" and c["task"] == task and c["model_a"] == c["model_b"] == name)
            lines.append(f"| {LABELS[name]} | {percent(arms['exposed'][task]['mean_BA'])} | {percent(arms['unexposed'][task]['mean_BA'])} | {100*c['BA_difference']:+.2f} | {interval(c['crossed_percentile_95'])} |")
        suffix = "seediv" if dataset == "SEEDIV" else "deap"
        lines += ["", f"![{dataset} all models](publication_runs/repeated_material_{suffix}/comparison.png)", "", f"![{dataset} exposure sensitivity](publication_runs/repeated_material_{suffix}/paired_comparison.png)", ""]
        if dataset == "SEEDIV":
            lines += ["SEED's primary target is stimulus-assigned neutral/negative/positive (chance BA 33.33%). The secondary conditional binary task excludes the same 270 neutral trials, retaining 810 trials and the unchanged neutral-mass floor/negative-tie rule. Duration uses full-trial length and no EEG; it is a diagnostic. Frozen REVE features reuse the existing audited adapter, 37 seconds of direct patches inside the normalized 40-second prefix, and one random encoder seed. No encoder is updated; complete target exclusion from its private pretraining corpus remains uncertified.", "", "| Model | Shared: binary BA | Unseen: binary BA |", "|---|---:|---:|"]
            for name, arms in report["models"]["combined"].items(): lines.append(f"| {LABELS[name]} | {percent(arms['exposed']['binary']['mean_BA'])} | {percent(arms['unexposed']['binary']['mean_BA'])} |")
            lines.append("")
        else:
            feasible = json.loads((runs/"repeated_material_deap/feasibility.json").read_text())
            ranges = {k: (min(r[k] for r in feasible), max(r[k] for r in feasible)) for k in ("train", "validation", "test")}
            label_meta = json.loads((runs/"repeated_material_deap/label_heterogeneity.json").read_text())
            lines += [f"DEAP uses individual self-reported valence, chance BA 50%, 1,264 retained trials, 32 people and 40 Experiment_id videos. {label_meta['materials_with_multiple_target_classes']} of40 videos have both positive and negative ratings across retained people; observed people per video range from {label_meta['observed_people_per_material_range'][0]} to {label_meta['observed_people_per_material_range'][1]}. [Complete label counts](publication_runs/repeated_material_deap/label_heterogeneity.json). Each fit has 24/4/4 disjoint training/validation/test people. After participant/class matching, trial counts range from {ranges['train'][0]}–{ranges['train'][1]} training, {ranges['validation'][0]}–{ranges['validation'][1]} validation and {ranges['test'][0]}–{ranges['test'][1]} test. Every split contains both classes. Exact training participant/class counts and test/validation trials match between arms; all test-video keys remain represented in exposed training after matching. No validation video enters training. Each trial is tested once per grouping.", "", "Video-prior predictions use only selected training labels associated with each video, Laplace-one smoothing and a global training-rate fallback for unseen videos. They use no EEG and no test-based selection. Both neural backbones have actual two-class heads (18,992 MLP versus 19,042 transformer parameters). These DEAP results do not independently authenticate first-party signals, reproduce a published Conformer or test a frozen DEAP foundation model.", ""]
    lines += ["## Every grouping and paired contrast", "", "Each table includes every declared contrast. Grouping0 is prior SEED initialization42, groupings1/2 are new. Repeats reuse all people/materials. Combined intervals do not treat repeats as new independent samples. Resample people and videos with paired fixed weights, using observed-cell class numerators and denominators. DEAP's sixteen excluded midpoint cells remain absent, and videos are not stratified by a single class. SEED retains session/native-emotion material strata. Intervals are unadjusted exploratory percentiles, conditional on fixed models/partitions/cohorts; no causal effect or equivalence inference is established.", ""]
    for dataset, (report, _, _) in evidence.items():
        lines += [f"### All {dataset} contrasts", "", "| Group | Contrast (A minus B) | Task | Difference (pp) | Person-only interval | Person/video interval |", "|---|---|---|---:|---:|---:|"]
        for c in report["contrasts"]:
            label = f"{LABELS[c['model_a']]}: unseen minus shared" if c["model_a"] == c["model_b"] else f"{LABELS[c['model_a']]} minus {LABELS[c['model_b']]} ({c['arm_a']})"
            lines.append(f"| {c['scope']} | {label} | {c['task']} | {100*c['BA_difference']:+.2f} | {interval(c['participant_percentile_95'])} | {interval(c['crossed_percentile_95'])} |")
        lines.append("")
    lines += ["## Selection, hardware and verification", "", "| Corpus | Model | Arm | Median selected update | Mean train BA | Mean validation BA | Mean test-fold BA (diagnostic) |", "|---|---|---|---:|---:|---:|---:|"]
    for (dataset, name, arm), rows in selected.groupby(["dataset", "model", "arm"]):
        lines.append(f"| {dataset} | {LABELS[name]} | {arm} | {rows.selected_step.median():.0f} | {percent(rows.train_BA.mean())} | {percent(rows.validation_BA.mean())} | {percent(rows.test_BA.mean())} |")
    lines += ["", f"The 680 new fits total **{selected.elapsed_seconds.sum():.1f} seconds**, with peak **{selected.peak_allocated_cuda_bytes.max()/2**20:.2f} MiB allocated CUDA tensors** on the RTX 3050. This excludes CUDA context/display/driver allocations, preprocessing, classical fitting and earlier frozen extraction. Every fit retains 600 updates and 24 validation evaluations; validation-only selection is replayed. The small held-out populations make selection noisy, especially SEED's twelve validation trials. The displayed mean test-fold BA is a training diagnostic; primary DEAP estimates aggregate original trial predictions.", "",
              "All 53 scientific-control tests passed before fitting. Independent numerical checks replay all 680 selected checkpoint train/validation/test predictions and scalers, reproduce initial weights and canonical batch signatures across paired arms, check exact split records and once-per-trial coverage, and independently refit every linear candidate and selected coefficient. The complete bootstrap analysis is recomputed exactly. DEAP preparation checked 32 raw-file hashes, spreadsheet metadata, 1,264 waveform hashes and ordered windows; continuous-rating CSV comparisons allow 1e-12 floating-point parsing differences while binary labels are exact. No label mapping was changed.", "",
              "The additional [DEAP source replay](publication_runs/cache_temporal_deap/raw_reproduction.json) reread all32 downloaded participant files and the corrected spreadsheet loader. It exactly regenerated all1,264 retained full filtered waveforms and first40-second sequence arrays, while checking all16 excluded midpoint trials. Maximum array difference was zero. This source replay occurred during SEED fitting and before DEAP fitting; it made no data or parameter changes and is not first-party authentication.", ""]
    for dataset, suffix in (("SEEDIV", "seediv"), ("DEAP", "deap")):
        v = json.loads((runs/f"repeated_material_{suffix}/verification.json").read_text())
        lines += [f"{dataset} maximum selected-neural probability replay error: {v['maximum_neural_probability_error']:.3g}; maximum linear error: {v['maximum_linear_probability_error']:.3g}; maximum coefficient-refit error: {v['maximum_coefficient_refit_error']:.3g}. [Verification](publication_runs/repeated_material_{suffix}/verification.json), [analysis replay](publication_runs/repeated_material_{suffix}/analysis_verification.json), [all group scores](publication_runs/repeated_material_{suffix}/all_groups.csv), [every test cell](publication_runs/repeated_material_{suffix}/all_cells.csv), [training diagnostics](publication_runs/repeated_material_{suffix}/training_diagnostics.csv) and [full comparison](publication_runs/repeated_material_{suffix}/comparison.json) retain complete evidence.", ""]
    lines += ["## Research decision limits", "", "Exposure changes training video content, difficulty and order along with identity. Repeated groupings quantify sensitivity within already inspected cohorts; they do not increase the number of people or independent video stimuli. One new optimizer initialization does not independently establish stability across optimization seeds. SEED stimulus-assigned classes and DEAP individual ratings have different semantics. Neither corpus result tests joint three-dataset training or leave-one-dataset-out generalization. Original SEED media hashes are unavailable, first-party DEAP signal authentication remains outstanding, and REVE's complete pretraining overlap is uncertified.", "", "The prior-work audit already found subject/material separation precedents. Holding out clips or adding a transformer alone is not a novel method. A defensible new paper would still need a clearly justified contribution, matched full GRU-XNet architectural baselines/ablations, broader generalization validation and an accurate rebuilt manuscript. Use the entire evidence to propose a direction; do not adopt a changed question automatically. The [readiness checklist](GRU-XNet_Publication_Readiness_2026-10-05.md) tracks these unfinished items. Show findings and get author approval before a pivot, then archive the then-current manuscript immediately before changing it.", "", "**Next experimental priority:** carry the original GRU-XNet and matched CBSAtt/BiLSTM control through the same source-only participant/video protocols, with a declared observation/normalization and matched training/selection budget. Add the contextual prior as a diagnostic. Test whether EEG adds predictive information beyond that prior by comparing EEG-plus-context with context alone; derive training-row priors by participant cross-fitting so a row's own label cannot enter its contextual feature, and derive held-out priors solely from training people. Declare this before fitting and retain both known/unseen-material settings. These are proposed diagnostics, not completed experiments or an adopted research question. Confirm comparable prior-work coverage before presenting a new contribution.", ""]
    path = workspace/"GRU-XNet_Repeated_Material_Findings_2026-10-06.md"; path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(path), "neural_fits": len(selected), "neural_fit_seconds": selected.elapsed_seconds.sum(), "peak_MiB": selected.peak_allocated_cuda_bytes.max()/2**20}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__); parser.add_argument("--workspace", type=Path, required=True); a = parser.parse_args(); create_report(a.workspace.resolve())
