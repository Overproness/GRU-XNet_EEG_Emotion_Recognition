"""Build a dated, evidence-linked research memo only after checkpoint verification."""
from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent


def describe(output, cache, destination):
    output, cache = output.resolve(), cache.resolve()
    verification = json.loads((output/"verification.json").read_text())
    if not verification["passed"]:
        raise ValueError("Verify the complete experiment before reporting")
    report = json.loads((output/"comparison.json").read_text())
    records = json.loads((output/"model_metrics.json").read_text())
    info = json.loads((cache/"prepared.json").read_text())
    def link(name):
        return (output/name).relative_to(WORKSPACE).as_posix()
    def title(name):
        return name.replace("mean_mlp","MLP").replace("transformer","Transformer").replace("_"," / ")
    neural = {name:value for name,value in report["conditions"].items() if not name.startswith("linear_")}
    table = ["| Model / representation / supervision | Three-class BA | Conditional binary BA | Three-class training BA |",
             "| --- | ---: | ---: | ---: |"]
    for name,value in neural.items():
        training = np.mean([r["training"]["coarse3"]["balanced_accuracy"] for r in records if r["condition"]==name])
        table.append(f"| {title(name)} | {100*value['mean_coarse3_BA']:.2f}% | {100*value['mean_binary_BA']:.2f}% | {100*training:.2f}% |")
    baselines = ["| Logistic input | Three-class BA | Conditional binary BA |",
                 "| --- | ---: | ---: |"]
    for name,value in report["conditions"].items():
        if name.startswith("linear_"):
            baselines.append(f"| {name.removeprefix('linear_')} | {100*value['coarse3']['balanced_accuracy']:.2f}% | {100*value['binary']['balanced_accuracy']:.2f}% |")
    pairs = ["| Comparison | Common task | Difference (pp) | Paired participant 95% interval (pp) |",
             "| --- | --- | ---: | ---: |"]
    for pair in report["paired_comparisons"]:
        low,high = np.array(pair["paired_subject_percentile_95"])*100
        pairs.append(f"| {title(pair['a'])} minus {title(pair['b'])} | {pair['task']} | {pair['mean_BA_difference']*100:+.2f} | [{low:+.2f}, {high:+.2f}] |")
    significant = [p for p in report["paired_comparisons"] if p["paired_subject_percentile_95"][0]>0 or p["paired_subject_percentile_95"][1]<0]
    best3 = max(neural,key=lambda k:neural[k]["mean_coarse3_BA"])
    best2 = max(neural,key=lambda k:neural[k]["mean_binary_BA"])
    mean_init = [f"- {title(name)}: three-class " + ", ".join(f"seed {seed}: {100*m['coarse3']['balanced_accuracy']:.2f}%" for seed,m in value["seeds"].items())
                 + "; binary " + ", ".join(f"seed {seed}: {100*m['binary']['balanced_accuracy']:.2f}%" for seed,m in value["seeds"].items()) + "."
                 for name,value in neural.items()]
    elapsed = sum(r["elapsed_seconds"] for r in records)
    memory = max(r["peak_allocated_cuda_bytes"] for r in records)/2**20
    steps = [r["selected_step"] for r in records]
    native = [f"- {title(name)}: {100*np.mean([m['native4']['balanced_accuracy'] for m in value['seeds'].values()]):.2f}% native four-class balanced accuracy (25% chance)."
              for name,value in neural.items() if name.endswith("native4")]
    standard = next(p for p in report["paired_comparisons"] if p["task"]=="coarse3"
                    and p["a"]=="transformer_absolute_coarse3" and p["b"]=="mean_mlp_absolute_coarse3")
    lo,hi = np.array(standard["paired_subject_percentile_95"])*100
    native_effects = [p for p in report["paired_comparisons"] if p["a"].endswith("native4") and p["b"].endswith("coarse3")]
    unclear_native = sum(p["paired_subject_percentile_95"][0]<=0<=p["paired_subject_percentile_95"][1] for p in native_effects)
    standard_neural = neural["transformer_absolute_coarse3"]
    standard_mlp = neural["mean_mlp_absolute_coarse3"]
    text = f"""# Transformer, representation and native-label findings

Date: 6 October 2026. **120 neural runs and 15 selected logistic models are complete and verified. The paper is still not ready for submission, and no new research question has been adopted.**

The highest observed neural three-class mean is **{100*neural[best3]['mean_coarse3_BA']:.2f}%** ({title(best3)}); the highest observed conditional binary mean is **{100*neural[best2]['mean_binary_BA']:.2f}%** ({title(best2)}). These are descriptive development results, not validation-selected winners for a new contribution. Of the 24 predeclared factor contrasts, **{len(significant)}** have unadjusted participant percentile intervals excluding zero. All contrasts and seeds are reported below; selecting a favourable contrast after inspection would require further independent confirmation.

The straightforward absolute-power/coarse-label comparison gives **{100*standard_neural['mean_coarse3_BA']:.2f}% transformer versus {100*standard_mlp['mean_coarse3_BA']:.2f}% MLP**, a **{standard['mean_BA_difference']*100:+.2f} percentage-point** primary-task difference with unadjusted interval **[{lo:+.2f}, {hi:+.2f}]**. This supports further temporal-model investigation, not a transformer-novelty or broad superiority claim. **{unclear_native} of {len(native_effects)}** native-minus-coarse objective intervals include zero, so preserving sadness/fear as separate training targets has not shown a clear benefit here. The absolute-power logistic control's **{100*report['conditions']['linear_absolute']['binary']['balanced_accuracy']:.2f}%** binary point estimate exceeds every neural binary mean in this phase. Representation and task matter; the primary-task gain should not be presented as an improvement on all emotion targets.

## What was tested

The [committed protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/Temporal_Native_Control_Protocol_2026-10-06.md) and [machine declaration](publication_runs/temporal_native_plan_2026-10-06.json) were saved before fitting. The full factorial compares temporal transformer versus mean MLP, absolute versus relative log-bandpower, and native four-emotion versus coarse three-valence supervision. Both label objectives see the **same 1,080 original trials**, including neutral, the same sampling stream and the same common evaluation tasks. Native supervision keeps sadness and fear separate; coarse supervision combines them as negative. This within-SEED-IV study does not test joint training with DEAP/GAMEEMO, so it cannot determine whether native supervision fixes cross-dataset negative transfer.

All 15 SEED-IV participants and three sessions are included. Five existing participant rotations use nine train, three validation and three test participants; seeds 42/43/44 initialize each model. All trials of a participant stay together. The common three-class task uses all trials, with 33.33% chance balanced accuracy; the conditional binary task excludes the same 270 neutral trials, with 50% chance. Native-head probabilities are grouped for common-task evaluation; binary probabilities are conditioned on negative plus positive. Native four-class accuracy is a distinct secondary measure and cannot be compared directly to binary accuracy.

Each trial contributes the first ten non-overlapping four-second bandpower windows: a fixed **40-second input** using 14 named electrodes and four bands. The transformer has two independently initialized four-head encoder layers, width 32, feedforward width 64, fixed positions in seconds and mean token pooling. The MLP averages the same ten windows before its 56 → 128 → 86 encoder. Coarse heads give 19,075 and 19,079 parameters respectively. The networks have nearly identical parameter counts and differ in latent width and computation, so the comparison cannot isolate attention's contribution. The transformer is independently written and is not a raw-waveform EEG-Conformer or pretrained foundation-model reproduction.

Relative features are log power fractions computed within each channel/window. Training-only scalers are frozen for validation/test and paired across objectives and architectures. Every arm receives the identical 600 batches for a fold/seed, each with 20 samples from each coarse class. The four native classes are not equally sampled: the two negative subtypes share the negative quota. Native heads begin with identical grouped coarse probabilities. Every run completes 600 AdamW updates with identical optimizer settings; validation every 25 updates selects common three-class BA, then balanced log loss, then the earliest exact tie. Test is evaluated afterward. Different selected checkpoints can have different actual training exposures despite the identical available budget.

## Results on common tasks

Neural values are out-of-fold trial BA averaged over the three initialization results. Training values average the 15 selected checkpoint training-fold BAs; they describe fit on known participants and are not directly a paired population comparison with the OOF score.

{chr(10).join(table)}

{chr(10).join(baselines)}

The duration diagnostic receives only the original full-trial window count, with a training-only scaler and validation-selected multinomial logistic classifier. It receives no EEG features. It quantifies label-duration association in this corpus and is excluded from claims about fixed-duration EEG recognition. It does **not** prove that the historical model exploited duration. Fixed prefix length removes direct sequence-length variation; the participants still watch familiar corpus stimuli. Zero-phase filtering and resampling use the full trial before cropping, so this is an offline protocol rather than strictly causal acquisition limited to 40 seconds.

The earlier 67.13% binary logistic control averages every available window and excludes neutral during training. The current experiment changes observation duration, training task, neutral inclusion, model width, and selection metric. Differences from that historical control cannot be attributed solely to architecture or label granularity. The factorial contrasts below use matched current inputs and trial populations.

## Paired effects and uncertainty

{chr(10).join(pairs)}

The 10,000 paired participant bootstrap draws use seed 20261006 and average the three initialization results within each resampled participant cohort. Seeds are not independent participants. The intervals condition on one participant grouping and these trained models; training folds overlap, and these participants were previously inspected in development. There is no multiplicity correction for the 24 exploratory contrasts and no new confirmatory cohort.

## All initialization results

{chr(10).join(mean_init)}

Secondary native four-class results, reported only within the native objective:

{chr(10).join(native)}

## Integrity, resources and saved evidence

- Re-extracted features from all 45 hashed raw SEED-IV source files agree exactly with the prior full-trial mean features for **all 1,080 trials**. Full waveforms agree exactly with the retained common-montage cache for all **810** overlapping nonneutral trials. Source labels, metadata and physical electrode ordering are checked. New sequence-cache fingerprint: `{info['fingerprint']}`.
- Verification replays **120 selected neural checkpoints**, **25,920 neural trial probability rows**, **15 linear coefficient models** and **3,240 linear probability rows**. It checks source/cache hashes, full participant coverage, validation selection, training-only scalers, identical 600-batch streams, checkpoint training/validation/test metrics and aggregate/bootstrap recomputation. Maximum replay probability error is `{verification['max_neural_probability_error']:.3g}`. Linear verification replays the saved coefficients; it does not independently refit every C candidate.
- Fitting time summed across neural runs: **{elapsed:.1f} seconds** ({elapsed/60:.2f} minutes), excluding preparation/replay/export. Maximum PyTorch allocated CUDA memory: **{memory:.2f} MiB**; CUDA context/driver and other processes are excluded. The runs used the author's RTX 3050 environment. Selected steps span **{min(steps)}–{max(steps)}**, median **{float(np.median(steps)):.1f}**, from a fixed 600-update available budget.
- [Environment snapshot]({link('environment.json')}) records package/runtime versions after replay. It is a post-run snapshot from the same active environment; dependencies were unchanged during this phase.
- [Verification]({link('verification.json')}), [all model metrics]({link('model_metrics.json')}), [aggregate comparisons]({link('comparison.json')}), [linear models and validation candidates]({link('linear_models.json')}). Individual checkpoints and histories remain local and are excluded from GitHub with raw EEG and feature arrays.
- [Common-task figure]({link('comparison.png')}), [paired-contrast figure]({link('paired_comparison.png')}). Probabilities are exported separately for each of the eight conditions to keep bounded review artifacts.

## Meaning of a negative-transfer contribution

Negative transfer means **adding a source dataset worsens prediction on a target dataset**, compared with training on that target alone under a fair protocol. A contribution in this area would establish a repeatable pattern, isolate an explanation through controlled interventions, or introduce and validate a method that prevents the harm. It does not mean merely publishing a low score. In our earlier study, linear pooling penalties were clearer, while every neural shared-joint versus target-only participant interval included zero. Causes such as label mismatch, electrode differences and optimization remain hypotheses, not demonstrated explanations.

The present transformer/label experiment broadens model exploration. It does not change that pooling conclusion, because it trains on SEED-IV alone. Keeping original emotions separate is itself an established idea; a model or loss change needs evidence and a gap from the closest work before it can be a strong-conference contribution.

## Remaining work and decision

The [new primary-source review](GRU-XNet_Transformer_Research_Update_2026-10-06.md) adds EEG-Conformer, One Model for All, PESD, an EMBC 2021 shared-stimulus study, and a July 2026 evaluation preprint. Broad claims about unified EEG models, transformer novelty, or the discovery that evaluation protocols matter already have substantial precedents. [Prior cross-target findings](GRU-XNet_Multitarget_Transfer_Findings_2026-10-05.md) and the [readiness checklist](GRU-XNet_Publication_Readiness_2026-10-05.md) still apply.

Before choosing a new contribution, the next useful validation is to separate both participants and stimulus/session material, and audit a modern pretrained baseline for checkpoint overlap, target-data access and hardware feasibility. SEED-IV session separation changes recording conditions and clips together; it is a robustness test and cannot isolate a video's causal influence. Current results do not establish performance on entirely unseen datasets or a single jointly selected checkpoint for all three corpora.

Still outstanding: convincing protocol-compatible reference comparisons; full GRU/LSTM/attention/montage/augmentation controls where relevant to the selected question; repeated participant groupings or final evaluation; full LODO evidence; first-party DEAP signal authentication; final novelty assessment; an updated manuscript, references, consistent figures and compiled PDF. Historical 95.91% is not replaced by these different tasks. Corrected existing labels remain unchanged.

**The manuscript source and historical PDF remain preserved, and the research question remains unchanged.** Show a concrete evidence-based proposal and obtain the author's explicit approval before adopting a changed question; archive the then-current source immediately beforehand. Completing this exploratory phase is not completing the publication project.
"""
    destination.write_text(text,encoding="utf-8")
    print(json.dumps({"saved":str(destination),"unadjusted_intervals_excluding_zero":len(significant),"best_three_class_condition":best3,"best_binary_condition":best2}))


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args = parser.parse_args()
    describe(args.output,args.cache,args.destination)
