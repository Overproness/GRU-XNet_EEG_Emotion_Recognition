"""Create evidence-linked memo and standalone figures after complete verification."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]; WORKSPACE=REPO.parent
sys.path.insert(0,str(REPO))
os.environ.setdefault("MPLCONFIGDIR",str(WORKSPACE/"publication_runs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gruxnet.data import write_json

TITLES={"mean_mlp":"Mean MLP","transformer":"Small transformer","bandpower":"Bandpower logistic",
        "reve_pretrained":"Frozen REVE pretrained","reve_random42":"Frozen REVE random42","duration":"Duration only (diagnostic)"}
ORDER=["mean_mlp","transformer","bandpower","reve_pretrained","reve_random42","duration"]


def environment():
    import scipy,sklearn,torch,transformers,safetensors,einops
    return {"snapshot":"Post-fit environment snapshot; no dependencies installed or replaced for this phase",
            "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,
            "torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
            "transformers":transformers.__version__,"safetensors":safetensors.__version__,"einops":einops.__version__,
            "device":torch.cuda.get_device_name(),"device_total_bytes":torch.cuda.get_device_properties(0).total_memory}


def report(runs,destination):
    session=runs/"session_stimulus_seediv"; reve=runs/"reve_frozen_seediv"; crossed=runs/"session_material_sensitivity"
    for folder in (session,reve,crossed):
        if not json.loads((folder/"verification.json").read_text())["passed"]: raise ValueError("Unverified study")
    a=json.loads((session/"comparison.json").read_text()); b=json.loads((reve/"comparison.json").read_text())
    sensitivity=json.loads((crossed/"comparison.json").read_text()); models={**a["models"],**b["models"]}
    for folder in (session,reve): write_json(folder/"environment.json",environment())
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),sharey=True)
    for ax,task,chance in zip(axes,("coarse3","binary"),(100/3,50)):
        for i,name in enumerate(ORDER):
            for protocol,offset,color,marker in (("same_session",-.12,"#226a92","o"),("mean_unseen",.12,"#bb6134","s")):
                item=sensitivity["models"][name][protocol][task]; low,high=np.array(item["crossed_percentile_95"])*100
                ax.plot([low,high],[i+offset,i+offset],color=color,lw=1.7)
                ax.plot(item["BA"]*100,i+offset,marker=marker,color=color,ms=6,
                        label=protocol.replace('_',' ') if i==0 else None)
        ax.axvline(chance,color="black",ls="--",lw=.8)
        ax.set_xlim(15,85); ax.grid(axis="x",alpha=.18)
        ax.set_xlabel("Trial balanced accuracy (%)")
        ax.set_title("Three-class valence: 1,080 trials" if task=="coarse3" else "Conditional binary: 810 trials")
    axes[0].set_yticks(range(6),labels=[TITLES[n] for n in ORDER]); axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.59,.92),ncol=2,fontsize=9)
    fig.suptitle("SEED-IV development: held-out people, familiar versus unseen source sessions",fontsize=12)
    fig.text(.5,.015,"Intervals: crossed participant/material bootstrap, fixed models and 3 sessions. Duration is a non-EEG diagnostic; REVE embeds 37s within 40s.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.94)); fig.savefig(crossed/"comparison.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,7.7),layout="constrained")
    matrices={name:np.array([[models[name]["source_test_cells"][f"{s}->{t}"]["mean_coarse3_BA"]*100 for t in (1,2,3)] for s in (1,2,3)]) for name in ORDER}
    low=np.floor(min(m.min() for m in matrices.values())/5)*5; high=np.ceil(max(m.max() for m in matrices.values())/5)*5
    for ax,name in zip(axes.ravel(),ORDER):
        values=matrices[name]; picture=ax.imshow(values,vmin=low,vmax=high,cmap="viridis")
        for (i,j),value in np.ndenumerate(values): ax.text(j,i,f"{value:.1f}",ha="center",va="center",color="white" if value<(low+high)/2 else "black",fontsize=11)
        ax.set_xticks(range(3),labels=[1,2,3]); ax.set_yticks(range(3),labels=[1,2,3])
        ax.set_xlabel("Test session"); ax.set_ylabel("Training/validation session"); ax.set_title(TITLES[name],fontsize=10)
    fig.colorbar(picture,ax=axes.ravel().tolist(),label="Three-class trial balanced accuracy (%)",shrink=.8)
    fig.suptitle("All nine source/test cells: identical held-out participant folds",fontsize=12)
    fig.savefig(crossed/"session_cells.png",dpi=180); plt.close(fig)
    def link(path): return path.relative_to(WORKSPACE).as_posix()
    def percentage(value): return f"{value*100:.2f}%"
    def interval(values): return "["+", ".join(f"{v*100:+.2f}" for v in values)+"]"
    table=["| Model | Same-session 3-class BA | Unseen-session 3-class BA | Same-session binary BA | Unseen-session binary BA |","| --- | ---: | ---: | ---: | ---: |"]
    for name in ORDER:
        m=models[name]; table.append("| "+TITLES[name]+" | "+" | ".join(percentage(m[p][k]) for k,p in [("mean_coarse3_BA","same_session"),("mean_coarse3_BA","mean_unseen"),("mean_binary_BA","same_session"),("mean_binary_BA","mean_unseen")])+" |")
    comparisons=["| Model A / protocol minus model B / protocol | Task | Difference (pp) | Crossed 95% interval (pp) |","| --- | --- | ---: | ---: |"]
    for c in sensitivity["contrasts"]:
        label=f"{TITLES[c['model_a']]} / {c['protocol_a']} minus {TITLES[c['model_b']]} / {c['protocol_b']}"
        comparisons.append(f"| {label} | {c['task']} | {c['BA_difference']*100:+.2f} | {interval(c['crossed_percentile_95'])} |")
    def contrast(name,pa,other,pb,task="coarse3"):
        return next(c for c in sensitivity["contrasts"] if (c["model_a"],c["protocol_a"],c["model_b"],c["protocol_b"],c["task"])==(name,pa,other,pb,task))
    drop=contrast("transformer","mean_unseen","transformer","same_session")
    gain=contrast("reve_pretrained","mean_unseen","reve_random42","mean_unseen")
    advantage=contrast("transformer","same_session","mean_mlp","same_session")
    state=json.loads((reve/"features.json").read_text()); records=json.loads((session/"model_metrics.json").read_text())
    text=f"""# Participant/session transfer and frozen pretraining findings

Date: 6 October 2026. **90 neural fits, 70 selected linear models and 34,560 test probability rows are complete and verified. No research question has been changed. The paper remains unready for submission.**

The small transformer's three-class balanced accuracy is **{percentage(models['transformer']['same_session']['mean_coarse3_BA'])} on familiar source sessions and {percentage(models['transformer']['mean_unseen']['mean_coarse3_BA'])} on unseen source sessions**. The paired change is **{drop['BA_difference']*100:+.2f} percentage points**, with crossed participant/material interval **{interval(drop['crossed_percentile_95'])}**. Its point advantage over the MLP disappears on unseen sessions. Even its familiar-session advantage has crossed interval **{interval(advantage['crossed_percentile_95'])}**, which includes zero. Holding people out alone does not guarantee robustness to a new recording session and its stimulus material.

Frozen pretrained REVE reaches **{percentage(models['reve_pretrained']['mean_unseen']['mean_coarse3_BA'])} unseen-session three-class BA**, versus **{percentage(models['reve_random42']['mean_unseen']['mean_coarse3_BA'])} for the same frozen architecture with random seed42 weights**. The direct weight comparison is **{gain['BA_difference']*100:+.2f} pp**, crossed interval **{interval(gain['crossed_percentile_95'])}**. The downloaded weights help under this adapter, subject to one random encoder initialization and incomplete independent checkpoint-corpus authentication. This does not establish a new pretraining method, superiority over the small models or a solution to session transfer. These are exploratory unadjusted intervals conditional on fixed folds, models and three observed sessions.

## Controlled access and complete results

All arms use the same 1,080 SEED-IV original trials, 15 people, three sessions and 14 named physical electrodes. Coarse targets are neutral / negative (sadness+fear) / positive. Five participant rotations have nine training, three validation and three test people; all sessions of held-out people are excluded from fitting and selection. Each source-session model trains on 216 trials and validates on 72 trials **from that source session alone**. Select once, then evaluate the same three held-out people's 72 trials in every session. No other-session inputs or labels enter scaling, fitting or checkpoint selection.

Same-session results use the nine-cell matrix's diagonal. Two cyclic unseen-source assignments each cover every target trial once; average their scores, not their probabilities. Target trials are identical in the comparisons. Three seeds initialize each small neural arm; their results are averaged, not counted as additional people. The [session protocol]({link(REPO/'docs/publication/Session_Stimulus_Control_Protocol_2026-10-06.md')}) was declared and pushed before its 90 fits. Its matched available exposure, class sampling and optimizer budget are unchanged from the preceding models. Source-specific checkpoint choices can differ.

{chr(10).join(table)}

Chance BA is 33.33% for three classes and 50% for binary. Conditional binary excludes the same 270 neutral trials and conditions negative/positive probabilities. The duration-only arm receives **full original trial length and no EEG**; it is a diagnostic and must not enter EEG-model superiority claims. Its 48.15% familiar-session and 47.69% unseen-session three-class scores show material-duration/label association within this corpus, not proof that the fixed-prefix EEG models use trial length.

With all three sessions available to training/validation people, frozen REVE gives **{percentage(models['reve_pretrained']['mixed_sessions']['mean_coarse3_BA'])} three-class / {percentage(models['reve_pretrained']['mixed_sessions']['mean_binary_BA'])} binary**, versus random **{percentage(models['reve_random42']['mixed_sessions']['mean_coarse3_BA'])} / {percentage(models['reve_random42']['mixed_sessions']['mean_binary_BA'])}**. Mixed-session fitting supplies three times as many training trials and includes the test session's material among other participants. It is a different access protocol, not an unseen-session result. The earlier 120-run mixed-session bandpower factorial is also a different fitting protocol and should not be used as an exact training-size comparison.

![Session comparison with crossed intervals]({link(crossed/'comparison.png')})

![All source/test cells]({link(crossed/'session_cells.png')})

## Participant and material uncertainty

The initial participant-only analysis holds all 72 corpus materials fixed. Its three-class transformer-minus-MLP familiar-session interval is +0.04 to +5.21 pp. The crossed interval is {interval(advantage['crossed_percentile_95'])}; that changes the interpretation. For duration, identical participant responses to fixed-length clips can make participant-only intervals degenerate. That does not mean performance on new clips is certain.

The [sensitivity declaration]({link(runs/'session_material_sensitivity_plan_2026-10-06.json')}) was added **after initial session outcomes were inspected and before REVE head outcomes**. It is an explicitly exploratory addendum. Resample 15 people and independently six clip-position keys in each of twelve session/native-emotion strata, weighting every participant/material cell by the product. Average seeds and source directions first. The same 10,000 draws are paired across arms; proportions and target trials are fixed. Training folds and checkpoints are not refit. The published design supports (session,trial-position) material keys; original stimulus-file hashes are unavailable.

{chr(10).join(comparisons)}

Complete participant-only intervals and per-seed/per-cell metrics remain in the [session summary]({link(session/'comparison.json')}) and [pretraining summary]({link(reve/'comparison.json')}). The [crossed summary]({link(crossed/'comparison.json')}) contains all 22 contrasts, point-score agreement checks, bound prediction hashes and resampling digests. Neither set of intervals is adjusted for multiplicity. These three sessions change recording conditions, trial order and films together; there is no causal isolation of a video-identity effect and no uncertainty over a wider population of recording sessions.

## Pretrained model provenance, adapter and hardware

Use official [REVE-Base](https://huggingface.co/brain-bzh/reve-base) revision `dc2a075c287bb2f6c04ee5875bd79535a0f7dba6` and [positions](https://huggingface.co/brain-bzh/reve-positions) revision `befa5b57a455b77cf302daf610c2e9ed8140bace`. Both safetensors digests and the four manually reviewed author Python files are bound. Load the reviewed code explicitly from local paths and weights strictly with safetensors. Keep the 69,189,632-parameter encoders in frozen evaluation mode. The same official physical positions and seed42 constructor are used for the random control. Downloaded code, weights, waveforms and embeddings stay local. The [audit]({link(runs/'reve_audit_2026-10-06/code_review.json')}) and [download manifest]({link(runs/'reve_audit_2026-10-06/download_manifest.json')}) record exact revisions and hashes.

The [paper's Appendix B](https://arxiv.org/html/2510.21585v1) and the [open-subset pretraining card](https://huggingface.co/datasets/brain-bzh/reve-dataset) do not name these target corpora. The card covers only part of the full pretraining corpus. This is **a limited public-provenance check, not independent proof that all target recordings and people are absent from checkpoint training**. The pinned REVE Responsible Use License v1.0 permits this aggregate research; model weights are not redistributed.

Regenerated prefixes agree exactly with all 1,080 prior bandpower sequences. The [input record]({link(runs/'cache_reve_input_seediv/prepared.json')}) binds the waveform cache. After offline 4–40 Hz filtering/full-trial resampling, crop 40 seconds at 128 Hz, resample the prefix to 200 Hz and zscore each channel using **only that observation's** mean/std, then clip at ±15. This is stateless input normalization, with no target-population calibration. It differs from REVE pretraining bandwidth and recording-session normalization. Split into ten four-second windows and mean final-layer channel/patch tokens, then mean windows to 512 trial features. The official patcher directly covers 3.70 seconds per window: **37 seconds of patches within a 40-second normalized observation**. Between-window temporal order and the remaining 0.30 seconds per window are not encoded. This authored adapter is not a reproduction of published FACED scores or a measurement of fine-tuning potential.

The [outcome-free pilot]({link(runs/'reve_audit_2026-10-06/feasibility.json')}) tested one fixed input without a classifier or score: batches 1/2/4/10 produced embedding differences below 7.63e-6. Batch10 extraction takes **{sum(r['elapsed_seconds'] for r in state['features'].values()):.1f} seconds for both 1,080-trial encoders**, with peak **{max(r['peak_allocated_cuda_bytes'] for r in state['features'].values())/2**20:.2f} MiB allocated CUDA tensor memory** per process. Memory excludes CUDA context, driver/display allocations and other processes. Each frozen state hash is unchanged before/after extraction. The 90 small-model fits total **{sum(r['elapsed_seconds'] for r in records):.1f} fit seconds**, peak **{max(r['peak_allocated_cuda_bytes'] for r in records)/2**20:.2f} MiB**. Training and frozen extraction are different workloads; these figures demonstrate local feasibility, not a fair speed comparison. The existing PyTorch environment was used without installing dependencies.

## Verification and preserved evidence

The [session verification]({link(session/'verification.json')}) replays all 90 selected checkpoints on train/validation and all three test sessions, all 19,440 neural probabilities, 30 selected linear coefficients and 6,480 linear probabilities, source-only scalers/selection and matched exposure streams. It does not independently refit its 120 classical candidates. The [REVE verification]({link(reve/'verification.json')}) checks exact frozen state hashes, 20 sampled embeddings (zero replay difference), **all 160 candidate classifier refits and 40 selected coefficient/scaler models**, all 8,640 probabilities, trial coverage and bootstrap summaries. Its selected coefficients refit exactly; test probability reconstruction error is at most 1.12e-16. Full 1,080-embedding recomputation is not repeated. The [crossed analysis]({link(crossed/'verification.json')}) recomputes every resampling contrast and checks original point estimates.

The initial generic linear verifier reconstructed probabilities using logsumexp normalization. Differences of at most 1.78e-15 were amplified to 2.58e-8 in one confident conditional-binary validation log loss. An [independent verifier]({link(REPO/'scripts/verify_reve_frozen_probe.py')}) uses sklearn's stable softmax calculation and retains the original 1e-10 log-loss tolerance. [The diagnostic]({link(reve/'verification_diagnostic.json')}) is retained. No training input, label, selected model, metric definition or result changed. Earlier source files bound to completed experiments remain unchanged.

The original manuscript and PDF remain [archived]({link(REPO/'docs/paper_archive/2026-10-05-pre-exploration/README.md')}). The maintained suite has **43 passing tests**. These checks support reproducibility; they do not establish novelty, general unseen-corpus performance or acceptance at a strong conference.

## Research decision and next work

**Do not adopt a negative-transfer, transformer or pretrained-model contribution from these results.** This phase shows a reproducible session robustness limitation and a useful existing pretrained representation under one adapter. Similar stimulus/session evaluation and foundation modeling already have prior work, including [EMBC 2021](https://www.paperhost.org/proceedings/embs/EMBC21/files/1565.pdf) and [REVE](https://arxiv.org/html/2510.21585v1). It supplies a stricter development test, not a demonstrated new method. This study is SEED-IV only and does not test pooled three-corpus training or negative-transfer mitigation.

The next discriminating experiment should hold people and clips out **within the same recording session**, using matched training sizes and material rotations. That would reduce the recording-day confound in this session experiment. Any source-only adaptation or proposed mitigation should then survive that test, all three corpora and an unseen-corpus protocol, with matched existing baselines. Before proposing a paper pivot, specify the narrow gap relative to prior work and obtain evidence beyond these already inspected development folds. A more effective or novel method remains unestablished.

The [readiness checklist]({link(WORKSPACE/'GRU-XNet_Publication_Readiness_2026-10-05.md')}) still marks full repeated-grouping/LOSO and LODO, the matched full GRU-XNet architecture ablations, first-party DEAP recording authentication, historical provenance, final novelty/venue assessment and a revised compilable paper as unfinished. No research-question change follows automatically from this report. Show any concrete proposed pivot and its evidence to the author, obtain approval and archive the then-current manuscript immediately before adopting it.
"""
    destination.write_text(text,encoding="utf-8")
    print(json.dumps({"memo":str(destination),"figures":2,"verified_probability_rows":34560}))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__); parser.add_argument("--runs",type=Path,required=True); parser.add_argument("--destination",type=Path,required=True)
    a=parser.parse_args(); report(a.runs.resolve(),a.destination.resolve())
