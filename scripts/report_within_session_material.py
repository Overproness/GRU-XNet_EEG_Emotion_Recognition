"""Render the complete verified matched-material study without changing fitted sources."""
from __future__ import annotations
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import platform
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO=Path(__file__).resolve().parents[1]
WORKSPACE=REPO.parent
NAMES={"mean_mlp":"Mean bandpower MLP","transformer":"Temporal transformer",
       "bandpower":"Bandpower logistic","duration":"Duration only (no EEG)",
       "reve_pretrained":"Frozen REVE pretrained","reve_random42":"Frozen REVE random42"}
ARMS=("exposed","unexposed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100*value:.2f}%"


def pp(value):
    return f"{100*value:+.2f}"


def ci(values):
    return f"[{100*values[0]:+.2f}, {100*values[1]:+.2f}]"


def link(path):
    return path.relative_to(WORKSPACE).as_posix()


def render(run,destination):
    verification=read(run/"verification.json")
    if not verification["passed"] or verification["neural_checkpoints_replayed"]!=540:
        raise ValueError("A full successful verification is required before reporting")
    if (sum(v["selected_heads"] for v in verification["linear"].values())!=360 or
        sum(v["candidates_independently_refitted"] for v in verification["linear"].values())!=1440 or
        verification["neural_probability_rows"]+sum(v["probability_rows"] for v in verification["linear"].values())!=21600):
        raise ValueError("Incomplete verified linear/prediction budget")
    comparison=read(run/"comparison.json"); models=comparison["models"]
    if set(models)!=set(NAMES) or len(comparison["contrasts"])!=20:
        raise ValueError("Incomplete model or contrast population")
    exposure=[c for c in comparison["contrasts"] if c["model_a"]==c["model_b"]]
    crossed_zero=sum(c["crossed_percentile_95"][0]<=0<=c["crossed_percentile_95"][1] for c in exposure)
    def contrast(a,arm,b,other,task="coarse3"):
        return next(c for c in comparison["contrasts"] if
                    (c["model_a"],c["arm_a"],c["model_b"],c["arm_b"],c["task"])==(a,arm,b,other,task))
    transformer_drop=contrast("transformer","unexposed","transformer","exposed")
    transformer_advantage=contrast("transformer","unexposed","mean_mlp","unexposed")
    pretrained_advantage=contrast("reve_pretrained","unexposed","reve_random42","unexposed")
    records=[]
    for name in ("mean_mlp","transformer"):
        for session in (1,2,3):
            records.extend(read(run/f"model_metrics_{name}_session{session}.json"))
    if len(records)!=540: raise ValueError("Incomplete neural metric records")
    fit_rows=[]; selection_rows=[]; cells=[]; seeds=[]
    for r in records:
        history=read(run/"models"/r["id"]/"history.json")
        chosen=next(h for h in history if h["step"]==r["selected_step"])
        fit_rows.append({"model":r["model"],"arm":r["arm"],"session":r["session"],
                        "rotation":r["rotation"],"seed":r["seed"],"fold":r["fold"],
                        "selected_step":r["selected_step"],
                        "train_BA":r["training"]["coarse3"]["balanced_accuracy"],
                        "validation_BA":chosen["validation"]["coarse3"]["balanced_accuracy"],
                        "test_BA":r["test"]["coarse3"]["balanced_accuracy"],
                        "fit_seconds":r["elapsed_seconds"],
                        "peak_allocated_cuda_bytes":r["peak_allocated_cuda_bytes"]})
    fits=pd.DataFrame(fit_rows)
    fits.to_csv(run/"training_diagnostics.csv",index=False)
    for name,conditions in models.items():
        for arm,item in conditions.items():
            for seed,metrics in item["seeds"].items():
                seeds.append({"model":name,"arm":arm,"seed":int(seed),
                              **{f"{task}_BA":metrics[task]["balanced_accuracy"] for task in ("coarse3","binary")}})
            for cell,per_seed in item["cells"].items():
                for seed,metrics in per_seed.items():
                    cells.append({"model":name,"arm":arm,"cell":cell,"seed":int(seed),
                                  **{f"{task}_BA":metrics[task]["balanced_accuracy"] for task in ("coarse3","binary")}})
    pd.DataFrame(cells).to_csv(run/"all_cells.csv",index=False)
    pd.DataFrame(seeds).to_csv(run/"all_seeds.csv",index=False)
    table=["| Model | Shared test materials: 3-class BA | Unseen test materials: 3-class BA | Shared: binary BA | Unseen: binary BA |",
           "|---|---:|---:|---:|---:|"]
    for name in NAMES:
        table.append("| "+NAMES[name]+" | "+" | ".join(
            pct(models[name][arm][task]["mean_BA"]) for task,arm in
            (("coarse3","exposed"),("coarse3","unexposed"),("binary","exposed"),("binary","unexposed")))+" |")
    contrasts=["| Paired contrast (A minus B) | Target | Difference (pp) | Person-only 95% interval | Person/material 95% interval |",
               "|---|---|---:|---:|---:|"]
    for c in comparison["contrasts"]:
        if c["model_a"]==c["model_b"]:
            name=f"{NAMES[c['model_a']]}: unseen minus shared"
        else:
            name=f"{NAMES[c['model_a']]} minus {NAMES[c['model_b']]} ({c['arm_a']})"
        contrasts.append(f"| {name} | {c['task']} | {pp(c['BA_difference'])} | {ci(c['participant_percentile_95'])} | {ci(c['crossed_percentile_95'])} |")
    for (model,arm),part in fits.groupby(["model","arm"],sort=False):
        selection_rows.append(f"| {NAMES[model]} | {arm} | {part.selected_step.median():.0f} | "
                              f"{pct(part.train_BA.mean())} | {pct(part.validation_BA.mean())} | "
                              f"{pct(part.test_BA.mean())} |")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
    positions=np.arange(len(NAMES))
    for ax,task,chance in zip(axes,("coarse3","binary"),(1/3,.5)):
        for arm,offset,color,label in (("exposed",-.12,"#2878b5","Shared materials"),
                                       ("unexposed",.12,"#c95636","Unseen materials")):
            points=np.array([models[n][arm][task]["mean_BA"] for n in NAMES])*100
            limits=np.array([models[n][arm][task]["crossed_percentile_95"] for n in NAMES])*100
            ax.hlines(positions+offset,limits[:,0],limits[:,1],color=color,lw=1.6)
            ax.scatter(points,positions+offset,color=color,label=label,s=30,zorder=3)
        ax.axvline(chance*100,color="gray",ls="--",lw=1)
        ax.set_title("Three-class valence" if task=="coarse3" else "Conditional binary valence")
        ax.set_xlabel("Trial balanced accuracy (%)")
        ax.grid(axis="x",alpha=.15)
    axes[0].set_yticks(positions,list(NAMES.values())); axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("Within-session held-out people: matched material exposure\nCrossed person/material intervals; development evidence",fontsize=12)
    fig.tight_layout(); fig.savefig(run/"comparison.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
    for ax,task in zip(axes,("coarse3","binary")):
        selected=[c for c in comparison["contrasts"] if c["task"]==task and c["model_a"]==c["model_b"]]
        for i,c in enumerate(selected):
            lo,hi=np.array(c["crossed_percentile_95"])*100
            ax.hlines(i,lo,hi,color="#2878b5",lw=2)
            ax.scatter(100*c["BA_difference"],i,color="#2878b5",s=30,zorder=3)
        ax.axvline(0,color="gray",ls="--",lw=1); ax.grid(axis="x",alpha=.15)
        ax.set_title("Three-class valence" if task=="coarse3" else "Conditional binary valence")
        ax.set_xlabel("Unseen minus shared balanced accuracy (pp)")
    axes[0].set_yticks(positions,list(NAMES.values())); axes[0].invert_yaxis()
    fig.suptitle("Paired material exposure contrasts\nUnadjusted crossed person/material 95% intervals",fontsize=12)
    fig.tight_layout(); fig.savefig(run/"paired_comparison.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    environment={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
                 "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
                 "device":torch.cuda.get_device_name(),"report_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run/"environment.json").write_text(json.dumps(environment,indent=2)+"\n",encoding="utf-8")
    message=f"""# Matched within-session material findings — 6 October 2026

**Completed and verified:** 540 neural fits, 360 selected linear heads, all 1,440 independently refitted linear candidates and 21,600 test probability rows. The primary question here is whether performance changes when test stimulus materials are absent from training, while holding participant groups, session, training size and test trials fixed. This is SEED-IV development evidence. **The manuscript and research question remain unchanged; no pivot is approved.**

**Research assessment:** all {crossed_zero} of {len(exposure)} material-exposure contrast intervals include zero when both people and materials are resampled. The transformer falls from {pct(models['transformer']['exposed']['coarse3']['mean_BA'])} to {pct(models['transformer']['unexposed']['coarse3']['mean_BA'])}, difference {pp(transformer_drop['BA_difference'])} pp, crossed interval {ci(transformer_drop['crossed_percentile_95'])}. These data allow a meaningful drop as well as a small or absent average effect; they do not establish equivalence. They do not support adopting a strong material-generalization or negative-transfer contribution yet.

On unseen materials, transformer-minus-MLP is {pp(transformer_advantage['BA_difference'])} pp with crossed interval {ci(transformer_advantage['crossed_percentile_95'])}. Frozen pretrained-minus-random REVE is {pp(pretrained_advantage['BA_difference'])} pp with crossed interval {ci(pretrained_advantage['crossed_percentile_95'])}; its unseen-material advantage is uncertain under this analysis, despite a positive point estimate. The current controls establish no clear new model advantage.

## Identical comparison population

The [protocol]({link(REPO/'docs/publication/Within_Session_Material_Control_Protocol_2026-10-06.md')}) and [plan]({link(run/'plan.json')}) were saved and pushed before fitting. Use all 1,080 original trials from 15 people, three sessions and 72 design-supported material keys. Each fit uses one session and disjoint nine/three/three training/validation/test people. Both arms have **108 training, 12 validation and 24 identical test trials**, with equal native-emotion counts and training people. Test clips are present among other training people in the shared-material arm, and excluded from training and selection in the unseen-material arm. Validation clips are disjoint from both training and test clips in both arms.

Within every session/native-emotion stratum, six clips receive a fixed random rank before fitting. Three rotations test every clip once; five participant rotations test every person once. Four training clips are common to the arms and eight are replaced. No test person's data enter fitting, normalization or selection across any session. Source-only feature scalers are refitted for every condition.

Neural models retain the previously declared 19,079-parameter mean MLP and 19,075-parameter temporal transformer. Each receives 600 balanced 60-trial updates, with identical participant/emotion/rank draw streams and initial weights in paired arms. Three initialization seeds are averaged, not counted as additional participants. Checkpoints are chosen from validation every 25 updates using three-class BA then balanced log loss. No augmentation, scheduler, target-population calibration or test-based choice occurs.

## All model results

{chr(10).join(table)}

Primary targets are neutral / negative (sadness and fear) / positive (happiness), with 33.33% chance BA. Conditional binary excludes the same 270 neutral trials, leaving 810 observations; chance BA is 50%. Binary probabilities retain the defined neutral-mass floor and negative tie rule. Targets are stimulus-assigned classes, not individual self-reported valence.

![All models and crossed intervals]({link(run/'comparison.png')})

The bandpower logistic control uses the same first-ten four-second frames, averaged per trial. The duration control uses full original trial length with **no EEG** and is a diagnostic, not an eligible EEG superiority baseline. The frozen pretrained and random42 REVE features reuse the [verified encoder audit and adapter]({link(WORKSPACE/'GRU-XNet_Session_Pretraining_Findings_2026-10-06.md')}); no encoder is updated. These heads use the same source-only participant/material splits, four C candidates and validation selection. Only one random encoder initialization is tested.

## All paired contrasts and uncertainty

Resample 15 people and independently six materials within each of twelve session/native-emotion strata using 10,000 fixed paired draws. Average neural-seed correctness within each person/material cell. Participant-only draws hold clips fixed; crossed draws also vary observed material keys. Folds, fitted models, three observed sessions and the split assignments remain fixed. Percentile intervals are exploratory and unadjusted across 20 contrasts.

{chr(10).join(contrasts)}

![Material exposure differences]({link(run/'paired_comparison.png')})

All seed metrics and all nine session/material-rotation cells are retained in [all_seeds.csv]({link(run/'all_seeds.csv')}) and [all_cells.csv]({link(run/'all_cells.csv')}); full metrics and both uncertainty analyses are in [comparison.json]({link(run/'comparison.json')}). Never choose a preferred seed or cell from its test score.

## Selection noise and hardware

Each validation set has only twelve trials: three neutral, six negative and three positive. This makes model selection noisy. The full 600-update budget is retained even when an early checkpoint is selected. The following means describe selected per-fit train/validation/test scores; primary headline scores above aggregate original out-of-fold trials and average seeds.

| Neural model | Arm | Median selected update | Mean train BA | Mean validation BA | Mean test BA |
|---|---|---:|---:|---:|---:|
{chr(10).join(selection_rows)}

The [540-fit diagnostic table]({link(run/'training_diagnostics.csv')}) records every selected update, split metric, time and allocation. Fits total **{fits.fit_seconds.sum():.1f} seconds**, with peak **{fits.peak_allocated_cuda_bytes.max()/2**20:.2f} MiB allocated CUDA tensors**, on the RTX 3050 using the existing PyTorch environment. Tensor memory excludes CUDA context, display/driver allocations and other processes. Frozen extraction occurred in the earlier phase and is not included in this fit-time measure.

## Verification, scope and research decision

The [verification record]({link(run/'verification.json')}) binds source hashes, input/cache metadata, material assignments and participant folds. It independently replays all 540 selected neural checkpoints and their train/validation/test metrics, training-only scalers, paired batch signatures, initial states and validation selection. Maximum neural probability replay error is {verification['max_neural_probability_error']:.3g}. It refits all 1,440 linear candidates, reproduces all 360 selected coefficients, checks every prediction row, exact once-per-trial coverage and recomputes the complete paired bootstrap. The scientific-control suite has 47 passing tests. Downloaded author code, weights, waveforms, embeddings and checkpoints remain local; bounded derived evidence is exported.

This comparison reduces the preceding recording-session confound by keeping both arms inside the same session index with the same people and test observations. It still changes training clip content, difficulty, chronological position and duration. It does **not** isolate a causal video-identity mechanism. Original media hashes are unavailable, so session/trial position is a published-design material proxy. This cohort has already been inspected in earlier development work. The REVE open-subset corpus documentation does not independently certify exclusion of all target recordings from the complete pretrained checkpoint.

Earlier session-control fits used 216 training and 72 validation trials; this experiment uses 108 and 12. Cross-phase score changes cannot be attributed only to material generalization. Within-phase pairs have equal access and size. These results do not test one jointly selected three-corpus model, unseen-corpus transfer or a new mitigation method.

The [focused prior-work audit]({link(WORKSPACE/'GRU-XNet_Material_Generalization_Research_Update_2026-10-06.md')}) records the EMBC 2021 subject/material precedent and recent stimulus-aware, data-centric and brain-region-transformer work. Holding out materials or adding a transformer alone is not an established novel contribution. Use the entire result to decide whether further robustness experiments are warranted; do not adopt a paper pivot automatically.

The earlier transformer session-transfer difference was -5.74 pp with crossed interval [-9.51,-2.21]. The present within-session estimate is smaller and less precise, but training/validation size and material access also differ between phases. It would be incorrect to infer that recording-day shift alone caused the earlier drop. Repeat independent material/participant groupings and extend a compatible binary control to DEAP before deciding whether a systematic robustness contribution is supported. The metadata audit identifies GAMEEMO's sparse game-condition and class coverage constraints; that extension needs a different explicitly declared protocol. No further grouping or corpus experiment is represented as completed here.

The [readiness checklist]({link(WORKSPACE/'GRU-XNet_Publication_Readiness_2026-10-05.md')}) still tracks unfinished matched full GRU-XNet ablations, broader repeated subject/LODO comparisons, first-party DEAP authentication, historical provenance, novelty assessment and a revised compilable paper. The manuscript remains [archived]({link(REPO/'docs/paper_archive/2026-10-05-pre-exploration/README.md')}). Show a concrete proposed research question with evidence, obtain author approval, and archive the then-current paper immediately before adopting any change.
"""
    destination.write_text(message,encoding="utf-8")
    print(json.dumps({"report":str(destination),"neural_fits":540,"selected_linear":360,"probability_rows":21600}))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    render(args.run.resolve(),args.destination.resolve())
