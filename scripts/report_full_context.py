"""Report only completely replayed full-width controls; not a fitting source."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.full_context_controls_v2 import ALL, NEURAL, ARMS
from gruxnet.data import sha256, write_json

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent/'publication_runs'
NAMES={'gru':'Full GRU-XNet','lstm':'Matched BiLSTM','cbsatt_local':'Local CBSAtt',
       'gru_context':'GRU + context','prior':'Video prior (no EEG)','context_logistic':'Calibrated context (no EEG)'}


def number(value, percentage=False): return f'{value*(100 if percentage else 1):.2f}'


def interval(c, percentage=False):
    scale=100 if percentage else 1; lo,hi=c['crossed_percentile_95']
    return f'{c["difference"]*scale:+.2f} [{lo*scale:+.2f}, {hi*scale:+.2f}]'


def report():
    values={}; checks={}; rows=[]; diagnostic=[]; alignment=[]
    for dataset in ('SEEDIV','DEAP'):
        folder=ROOT/'within_video_alignment'
        if not json.loads((folder/f'verification_{dataset.lower()}.json').read_text())['passed']: raise ValueError('Alignment replay required before report')
        result=json.loads((folder/f'comparison_{dataset.lower()}.json').read_text())
        if result['eligible_trials']!=result['total_trials']: raise ValueError('Report the diagnostic eligible-cohort exclusions explicitly')
        if dataset=='DEAP':
            for model in NEURAL:
                for arm in ARMS:
                    for which in ('BA','logloss'):
                        c=result['models'][model][arm]['binary_'+which]; scale=100 if which=='BA' else 1
                        lo,hi=c['crossed_dyadic_percentile_95']
                        alignment.append(f'| {NAMES[model]} | {arm} | {which} | {c["difference"]*scale:+.2f} [{lo*scale:+.2f}, {hi*scale:+.2f}] | {c["nonestimable_bootstrap_draws"]} |')
    for dataset in ('SEEDIV','DEAP'):
        folder=ROOT/f'full_context_v2_{dataset.lower()}'
        verification=json.loads((folder/'verification.json').read_text())
        analysis=json.loads((folder/'analysis_verification.json').read_text())
        if not verification['passed'] or not analysis['passed']: raise ValueError('Complete verification required')
        value=json.loads((folder/'comparison.json').read_text()); values[dataset]=value; checks[dataset]=verification
        task='coarse3' if dataset=='SEEDIV' else 'binary'
        for name in ALL:
            a=value['models'][name]['exposed'][task]; b=value['models'][name]['unexposed'][task]
            rows.append(f'| {dataset} | {NAMES[name]} | {number(a["balanced_accuracy"],True)} | {number(b["balanced_accuracy"],True)} | {number(a["balanced_log_loss"])} | {number(b["balanced_log_loss"])} |')
        records=[json.loads((folder/'models'/i['id']/'metrics.json').read_text()) for i in json.loads((folder/'model_index.json').read_text())]
        dataset_diag=[]
        for record in records:
            final=record['history'][-1]
            row={k:record[k] for k in ('id','model','arm','session','rotation','fold','parameters','selected_step','elapsed_seconds','peak_allocated_cuda_bytes')}
            row.update(selected_training_BA=record['training'][task]['balanced_accuracy'],selected_test_BA=record['test'][task]['balanced_accuracy'],
                       selected_validation_BA=next(h['validation'][task]['balanced_accuracy'] for h in record['history'] if h['step']==record['selected_step']),
                       final_validation_BA=final['validation'][task]['balanced_accuracy'],final_training_batch_loss=final['training_loss'])
            dataset_diag.append(row); diagnostic.append(dict(dataset=dataset,**row))
        pd.DataFrame(dataset_diag).to_csv(folder/'diagnostics.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(12,4.7)); pos=np.arange(len(ALL)); width=.35
        for j,stat in enumerate(('balanced_accuracy','balanced_log_loss')):
            for k,arm in enumerate(ARMS):
                vals=[value['models'][n][arm][task][stat]*(100 if j==0 else 1) for n in ALL]
                axes[j].barh(pos+(k-.5)*width,vals,height=width,label='Shared videos' if arm=='exposed' else 'Unseen videos',color=('#3877ad','#de9541')[k])
            axes[j].set_yticks(pos,[NAMES[n] for n in ALL]); axes[j].invert_yaxis()
            axes[j].set_xlabel('Trial balanced accuracy (%)' if j==0 else 'Balanced log loss (lower is better)')
            axes[j].grid(axis='x',alpha=.2); axes[j].legend(fontsize=8)
            if j==0: axes[j].set_xlim(0,105)
        fig.suptitle(f'{dataset}: full-width models, fixed grouping 1 / 200 updates')
        fig.tight_layout(); fig.savefig(folder/'comparison.png',dpi=150); plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7)); steps=list(range(10,201,10))
        for ax,name in zip(axes.flat,NEURAL):
            for arm,color in zip(ARMS,('#3877ad','#de9541')):
                subset=[r for r in records if r['model']==name and r['arm']==arm]
                curves=np.array([[h['validation'][task]['balanced_accuracy']*100 for h in r['history']] for r in subset])
                ax.plot(steps,curves.mean(0),color=color,label='Shared videos' if arm=='exposed' else 'Unseen videos')
                ax.fill_between(steps,np.quantile(curves,.25,axis=0),np.quantile(curves,.75,axis=0),alpha=.12,color=color)
            ax.set_title(NAMES[name]); ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Validation balanced accuracy (%)'); ax.grid(alpha=.2)
            ax.legend(fontsize=8); ax.set_ylim(0,100)
        fig.suptitle(f'{dataset}: descriptive mean / fold interquartile range; validation videos unseen')
        fig.tight_layout(); fig.savefig(folder/'validation_curves.png',dpi=150); plt.close(fig)
        (folder/'README.md').write_text(f'# {dataset} matched full-model controls\n\nAll {len(records)} full neural fits and selected checkpoints replayed. Context-only candidates independently refitted. The comparison contains every predeclared accuracy and log-loss contrast; diagnoses and plots retain all models/arms. One fixed grouping and 200-update budget: development controls, without a convergence or publication-readiness claim. Source first-party authentication and historical-score provenance remain outstanding.\n',encoding='utf-8')
    diagnostics=pd.DataFrame(diagnostic)
    fit_seconds=diagnostics.elapsed_seconds.sum(); peak=diagnostics.peak_allocated_cuda_bytes.max()/2**20
    summary=[]
    for (dataset,name,arm),part in diagnostics.groupby(['dataset','model','arm'],sort=False):
        summary.append(f'| {dataset} | {NAMES[name]} | {arm} | {int(part.parameters.iloc[0]):,} | {part.selected_step.median():.0f} | {int(part.selected_step.eq(200).sum())}/{len(part)} | {part.selected_training_BA.mean()*100:.2f} | {part.selected_validation_BA.mean()*100:.2f} |')
    contrasts=[]; architecture=[]
    for dataset,value in values.items():
        task='coarse3' if dataset=='SEEDIV' else 'binary'
        for c in value['contrasts']:
            if c['task']==task and c['model_a'] in ('lstm','cbsatt_local') and c['model_b']=='gru':
                architecture.append(f'| {dataset} | {NAMES[c["model_a"]]} | {c["arm_a"]} | {c["statistic"]} | {interval(c,c["statistic"]=="BA")} |')
            if c['task']!=task or c['model_a']!='gru_context' or c['model_b'] not in ('prior','context_logistic'): continue
            contrasts.append(f'| {dataset} | {c["arm_a"]} | {NAMES[c["model_b"]]} | {c["statistic"]} | {interval(c,c["statistic"]=="BA")} |')
    exposure=[]
    for dataset,value in values.items():
        task='coarse3' if dataset=='SEEDIV' else 'binary'
        for c in value['contrasts']:
            if c['task']==task and c['statistic']=='BA' and c['model_a']==c['model_b']:
                exposure.append(f'| {dataset} | {NAMES[c["model_a"]]} | {interval(c,True)} |')
    text='''# Full-width GRU-XNet and contextual-prior findings — 6 October 2026

All 680 full-model fits are complete and independently replayed: 360 SEED-IV and 320 DEAP. There are also 170 source-only context calibration cells, with all 680 regularization candidates independently refitted. The study changes neither the manuscript nor the main research question. It is one predeclared participant/video grouping with a fixed 200-update training budget, not a historical-score reproduction or a submission-ready result.

## Matched primary results

Balanced accuracy is a percentage; balanced log loss is in natural-log units (lower is better). Shared-video and unseen-video arms have identical held-out people, validation/test trials and matched training person/class counts. The study uses all 1080 SEED-IV trials with neutral/negative/positive classes, and all 1264 retained DEAP trials with individual binary valence. These different targets should not be compared as a common leaderboard. The earlier feature controls used the same first 40 seconds, but more updates, larger batches and much smaller models, so their scores are not architecture-only comparisons.

| Dataset | Model | Shared-video BA | Unseen-video BA | Shared log loss | Unseen log loss |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## Matched reference-model contrasts

Each reference minus EEG-only GRU-XNet, using paired crossed participant/video percentile ranges. Positive balanced-accuracy differences (percentage points) or negative balanced-log-loss differences favor the reference. An interval spanning zero does not establish equivalence. The single base initialization scheme and fixed training budget leave optimization and initialization robustness unresolved; CNN/recurrent widths are matched for the BiLSTM swap, but parameter counts differ.

| Dataset | Reference | Arm | Statistic | Reference minus GRU [95% interval] |
|---|---|---|---|---|
'''+ '\n'.join(architecture)+'''

## Does EEG improve on context alone?

Training context excludes every label from the participant receiving that feature. Validation/test context comes only from outer training labels. A video receives Laplace-one per-class counts; an unknown video receives the global training distribution. The EEG-plus-context model adds the fixed log prior to full GRU logits and learns the residual; it has exactly the same EEG weights at initialization and parameter count as EEG-only GRU. Both the raw prior and source-validation-selected logistic calibration are retained as context-only controls.

These are EEG-plus-context minus each context-only comparator, with paired crossed participant/video 95% percentile intervals. Positive BA differences (percentage points) and negative log-loss differences favor EEG-plus-context. All intervals are conditional on these fixed fits/cohorts/grouping and unadjusted exploratory comparisons; improvement in one statistic does not imply general information gains or equivalence on the other.

| Dataset | Arm | Context-only comparator | Statistic | Difference [95% interval] |
|---|---|---|---|---|
'''+ '\n'.join(contrasts)+'''

SEED-IV's emotional labels are assigned to the videos and shared across participants: a known video can identify its target using training labels, without EEG. This is an intended contextual diagnostic, not a deployable EEG classifier or evidence that an EEG model used video identity. DEAP videos can receive both individual valence classes, so normative context remains imperfect. Neither dataset supplies test ratings to the models. Cross-fitting removes direct own-label leakage, but training videos have known priors while validation videos are unseen; context features consequently shift in distribution across roles. A residual model may fail to improve because of optimization, budget or that shift, even if EEG contains information.

## Correct individual EEG versus exchanged EEG within the same video

This no-refit supplement was declared while the full-model batch was running, before aggregate primary results. For every trial, compare its selected prediction with the scores obtained from **every other held-out participant who watched the same video in the same model/cell**. The video's source-only context and normalizer are identical, so exchanging verified donor probabilities exactly represents EEG exchange with recipient labels/context retained. Average correctness/log loss across donors, without ensembling probabilities. All 2344 trials have at least one other held-out donor, so the diagnostic excludes none.

These DEAP differences are correctly aligned minus exchanged EEG. Positive BA (percentage points) and negative log loss favor the correctly paired recording. Intervals use paired dyadic recipient-and-donor participant weights and video weights, with identical observed-class denominators. Nonestimable draws are disclosed rather than replaced. All six models, both arms and both SEED tasks are retained in the machine-readable comparisons.

| Model | Arm | Statistic | Aligned minus exchanged [crossed dyadic 95% interval] | Nonestimable draws |
|---|---|---|---:|---:|
'''+ '\n'.join(alignment)+'''

Both source-only context controls are invariant to exchange. SEED-IV's assigned video labels imply zero aggregate differences for all models, including the symmetric pair-weighted bootstrap; that mathematical sanity check passes. This does not rule out useful EEG for classifying *unseen* SEED-IV videos. On DEAP, an alignment effect would support recording/rating association conditional on video in these fits. It could also involve stable participant traits, demographics or artifacts, and would not identify a causal physiological emotion mechanism. It uses a limited donor cohort and one fixed development partition.

## Material sensitivity

Unseen-video minus shared-video balanced accuracy; all six models retained. The arm change replaces training recordings/content and can change difficulty, order or subject responses. It does not isolate a causal effect of identity and is not evidence of joint multi-corpus negative transfer.

| Dataset | Model | Difference in percentage points [crossed 95% interval] |
|---|---|---:|
'''+ '\n'.join(exposure)+'''

All architecture contrasts, both primary and secondary SEED-IV tasks, and both uncertainty calculations are preserved in the machine-readable comparisons, including nonsignificant results. SEED-IV secondary binary results condition positive probability on positive/negative mass and exclude all 270 neutral trials. The resampling ranges are exploratory: nominal coverage has not been demonstrated for these small fixed partitions and dyadic dependence, and no uniformly valid bootstrap claim is made.

## Budget and training behavior

| Dataset | Model | Arm | Parameters | Median selected update | Selected at last update | Mean selected train BA (%) | Mean selected validation BA (%) |
|---|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(summary)+f'''

Recorded neural fitting time is {fit_seconds/60:.2f} minutes; maximum PyTorch-allocated CUDA memory is {peak:.2f} MiB. These exclude data preparation, environment/driver memory, reporting, independent replay and context fitting. Per-cell training metrics, complete 20-point validation histories, resource counts and selected checkpoint positions are retained. A checkpoint chosen at the budget limit is a reason to consider a separately predeclared longer experiment; a checkpoint selected earlier is not proof of convergence. Small validation populations and repeated checkpoint comparisons introduce selection uncertainty.

## Implementation fidelity and verification

The dynamic model retains the full original 32/64/128 independent electrode CNN widths, two-layer bidirectional 128-unit GRU, four-head attention with residual normalization and the 256/128 classifier. The matched BiLSTM swaps only recurrence; same hidden width does not mean equal parameter count. Both use the same initialization seed and common CNN initialization, but different recurrent parameter sizes consume different random draws before attention/head initialization; those later weights are not identical. GRU versus GRU-plus-context starts with the entire EEG state identical. The second reference reproduces the local CBSAtt implementation, including global pooling and its one-step, single-layer LSTM. It is not certified author code or a reproduction of published performance.

Inputs are adapted to the same physical common14 channels and 40-second offline prefix, with real 4–40 Hz bins and 79 STFT frames, producing nine ordered recurrent steps after three pools. This differs from the historical interpolated 129×126/max-channel/augmented pooled configuration. Grouped convolutions preserve distinct weights and BatchNorm statistics per electrode. Weight-mapped forward outputs for both binary and three-class heads match the original local implementations within the predeclared tolerance. All 2344 input spectrograms independently match SciPy exactly after double-precision transform calculation and float32 storage; the original tolerance was retained through the preflight precision correction.

Every fit binds its input and source hashes, original trial split, canonical training draws, initial state, training-only scaler, participant-excluded context, history, selected checkpoint and predictions. Independent verification reconstructs all these, replays selected training/validation/test predictions and selection rules, refits every context candidate and checks once-per-trial out-of-fold coverage. This is selected-checkpoint replay, not rerunning every neural optimization trajectory. Largest neural probability error: SEED-IV {checks['SEEDIV']['maximum_neural_probability_error']:.3g}; DEAP {checks['DEAP']['maximum_neural_probability_error']:.3g}. Analysis was independently recomputed exactly before reporting.

The first partial batch was stopped after 15 completed fits when a training-mode CBSAtt dropout/pooling discrepancy was found. Revision 2 corrects the ordering and adds training-mode checks. The partial fits and original source hashes are preserved as excluded preflight evidence; all 680 reported fits were restarted under the corrected declaration without changing splits, budgets or hyperparameters.

## Remaining publication work

These development results do not settle full-model convergence, independent optimizer/grouping robustness, unseen-corpus transfer, all-corpus joint learning, original first-party DEAP signal authentication, or historical 95.91% provenance. The manuscript still needs an approved contribution, revised claims, matched final experiments and compilable source figures. No new research question has been adopted. A concrete proposal requires the author's approval, and the then-current paper must be archived immediately before an approved pivot.

## Evidence and reproduction

- [Predeclared protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/Full_Context_Control_Protocol_2026-10-06.md)
- [Additional primary prior work and inference limits](GRU-XNet_Context_Alignment_Research_Update_2026-10-06.md)
- [Public-probability reanalysis check](publication_runs/full_context_public_reanalysis_2026-10-06.json)
- [Machine-readable plan](publication_runs/full_context_v2_plan_2026-10-06.json)
- [Within-video declaration](publication_runs/within_video_alignment_plan_2026-10-06.json), [DEAP alignment contrasts](publication_runs/within_video_alignment/comparison_deap.json), [SEED-IV sanity controls](publication_runs/within_video_alignment/comparison_seediv.json)
- [SEED-IV complete contrasts](publication_runs/full_context_v2_seediv/comparison.json), [verification](publication_runs/full_context_v2_seediv/verification.json), [diagnostics](publication_runs/full_context_v2_seediv/diagnostics.csv), [scores](publication_runs/full_context_v2_seediv/comparison.png), [validation curves](publication_runs/full_context_v2_seediv/validation_curves.png)
- [DEAP complete contrasts](publication_runs/full_context_v2_deap/comparison.json), [verification](publication_runs/full_context_v2_deap/verification.json), [diagnostics](publication_runs/full_context_v2_deap/diagnostics.csv), [scores](publication_runs/full_context_v2_deap/comparison.png), [validation curves](publication_runs/full_context_v2_deap/validation_curves.png)

With the already prepared local datasets/caches, run `python scripts/full_context_controls_v2.py plan --root ../publication_runs` once on a fresh study, prepare each dataset, run `python scripts/audit_full_context_inputs.py` and `python scripts/audit_full_context_models_v2.py`. Declare the supplement with `python scripts/within_video_alignment.py plan` before running `python scripts/full_context_controls_v2.py batch --root ../publication_runs`; after both primary studies pass replay, run `python scripts/within_video_alignment.py analyze` and `python scripts/report_full_context.py`. Existing declarations/caches are protected; the batch resumes only hash-matching fitted records. A public checkout can run `python scripts/verify_publication_export.py --export-only` and `python scripts/verify_full_context_export.py` to check artifact integrity and recompute all reported probability-based numbers without raw EEG or checkpoints. Selected-model replay requires those local files.
'''
    (REPO.parent/'GRU-XNet_Full_Context_Findings_2026-10-06.md').write_text(text,encoding='utf-8')
    print(json.dumps({'neural_fits':len(diagnostics),'fit_minutes':fit_seconds/60,'peak_allocated_MiB':peak,'contrasts':sum(len(v['contrasts']) for v in values.values())},indent=2))

if __name__=='__main__': report()
