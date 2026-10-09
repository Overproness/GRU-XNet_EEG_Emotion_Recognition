"""Render complete audited source-learning tables; no new fitting or recipe selection."""
from pathlib import Path
import json
import pandas as pd
REPO=Path(__file__).resolve().parents[1]
STUDY='cbramod_learning_2026-10-09'
BASE=REPO/'results/development'/STUDY
LINK=f'GRU-XNet_EEG_Emotion_Recognition/results/development/{STUDY}'


def pair(frame,column,percentage=False,scientific=False):
    frame=frame.sort_values('group')
    if frame.group.tolist()!=[1,2]:raise ValueError('Missing report grouping')
    return ' / '.join(f'{x*100:.2f}' if percentage else (f'{x:.2e}' if scientific else f'{x:.4f}') for x in frame[column])


def main():
    proof=json.loads((BASE/'postfit_analysis/verification.json').read_text())
    if not proof['complete']:raise ValueError('Complete audited findings required')
    selected=pd.read_csv(BASE/'postfit_analysis/selected_metrics.csv')
    capacity=pd.read_csv(BASE/'postfit_analysis/capacity_outcomes.csv')
    all_metrics=pd.read_csv(BASE/'postfit_analysis/all_metrics.csv')
    if len(selected)!=96 or len(capacity)!=16 or len(all_metrics)!=816:raise ValueError('Incomplete report coverage')
    passed=int(capacity.capacity_pass.sum())
    fitting=json.loads((BASE/'verification.json').read_text())
    neural_records=[json.loads(p.read_text()) for kind in ('tiny','long') for p in (BASE/kind).glob('*/record.json')]
    if len(neural_records)!=32 or fitting['states_checked']!=304 or fitting['metric_sets']!=816:
        raise ValueError('Incomplete state/resource coverage')
    peak=max(r['peak_allocated_bytes'] for r in neural_records)
    selected_train=selected[selected.role=='train']
    early_heads=int(((selected_train.kind=='head')&(selected_train.step==200)).sum())
    early_fine=int(((selected_train.kind=='long')&(selected_train.trainable==True)&(selected_train.step==200)).sum())
    cap_min=capacity.balanced_log_loss.min();cap_max=capacity.balanced_log_loss.max()
    text=f'''# CBraMod source capacity, head optimization and longer schedules

Completed 9 October 2026. The authorized diagnostic completes all **80 trajectories, 304 states and 816 probability metric sets**. All 48 cached-feature heads, 16 tiny capacity controls and 16 longer neural runs verify. The main research question and manuscript remain unchanged. Existing cohorts and source selections remain development evidence.

## Findings and immediate recommendation

**The tiny-set capacity concern is resolved under the easier declared setup; reliable source generalization is still unresolved.** All sixteen real/permuted controls reach 100% training balanced accuracy; their final balanced cross entropy ranges from {cap_min:.2e} to {cap_max:.2e}. This includes both pretrained and random encoders in both datasets/groupings. The code and native inputs can support memorization. Those runs disable dropout, smoothing and decay together, so they neither isolate one optimization factor nor demonstrate useful emotion prediction.

Training-only feature standardization improves fitting under the matched head grid. At the fixed LR 0.01 / final 2,400 endpoint, pretrained standardized-head training BA is **76.27/70.08% on DEAP and 98.15/100% on SEED-IV**, versus raw-head **60.21/58.73% and 62.35/69.14%**. These endpoints are reported as learning diagnostics; validation does not select them. All four pretrained standardized source-selected heads have worse unseen-validation point log loss than their raw counterparts. Selected unseen BA changes in opposite directions between groupings: DEAP 64.12/41.90%, SEED-IV 22.22/50%. Therefore scaling gives a fitting signal without a robust validation lead.

Longer full-source fine-tuning also demonstrates learning, especially from random initialization. At update 1,200, random fine-tuned training BA reaches **78.87/82.52% on DEAP and 100/99.38% on SEED-IV**. Pretrained fine-tuned training BA is **57.58/58.02% and 69.14/67.28%**. Pretrained DEAP learning remains limited under this particular regularized recipe; neither the tiny capacity success nor sparse longer checkpoints certifies full-source convergence.

Training improvements do not consistently carry through to validation. Random fine-tuned SEED-IV final unseen BA is **27.78/22.22%**, with losses **1.6610/2.1087** against the uniform reference ln(3)=1.0986. The source-selected pretrained fine-tuned unseen BA is **44.90/42.11% on DEAP and 44.44/33.33% on SEED-IV**; the matched frozen values are **51.57/44.74% and 38.89/38.89%**. Pretrained fine-tuning has worse unseen-validation point loss than its frozen counterpart in all four panels. There is no consistent gain across groupings, metrics and matched controls; no population significance claim is made.

**{early_heads}/16 head selections and {early_fine}/8 fine-tuning selections choose the earliest eligible 200-update checkpoint.** All runs nevertheless finish their declared schedules. The early source selections and rising validation loss are evidence against treating a larger budget alone as the current solution. They are not a proof of global convergence or of absent EEG information.

The next bounded priority is **matched representation conditioning and training-mode diagnosis**, rather than adding more encoders now. Keep the same ten-second training stream, pooling/head, initialization, labels, learning rates, budget and source boundaries; isolate training-only embedding scaling and encoder dropout in separately declared controls. The current cached forty-second head versus stochastic ten-second neural comparison changes multiple factors and cannot identify which causes limited pretrained learning. Physical input calibration and native four-emotion SEED-IV suitability also remain open; do not change amplitudes, labels or exclusions in response to validation scores. A larger encoder/held-out programme needs a reproducible matched validation lead first, followed by independent confirmation and a distinct prior-work gap. This recommendation is exploratory and does not change the main paper question.

## Fixed design and interpretation limits

The [protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/CBraMod_Learning_Protocol_2026-10-09.md) was declared and pushed before fitting in commit `148ee4ca8`; plan SHA256 `d862d821ca1b8c046aad3bd3b60605c2f58792876ad6ec5dac6a9b59c8cee34e`. All 48 bound source files remain unchanged. Both reused participant/video groupings, native 32/62 channels, forty-second input, corrected DEAP binary individual valence and existing SEED-IV assigned coarse three-class task are preserved. No new outer-test inference, corpus, original-label mapping or calibration change is made. Actual waveform/feature hashes are rechecked before declaration and after fitting.

Both source-validation roles contain participants excluded from their grouping's training. Familiar validation uses training videos; unseen validation additionally holds out videos. The separate outer-test participants/materials remain unused. All roles comprise original trials, with training-only sampling and no synthetic augmentation ancestry.

Cached averaged 200-dimensional features compare raw/training-only standardization and head rates `0.001*sqrt(6/256)`, 0.001 and 0.01. Every FP64 CPU head uses 2,400 AdamW updates, balanced six-trial draws, smoothing 0.1, weight decay 0.05, clipping at 1 and cosine decay. Initial/200/800/2,400 states and full source probabilities are retained. Each pretraining/scaling condition selects among three rates and three positive steps using equal familiar/unseen balanced log loss. Cached embeddings have no training dropout or window variation, so this is not an exact substitute for end-to-end training.

Each tiny run uses twelve original training trials and one fixed ten-second window per trial, pretrained/random encoders and real/class-preserving permuted training targets. Original labels remain unchanged; copied capacity targets are explicitly marked. All dropout, smoothing and weight decay are disabled; constant head LR 0.001/encoder LR 1e-4, clipping at 1 and 400 updates deliberately make this an easier fitting test. Balanced draw pools follow the diagnostic target, so real/permuted observation ordering can differ. Initial/200/400 states are retained, with only `capacity_train` predictions. Fixed success requires BA >=0.95 AND unsmoothed balanced CE <=0.10 at update 400. **{passed}/16 capacity controls pass this declared criterion.** Memorization, including random labels, does not demonstrate emotion generalization.

Longer runs compare pretrained/random × frozen/trainable encoders through 1,200 updates. Head LR 0.001 and encoder LR 1e-4 were fixed before head/capacity outcomes. Both encoder conditions preserve training-mode dropout; use balanced six-trial/window streams, smoothing 0.1, weight decay 0.05, clipping at 1 and cosine horizon 1,200. Inference averages four window logits before softmax. The first 200 sampling draws match the previous study, but head rate/cosine horizon differ: old/new differences cannot be attributed solely to duration. Within this study, initial/200/600/1,200 are checkpoints of one continuation. Each condition selects among three positive steps by the same source-only criterion; all sixteen long conditions run irrespective of cheap diagnostic results.

## All tiny capacity outcomes

Every slash separates grouping 1/grouping 2. Accuracy is balanced percent; loss is balanced unsmoothed cross entropy on the twelve training observations. Passing requires both conditions, and all failed controls are retained.

| Dataset | Encoder | Target | Final BA (%) | Final loss | Pass, groups 1 / 2 |
| --- | --- | --- | ---: | ---: | --- |
'''
    for dataset in ('DEAP','SEEDIV'):
        for pretrained in (True,False):
            for target in ('real','permuted'):
                data=capacity[(capacity.dataset==dataset)&(capacity.pretrained==pretrained)&(capacity.target==target)].sort_values('group')
                flags=' / '.join('yes' if v else 'no' for v in data.capacity_pass)
                text+=f'| {dataset} | {"Pretrained" if pretrained else "Random42"} | {target} | {pair(data,"balanced_accuracy",True)} | {pair(data,"balanced_log_loss",scientific=True)} | {flags} |\n'
    text+=f'\n![Every fixed capacity outcome]({LINK}/postfit_analysis/capacity_outcomes.png)\n\n'
    text+='## All source-selected head and longer-neural outcomes\n\n'
    text+='Each slash again denotes grouping 1/grouping 2. These familiar/unseen source-validation observations participate in selection. Uniform BA/loss references are 50%/ln(2) for DEAP and 33.33%/ln(3) for SEED-IV. There are no untouched-test scores or population intervals. Sixteen head conditions have nine candidates each; sixteen long conditions have three each.\n\n'
    for dataset in ('DEAP','SEEDIV'):
        text+=f'### {dataset}\n\n| Model | Selected setting, groups 1 / 2 | Train BA (%) | Familiar BA (%) | Unseen BA (%) | Familiar loss | Unseen loss |\n'
        text+='| --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
        for kind in ('head','long'):
            field='scaled' if kind=='head' else 'trainable'
            for pretrained in (True,False):
                for variant in (False,True):
                    data=selected[(selected.dataset==dataset)&(selected.kind==kind)&(selected.pretrained==pretrained)&(selected[field]==variant)]
                    parts={r:data[data.role==r].sort_values('group') for r in ('train','validation_familiar','validation_unseen')}
                    name=('Pretrained' if pretrained else 'Random42')+(': standardized head' if variant else ': raw head') if kind=='head' else \
                         ('Pretrained' if pretrained else 'Random42')+(': fine-tuned long' if variant else ': frozen long')
                    settings=' / '.join(f'{int(r.step)} updates'+(f', LR={r.head_rate:.4g}' if kind=='head' else '') for _,r in parts['train'].iterrows())
                    text+=f'| {name} | {settings} | {pair(parts["train"],"balanced_accuracy",True)} | {pair(parts["validation_familiar"],"balanced_accuracy",True)} | {pair(parts["validation_unseen"],"balanced_accuracy",True)} | {pair(parts["validation_familiar"],"balanced_log_loss")} | {pair(parts["validation_unseen"],"balanced_log_loss")} |\n'
        text+='\n'
    text+=f'''## Complete source learning and exposure

All initial probabilities are retained and independently recomputed. The figures show every rate/scaling trajectory, including unselected outcomes. Initial points are diagnostics and cannot be selected. Mean minibatch loss, last preclip head/encoder/global gradient norms, rates and aggregate trial/window exposure remain in case records/history. Sparse checkpoints, higher training accuracy or passing a memorization check do not establish convergence.

![All pretrained head trajectories]({LINK}/postfit_analysis/head_pretrained_learning.png)

![All random head trajectories]({LINK}/postfit_analysis/head_random_learning.png)

![All longer neural trajectories]({LINK}/postfit_analysis/long_source_learning.png)

The [complete exposure table]({LINK}/postfit_analysis/training_exposure.csv) has 224 independently reconstructed points from declared seeds and ordered training metadata. By update 1,200, every original training observation is drawn: all 358/344 DEAP and 108/108 SEED-IV trials. Distinct observed trial/window pairs are 1,418/1,432 and 1,368/1,376 on DEAP, and all 432/432 on both SEED-IV panels. The schedule therefore reaches almost all available ten-second windows, without establishing that every observation is adequately optimized.

All eight final fine-tuning gradient snapshots have finite, nonzero head/encoder norms. Their final preclip global norms range from 3.21 to 118.01, exceeding the declared clip threshold of 1; clipping acts on those recorded updates. These sparse snapshots neither estimate the frequency of clipping nor prove it causes limited learning. Frozen/updated module state digests and independent probability replay check actual parameter changes. The exposure reconstruction is not an independent replay of every optimizer update. Each capacity condition uses only twelve training observations; these thresholds do not certify generalization on the full source population.

## Verification and publication

Twenty relevant tests pass. All 192 cached-head states independently check their training-only scaler and NumPy coefficient links; all 112 complete neural states strictly restore and probabilities replay at a different inference batch size, including initial states. Frozen/updated module digests, source-only membership/permutation/windows, paired initialization/draws and all 32 source selections verify. No original code/input drift or outcome-dependent budget change is permitted.

Across all 32 GPU trajectories, the maximum recorded CUDA tensor allocation is {peak:,} bytes ({peak/1024**3:.3f} GiB), within the RTX 3050's 6 GB. This excludes driver/other-process memory. Independent state replay has maximum probability difference {fitting['maximum_probability_abs']:.3g} and maximum metric difference {fitting['maximum_metric_abs']:.3g}. Public probability reanalysis has maximum metric discrepancy {proof['maximum_abs_metric_discrepancy']:.3g}; it needs no private EEG or checkpoints.

The [public reanalysis proof]({LINK}/postfit_analysis/verification.json) independently checks all 816 new and 288 prior probability metric sets, all sixteen capacity outcomes, 32 source selections and 448 descriptive balanced-loss/accuracy contrast points. It checks matched metadata/source-role boundaries, tiny training-only labels/windows, paired state certificates, 224 exposure points and canonical artifact hashes. This introduces no fit, test access, population uncertainty or global method selection. The raw EEG, embeddings, coefficients/checkpoints and per-trial amplitude values remain private; verified probabilities/labels/anonymous IDs are published under the author's existing explicit approval. Permuted targets are clearly marked, and original task labels remain available in tiny definitions. All four scientific figures were visually inspected; PNG/SVG copies are retained.

Public-only numerical verification needs no GPU or raw EEG:

```powershell
conda activate pytorch
python scripts/analyze_cbramod_learning.py verify
python scripts/verify_publication_export.py --export-only
```

Run from `GRU-XNet_EEG_Emotion_Recognition/`. The completed local worker can be checked with `python scripts/cbramod_learning.py run`: verified completed artifacts are checked without overwriting records. Full state/raw-input replay additionally requires the retained local datasets/assets/states. [Fitting proof]({LINK}/verification.json), [all candidates/selections]({LINK}/summary.json), [all probability metrics]({LINK}/postfit_analysis/all_metrics.csv), [all descriptive contrasts]({LINK}/postfit_analysis/contrasts.json), [complete progress]({LINK}/progress.json), [canonical export]({LINK}/export_manifest.json).

## Scientific decision gate

This diagnoses source learning, not a new method or a confirmed architecture ranking. Existing-encoder scaling, optimization and longer training alone do not establish novelty. The same people/videos remain reused development cohorts, source validation is selected repeatedly, one initialization per encoder/head is used, and tiny/long conditions differ in several regularization factors. No joint training or unseen-corpus experiment is made here; weak generalization does not prove absence of EEG information or negative transfer.

Any larger encoder programme still needs a robust matched lead, a specific gap in prior work, independently declared fold-local controls and broader initialization/participant/material confirmation. Physical calibration, first-party DEAP signals, actual checkpoint training membership, historical synthetic lineage/results and missing manuscript figures remain unresolved. Before adopting a different main question, show measured evidence and a concrete proposal, obtain explicit author approval and immediately archive the then-current manuscript. The current manuscript and question are unchanged.
'''
    target=REPO.parent/'GRU-XNet_CBraMod_Learning_Findings_2026-10-09.md'
    if target.exists():raise FileExistsError('Preserve source learning report')
    target.write_text(text,encoding='utf-8')
    print(json.dumps({'report':target.name,'capacity_outcomes':16,'selected_table_rows':16,'source_selected_conditions':32}))


if __name__=='__main__':main()
