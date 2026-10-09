"""Render every audited conditioning/dropout endpoint and matched effect."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_conditioning import verify, STUDY, ROLES
BASE = REPO/'results/development'/STUDY
LINK = f'GRU-XNet_EEG_Emotion_Recognition/results/development/{STUDY}'


def pair(frame, column, percent=False):
    part = frame.sort_values('group')
    if part.group.tolist() != [1, 2]:
        raise ValueError('Missing or duplicate report grouping')
    return ' / '.join(f'{v*100:.2f}' if percent else f'{v:.4f}' for v in part[column])


def label(pretrained, trainable, scaled, dropout):
    return ('Pretrained' if pretrained else 'Random42')+(' fine-tuned' if trainable else ' frozen')+ \
        ('; standardized' if scaled else '; raw')+('; dropout on' if dropout else '; dropout off')


def render():
    verify(BASE)
    analysis = BASE/'postfit_analysis'
    rows = pd.read_csv(analysis/'all_metrics.csv'); selected = pd.read_csv(analysis/'selected_metrics.csv')
    history = pd.read_csv(analysis/'gradient_history.csv'); norms = pd.read_csv(analysis/'normalizer_diagnostics.csv')
    points = pd.DataFrame(json.loads((analysis/'contrasts.json').read_text())['contrasts'])
    fitting = json.loads((BASE/'verification.json').read_text())
    proof = json.loads((analysis/'verification.json').read_text())
    plan = json.loads((BASE/'plan.json').read_text())
    records = [json.loads(p.read_text()) for p in (BASE/'runs').glob('*/record.json')]
    if len(records) != 64 or len(rows) != 768 or len(selected) != 192 or len(history) != 768:
        raise ValueError('Incomplete report coverage')
    peak = max(r['peak_allocated_bytes'] for r in records)
    selections = selected[selected.role == 'train']
    early = int((selections.step == 200).sum())
    text = f'''# Matched CBraMod embedding normalization and encoder dropout

Completed 9 October 2026. **All 64 conditions, 256 full states and 768 probability metric sets verify.** Sixteen raw/dropout-on controls are reused exactly; 48 new trajectories complete all 1,200 updates. Eight initial-encoder normalizers use only original training windows and remain fixed throughout fine-tuning. No outer-test inference, manuscript change or research-question pivot is made.

**Decision: the grid provides a source-fitting result, but no robust generalization lead that warrants a larger encoder study.** With pretrained fine-tuning, training-only standardization plus encoder dropout off raises final training BA from 57.58/58.02% to 93.24/91.34% on DEAP and from 69.14/67.28% to 100/100% on SEED-IV. Final unseen-validation loss is nevertheless 0.8519/0.9141 and 1.3400/1.3366, respectively, worse than uniform predictions on all four panels. These are grouping 1/2 results, not independent confirmation. The [contribution decision and bounded proposal](GRU-XNet_Contribution_Decision_2026-10-09.md) explain why the next step is adapter/target reassessment rather than another broad grid.

The [protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/CBraMod_Conditioning_Protocol_2026-10-09.md) was frozen and pushed before task feature preparation/fitting in commit `0fd236f7f`. All {len(plan['source_sha256'])} bound source/test files and predecessor bindings remain exact. Pretrained/random42 × frozen/trainable × raw/standardized × encoder dropout on/off are crossed in DEAP/SEED-IV groupings 1 and 2. Native 32/62-channel prepared 200-Hz forty-second inputs and four disjoint ten-second windows stay fixed. DEAP uses corrected individual binary valence; SEED-IV uses the existing assigned coarse three-class target, retaining original labels. Input calibration, preprocessing, original labels and exclusion rules are unchanged.

All conditions preserve head/random/dropout initialization seeds and exact balanced six-observation/window streams, pooling, optimizer, label smoothing, decay, global clipping and 1,200-update cosine schedule. Dropout-off disables all 61 module and internal-attention sites; it deliberately changes mask consumption while preserving observation/window streams. Raw bypasses affine arithmetic. Standardized embeddings use FP64 training-window moments and population standard deviations, floor 1e-6, then fixed FP32 buffers. They are never fitted to validation/current-encoder features. The head is a 200-dimensional pooled linear classifier, not the published default multilayer flattened fine-tuning head.

**Primary effects hold update duration fixed at 200/600/1,200.** Balanced log loss is primary and balanced accuracy secondary. Independently selected checkpoints are secondary because durations may differ. Initial probabilities are retained as diagnostics. All {proof['fixed_step_contrast_points']} fixed-step and {proof['selected_secondary_points']} selected contrast points are available, including {proof['primary_positive_validation_points']} positive-step validation points. Every grouping and condition survives. Binary/three-class losses are not pooled into a common-task score; no global winning recipe or population interval is selected.

At the final matched step, normalization with dropout on worsens pretrained fine-tuned unseen loss in all four panels; normalization with dropout off worsens it in three. Disabling dropout on raw embeddings improves it in one of four panels. Disabling dropout on standardized embeddings improves both SEED-IV panels but worsens both DEAP panels. Frozen random heads have smaller unseen-loss improvements from disabling dropout in all four panels at either scaling, but their final losses still exceed uniform predictions. Thus the complete grid contains improvements without a convincing cross-panel predictive lead. Frequent clipping accompanies several fits; it is not an established explanation of their generalization failure.

Secondary checkpoint selection does not rescue the combined standardized/dropout-off pretrained recipe: unseen losses are 0.8228/0.7392 on DEAP and 1.1892/1.3366 on SEED-IV, still above uniform. An isolated accuracy improvement cannot override the declared primary loss. The full tables below preserve such outcomes, the earlier matched checkpoints and every random/frozen control.

## Complete fixed-update endpoints

Every slash separates grouping 1/grouping 2. BA is balanced accuracy in percent. Familiar and unseen validation exclude training participants; unseen validation also excludes training stimulus materials. The uniform reference is BA 50%/loss ln(2)=0.6931 for DEAP and BA 33.33%/loss ln(3)=1.0986 for SEED-IV. These are repeatedly reused development panels, not untouched test scores. Tables include every condition at update 1,200; complete 200/600/initial values remain in [all metrics]({LINK}/postfit_analysis/all_metrics.csv).

'''
    for dataset in ('DEAP', 'SEEDIV'):
        text += f'### {dataset}: fixed update 1,200\n\n| Condition | Train BA (%) | Familiar BA (%) | Unseen BA (%) | Familiar loss | Unseen loss |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
        for p in (True, False):
            for t in (False, True):
                for s in (False, True):
                    for d in (True, False):
                        case = rows[(rows.dataset == dataset)&(rows.pretrained == p)&(rows.trainable == t)&(rows.scaled == s)&(rows.dropout == d)&(rows.step == 1200)]
                        train, unseen, familiar = [case[case.role == role] for role in ROLES]
                        text += '| '+label(p, t, s, d)+' | '+pair(train, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_accuracy', True)+' | '+pair(unseen, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_log_loss')+' | '+pair(unseen, 'balanced_log_loss')+' |\n'
        text += '\n'
    text += '## Matched final-step factor effects\n\nEach entry is changed minus reference at the same update 1,200. Negative loss difference favors the changed condition; positive BA difference favors it. Normalization means standardized minus raw at the stated dropout. Dropout-off means off minus on at the stated scaling. Interaction is the normalization effect with dropout off minus its effect with dropout on; its sign alone is not an overall performance benefit. Earlier matched checkpoints and all initial/training contrasts are retained in [complete contrasts]('+LINK+'/postfit_analysis/contrasts.json).\n\n'
    names = {'normalization_dropout_on': 'Normalization, dropout on', 'normalization_dropout_off': 'Normalization, dropout off',
             'dropout_off_raw': 'Dropout off, raw', 'dropout_off_standardized': 'Dropout off, standardized', 'interaction': 'Interaction'}
    for dataset in ('DEAP', 'SEEDIV'):
        text += f'### {dataset}: all factor effects\n\n| Encoder | Contrast | Train loss delta | Familiar loss delta | Unseen loss delta | Unseen BA delta (pp) |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
        for p in (True, False):
            for t in (False, True):
                for name, title in names.items():
                    case = points[(points.scope == 'fixed_step')&(points.step == 1200)&(points.dataset == dataset)&(points.pretrained == p)&(points.trainable == t)&(points.contrast == name)]
                    losses = case[case.metric == 'balanced_log_loss']
                    train, unseen, familiar = [losses[losses.role == role] for role in ROLES]
                    accuracy = case[(case.metric == 'balanced_accuracy')&(case.role == ROLES[1])]
                    text += '| '+('Pretrained' if p else 'Random42')+(' fine-tuned' if t else ' frozen')+' | '+title+' | '+pair(train, 'delta')+' | '+pair(familiar, 'delta')+' | '+pair(unseen, 'delta')+' | '+pair(accuracy, 'delta', True)+' |\n'
        text += '\n'
    text += f'## Checkpoint-selected outcomes: secondary\n\nEach trajectory selects among 200/600/1,200 by equal familiar/unseen balanced loss, then mean BA and stable first candidate. All 64 selections independently verify; {early}/64 select 200. Differing durations make these secondary comparisons. A selected-validation score is not an independent test gain.\n\n'
    for dataset in ('DEAP', 'SEEDIV'):
        text += f'### {dataset}: all selected conditions\n\n| Condition | Selected updates | Train BA (%) | Familiar BA (%) | Unseen BA (%) | Familiar loss | Unseen loss |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
        for p in (True, False):
            for t in (False, True):
                for s in (False, True):
                    for d in (True, False):
                        case = selected[(selected.dataset == dataset)&(selected.pretrained == p)&(selected.trainable == t)&(selected.scaled == s)&(selected.dropout == d)]
                        train, unseen, familiar = [case[case.role == role] for role in ROLES]
                        steps = ' / '.join(str(v) for v in train.sort_values('group').step)
                        text += '| '+label(p, t, s, d)+' | '+steps+' | '+pair(train, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_accuracy', True)+' | '+pair(unseen, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_log_loss')+' | '+pair(unseen, 'balanced_log_loss')+' |\n'
        text += '\n'
    text += f'''## Source-only statistics, clipping and numerical checks

The eight source statistic sets independently replay the canonical-batch features exactly and reproduce FP64 moments with maximum discrepancy {norms.independent_scaler_max_abs.max():.2e}. The alternate batch-8 training-feature diagnostic has maximum raw discrepancy {norms.batch8_feature_max_abs.max():.2e}, and standardized discrepancy {norms.batch8_normalized_feature_max_abs.max():.2e}. This is a numerical diagnostic, not whole-model batch invariance. Complete-state replay uses canonical batch 16, FP32 and the explicit backbone/token-mean/fixed-affine/functional-linear operator order. Maximum full-state probability discrepancy is {fitting['maximum_probability_abs']:.2e}; maximum replay metric discrepancy is {fitting['maximum_metric_abs']:.2e}. Public metric recomputation differs by at most {proof['maximum_abs_metric_discrepancy']:.2e}.

New histories retain clipping counts across every 100-update interval. Reused controls lack those counts and are excluded from frequency estimates. The following frequencies are actual preclip norms exceeding 1 among all 1,200 updates of each new trajectory; gradient clipping is an observation, not proof of the cause of weak generalization.

| Dataset | New condition | Updates clipped (%), groups 1 / 2 |
| --- | --- | ---: |
'''
    new = history[~history.reused].groupby(['dataset', 'group', 'pretrained', 'trainable', 'scaled', 'dropout'], as_index=False).clipped_updates_last100.sum()
    new['fraction'] = new.clipped_updates_last100/1200
    for dataset in ('DEAP', 'SEEDIV'):
        for p in (True, False):
            for t in (False, True):
                for s in (False, True):
                    for d in (True, False):
                        if not s and d:
                            continue
                        case = new[(new.dataset == dataset)&(new.pretrained == p)&(new.trainable == t)&(new.scaled == s)&(new.dropout == d)]
                        text += '| '+dataset+' | '+label(p, t, s, d)+' | '+pair(case, 'fraction', True)+' |\n'
    text += f'''\nThe public audit checks all 768 metric sets, 64 selections, 192 reconstructed exposure points, 768 history points, eight normalizer proofs, sixteen byte-exact anchors, paired initializations/streams and four participant/trial/material role boundaries. Peak actual tensor allocation is {peak/(1024**3):.3f} GiB, excluding driver/desktop memory. Thirty fitting-related tests and two independent interaction-analysis checks pass.

## Figures and reproducibility

The five original PNG/SVG pairs in `postfit_analysis/` remain exact. A separately bound presentation revision in `linear_figures/` uses linear axes with readable numeric ticks, reproduces all 768 loss coordinates and 320 final validation-effect coordinates, and changes no numerical outcome. All five revised PNGs were visually inspected, including legends and axis labels; standalone SVGs are retained.

'''
    for name in ('pretrained_frozen_learning', 'pretrained_finetune_learning', 'random_frozen_learning', 'random_finetune_learning', 'matched_final_step_effects'):
        text += '!['+name.replace('_', ' ')+']('+LINK+'/linear_figures/'+name+'.png)\n\n'
    text += f'''Public-only numerical verification needs no GPU/raw EEG:

```powershell
conda activate pytorch
python scripts/analyze_cbramod_conditioning.py verify
python scripts/plot_cbramod_conditioning.py verify
python scripts/verify_publication_export.py --export-only
```

Run from `GRU-XNet_EEG_Emotion_Recognition/`. Local full-state checking additionally requires retained datasets, official assets, scaler arrays and checkpoints. [Fitting proof]({LINK}/verification.json), [independent public proof]({LINK}/postfit_analysis/verification.json), [all selections]({LINK}/summary.json), [gradient histories]({LINK}/postfit_analysis/gradient_history.csv), [source exposure]({LINK}/postfit_analysis/training_exposure.csv), [normalizer diagnostics]({LINK}/postfit_analysis/normalizer_diagnostics.csv), [canonical export]({LINK}/export_manifest.json).

Verified probabilities, labels and anonymous references are released under the author's existing explicit approval. EEG, embeddings, scaler/weight arrays, coefficients and per-trial physical amplitudes remain local. State replay checks model/output integrity, not every optimizer update or external recording/pretraining authenticity.

## Interpretation and decision gate

These matched controls isolate the tested embedding conditioning and encoder dropout recipe. They do not test native four-emotion SEED-IV, complete joint training, unseen-corpus transfer or all published CBraMod adapters. One head/random initialization and repeatedly reused source panels cannot establish population confirmation. Physical input calibration, first-party DEAP signal authentication, actual checkpoint training membership and historical result/figure provenance remain unresolved.

Full-source fitting is now demonstrated under the combined standardized/dropout-off pretrained recipe, without a reliable predictive lead. This resolves the narrower capacity concern under that recipe; it does not establish convergence, adequacy of the raw/default recipe or useful emotion generalization. There is no qualifying lead for a larger confirmation programme from this grid. The [decision proposal](GRU-XNet_Contribution_Decision_2026-10-09.md) recommends a bounded authenticated-head control before any larger encoder expansion; it is not launched or adopted by this report.

A better normalization/dropout recipe for an existing model is not itself scientific novelty. The [contribution research update](GRU-XNet_Contribution_Research_Update_2026-10-09.md) records close prior work and a limited official-readout shape audit. Any actual changed main question still requires measured findings, a concrete proposal, explicit author approval and a fresh archive immediately before adoption. The manuscript and current question remain unchanged.
'''
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO.parent/'GRU-XNet_CBraMod_Conditioning_Findings_2026-10-09.md')
    args = parser.parse_args()
    args.output.write_text(render(), encoding='utf-8')
    print(json.dumps({'report': str(args.output.resolve())}))
