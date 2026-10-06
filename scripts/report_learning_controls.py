"""Report the completed, replayed source learning and normalization studies."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent/'publication_runs'
sys.path.insert(0, str(REPO))
from gruxnet.data import write_json

NAMES = {'gru': 'GRU-XNet', 'lstm': 'Matched BiLSTM', 'cbsatt_local': 'Local CBSAtt', 'eegnet': 'Author-checked EEGNet'}


def report():
    output = ROOT/'learning_controls_2026-10-06'; bn = ROOT/'source_bn_diagnostic_2026-10-06'
    tiny = ROOT/'memo_bn_check_2026-10-06'
    verified = json.loads((output/'verification.json').read_text())
    normalization = json.loads((bn/'verification.json').read_text())
    tiny_proof = json.loads((tiny/'verification.json').read_text())
    if not verified['passed'] or not verified['complete'] or not normalization['passed'] or not tiny_proof['passed']:
        raise ValueError('Complete original and normalization replay required')
    records = json.loads((output/'records.json').read_text())
    if len(records) != 96: raise ValueError('All declared fits required')
    differences = json.loads((bn/'comparison.json').read_text())['cases']
    tiny_result = json.loads((tiny/'comparison.json').read_text())
    rows = []; memorization = []; curve_rows = []
    for r in records:
        job = r['job']; dataset = job['dataset']; task = 'coarse3' if dataset == 'SEEDIV' else 'binary'
        item = {'id': r['id'], **job, 'selected_step': r['selected_step'], 'parameters': r['parameters'],
                'elapsed_seconds': r['elapsed_seconds'], 'peak_allocated_cuda_bytes': r['peak_allocated_cuda_bytes']}
        for step, point in r['checkpoints'].items():
            for part in ('training', 'validation'):
                if point[part] is None: continue
                for field, short in (('balanced_accuracy', 'BA'), ('balanced_log_loss', 'logloss')):
                    item[f'{part}_{short}_{step}'] = point[part][task][field]
        rows.append(item)
        if job['kind'] == 'memorization':
            point = r['checkpoints']['400']['training'][task]
            memorization.append(f'| {dataset} | {NAMES[job["model"]]} | {job["label_mode"]} | {point["balanced_accuracy"]*100:.2f} | {point["balanced_log_loss"]:.4g} | {"yes" if r["memorization_criterion_met"] else "no"} |')
        else:
            history = json.loads((output/'fits'/r['id']/'history.json').read_text())
            for h in history:
                curve_rows.append({**job, 'step': h['step'],
                                   'training_BA': h['training'][task]['balanced_accuracy'],
                                   'validation_BA': h['validation'][task]['balanced_accuracy'],
                                   'training_logloss': h['training'][task]['balanced_log_loss'],
                                   'validation_logloss': h['validation'][task]['balanced_log_loss']})
    df = pd.DataFrame(rows); df.to_csv(output/'summary.csv', index=False)
    curves = pd.DataFrame(curve_rows); main = df[df.kind.eq('source_curve')].copy()
    means = []; selected_rows = []
    for (dataset, model, lr), part in main.groupby(['dataset', 'model', 'lr'], sort=False):
        fields = [part[f'{p}_BA_{s}'].mean()*100 for p in ('training', 'validation') for s in (200, 600, 1200)]
        loss = [part[f'validation_logloss_{s}'].mean() for s in (200, 1200)]
        means.append(f'| {dataset} | {NAMES[model]} | {lr:g} | {len(part)} | '+' | '.join(f'{v:.2f}' for v in fields+loss)+' |')
        selected_rows.append(f'| {dataset} | {NAMES[model]} | {lr:g} | {part.selected_step.median():.0f} | {int(part.selected_step.gt(200).sum())}/{len(part)} | {int(part.selected_step.eq(1200).sum())}/{len(part)} |')
    for dataset in ('SEEDIV', 'DEAP'):
        fig, axes = plt.subplots(4, 2, figsize=(12, 12.5), sharex=True, sharey=True)
        for row, model in enumerate(NAMES):
            for col, lr in enumerate((.001, .0003)):
                ax = axes[row, col]
                subset = curves[curves.dataset.eq(dataset) & curves.model.eq(model) & curves.lr.eq(lr)]
                for arm, style in (('exposed', '-'), ('unexposed', '--')):
                    data = subset[subset.arm.eq(arm)]
                    for part, color in (('training', '#3575a9'), ('validation', '#d27e23')):
                        table = data.groupby('step')[part+'_BA'].agg(['mean', 'min', 'max'])*100
                        ax.plot(table.index, table['mean'], style, color=color, linewidth=1.6)
                        ax.fill_between(table.index, table['min'], table['max'], color=color, alpha=.07)
                ax.set_title(f'{NAMES[model]}, learning rate {lr:g}', fontsize=10)
                ax.set_ylim(0, 103); ax.grid(alpha=.2); ax.axvline(200, color='gray', linewidth=.7, alpha=.6)
                if col == 0: ax.set_ylabel('Balanced accuracy (%)')
                if row == 3: ax.set_xlabel('Optimizer updates')
        label = 'SEED-IV (three classes, session 1)' if dataset == 'SEEDIV' else 'DEAP (individual binary valence)'
        fig.suptitle(f'{label}: fixed source training/validation panels', y=.995)
        handles = [plt.Line2D([0], [0], color=color, linestyle=style, label=f'{role}, {arm} arm')
                   for role, color in (('Training', '#3575a9'), ('Validation', '#d27e23'))
                   for arm, style in (('exposed', '-'), ('unexposed', '--'))]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .975), ncol=2, frameon=False)
        fig.text(.5, .008, 'Lines: panel means; shading: panel range, without population uncertainty. Validation videos are unseen in both arms.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .92)); fig.savefig(output/f'learning_curves_{dataset.lower()}.png', dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
        bdf = pd.DataFrame(differences)
        for ax, role in zip(axes, ('train', 'validation')):
            for model, color in zip(NAMES, ('#3575a9', '#d27e23', '#578b50', '#8c64a6')):
                part = bdf[bdf.dataset.eq(dataset) & bdf.model.eq(model) & bdf.part.eq(role)]
                ax.scatter(part.before_BA*100, part.after_BA*100, s=25, alpha=.6, color=color, label=NAMES[model])
            ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=1)
            ax.set_title('Source training' if role == 'train' else 'Source validation'); ax.grid(alpha=.2)
            ax.set_xlabel('Original BatchNorm BA (%)'); ax.set_xlim(-2, 103); ax.set_ylim(-2, 103)
        axes[0].set_ylabel('Source population BatchNorm BA (%)')
        fig.suptitle(f'{label}: post-hoc normalization diagnostic; weights unchanged', y=.99)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
        fig.tight_layout(rect=(0, .13, 1, .94)); fig.savefig(bn/f'calibration_{dataset.lower()}.png', dpi=150); plt.close(fig)
    bdf = pd.DataFrame(differences); bn_rows = []
    for (dataset, model, lr, stage, part), chunk in bdf.groupby(['dataset', 'model', 'lr', 'stage', 'part'], sort=False):
        bn_rows.append(f'| {dataset} | {NAMES[model]} | {lr:g} | {stage} | {part} | {len(chunk)} | {chunk.BA_difference.mean()*100:+.2f} | {chunk.logloss_difference.mean():+.3f} |')
    repeat_rows = []
    for (dataset, model, group, arm, lr), chunk in main[main.model.isin(['gru', 'eegnet'])].groupby(['dataset', 'model', 'group', 'arm', 'lr'], sort=False):
        values = chunk.set_index('initialization')
        repeat_rows.append(f'| {dataset} | {NAMES[model]} | {group} | {arm} | {lr:g} | {values.at[42,"validation_BA_1200"]*100:.2f} | {values.at[91,"validation_BA_1200"]*100:.2f} |')
    tiny_rows = []
    for case in tiny_result['cases']:
        tiny_rows.append(f'| {case["dataset"]} | {NAMES[case["model"]]} | {case["label_mode"]} | {case["before_BA"]*100:.2f} | {case["after_BA"]*100:.2f} | {case["before_logloss"]:.4g} | {case["after_logloss"]:.4g} | {"yes" if case["criterion_after"] else "no"} |')
    source = {dataset: {} for dataset in ('SEEDIV', 'DEAP')}
    for (dataset, model, lr), part in main.groupby(['dataset', 'model', 'lr'], sort=False):
        source[dataset].setdefault(model, {})[str(lr)] = {
            'source_panels': len(part), 'mean_train_BA_200': part.training_BA_200.mean(),
            'mean_train_BA_600': part.training_BA_600.mean(),
            'mean_train_BA_1200': part.training_BA_1200.mean(), 'mean_val_BA_200': part.validation_BA_200.mean(),
            'mean_val_BA_600': part.validation_BA_600.mean(),
            'mean_val_BA_1200': part.validation_BA_1200.mean(), 'mean_val_logloss_200': part.validation_logloss_200.mean(),
            'mean_val_logloss_600': part.validation_logloss_600.mean(),
            'mean_val_logloss_1200': part.validation_logloss_1200.mean(),
            'selected_after_200': int(part.selected_step.gt(200).sum())}
    result = {'development_only': True, 'source_only': True, 'research_question_change_approved': False,
              'fits': len(records), 'memorization_criterion_passes': sum(r['memorization_criterion_met'] is True for r in records),
              'means': source, 'normalization_cases': 160, 'tiny_normalization_cases': 16,
              'recalibrated_memorization_criterion_passes': tiny_result['criterion_after'],
              'inference': 'Descriptive source diagnostics; incomplete validation panels, reused cohorts and unequal representation/parameter counts. No held-out test, convergence or population-generalization claim.'}
    write_json(output/'comparison.json', result)
    text = '''# Source learning, baseline authentication and normalization findings — 6 October 2026

All **96 declared fits** are complete: sixteen tiny-batch memorization checks and eighty source-training/validation trajectories through 1,200 updates. GRU and author-checked EEGNet include both declared groupings and initializations 42/91. Twelve older full-model 200-update prefixes reproduce their exact selected state and complete source-validation history. Final and selected checkpoints are independently replayed. Separately declared **post-hoc** BatchNorm diagnostics cover all 160 selected/final source states and all sixteen tiny-batch final states; every learned parameter is unchanged and every source moment/probability is independently replayed. The scientific-control suite has **71 passing tests**.

These are source-only development diagnostics. They **do not evaluate outer-test performance** and do not change the manuscript or main research question. SEED-IV uses session 1 / rotation 0 / fold 0, with 108 source-training and 12 validation trials. DEAP uses rotation 0 / fold 0, with 358 (group 1) or 344 (group 2) training and 32 validation trials. Groupings can change validation people/videos; initializations are compared on the same population within each grouping/arm. These are repeated source panels, not new participants or complete OOF confirmation. Previously inspected cohorts remain development cohorts.

## Fixed-budget source results

All learning rates are retained. BA is a percentage; balanced log loss is lower-is-better. Each entry is an unweighted descriptive panel mean. GRU/EEGNet have eight panels per rate, the local references two; widths, parameter counts and raw versus STFT representations differ. Scores do not form an architecture leaderboard. The 200/600/1,200 points come from the same uninterrupted trajectory, not separate restarts.

| Dataset | Model | LR | Panels | Train BA 200 | Train BA 600 | Train BA 1,200 | Val BA 200 | Val BA 600 | Val BA 1,200 | Val loss 200 | Val loss 1,200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(means)+'''

The validation panels are small, and expanding the checkpoint search increases selection opportunity. A larger source-selected validation score is not evidence of better test generalization. Fixed-step validation loss and accuracy are therefore retained alongside selection positions. A checkpoint chosen early is not proof of optimization convergence; a final checkpoint is not proof that a longer budget would help.

| Dataset | Model | LR | Median selected update | Selected after 200 | Selected at 1,200 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(selected_rows)+'''

## Can the implementation memorize training trials?

Twelve distinct source training trials, class-balanced, are fit for 400 updates with the declared dropout and constraints. The shuffled targets preserve class counts. The descriptive criterion requires BA at least 95% **and** balanced log loss below 0.15; failure can reflect limited confidence even when classification is perfect. All outcomes are retained. Success establishes capacity on a tiny batch, not physiological emotion information or generalization.

| Dataset | Model | Targets | Train BA (%) | Train balanced loss | Strict criterion met |
|---|---|---|---:|---:|---|
'''+ '\n'.join(memorization)+'''

## Balanced tiny-batch normalization check

The EEGNet SEED tiny-batch sampled training-mode losses reached 0.0031/0.0108 while ordinary inference losses were 6.97/3.60. A separate supplement was declared after fifty source fits had completed, explicitly retaining this post-hoc observation. It uses the same twelve class-balanced trials, fixed labels, input scaling and final learned weights as each original tiny fit. Three-pass source moments are calculated with dropout off. Unlike full-population recalibration below, the tiny-batch check does not change the class prior; it still cannot separate moving-average lag from dropout/inference activation-distribution differences. Every architecture, corpus and real/shuffled target case is retained.

| Dataset | Model | Targets | Original BA (%) | Recalibrated BA (%) | Original loss | Recalibrated loss | Recalibrated strict criterion |
|---|---|---|---:|---:|---:|---:|---|
'''+ '\n'.join(tiny_rows)+'''

## Running normalization statistics versus learned weights

This supplement was declared after nine completed source fits were available and some source curves had been inspected; it is explicitly post-hoc. It changes only BatchNorm running means/variances. Dropout is off, all actual source training trials receive equal weight, and moment calculation uses float64 sums/squares in three feed-forward passes. It differs from balanced mini-batch statistics used during optimization. Every learned parameter, original normalizer and originally selected optimizer checkpoint is retained. Validation/test inputs do not enter moments and no new checkpoint is selected after recalibration.

Mean after-minus-before changes are descriptive. Positive BA changes are percentage points; negative loss changes favor recalibration. Both original selected and final states and every model/rate/arm/group/initialization are preserved; a helpful training change need not improve validation. The diagnostic does not make population-generalization or BN-method-novelty claims.

| Dataset | Model | LR | State | Source role | Cases | BA change (pp) | Loss change |
|---|---|---:|---|---|---:|---:|---:|
'''+ '\n'.join(bn_rows)+'''

## Initialization and grouping sensitivity

Final-update source-validation BA. Both initializations are shown; do not select a best seed or grouping. These reused source cohorts and partial validation panels cannot establish optimizer robustness for the complete test task.

| Dataset | Model | Group | Arm | LR | Init 42 BA (%) | Init 91 BA (%) |
|---|---|---:|---|---:|---:|---:|
'''+ '\n'.join(repeat_rows)+f'''

## Verification and practical scope

The author-code EEGNet check executes the pinned original TensorFlow function and checks the PyTorch port's binary/three-class outputs, deterministic dropout-zero training, moving BatchNorm states, max-norm constraints and parameter counts. Dropout 0.5 is retained during real fitting. EEGNet has 6,450 binary or 9,011 three-class trainable parameters. Inputs and balanced AdamW optimization differ from the authors' original applications; published scores and training trajectories are not reproduced. The local CBSAtt reference remains unauthenticated author code.

Independent source replay checks all source/data/config/split/scaler/draw/initial/checkpoint bindings, the final and source-selected predictions and selection rule, and source/test disjointness. It checks {verified['probability_metric_sets_checked']} exported probability metric sets and performs {verified['checkpoint_states_replayed']} final/selected state replays, including repeats when states coincide. The largest neural probability error is {verified['maximum_neural_probability_error']:.3g}. All twelve old prefixes pass without reading previous test probabilities. The post-hoc normalization replay reconstructs all float32 moment arrays exactly and reaches maximum probability error {normalization['maximum_probability_error']:.3g}. Replay is not a rerun of all optimizer updates or authentication of first-party EEG.

The sixteen tiny-batch recalibrations also reconstruct source moments exactly and replay original/after probabilities with maximum error {tiny_proof['maximum_probability_error']:.3g}; the strict capacity criterion passes {tiny_result['criterion_before']}/16 original states and {tiny_result['criterion_after']}/16 recalibrated states. Public probability checks independently recompute all 640 full-population before/after metric sets and all 32 tiny-batch metric sets, with no raw EEG or checkpoint access.

Recorded original fitting time is {df.elapsed_seconds.sum()/60:.2f} minutes; maximum PyTorch-allocated CUDA memory is {df.peak_allocated_cuda_bytes.max()/2**20:.2f} MiB. These exclude preparation, driver/background allocations, software tests and independent replay/normalization diagnostics. Checkpoints, waveform caches, isolated compatibility dependencies and downloaded third-party source remain local; bounded probabilities, source records and charts are exported.

## Publication consequence and next valid experiments

This phase distinguishes limited early learning, later training fit, stochastic/normalization sensitivity and source-validation behavior; it does not demonstrate a publishable model advantage or useful incremental EEG prediction on an independently evaluated test population. Simply increasing the budget or selecting the most favorable source panel is insufficient. No global recipe is selected from these partial panels.

The next broad controls require a new declaration with per-outer-fold source validation for learning rate, training duration and any source-only normalization variant, preserving every comparison and independent initialization/grouping repeats. EEGNet plus a participant-excluded video prior should then be compared with both context-only controls. Full joint all-corpus/LODO studies, historical provenance and first-party signal authentication still remain. Our [latest prior-work audit](GRU-XNet_Learning_Control_Research_Update_2026-10-06.md) includes direct familiar-video prior/physiology work published as a public preprint on 2 October; the general contextual-prior idea alone is not an established new contribution.

Changing the paper's main question requires a concrete evidence-backed proposal, author approval and an immediate fresh archive of the then-current paper. No pivot has been adopted.

## Evidence and reproduction

- [Source protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/Learning_Control_Protocol_2026-10-06.md), [plan](publication_runs/learning_controls_2026-10-06/plan.json), [complete source replay](publication_runs/learning_controls_2026-10-06/verification.json), [public probability check](publication_runs/learning_controls_2026-10-06/public_verification.json)
- [Complete source comparison](publication_runs/learning_controls_2026-10-06/comparison.json), [per-fit summary](publication_runs/learning_controls_2026-10-06/summary.csv), [SEED-IV curves](publication_runs/learning_controls_2026-10-06/learning_curves_seediv.png), [DEAP curves](publication_runs/learning_controls_2026-10-06/learning_curves_deap.png)
- [Post-hoc normalization protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/Source_BN_Diagnostic_Protocol_2026-10-06.md), [declaration](publication_runs/source_bn_diagnostic_2026-10-06/plan.json), [replay](publication_runs/source_bn_diagnostic_2026-10-06/verification.json), [all differences](publication_runs/source_bn_diagnostic_2026-10-06/comparison.json)
- [SEED-IV normalization plot](publication_runs/source_bn_diagnostic_2026-10-06/calibration_seediv.png), [DEAP normalization plot](publication_runs/source_bn_diagnostic_2026-10-06/calibration_deap.png)
- [Tiny-batch post-hoc declaration](publication_runs/memo_bn_check_2026-10-06/plan.json), [local replay](publication_runs/memo_bn_check_2026-10-06/verification.json), [all tiny contrasts](publication_runs/memo_bn_check_2026-10-06/comparison.json), [public check](publication_runs/memo_bn_check_2026-10-06/public_verification.json)

On a fresh local study with the previous verified caches and author audit, declare with `python scripts/learning_controls.py plan` before `run`. After completion run `python scripts/audit_learning_controls.py`, then the separately declared `source_bn_diagnostic.py run`/`verify` and `memo_bn_check.py run`/`verify`. Run `python scripts/verify_source_bn_export.py --root ../publication_runs` and `python scripts/verify_memo_bn_export.py --root ../publication_runs` before `python scripts/report_learning_controls.py`, then `python scripts/verify_learning_report.py --root ../publication_runs`. Preserve the post-hoc timing in any reproduction. A public checkout can use `python scripts/audit_learning_controls.py --export-only`, `python scripts/verify_source_bn_export.py`, `python scripts/verify_memo_bn_export.py` and `python scripts/verify_learning_report.py` without raw EEG or neural checkpoints. On Windows, Git may require repository-local `core.longpaths=true` for this existing long workspace path.
'''
    (REPO.parent/'GRU-XNet_Learning_Control_Findings_2026-10-06.md').write_text(text, encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': report()
