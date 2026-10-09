"""Render every audited endpoint, matched effect, selection and spectral reference."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_readout_boundary_adapter import api, PUBLIC
from scripts.compare_cbramod_readout_spectral import verify as verify_spectral
from scripts.plot_cbramod_readout import verify as verify_figures

LINK = f'GRU-XNet_EEG_Emotion_Recognition/results/development/{PUBLIC.name}'
HEADS = ('pooled_linear', 'pooled_mlp', 'flattened_mlp')
NAMES = {'pooled_linear': 'Pooled linear', 'pooled_mlp': 'Pooled MLP', 'flattened_mlp': 'Flattened MLP'}
ROLES = ('train', 'validation_unseen', 'validation_familiar')


def pair(frame, column, percent=False):
    part = frame.sort_values('group')
    if part.group.tolist() != [1, 2]:
        raise ValueError('Missing or duplicate report grouping')
    return ' / '.join(f'{v*100:.2f}' if percent else f'{v:.4f}' for v in part[column])


def render():
    api()['verify'](PUBLIC); verify_spectral(); verify_figures()
    base = PUBLIC/'postfit_analysis'
    rows = pd.read_csv(base/'all_metrics.csv'); selected = pd.read_csv(base/'selected_metrics.csv')
    history = pd.read_csv(base/'gradient_history.csv'); spectral = pd.read_csv(PUBLIC/'spectral_reference/spectral_metrics.csv')
    contrasts = pd.DataFrame(json.loads((base/'contrasts.json').read_text())['contrasts'])
    proof = json.loads((PUBLIC/'verification.json').read_text())
    analysis = json.loads((base/'verification.json').read_text())
    plan = json.loads((PUBLIC/'plan.json').read_text())
    records = [json.loads(p.read_text()) for p in (PUBLIC/'runs').glob('*/record.json')]
    if len(records) != 24 or len(rows) != 288 or len(selected) != 72 or len(spectral) != 24:
        raise ValueError('Incomplete report coverage')
    early = int(((selected.role == 'train')&(selected.selected_step == 200)).sum())
    peak = max(r['peak_allocated_bytes'] for r in records)/(1024**3)
    text = f'''# Matched nonlinear CBraMod readout findings

Declared 9 October 2026; completion recorded at `{proof['created_utc']}`. **All 24 conditions, 96 full states and 288 probability metric sets verify:** sixteen new 1,200-update trajectories and eight exact pooled-linear controls. Forty relevant pre-fit tests, four supplementary boundary tests and a complete synthetic public-grid audit pass. No new outer-test inference, label change, manuscript change or adopted main question is made.

The [protocol](GRU-XNet_EEG_Emotion_Recognition/docs/publication/CBraMod_Readout_Protocol_2026-10-09.md) and [declaration]({LINK}/plan.json) were pushed before task fitting in commit `6a791a4a5`. All {len(plan['source_sha256'])} frozen source/analysis/test files remain exact. The [pre-fit boundary adapter]({LINK}/analysis_boundary_adapter_declaration.json) preserves the original analyzer while correcting its overly strict validation-subject guard. Training people are excluded from both validation roles; familiar/unseen validation deliberately share held-out people with disjoint trials. All trial and material exclusions remain unchanged. The adapter was declared with zero new trajectories begun; four rejection/acceptance tests and a complete synthetic-only public-grid replay cover it.

The [interpretation and decision](GRU-XNet_Readout_Decision_2026-10-09.md) distinguish a useful predictive lead from training-only improvement. The tables below retain all conditions and primary/secondary outcomes; no global winning head or population interval is selected.

## What is matched

DEAP/SEED-IV grouping 1/2 retain native 32/62-channel prepared 200-Hz forty-second arrays divided by 100, four disjoint ten-second windows, corrected individual binary DEAP valence and assigned coarse three-class SEED-IV labels. All cases fine-tune the same pretrained/random42 encoder with encoder dropout on. Head initialization seed 4242, the exact balanced observation/window streams, AdamW rates 0.001/0.0001, decay 0.05, smoothing 0.1, clipping 1 and 1,200-update cosine schedule are fixed.

Pooled linear controls average tokens into 200 dimensions. New pooled/flattened MLPs use a 200-hidden-unit linear layer, ELU, dropout 0.1 and a two-/three-class output. Head parameter counts are 402/603 for linear, 40,602/40,803 for pooled MLP and 12,800,602/24,800,803 for flattened MLP. Head-family changes add dropout; flattening adds parameters and retains positional information. These are not pure pooling or capacity effects. New head dropout uses a separate `424243 + update` stream and restores the encoder's global RNG. Exact CPU/CUDA mask and RNG checks pass.

Pinned author two-layer operators, outputs and input/parameter gradients match in synthetic training/evaluation checks. The SEED-V standalone input adapts one to ten patches and bypasses premature wrapper flattening; the pooled control supplies mean tokens. These are explicit adapters, not reproductions of published CBraMod scores or the larger default three-layer head. Existing physical calibration, recording authentication and checkpoint membership limits remain.

## Source panel coverage

Entries show people / trials / materials. Four windows from a trial do not create four independent observations. Familiar and unseen validation share the same held-out people, with different trials/material sets. Both validation roles are repeatedly reused development data.

DEAP inherits the earlier deterministic per-training-participant/class matching between exposed and unexposed arms. Eligible training-participant labels from both arms determine retained counts; no new matching or exclusion is introduced here. The current head comparisons are conditional on that prepared population, rather than every available source trial. A strictly sealed new-material confirmation should construct its source population without consulting labels from the excluded materials, including labels from training participants.

| Dataset/group | Training | Familiar validation | Unseen validation |
| --- | ---: | ---: | ---: |
'''
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            folder = PUBLIC/'runs'/f'{dataset.lower()}_g{group}_pretrained_pooled_linear'
            cells = []
            for role in ('train', 'validation_familiar', 'validation_unseen'):
                frame = pd.read_csv(folder/f'step0_{role}.csv', usecols=['trial_id', 'subject_id', 'material_key'])
                cells.append(f'{frame.subject_id.nunique()} / {len(frame)} / {frame.material_key.nunique()}')
            text += '| '+dataset+f' / {group} | '+' | '.join(cells)+' |\n'
    text += f'''
SEED-IV unseen validation has only twelve trials from three people and four materials per grouping; DEAP has thirty-two trials from four people and eight materials. Apparent BA differences can therefore come from a few predictions. The two groupings are not independent population replications.

## Complete fixed-duration results

Slashes separate grouping 1 / grouping 2. BA is balanced accuracy in percent. Balanced log loss is primary; BA is secondary. Uniform references are loss 0.6931/BA 50% on DEAP and loss 1.0986/BA 33.33% on SEED-IV. These small repeatedly reused source-validation panels are development evidence, not independent tests. All 0/200/600/1,200 outcomes remain in [all metrics]({LINK}/postfit_analysis/all_metrics.csv).

'''
    for dataset in ('DEAP', 'SEEDIV'):
        text += f'### {dataset}: fixed update 1,200\n\n| Encoder/head | Train BA (%) | Familiar BA (%) | Unseen BA (%) | Familiar loss | Unseen loss |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
        for pretrained in (True, False):
            for head in HEADS:
                part = rows[(rows.dataset == dataset)&(rows.pretrained == pretrained)&(rows['head'] == head)&(rows.step == 1200)]
                train, unseen, familiar = [part[part.role == role] for role in ROLES]
                text += '| '+('Pretrained' if pretrained else 'Random42')+' / '+NAMES[head]+' | '+pair(train, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_accuracy', True)+' | '+pair(unseen, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_log_loss')+' | '+pair(unseen, 'balanced_log_loss')+' |\n'
        text += '\n'
    text += '## All matched head effects at update 1,200\n\nChanged minus reference: negative loss favors the changed head; positive BA difference favors it. Earlier fixed-step, initial, training and all pretraining contrasts remain in [complete contrasts]('+LINK+'/postfit_analysis/contrasts.json). No independent uncertainty is estimated from these reused panels.\n\n| Dataset/encoder | Head contrast | Train loss delta | Familiar loss delta | Unseen loss delta | Unseen BA delta (pp) |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
    for dataset in ('DEAP', 'SEEDIV'):
        for pretrained in (True, False):
            for changed, reference in (('pooled_mlp', 'pooled_linear'), ('flattened_mlp', 'pooled_linear'), ('flattened_mlp', 'pooled_mlp')):
                part = contrasts[(contrasts.kind == 'head')&(contrasts.scope == 'fixed_step')&(contrasts.step == 1200)&(contrasts.dataset == dataset)&(contrasts.pretrained == pretrained)&(contrasts.changed == changed)&(contrasts.reference == reference)]
                loss = part[part.metric == 'balanced_log_loss']
                train, unseen, familiar = [loss[loss.role == role] for role in ROLES]
                ba = part[(part.metric == 'balanced_accuracy')&(part.role == 'validation_unseen')]
                text += '| '+dataset+' / '+('pretrained' if pretrained else 'random42')+' | '+NAMES[changed]+' − '+NAMES[reference]+' | '+pair(train, 'delta')+' | '+pair(familiar, 'delta')+' | '+pair(unseen, 'delta')+' | '+pair(ba, 'delta', True)+' |\n'
    text += '\n## All matched pretrained-minus-random effects at update 1,200\n\nLoss-negative/BA-positive favors pretrained under that exact readout. A pretrained/random contrast is conditional on one random initialization and this adapter.\n\n| Dataset/head | Train loss delta | Familiar loss delta | Unseen loss delta | Unseen BA delta (pp) |\n| --- | ---: | ---: | ---: | ---: |\n'
    for dataset in ('DEAP', 'SEEDIV'):
        for head in HEADS:
            part = contrasts[(contrasts.kind == 'pretraining')&(contrasts.scope == 'fixed_step')&(contrasts.step == 1200)&(contrasts.dataset == dataset)&(contrasts['head'] == head)]
            loss = part[part.metric == 'balanced_log_loss']
            train, unseen, familiar = [loss[loss.role == role] for role in ROLES]
            ba = part[(part.metric == 'balanced_accuracy')&(part.role == 'validation_unseen')]
            text += '| '+dataset+' / '+NAMES[head]+' | '+pair(train, 'delta')+' | '+pair(familiar, 'delta')+' | '+pair(unseen, 'delta')+' | '+pair(ba, 'delta', True)+' |\n'
    text += f'\n## Checkpoint selection: secondary\n\nAll 24 selections independently verify; {early}/24 choose update 200. Each selects among 200/600/1,200 by equal familiar/unseen balanced loss, then mean BA and stable first candidate. Different selected durations make these secondary comparisons. Selected validation is not an independent test gain.\n\n| Dataset/encoder/head | Updates | Train BA (%) | Familiar BA (%) | Unseen BA (%) | Familiar loss | Unseen loss |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
    for dataset in ('DEAP', 'SEEDIV'):
        for pretrained in (True, False):
            for head in HEADS:
                part = selected[(selected.dataset == dataset)&(selected.pretrained == pretrained)&(selected['head'] == head)]
                train, unseen, familiar = [part[part.role == role] for role in ROLES]
                steps = ' / '.join(str(s) for s in train.sort_values('group').selected_step)
                text += '| '+dataset+' / '+('pretrained' if pretrained else 'random42')+' / '+NAMES[head]+' | '+steps+' | '+pair(train, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_accuracy', True)+' | '+pair(unseen, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_log_loss')+' | '+pair(unseen, 'balanced_log_loss')+' |\n'
    text += '\n## Existing spectral references on identical panels\n\nAll eight older absolute/relative spectral logistic heads are retained. Their declared source-only selections among eight C values and every released probability/metric are rechecked, with exact participant/trial/material/label correspondence to the neural controls. Representation, optimizer and search budget differ; these are predictive references, not a pure architecture comparison. They reuse the same development validation. All 576 fixed-step and 192 selected neural-minus-spectral contrasts remain in the [reference comparison]('+LINK+'/spectral_reference/contrasts.json).\n\n| Dataset/reference | Train BA (%) | Familiar BA (%) | Unseen BA (%) | Familiar loss | Unseen loss |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for dataset in ('DEAP', 'SEEDIV'):
        for model in ('band_absolute', 'band_relative'):
            part = spectral[(spectral.dataset == dataset)&(spectral.model == model)]
            train, unseen, familiar = [part[part.role == role] for role in ROLES]
            text += '| '+dataset+' / '+model+' | '+pair(train, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_accuracy', True)+' | '+pair(unseen, 'balanced_accuracy', True)+' | '+pair(familiar, 'balanced_log_loss')+' | '+pair(unseen, 'balanced_log_loss')+' |\n'
    text += '\n## Clipping, exposure and integrity\n\nNew histories retain the actual number of preclip norms exceeding 1 across all 1,200 updates; old anchors lack these counts and are excluded. Clipping frequency is an observation, not an established cause of generalization failure. All 72 source-exposure points independently reconstruct exact balanced trial/window draws.\n\n| Dataset/encoder/head | Updates clipped (%), groups 1 / 2 |\n| --- | ---: |\n'
    clipping = history[~history.reused].groupby(['dataset', 'group', 'pretrained', 'head'], as_index=False).clipped_updates_last100.sum()
    clipping['frequency'] = clipping.clipped_updates_last100/1200
    for dataset in ('DEAP', 'SEEDIV'):
        for pretrained in (True, False):
            for head in HEADS[1:]:
                part = clipping[(clipping.dataset == dataset)&(clipping.pretrained == pretrained)&(clipping['head'] == head)]
                text += '| '+dataset+' / '+('pretrained' if pretrained else 'random42')+' / '+NAMES[head]+' | '+pair(part, 'frequency', True)+' |\n'
    text += '\n## Random flattened-head learning failure\n\nThe four random flattened heads end at chance BA on training and both validation roles. All eight final validation tables have exactly zero probability range across trials; three training tables are also constant and the fourth differs by only about 1.22e-7. Their final saved encoder-gradient norms are zero. This is a degenerate learning result under this adapted head and schedule, not a successful predictive baseline or evidence of generally inferior random representations. It does not establish an activation/optimizer cause or a failure in the authors\' published experiments. All four degenerate runs remain in the grid.\n\nProbability range is the maximum column-wise range across observations. Distance from uniform is the largest absolute class-probability difference from 1/K. These descriptive quantities derive directly from released probabilities and saved history; no new training, EEG inference or metric-selection rule is introduced.\n\n| Dataset/group | Train probability range | Unseen range | Familiar range | Maximum distance from uniform | Final saved encoder gradient norm |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            folder = PUBLIC/'runs'/f'{dataset.lower()}_g{group}_random42_flattened_mlp'
            ranges = []; distances = []
            for role in ROLES:
                frame = pd.read_csv(folder/f'step1200_{role}.csv')
                probability = frame[[f'p{c}' for c in range(2 if dataset == 'DEAP' else 3)]].to_numpy()
                ranges.append(float(np.ptp(probability, axis=0).max()))
                distances.append(float(abs(probability-1/probability.shape[1]).max()))
            part = history[(history.dataset == dataset)&(history.group == group)&(~history.pretrained)&(history['head'] == 'flattened_mlp')&(history.step == 1200)]
            if len(part) != 1:
                raise ValueError('Missing final random-head gradient diagnostic')
            norm = float(part.encoder_gradient_L2_before_clip.iloc[0])
            text += '| '+dataset+f' / {group} | '+' | '.join(f'{v:.3e}' for v in (*ranges, max(distances), norm))+' |\n'
    text += f'''\nStrict replay checks all 96 states using canonical batch 16 and explicit functional readouts. Maximum probability discrepancy is {proof['maximum_probability_abs']:.2e}; maximum replay metric discrepancy is {proof['maximum_metric_abs']:.2e}. Public metric recomputation differs by at most {analysis['maximum_abs_metric_discrepancy']:.2e}. Peak actual tensor allocation is {peak:.3f} GiB, excluding driver/desktop memory. All original input/feature hashes and exact anchor bindings are checked at completion. This validates state/output integrity, not every optimizer update or external dataset/pretraining authenticity. Updated parameter digests alone do not establish supervised learning: weight decay can change tensors even when a saved encoder gradient is zero.

## Complete scientific figures

Four PNG/SVG pairs retain all 288 loss and 144 final validation-effect coordinates, with linear numeric axes and all conditions. All four PNGs are visually inspected after generation; figure coordinates and bytes have separate bindings.

'''
    for name in ('pretrained_readout_learning', 'random_readout_learning', 'head_final_step_effects', 'pretraining_final_step_effects'):
        text += '!['+name.replace('_', ' ')+']('+LINK+'/figures/'+name+'.png)\n\n'
    text += f'''## Reproduction and limits

Run from `GRU-XNet_EEG_Emotion_Recognition/` without GPU/raw EEG for numerical reanalysis:

```powershell
conda activate pytorch
python scripts/analyze_cbramod_readout_boundary_adapter.py verify
python scripts/compare_cbramod_readout_spectral.py verify
python scripts/plot_cbramod_readout.py verify
python scripts/verify_publication_export.py --export-only
```

[Fitting proof]({LINK}/verification.json), [public numerical proof]({LINK}/postfit_analysis/verification.json), [all selections]({LINK}/summary.json), [spectral proof]({LINK}/spectral_reference/verification.json), [gradient history]({LINK}/postfit_analysis/gradient_history.csv), [source exposure]({LINK}/postfit_analysis/training_exposure.csv), [figure proof]({LINK}/figures/verification.json) and [canonical export]({LINK}/export_manifest.json) retain the evidence. Local full-state replay additionally needs the preserved private checkpoints, datasets and official assets.

Probabilities, labels and anonymous references are released under the author's existing explicit approval. EEG, embeddings, scaler/weight arrays and per-trial physical amplitudes remain local. Reused small validation panels and one head/random initialization cannot establish a population result or a general encoder ranking. No confidence intervals, joint three-dataset training, unseen-corpus transfer, native four-emotion comparison or general absence of EEG information are claimed. First-party DEAP signal authentication, physical calibration and actual pretraining membership remain unresolved.

A qualifying lead requires consistent primary-loss benefit across both groupings of each claimed corpus, stable matched-step evidence and meaningful prediction against uniform/spectral references. Training-only improvement is insufficient. The [decision update](GRU-XNet_Readout_Decision_2026-10-09.md) states whether this criterion is met and what follows. An existing head change alone does not establish novelty. Actual main-question adoption still requires measured evidence, an author-approved concrete proposal and an immediate fresh manuscript archive. The manuscript and current question remain unchanged.
'''
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO.parent/'GRU-XNet_CBraMod_Readout_Findings_2026-10-09.md')
    args = parser.parse_args(); args.output.write_text(render(), encoding='utf-8')
    print(json.dumps({'report': str(args.output.resolve())}))
