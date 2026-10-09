"""Public selected-prediction reanalysis and descriptive source-only figures."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_probe import STUDY, ASSETS, MODELS, ROLES, sha, atomic, stamp


def main():
    output = REPO.parent/'publication_runs'/STUDY
    public = REPO/'results/development'/STUDY
    proof = json.loads((output/'verification.json').read_text())
    if not proof['complete'] or sha(output/'summary.json') != proof['summary_sha256']:
        raise ValueError('Verified complete study required')
    folder = output/'postfit_analysis'
    if folder.exists():
        raise FileExistsError('Preserve completed descriptive analysis')
    folder.mkdir(parents=True)
    bindings = {p.relative_to(public).as_posix(): sha(p) for p in public.rglob('*') if p.is_file()}
    atomic(folder/'declaration.json', {'created_utc': stamp(), 'postfit_descriptive': True,
        'analysis_source_sha256': sha(Path(__file__)), 'public_input_sha256': bindings,
        'definition': 'Independently recompute all72selected train/validation probability metric sets from public CSVs. Recheck identical panel observations across models. Report all96pretrained-minus-random/absolute/relative validation metric contrasts (4panels*2readouts*3references*2validationroles*2metrics). No new fitting, model selection, confidence interval or test access. Figure uses individual group points, no means/uncertainty/significance.'})
    rows = []; maximum = 0.; matched = {}
    for fit in sorted((public/'fits').iterdir()):
        record = json.loads((fit/'record.json').read_text())
        for role in ROLES:
            pframe = pd.read_csv(fit/f'{role}.csv')
            y = pframe.label.to_numpy(dtype=int)
            p = pframe[[f'p{c}' for c in range(y.max()+1)]].to_numpy()
            score = {'n': len(y), 'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(1))),
                'balanced_log_loss': float(np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
                                                      weights=compute_sample_weight('balanced', y)))}
            for key in score:
                delta = abs(score[key]-record['metrics'][role][key])
                maximum = max(maximum, delta)
                if delta > 2e-11: raise ValueError('Public metric discrepancy')
            job = record['job']; identity = (job['dataset'], job['group'], role)
            metadata = pframe[['trial_id', 'subject_id', 'material_key', 'original_label', 'label']]
            if identity in matched: pd.testing.assert_frame_equal(metadata, matched[identity])
            else: matched[identity] = metadata
            rows.append({**job, 'role': role, 'selected_C': record['selected_C'], **score})
    data = pd.DataFrame(rows)
    if len(data) != 72: raise ValueError('Incomplete selected metric coverage')
    data.to_csv(folder/'selected_metrics.csv', index=False)
    contrasts = []
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            for readout in ('average', 'flatten'):
                for reference in (f'random42_{readout}', 'band_absolute', 'band_relative'):
                    for role in ROLES[1:]:
                        a = data[(data.dataset == dataset)&(data.group == group)&(data.role == role)&(data.model == f'pretrained_{readout}')].iloc[0]
                        b = data[(data.dataset == dataset)&(data.group == group)&(data.role == role)&(data.model == reference)].iloc[0]
                        for name in ('balanced_accuracy', 'balanced_log_loss'):
                            contrasts.append({'dataset': dataset, 'group': group, 'role': role,
                                'model': f'pretrained_{readout}', 'reference': reference, 'metric': name,
                                'difference': float(a[name]-b[name]), 'population_interval': None})
    if len(contrasts) != 96: raise ValueError('Incomplete declared contrasts')
    atomic(folder/'contrasts.json', {'development_only': True, 'contrasts': contrasts})
    os.environ.setdefault('MPLCONFIGDIR', str(REPO.parent/'publication_runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ('Pretrained average', 'Pretrained flatten', 'Random average', 'Random flatten',
              'Absolute band power', 'Relative band power')
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), constrained_layout=True)
    for row, dataset in enumerate(('DEAP', 'SEEDIV')):
        for col, metric in enumerate(('balanced_accuracy', 'balanced_log_loss')):
            ax = axes[row, col]
            for group, color, marker in ((1, '#2463aa', 'o'), (2, '#bd4c25', 's')):
                part = data[(data.dataset == dataset)&(data.group == group)&(data.role == 'validation_unseen')].set_index('model')
                values = [part.loc[m, metric]*(100 if col == 0 else 1) for m in MODELS]
                ax.plot(values, np.arange(6)+(group-1.5)*.12, marker=marker, linestyle='none',
                        markersize=6, color=color, label=f'Grouping {group}')
            chance = (50 if dataset == 'DEAP' else 100/3) if col == 0 else np.log(2 if dataset == 'DEAP' else 3)
            ax.axvline(chance, color='#777777', linewidth=1, linestyle='--', label='Uniform reference')
            ax.set_yticks(range(6), labels if col == 0 else ['']*6)
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
            ax.set_title(f'{dataset}: unseen source-validation')
            ax.set_xlabel('Balanced accuracy (%) — higher is better' if col == 0 else 'Balanced log loss — lower is better')
            if col == 0: ax.set_xlim(0, 100)
    axes[0, 1].legend(loc='best', fontsize=8)
    fig.suptitle('Frozen CBraMod diagnostic: reused source panels, no outer-test inference', fontsize=13)
    for suffix in ('png', 'svg'):
        fig.savefig(folder/f'cbramod_source_controls.{suffix}', dpi=180)
    plt.close(fig)
    result = {'complete': True, 'selected_metric_sets': len(rows), 'paired_contrast_points': len(contrasts),
              'maximum_abs_metric_discrepancy': maximum, 'matched_panel_metadata_sets': len(matched),
              'declaration_sha256': sha(folder/'declaration.json'),
              'artifact_sha256': {p.name: sha(p) for p in folder.iterdir() if p.name not in ('declaration.json', 'verification.json')}}
    atomic(folder/'verification.json', result)
    # The frozen study exporter handles JSON/CSV/Markdown. Copy these standalone
    # scientific figures separately and preserve their hashes in this proof.
    import shutil
    destination = public/'postfit_analysis'; destination.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir(): shutil.copyfile(path, destination/path.name)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
