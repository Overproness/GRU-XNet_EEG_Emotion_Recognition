"""Independent public source-metric check and complete descriptive recipe contrasts.

Added after declaration; never imported by fitting, tuning or the worker. No raw
EEG, private weights, test outcomes, bootstrap or optimizer replay is used here.
"""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[1]
STUDY = 'preprocessing_diagnostic_v2_2026-10-09'
ROLES = ('train', 'validation_unseen', 'validation_familiar')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def independent(path, expected):
    rows = pd.read_csv(path)
    columns = sorted([c for c in rows if c.startswith('p') and c[1:].isdigit()], key=lambda c: int(c[1:]))
    p = rows[columns].to_numpy(dtype=float); y = rows.label.to_numpy(dtype=int)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-6):
        raise ValueError('Invalid probabilities')
    labels = sorted(set(y))
    if labels != list(range(len(columns))) or len(y) != expected['n']:
        raise ValueError('Population/class mismatch')
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.))
    actual = {'n': len(y), 'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(1))),
              'balanced_log_loss': float(np.mean([loss[y == c].mean() for c in labels]))}
    error = max(abs(actual[f]-expected[f]) for f in ('balanced_accuracy', 'balanced_log_loss'))
    if error > 1e-6:
        raise ValueError('Independent source metric mismatch')
    return error


def score(candidate):
    a, b = (candidate['metrics'][r] for r in ROLES[1:])
    return ((a['balanced_log_loss']+b['balanced_log_loss'])/2,
            -(a['balanced_accuracy']+b['balanced_accuracy'])/2)


def contrasts(selected):
    rows = []
    for (dataset, group, montage, norm, seconds, baseline), first in selected.items():
        if baseline:
            continue
        references = []
        if montage == 'native':
            references.append(('native_minus_common14', (dataset, group, 'common14', norm, seconds, False)))
        if norm == 'trial_zscore':
            references.append(('trial_zscore_minus_source_channel', (dataset, group, montage, 'source_channel', seconds, False)))
        if seconds == 4:
            references.append(('4s_minus_40s', (dataset, group, montage, norm, 40, False)))
        for label, other in references:
            for role in ROLES:
                for metric in ('balanced_accuracy', 'balanced_log_loss'):
                    rows.append({'contrast': label, 'dataset': dataset, 'group': group,
                        'montage': montage, 'normalization': norm, 'seconds': seconds,
                        'role': role, 'metric': metric,
                        'difference': first['metrics'][role][metric]-selected[other]['metrics'][role][metric]})
    for key, first in selected.items():
        dataset, group, montage, norm, seconds, baseline = key
        if not baseline:
            continue
        other = selected[(dataset, group, montage, norm, seconds, False)]
        for role in ROLES:
            for metric in ('balanced_accuracy', 'balanced_log_loss'):
                rows.append({'contrast': 'baseline_minus_no_baseline', 'dataset': dataset,
                    'group': group, 'montage': montage, 'normalization': norm, 'seconds': seconds,
                    'role': role, 'metric': metric, 'difference': first['metrics'][role][metric]-other['metrics'][role][metric]})
    return pd.DataFrame(rows)


def plot(public, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for row, (dataset, group) in enumerate((('DEAP', 1), ('DEAP', 2), ('SEEDIV', 1), ('SEEDIV', 2))):
        for column, seconds in enumerate((4, 40)):
            ax = axes[row, column]
            for montage, color in (('common14', '#3465a4'), ('native', '#cf6b1a')):
                folder = public/'fits'/f'{dataset.lower()}_g{group}_{montage}_source_channel_{seconds}s_b0_lr0.001'
                history = json.loads((folder/'history.json').read_text())
                x = [h['step'] for h in history]
                for role, style, name in (('train', '--', 'train'), ('validation_unseen', '-', 'unseen validation'),
                                          ('validation_familiar', ':', 'familiar validation')):
                    y = [h['metrics'][role]['balanced_accuracy'] for h in history]
                    ax.plot(x, y, style, color=color, linewidth=1.5, label=f'{montage}: {name}')
            ax.axhline(.5 if dataset == 'DEAP' else 1/3, color='#888888', linewidth=.8, alpha=.6)
            ax.set_ylim(0, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1.))
            ax.set_title(f'{dataset}, grouping {group}, {seconds}s input', fontsize=11)
            ax.grid(alpha=.18)
            if column == 0: ax.set_ylabel('Balanced accuracy')
            if row == 3: ax.set_xlabel('Training updates')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5, .965), fontsize=9)
    fig.suptitle('Source learning curves: fixed LR 0.001, source-channel normalization', fontsize=13, y=.995)
    fig.text(.5, .012, 'EMA evaluation; one initialization; small reused source panels; no outer-test inference or uncertainty interval.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .90))
    fig.savefig(output/'source_learning_curves.png', dpi=220)
    fig.savefig(output/'source_learning_curves.svg')
    svg = output/'source_learning_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)


def run(public):
    completion = json.loads((public/'verification.json').read_text())
    if not completion.get('complete') or not completion.get('passed'):
        raise ValueError('Complete verified grid required before comparison')
    plan = json.loads((public/'plan.json').read_text())
    expected_jobs = {(j['dataset'], j['group'], json.dumps(j['recipe'], sort_keys=True), j['lr']) for j in plan['jobs']}
    folders = sorted((public/'fits').glob('*'))
    if len(folders) != 72:
        raise ValueError('Unexpected neural coverage')
    metric_sets, checked, maximum = 0, 0, 0.
    seen, grouped, prefixes = set(), {}, []
    for folder in folders:
        record = json.loads((folder/'record.json').read_text())
        certificate = json.loads((folder/'verification.json').read_text())
        if not certificate['passed'] or certificate['record_sha256'] != sha(folder/'record.json'):
            raise ValueError('Invalid fit certificate')
        if record['plan_sha256'] != sha(public/'plan.json'):
            raise ValueError('Wrong fit declaration')
        for name, expected in record['artifact_sha256'].items():
            if Path(name).suffix in ('.json', '.csv'):
                if sha(folder/name) != expected:
                    raise ValueError('Changed public artifact')
                checked += 1
        for h in json.loads((folder/'history.json').read_text()):
            for role in ROLES:
                maximum = max(maximum, independent(folder/f'curve{h["step"]}_{role}.csv', h['metrics'][role]))
                metric_sets += 1
        if record['selected'] != min(record['candidates'], key=score)['id']:
            raise ValueError('Wrong within-trajectory source selection')
        for c in record['candidates']:
            for role in ROLES:
                maximum = max(maximum, independent(folder/f'{c["id"]}_{role}.csv', c['metrics'][role]))
                metric_sets += 1
        j = record['job']; recipe = j['recipe']
        job_key = (j['dataset'], j['group'], json.dumps(recipe, sort_keys=True), j['lr'])
        if job_key in seen: raise ValueError('Duplicate neural job')
        seen.add(job_key)
        key = (j['dataset'], j['group'], recipe['montage'], recipe['normalization'], recipe['seconds'], recipe['baseline'])
        grouped.setdefault(key, []).extend([{**c, 'lr': j['lr'], 'fit': folder.name} for c in record['candidates']])
        if recipe == {'montage': 'common14', 'normalization': 'source_channel', 'seconds': 40, 'baseline': False}:
            old = REPO/'results/development/learning_controls_2026-10-06/fits'/f'source_curve_{j["dataset"].lower()}_eegnet_{j["group"]}_42_unexposed_{j["lr"]}_original'/'record.json'
            if old.exists():
                earlier = json.loads(old.read_text())
                state = next(c['state_digest'] for c in record['candidates'] if c['id'] == 'step200_ema')
                prefixes.append({'fit': folder.name, 'previous_record_sha256': sha(old),
                    'exact_200_state_match': state == earlier['checkpoints']['200']['state_digest']})
    if seen != expected_jobs:
        raise ValueError('Missing/unexpected declared neural jobs')
    classical_count = 0
    for folder in sorted((public/'classical').glob('*')):
        record = json.loads((folder/'record.json').read_text())
        certificate = json.loads((folder/'verification.json').read_text())
        if not certificate['passed'] or certificate['record_sha256'] != sha(folder/'record.json'):
            raise ValueError('Invalid classical certificate')
        for name, expected in record['artifact_sha256'].items():
            if Path(name).suffix == '.csv':
                if sha(folder/name) != expected: raise ValueError('Changed classical probabilities')
                checked += 1
        for recipe in record['recipes']:
            if recipe['selected'] != min(recipe['candidates'], key=score)['id']:
                raise ValueError('Wrong classical source selection')
            for c in recipe['candidates']:
                for role in ROLES:
                    maximum = max(maximum, independent(folder/f'{c["id"]}_{role}.csv', c['metrics'][role]))
                    metric_sets += 1
                classical_count += 1
        for role in ROLES:
            maximum = max(maximum, independent(folder/f'context_prior_{role}.csv', record['context'][role]))
            metric_sets += 1
    if classical_count != 80 or len(grouped) != 36:
        raise ValueError('Incomplete classical or neural recipe coverage')
    selected = {key: min(candidates, key=score) for key, candidates in grouped.items()}
    rows = []
    for key, c in selected.items():
        for role in ROLES:
            rows.append(dict(zip(('dataset', 'group', 'montage', 'normalization', 'seconds', 'baseline'), key)) |
                {'role': role, 'lr': c['lr'], 'updates': c['step'], 'BN': c['normalization'], **c['metrics'][role]})
    output = public/'postfit_analysis'
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'selected_recipe_metrics.csv', index=False)
    contrasts(selected).to_csv(output/'paired_recipe_contrasts.csv', index=False)
    plot(public, output)
    result = {'passed': True, 'plan_sha256': sha(public/'plan.json'), 'analyzer_sha256': sha(Path(__file__)),
        'neural_trajectories': 72, 'classical_candidates': classical_count,
        'metric_sets_independently_recomputed': metric_sets, 'public_artifact_hashes_checked': checked,
        'maximum_metric_error': maximum, 'previous_source_prefixes': prefixes,
        'test_predictions_used': 0,
        'scope': 'Independent public-probability point metrics and source-selection logic, descriptive matched recipe contrasts and stored source-state prefix comparisons. No EEG/weights, bootstrap endpoints, optimizer or first-party signal authentication; plots use fixed LR/EMA and do not select a successful recipe.'}
    write(output/'verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, default=REPO/'results/development'/STUDY)
    args = parser.parse_args()
    run(args.public.resolve())
