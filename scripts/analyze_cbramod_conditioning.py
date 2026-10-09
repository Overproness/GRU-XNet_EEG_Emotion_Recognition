"""Public-only matched conditioning/dropout analysis; no EEG or GPU needed."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_learning import sha, read, write, criterion, reconstruct_exposure

STUDY = 'cbramod_conditioning_2026-10-09'
PRIOR = 'cbramod_learning_2026-10-09'
ROLES = ('train', 'validation_unseen', 'validation_familiar')
METRICS = ('balanced_log_loss', 'balanced_accuracy')
STEPS = (0, 200, 600, 1200)
FACTORS = ('dataset', 'group', 'pretrained', 'trainable', 'scaled', 'dropout')
BASE = FACTORS[:4]


def collect(public):
    proof, summary, plan = [read(public/n) for n in ('verification.json', 'summary.json', 'plan.json')]
    if not proof['complete'] or proof['plan_sha256'] != sha(public/'plan.json') or proof['summary_sha256'] != sha(public/'summary.json'):
        raise ValueError('Completed verified factorial required')
    if not summary['development_only'] or summary['outer_test_inferences'] or summary['research_question_change_approved']:
        raise ValueError('Unexpected scientific scope')
    for name, checksum in plan['source_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError('Changed frozen source: '+name)
    if len(plan['source_sha256']) != 52 or len(plan['jobs']) != 64:
        raise ValueError('Wrong declared source/factorial count')
    folders = sorted((public/'runs').iterdir())
    if len(folders) != 64:
        raise ValueError('Incomplete condition directories')
    rows = []; records = []; panels = {}; exposure = []; history = []; maximum = 0.
    previous = REPO/'results/development'/PRIOR
    for folder in folders:
        record, audit = read(folder/'record.json'), read(folder/'verification.json')
        job = record['job']; classes = 2 if job['dataset'] == 'DEAP' else 3
        if job not in plan['jobs'] or record['plan_sha256'] != sha(public/'plan.json'):
            raise ValueError('Undeclared or stale case')
        if not audit['complete'] or audit['record_sha256'] != sha(folder/'record.json') or audit['states_checked'] != 4 or audit['metric_sets'] != 12:
            raise ValueError('Wrong case replay certificate')
        if [i['step'] for i in record['candidates']] != list(STEPS) or record['reused'] != (not job['scaled'] and job['dropout']):
            raise ValueError('Wrong state/anchor coverage')
        records.append({**record, 'trajectory': folder.name})
        for name, checksum in record['artifact_sha256'].items():
            if name.endswith(('.csv', '.json')) and sha(folder/name) != checksum:
                raise ValueError('Changed released case artifact')
        if record['reused']:
            name = f'{job["dataset"].lower()}_g{job["group"]}_{"pretrained" if job["pretrained"] else "random42"}_{"finetune" if job["trainable"] else "frozen"}'
            source = previous/'long'/name
            original = read(source/'record.json')
            if sha(source/'record.json') != record['source_record_sha256'] or sha(source/'verification.json') != record['source_verification_sha256']:
                raise ValueError('Changed anchor input binding')
            if original['candidates'] != record['candidates']:
                raise ValueError('Changed original anchor states/metrics')
            for name in record['artifact_sha256']:
                if sha(source/name) != sha(folder/name):
                    raise ValueError('Reused public anchor bytes differ')
        else:
            schema = record['dropout_schema']
            if len(schema) != 61 or not job['dropout'] and any(i['p'] for i in schema):
                raise ValueError('Incorrect dropout intervention')
            if job['dropout'] and not any(i['p'] for i in schema):
                raise ValueError('Dropout-on configuration disabled')
        for item in record['candidates']:
            for role in ROLES:
                frame = pd.read_csv(folder/f'step{item["step"]}_{role}.csv')
                allowed = {'trial_id', 'subject_id', 'material_key', 'original_label', 'label', 'role', *[f'p{c}' for c in range(classes)]}
                if set(frame.columns) != allowed or frame.trial_id.duplicated().any() or not (frame.role == role).all():
                    raise ValueError('Invalid public probability schema')
                y = frame.label.to_numpy(dtype=int)
                p = frame[[f'p{c}' for c in range(classes)]].to_numpy(dtype=float)
                if set(y) != set(range(classes)) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                    raise ValueError('Invalid probabilities/classes')
                np.testing.assert_allclose(p.sum(1), 1., rtol=0, atol=2e-12)
                scores = {'n': len(y), 'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(1))),
                    'balanced_log_loss': float(np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
                                                         weights=compute_sample_weight('balanced', y)))}
                for metric, value in scores.items():
                    error = abs(value-item['metrics'][role][metric]); maximum = max(maximum, error)
                    if error > 2e-11:
                        raise ValueError('Independent metric discrepancy')
                key = (job['dataset'], job['group'], role)
                metadata = frame[['trial_id', 'subject_id', 'material_key', 'original_label', 'label']]
                if key in panels:
                    pd.testing.assert_frame_equal(metadata, panels[key])
                else:
                    panels[key] = metadata
                rows.append({**job, 'trajectory': folder.name, 'reused': record['reused'], 'step': item['step'],
                    'role': role, **scores, 'predicted_classes': len(np.unique(p.argmax(1)))})
        train = panels[(job['dataset'], job['group'], 'train')]
        for item in reconstruct_exposure(train, record, 'long'):
            exposure.append({**job, 'trajectory': folder.name, **item})
        trajectory_history = read(folder/'history.json')
        if [i['step'] for i in trajectory_history] != list(range(100, 1201, 100)):
            raise ValueError('Incomplete gradient/loss history')
        for item in trajectory_history:
            required = ('mean_last100_minibatch_loss', 'gradient_L2_before_clip', 'head_gradient_L2_before_clip',
                        'encoder_gradient_L2_before_clip', 'head_lr', 'encoder_lr')
            if not all(np.isfinite(item[k]) for k in required):
                raise ValueError('Nonfinite history')
            if not record['reused'] and not 0 <= item['clipped_updates_last100'] <= 100:
                raise ValueError('Wrong clipping count')
            history.append({**job, 'trajectory': folder.name, 'reused': record['reused'], **item})
    if len(rows) != 768 or len(exposure) != 192 or sum(r['reused'] for r in records) != 16:
        raise ValueError('Wrong complete coverage')
    if len({tuple(r['job'][k] for k in FACTORS) for r in records}) != 64:
        raise ValueError('Duplicate factorial conditions')
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            train, unseen, familiar = [panels[(dataset, group, role)] for role in ROLES]
            if set(train.subject_id) & (set(unseen.subject_id) | set(familiar.subject_id)):
                raise ValueError('Participant boundary failed')
            if set(train.trial_id) & (set(unseen.trial_id) | set(familiar.trial_id)) or set(unseen.trial_id) & set(familiar.trial_id):
                raise ValueError('Trial boundary failed')
            if set(train.material_key) & set(unseen.material_key) or not set(familiar.material_key) <= set(train.material_key):
                raise ValueError('Material boundary failed')
            panel = [r for r in records if (r['job']['dataset'], r['job']['group']) == (dataset, group)]
            if len({r['initial_head_digest'] for r in panel}) != 1 or len({r['sampling_digest'] for r in panel}) != 1:
                raise ValueError('Unmatched head or sampling')
            for pretrained in (True, False):
                block = [r for r in panel if r['job']['pretrained'] == pretrained]
                if len({r['initial_encoder_digest'] for r in block}) != 1:
                    raise ValueError('Unmatched encoder initialization')
    normalizers = []
    for folder in sorted((public/'normalizers').iterdir()):
        record, audit = read(folder/'record.json'), read(folder/'verification.json')
        job = record['job']; train = panels[(job['dataset'], job['group'], 'train')]
        if job not in plan['normalizer_jobs'] or record['plan_sha256'] != sha(public/'plan.json'):
            raise ValueError('Wrong normalizer declaration')
        if not audit['complete'] or audit['record_sha256'] != sha(folder/'record.json') or audit['validation_statistics_accessed']:
            raise ValueError('Invalid source-statistic proof')
        if record['training_trial_ids'] != train.trial_id.tolist() or record['training_trials'] != len(train):
            raise ValueError('Normalizer training membership failed')
        if record['scale_floor'] != 1e-6 or record['encoder_mode'] != 'eval' or record['extraction_batch'] != 16:
            raise ValueError('Changed conditioner definition')
        if not audit['fixed_batch_features_exact'] or audit['source_windows'] != 4*len(train):
            raise ValueError('Wrong statistic reconstruction coverage')
        for case in records:
            if all(case['job'][k] == job[k] for k in job):
                if case['initial_encoder_digest'] != record['initial_encoder_digest']:
                    raise ValueError('Wrong statistic encoder')
                if case['job']['scaled'] and case['normalizer_record_sha256'] != sha(folder/'record.json'):
                    raise ValueError('Conditioner not fixed to source record')
        normalizers.append({**job, **{k: audit[k] for k in ('source_trials', 'source_windows', 'independent_scaler_max_abs',
                             'batch8_feature_max_abs', 'batch8_normalized_feature_max_abs')}})
    if len(normalizers) != 8:
        raise ValueError('Missing normalizer proof')
    byname = {r['trajectory']: r for r in records}; selected = []; seen = set()
    for condition in summary['selected']:
        name = condition['trajectory']; record = byname[name]
        choices = [i for i in record['candidates'] if i['step'] > 0]
        if name in seen or condition['job'] != record['job'] or condition['candidates'] != choices:
            raise ValueError('Incorrect selected coverage')
        seen.add(name)
        if condition['selected'] != choices[criterion(choices)]:
            raise ValueError('Independent source selection mismatch')
        selected.extend(row for row in rows if row['trajectory'] == name and row['step'] == condition['selected']['step'])
    if len(seen) != 64 or len(selected) != 192:
        raise ValueError('Missing selections')
    return tuple(pd.DataFrame(x) for x in (rows, selected, exposure, history, normalizers)), maximum


def contrasts(rows, selected):
    result = []
    for scope, frame, keys in (('fixed_step', rows, (*BASE, 'step', 'role')),
                               ('selected_secondary', selected, (*BASE, 'role'))):
        for key, block in frame.groupby(list(keys), sort=True):
            base = dict(zip(keys, key)); cells = {(bool(r.scaled), bool(r.dropout)): r for r in block.itertuples()}
            if len(cells) != 4:
                raise ValueError('Missing matched factorial cell')
            for metric in METRICS:
                def value(s, d): return float(getattr(cells[(s, d)], metric))
                points = (
                    ('normalization_dropout_on', value(True, True)-value(False, True), (True, True), (False, True)),
                    ('normalization_dropout_off', value(True, False)-value(False, False), (True, False), (False, False)),
                    ('dropout_off_raw', value(False, False)-value(False, True), (False, False), (False, True)),
                    ('dropout_off_standardized', value(True, False)-value(True, True), (True, False), (True, True)),
                    ('interaction', (value(True, False)-value(False, False))-(value(True, True)-value(False, True)), None, None))
                for name, delta, changed, reference in points:
                    point = {'scope': scope, **base, 'metric': metric, 'contrast': name, 'delta': delta,
                             'preferred_direction': 'lower' if metric == 'balanced_log_loss' else 'higher'}
                    if changed is not None:
                        point.update({'changed_step': int(cells[changed].step), 'reference_step': int(cells[reference].step)})
                    else:
                        point['cell_steps'] = {f'{int(s)}_{int(d)}': int(cells[(s, d)].step) for s, d in cells}
                    result.append(point)
    if len(result) != 2400 or sum(p['scope'] == 'fixed_step' for p in result) != 1920:
        raise ValueError('Wrong contrast coverage')
    return result


TABLES = ('all_metrics', 'selected_metrics', 'training_exposure', 'gradient_history', 'normalizer_diagnostics')


def generate(public):
    # Require the complete, checked grid before creating a descriptive declaration.
    frames, maximum = collect(public)
    points = contrasts(frames[0], frames[1])
    output = REPO.parent/'publication_runs'/STUDY/'postfit_analysis'
    if output.exists() or (public/'postfit_analysis').exists():
        raise FileExistsError('Preserve postfit analysis')
    manifest = read(public/'export_manifest.json')
    for item in manifest['files']:
        if sha(public/item['file']) != item['sha256']:
            raise ValueError('Changed completed publication')
    output.mkdir()
    shutil.copyfile(public/'export_manifest.json', output/'input_export_manifest.json')
    bindings = {p.relative_to(REPO).as_posix(): sha(p) for folder in (public, REPO/'results/development'/PRIOR)
                for p in folder.rglob('*') if p.is_file() and p.name != 'export_manifest.json'}
    write(output/'declaration.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'postfit_descriptive': True,
        'analysis_source_sha256': sha(Path(__file__)), 'previous_analyzer_sha256': sha(REPO/'scripts/analyze_cbramod_learning.py'),
        'public_input_sha256': bindings, 'input_manifest_sha256': sha(output/'input_export_manifest.json'),
        'definition': 'Recompute all768 public metric sets,64source selections,192exposure points,768history points and8normalizer certificates. Check64factorial cells, pairedinitializations/streams,12metadata sets,4source participant/trial/material boundaries and16byte-exact old anchors. All1920same-step contrast points (960positive-step validation primary) and480checkpoint-selected secondary points retained. Five scientific figures show everycondition and finalmatched effects. No new fit, outer test, population uncertainty, global winning recipe or approved paper pivot.'})
    for name, frame in zip(TABLES, frames):
        frame.to_csv(output/(name+'.csv'), index=False)
    write(output/'contrasts.json', {'development_only': True, 'contrasts': points})
    figures(output, frames[0], points)
    proof = {'complete': True, 'probability_metric_sets': 768, 'source_selected_conditions': 64, 'selected_metric_sets': 192,
        'exposure_points': 192, 'history_points': 768, 'normalizers': 8, 'reused_exact_anchors': 16,
        'fixed_step_contrast_points': 1920, 'primary_positive_validation_points': 960, 'selected_secondary_points': 480,
        'maximum_abs_metric_discrepancy': maximum, 'declaration_sha256': sha(output/'declaration.json'),
        'artifact_sha256': {p.name: sha(p) for p in output.iterdir() if p.name not in ('declaration.json', 'verification.json')}}
    write(output/'verification.json', proof)
    destination = public/'postfit_analysis'; destination.mkdir()
    for path in output.iterdir():
        shutil.copyfile(path, destination/path.name)
    manifest['files'] += [{'file': p.relative_to(public).as_posix(), 'sha256': sha(p)} for p in sorted(destination.iterdir())]
    manifest['postfit_analysis_source_sha256'] = sha(Path(__file__))
    write(public/'export_manifest.json', manifest)
    print(json.dumps(proof))


def verify(public):
    folder = public/'postfit_analysis'
    proof, declaration = read(folder/'verification.json'), read(folder/'declaration.json')
    if not proof['complete'] or sha(folder/'declaration.json') != proof['declaration_sha256']:
        raise ValueError('Changed postfit declaration')
    if sha(Path(__file__)) != declaration['analysis_source_sha256'] or sha(REPO/'scripts/analyze_cbramod_learning.py') != declaration['previous_analyzer_sha256']:
        raise ValueError('Changed analysis code')
    if sha(folder/'input_export_manifest.json') != declaration['input_manifest_sha256']:
        raise ValueError('Changed analysis input snapshot')
    for name, checksum in declaration['public_input_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError('Changed public analysis input')
    for name, checksum in proof['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Changed analysis artifact')
    manifest = read(public/'export_manifest.json'); seen = set()
    for item in manifest['files']:
        if item['file'] in seen or sha(public/item['file']) != item['sha256']:
            raise ValueError('Invalid canonical study export')
        seen.add(item['file'])
    if seen != {p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file() and p != public/'export_manifest.json'}:
        raise ValueError('Incomplete canonical study export')
    frames, maximum = collect(public)
    for name, frame in zip(TABLES, frames):
        pd.testing.assert_frame_equal(pd.read_csv(folder/(name+'.csv')), frame, rtol=0, atol=2e-11, check_exact=False)
    if read(folder/'contrasts.json')['contrasts'] != contrasts(frames[0], frames[1]):
        raise ValueError('Changed factorial effects')
    print(json.dumps({'passed': True, 'metric_sets': 768, 'source_selections': 64, 'exposure_points': 192,
        'normalizers': 8, 'reused_exact_anchors': 16, 'fixed_step_contrasts': 1920, 'selected_secondary_contrasts': 480,
        'maximum_abs_metric_discrepancy': maximum,
        'scope': 'Public predictions, all matched effects, source selections/boundaries/normalizer metadata and anchor bytes; no full optimizer replay or untouched test interpretation.'}))


def figures(output, rows, points):
    os.environ.setdefault('MPLCONFIGDIR', str(REPO.parent/'publication_runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = (('DEAP', 1), ('DEAP', 2), ('SEEDIV', 1), ('SEEDIV', 2))
    colors = {(False, True): '#596776', (False, False): '#2463aa', (True, True): '#bd4c25', (True, False): '#348344'}
    def save(fig, name):
        for extension in ('png', 'svg'):
            fig.savefig(output/f'{name}.{extension}', dpi=150)
        plt.close(fig)
    for pretrained in (True, False):
        for trainable in (False, True):
            fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
            for i, (dataset, group) in enumerate(panels):
                part = rows[(rows.dataset == dataset) & (rows.group == group) & (rows.pretrained == pretrained) & (rows.trainable == trainable)]
                for j, role in enumerate(ROLES):
                    ax = axes[i, j]
                    for (scaled, dropout), trajectory in part[part.role == role].groupby(['scaled', 'dropout']):
                        curve = trajectory.sort_values('step')
                        ax.plot(curve.step, curve.balanced_log_loss, color=colors[(scaled, dropout)],
                            linestyle='-' if scaled else '--', marker='o', markersize=3,
                            label=('Standardized' if scaled else 'Raw')+(', dropout on' if dropout else ', dropout off'))
                    ax.axhline(np.log(2 if dataset == 'DEAP' else 3), color='#999999', linestyle=':', linewidth=.8)
                    ax.set_yscale('symlog', linthresh=2.)
                    ax.set_title(f'{dataset}, group {group}: '+{'train': 'training', 'validation_unseen': 'unseen material', 'validation_familiar': 'familiar material'}[role])
                    ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Balanced log loss (symlog above 2)'); ax.grid(alpha=.2)
                    ax.set_xticks(STEPS)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9)
            name = ('pretrained' if pretrained else 'random')+'_'+('finetune' if trainable else 'frozen')
            fig.suptitle(name.replace('_', ' ').capitalize()+': all matched source trajectories; reused development panels', fontsize=13)
            save(fig, name+'_learning')
    frame = pd.DataFrame(points)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
    names = ('normalization_dropout_on', 'normalization_dropout_off', 'dropout_off_raw', 'dropout_off_standardized', 'interaction')
    labels = ('Normalization: dropout on', 'Normalization: dropout off', 'Dropout off: raw', 'Dropout off: standardized', 'Interaction')
    for i, (dataset, group) in enumerate(panels):
        part = frame[(frame.scope == 'fixed_step') & (frame.step == 1200) & (frame.dataset == dataset) & (frame.group == group) & (frame.role != 'train')]
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for pretrained in (True, False):
                for trainable in (False, True):
                    for role in ROLES[1:]:
                        block = part[(part.pretrained == pretrained) & (part.trainable == trainable) & (part.role == role) & (part.metric == metric)].set_index('contrast')
                        offset = (.10 if pretrained else -.10)+(.035 if trainable else -.035)+(.013 if role == ROLES[1] else -.013)
                        values = [block.loc[n, 'delta']*(100 if metric == 'balanced_accuracy' else 1) for n in names]
                        ax.plot(values, np.arange(5)+offset, linestyle='none', color='#2463aa' if pretrained else '#bd4c25',
                            marker='o' if trainable else 's', markersize=5, markerfacecolor='auto' if role == ROLES[1] else 'none',
                            label=('Pretrained' if pretrained else 'Random')+(' fine-tuned' if trainable else ' frozen')+(' unseen' if role == ROLES[1] else ' familiar'))
            ax.axvline(0, color='#777777', linestyle='--'); ax.grid(axis='x', alpha=.2); ax.invert_yaxis()
            ax.set_yticks(range(5), labels if j == 0 else ['']*5)
            ax.set_title(f'{dataset}, group {group}: fixed update 1,200')
            ax.set_xlabel('Loss difference (symlog; lower better)' if j == 0 else 'Balanced accuracy difference (pp; higher better)')
            if j == 0:
                ax.set_xscale('symlog', linthresh=.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=8)
    fig.suptitle('Matched final-step effects: both validation roles; descriptive development points', fontsize=13)
    save(fig, 'matched_final_step_effects')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'verify'))
    args = parser.parse_args()
    (generate if args.action == 'generate' else verify)(REPO/'results/development'/STUDY)
