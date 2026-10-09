"""Frozen public numerical analysis of matched head/pretraining contrasts."""
from pathlib import Path
import argparse
import json
import sys
import shutil
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_learning import sha, read, write, criterion, reconstruct_exposure

STUDY = 'cbramod_readout_2026-10-09'
PRIOR = 'cbramod_conditioning_2026-10-09'
HEADS = ('pooled_linear', 'pooled_mlp', 'flattened_mlp')
ROLES = ('train', 'validation_unseen', 'validation_familiar')
STEPS = (0, 200, 600, 1200)
METRICS = ('balanced_log_loss', 'balanced_accuracy')
KEYS = ('dataset', 'group', 'pretrained', 'head')


def measures(y, p):
    classes = p.shape[1]
    if set(y) != set(range(classes)):
        raise ValueError('Missing task class')
    losses = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
    return {'n': len(y), 'balanced_accuracy': float(np.mean([np.mean(p.argmax(1)[y == c] == c) for c in range(classes)])),
        'balanced_log_loss': float(np.mean([np.mean(losses[y == c]) for c in range(classes)]))}


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def contrasts(rows, scope):
    points = []
    keys = ('dataset', 'group', 'step', 'role')
    for base, block in rows.groupby(list(keys), sort=True):
        group = dict(zip(keys, map(scalar, base)))
        for pretrained in (True, False):
            part = block[block.pretrained == pretrained].set_index('head')
            if len(part) != 3 or set(part.index) != set(HEADS):
                raise ValueError('Missing head contrast cell')
            for changed, reference in (('pooled_mlp', 'pooled_linear'), ('flattened_mlp', 'pooled_linear'), ('flattened_mlp', 'pooled_mlp')):
                for metric in METRICS:
                    points.append({**group, 'scope': scope, 'kind': 'head', 'pretrained': pretrained,
                        'changed': changed, 'reference': reference, 'metric': metric,
                        'delta': float(part.loc[changed, metric]-part.loc[reference, metric])})
        for head in HEADS:
            part = block[block['head'] == head].set_index('pretrained')
            if len(part) != 2 or set(part.index) != {True, False}:
                raise ValueError('Missing pretraining contrast cell')
            for metric in METRICS:
                points.append({**group, 'scope': scope, 'kind': 'pretraining', 'head': head,
                    'changed': 'pretrained', 'reference': 'random42', 'metric': metric,
                    'delta': float(part.loc[True, metric]-part.loc[False, metric])})
    return points


def collect(public):
    proof, summary, plan = [read(public/n) for n in ('verification.json', 'summary.json', 'plan.json')]
    if not proof['complete'] or proof['plan_sha256'] != sha(public/'plan.json') or proof['summary_sha256'] != sha(public/'summary.json'):
        raise ValueError('Completed verified readout grid required')
    if not summary['development_only'] or summary['outer_test_inferences'] or summary['research_question_change_approved']:
        raise ValueError('Unexpected readout scientific scope')
    for name, checksum in plan['source_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError('Changed frozen readout source')
    folders = sorted((public/'runs').iterdir())
    if len(folders) != 24 or len(plan['jobs']) != 24:
        raise ValueError('Incomplete public readout grid')
    rows = []; selections = []; records = []; exposure = []; history = []; panels = {}; maximum = 0.
    for folder in folders:
        record, audit = read(folder/'record.json'), read(folder/'verification.json')
        job = record['job']; classes = 2 if job['dataset'] == 'DEAP' else 3
        if job not in plan['jobs'] or record['plan_sha256'] != sha(public/'plan.json'):
            raise ValueError('Undeclared readout record')
        if not audit['complete'] or audit['record_sha256'] != sha(folder/'record.json') or audit['states_checked'] != 4 or audit['metric_sets'] != 12:
            raise ValueError('Wrong readout replay certificate')
        if [i['step'] for i in record['candidates']] != list(STEPS) or record['reused'] != (job['head'] == 'pooled_linear'):
            raise ValueError('Wrong readout state/anchor coverage')
        records.append(record)
        for name, checksum in record['artifact_sha256'].items():
            if name.endswith(('.json', '.csv')) and sha(folder/name) != checksum:
                raise ValueError('Changed readout public artifact')
        if record['reused']:
            name = f'{job["dataset"].lower()}_g{job["group"]}_{"pretrained" if job["pretrained"] else "random42"}_finetune_raw_dropout_on'
            source = REPO/'results/development'/PRIOR/'runs'/name
            original = read(source/'record.json')
            if sha(source/'record.json') != record['source_record_sha256'] or sha(source/'verification.json') != record['source_verification_sha256'] or original['candidates'] != record['candidates']:
                raise ValueError('Changed readout anchor binding')
            for name in record['artifact_sha256']:
                if sha(source/name) != sha(folder/name):
                    raise ValueError('Public readout anchor bytes differ')
        else:
            if len(record['dropout_schema']) != 61 or not any(i['p'] for i in record['dropout_schema']):
                raise ValueError('Changed encoder dropout-on mode')
            if record['head_dropout_rng'] != {'p': .1, 'seed': 424243, 'index': 'seed + update; encoder global RNG restored'}:
                raise ValueError('Changed head dropout stream')
        for item in record['candidates']:
            for role in ROLES:
                frame = pd.read_csv(folder/f'step{item["step"]}_{role}.csv')
                metadata = ['trial_id', 'subject_id', 'material_key', 'original_label', 'label']
                if set(frame.columns) != {*metadata, 'role', *[f'p{c}' for c in range(classes)]} or frame.trial_id.duplicated().any() or not (frame.role == role).all():
                    raise ValueError('Wrong readout probability schema')
                y = frame.label.to_numpy(dtype=int); p = frame[[f'p{c}' for c in range(classes)]].to_numpy()
                if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                    raise ValueError('Invalid readout probability')
                np.testing.assert_allclose(p.sum(1), 1, atol=2e-12, rtol=0)
                score = measures(y, p)
                for name, value in score.items():
                    error = abs(value-item['metrics'][role][name]); maximum = max(maximum, error)
                    if error > 2e-11:
                        raise ValueError('Readout public metric differs')
                key = (job['dataset'], job['group'], role)
                if key not in panels:
                    panels[key] = frame[metadata]
                else:
                    pd.testing.assert_frame_equal(panels[key], frame[metadata])
                rows.append({**job, 'trajectory': folder.name, 'step': item['step'], 'role': role, **score})
        choices = [i for i in record['candidates'] if i['step'] > 0]
        selected = choices[criterion(choices)]
        saved = [i for i in summary['selected'] if i['job'] == job]
        if len(saved) != 1 or saved[0]['selected'] != selected or saved[0]['candidates'] != choices:
            raise ValueError('Readout source selection differs')
        for role in ROLES:
            selections.append({**job, 'trajectory': folder.name, 'selected_step': selected['step'], 'step': -1,
                'role': role, **selected['metrics'][role]})
        train = pd.read_csv(folder/'step0_train.csv')
        exposure.extend({**job, **e} for e in reconstruct_exposure(train, record, 'long'))
        hist = read(folder/'history.json')
        if [h['step'] for h in hist] != list(range(100, 1201, 100)):
            raise ValueError('Missing readout loss history')
        history.extend({**job, 'reused': record['reused'], **h} for h in hist)
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            roles = {r: panels[(dataset, group, r)] for r in ROLES}
            for i, a in enumerate(ROLES):
                for b in ROLES[i+1:]:
                    for name in ('trial_id', 'subject_id'):
                        if set(roles[a][name]) & set(roles[b][name]):
                            raise ValueError('Readout role leakage')
            if set(roles['train'].material_key) & set(roles['validation_unseen'].material_key):
                raise ValueError('Training material in unseen validation')
            if not set(roles['validation_familiar'].material_key) <= set(roles['train'].material_key):
                raise ValueError('Familiar validation contains unseen material')
            block = [r for r in records if (r['job']['dataset'], r['job']['group']) == (dataset, group)]
            if len({r['sampling_digest'] for r in block}) != 1:
                raise ValueError('Unpaired public readout streams')
            for pretrained in (True, False):
                if len({r['initial_encoder_digest'] for r in block if r['job']['pretrained'] == pretrained}) != 1:
                    raise ValueError('Unpaired public encoder initialization')
        for head in HEADS:
            block = [r for r in records if r['job']['dataset'] == dataset and r['job']['head'] == head]
            if len({r['initial_head_digest'] for r in block}) != 1:
                raise ValueError('Unpaired public head initialization')
    frames = {name: pd.DataFrame(data) for name, data in (
        ('all_metrics', rows), ('selected_metrics', selections), ('training_exposure', exposure), ('gradient_history', history))}
    points = contrasts(frames['all_metrics'], 'fixed_step')+contrasts(frames['selected_metrics'], 'selected_secondary')
    if len(rows) != 288 or len(selections) != 72 or len(exposure) != 72 or len(history) != 288 or len(points) != 1080:
        raise ValueError('Incomplete readout numerical coverage')
    return frames, points, maximum


def generate(public):
    frames, points, maximum = collect(public)
    output = REPO.parent/'publication_runs'/STUDY/'postfit_analysis'
    destination = public/'postfit_analysis'
    if output.exists() or destination.exists():
        raise FileExistsError('Preserve readout postfit analysis')
    output.mkdir()
    inputs = read(public/'export_manifest.json')
    write(output/'declaration.json', {'source_sha256': sha(Path(__file__)),
        'fit_plan_sha256': sha(public/'plan.json'), 'scope': 'Frozen matched head/pretraining contrasts; source-selected secondary, no test inference/intervals.'})
    write(output/'input_export_manifest.json', inputs)
    for name, frame in frames.items():
        frame.to_csv(output/f'{name}.csv', index=False)
    write(output/'contrasts.json', {'contrasts': points})
    proof = {'complete': True, 'declaration_sha256': sha(output/'declaration.json'),
        'metric_sets': 288, 'source_selections': 24, 'selected_metric_sets': 72,
        'exposure_points': 72, 'history_points': 288, 'reused_exact_anchors': 8,
        'fixed_step_contrasts': 864, 'selected_secondary_contrasts': 216,
        'primary_positive_validation_contrasts': 432, 'maximum_abs_metric_discrepancy': maximum,
        'artifact_sha256': {p.name: sha(p) for p in output.iterdir()}}
    write(output/'verification.json', proof)
    destination.mkdir()
    for path in output.iterdir():
        shutil.copyfile(path, destination/path.name)
    for item in inputs['files']:
        if sha(public/item['file']) != item['sha256']:
            raise ValueError('Changed pre-analysis export')
    inputs['files'] += [{'file': p.relative_to(public).as_posix(), 'sha256': sha(p)} for p in sorted(destination.iterdir())]
    write(public/'export_manifest.json', inputs)
    verify(public)


def verify(public):
    folder = public/'postfit_analysis'; proof = read(folder/'verification.json'); declaration = read(folder/'declaration.json')
    if not proof['complete'] or sha(folder/'declaration.json') != proof['declaration_sha256'] or sha(Path(__file__)) != declaration['source_sha256']:
        raise ValueError('Changed readout analysis binding')
    for name, checksum in proof['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Changed readout analysis artifact')
    for item in read(public/'export_manifest.json')['files']:
        if sha(public/item['file']) != item['sha256']:
            raise ValueError('Changed canonical readout export')
    frames, points, maximum = collect(public)
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(pd.read_csv(folder/f'{name}.csv'), frame, check_dtype=False, rtol=2e-11, atol=2e-11)
    if read(folder/'contrasts.json')['contrasts'] != points:
        raise ValueError('Changed readout contrasts')
    print(json.dumps({'passed': True, 'metric_sets': 288, 'source_selections': 24,
        'exposure_points': 72, 'reused_exact_anchors': 8, 'contrasts': 1080,
        'maximum_abs_metric_discrepancy': maximum, 'outer_test_inferences': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'verify'))
    args = parser.parse_args()
    public = REPO/'results/development'/STUDY
    (generate if args.action == 'generate' else verify)(public)
