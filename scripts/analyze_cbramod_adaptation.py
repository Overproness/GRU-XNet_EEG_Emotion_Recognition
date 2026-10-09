"""Post-fit descriptive analysis and independent public probability/selection audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

REPO = Path(__file__).resolve().parents[1]
STUDY = 'cbramod_adaptation_2026-10-09'
OLD = 'cbramod_source_probe_2026-10-09'
ROLES = ('train', 'validation_unseen', 'validation_familiar')
MODELS = ('pretrained_average', 'pretrained_flatten', 'random42_average',
          'random42_flatten', 'band_absolute', 'band_relative')
METRICS = ('balanced_accuracy', 'balanced_log_loss')
C_GRID = (1e-6, 1e-5, 1e-4, 1e-3, .01, .1, 1., 10.)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def choice(items):
    return min(range(len(items)), key=lambda i: (
        sum(items[i]['metrics'][r]['balanced_log_loss'] for r in ROLES[1:])/2,
        -sum(items[i]['metrics'][r]['balanced_accuracy'] for r in ROLES[1:])/2, i))


def collect(public, predecessor):
    proof = read(public/'verification.json')
    if not proof['complete'] or sha(public/'summary.json') != proof['summary_sha256']:
        raise ValueError('Complete verified summary required')
    if sha(public/'plan.json') != proof['plan_sha256']:
        raise ValueError('Wrong plan binding')
    plan = read(public/'plan.json')
    for name, checksum in plan['source_sha256'].items():
        if sha(REPO/name) != checksum: raise ValueError('Frozen experiment source changed: '+name)
    summary = read(public/'summary.json')
    if not summary['development_only'] or summary['outer_test_inferences'] != 0 or plan['research_question_change_approved']:
        raise ValueError('Unexpected study scope')
    panels = {}; rows = []; selected = []; oldrows = []; maximum = 0.
    neural_candidates = {}; neural_records = []; initialization = {}; trajectory_names = {}

    def probability_row(folder, filename, record_metrics, job, model, kind, step=0):
        nonlocal maximum
        frame = pd.read_csv(folder/filename)
        classes = 2 if job['dataset'] == 'DEAP' else 3
        allowed = {'trial_id', 'subject_id', 'material_key', 'original_label', 'label', 'role',
                   *[f'p{c}' for c in range(classes)]}
        if set(frame.columns) != allowed or frame.trial_id.duplicated().any():
            raise ValueError('Unexpected public columns or duplicate observations')
        role = filename.split('_', 1)[1][:-4] if step else filename[:-4]
        if not (frame.role == role).all(): raise ValueError('Role mismatch')
        p = frame[[f'p{c}' for c in range(classes)]].to_numpy(dtype=float)
        y = frame.label.to_numpy(dtype=int)
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError('Invalid probabilities')
        np.testing.assert_allclose(p.sum(1), 1., atol=2e-12, rtol=0)
        if set(y) != set(range(classes)): raise ValueError('Incomplete panel classes')
        scores = {'n': len(y), 'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(1))),
                  'balanced_log_loss': float(np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
                                                       weights=compute_sample_weight('balanced', y)))}
        for name, value in scores.items():
            delta = abs(value-record_metrics[role][name]); maximum = max(maximum, delta)
            if delta > 2e-11: raise ValueError('Independent public metric mismatch')
        identity = (job['dataset'], job['group'], role)
        metadata = frame[['trial_id', 'subject_id', 'material_key', 'original_label', 'label']]
        if identity in panels: pd.testing.assert_frame_equal(metadata, panels[identity])
        else: panels[identity] = metadata
        return {'dataset': job['dataset'], 'group': job['group'], 'kind': kind, 'model': model,
                'trajectory': folder.name, 'step': step, 'role': role, **scores,
                'predicted_classes': int(len(np.unique(p.argmax(1))))}

    linear_results = {}
    for folder in sorted((public/'linear').iterdir()):
        record = read(folder/'record.json'); audit = read(folder/'verification.json')
        if not audit['complete'] or sha(folder/'record.json') != audit['record_sha256']:
            raise ValueError('Stale linear proof')
        candidates = record['candidates']
        if len(candidates) != 8 or tuple(i['C'] for i in candidates) != C_GRID:
            raise ValueError('Expanded C coverage mismatch')
        chosen = choice(candidates)
        if chosen != record['selected_id'] or record['selected_C'] != C_GRID[chosen]:
            raise ValueError('Expanded selection mismatch')
        if candidates[chosen]['metrics'] != record['metrics']:
            raise ValueError('Selected candidate metrics mismatch')
        job = record['job']; linear_results[(job['dataset'], job['group'], job['model'])] = record
        for role in ROLES:
            if sha(folder/f'{role}.csv') != record['artifact_sha256'][f'{role}.csv']:
                raise ValueError('Changed selected probability bytes')
            row = probability_row(folder, f'{role}.csv', record['metrics'], job, job['model'], 'expanded_linear')
            row['selected_C'] = record['selected_C']; rows.append(row); selected.append(row)
        oldfolder = predecessor/'fits'/folder.name
        oldrecord = read(oldfolder/'record.json')
        if sha(oldfolder/'record.json') != record['predecessor_record_sha256']:
            raise ValueError('Wrong predecessor binding')
        for role in ROLES:
            row = probability_row(oldfolder, f'{role}.csv', oldrecord['metrics'], job, job['model'], 'original_linear')
            row['selected_C'] = oldrecord['selected_C']; oldrows.append(row)
    if len(linear_results) != 24: raise ValueError('Incomplete linear coverage')

    for folder in sorted((public/'neural').iterdir()):
        record = read(folder/'record.json'); audit = read(folder/'verification.json')
        if not audit['complete'] or sha(folder/'record.json') != audit['record_sha256']:
            raise ValueError('Stale neural proof')
        job = record['job']; neural_records.append(record)
        trajectory_names[id(record)] = folder.name
        model = ('pretrained' if job['pretrained'] else 'random42')+'_'+('finetune' if job['trainable'] else 'frozen')
        key = (job['dataset'], job['group'], job['pretrained'], job['trainable'])
        if [i['step'] for i in record['candidates']] != [50, 200]: raise ValueError('Wrong steps')
        for item in record['candidates']:
            neural_candidates[(folder.name, item['step'])] = item
            for role in ROLES:
                name = f'step{item["step"]}_{role}.csv'
                if sha(folder/name) != record['artifact_sha256'][name]: raise ValueError('Changed neural probabilities')
                rows.append(probability_row(folder, name, item['metrics'], job, model, 'neural_candidate', item['step']))
        initialization.setdefault(key[:2], []).append(record)
    if len(neural_records) != 24 or len(neural_candidates) != 48:
        raise ValueError('Incomplete neural coverage')
    for panel, records in initialization.items():
        if len({r['head_initial_digest'] for r in records}) != 1 or len({r['sampling_digest'] for r in records}) != 1:
            raise ValueError('Unmatched head/sample streams')
        for pretrained in (True, False):
            part = [r for r in records if r['job']['pretrained'] == pretrained]
            if len({r['encoder_initial_digest'] for r in part}) != 1: raise ValueError('Unmatched encoders')

    # Independently reconstruct the predeclared job order, including LR ties.
    neural_selected = {}
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            for pretrained in (True, False):
                for trainable in (False, True):
                    condition = (dataset, group, pretrained, trainable)
                    records = [r for r in neural_records if tuple(r['job'][k] for k in
                               ('dataset', 'group', 'pretrained', 'trainable')) == condition]
                    rates = [1e-4, 3e-5] if trainable else [0.]
                    ordered = []
                    for rate in rates:
                        parts = [r for r in records if r['job']['encoder_rate'] == rate]
                        if len(parts) != 1: raise ValueError('Missing/duplicate rate')
                        trajectory = trajectory_names[id(parts[0])]
                        for item in parts[0]['candidates']: ordered.append({**item, 'trajectory': trajectory})
                    chosen = ordered[choice(ordered)]
                    published = [i for i in summary['neural'] if tuple(i[k] for k in
                                 ('dataset', 'group', 'pretrained', 'trainable')) == condition]
                    if len(published) != 1: raise ValueError('Missing neural selection')
                    result = published[0]
                    if (result['selected_trajectory'], result['selected_step']) != (chosen['trajectory'], chosen['step']):
                        raise ValueError('Independent neural selection mismatch')
                    if result['metrics'] != chosen['metrics']: raise ValueError('Selected neural metrics changed')
                    if len(result['selection_candidates']) != len(ordered): raise ValueError('Candidate coverage changed')
                    for saved, expected in zip(result['selection_candidates'], ordered):
                        if (saved['trajectory'], saved['step'], saved['metrics']) != (expected['trajectory'], expected['step'], expected['metrics']):
                            raise ValueError('Published candidate ordering changed')
                    neural_selected[condition] = result
                    for role in ROLES:
                        item = next(r for r in rows if r['kind'] == 'neural_candidate' and r['trajectory'] == chosen['trajectory']
                                    and r['step'] == chosen['step'] and r['role'] == role)
                        selected.append({**item, 'kind': 'selected_neural', 'encoder_rate': result['encoder_rate']})
    for result in summary['linear']:
        job = result['job']; record = linear_results[(job['dataset'], job['group'], job['model'])]
        if any(result[k] != record[k] for k in ('selected_C', 'selected_id', 'metrics')):
            raise ValueError('Published linear summary differs')
    if len(rows) != 216 or len(oldrows) != 72 or len(selected) != 120 or len(panels) != 12:
        raise ValueError('Incomplete probability/selection coverage')
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            train, unseen, familiar = [panels[(dataset, group, role)] for role in ROLES]
            if set(train.subject_id)&(set(unseen.subject_id)|set(familiar.subject_id)):
                raise ValueError('Training/validation participant overlap')
            if set(train.trial_id)&(set(unseen.trial_id)|set(familiar.trial_id)) or set(unseen.trial_id)&set(familiar.trial_id):
                raise ValueError('Observation role overlap')
            if set(train.material_key)&set(unseen.material_key) or not set(familiar.material_key)<=set(train.material_key):
                raise ValueError('Incorrect familiar/unseen material boundary')
    return pd.DataFrame(rows), pd.DataFrame(oldrows), pd.DataFrame(selected), neural_records, maximum


def comparisons(selected, old):
    points = []
    def add(a, b, comparison):
        for metric in METRICS:
            points.append({'comparison': comparison, 'dataset': a['dataset'], 'group': int(a['group']),
                           'role': a['role'], 'model': a['model'], 'reference': b['model'], 'metric': metric,
                           'difference': float(a[metric]-b[metric]), 'population_interval': None})
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            for role in ROLES[1:]:
                panel = selected[(selected.dataset == dataset)&(selected.group == group)&(selected.role == role)].set_index('model', drop=False)
                before = old[(old.dataset == dataset)&(old.group == group)&(old.role == role)].set_index('model', drop=False)
                for model in MODELS: add(panel.loc[model], before.loc[model], 'expanded_minus_original_linear')
                for prefix in ('pretrained', 'random42'):
                    add(panel.loc[prefix+'_finetune'], panel.loc[prefix+'_frozen'], 'finetune_minus_frozen')
                for condition in ('frozen', 'finetune'):
                    add(panel.loc['pretrained_'+condition], panel.loc['random42_'+condition], 'pretrained_minus_random42')
                for model in ('pretrained_frozen', 'pretrained_finetune', 'random42_frozen', 'random42_finetune'):
                    for reference in ('band_absolute', 'band_relative'):
                        add(panel.loc[model], panel.loc[reference], 'selected_neural_minus_expanded_spectral')
    if len(points) != 288: raise ValueError('Incomplete descriptive contrast coverage')
    return points


def exposure(public):
    """Expected coverage of the declared stream, not an optimizer-trajectory replay."""
    rows=[]
    for dataset in ('DEAP','SEEDIV'):
        for group in (1,2):
            frame=pd.read_csv(public/'linear'/f'{dataset.lower()}_g{group}_pretrained_average'/'train.csv')
            y=frame.label.to_numpy(dtype=int); classes=2 if dataset=='DEAP' else 3
            rng=np.random.default_rng(20261009); window_rng=np.random.default_rng(20261010)
            pools=[np.flatnonzero(y==c) for c in range(classes)]
            draws=np.array([rng.permutation(np.concatenate([rng.choice(pool,6//classes,replace=True)
                                                           for pool in pools])) for _ in range(200)])
            windows=window_rng.integers(0,4,size=draws.shape)
            for step in (50,200):
                pairs=set(zip(draws[:step].ravel(),windows[:step].ravel()))
                counts=np.bincount(draws[:step].ravel(),minlength=len(frame))
                rows.append({'dataset':dataset,'group':group,'step':step,'training_observations':len(frame),
                             'window_draws':int(draws[:step].size),'unique_observations_seen':int(np.count_nonzero(counts)),
                             'unique_observation_window_pairs':len(pairs),'available_observation_window_pairs':4*len(frame),
                             'minimum_draws_per_observation':int(counts.min()),'maximum_draws_per_observation':int(counts.max())})
    return pd.DataFrame(rows)


def plots(folder, selected, records):
    os.environ.setdefault('MPLCONFIGDIR', str(REPO.parent/'publication_runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = (*MODELS, 'pretrained_frozen', 'pretrained_finetune', 'random42_frozen', 'random42_finetune')
    labels = ('Pretrained average', 'Pretrained flatten', 'Random average', 'Random flatten',
              'Absolute band power', 'Relative band power', 'Pretrained: frozen neural',
              'Pretrained: fine-tuned', 'Random: frozen neural', 'Random: fine-tuned')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for i, dataset in enumerate(('DEAP', 'SEEDIV')):
        for j, metric in enumerate(METRICS):
            ax = axes[i,j]
            for group, color, marker in ((1, '#2463aa', 'o'), (2, '#bd4c25', 's')):
                panel = selected[(selected.dataset == dataset)&(selected.group == group)&(selected.role == 'validation_unseen')].set_index('model')
                ax.plot([panel.loc[m,metric]*(100 if j == 0 else 1) for m in names],
                        np.arange(10)+(group-1.5)*.12, linestyle='none', marker=marker,
                        color=color, markersize=6, label=f'Grouping {group}')
            chance = (50 if dataset == 'DEAP' else 100/3) if j == 0 else np.log(2 if dataset == 'DEAP' else 3)
            ax.axvline(chance, color='#777777', linestyle='--', linewidth=1, label='Uniform reference')
            ax.axhline(5.5, color='#aaaaaa', linewidth=.8)
            ax.set_yticks(range(10), labels if j == 0 else ['']*10); ax.invert_yaxis()
            ax.set_title(f'{dataset}: unseen source-validation'); ax.grid(axis='x', alpha=.2)
            ax.set_xlabel('Balanced accuracy (%) — higher is better' if j == 0 else 'Balanced log loss — lower is better')
            if j == 0: ax.set_xlim(0,100)
    axes[0,1].legend(fontsize=9)
    fig.suptitle('Regularization and matched adaptation: reused source panels, no outer-test inference', fontsize=13)
    for suffix in ('png', 'svg'): fig.savefig(folder/f'selected_source_controls.{suffix}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4,2,figsize=(14,13),constrained_layout=True)
    colors = {0.: '#2463aa', 1e-4: '#bd4c25', 3e-5: '#348344'}
    for i, (dataset, group) in enumerate((('DEAP',1),('DEAP',2),('SEEDIV',1),('SEEDIV',2))):
        for record in records:
            job = record['job']
            if (job['dataset'],job['group']) != (dataset,group): continue
            items = [record['initial_metrics']]+[c['metrics'] for c in record['candidates']]
            condition = 'frozen' if not job['trainable'] else f'fine LR={job["encoder_rate"]:g}'
            label = ('Pretrained' if job['pretrained'] else 'Random')+': '+condition
            for j in range(2):
                values = [m['train']['balanced_log_loss'] if j == 0 else
                          sum(m[r]['balanced_log_loss'] for r in ROLES[1:])/2 for m in items]
                axes[i,j].plot([0,50,200], values, marker='o', markersize=4, color=colors[job['encoder_rate']],
                               linestyle='-' if job['pretrained'] else '--', label=label)
        for j in range(2):
            axes[i,j].axhline(np.log(2 if dataset == 'DEAP' else 3),color='#777777',linewidth=.8,linestyle=':')
            axes[i,j].set_title(f'{dataset}, grouping {group}: '+('training' if j == 0 else 'equal familiar/unseen validation'))
            axes[i,j].set_xlabel('Optimizer updates'); axes[i,j].set_ylabel('Balanced log loss')
            axes[i,j].set_xticks([0,50,200]); axes[i,j].grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('All 24 finite-budget trajectories; initial points are unselected diagnostics',fontsize=13)
    for suffix in ('png','svg'): fig.savefig(folder/f'all_source_learning_points.{suffix}',dpi=160)
    plt.close(fig)


def generate(public, predecessor):
    folder = REPO.parent/'publication_runs'/STUDY/'postfit_analysis'
    if folder.exists(): raise FileExistsError('Preserve completed post-fit analysis')
    if not read(public/'verification.json')['complete']: raise ValueError('Wait for full verified study')
    folder.mkdir(parents=True)
    shutil.copyfile(public/'export_manifest.json',folder/'input_export_manifest.json')
    bindings = {f'current/{p.relative_to(public).as_posix()}': sha(p) for p in public.rglob('*')
                if p.is_file() and p.name != 'export_manifest.json'}
    # Bind only the old records/probability tables actually read by this analysis.
    for p in (predecessor/'fits').rglob('*'):
        if p.is_file() and (p.name == 'record.json' or p.suffix == '.csv'):
            bindings['predecessor/'+p.relative_to(predecessor).as_posix()] = sha(p)
    write(folder/'declaration.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'postfit_descriptive': True, 'analysis_source_sha256': sha(Path(__file__)), 'public_input_sha256': bindings,
        'input_export_manifest_sha256': sha(folder/'input_export_manifest.json'),
        'definition': 'Independently recompute all216 new and72 prior public probability metric sets; check twelve matched panel metadata sets, all four participant/trial/material source-role boundaries and all24 expanded/16 neural selections. Report all288 loss/accuracy contrasts:96 expanded-minus-original,32 fine-minus-frozen,32 pretrained-minus-random,128 neural-minus-each-spectral. Reconstruct eight aggregate50/200-step sampling-coverage points from declared seeds and ordered public training labels, without claiming optimizer-update replay. Individual groups only: no confidence interval, global recipe selection, new fit or test access. Plot all selected unseen results and all24 initial/50/200 source-loss trajectories. Initial metrics are bound diagnostics without an independently published initial probability table.'})
    rows, old, selected, records, maximum = collect(public, predecessor)
    rows.to_csv(folder/'candidate_metrics.csv', index=False)
    old.to_csv(folder/'predecessor_selected_metrics.csv', index=False)
    selected.to_csv(folder/'selected_metrics.csv', index=False)
    exposure(public).to_csv(folder/'expected_training_exposure.csv',index=False)
    write(folder/'contrasts.json', {'development_only': True, 'contrasts': comparisons(selected,old)})
    plots(folder,selected,records)
    result = {'complete': True, 'new_public_probability_metric_sets': len(rows), 'prior_public_probability_metric_sets': len(old),
              'selected_metric_sets': len(selected), 'matched_panel_metadata_sets': 12, 'source_role_boundary_panels': 4,
              'independent_expanded_selections': 24, 'independent_neural_selections': 16,
              'descriptive_contrast_points': 288, 'maximum_abs_metric_discrepancy': maximum,
              'declaration_sha256': sha(folder/'declaration.json'),
              'artifact_sha256': {p.name: sha(p) for p in folder.iterdir() if p.name not in ('declaration.json','verification.json')}}
    write(folder/'verification.json',result)
    destination=public/'postfit_analysis'; destination.mkdir()
    for p in folder.iterdir(): shutil.copyfile(p,destination/p.name)
    manifest=read(public/'export_manifest.json')
    for item in manifest['files']:
        if sha(public/item['file']) != item['sha256']: raise ValueError('Changed completed study export')
    manifest['files'] += [{'file':p.relative_to(public).as_posix(),'sha256':sha(p)}
                          for p in sorted(destination.iterdir())]
    manifest['postfit_analysis_source_sha256']=sha(Path(__file__))
    write(public/'export_manifest.json',manifest)
    print(json.dumps(result))


def verify(public, predecessor):
    folder = public/'postfit_analysis'; proof = read(folder/'verification.json')
    if not proof['complete'] or sha(folder/'declaration.json') != proof['declaration_sha256']:
        raise ValueError('Changed post-fit declaration')
    declaration=read(folder/'declaration.json')
    if sha(Path(__file__)) != declaration['analysis_source_sha256']: raise ValueError('Analysis source changed')
    if sha(folder/'input_export_manifest.json') != declaration['input_export_manifest_sha256']:
        raise ValueError('Changed preanalysis export snapshot')
    manifest=read(public/'export_manifest.json'); seen=set()
    for item in manifest['files']:
        if item['file'] in seen or sha(public/item['file']) != item['sha256']:
            raise ValueError('Duplicate/changed completed public export')
        seen.add(item['file'])
    actual={p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file() and p!=public/'export_manifest.json'}
    if seen != actual: raise ValueError('Incomplete canonical public manifest')
    for name, digest in declaration['public_input_sha256'].items():
        prefix, relative=name.split('/',1); source=public if prefix=='current' else predecessor
        if sha(source/relative) != digest: raise ValueError('Changed declared public input: '+name)
    for name,digest in proof['artifact_sha256'].items():
        if sha(folder/name)!=digest: raise ValueError('Changed post-fit artifact')
    rows,old,selected,records,maximum=collect(public,predecessor)
    for name,frame in (('candidate_metrics.csv',rows),('predecessor_selected_metrics.csv',old),('selected_metrics.csv',selected)):
        pd.testing.assert_frame_equal(pd.read_csv(folder/name),frame,check_exact=False,rtol=0,atol=2e-11)
    saved=read(folder/'contrasts.json')['contrasts']; points=comparisons(selected,old)
    if saved != points: raise ValueError('Public contrast mismatch')
    pd.testing.assert_frame_equal(pd.read_csv(folder/'expected_training_exposure.csv'),exposure(public))
    print(json.dumps({'passed':True,'new_public_probability_metric_sets':len(rows),
                      'prior_public_probability_metric_sets':len(old),'selected_metric_sets':len(selected),
                      'descriptive_contrast_points':len(points),'maximum_abs_metric_discrepancy':maximum,
                      'scope':'Public probabilities, metadata pairing, complete source selections and descriptive points; no raw-waveform authentication, optimizer-update replay or population uncertainty.'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('generate','verify'))
    args=parser.parse_args()
    public=REPO/'results/development'/STUDY; predecessor=REPO/'results/development'/OLD
    (generate if args.action=='generate' else verify)(public,predecessor)
