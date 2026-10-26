"""Fold-local tuning and complete repeated held-out controls.

Earlier frozen studies are imported without changing their fitting sources.
All output paths are relative to the study, including checkpoint certificates.
"""
from __future__ import annotations
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .data import digest, sha256, write_json
from .learning_controls import SOURCES as PREVIOUS_SOURCES, draw_stream, identifier as old_identifier
from .full_context_controls_v2 import prior, infer
from .grouped_material_controls import annotate, cells, indices, metrics, frame, GROUPS
from .full_context_models_v2 import FullControl
from .eegnet_control import EEGNetControl
from .material_controls import state_digest
from .train import seed_everything
from scripts.source_bn_diagnostic import calibrate

REPO = Path(__file__).resolve().parents[1]
STUDY = 'heldout_tuning_2026-10-06'
MODELS = ('eegnet', 'eegnet_context', 'gru')
ARMS = ('exposed', 'unexposed')
INITIALIZATIONS = (42, 91)
RATES = (.001, .0003)
STEPS = (200, 600, 1200)
NORMS = ('ema', 'source_population')
C_VALUES = (.01, .1, 1., 10.)
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES, 'scripts/source_bn_diagnostic.py',
    'scripts/audit_learning_controls.py', 'gruxnet/heldout_tuning.py',
    'scripts/heldout_tuning.py', 'scripts/audit_heldout_tuning.py',
    'scripts/analyze_heldout_tuning.py', 'scripts/export_heldout_tuning.py', 'scripts/within_video_alignment.py')))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    write_json(temporary, value)
    temporary.replace(path)


def case_id(job):
    return (f'{job["dataset"].lower()}_{job["model"]}_g{job["group"]}_i{job["initialization"]}'
            f'_s{job["session"]}_r{job["rotation"]}_f{job["fold"]}_{job["arm"]}')


def context_id(job):
    return (f'{job["dataset"].lower()}_g{job["group"]}_s{job["session"]}'
            f'_r{job["rotation"]}_f{job["fold"]}_{job["arm"]}')


def make_model(name, classes):
    return FullControl('gru', classes) if name == 'gru' else EEGNetControl(classes, context=name == 'eegnet_context')


def keep_checkpoint(job):
    return job['model'] != 'gru' or (job['rotation'] == 0 and job['fold'] == 0)


def partitions(base, job):
    table = annotate(base, job['dataset'], job['group'])
    fs, _ = cells(table, job['dataset'], job['group'])
    fold = next(f for f in fs if f['fold'] == job['fold'])
    idx = indices(table, fold, job['session'], job['rotation'], job['arm'], job['dataset'], job['group'])
    videos = set(table.iloc[idx['train']].material_key)
    eligible = table.subject_id.isin(fold['validation']) & table.material_key.isin(videos)
    if job['dataset'] == 'SEEDIV': eligible &= table.session.eq(job['session'])
    idx = dict(idx)
    idx['validation_unseen'] = idx.pop('validation')
    idx['validation_familiar'] = np.flatnonzero(eligible.to_numpy())
    classes = 3 if job['dataset'] == 'SEEDIV' else 2
    roles = list(idx)
    for role, rows in idx.items():
        if len(rows) == 0 or set(table.iloc[rows].label) != set(range(classes)):
            raise ValueError(f'Empty/missing-class panel: {case_id(job)} {role}')
    for k, first in enumerate(roles):
        for second in roles[k+1:]:
            if set(idx[first]) & set(idx[second]): raise ValueError('Trial role overlap')
    source_people = set(table.iloc[idx['train']].subject_id)
    val_people = set(table.iloc[idx['validation_unseen']].subject_id) | set(table.iloc[idx['validation_familiar']].subject_id)
    test_people = set(table.iloc[idx['test']].subject_id)
    if source_people & val_people or source_people & test_people or val_people & test_people:
        raise ValueError('Participant role overlap')
    if job['arm'] == 'unexposed':
        for role in ('train', 'validation_unseen', 'validation_familiar'):
            if set(table.iloc[idx[role]].material_key) & set(table.iloc[idx['test']].material_key):
                raise ValueError('Unseen target video accessed in source panel')
    return table, idx


def jobs_and_feasibility(bases):
    jobs = []; contexts = []; checks = []; retained = 0
    # Complete EEGNet and EEG-plus-context coverage before the expensive GRU grid.
    for model in MODELS:
        for dataset in ('DEAP', 'SEEDIV'):
            for group in GROUPS:
                table = annotate(bases[dataset], dataset, group)
                _, conditions = cells(table, dataset, group)
                for initialization in INITIALIZATIONS:
                    tested = []
                    for session, rotation, fold in conditions:
                        pair = []
                        for arm in ARMS:
                            job = dict(dataset=dataset, model=model, group=group, initialization=initialization,
                                       session=session, rotation=rotation, fold=fold['fold'], arm=arm)
                            other, idx = partitions(bases[dataset], job); pair.append(idx)
                            jobs.append(job); retained += int(keep_checkpoint(job))
                            if model == MODELS[0] and initialization == INITIALIZATIONS[0]:
                                contexts.append({k: v for k, v in job.items() if k not in ('model', 'initialization')})
                                checks.append({'id': context_id(job), 'counts': {k: len(v) for k, v in idx.items()},
                                               'class_counts': {k: other.iloc[v].label.value_counts().sort_index().to_dict() for k,v in idx.items()}})
                        for role in ('validation_unseen', 'test'): np.testing.assert_array_equal(pair[0][role], pair[1][role])
                        # Exposure arms have the same number of source trials per person/class.
                        a = other.iloc[pair[0]['train']].groupby(['subject_id','label']).size()
                        b = other.iloc[pair[1]['train']].groupby(['subject_id','label']).size()
                        pd.testing.assert_series_equal(a, b)
                        tested.extend(other.iloc[pair[0]['test']].trial_id)
                    if len(tested) != len(table) or set(tested) != set(table.trial_id):
                        raise ValueError('Incomplete once-per-trial OOF partition')
    if len(jobs) != 2040 or len(contexts) != 340: raise ValueError('Unexpected study size')
    return jobs, contexts, checks, retained


def choose(candidates, dataset):
    task = 'coarse3' if dataset == 'SEEDIV' else 'binary'
    def score(candidate):
        values = [candidate['validation'][p][task] for p in ('validation_unseen','validation_familiar')]
        return (np.mean([m['balanced_log_loss'] for m in values]),
                -np.mean([m['balanced_accuracy'] for m in values]))
    return min(candidates, key=score)  # Stable first-declared tie.


def normalizer(values, train, waveform):
    axes = (0, 2) if waveform else (0, 3)
    source = values[train].astype(np.float64)
    mean = source.mean(axes, keepdims=True)
    scale = np.maximum(source.std(axes, keepdims=True), 1e-6)
    return mean, scale


def population(model, training):
    # Context changes only output logits; CNN population moments need no prior.
    original = getattr(model, 'context', None)
    if original is not None: model.context = False
    try: return calibrate(model, training)
    finally:
        if original is not None: model.context = original


def probabilities(model, inputs, log_prior):
    return infer(model, inputs, log_prior, np.arange(len(inputs)))


def prediction_frame(table, idx, p, job):
    rows = frame(table, idx, p, job['model'], job['arm'], job['session'], job['rotation'], job['fold'], job['group'])
    rows['seed'] = job['initialization']
    return rows.reset_index(drop=True)


def source_priors(table, idx, classes):
    return {role: prior(table, idx['train'], rows, classes, crossfit=role == 'train')
            for role, rows in idx.items() if role != 'test'}


def prefix_guard(root, job, lr, actual):
    if job['model'] == 'eegnet_context' or job['session'] != 1 or job['rotation'] != 0 or job['fold'] != 0: return None
    previous_job = dict(kind='source_curve', dataset=job['dataset'], model=job['model'], group=job['group'],
                        initialization=job['initialization'], arm=job['arm'], lr=lr, label_mode='original')
    path = root/'learning_controls_2026-10-06/fits'/old_identifier(previous_job)/'record.json'
    old = json.loads(path.read_text())
    expected = old['checkpoints']['200']['state_digest']
    if actual != expected: raise ValueError('Exact previous 200-update source prefix failed')
    return {'record_sha256': sha256(path), 'step': 200, 'state_digest': actual, 'exact': True}


def fit_neural(job, x, wave, base, output, root, device='cuda'):
    folder = output/'fits'/case_id(job); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'record.json').exists(): return json.loads((folder/'record.json').read_text())
    if (folder/'selection.json').exists():
        return finish_neural(job, x, wave, base, output, device)
    table, idx = partitions(base, job)
    classes = 3 if job['dataset'] == 'SEEDIV' else 2
    init = job['initialization']+1000*job['fold']+10000*job['rotation']
    sampled, signature = draw_stream(table, idx['train'], init, classes, STEPS[-1])
    allowed = np.sort(np.concatenate([v for k,v in idx.items() if k != 'test']))
    remap = np.full(len(table), -1, dtype=int); remap[allowed] = np.arange(len(allowed))
    if set(allowed) & set(idx['test']): raise ValueError('Outer test input entered fitting')
    values = wave if job['model'] != 'gru' else x
    mean, scale = normalizer(values[allowed], remap[idx['train']], job['model'] != 'gru')
    source = torch.from_numpy(((values[allowed]-mean)/scale).astype(np.float32)).to(device)
    target = torch.tensor(table.iloc[allowed].label.to_numpy(dtype=np.int64), device=device)
    priors = source_priors(table, idx, classes)
    q = torch.empty((len(allowed), classes), dtype=torch.float32, device=device)
    for role, p in priors.items(): q[remap[idx[role]]] = torch.tensor(np.log(p).astype(np.float32), device=device)
    candidates = []; prefixes = []; best_state = None; best = None; initial_digest = None
    start = time.perf_counter(); peak = 0
    for lr in RATES:
        seed_everything(init); model = make_model(job['model'], classes).to(device)
        initial = state_digest(model.state_dict())
        if initial_digest is not None and initial != initial_digest: raise ValueError('Unpaired LR initial states')
        initial_digest = initial
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        seed_everything(init+1000000)
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        losses = []
        for step, batch in enumerate(sampled, 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            positions = remap[batch]
            loss = nn.functional.cross_entropy(model(source[positions], q[positions]), target[positions])
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if job['model'] != 'gru': model.constrain()
            losses.append(float(loss.item()))
            if step not in STEPS: continue
            ema_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            ema_digest = state_digest(ema_state)
            if step == 200:
                guard = prefix_guard(root, job, lr, ema_digest)
                if guard: prefixes.append(dict(lr=lr, **guard))
            before_rng = torch.get_rng_state().clone()
            before_cuda = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
            for norm in NORMS:
                if norm == 'source_population': population(model, source[remap[idx['train']]])
                val = {}
                for panel in ('validation_unseen', 'validation_familiar'):
                    positions = remap[idx[panel]]
                    p = probabilities(model, source[positions], q[positions])
                    filename = f'candidate_lr{lr}_step{step}_{norm}_{panel}.csv'
                    prediction_frame(table, idx[panel], p, job).to_csv(folder/filename, index=False)
                    val[panel] = metrics(table, idx[panel], p, job['dataset'])
                candidate = {'id': f'lr{lr}_step{step}_{norm}', 'lr': lr, 'step': step, 'normalization': norm,
                             'validation': val, 'state_digest': state_digest(model.state_dict()),
                             'training_batch_loss_mean_since_previous_budget': float(np.mean(losses))}
                candidates.append(candidate)
                selected = choose(candidates, job['dataset'])
                if best is None or selected['id'] != best['id']:
                    best = selected
                    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            if state_digest(model.state_dict()) != ema_digest: raise ValueError('Trajectory BN state was not restored')
            if not torch.equal(before_rng, torch.get_rng_state()): raise ValueError('Evaluation changed CPU training RNG')
            if device == 'cuda' and not torch.equal(before_cuda, torch.cuda.get_rng_state()): raise ValueError('Evaluation changed CUDA training RNG')
            losses = []
        if device == 'cuda': peak = max(peak, torch.cuda.max_memory_allocated())
        del model, optimizer, ema_state
    torch.save({'state': best_state, 'mean': mean, 'scale': scale}, folder/'selected.pt')
    candidate_files = sorted(p.name for p in folder.glob('candidate_*.csv'))
    selection = {'job': job, 'config_sha256': sha256(output/'config.json'), 'created_utc': stamp(),
                 'candidates': candidates, 'selected': best['id'], 'selected_state_digest': state_digest(best_state),
                 'checkpoint_sha256': sha256(folder/'selected.pt'), 'initial_state_digest': initial_digest,
                 'canonical_draw_signature': signature, 'batch_digest': digest(sampled.tolist()),
                 'split_trials': {k: table.iloc[v].trial_id.tolist() for k,v in idx.items()},
                 'source_accessed_trials': table.iloc[allowed].trial_id.tolist(), 'prefix_replay': prefixes,
                 'artifact_sha256': {f: sha256(folder/f) for f in candidate_files}}
    atomic_json(folder/'selection.json', selection)  # SEALED BEFORE outer-test inference.
    del source, target, q
    del best_state
    return finish_neural(job, x, wave, base, output, device, time.perf_counter()-start, peak)


def finish_neural(job, x, wave, base, output, device='cuda', fitting_seconds=None, peak=None):
    folder = output/'fits'/case_id(job)
    selection = json.loads((folder/'selection.json').read_text())
    if selection['job'] != job or selection['config_sha256'] != sha256(output/'config.json'):
        raise ValueError('Changed sealed case configuration')
    for name, expected in selection['artifact_sha256'].items():
        if sha256(folder/name) != expected: raise ValueError('Changed sealed candidate')
    if sha256(folder/'selected.pt') != selection['checkpoint_sha256']: raise ValueError('Changed sealed weights')
    checkpoint = torch.load(folder/'selected.pt', map_location='cpu', weights_only=False)
    table, idx = partitions(base, job)
    if {k: table.iloc[v].trial_id.tolist() for k,v in idx.items()} != selection['split_trials']:
        raise ValueError('Changed sealed partition')
    classes = 3 if job['dataset'] == 'SEEDIV' else 2
    mean, scale = checkpoint['mean'], checkpoint['scale']
    values = wave if job['model'] != 'gru' else x
    model = make_model(job['model'], classes).to(device); model.load_state_dict(checkpoint['state'])
    candidate_files = list(selection['artifact_sha256'])
    selected_metrics = {}; files = list(candidate_files)
    for role, rows in idx.items():
        inputs = torch.from_numpy(((values[rows]-mean)/scale).astype(np.float32)).to(device)
        p_prior = prior(table, idx['train'], rows, classes, crossfit=role == 'train')
        q = torch.tensor(np.log(p_prior).astype(np.float32), device=device)
        p = probabilities(model, inputs, q)
        filename = f'predictions_{role}.csv'; prediction_frame(table, rows, p, job).to_csv(folder/filename, index=False)
        files.append(filename); selected_metrics[role] = metrics(table, rows, p, job['dataset'])
        del inputs, q
    record = {'id': case_id(job), 'job': job, 'config_sha256': sha256(output/'config.json'),
              'selection_sha256': sha256(folder/'selection.json'), 'metrics': selected_metrics,
              'checkpoint_retained': keep_checkpoint(job), 'checkpoint_sha256': sha256(folder/'selected.pt'),
              'elapsed_seconds': fitting_seconds, 'peak_allocated_cuda_bytes': peak,
              'parameters': sum(p.numel() for p in model.parameters()),
              'artifact_sha256': {f: sha256(folder/f) for f in files}}
    atomic_json(folder/'record.json', record)
    del model, checkpoint
    return record


def fit_context(job, base, output):
    folder = output/'contexts'/context_id(job); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'record.json').exists(): return json.loads((folder/'record.json').read_text())
    if (folder/'selection.json').exists(): return finish_context(job,base,output)
    full_job = dict(job, model='context_logistic', initialization=42)
    table, idx = partitions(base, full_job); classes = 3 if job['dataset'] == 'SEEDIV' else 2
    q = source_priors(table, idx, classes)
    scaler = StandardScaler().fit(np.log(q['train']))
    inputs = {k: scaler.transform(np.log(v)) for k,v in q.items()}
    candidates = []; models = []
    for c in C_VALUES:
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, tol=1e-6, random_state=42).fit(inputs['train'], table.iloc[idx['train']].label)
        if int(model.n_iter_.max()) >= 2000: raise ValueError('Context calibrator failed declared iteration budget')
        val = {}
        for role in ('validation_unseen','validation_familiar'):
            p = model.predict_proba(inputs[role]); val[role] = metrics(table, idx[role], p, job['dataset'])
            prediction_frame(table, idx[role], p, full_job).to_csv(folder/f'candidate_C{c}_{role}.csv', index=False)
        candidates.append({'id': str(c), 'C': c, 'validation': val}); models.append(model)
    chosen = choose(candidates, job['dataset']); model = models[candidates.index(chosen)]
    files = sorted(p.name for p in folder.glob('candidate_*.csv'))
    selection = {'job': job, 'config_sha256': sha256(output/'config.json'), 'created_utc': stamp(),
                 'candidates': candidates, 'selected': chosen['id'],
                 'split_trials': {k: table.iloc[v].trial_id.tolist() for k,v in idx.items()},
                 'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                 'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
                 'classes': model.classes_.tolist(), 'artifact_sha256': {f: sha256(folder/f) for f in files}}
    atomic_json(folder/'selection.json', selection)
    return finish_context(job,base,output)


def finish_context(job,base,output):
    folder=output/'contexts'/context_id(job)
    selection=json.loads((folder/'selection.json').read_text())
    if selection['job']!=job or selection['config_sha256']!=sha256(output/'config.json'): raise ValueError('Changed sealed context configuration')
    for name,expected in selection['artifact_sha256'].items():
        if sha256(folder/name)!=expected: raise ValueError('Changed context candidate')
    full_job=dict(job,model='context_logistic',initialization=42)
    table,idx=partitions(base,full_job); classes=3 if job['dataset']=='SEEDIV' else 2
    if {k:table.iloc[v].trial_id.tolist() for k,v in idx.items()}!=selection['split_trials']: raise ValueError('Changed context partition')
    q=source_priors(table,idx,classes)
    scaler=StandardScaler().fit(np.log(q['train']))
    np.testing.assert_array_equal(scaler.mean_,selection['scaler_mean']); np.testing.assert_array_equal(scaler.scale_,selection['scaler_scale'])
    selected=next(c for c in selection['candidates'] if c['id']==selection['selected'])
    model=LogisticRegression(C=selected['C'],class_weight='balanced',max_iter=2000,tol=1e-6,random_state=42).fit(scaler.transform(np.log(q['train'])),table.iloc[idx['train']].label)
    np.testing.assert_array_equal(model.coef_,selection['coef']); np.testing.assert_array_equal(model.intercept_,selection['intercept'])
    files=list(selection['artifact_sha256'])
    result = {}
    for role, rows in idx.items():
        if role == 'test': q[role] = prior(table, idx['train'], rows, classes)
        result[role] = {}
        for name, p in (('prior', q[role]), ('context_logistic', model.predict_proba(scaler.transform(np.log(q[role]))))):
            filename = f'predictions_{name}_{role}.csv'; files.append(filename)
            prediction_frame(table, rows, p, dict(full_job, model=name)).to_csv(folder/filename, index=False)
            result[role][name] = metrics(table, rows, p, job['dataset'])
    record = {'id': context_id(job), 'job': job, 'config_sha256': sha256(output/'config.json'),
              'selection_sha256': sha256(folder/'selection.json'), 'metrics': result,
              'artifact_sha256': {f: sha256(folder/f) for f in files}}
    atomic_json(folder/'record.json', record)
    return record
