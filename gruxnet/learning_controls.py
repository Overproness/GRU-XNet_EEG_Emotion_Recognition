"""Source-only learning diagnostics. Frozen previous fitting sources untouched."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import digest, sha256, write_json
from .eegnet_control import EEGNetControl
from .full_context_models_v2 import FullControl
from .full_context_controls_v2 import load, normalize, infer, SOURCES as PREVIOUS_SOURCES
from .grouped_material_controls import annotate, cells, indices, metrics, key, GROUPS
from .material_controls import state_digest
from .reve_probe import load_inputs
from .train import seed_everything

REPO = Path(__file__).resolve().parents[1]
MODELS = ('gru', 'lstm', 'cbsatt_local', 'eegnet')
LEARNING_RATES = (.001, .0003)
UPDATES = 1200
SOURCES = tuple(dict.fromkeys(('gruxnet/learning_controls.py', 'scripts/learning_controls.py',
           'gruxnet/eegnet_control.py', 'scripts/audit_eegnet_author_tf.py',
           'scripts/audit_eegnet_control.py', 'gruxnet/full_context_models_v2.py',
           'gruxnet/full_context_controls_v2.py', 'gruxnet/grouped_material_controls.py',
           'gruxnet/material_controls.py', 'gruxnet/temporal_controls.py',
           'gruxnet/reve_probe.py', 'gruxnet/data.py', 'gruxnet/train.py', *PREVIOUS_SOURCES)))


def jobs():
    result = []
    for dataset in ('SEEDIV', 'DEAP'):
        for model in MODELS:
            for label_mode in ('original', 'permuted'):
                result.append(dict(kind='memorization', dataset=dataset, model=model,
                                   group=1, initialization=42, arm='unexposed', lr=.001,
                                   label_mode=label_mode, updates=400))
            groups = (1, 2) if model in ('gru', 'eegnet') else (1,)
            seeds = (42, 91) if model in ('gru', 'eegnet') else (42,)
            for group in groups:
                for initialization in seeds:
                    for arm in ('exposed', 'unexposed'):
                        for lr in LEARNING_RATES:
                            result.append(dict(kind='source_curve', dataset=dataset, model=model,
                                               group=group, initialization=initialization,
                                               arm=arm, lr=lr, label_mode='original', updates=UPDATES))
    assert len(result) == 96
    return result


def identifier(job):
    return '_'.join(str(job[k]) for k in ('kind', 'dataset', 'model', 'group',
                                        'initialization', 'arm', 'lr', 'label_mode')).lower()


def plan(root):
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'development_only': True,
            'research_question_change_approved': False,
            'phase': 'Source-only learning, authenticated-architecture baseline, and source-panel replication',
            'jobs': jobs(), 'fits': {'memorization': 16, 'source_curve': 80, 'total': 96},
            'source_sha256': {s: sha256(REPO/s) for s in SOURCES},
            'prior_fit_configs': {d: sha256(root/f'full_context_v2_{d.lower()}/config.json') for d in ('SEEDIV', 'DEAP')},
            'author_verification_sha256': sha256(root/'eegnet_author_audit_2026-10-06/port_verification.json'),
            'panel': 'Fixed session1/rotation0/fold0 in each corpus and grouping. SEED-IV uses session1 only; DEAP uses first declared rotation/fold only. Source training and validation people/videos retained exactly from previous split logic. No outer-test features or ratings passed to a model/scaler/prior/selection. This is a development diagnostic, not complete OOF/independent test confirmation or new participants.',
            'replication': 'GRU and EEGNet: full two-group by two-base-initialization factorial (42,91) on the fixed source panel, both arms and learning rates. LocalCBSAtt/BiLSTM: grouping1/init42 only. Groupings are existing predeclared 20261007/17 and20261008/18. These reuse cohorts and small validation panels; no population-level intervals or architecture ranking.',
            'optimization': '1200 AdamW updates, balanced12 trial draws, clip1, wd.01, constant LR.001 and.0003 both retained. No AMP/scheduler/earlystop/augmentation. Same canonical person/native-class/rank streams across model/arms/groups; initialization identical within architecture across arms/groupings. Validation at every10 through200 then every50 through1200; full-source train metrics at these checkpoints, plus pretraining. Retain final/selected states and complete train/validation probabilities at200/600/1200 and selected. Selected source-validation maxBA then minbalancedlogloss then firsttie; no pooled recipe selection or borrowing these panels to set hyperparameters for another outer fold.',
            'budget': '200/600/1200 comparisons are from the SAME uninterrupted trajectory, not a claim of convergence. Final-checkpoint train statistics separately recorded. Both learning rates retained, no test-driven extension or success-based stopping. Reference default dropout.5 retained. 1200 predeclared before real fitting based on synthetic resource pilot, not validation accuracy.',
            'normalization': 'GRU/reference use prior independently verified logSTFT/common14/first40s with source-only channel/frequency mean/std. EEGNet sees verified common14 raw40s at128Hz, source-only channel mean/std over source trials/time, stdfloor1e-6. Same offline full-trial filtering/prefix scope; differing raw/STFT representations mean baseline comparison is not architecture-only.',
            'eegnet': 'Authors EEGNet-8,2 defaults kernel64,F1=8,D=2,F2=16,Dropout.5,BN eps.001 momentum.99 Keras(.01 PyTorch),SAME asymmetric padding,channels-last flatten,maxnorm spatial1/head.25 after EVERY update. Keras Glorot fan definitions preserved; random draws not identical across frameworks. Port matches executed pinned original in eval/dropout-zero train,BN updates/constraints/counts. Our balanced AdamW/trial40s experiment is not author optimizer/published-score reproduction.',
            'memorization': 'Choose12 distinct training trials,4/classSEED or6/classDEAP, SHA256 deterministic order in grouping1/unexposed/session1/rotation0/fold0. Separate original labels and fixed label permutation preserving counts. Fit400updates on entire tiny batch, source-only tiny-batch normalization. No validation/test. Criterion95%BA andbalancedlogloss<.15 is descriptive capacity check, no success stopping or requirement to remove a failed model. Save final state/predictions and curves; training dropout kept. Permuted-label fit tests memorization capacity, not useful EEG emotion information.',
            'prefix_replay': 'For original full models group1/init42/lr.001, verify every prior source-validation checkpoint and sampled training loss through200 and exact state digest at prior selected step. Prior test predictions are never read. Any prefix discrepancy is reported as a reproducibility failure and stops that fit before interpretation; tolerance1e-6 probabilities/metrics,1e-5 trainingloss, exact stateSHA.',
            'verification': 'Bind original input/source/config/author/split/scaler/draw/initial/checkpoint hashes. Independently replay final and source-selected states, reconstruct full train/validation probability metrics with independent sklearn/class-meanlogloss, inspect selection rule and no-test row disjointness. Prior prefix metrics/state matched where applicable. Check every fit, including weak/failed learning outcomes; not full optimization rerun.',
            'gate': 'No main-question/manuscript change. Full independent held-out confirmation after source diagnostics needs separately predeclared outer-fold-valid recipes; author approval/archive still precedes a research-question pivot.'}


def raw_inputs(dataset, root, table):
    if dataset == 'SEEDIV':
        raw, other, source = load_inputs(root/'cache_reve_input_seediv')
        pd.testing.assert_frame_equal(table, other, check_exact=True)
        return raw, {'prefix_cache_fingerprint': source['fingerprint']}
    lineage = json.loads((root/'cache_common14/lineage.json').read_text())
    records = {r['trial_id']: r for r in lineage if r['dataset'] == 'DEAP'}
    raw = np.empty((len(table), 14, 5120), dtype=np.float32)
    hashes = {}
    for i, row in table.iterrows():
        record = records[row.trial_id]; path = root/'cache_common14'/record['cache_file']
        actual = sha256(path)
        if actual != record['cache_sha256']: raise ValueError('Changed raw EEG cache')
        hashes[row.trial_id] = actual
        raw[i] = np.load(path, allow_pickle=False)[:10].transpose(1, 0, 2).reshape(14, 5120)
    return raw, {'waveform_files_checked': len(hashes), 'ordered_file_hashes_digest': digest(hashes)}


def draw_stream(table, train, seed, classes, updates):
    rows = table.iloc[train].copy(); rows['position'] = train
    rows = rows.sort_values(['subject_id', 'training_class', 'trial_id'])
    rows['rank'] = rows.groupby(['subject_id', 'training_class']).cumcount()
    keys = {r.position: f'{r.subject_id}:{r.training_class}:{r.rank}' for r in rows.itertuples()}
    pools = [rows.loc[rows.label.eq(c), 'position'].to_numpy() for c in range(classes)]
    rng = np.random.default_rng(seed)
    sampled = np.stack([np.concatenate([rng.choice(p, 12//classes, replace=True) for p in pools])
                        for _ in range(updates)])
    return sampled, digest([[keys[int(i)] for i in batch] for batch in sampled])


def tiny_indices(table, train, classes):
    chosen = []
    for label in range(classes):
        eligible = [int(i) for i in train if table.at[i, 'label'] == label]
        eligible.sort(key=lambda i: digest(['memorization', 20261006, table.at[i, 'trial_id']]))
        chosen.extend(eligible[:12//classes])
    if len(set(chosen)) != 12: raise ValueError('Need12 distinct training trials')
    return np.asarray(chosen, dtype=int)


def normalize_waveform(x, train):
    mean = x[train].astype(np.float64).mean((0, 2), keepdims=True)
    scale = np.maximum(x[train].astype(np.float64).std((0, 2), keepdims=True), 1e-6)
    return ((x-mean)/scale).astype(np.float32), mean, scale


def probability_frame(table, rows, p):
    result = table.iloc[rows][['trial_id', 'subject_id', 'material_key', 'label', 'original_label']].copy()
    for c in range(p.shape[1]): result[f'p{c}'] = p[:, c]
    return result.reset_index(drop=True)


def make_model(name, classes):
    return EEGNetControl(classes) if name == 'eegnet' else FullControl(name, classes)


def fit(job, x, wave, base, info, output, root, config_hash, device='cuda'):
    folder = output/'fits'/identifier(job); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'record.json').exists():
        record = json.loads((folder/'record.json').read_text())
        if record['config_sha256'] != config_hash: raise ValueError('Changed resumed study')
        for name, expected in record['artifact_sha256'].items():
            if sha256(folder/name) != expected: raise ValueError('Changed resumed artifact')
        return record
    table = annotate(base, job['dataset'], job['group'])
    folds, _ = cells(table, job['dataset'], job['group']); fold = folds[0]
    idx = indices(table, fold, 1, 0, job['arm'], job['dataset'], job['group'])
    classes = 3 if job['dataset'] == 'SEEDIV' else 2
    task = 'coarse3' if classes == 3 else 'binary'
    tiny = job['kind'] == 'memorization'
    if tiny:
        idx = dict(idx); idx['train'] = tiny_indices(table, idx['train'], classes); idx['validation'] = np.array([], dtype=int)
        if job['label_mode'] == 'permuted':
            table.loc[idx['train'], 'label'] = np.random.default_rng(20261006).permutation(table.loc[idx['train'], 'label'].to_numpy())
            # Primary training metrics must use the synthetic target, including coarse3.
            if classes == 3:
                table.loc[idx['train'], 'original_label'] = np.array([0, 1, 3])[table.loc[idx['train'], 'label']]
            else:
                table.loc[idx['train'], 'original_label'] = np.array([1, 9])[table.loc[idx['train'], 'label']]
    allowed = np.sort(np.concatenate([idx['train'], idx['validation']]))
    if set(allowed) & set(idx['test']): raise ValueError('Outer-test row accessed')
    remap = np.full(len(table), -1, dtype=int); remap[allowed] = np.arange(len(allowed))
    source_x = (wave if job['model'] == 'eegnet' else x)[allowed]
    normalizer = normalize_waveform if job['model'] == 'eegnet' else normalize
    normalized, mean, scale = normalizer(source_x, remap[idx['train']])
    data = torch.from_numpy(normalized).to(device)
    target = torch.from_numpy(table.iloc[allowed].label.to_numpy(dtype=np.int64)).to(device)
    init = job['initialization']
    sampled, signature = draw_stream(table, idx['train'], init, classes, job['updates'])
    if tiny:
        sampled = np.tile(idx['train'], (job['updates'], 1)); signature = digest(sampled.tolist())
    seed_everything(init); model = make_model(job['model'], classes).to(device)
    initial = state_digest(model.state_dict()); seed_everything(init+1000000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job['lr'], weight_decay=.01)
    q = torch.zeros((len(allowed), classes), device=device)
    validation = remap[idx['validation']]; training = remap[idx['train']]
    history = []; selected = None; best = None; selected_state = None; selected_p = None
    gradient = []; saved = {}; artifacts = []; prior_replay = None
    if not tiny and job['group'] == 1 and init == 42 and job['lr'] == .001 and job['model'] != 'eegnet':
        previous = root/f'full_context_v2_{job["dataset"].lower()}'/'models'/f'{job["model"]}_{job["arm"]}_session1_rotation0_fold0'
        old = json.loads((previous/'metrics.json').read_text())
        old_checkpoint = torch.load(previous/'selected.pt', map_location='cpu', weights_only=False)
        old_digest = state_digest(old_checkpoint['state']); del old_checkpoint
        prior_replay = {'prior_record_sha256': sha256(previous/'metrics.json'), 'selected_step': old['selected_step'],
                        'expected_selected_state_digest': old_digest, 'maximum_validation_metric_error': 0.,
                        'maximum_training_batch_loss_error': 0.}
    start = time.perf_counter()
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    pretraining = metrics(table, idx['train'], infer(model, data, q, training), job['dataset'])
    seed_everything(init+1000000)
    for step, batch in enumerate(sampled, 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(data[remap[batch]], q[remap[batch]]), target[remap[batch]])
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        loss.backward()
        if step in (1, 10, 200, 600, job['updates']):
            modules = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    block = name.split('.')[0]
                    modules.setdefault(block, []).append(float(p.grad.detach().square().sum().item()))
            gradient.append({'step': step, 'before_clip_L2': {k: float(np.sqrt(sum(v))) for k, v in modules.items()}})
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        if job['model'] == 'eegnet': model.constrain()
        if prior_replay and step == old['selected_step']:
            actual = state_digest(model.state_dict())
            if actual != old_digest: raise ValueError('Original200-step prefix state does not replay exactly')
            prior_replay['selected_state_matched'] = True
        interval = 10 if step <= 200 else 50
        if step % interval == 0 or step == job['updates']:
            p_train = infer(model, data, q, training)
            train_metrics = metrics(table, idx['train'], p_train, job['dataset'])
            val_metrics = None if tiny else metrics(table, idx['validation'], infer(model, data, q, validation), job['dataset'])
            history.append({'step': step, 'training_batch_loss': float(loss.item()),
                            'training': train_metrics, 'validation': val_metrics})
            if prior_replay and step <= 200:
                prior = next(h for h in old['history'] if h['step'] == step)
                error = max(abs(prior['validation'][task][f]-val_metrics[task][f]) for f in ('balanced_accuracy', 'balanced_log_loss'))
                loss_error = abs(prior['training_loss']-float(loss.item()))
                if error > 1e-6 or loss_error > 1e-5: raise ValueError('Original source prefix metric mismatch')
                prior_replay['maximum_validation_metric_error'] = max(error, prior_replay['maximum_validation_metric_error'])
                prior_replay['maximum_training_batch_loss_error'] = max(loss_error, prior_replay['maximum_training_batch_loss_error'])
            score = (step,) if tiny else key(val_metrics, job['dataset'])
            if best is None or score > best:
                best = score; selected = step; selected_state = copy.deepcopy(model.state_dict())
                selected_p = {part: infer(model, data, q, remap[idx[part]]) for part in ('train', 'validation') if len(idx[part])}
            if step in (200, 600, job['updates']):
                saved[str(step)] = {'state_digest': state_digest(model.state_dict()), 'training': train_metrics, 'validation': val_metrics}
                for part in ('train', 'validation'):
                    if len(idx[part]):
                        path = f'predictions_{part}_step{step}.csv'
                        probability_frame(table, idx[part], infer(model, data, q, remap[idx[part]])).to_csv(folder/path, index=False)
                        artifacts.append(path)
    final_state = {k: v.cpu() for k, v in model.state_dict().items()}
    selected_state = {k: v.cpu() for k, v in selected_state.items()}
    torch.save({'state': final_state, 'mean': mean, 'scale': scale}, folder/'final.pt'); artifacts.append('final.pt')
    if selected == job['updates']:
        selected_checkpoint = 'final.pt'
    else:
        torch.save({'state': selected_state, 'mean': mean, 'scale': scale}, folder/'selected.pt')
        artifacts.append('selected.pt'); selected_checkpoint = 'selected.pt'
    for part, p in selected_p.items():
        path = f'predictions_{part}_selected.csv'
        probability_frame(table, idx[part], p).to_csv(folder/path, index=False); artifacts.append(path)
    write_json(folder/'history.json', history); artifacts.append('history.json')
    record = {'id': identifier(job), 'job': job, 'config_sha256': config_hash, 'input_fingerprint': info['fingerprint'],
              'initial_state_digest': initial, 'canonical_draw_signature': signature, 'batch_digest': digest(sampled.tolist()),
              'split_trials': {p: table.iloc[rows].trial_id.tolist() for p, rows in idx.items()},
              'split_digest': digest({p: table.iloc[rows].trial_id.tolist() for p, rows in idx.items()}),
              'model_accessed_trials': table.iloc[allowed].trial_id.tolist(),
              'targets_digest': digest(table.iloc[allowed].label.tolist()), 'pretraining': pretraining,
              'selected_step': selected, 'selected_checkpoint': selected_checkpoint,
              'selected_state_digest': state_digest(selected_state), 'final_state_digest': state_digest(final_state),
              'checkpoints': saved, 'gradient_checks': gradient, 'prior_prefix_replay': prior_replay,
              'parameters': sum(p.numel() for p in model.parameters()),
              'elapsed_seconds': time.perf_counter()-start,
              'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
              'memorization_criterion_met': bool(history[-1]['training'][task]['balanced_accuracy'] >= .95 and
                                                history[-1]['training'][task]['balanced_log_loss'] < .15) if tiny else None,
              'artifact_sha256': {name: sha256(folder/name) for name in artifacts}}
    write_json(folder/'record.json', record)
    del model, optimizer, selected_state, final_state, data, target, q
    return record


def run(root, output, device='cuda'):
    declaration = json.loads((output/'plan.json').read_text())
    config = json.loads((output/'config.json').read_text())
    if config['plan_sha256'] != sha256(output/'plan.json'): raise ValueError('Changed frozen declaration')
    for field, actual in (('device', device), ('torch', torch.__version__),
                          ('cuda', torch.version.cuda), ('cudnn', torch.backends.cudnn.version())):
        if config[field] != actual: raise ValueError('Changed fitting environment')
    if sha256(root/'eegnet_author_audit_2026-10-06/port_verification.json') != declaration['author_verification_sha256']:
        raise ValueError('Changed author-code audit')
    for dataset, expected in declaration['prior_fit_configs'].items():
        if sha256(root/f'full_context_v2_{dataset.lower()}/config.json') != expected: raise ValueError('Changed previous fit binding')
    for name, expected in declaration['source_sha256'].items():
        if sha256(REPO/name) != expected: raise ValueError(f'Changed frozen fitting source:{name}')
    config_hash = sha256(output/'config.json')
    records = []
    for dataset in ('SEEDIV', 'DEAP'):
        x, base, info = load(root/f'cache_full_context_{dataset.lower()}')
        wave, raw_record = raw_inputs(dataset, root, base)
        write_json(output/f'waveform_binding_{dataset.lower()}.json', raw_record)
        for job in declaration['jobs']:
            if job['dataset'] != dataset: continue
            record = fit(job, x, wave, base, info, output, root, config_hash, device)
            records.append(record)
            write_json(output/f'records_{dataset.lower()}.json', records if dataset == 'SEEDIV' else [r for r in records if r['job']['dataset'] == dataset])
            print(json.dumps({'completed': len(records), 'total': len(declaration['jobs']), 'id': record['id'],
                              'selected_step': record['selected_step'], 'final_training': record['checkpoints'][str(job['updates'])]['training'],
                              'final_validation': record['checkpoints'][str(job['updates'])]['validation']}), flush=True)
        del x, wave
    write_json(output/'records.json', records)
    return records
