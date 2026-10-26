"""Predeclared full-width EEG/context controls; earlier fit sources untouched."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from .data import digest, sha256, write_json
from .full_context_models import FullControl, spectrogram
from .grouped_material_controls import (annotate, cells, indices, ARMS, metrics, key,
    frame, probabilities, load_deap, weights, bootstrap, check_rows, check_metrics, check_coverage)
from .temporal_controls import load as load_temporal, sequence_features, C_VALUES
from .reve_probe import load_inputs
from .material_controls import state_digest
from .train import seed_everything

REPO = Path(__file__).resolve().parents[1]
NEURAL = ('gru', 'lstm', 'cbsatt_local', 'gru_context')
ALL = (*NEURAL, 'prior', 'context_logistic')
UPDATES = 200
INTERVAL = 10
GROUP = 1
SOURCES = ['gruxnet/full_context_models.py', 'gruxnet/full_context_controls.py',
           'scripts/full_context_controls.py', 'scripts/audit_full_context_inputs.py', 'gruxnet/grouped_material_controls.py',
           'gruxnet/temporal_controls.py', 'gruxnet/material_controls.py',
           'gruxnet/session_controls.py', 'gruxnet/data.py', 'gruxnet/prepare.py',
           'gruxnet/reve_probe.py', 'gruxnet/train.py', 'scripts/native_seediv_diagnostic.py']


def plan():
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'development_only': True,
            'research_question_change_approved': False, 'group': GROUP,
            'spectrogram_precision': 'STFT and log1p computed in float64, then stored and modeled in float32; independent SciPy replay before fitting.',
            'datasets': ['SEEDIV', 'DEAP'], 'neural_models': list(NEURAL),
            'neural_fits': {'SEEDIV': 360, 'DEAP': 320, 'total': 680},
            'observation': 'Same first40s common14 electrodes as preceding feature controls; offline full-trial4..40Hz filtering before prefix. Hann STFT128,hop64,centerFalse; bins4 through40Hz inclusive,log1p(abs/64); [14,37,79]. No amplitude normalization within test participant, interpolation, augmentation, AMP or target adaptation. Channel/frequency mean/std over training trials and time only, floor1e-6. Full prefix directly represented, recurrent input9 ordered CNN time steps.',
            'architecture': 'Original dynamic GRU-XNet full widths32/64/128, separate electrode CNN parameters and BatchNorm, three2x2 pools, ReLU/dropout.5, two-layer bidirectional128 GRU, custom4head256attention/residual/LayerNorm, classifier256/128. Grouped vectorization numerically compared with exact local original. Matched BiLSTM changes recurrence only (same hidden width/layers, unequal parameter counts explicitly reported). Local CBSAtt reference preserves original global pooling, one-step single-layer BiLSTM, attention with residualLayerNorm and128classifier. Local reference reproduction is not certified author-code or published-score reproduction. Physical-input/common14/coarse3 adaptations differ from historical129x126/maxchannels/pooledaugmented experiment.',
            'partitions': 'Reuse only predeclared grouping1 (participant20261007,material20261017), every SEED3session*3rotation*5fold or DEAP5rotation*8fold, both exposed/unexposed arms. Same exact matching and validation/test trials as completed feature controls. One initialization42+1000fold+10000rotation; no selection between groupings. No pooled-corpus experiment.',
            'optimization': '200 AdamWupdates,12 balanced trial draws (4/classSEED,6/classDEAP),lr.001,wd.01,clip1,FP32 deterministicCUDA. Same canonical participant/class/rank draw stream per cell across all neural models/arms; reset trainingRNGinit+1000000. Validation every10,max primary trialBA then min balancedlogloss, first tie. Test predictions produced only after checkpoint selection. Budget is first-pass matched control, not claim of convergence; no test-driven extension. GRU andGRUcontext identical initial EEG state.',
            'context': 'Label-only material prior: Laplace1 per-class per-video counts, unseen key uses Laplace1 global training class counts. For every training row remove ALL rows belonging to that participant before constructing its prior, including global fallback. Validation/test priors use only outer training participants, never validation/test labels. GRUcontext adds fixed log prior to full GRU logits, no extra parameters; model must learn EEG residual. Diagnostic raw prior and learned balanced logistic context-only calibration(logprior inputs,training-onlyscaler,C.01/.1/1/10 selected on same sourcevalidation) retained. Known training videos versus unseen validation/test prior distributions explicitly acknowledged.',
            'evaluation': 'PrimarySEEDcoarse3/all1080 andDEAPbinary/all1264. SecondarySEEDconditionalbinary/810 excludesneutral. Complete once-per-trial OOF; do not average unequal foldBAs. Report all models/arms, train/validation curves, selected step and parameter/resource counts. EEG adds information only if EEG+context improves beyond context controls, with uncertainty and conditional/finite-budget limitations. Accuracy alone is insufficient; primary paired balanced logloss andBA contrasts.',
            'uncertainty': '10000 paired participant and crossed participant/video percentile draws seed20261006, same previous weights and observed-cell denominators; SEED6videos per12session/native strata, DEAP40unstratified videos. Conditional fixed fits and one grouping, unadjusted exploratory comparisons. Include every model unexposed-minus-exposed, LSTM-minusGRU, localCBSAtt-minusGRU, GRUcontext-minusGRU and GRUcontext-minusboth context controls for both arms and tasks. Negative logloss difference favors first model.',
            'gate': 'No manuscript/research-question change; show findings and obtain explicit author approval and archive then-current paper before adopting a pivot.'}


def prepare(dataset, root, output):
    if output.exists(): raise FileExistsError(output)
    if dataset == 'SEEDIV':
        raw, table, upstream = load_inputs(root/'cache_reve_input_seediv')
        features, other, temporal = load_temporal(root/'cache_temporal_native_seediv')
        pd.testing.assert_frame_equal(table, other, check_exact=True)
        source = {'raw_prefix': upstream, 'temporal': temporal}
    else:
        features, table, temporal = load_deap(root/'cache_temporal_deap')
        lineage = json.loads((root/'cache_common14/lineage.json').read_text())
        records = {r['trial_id']: r for r in lineage if r['dataset'] == 'DEAP'}
        raw = np.empty((len(table), 14, 5120), dtype=np.float32)
        for i, row in table.iterrows():
            record = records[row.trial_id]; path = root/'cache_common14'/record['cache_file']
            if sha256(path) != record['cache_sha256']: raise ValueError('Changed raw waveform cache')
            raw[i] = np.load(path, allow_pickle=False)[:10].transpose(1, 0, 2).reshape(14, 5120)
        source = {'temporal': temporal, 'raw_reproduction': json.loads((root/'cache_temporal_deap/raw_reproduction.json').read_text())}
        if not source['raw_reproduction']['passed']: raise ValueError('Source replay required')
    for i in range(len(table)):
        np.testing.assert_array_equal(sequence_features(raw[i].reshape(14, 10, 512).transpose(1, 0, 2)), features[i])
    seed_everything(42)
    chunks = [spectrogram(torch.from_numpy(np.array(raw[i:i+32]))).numpy() for i in range(0, len(raw), 32)]
    x = np.concatenate(chunks)
    if x.shape != (len(table), 14, 37, 79) or not np.isfinite(x).all(): raise ValueError('Invalid STFT')
    output.mkdir(parents=True); np.save(output/'spectrograms.npy', x, allow_pickle=False)
    table.to_csv(output/'trials.csv', index=False)
    info = {'dataset': dataset, 'shape': list(x.shape), 'source': source,
            'exact_bandpower_prefix_matches': len(table), 'spectrograms_sha256': sha256(output/'spectrograms.npy'),
            'trials_sha256': sha256(output/'trials.csv'), 'transform': plan()['observation']}
    info['fingerprint'] = digest(info); write_json(output/'prepared.json', info)
    return info


def load(cache):
    info = json.loads((cache/'prepared.json').read_text()); check = dict(info); fp = check.pop('fingerprint')
    if digest(check) != fp: raise ValueError('Changed cache metadata')
    for f, k in (('spectrograms.npy', 'spectrograms_sha256'), ('trials.csv', 'trials_sha256')):
        if sha256(cache/f) != info[k]: raise ValueError('Changed prepared input')
    return np.load(cache/'spectrograms.npy', allow_pickle=False), pd.read_csv(cache/'trials.csv'), info


def prior(table, train, tested, classes, crossfit=False):
    result = []
    source = table.iloc[train]
    for row in table.iloc[tested].itertuples():
        eligible = source[source.subject_id != row.subject_id] if crossfit else source
        if not len(eligible): raise ValueError('No other source participants')
        video = eligible[eligible.material_key == row.material_key]
        selected = video if len(video) else eligible
        counts = np.bincount(selected.label.to_numpy(dtype=int), minlength=classes)+1
        result.append(counts/counts.sum())
    return np.array(result)


def contexts(table, idx, classes):
    q = np.zeros((len(table), classes), dtype=np.float64)
    for part, rows in idx.items(): q[rows] = prior(table, idx['train'], rows, classes, crossfit=part == 'train')
    return q


def batches(table, train, seed, classes):
    rows = table.iloc[train].copy(); rows['position'] = train
    rows = rows.sort_values(['subject_id', 'training_class', 'trial_id'])
    rows['rank'] = rows.groupby(['subject_id', 'training_class']).cumcount()
    keys = {r.position: f'{r.subject_id}:{r.training_class}:{r.rank}' for r in rows.itertuples()}
    pools = [rows.loc[rows.label == c, 'position'].to_numpy() for c in range(classes)]
    rng = np.random.default_rng(seed)
    sampled = np.stack([np.concatenate([rng.choice(p, 12//classes, replace=True) for p in pools]) for _ in range(UPDATES)])
    return sampled, digest([[keys[int(i)] for i in row] for row in sampled])


def normalize(x, train):
    mean = x[train].astype(np.float64).mean((0, 3), keepdims=True)
    scale = np.maximum(x[train].astype(np.float64).std((0, 3), keepdims=True), 1e-6)
    return ((x-mean)/scale).astype(np.float32), mean, scale


@torch.no_grad()
def infer(model, data, q, idx):
    model.eval()
    return np.concatenate([torch.softmax(model(data[p], q[p]), dim=1).cpu().numpy()
                           for p in (idx[i:i+12] for i in range(0, len(idx), 12))])


def configuration(dataset, info, plan_path, device):
    audit_path = plan_path.parent/f'cache_full_context_{dataset.lower()}'/'input_audit.json'
    if not audit_path.exists(): audit_path = plan_path.parent.parent/f'cache_full_context_{dataset.lower()}'/'input_audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['passed'] or audit['cache_fingerprint'] != info['fingerprint']: raise ValueError('Verified input audit required')
    return {'dataset': dataset, 'cache_fingerprint': info['fingerprint'], 'input_audit_sha256': sha256(audit_path), 'plan_sha256': sha256(plan_path),
            'source_sha256': {f: sha256(REPO/f) for f in SOURCES},
            'device': device, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
            'group': GROUP, 'updates': UPDATES, 'interval': INTERVAL, 'initializations': [42]}


def fit_neural(x, table, dataset, session, rotation, fold, arm, name, output, device):
    identifier = f'{name}_{arm}_session{session}_rotation{rotation}_fold{fold["fold"]}'
    folder = output/'models'/identifier; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'metrics.json').exists():
        record = json.loads((folder/'metrics.json').read_text())
        if record['config_sha256'] != sha256(output/'config.json'): raise ValueError('Changed resumed study')
        return record, pd.read_csv(folder/'predictions.csv')
    idx = indices(table, fold, session, rotation, arm, dataset, GROUP)
    classes = 3 if dataset == 'SEEDIV' else 2; init = 42+1000*fold['fold']+10000*rotation
    sampled, signature = batches(table, idx['train'], init, classes)
    normalized, mean, scale = normalize(x, idx['train'])
    data = torch.from_numpy(normalized).to(device); q_np = contexts(table, idx, classes)
    q = torch.from_numpy(np.log(np.maximum(q_np, 1e-12)).astype(np.float32)).to(device)
    target = torch.from_numpy(table.label.to_numpy(dtype=np.int64)).to(device)
    seed_everything(init); model = FullControl(name, classes).to(device); initial = state_digest(model.state_dict())
    seed_everything(init+1000000); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []; best = None; selected = None; state = None
    torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
    start = time.perf_counter()
    for step, batch in enumerate(sampled, 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(data[batch], q[batch]), target[batch])
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % INTERVAL == 0:
            val = metrics(table, idx['validation'], infer(model, data, q, idx['validation']), dataset)
            history.append({'step': step, 'training_loss': float(loss.item()), 'validation': val})
            if best is None or key(val, dataset) > best:
                best = key(val, dataset); selected = step; state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    training = metrics(table, idx['train'], infer(model, data, q, idx['train']), dataset)
    p = infer(model, data, q, idx['test']); tested = metrics(table, idx['test'], p, dataset)
    rows = frame(table, idx['test'], p, name, arm, session, rotation, fold['fold'], GROUP)
    rows.to_csv(folder/'predictions.csv', index=False); write_json(folder/'history.json', history)
    torch.save({'state': {k: v.cpu() for k, v in state.items()}, 'mean': mean, 'scale': scale, 'selected_step': selected}, folder/'selected.pt')
    record = {'id': identifier, 'model': name, 'arm': arm, 'session': session, 'rotation': rotation, 'fold': fold['fold'],
              'training': training, 'test': tested, 'selected_step': selected, 'history': history,
              'parameters': sum(p.numel() for p in model.parameters()), 'initial_state_sha256': initial,
              'canonical_draw_signature': signature, 'batch_digest': digest(sampled.tolist()),
              'split_sha256': digest({part: table.iloc[v].trial_id.tolist() for part, v in idx.items()}),
              'context_sha256': digest(q_np.tolist()), 'elapsed_seconds': time.perf_counter()-start,
              'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
              'config_sha256': sha256(output/'config.json'), 'checkpoint_sha256': sha256(folder/'selected.pt'),
              'history_sha256': sha256(folder/'history.json'), 'predictions_sha256': sha256(folder/'predictions.csv')}
    write_json(folder/'metrics.json', record)
    del model, optimizer, state, data, target, q
    return record, rows


def fit_context(table, dataset, output, verify=False):
    classes = 3 if dataset == 'SEEDIV' else 2; records = []; predictions = {n: {a: [] for a in ARMS} for n in ('prior', 'context_logistic')}
    for session, rotation, fold in cells(table, dataset, GROUP)[1]:
        for arm in ARMS:
            idx = indices(table, fold, session, rotation, arm, dataset, GROUP); q = contexts(table, idx, classes)
            scaler = StandardScaler().fit(np.log(q[idx['train']]))
            z = scaler.transform(np.log(np.maximum(q, 1e-12))); candidates = []; estimators = []
            for c in C_VALUES:
                est = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, tol=1e-6, random_state=42)
                est.fit(z[idx['train']], table.iloc[idx['train']].label); estimators.append(est)
                candidates.append({'C': c, 'validation': metrics(table, idx['validation'], est.predict_proba(z[idx['validation']]), dataset)})
            chosen = max(range(4), key=lambda j: key(candidates[j]['validation'], dataset)); est = estimators[chosen]
            record = {'arm': arm, 'session': session, 'rotation': rotation, 'fold': fold['fold'], 'context_sha256': digest(q.tolist()),
                      'candidates': candidates, 'selected_C': C_VALUES[chosen], 'coefficients': est.coef_.tolist(), 'intercept': est.intercept_.tolist(),
                      'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist()}
            for name, p in (('prior', q[idx['test']]), ('context_logistic', est.predict_proba(z[idx['test']]))):
                record[name] = metrics(table, idx['test'], p, dataset)
                predictions[name][arm].append(frame(table, idx['test'], p, name, arm, session, rotation, fold['fold'], GROUP))
            records.append(record)
    if verify:
        if json.loads((output/'context_records.json').read_text()) != json.loads(json.dumps(records)): raise ValueError('Context candidates/refit failed exact replay')
        for name, arms in predictions.items():
            for arm, parts in arms.items(): check_rows(pd.read_csv(output/f'predictions_{name}_{arm}.csv'), pd.concat(parts, ignore_index=True), 1e-14)
    else:
        write_json(output/'context_records.json', records)
        for name, arms in predictions.items():
            for arm, parts in arms.items(): pd.concat(parts, ignore_index=True).to_csv(output/f'predictions_{name}_{arm}.csv', index=False)
    return len(records)


def run(dataset, cache, output, plan_path, device='cuda'):
    x, original, info = load(cache); table = annotate(original, dataset, GROUP)
    output.mkdir(parents=True, exist_ok=True); config = configuration(dataset, info, plan_path, device)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text()) != config: raise ValueError('Frozen config changed')
    write_json(output/'config.json', config); write_json(output/'plan.json', json.loads(plan_path.read_text()))
    folds, conditions = cells(table, dataset, GROUP); write_json(output/'folds.json', folds)
    index = []; collected = {(n, a): [] for n in NEURAL for a in ARMS}; started = time.perf_counter()
    for session, rotation, fold in conditions:
        for name in NEURAL:
            for arm in ARMS:
                record, rows = fit_neural(x, table, dataset, session, rotation, fold, arm, name, output, device)
                index.append({'id': record['id'], 'metrics_sha256': sha256(output/'models'/record['id']/'metrics.json')})
                collected[name, arm].append(rows)
                print(f'{dataset} {len(index)}/{len(conditions)*8}: {record["id"]}, selected{record["selected_step"]}, fit{record["elapsed_seconds"]:.1f}s; batch{(time.perf_counter()-started)/60:.1f}min', flush=True)
        write_json(output/'model_index.json', index)
        # Small reviewable record chunks; checkpoints remain local.
        records = [json.loads((output/'models'/i['id']/'metrics.json').read_text()) for i in index if f'session{session}_rotation{rotation}_' in i['id']]
        write_json(output/f'metrics_session{session}_rotation{rotation}.json', records)
    for (name, arm), parts in collected.items():
        rows = pd.concat(parts, ignore_index=True); check_coverage(rows, table)
        rows.to_csv(output/f'predictions_{name}_{arm}.csv', index=False)
    fit_context(table, dataset, output)
    return index


def verify(dataset, cache, output, device='cuda'):
    x, original, info = load(cache); table = annotate(original, dataset, GROUP)
    saved_config = json.loads((output/'config.json').read_text())
    if configuration(dataset, info, output/'plan.json', device) != saved_config: raise ValueError('Changed source/input/config')
    folds, conditions = cells(table, dataset, GROUP)
    if json.loads((output/'folds.json').read_text()) != folds: raise ValueError('Changed folds')
    index = json.loads((output/'model_index.json').read_text()); expected = {f'{n}_{a}_session{s}_rotation{r}_fold{f["fold"]}' for s, r, f in conditions for n in NEURAL for a in ARMS}
    if len(index) != len(expected) or {i['id'] for i in index} != expected: raise ValueError('Incomplete batch')
    signatures = {}; initial = {}; collected = {}; maximum = 0.
    for number, item in enumerate(index, 1):
        folder = output/'models'/item['id']; record = json.loads((folder/'metrics.json').read_text())
        if sha256(folder/'metrics.json') != item['metrics_sha256'] or record['config_sha256'] != sha256(output/'config.json'): raise ValueError('Changed fit record')
        for f, k in (('selected.pt', 'checkpoint_sha256'), ('history.json', 'history_sha256'), ('predictions.csv', 'predictions_sha256')):
            if sha256(folder/f) != record[k]: raise ValueError('Changed fit artifact')
        idx = indices(table, folds[record['fold']], record['session'], record['rotation'], record['arm'], dataset, GROUP)
        if digest({p: table.iloc[v].trial_id.tolist() for p, v in idx.items()}) != record['split_sha256']: raise ValueError('Changed original trial split')
        classes = 3 if dataset == 'SEEDIV' else 2; init = 42+1000*record['fold']+10000*record['rotation']
        sampled, signature = batches(table, idx['train'], init, classes)
        if signature != record['canonical_draw_signature'] or digest(sampled.tolist()) != record['batch_digest']: raise ValueError('Unpaired draws')
        signatures.setdefault((record['session'], record['rotation'], record['fold']), set()).add(signature)
        seed_everything(init); model = FullControl(record['model'], classes).to(device)
        if state_digest(model.state_dict()) != record['initial_state_sha256'] or sum(p.numel() for p in model.parameters()) != record['parameters']: raise ValueError('Changed initialization/architecture')
        initial.setdefault((record['model'].replace('gru_context', 'gru'), record['rotation'], record['fold']), set()).add(record['initial_state_sha256'])
        checkpoint = torch.load(folder/'selected.pt', map_location='cpu', weights_only=False); model.load_state_dict(checkpoint['state'])
        normalized, mean, scale = normalize(x, idx['train'])
        np.testing.assert_array_equal(mean, checkpoint['mean']); np.testing.assert_array_equal(scale, checkpoint['scale'])
        data = torch.from_numpy(normalized).to(device); q_np = contexts(table, idx, classes)
        if digest(q_np.tolist()) != record['context_sha256']: raise ValueError('Changed participant-excluded context')
        q = torch.from_numpy(np.log(np.maximum(q_np, 1e-12)).astype(np.float32)).to(device)
        history = json.loads((folder/'history.json').read_text())
        if history != record['history'] or [h['step'] for h in history] != list(range(INTERVAL, UPDATES+1, INTERVAL)): raise ValueError('Changed update/selection budget')
        best = max(history, key=lambda h: key(h['validation'], dataset))
        if best['step'] != record['selected_step'] or checkpoint['selected_step'] != record['selected_step']: raise ValueError('Wrong checkpoint choice')
        check_metrics(metrics(table, idx['validation'], infer(model, data, q, idx['validation']), dataset), best['validation'])
        check_metrics(metrics(table, idx['train'], infer(model, data, q, idx['train']), dataset), record['training'])
        p = infer(model, data, q, idx['test']); check_metrics(metrics(table, idx['test'], p, dataset), record['test'])
        expected_rows = frame(table, idx['test'], p, record['model'], record['arm'], record['session'], record['rotation'], record['fold'], GROUP)
        rows = pd.read_csv(folder/'predictions.csv'); maximum = max(maximum, check_rows(rows, expected_rows, 1e-6))
        collected.setdefault((record['model'], record['arm']), []).append(rows)
        del model, data, q
        if number % 40 == 0: print(f'Replayed {dataset} {number}/{len(index)}', flush=True)
    if any(len(v) != 1 for v in (*signatures.values(), *initial.values())): raise ValueError('Unpaired streams/initial states')
    for (name, arm), parts in collected.items():
        rows = pd.read_csv(output/f'predictions_{name}_{arm}.csv'); check_coverage(rows, table)
        check_rows(rows, pd.concat(parts, ignore_index=True), 1e-15)
    for session in sorted(table.session.unique()):
        for rotation in range(3 if dataset == 'SEEDIV' else 5):
            records = [json.loads((output/'models'/i['id']/'metrics.json').read_text()) for i in index if f'session{session}_rotation{rotation}_' in i['id']]
            if json.loads((output/f'metrics_session{session}_rotation{rotation}.json').read_text()) != records: raise ValueError('Changed public fit chunks')
    context_count = fit_context(table, dataset, output, verify=True)
    for name in ('prior', 'context_logistic'):
        for arm in ARMS: check_coverage(pd.read_csv(output/f'predictions_{name}_{arm}.csv'), table)
    result = {'passed': True, 'dataset': dataset, 'neural_fits_replayed': len(index), 'context_cells_refitted': context_count,
              'context_candidates_refitted': 4*context_count, 'maximum_neural_probability_error': maximum,
              'config_sha256': sha256(output/'config.json'), 'scope': 'Input/source bindings, exact original trial partitions, paired draws/initializations, train-only normalizers and participant-crossfit priors, all selected checkpoint train/validation/test replay, all20checkpoints selection histories, completeOOF and exact context-logistic candidate refits. Not full training trajectory or first-party signal authentication.'}
    write_json(output/'verification.json', result); return result


def statistic(rows, dataset, task, subjects, materials, sw, mw, which):
    if which == 'BA': return bootstrap(rows, dataset, task, subjects, materials, sw, mw)
    rows = rows.copy(); p = probabilities(rows)
    if dataset == 'SEEDIV' and task == 'binary':
        selected = rows.original_label.ne(0).to_numpy(); rows = rows[selected].copy(); p = p[selected]
        pos = p[:, 2]/np.maximum(p[:, 1:].sum(1), 1e-12); p = np.stack([1-pos, pos], axis=1)
        target = (rows.original_label == 3).astype(int).to_numpy()
    else: target = rows.label.to_numpy(dtype=int)
    rows['target'] = target; rows['loss'] = -np.log(np.clip(p[np.arange(len(p)), target], 1e-12, 1))
    cell = rows.groupby(['subject_id', 'material_key']).agg(target=('target', 'first'), loss=('loss', 'mean'))
    points = []; samples = []
    for c in range(p.shape[1]):
        included = cell.target.eq(c).astype(float).unstack().reindex(index=subjects, columns=materials).fillna(0).to_numpy()
        value = (cell.loss*cell.target.eq(c)).unstack().reindex(index=subjects, columns=materials).fillna(0).to_numpy()
        denominator = ((sw@included)*mw).sum(1)
        if (denominator <= 0).any(): raise ValueError('Lost class in bootstrap')
        points.append(value.sum()/included.sum()); samples.append(((sw@value)*mw).sum(1)/denominator)
    return float(np.mean(points)), np.mean(samples, axis=0)


def analyze(dataset, output, write=True):
    if not json.loads((output/'verification.json').read_text())['passed']: raise ValueError('Replay before analysis')
    frames = {n: {a: pd.read_csv(output/f'predictions_{n}_{a}.csv') for a in ARMS} for n in ALL}
    table = frames['gru']['exposed']; subjects, materials, sw, mw = weights(table, dataset); fixed = np.ones_like(mw)
    tasks = ('coarse3', 'binary') if dataset == 'SEEDIV' else ('binary',)
    result = {'development_only': True, 'research_question_change_approved': False, 'dataset': dataset, 'group': GROUP, 'models': {}, 'contrasts': []}
    distribution = {}
    for name, arms in frames.items():
        result['models'][name] = {}; distribution[name] = {}
        for arm, rows in arms.items():
            check_coverage(rows, table); item = metrics(rows, np.arange(len(rows)), probabilities(rows), dataset)
            result['models'][name][arm] = item; distribution[name][arm] = {}
            for task in tasks:
                distribution[name][arm][task] = {}
                for which, field in (('BA', 'balanced_accuracy'), ('logloss', 'balanced_log_loss')):
                    point, crossed = statistic(rows, dataset, task, subjects, materials, sw, mw, which)
                    _, participant = statistic(rows, dataset, task, subjects, materials, sw, fixed, which)
                    if abs(point-item[task][field]) > 1e-10: raise ValueError('Point mismatch')
                    distribution[name][arm][task][which] = (crossed, participant)
    pairs = [(n, 'unexposed', n, 'exposed') for n in ALL]
    pairs += [(n, a, 'gru', a) for n in ('lstm', 'cbsatt_local', 'gru_context') for a in ARMS]
    pairs += [('gru_context', a, n, a) for n in ('prior', 'context_logistic') for a in ARMS]
    for task in tasks:
        for which, field in (('BA', 'balanced_accuracy'), ('logloss', 'balanced_log_loss')):
            for n, a, m, b in pairs:
                ca, pa = distribution[n][a][task][which]; cb, pb = distribution[m][b][task][which]
                result['contrasts'].append({'task': task, 'statistic': which, 'model_a': n, 'arm_a': a, 'model_b': m, 'arm_b': b,
                    'difference': result['models'][n][a][task][field]-result['models'][m][b][task][field],
                    'crossed_percentile_95': np.quantile(ca-cb, [.025, .975]).tolist(),
                    'participant_percentile_95': np.quantile(pa-pb, [.025, .975]).tolist()})
    result['bootstrap'] = {'draws': 10000, 'seed': 20261006, 'subject_weight_digest': digest(sw.tolist()), 'material_weight_digest': digest(mw.tolist()),
                           'scope': 'Paired observed-cell subject and crossed person/video percentile intervals, conditional fixed models/one grouping/cohorts; unadjusted exploratory. BA positive/logloss negative favors modelA.'}
    if write: write_json(output/'comparison.json', result)
    return result
