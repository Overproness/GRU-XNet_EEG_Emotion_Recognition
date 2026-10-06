"""Independent source diagnostic metrics, source boundaries and checkpoint replay."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.data import sha256, digest, write_json
from gruxnet.learning_controls import (identifier, raw_inputs, draw_stream, tiny_indices, make_model)
from gruxnet.full_context_controls_v2 import load
from gruxnet.grouped_material_controls import annotate, cells, indices
from gruxnet.material_controls import state_digest
from gruxnet.train import seed_everything


def independent_metrics(frame, dataset):
    # Stored neural probabilities originate in float32; preserve its loss
    # arithmetic rather than misclassifying harmless dtype-rounding as corruption.
    p = frame[[c for c in frame if c.startswith('p') and c[1:].isdigit()]].to_numpy(dtype=np.float32)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any(): raise ValueError('Invalid probability')
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-6, rtol=0)
    target = frame.label.to_numpy(dtype=int)
    tasks = {'coarse3' if dataset == 'SEEDIV' else 'binary': (target, p)}
    if dataset == 'SEEDIV':
        keep = frame.original_label.ne(0).to_numpy()
        pos = p[keep, 2]/np.maximum(p[keep, 1:].sum(1), 1e-12)
        tasks['binary'] = ((frame.original_label.to_numpy()[keep] == 3).astype(int), np.stack([1-pos, pos], 1))
    result = {}
    for name, (y, z) in tasks.items():
        predicted = z.argmax(1); losses = -np.log(np.clip(z[np.arange(len(y)), y], 1e-12, 1))
        result[name] = {'n': len(y), 'accuracy': float(accuracy_score(y, predicted)),
                        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
                        'macro_f1': float(f1_score(y, predicted, labels=range(z.shape[1]), average='macro', zero_division=0)),
                        'balanced_log_loss': float(np.mean([losses[y == c].mean() for c in range(z.shape[1])])),
                        'confusion_matrix': confusion_matrix(y, predicted, labels=range(z.shape[1])).tolist()}
    return result


def compare_metrics(actual, expected):
    if actual.keys() != expected.keys(): raise ValueError('Wrong target metrics')
    for task in actual:
        for name, value in actual[task].items():
            if name in ('n', 'confusion_matrix'):
                if value != expected[task][name]: raise ValueError('Metric count/confusion mismatch')
            else:
                np.testing.assert_allclose(value, expected[task][name], atol=1e-7, rtol=0)


@torch.no_grad()
def predict(model, inputs):
    model.eval()
    return np.concatenate([torch.softmax(model(inputs[i:i+12]), 1).cpu().numpy()
                           for i in range(0, len(inputs), 12)])


def verify(output, runs=None, device='cuda', partial=False):
    declaration = json.loads((output/'plan.json').read_text())
    config = json.loads((output/'config.json').read_text())
    if sha256(output/'plan.json') != config['plan_sha256']: raise ValueError('Changed declaration')
    for name, expected in declaration['source_sha256'].items():
        if sha256(REPO/name) != expected: raise ValueError('Changed fitting source')
    missing = [identifier(j) for j in declaration['jobs'] if not (output/'fits'/identifier(j)/'record.json').exists()]
    if missing and not partial: raise ValueError(f'Incomplete study: {len(missing)} missing fits')
    complete = [j for j in declaration['jobs'] if identifier(j) not in missing]
    raw = {}; spectrograms = {}; tables = {}; cache_info = {}
    if runs is not None:
        for dataset in ('SEEDIV', 'DEAP'):
            if not any(j['dataset'] == dataset for j in complete): continue
            x, table, info = load(runs/f'cache_full_context_{dataset.lower()}')
            wave, binding = raw_inputs(dataset, runs, table)
            if binding != json.loads((output/f'waveform_binding_{dataset.lower()}.json').read_text()):
                raise ValueError('Raw waveform binding mismatch')
            spectrograms[dataset], raw[dataset], tables[dataset], cache_info[dataset] = x, wave, table, info
    records = []; checked_points = replayed_states = prefix_checks = 0; max_error = 0.; init_pairs = {}; draw_pairs = {}
    input_hashes = {}
    for job in complete:
        folder = output/'fits'/identifier(job)
        r = json.loads((folder/'record.json').read_text())
        if r['job'] != job or r['id'] != identifier(job) or r['config_sha256'] != sha256(output/'config.json'):
            raise ValueError('Job/config identity mismatch')
        records.append(r); dataset = job['dataset']; classes = 3 if dataset == 'SEEDIV' else 2
        task = 'coarse3' if classes == 3 else 'binary'; tiny = job['kind'] == 'memorization'
        train_ids = set(r['split_trials']['train']); val_ids = set(r['split_trials']['validation']); test_ids = set(r['split_trials']['test'])
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids: raise ValueError('Trial-role overlap')
        if set(r['model_accessed_trials']) != train_ids | val_ids: raise ValueError('Unexpected model input')
        if set(r['model_accessed_trials']) & test_ids: raise ValueError('Model accessed outer test')
        if digest(r['split_trials']) != r['split_digest']: raise ValueError('Changed split metadata')
        for name, expected in r['artifact_sha256'].items():
            path = folder/name
            if name.endswith('.pt') and runs is None: continue
            if sha256(path) != expected: raise ValueError('Changed diagnostic artifact')
            input_hashes[str(path.relative_to(output))] = sha256(path)
        history = json.loads((folder/'history.json').read_text())
        expected_steps = list(range(10, 201, 10))+list(range(250, job['updates']+1, 50))
        if [h['step'] for h in history] != expected_steps: raise ValueError('Incomplete trajectory')
        if tiny:
            selected = job['updates']
        else:
            selected = max(history, key=lambda h: (h['validation'][task]['balanced_accuracy'],
                                                   -h['validation'][task]['balanced_log_loss']))['step']
        if selected != r['selected_step']: raise ValueError('Source selection rule mismatch')
        expected_selected_file = 'final.pt' if selected == job['updates'] else 'selected.pt'
        if r['selected_checkpoint'] != expected_selected_file: raise ValueError('Selected-state filename mismatch')
        chosen_metrics = next(h for h in history if h['step'] == selected)
        for step, checkpoint in r['checkpoints'].items():
            position = next(h for h in history if h['step'] == int(step))
            for part in ('train', 'validation'):
                if not r['split_trials'][part]: continue
                frame = pd.read_csv(folder/f'predictions_{part}_step{step}.csv')
                if frame.trial_id.tolist() != r['split_trials'][part]: raise ValueError('Incomplete probability population')
                if frame.trial_id.duplicated().any(): raise ValueError('Repeated source row')
                compare_metrics(independent_metrics(frame, dataset), checkpoint['training' if part == 'train' else 'validation'])
                compare_metrics(independent_metrics(frame, dataset), position['training' if part == 'train' else 'validation'])
                checked_points += 1
            if int(step) == job['updates'] and checkpoint['state_digest'] != r['final_state_digest']:
                raise ValueError('Final-state digest mismatch')
        for part in ('train', 'validation'):
            if not r['split_trials'][part]: continue
            frame = pd.read_csv(folder/f'predictions_{part}_selected.csv')
            if frame.trial_id.tolist() != r['split_trials'][part]: raise ValueError('Wrong selected cohort')
            compare_metrics(independent_metrics(frame, dataset), chosen_metrics['training' if part == 'train' else 'validation'])
            checked_points += 1
        if tiny:
            final = history[-1]['training'][task]
            if bool(final['balanced_accuracy'] >= .95 and final['balanced_log_loss'] < .15) != r['memorization_criterion_met']:
                raise ValueError('Capacity criterion changed')
        else:
            # Initialization is paired across LR, arm and grouping, within architecture.
            pair = (dataset, job['model'], job['initialization'])
            if pair in init_pairs and init_pairs[pair] != r['initial_state_digest']: raise ValueError('Initialization pairing broken')
            init_pairs[pair] = r['initial_state_digest']
            stream = (dataset, job['group'], job['initialization'])
            if stream in draw_pairs and draw_pairs[stream] != r['canonical_draw_signature']: raise ValueError('Model/arm draw pairing broken')
            draw_pairs[stream] = r['canonical_draw_signature']
        prefix = r['prior_prefix_replay']
        if prefix:
            if not prefix.get('selected_state_matched'): raise ValueError('Prior prefix state not matched')
            if prefix['maximum_validation_metric_error'] > 1e-6 or prefix['maximum_training_batch_loss_error'] > 1e-5:
                raise ValueError('Prior prefix tolerance changed')
            prefix_checks += 1
        for check in r['gradient_checks']:
            if not check['before_clip_L2'] or not all(np.isfinite(v) and v >= 0 for v in check['before_clip_L2'].values()):
                raise ValueError('Invalid gradient diagnostic')
        if runs is None: continue
        base = tables[dataset]; table = annotate(base, dataset, job['group'])
        folds, _ = cells(table, dataset, job['group'])
        idx = indices(table, folds[0], 1, 0, job['arm'], dataset, job['group'])
        if tiny:
            idx = dict(idx); idx['train'] = tiny_indices(table, idx['train'], classes); idx['validation'] = np.array([], dtype=int)
            if job['label_mode'] == 'permuted':
                permuted = np.random.default_rng(20261006).permutation(table.loc[idx['train'], 'label'].to_numpy())
                table.loc[idx['train'], 'label'] = permuted
                table.loc[idx['train'], 'original_label'] = (np.array([0, 1, 3]) if classes == 3 else np.array([1, 9]))[permuted]
        actual_split = {p: table.iloc[rows].trial_id.tolist() for p, rows in idx.items()}
        if actual_split != r['split_trials'] or cache_info[dataset]['fingerprint'] != r['input_fingerprint']:
            raise ValueError('Original split/input changed')
        allowed = np.sort(np.concatenate([idx['train'], idx['validation']]))
        if table.iloc[allowed].trial_id.tolist() != r['model_accessed_trials'] or digest(table.iloc[allowed].label.tolist()) != r['targets_digest']:
            raise ValueError('Wrong source targets/ordering')
        remap = np.full(len(table), -1, dtype=int); remap[allowed] = np.arange(len(allowed))
        sampled, canonical = draw_stream(table, idx['train'], job['initialization'], classes, job['updates'])
        if tiny:
            sampled = np.tile(idx['train'], (job['updates'], 1)); canonical = digest(sampled.tolist())
        if canonical != r['canonical_draw_signature'] or digest(sampled.tolist()) != r['batch_digest']: raise ValueError('Changed sampler')
        values = (raw[dataset] if job['model'] == 'eegnet' else spectrograms[dataset])[allowed]
        axes = (0, 2) if job['model'] == 'eegnet' else (0, 3)
        mean = values[remap[idx['train']]].astype(np.float64).mean(axes, keepdims=True)
        scale = np.maximum(values[remap[idx['train']]].astype(np.float64).std(axes, keepdims=True), 1e-6)
        normalized = torch.from_numpy(((values-mean)/scale).astype(np.float32)).to(device)
        seed_everything(job['initialization']); model = make_model(job['model'], classes).to(device)
        if state_digest(model.state_dict()) != r['initial_state_digest']: raise ValueError('Initial state mismatch')
        if sum(p.numel() for p in model.parameters()) != r['parameters']: raise ValueError('Parameter count changed')
        for stage, filename, state_hash in (('final', 'final.pt', r['final_state_digest']),
                                             ('selected', r['selected_checkpoint'], r['selected_state_digest'])):
            checkpoint = torch.load(folder/filename, map_location='cpu', weights_only=False)
            if state_digest(checkpoint['state']) != state_hash: raise ValueError('Checkpoint state changed')
            np.testing.assert_array_equal(mean, checkpoint['mean']); np.testing.assert_array_equal(scale, checkpoint['scale'])
            model.load_state_dict(checkpoint['state']); replayed_states += 1
            for part in ('train', 'validation'):
                if not len(idx[part]): continue
                path = folder/(f'predictions_{part}_selected.csv' if stage == 'selected' else f'predictions_{part}_step{job["updates"]}.csv')
                frame = pd.read_csv(path)
                metadata = table.iloc[idx[part]][['trial_id', 'subject_id', 'material_key', 'label', 'original_label']].reset_index(drop=True)
                pd.testing.assert_frame_equal(metadata, frame[metadata.columns], check_exact=True, check_dtype=False)
                expected = frame[[f'p{c}' for c in range(classes)]].to_numpy()
                actual = predict(model, normalized[remap[idx[part]]])
                error = float(np.max(np.abs(actual-expected))); max_error = max(error, max_error)
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
                subjects = set(frame.subject_id)
                if subjects & set(table.iloc[idx['test']].subject_id): raise ValueError('Source/test participant overlap')
        del model, normalized
        print(json.dumps({'verified': len(records), 'id': r['id'], 'maximum_probability_error': max_error}), flush=True)
    result = {'passed': True, 'complete': not missing, 'fits_checked': len(records), 'fits_missing': missing,
              'probability_metric_sets_checked': checked_points, 'checkpoint_states_replayed': replayed_states,
              'prior_prefix_checks': prefix_checks, 'maximum_neural_probability_error': max_error if runs is not None else None,
              'source_sha256': sha256(Path(__file__)), 'fitting_plan_sha256': sha256(output/'plan.json'),
              'input_digest': digest(input_hashes), 'source_only': True,
              'scope': 'Independent probability metrics, complete trajectories/selection, artifact/source bindings, declared source/test row boundaries and paired initializations/draws.'+
                       (' Original cache/split/scaler reconstruction and final/selected checkpoint inference replay.' if runs is not None else
                        ' Public probabilities only; no waveform/scaler/model inference replay.')+
                       ' Neither full optimization-trajectory rerun nor complete outer-test confirmation.'}
    if not partial:
        filename = 'verification.json' if runs is not None else 'public_verification.json'
        write_json(output/filename, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); root = args.root.resolve()
    if args.export_only and root == (REPO.parent/'publication_runs').resolve(): root = REPO/'results/development'
    verify(root/'learning_controls_2026-10-06', None if args.export_only else root, args.device, args.partial)
