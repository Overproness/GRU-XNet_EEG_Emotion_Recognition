"""Declared post-hoc source-only population BatchNorm recalibration diagnostic."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.data import digest, sha256, write_json
from gruxnet.learning_controls import identifier, raw_inputs, make_model
from gruxnet.full_context_controls_v2 import load
from gruxnet.train import seed_everything
from scripts.audit_learning_controls import predict, independent_metrics, compare_metrics


@torch.no_grad()
def calibrate(model, training):
    """Three sequential passes: true pooled source moments, no dropout/update."""
    model.eval()
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, torch.nn.BatchNorm2d)]
    if len(layers) != 3: raise ValueError('Expected three reference BatchNorm layers')
    records = {}
    for name, layer in layers:
        total = squares = None; count = 0
        def collect(module, inputs):
            nonlocal total, squares, count
            values = inputs[0].detach().to(torch.float64)
            reduced = values.sum((0, 2, 3)); second = values.square().sum((0, 2, 3))
            total = reduced if total is None else total+reduced
            squares = second if squares is None else squares+second
            count += values.shape[0]*values.shape[2]*values.shape[3]
        hook = layer.register_forward_pre_hook(collect)
        try:
            for start in range(0, len(training), 12): model(training[start:start+12])
        finally:
            hook.remove()
        mean = total/count; variance = torch.clamp(squares/count-mean.square(), min=1e-12)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all(): raise ValueError('Invalid pooled moments')
        layer.running_mean.copy_(mean.float()); layer.running_var.copy_(variance.float())
        records[name] = {'observations_per_feature': count,
                         'mean': layer.running_mean.cpu().numpy(), 'variance': layer.running_var.cpu().numpy()}
    return records


def declare(root):
    output = root/'source_bn_diagnostic_2026-10-06'
    if output.exists(): raise FileExistsError('Do not replace the diagnostic declaration')
    upstream = root/'learning_controls_2026-10-06'
    declaration = json.loads((upstream/'plan.json').read_text())
    known = [{'id': folder.name, 'record_sha256': sha256(folder/'record.json')}
             for folder in sorted((upstream/'fits').iterdir()) if (folder/'record.json').exists()]
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'post_hoc': True,
              'reason': 'Declared after inspecting early source training/validation curves showing fluctuating evaluation-mode training accuracy and worsening validation log loss. No outer-test outcomes consulted. Tests whether population BN moments contribute; no claim of original pre-specification.',
              'known_completed_fits_at_declaration': known,
              'upstream_plan_sha256': sha256(upstream/'plan.json'),
              'source_sha256': {f: sha256(REPO/f) for f in ('scripts/source_bn_diagnostic.py', 'scripts/audit_learning_controls.py',
                               'gruxnet/learning_controls.py', 'gruxnet/eegnet_control.py', 'gruxnet/full_context_models_v2.py')},
              'evaluations': 160,
              'jobs': [{'fit_id': identifier(j), 'stage': stage} for j in declaration['jobs']
                       if j['kind'] == 'source_curve' for stage in ('selected', 'final')],
              'procedure': 'Keep all learned parameters and upstream selected checkpoint unchanged. Disable dropout/eval mode throughout. For three BN layers in feed-forward order, pass only all actual source training rows and collect pre-BN activation sums/squares in float64 across rows/spatial/time. Set running mean and population variance(E[x2]-E[x]^2, floor1e-12). Later layers see earlier recalibrated moments. Every original trial has equal weight; use original possibly imbalanced source-trial population, not balanced sampled batches. Do not update affine parameters, optimizer, labels, scalers or checkpoint selection; no validation/test inputs enter moments.',
              'evaluation': 'Compare source-train and source-validation BA/logloss before/after separately for each selected/final state, retaining every model/LR/group/initialization/arm. Save changed BN moments, full probabilities and source-trial identities. No outer-test evaluation, model preference or new source-global recipe chosen.',
              'verification': 'Verify learned parameters unchanged and originals reproduced; independently reconstruct all source moments, require bit-exact float32 stored moments and replay all after-calibration probabilities. Compute metrics independently. Public checks validate probabilities/record contrasts without EEG/weights; not full calibration reconstruction.',
              'inference': 'Exploratory diagnostic conditional on source panels; no population interval, optimization convergence, novel BN method or test-generalization claim. Broad confirmation needs a separate fold-valid declaration.',
              'research_question_change_approved': False}
    output.mkdir(); write_json(output/'plan.json', result)
    print(json.dumps({'declared_evaluations': 160, 'known_source_fits': len(known), 'post_hoc': True}))


def run(root, verify=False):
    seed_everything(20261006)
    output = root/'source_bn_diagnostic_2026-10-06'; upstream = root/'learning_controls_2026-10-06'
    declaration = json.loads((output/'plan.json').read_text())
    if sha256(upstream/'plan.json') != declaration['upstream_plan_sha256']: raise ValueError('Changed original study')
    for source, expected in declaration['source_sha256'].items():
        if sha256(REPO/source) != expected: raise ValueError('Changed diagnostic source')
    if not json.loads((upstream/'verification.json').read_text())['complete']: raise ValueError('Complete upstream replay required')
    records = []; max_error = 0.
    for dataset in ('SEEDIV', 'DEAP'):
        x, base, info = load(root/f'cache_full_context_{dataset.lower()}')
        wave, _ = raw_inputs(dataset, root, base)
        positions = {trial: i for i, trial in enumerate(base.trial_id)}
        jobs = [j for j in declaration['jobs'] if f'_{dataset.lower()}_' in j['fit_id']]
        for j in jobs:
            folder = output/'fits'/f'{j["fit_id"]}_{j["stage"]}'; source = upstream/'fits'/j['fit_id']
            record_path = folder/'record.json'
            if record_path.exists() and not verify:
                r = json.loads(record_path.read_text())
                if r['plan_sha256'] != sha256(output/'plan.json') or r['source_record_sha256'] != sha256(source/'record.json'):
                    raise ValueError('Changed resumed diagnostic')
                records.append(r); continue
            original = json.loads((source/'record.json').read_text()); job = original['job']
            classes = 3 if dataset == 'SEEDIV' else 2
            selected = j['stage'] == 'selected'
            filename = original['selected_checkpoint'] if selected else 'final.pt'
            checkpoint = torch.load(source/filename, map_location='cpu', weights_only=False)
            indices = {p: np.array([positions[t] for t in original['split_trials'][p]]) for p in ('train', 'validation')}
            allowed = np.sort(np.concatenate(list(indices.values())))
            remap = {int(v): k for k, v in enumerate(allowed)}
            values = (wave if job['model'] == 'eegnet' else x)[allowed]
            inputs = torch.from_numpy(((values-checkpoint['mean'])/checkpoint['scale']).astype(np.float32)).cuda()
            model = make_model(job['model'], classes).cuda(); model.load_state_dict(checkpoint['state'])
            parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            before = {}
            for part in ('train', 'validation'):
                path = source/(f'predictions_{part}_selected.csv' if selected else 'predictions_'+part+'_step1200.csv')
                rows = pd.read_csv(path)
                p = predict(model, inputs[[remap[int(i)] for i in indices[part]]])
                expected = rows[[f'p{c}' for c in range(classes)]].to_numpy()
                error = float(np.max(np.abs(p-expected))); max_error = max(error, max_error)
                np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
                before[part] = independent_metrics(rows, dataset)
            statistics = calibrate(model, inputs[[remap[int(i)] for i in indices['train']]])
            for name, parameter in model.named_parameters():
                if not torch.equal(parameter.detach().cpu(), parameters[name]): raise ValueError('Learned parameter was changed')
            arrays = {f'{name}:{field}': item[field] for name, item in statistics.items() for field in ('mean', 'variance')}
            counts = {name: item['observations_per_feature'] for name, item in statistics.items()}
            after = {}
            if not verify: folder.mkdir(parents=True, exist_ok=True)
            if verify:
                stored = np.load(folder/'moments.npz', allow_pickle=False)
                if set(stored.files) != set(arrays): raise ValueError('Moment keys changed')
                for name, value in arrays.items(): np.testing.assert_array_equal(value, stored[name])
            else:
                np.savez(folder/'moments.npz', **arrays)
            for part in ('train', 'validation'):
                metadata = pd.read_csv(source/(f'predictions_{part}_selected.csv' if selected else f'predictions_{part}_step1200.csv'))
                p = predict(model, inputs[[remap[int(i)] for i in indices[part]]])
                for c in range(classes): metadata[f'p{c}'] = p[:, c]
                path = folder/f'predictions_{part}.csv'
                if verify:
                    saved = pd.read_csv(path)
                    expected = saved[[f'p{c}' for c in range(classes)]].to_numpy()
                    error = float(np.max(np.abs(p-expected))); max_error = max(error, max_error)
                    np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
                    metadata = saved
                else: metadata.to_csv(path, index=False)
                after[part] = independent_metrics(metadata, dataset)
            result = {'id': folder.name, 'job': job, 'stage': j['stage'], 'plan_sha256': sha256(output/'plan.json'),
                      'source_record_sha256': sha256(source/'record.json'), 'checkpoint_sha256': sha256(source/filename),
                      'source_training_trials': original['split_trials']['train'], 'moments_population_counts': counts,
                      'parameters_unchanged': True, 'before': before, 'after': after,
                      'artifact_sha256': {name: sha256(folder/name) for name in ('moments.npz', 'predictions_train.csv', 'predictions_validation.csv')}}
            if verify:
                if result != json.loads(record_path.read_text()): raise ValueError('Diagnostic record failed exact replay')
            else: write_json(record_path, result)
            records.append(result)
            del inputs, model, parameters
            print(json.dumps({'verified' if verify else 'completed': len(records), 'total': 160, 'id': result['id']}), flush=True)
        del x, wave
    if len(records) != declaration['evaluations']: raise ValueError('Incomplete diagnostic')
    if verify:
        write_json(output/'verification.json', {'passed': True, 'evaluations': len(records),
                                               'maximum_probability_error': max_error, 'source_moments_exact': True,
                                               'scope': 'All selected/final source BN moments reconstructed, frozen learned parameters checked, all source train/validation predictions replayed; no outer-test or optimization-trajectory replay.'})
    else: write_json(output/'records.json', records)
    return records


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'verify'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    args = parser.parse_args()
    if args.command == 'plan': declare(args.root.resolve())
    else: run(args.root.resolve(), args.command == 'verify')
