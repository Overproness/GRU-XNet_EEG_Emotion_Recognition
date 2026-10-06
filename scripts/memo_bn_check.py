"""Post-hoc source-only BN check on all sixteen class-balanced tiny batches."""
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
from scripts.source_bn_diagnostic import calibrate
from scripts.audit_learning_controls import predict, independent_metrics, compare_metrics


def plan(root):
    output = root/'memo_bn_check_2026-10-06'
    if output.exists(): raise FileExistsError('Do not replace declared tiny-batch diagnostic')
    upstream = root/'learning_controls_2026-10-06'
    study = json.loads((upstream/'plan.json').read_text())
    known = [f.parent.name for f in sorted((upstream/'fits').glob('*/record.json'))]
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'post_hoc': True,
              'reason': 'EEGNet SEED tiny-batch training-mode loss0.003/0.011 but inference loss6.97/3.60 inspected before declaration. Test source-population BN without a class-prior change because the twelve training trials are already balanced. Other models/labels/corpora retained, no outer-test access.',
              'known_completed_fits': known, 'upstream_plan_sha256': sha256(upstream/'plan.json'),
              'jobs': [identifier(j) for j in study['jobs'] if j['kind'] == 'memorization'],
              'source_sha256': {s: sha256(REPO/s) for s in ('scripts/memo_bn_check.py', 'scripts/source_bn_diagnostic.py',
                               'scripts/audit_learning_controls.py', 'gruxnet/learning_controls.py',
                               'gruxnet/eegnet_control.py', 'gruxnet/full_context_models_v2.py')},
              'procedure': 'All16 final tiny-batch states, same12 balanced source trials/targets and input normalizer. Use frozen three-pass eval-mode float64 source-moment function, no dropout, gradients, affine-weight change, optimizer or checkpoint reselection. Retain original and recalibrated full-source probabilities/moments/metrics and criterion95%BA plusloss<.15.',
              'scope': 'Explicit post-hoc capacity/evaluation-state diagnostic; does not separate EMA drift from dropout-distribution shift, establish emotion information, convergence or held-out prediction. No research-question change.',
              'research_question_change_approved': False}
    output.mkdir(); write_json(output/'plan.json', result)
    print(json.dumps({'post_hoc': True, 'tiny_batches': len(result['jobs']), 'known_fits': len(known)}))


def run(root, verify=False):
    seed_everything(20261006)
    output = root/'memo_bn_check_2026-10-06'; upstream = root/'learning_controls_2026-10-06'
    declaration = json.loads((output/'plan.json').read_text())
    if sha256(upstream/'plan.json') != declaration['upstream_plan_sha256']: raise ValueError('Source plan changed')
    for s, expected in declaration['source_sha256'].items():
        if sha256(REPO/s) != expected: raise ValueError('Diagnostic source changed')
    if not json.loads((upstream/'verification.json').read_text())['complete']: raise ValueError('Complete upstream replay required')
    records = []; maximum = 0.
    for dataset in ('SEEDIV', 'DEAP'):
        x, base, _ = load(root/f'cache_full_context_{dataset.lower()}')
        wave, _ = raw_inputs(dataset, root, base)
        positions = {t: i for i, t in enumerate(base.trial_id)}
        for name in declaration['jobs']:
            if f'_{dataset.lower()}_' not in name: continue
            source = upstream/'fits'/name; folder = output/'fits'/name
            original = json.loads((source/'record.json').read_text()); job = original['job']
            checkpoint = torch.load(source/'final.pt', map_location='cpu', weights_only=False)
            rows = pd.read_csv(source/'predictions_train_step400.csv')
            if rows.trial_id.tolist() != original['split_trials']['train']: raise ValueError('Changed tiny population')
            if set(rows.trial_id) & set(original['split_trials']['test']): raise ValueError('Outer test entered capacity check')
            selected = np.array([positions[t] for t in rows.trial_id], dtype=int)
            values = (wave if job['model'] == 'eegnet' else x)[selected]
            inputs = torch.from_numpy(((values-checkpoint['mean'])/checkpoint['scale']).astype(np.float32)).cuda()
            classes = 3 if dataset == 'SEEDIV' else 2; task = 'coarse3' if dataset == 'SEEDIV' else 'binary'
            model = make_model(job['model'], classes).cuda(); model.load_state_dict(checkpoint['state'])
            parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            p = predict(model, inputs); expected = rows[[f'p{c}' for c in range(classes)]].to_numpy()
            maximum = max(maximum, float(np.max(np.abs(p-expected))))
            np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
            before = independent_metrics(rows, dataset)
            compare_metrics(before, original['checkpoints']['400']['training'])
            stats = calibrate(model, inputs)
            arrays = {f'{n}:{field}': r[field] for n, r in stats.items() for field in ('mean', 'variance')}
            for n, parameter in model.named_parameters():
                if not torch.equal(parameters[n], parameter.detach().cpu()): raise ValueError('A learned weight changed')
            after_rows = rows.copy(); p = predict(model, inputs)
            for c in range(classes): after_rows[f'p{c}'] = p[:, c]
            if verify:
                saved = np.load(folder/'moments.npz', allow_pickle=False)
                for n, value in arrays.items(): np.testing.assert_array_equal(value, saved[n])
                stored = pd.read_csv(folder/'predictions_train.csv')
                expected = stored[[f'p{c}' for c in range(classes)]].to_numpy()
                maximum = max(maximum, float(np.max(np.abs(p-expected))))
                np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
                after_rows = stored
            else:
                folder.mkdir(parents=True, exist_ok=True); np.savez(folder/'moments.npz', **arrays)
                after_rows.to_csv(folder/'predictions_train.csv', index=False)
            after = independent_metrics(after_rows, dataset)
            result = {'id': name, 'job': job, 'before': before, 'after': after,
                      'criterion_before': bool(before[task]['balanced_accuracy'] >= .95 and before[task]['balanced_log_loss'] < .15),
                      'criterion_after': bool(after[task]['balanced_accuracy'] >= .95 and after[task]['balanced_log_loss'] < .15),
                      'parameters_unchanged': True, 'source_trials': rows.trial_id.tolist(),
                      'source_record_sha256': sha256(source/'record.json'), 'plan_sha256': sha256(output/'plan.json'),
                      'artifact_sha256': {p: sha256(folder/p) for p in ('moments.npz', 'predictions_train.csv')}}
            if verify:
                if result != json.loads((folder/'record.json').read_text()): raise ValueError('Capacity diagnosis replay changed')
            else: write_json(folder/'record.json', result)
            records.append(result); del inputs, model, parameters
    if len(records) != 16: raise ValueError('Incomplete tiny-batch control')
    if verify: write_json(output/'verification.json', {'passed': True, 'source_moments_exact': True, 'cases': 16,
                      'maximum_probability_error': maximum, 'scope': 'All tiny-batch moments reconstructed exactly, learned weights unchanged, original/after probabilities replayed; no validation/test evaluation or full optimization rerun.'})
    else: write_json(output/'records.json', records)
    print(json.dumps({'cases': 16, 'verified': verify, 'criterion_before': sum(r['criterion_before'] for r in records),
                      'criterion_after': sum(r['criterion_after'] for r in records)}))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__); parser.add_argument('command', choices=('plan', 'run', 'verify'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    args = parser.parse_args()
    if args.command == 'plan': plan(args.root.resolve())
    else: run(args.root.resolve(), args.command == 'verify')
