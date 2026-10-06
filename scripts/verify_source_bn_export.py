"""Recompute every source-normalization contrast from exported probabilities."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.data import digest, sha256, write_json
from scripts.audit_learning_controls import independent_metrics, compare_metrics


def analyze(root, write=True):
    output = root/'source_bn_diagnostic_2026-10-06'
    upstream = root/'learning_controls_2026-10-06'
    plan = json.loads((output/'plan.json').read_text())
    if sha256(upstream/'plan.json') != plan['upstream_plan_sha256']: raise ValueError('Changed source plan')
    for name, expected in plan['source_sha256'].items():
        if sha256(REPO/name) != expected: raise ValueError('Changed diagnostic source')
    verification = json.loads((output/'verification.json').read_text())
    if not verification['passed'] or verification['evaluations'] != 160: raise ValueError('Complete calibration replay required')
    rows = []; hashes = {}; metrics_checked = 0
    for job in plan['jobs']:
        folder = output/'fits'/f'{job["fit_id"]}_{job["stage"]}'
        source = upstream/'fits'/job['fit_id']
        r = json.loads((folder/'record.json').read_text())
        s = json.loads((source/'record.json').read_text())
        if r['source_record_sha256'] != sha256(source/'record.json') or r['plan_sha256'] != sha256(output/'plan.json'):
            raise ValueError('Changed evidence binding')
        if r['job'] != s['job'] or r['stage'] != job['stage'] or r['id'] != folder.name:
            raise ValueError('Wrong declared case')
        checkpoint = s['selected_checkpoint'] if job['stage'] == 'selected' else 'final.pt'
        if s['artifact_sha256'][checkpoint] != r['checkpoint_sha256']: raise ValueError('Wrong weight identity')
        if r['source_training_trials'] != s['split_trials']['train'] or not r['parameters_unchanged']:
            raise ValueError('Wrong calibration population/parameter record')
        dataset = r['job']['dataset']; task = 'coarse3' if dataset == 'SEEDIV' else 'binary'
        for part in ('train', 'validation'):
            before_path = source/(f'predictions_{part}_selected.csv' if job['stage'] == 'selected' else f'predictions_{part}_step1200.csv')
            after_path = folder/f'predictions_{part}.csv'
            if sha256(after_path) != r['artifact_sha256'][after_path.name]: raise ValueError('Changed calibration probabilities')
            before = pd.read_csv(before_path); after = pd.read_csv(after_path)
            metadata = ['trial_id', 'subject_id', 'material_key', 'label', 'original_label']
            pd.testing.assert_frame_equal(before[metadata], after[metadata], check_exact=True)
            if before.trial_id.tolist() != s['split_trials'][part] or before.trial_id.duplicated().any():
                raise ValueError('Incomplete source population')
            bm = independent_metrics(before, dataset); am = independent_metrics(after, dataset)
            compare_metrics(bm, r['before'][part]); compare_metrics(am, r['after'][part]); metrics_checked += 2
            rows.append({'id': r['id'], **r['job'], 'stage': job['stage'], 'part': part,
                         'before_BA': bm[task]['balanced_accuracy'], 'after_BA': am[task]['balanced_accuracy'],
                         'BA_difference': am[task]['balanced_accuracy']-bm[task]['balanced_accuracy'],
                         'before_logloss': bm[task]['balanced_log_loss'], 'after_logloss': am[task]['balanced_log_loss'],
                         'logloss_difference': am[task]['balanced_log_loss']-bm[task]['balanced_log_loss']})
            for path in (before_path, after_path): hashes[str(path.relative_to(root))] = sha256(path)
    if len(rows) != 320: raise ValueError('Missing normalization cases')
    result = {'development_only': True, 'post_hoc': True, 'research_question_change_approved': False,
              'cases': rows, 'scope': 'Every selected/final state, both source roles and every declared model/LR/group/initialization/arm; source-only normalization diagnostic, without test evaluation or population intervals.'}
    if write:
        write_json(output/'comparison.json', result); pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
        write_json(output/'public_verification.json', {'passed': True, 'evaluations': len(plan['jobs']),
                    'probability_metric_sets_independently_recomputed': metrics_checked,
                    'primary_point_differences_recomputed': len(rows)*2,
                    'source_sha256': sha256(Path(__file__)), 'input_digest': digest(hashes),
                    'scope': 'Source/record/probability bindings and independent probability metrics/paired changes; no raw EEG, checkpoint inference or BN-moment reconstruction.'})
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'results/development')
    args = parser.parse_args(); root = args.root.resolve()
    result = analyze(root)
    if analyze(root, write=False) != result: raise ValueError('Analysis recomputation changed')
    print(json.dumps({'passed': True, 'paired_cases': len(result['cases']), 'point_differences': len(result['cases'])*2}))
