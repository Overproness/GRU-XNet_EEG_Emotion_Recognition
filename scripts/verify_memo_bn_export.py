"""Check all tiny-batch normalization contrasts from public probabilities."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.data import digest, sha256, write_json
from gruxnet.learning_controls import identifier
from scripts.audit_learning_controls import independent_metrics, compare_metrics


def analyze(root, write=True):
    output = root/'memo_bn_check_2026-10-06'
    upstream = root/'learning_controls_2026-10-06'
    plan = json.loads((output/'plan.json').read_text())
    study = json.loads((upstream/'plan.json').read_text())
    expected_jobs = [identifier(j) for j in study['jobs'] if j['kind'] == 'memorization']
    if plan['jobs'] != expected_jobs or len(expected_jobs) != 16:
        raise ValueError('Wrong tiny-batch declaration')
    if sha256(upstream/'plan.json') != plan['upstream_plan_sha256']:
        raise ValueError('Changed source plan')
    for name, expected in plan['source_sha256'].items():
        if sha256(REPO/name) != expected: raise ValueError('Changed diagnostic source')
    proof = json.loads((output/'verification.json').read_text())
    if not proof['passed'] or not proof['source_moments_exact'] or proof['cases'] != 16:
        raise ValueError('Complete tiny-batch replay required')
    rows = []; hashes = {}; metric_sets = 0
    for name in expected_jobs:
        folder = output/'fits'/name; source = upstream/'fits'/name
        record = json.loads((folder/'record.json').read_text())
        original = json.loads((source/'record.json').read_text())
        if record['id'] != name or record['job'] != original['job']:
            raise ValueError('Wrong declared case')
        if record['source_record_sha256'] != sha256(source/'record.json') or record['plan_sha256'] != sha256(output/'plan.json'):
            raise ValueError('Changed record bindings')
        before_path = source/'predictions_train_step400.csv'
        after_path = folder/'predictions_train.csv'
        if sha256(before_path) != original['artifact_sha256'][before_path.name] or sha256(after_path) != record['artifact_sha256'][after_path.name]:
            raise ValueError('Changed tiny-batch probabilities')
        before = pd.read_csv(before_path); after = pd.read_csv(after_path)
        metadata = ['trial_id', 'subject_id', 'material_key', 'label', 'original_label']
        pd.testing.assert_frame_equal(before[metadata], after[metadata], check_exact=True)
        if before.trial_id.tolist() != original['split_trials']['train'] or before.trial_id.tolist() != record['source_trials']:
            raise ValueError('Wrong capacity population')
        if len(before) != 12 or before.trial_id.duplicated().any() or set(before.trial_id) & set(original['split_trials']['test']):
            raise ValueError('Wrong source-only membership')
        counts = before.label.value_counts()
        if counts.nunique() != 1 or len(counts) != (3 if record['job']['dataset'] == 'SEEDIV' else 2):
            raise ValueError('Tiny batch is not class balanced')
        if not record['parameters_unchanged']: raise ValueError('Learned parameter-change record')
        dataset = record['job']['dataset']; task = 'coarse3' if dataset == 'SEEDIV' else 'binary'
        bm = independent_metrics(before, dataset); am = independent_metrics(after, dataset)
        compare_metrics(bm, record['before']); compare_metrics(am, record['after']); metric_sets += 2
        criterion = lambda m: bool(m[task]['balanced_accuracy'] >= .95 and m[task]['balanced_log_loss'] < .15)
        if criterion(bm) != record['criterion_before'] or criterion(am) != record['criterion_after']:
            raise ValueError('Wrong strict capacity criterion')
        rows.append({'id': name, **record['job'], 'before_BA': bm[task]['balanced_accuracy'],
                     'after_BA': am[task]['balanced_accuracy'],
                     'before_logloss': bm[task]['balanced_log_loss'], 'after_logloss': am[task]['balanced_log_loss'],
                     'criterion_before': criterion(bm), 'criterion_after': criterion(am)})
        for path in (before_path, after_path, folder/'record.json', source/'record.json'):
            hashes[str(path.relative_to(root))] = sha256(path)
    result = {'post_hoc': True, 'source_only': True, 'cases': rows,
              'criterion_before': sum(r['criterion_before'] for r in rows),
              'criterion_after': sum(r['criterion_after'] for r in rows),
              'scope': 'All sixteen class-balanced tiny training populations, both label modes and corpora; unchanged learned weights. No validation/test inference, physiological information or convergence claim.'}
    if write:
        write_json(output/'comparison.json', result)
        pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
        write_json(output/'public_verification.json', {'passed': True, 'cases': 16,
                   'probability_metric_sets_independently_recomputed': metric_sets,
                   'source_sha256': sha256(Path(__file__)), 'input_digest': digest(hashes),
                   'scope': 'Public source, membership, record and probability bindings with independent metrics/capacity criteria; moment-array reconstruction and checkpoint replay require local data/weights.'})
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'results/development')
    args = parser.parse_args(); root = args.root.resolve()
    result = analyze(root)
    if analyze(root, write=False) != result: raise ValueError('Recomputation changed')
    print(json.dumps({'passed': True, 'cases': len(result['cases']),
                      'criterion_before': result['criterion_before'], 'criterion_after': result['criterion_after']}))
