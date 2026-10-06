"""Independently check source-report means from probabilities, not stored metrics."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.data import digest, sha256, write_json
from gruxnet.learning_controls import identifier
from scripts.audit_learning_controls import independent_metrics


def verify(root):
    output = root/'learning_controls_2026-10-06'
    comparison = json.loads((output/'comparison.json').read_text())
    declaration = json.loads((output/'plan.json').read_text())
    rows = []; hashes = {}; capacity_passes = 0
    for job in declaration['jobs']:
        folder = output/'fits'/identifier(job); record = json.loads((folder/'record.json').read_text())
        dataset = job['dataset']; task = 'coarse3' if dataset == 'SEEDIV' else 'binary'
        if job['kind'] == 'memorization':
            path = folder/'predictions_train_step400.csv'; scores = independent_metrics(pd.read_csv(path), dataset)[task]
            capacity_passes += int(scores['balanced_accuracy'] >= .95 and scores['balanced_log_loss'] < .15)
            hashes[str(path.relative_to(output))] = sha256(path); continue
        row = {'dataset': dataset, 'model': job['model'], 'lr': job['lr'], 'selected_step': record['selected_step']}
        for step in (200, 600, 1200):
            for role in ('train', 'validation'):
                path = folder/f'predictions_{role}_step{step}.csv'
                scores = independent_metrics(pd.read_csv(path), dataset)[task]
                row[f'{role}_BA_{step}'] = scores['balanced_accuracy']
                row[f'{role}_logloss_{step}'] = scores['balanced_log_loss']
                hashes[str(path.relative_to(output))] = sha256(path)
        rows.append(row)
    df = pd.DataFrame(rows); checked = 0
    if comparison['fits'] != 96 or comparison['memorization_criterion_passes'] != capacity_passes:
        raise ValueError('Reported study counts disagree')
    for (dataset, model, lr), part in df.groupby(['dataset', 'model', 'lr']):
        expected = comparison['means'][dataset][model][str(lr)]
        if expected['source_panels'] != len(part) or expected['selected_after_200'] != int(part.selected_step.gt(200).sum()):
            raise ValueError('Reported panel/selection counts disagree')
        for step in (200, 600, 1200):
            for role, prefix in (('train', 'train'), ('validation', 'val')):
                for field in ('BA', 'logloss') if role == 'validation' else ('BA',):
                    actual = part[f'{role}_{field}_{step}'].mean()
                    np.testing.assert_allclose(actual, expected[f'mean_{prefix}_{field}_{step}'], atol=1e-7, rtol=0)
                    checked += 1
    result = {'passed': True, 'aggregate_point_metrics_recomputed': checked, 'complete_source_fits': len(rows),
              'memorization_fits': 16, 'capacity_criterion_passes': capacity_passes,
              'source_sha256': sha256(Path(__file__)), 'input_digest': digest(hashes),
              'scope': 'All descriptive source-report aggregate BA/logloss means independently computed from original probability tables; no population inference or EEG/checkpoint replay.'}
    write_json(output/'report_verification.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'results/development')
    args = parser.parse_args(); verify(args.root.resolve())
