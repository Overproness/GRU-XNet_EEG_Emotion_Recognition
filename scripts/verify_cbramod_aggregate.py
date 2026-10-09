"""Check aggregate export integrity, disclosure scope and all selected contrasts."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_probe import sha, atomic


def inspect(value):
    if isinstance(value, dict):
        if {'trial_id', 'subject_id', 'original_label', 'coef', 'intercept', 'p0', 'p1', 'p2'} & set(value):
            raise ValueError('Row-level identity/probability/parameter field in aggregate export')
        for item in value.values(): inspect(item)
    elif isinstance(value, list):
        for item in value: inspect(item)


def main():
    folder = REPO/'results/development/cbramod_source_probe_aggregate_2026-10-09'
    manifest = json.loads((folder/'export_manifest.json').read_text())
    for item in manifest['files']:
        target = (folder/item['file']).resolve()
        if not target.is_relative_to(folder.resolve()) or sha(target) != item['sha256']:
            raise ValueError('Changed/escaped aggregate artifact')
        if target.suffix == '.json': inspect(json.loads(target.read_text()))
        if target.suffix == '.csv' and {'trial_id', 'subject_id', 'p0', 'p1', 'p2', 'original_label'} & set(pd.read_csv(target).columns):
            raise ValueError('Individual row fields in CSV')
    data = pd.read_csv(folder/'postfit_analysis/selected_metrics.csv')
    if len(data) != 72 or data.duplicated(['dataset', 'group', 'model', 'role']).any():
        raise ValueError('Incomplete/duplicate selected metric coverage')
    pivot = data.set_index(['dataset', 'group', 'model', 'role'])
    contrasts = json.loads((folder/'postfit_analysis/contrasts.json').read_text())['contrasts']
    if len(contrasts) != 96: raise ValueError('Incomplete contrast count')
    maximum = 0.
    for item in contrasts:
        a = pivot.loc[(item['dataset'], item['group'], item['model'], item['role']), item['metric']]
        b = pivot.loc[(item['dataset'], item['group'], item['reference'], item['role']), item['metric']]
        maximum = max(maximum, abs(float(a-b)-item['difference']))
    if maximum > 2e-12: raise ValueError('Aggregate contrast mismatch')
    heads = json.loads((folder/'all_candidate_metrics.json').read_text())['heads']
    if len(heads) != 24: raise ValueError('Incomplete candidate head coverage')
    for head in heads:
        keys = []
        for item in head['candidates']:
            primary = np.mean([item['metrics'][role]['balanced_log_loss'] for role in ('validation_unseen', 'validation_familiar')])
            secondary = -np.mean([item['metrics'][role]['balanced_accuracy'] for role in ('validation_unseen', 'validation_familiar')])
            keys.append((primary, secondary, item['id']))
        if min(keys)[2] != head['selected_id']:
            raise ValueError('Public candidate selection mismatch')
    result = {'complete': True, 'aggregate_files_verified': len(manifest['files']),
              'disallowed_individual_fields_present': False, 'public_metric_sets': 72,
              'public_selections_rechecked': 24, 'contrast_points_rechecked': 96,
              'max_contrast_abs': maximum, 'verification_source_sha256': sha(Path(__file__)),
              'scope': 'Aggregate byte integrity, obvious row-field exclusion, source selection and paired point arithmetic. No trial-level public reanalysis or certification that arbitrary renamed data fields cannot be identifying.'}
    atomic(folder/'publication_verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
