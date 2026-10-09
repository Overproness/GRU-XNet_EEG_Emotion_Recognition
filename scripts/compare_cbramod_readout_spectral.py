"""Recheck existing spectral selections on exactly the readout study's panels."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_learning import sha, read, write, criterion
from scripts.analyze_cbramod_readout import measures
from scripts.analyze_cbramod_readout_boundary_adapter import api, PUBLIC

SPECTRAL = REPO/'results/development/cbramod_adaptation_2026-10-09'
MODELS = ('band_absolute', 'band_relative')
ROLES = ('train', 'validation_unseen', 'validation_familiar')


def collect():
    api()['verify'](PUBLIC)
    rows = pd.read_csv(PUBLIC/'postfit_analysis/all_metrics.csv')
    selected = pd.read_csv(PUBLIC/'postfit_analysis/selected_metrics.csv')
    measures_rows = []; points = []; inputs = {}; maximum = 0.
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            anchor = PUBLIC/'runs'/f'{dataset.lower()}_g{group}_pretrained_pooled_linear'
            for model in MODELS:
                folder = SPECTRAL/'linear'/f'{dataset.lower()}_g{group}_{model}'
                record, audit = read(folder/'record.json'), read(folder/'verification.json')
                if not audit['complete'] or audit['record_sha256'] != sha(folder/'record.json') or record['plan_sha256'] != sha(SPECTRAL/'plan.json'):
                    raise ValueError('Unverified spectral selection')
                choice = record['candidates'][criterion(record['candidates'])]
                if choice['id'] != record['selected_id'] or choice['C'] != record['selected_C']:
                    raise ValueError('Existing spectral source selection differs')
                for name in ('record.json', 'verification.json', *[f'{r}.csv' for r in ROLES]):
                    path = folder/name; inputs[path.relative_to(REPO).as_posix()] = sha(path)
                for role in ROLES:
                    frame = pd.read_csv(folder/f'{role}.csv')
                    original = pd.read_csv(anchor/f'step0_{role}.csv')
                    metadata = ['trial_id', 'subject_id', 'material_key', 'original_label', 'label', 'role']
                    pd.testing.assert_frame_equal(frame[metadata], original[metadata])
                    classes = 2 if dataset == 'DEAP' else 3
                    p = frame[[f'p{c}' for c in range(classes)]].to_numpy(); y = frame.label.to_numpy(dtype=int)
                    np.testing.assert_allclose(p.sum(1), 1, rtol=0, atol=2e-12)
                    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                        raise ValueError('Invalid spectral probabilities')
                    score = measures(y, p)
                    for name, value in score.items():
                        error = abs(value-record['metrics'][role][name]); maximum = max(maximum, error)
                        if error > 2e-11:
                            raise ValueError('Existing spectral metric differs')
                    measures_rows.append(dict(dataset=dataset, group=group, model=model, role=role,
                        selected_C=record['selected_C'], **score))
                    if role == 'train':
                        continue
                    for scope, source in (('fixed_step', rows[rows.step > 0]), ('selected_secondary', selected)):
                        block = source[(source.dataset == dataset)&(source.group == group)&(source.role == role)]
                        for _, row in block.iterrows():
                            for metric in ('balanced_log_loss', 'balanced_accuracy'):
                                points.append({'scope': scope, 'dataset': dataset, 'group': group, 'head': row['head'],
                                    'pretrained': bool(row.pretrained), 'step': int(row.step), 'role': role,
                                    'reference': model, 'metric': metric, 'delta': float(row[metric]-score[metric])})
    if len(measures_rows) != 24 or len(points) != 768:
        raise ValueError('Incomplete spectral comparison coverage')
    return pd.DataFrame(measures_rows), points, inputs, maximum


def generate():
    rows, points, inputs, maximum = collect()
    output = REPO.parent/'publication_runs'/PUBLIC.name/'spectral_reference'
    destination = PUBLIC/'spectral_reference'
    if output.exists() or destination.exists():
        raise FileExistsError('Preserve spectral comparison')
    output.mkdir()
    write(output/'declaration.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(Path(__file__)), 'input_sha256': inputs,
        'neural_analysis_sha256': sha(PUBLIC/'postfit_analysis/verification.json'),
        'scope': 'Descriptive comparison to eight existing source-selected absolute/relative spectral logistic heads, exactly matching metadata. Different representation/optimizer/search budget; not a pure architecture contrast or new task fitting.'})
    rows.to_csv(output/'spectral_metrics.csv', index=False)
    write(output/'contrasts.json', {'contrasts': points})
    write(output/'verification.json', {'complete': True, 'declaration_sha256': sha(output/'declaration.json'),
        'spectral_heads': 8, 'metric_sets': 24, 'fixed_step_contrasts': 576, 'selected_secondary_contrasts': 192,
        'maximum_abs_metric_discrepancy': maximum,
        'artifact_sha256': {p.name: sha(p) for p in output.iterdir()}})
    destination.mkdir()
    for path in output.iterdir():
        shutil.copyfile(path, destination/path.name)
    manifest = read(PUBLIC/'export_manifest.json')
    for item in manifest['files']:
        if sha(PUBLIC/item['file']) != item['sha256']:
            raise ValueError('Changed neural export')
    manifest['files'] += [{'file': p.relative_to(PUBLIC).as_posix(), 'sha256': sha(p)} for p in sorted(destination.iterdir())]
    write(PUBLIC/'export_manifest.json', manifest)
    verify()


def verify():
    folder = PUBLIC/'spectral_reference'; declaration = read(folder/'declaration.json'); proof = read(folder/'verification.json')
    if not proof['complete'] or proof['declaration_sha256'] != sha(folder/'declaration.json') or declaration['source_sha256'] != sha(Path(__file__)):
        raise ValueError('Changed spectral comparison binding')
    for name, checksum in declaration['input_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError('Changed spectral reference input')
    for name, checksum in proof['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Changed spectral comparison output')
    rows, points, _, maximum = collect()
    pd.testing.assert_frame_equal(pd.read_csv(folder/'spectral_metrics.csv'), rows, check_dtype=False, rtol=2e-11, atol=2e-11)
    if read(folder/'contrasts.json')['contrasts'] != points:
        raise ValueError('Changed spectral comparison points')
    print(json.dumps({'passed': True, 'spectral_heads': 8, 'metric_sets': 24, 'contrasts': 768,
        'maximum_abs_metric_discrepancy': maximum}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('generate', 'verify'))
    args = parser.parse_args(); (generate if args.action == 'generate' else verify)()
