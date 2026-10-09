"""Publishable aggregate-only evidence; row predictions/participant IDs excluded.

An automatic approval review rejected GitHub egress of row-level evidence.
This exporter is a safer publication scope, not permission to push those rows.
"""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_probe import STUDY, ASSETS, ROLES, sha, atomic, stamp

DEST = 'cbramod_source_probe_aggregate_2026-10-09'


def export():
    output = REPO.parent/'publication_runs'/STUDY
    proof = json.loads((output/'verification.json').read_text())
    if not proof['complete']:
        raise ValueError('Complete verified study required')
    destination = REPO/'results/development'/DEST
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    # These files contain only model/configuration/aggregate metric information.
    for name in ('plan.json', 'progress.json', 'config.json', 'summary.json', 'verification.json'):
        shutil.copyfile(output/name, destination/name)
    postfit = output/'postfit_analysis'
    postdest = destination/'postfit_analysis'; postdest.mkdir(parents=True, exist_ok=True)
    for name in ('selected_metrics.csv', 'contrasts.json', 'cbramod_source_controls.png',
                 'cbramod_source_controls.svg', 'verification.json'):
        shutil.copyfile(postfit/name, postdest/name)
    # The detailed analysis declaration remains local: its input file hashes
    # include the withheld row-level export. Bind it without releasing rows.
    atomic(postdest/'scope.json', {'created_utc': stamp(),
        'local_declaration_sha256': sha(postfit/'declaration.json'),
        'analysis_source_sha256': sha(REPO/'scripts/analyze_cbramod_source_probe.py'),
        'scope': 'Aggregate metrics, all96paired descriptive contrasts and standalone figures only; full local point reanalysis additionally used withheld predictions.'})
    candidates = []
    for folder in sorted((output/'fits').iterdir()):
        record = json.loads((folder/'record.json').read_text())
        candidates.append({'job': record['job'], 'selected_id': record['selected_id'],
                           'selected_C': record['selected_C'], 'candidates': record['candidates'],
                           'local_record_sha256': sha(folder/'record.json'),
                           'local_verification_sha256': sha(folder/'verification.json')})
    atomic(destination/'all_candidate_metrics.json', {'heads': candidates,
        'excludes': 'Per-trial probability arrays, features, fitted coefficients, participant identities'})
    amplitude = {}
    for dataset in ('deap', 'seediv'):
        cache = output/'inputs'/dataset
        inputs = json.loads((cache/'prepared.json').read_text())
        features = json.loads((cache/'features.json').read_text())
        data = pd.read_csv(cache/'amplitude_statistics.csv')
        amplitude[dataset] = {'source_union_trials': len(data), 'channels': len(inputs['channels']),
            'physical_calibration_independently_authenticated': False,
            'rms_numeric_min_median_max': [float(data.rms_numeric.min()), float(data.rms_numeric.median()), float(data.rms_numeric.max())],
            'median_fraction_abs_over100': float(data.fraction_abs_over100.median()),
            'encoder_states': features['encoder_states'],
            'local_prepared_sha256': sha(cache/'prepared.json'), 'local_features_sha256': sha(cache/'features.json')}
    atomic(destination/'input_aggregate.json', amplitude)
    for path in destination.rglob('*'):
        if path.is_file() and path.name != 'export_manifest.json':
            records.append({'file': path.relative_to(destination).as_posix(), 'sha256': sha(path)})
    atomic(destination/'export_manifest.json', {'files': records,
        'scope': 'Aggregate-only authorized publication alternative following review rejection of trial-level egress.',
        'withheld': 'EEG, embeddings, trial-level probability exports, participant/video-linked metadata, fitted coefficients, full author source and weights remain local.',
        'independent_reproduction_limit': 'Aggregate calculations and declared candidate selection can be inspected publicly; full trial probability reanalysis requires the retained local study or fresh separately authorized reproduction.'})
    for item in records:
        if sha(destination/item['file']) != item['sha256']: raise ValueError('Aggregate export mismatch')
    print(json.dumps({'aggregate_files': len(records), 'destination': DEST,
                      'trial_probability_arrays_exported': 0, 'participant_ids_exported': 0}))


if __name__ == '__main__':
    export()
