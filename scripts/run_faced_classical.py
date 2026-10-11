"""Freeze, execute and verify the declared FACED controls with a source-first gate."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO))
import numpy as np
import scipy
import sklearn
from gruxnet.faced_classical_contract import context_features, spectral_features
from gruxnet.faced_classical_runner import source_gate, activate_after_source, run_controls
from gruxnet.faced_source_quality import read_interval, selected_header, quality_metrics, QUALITY_RULES
from gruxnet.faced_source_notch import notched_late_trial_windows
from gruxnet.faced_proposal_preprocessing_v2 import common_indices, calibrate_common_counts, late_segment_bounds
from gruxnet.faced_targets import read_declared_item10
from scripts.audit_faced_raw import read_header

PROGRAMME = REPO/'results/development/faced_development_protocol_2026-10-11/plan.json'
RAW_EVIDENCE = REPO/'results/development/faced_raw_2026-10-11'
TARGET = REPO/'results/development/faced_target_contract_2026-10-11'
ROLES = REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json'
PUBLIC = REPO/'results/development/faced_classical_execution_2026-10-11'
PRIVATE = WORKSPACE/'publication_runs/faced_classical_execution_2026-10-11'
RAW = WORKSPACE/'Data/Data'


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    if path.exists(): raise ValueError('Preserve previous receipt: '+path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def array_digest(array):
    x = np.ascontiguousarray(array)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__}


def dependencies():
    paths = [Path(__file__), REPO/'gruxnet/faced_classical_runner.py',
             REPO/'tests/test_faced_classical_runner.py', REPO/'gruxnet/faced_source_quality.py',
             REPO/'scripts/audit_faced_raw.py', PROGRAMME, TARGET/'packet_verification.json',
             RAW_EVIDENCE/'headers.json', RAW_EVIDENCE/'extraction.json',
             RAW_EVIDENCE/'joins_numeric_projection.json', RAW_EVIDENCE/'verification.json',
             REPO/'results/development/emo_mc_qualification_2026-10-10/reservation.json',
             WORKSPACE/'report.tex']
    for relative in load(PROGRAMME)['bindings']: paths.append(REPO/relative)
    return {p.relative_to(WORKSPACE).as_posix(): digest(p) for p in paths}


def programme_check():
    p = load(PROGRAMME)
    for relative, expected in p['bindings'].items():
        if digest(REPO/relative) != expected: raise ValueError('Bounded programme dependency changed')
    for name in (RAW_EVIDENCE/'verification.json', TARGET/'verification.json'):
        for relative, expected in load(name)['bindings'].items():
            if digest(REPO/relative) != expected: raise ValueError('Source qualification dependency changed')
    return p


def partition(p):
    roles = load(ROLES)
    all_pairs = [(r['subject'], r['vid']) for r in p['input_access']['future_development_pairs']]
    src = sorted(pair for pair in all_pairs if pair[0] in roles['development_source'])
    val = sorted(pair for pair in all_pairs if pair[0] in roles['development_validation'])
    if len(set(all_pairs)) != 1260 or len(src) != 980 or len(val) != 280:
        raise ValueError('Execution identities differ from declared union')
    if any(person in roles['confirmation'] or vid not in (2,3,4,6,9,11,12,17,18,20,23,25,27,28) for person, vid in all_pairs):
        raise PermissionError('Protected participant/material in original access plan')
    return src, val


def declare():
    p = programme_check(); source, validation = partition(p)
    save(PUBLIC/'activation_plan.json', {
        'date': '2026-10-11', 'bindings': dependencies(), 'runtime': runtime(),
        'programme_sha256': digest(PROGRAMME), 'source_pairs': [{'subject': s, 'vid': v} for s, v in source],
        'validation_pairs': [{'subject': s, 'vid': v} for s, v in validation],
        'source_gate': 'Finish all 980 source trials before any rating or validation EEG; any frozen review flag holds both. Numerical/authentication failure stops execution and records incomplete coverage.',
        'source_review_flags': {**QUALITY_RULES, 'processed_49_51_to_1_45_power_ratio_above': .10},
        'quality_flag_handling': 'Complete the source rectangle even if review flags appear, retaining all flags/trials. Do not alter preprocessing, exclude or fit. No outcome-based repair.',
        'validation': 'Only after source pass; retain all flags. Authenticated, numerically unavailable EEG retains the trial with declared inference fallback; authentication failures stop access.',
        'decoder': 'Every selected original signed count independently scalar checked; header ADC conversion also checked by independently expressed gain and offset; no unselected channel/sample decode.',
        'original_data_integrity': 'Whole-file SHA and technical header before access; data SHA again after all selected intervals; opaque evt/MAT companion hashes; never interpret recordInformation.json.',
        'cache': 'Per-trial context/150-spectral features and hashes, access ledgers and QC private; no raw waveforms persisted. Item10 cache only after gate. Public anonymous QC/predictions permitted by author.',
        'fitting': 'Exact 45-candidate and 45 selected-alpha OOF fits; scalers/source priors refit on each exact source set. No target-derived feature, neural fit or GPU.',
        'bootstrap': '4000 paired crossed participant/film draws per rotation with seed425401; exploratory, conditional on validation selection.',
        'time_cap_seconds': 7200, 'cap_start': 'Immediately before original-data execution, includes source and validation preparation, target access, fitting and analysis.',
        'failure_receipts': 'No overwrite/resume of a partial execution; retain partial rows, exact counts, job states and immutable failure receipt. A continuation requires a separate declaration.',
        'lead': 'Numerical criteria plus explicit source-capacity/coverage interpretation required before proposing further confirmation; reused rotations are not independent.',
        'original_ratings_decoded_at_declaration': 0, 'original_ridge_fits_at_declaration': 0,
        'confirmation_access': False, 'research_question_changed': False,
        'documentation': ['https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html',
                          'https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html']})
    print({'runner_frozen': True, 'source_trials': 980, 'original_ratings_decoded': 0}, flush=True)


def check():
    p = programme_check(); activation = load(PUBLIC/'activation_plan.json')
    if activation['bindings'] != dependencies() or activation['runtime'] != runtime():
        raise ValueError('Frozen runner bindings/runtime differ')
    source, val = partition(p)
    if activation['source_pairs'] != [{'subject': s, 'vid': v} for s, v in source] or activation['validation_pairs'] != [{'subject': s, 'vid': v} for s, v in val]:
        raise ValueError('Frozen original-access identities differ')
    return p, activation, source, val


class Execution:
    def __init__(self, p, activation):
        self.programme = p; self.activation = activation; self.started = time.monotonic()
        self.headers = {r['subject']: r['data.bdf'] for r in load(RAW_EVIDENCE/'headers.json')['records']}
        self.joins = {r['subject']: r for r in load(RAW_EVIDENCE/'joins_numeric_projection.json')['records']}
        self.files = {r['member']: r for r in load(RAW_EVIDENCE/'extraction.json')['records']}
        self.catalogue = {r['clip']: r for r in load(REPO/'results/development/faced_codebooks_2026-10-10/audit.json')['stimuli']}
        self.storage = {r['subject']: r for r in load(TARGET/'storage.json')['records']}
        self.records = {'source': [], 'validation': []}; self.authenticated = {}
        self.features = {'context': {}, 'eeg': {}}; self.metadata = {}; self.targets = {}
        self.ratings_decoded = 0; self.decoded_values = 0; self.fit_records = []
        self.prediction_rows = []; self.target_access = []; self.gate = None

    def budget(self):
        if time.monotonic()-self.started > self.activation['time_cap_seconds']:
            raise TimeoutError('Declared two-hour original-data execution cap reached')

    def authenticate(self, subject):
        self.budget()
        if subject in self.authenticated: return
        root = RAW/subject.replace('sub-', 'sub'); path = root/'data.bdf'
        h = self.headers[subject]
        if read_header(path) != h: raise ValueError('Qualified technical header changed')
        for name in ('data.bdf', 'evt.bdf', 'After_remarks.mat'):
            if digest(root/name) != self.files[f'Data/{root.name}/{name}']['sha256']:
                raise ValueError('Original source/companion bytes changed')
        self.authenticated[subject] = {'data_sha256': self.files[f'Data/{root.name}/data.bdf']['sha256'],
                                      'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}

    def prepare(self, pairs, role):
        permitted = {(r['subject'], r['vid']) for r in self.activation[role+'_pairs']}
        if set(pairs) != permitted or len(pairs) != len(permitted):
            raise PermissionError('Original EEG request differs from frozen role')
        if role == 'validation' and (self.gate is None or not self.gate['passed']):
            raise PermissionError('Source quality has not cleared validation access')
        people = sorted({p for p, v in pairs})
        for subject in people:
            self.authenticate(subject); h = self.headers[subject]; root = RAW/subject.replace('sub-', 'sub')
            path = root/'data.bdf'; rate = int(h['sample_rates_hz'][0])
            indices = common_indices(h['columns']['labels']); projected = selected_header(h, indices)
            for person, vid in [pair for pair in pairs if pair[0] == subject]:
                self.budget()
                spans = [s for s in self.joins[person]['spans'] if s['vid'] == vid]
                if len(spans) != 1 or not self.joins[person]['passed']: raise ValueError('Unqualified trial join')
                span = spans[0]
                start, stop = late_segment_bounds(span['start_seconds'], span['end_seconds'], rate,
                                                 h['record_count']*int(h['columns']['samples_per_record'][0]))
                metadata = {**self.catalogue[vid], 'trial': span['trial'], 'vid': vid,
                            'family': self.catalogue[vid]['source_film_sha256'],
                            'acquisition_group': ('legacy_' if int(subject[-3:]) < 61 else 'modern_')+str(rate)}
                pair = (person, vid); self.metadata[pair] = metadata
                context = context_features(metadata); self.features['context'][pair] = context
                row = {'subject': person, 'vid': vid, 'trial': span['trial'], 'role': role,
                       'acquisition_group': metadata['acquisition_group'], 'native_rate_hz': rate,
                       'sample_interval': [start, stop], 'original_header_sha256': h['header_sha256'],
                       'original_data_sha256': self.authenticated[subject]['data_sha256'],
                       'channel_indices': indices, 'context_feature_sha256': array_digest(context),
                       'technical_passed': False, 'review_flags': [], 'samples_decoded': 0}
                try:
                    with path.open('rb') as stream: counts, access = read_interval(stream, h, start, stop, indices, True)
                    self.decoded_values += counts.size; row['samples_decoded'] = int(counts.size)
                    save(PRIVATE/f'{person}_clip{vid:02d}_access.json', access)
                    row.update({'access_ledger_file_sha256': digest(PRIVATE/f'{person}_clip{vid:02d}_access.json'),
                                'range_ledger_sha256': access['range_ledger_sha256'],
                                'scalar_reference_checked': access['all_counts_scalar_reference_checked'],
                                'extra_boundary_samples_decoded': access['extra_boundary_samples_decoded'],
                                'count_array_sha256': array_digest(counts)})
                    values = calibrate_common_counts(counts, person, projected)
                    cols = projected['columns']; dmin = np.asarray(cols['digital_min'], float)[:, None]
                    pmin = np.asarray(cols['physical_min'], float)[:, None]
                    gain = (np.asarray(cols['physical_max'], float)[:, None]-pmin)/(np.asarray(cols['digital_max'], float)[:, None]-dmin)
                    error = float(np.max(np.abs(values-(counts.astype(float)*gain+pmin-dmin*gain))))
                    if error > 1e-8: raise ValueError('Independent gain/offset calibration differs')
                    windows = notched_late_trial_windows(values, rate)
                    quality = quality_metrics(counts, values, windows, rate)
                    spectral = spectral_features(windows)
                    flags = []
                    for channel in quality['channels']:
                        if channel['processed_49_51_to_1_45_power_ratio'] > .10:
                            channel['review_flags'].append('residual_50hz_power')
                        flags.extend({'channel': channel['channel'], 'flag': flag} for flag in channel['review_flags'])
                    self.features['eeg'][pair] = spectral
                    cache = PRIVATE/f'{person}_clip{vid:02d}_features.npz'
                    if cache.exists(): raise ValueError('Preserve previous feature cache')
                    with cache.open('xb') as stream: np.savez_compressed(stream, context=context, spectral=spectral)
                    save(PRIVATE/f'{person}_clip{vid:02d}_quality.json', quality)
                    row.update({'technical_passed': True, 'review_flags': flags,
                                'feature_cache_sha256': digest(cache), 'spectral_feature_sha256': array_digest(spectral),
                                'quality_file_sha256': digest(PRIVATE/f'{person}_clip{vid:02d}_quality.json'),
                                'processed_window_sha256': array_digest(windows),
                                'calibration_max_absolute_error_uv': error,
                                'processed_sd_uv_min_median_max': quality['processed_channel_sd_uv_min_median_max'],
                                'maximum_processed_peak_to_peak_uv': max(c['processed_peak_to_peak_uv'] for c in quality['channels']),
                                'maximum_processed_line_ratio': max(c['processed_49_51_to_1_45_power_ratio'] for c in quality['channels']),
                                'valid_windows': 7, 'windows_dropped': 0})
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    row['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
                    self.records[role].append(row)
                    if role == 'source': raise
                    # Numerically unavailable, already authenticated validation input.
                    self.features['eeg'].pop(pair, None)
                    continue
                self.records[role].append(row)
                auth = self.authenticated[person]
                if path.stat().st_size != auth['size'] or path.stat().st_mtime_ns != auth['mtime_ns']:
                    raise ValueError('Original file changed during selective access')
            if digest(path) != self.authenticated[subject]['data_sha256']:
                raise ValueError('Original file changed after selective access')
            print({'role': role, 'participant_complete': subject, 'trials_complete': len(self.records[role]),
                   'flagged_trials': sum(bool(r['review_flags']) for r in self.records[role]),
                   'elapsed_seconds': round(time.monotonic()-self.started, 1)}, flush=True)

    def open_targets(self, allowed_pairs):
        if self.gate is None or not self.gate['passed'] or not (PUBLIC/'source_gate.json').exists():
            raise PermissionError('Persisted source quality gate required before item10 access')
        save(PUBLIC/'outcome_activation.json', {'source_gate_sha256': digest(PUBLIC/'source_gate.json'),
             'activation_plan_sha256': digest(PUBLIC/'activation_plan.json'),
             'permitted_item10_pairs': len(allowed_pairs), 'confirmation_access': False,
             'source_quality_passed': True, 'ratings_decoded_before_activation': 0})
        for subject in sorted({p for p, v in allowed_pairs}):
            self.budget(); storage = self.storage[subject]
            observed = []
            def observer(event):
                self.ratings_decoded += 1; observed.append(event)
            rows, scope = read_declared_item10(RAW, subject, allowed_pairs,
                storage['original_file_sha256'], self.joins[subject]['identity_rows'],
                {r['vid']: r['score_storage'] for r in storage['scores']}, observer=observer)
            for row in rows: self.targets[subject, row['vid']] = row['valence']
            self.target_access.append({'subject': subject, 'scope': scope, 'observed_selected_item_offsets': observed})
        if len(self.targets) != 1260 or self.ratings_decoded != 1260:
            raise ValueError('Target access differs from declared union')
        save(PRIVATE/'targets.json', [{'subject': p, 'vid': v, 'valence': t} for (p,v),t in sorted(self.targets.items())])
        save(PUBLIC/'target_access.json', {'records': self.target_access, 'item10_values_decoded': self.ratings_decoded,
            'other_rating_values_decoded': 0, 'protected_rating_values_decoded': 0,
            'targets_private_sha256': digest(PRIVATE/'targets.json')})

    def retain_fit(self, job, fitted):
        self.budget(); index = len(self.fit_records)
        path = PRIVATE/f'model_{index:03d}.npz'
        if path.exists(): raise ValueError('Preserve previous model state')
        with path.open('xb') as stream:
            np.savez_compressed(stream, mean=fitted['scaler'].mean_, scale=fitted['scaler'].scale_,
                 variance=fitted['scaler'].var_, coef=fitted['estimator'].coef_,
                 intercept=fitted['estimator'].intercept_)
        self.fit_records.append({**job, 'index': index, 'source_pairs': [{'subject': p, 'vid': v} for p,v in fitted['source_pairs']],
                                 'source_identity_sha256': fitted['source_identity_sha256'],
                                 'private_state_sha256': digest(path)})


def execute():
    p, activation, source, validation = check()
    if (PUBLIC/'execution_started.json').exists(): raise ValueError('Preserve earlier execution; no implicit resume')
    save(PUBLIC/'execution_started.json', {'activation_plan_sha256': digest(PUBLIC/'activation_plan.json'),
         'user_authorized_conditional_execution': True, 'source_trials': 980, 'ratings_decoded': 0})
    x = Execution(p, activation); status = 'running'; failure = None
    try:
        x.prepare(source, 'source')
        x.gate = source_gate(x.records['source'], source)
        save(PUBLIC/'source_records.json', {'records': x.records['source']})
        save(PUBLIC/'source_gate.json', {**x.gate, 'source_records_sha256': digest(PUBLIC/'source_records.json'),
                                       'ratings_decoded_at_gate': x.ratings_decoded,
                                       'original_ridge_fits_at_gate': len(x.fit_records)})
        if not x.gate['passed']:
            status = 'held_by_source_quality'
        else:
            x.prepare(validation, 'validation')
            activate_after_source(x.records['source'], source,
                                  lambda: x.open_targets(set(source+validation)))
            results = run_controls(p, x.features, x.targets, x.metadata,
                                   x.retain_fit, x.prediction_rows.extend, x.budget)
            x.budget()
            if results['ridge_fits'] != 90 or results['prior_estimations'] != 72:
                raise ValueError('Completed fit counts differ from declaration')
            save(PUBLIC/'comparison.json', results)
            status = 'completed_controls'
    except Exception as exc:
        status = 'failed_or_truncated'
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if not (PUBLIC/'source_records.json').exists(): save(PUBLIC/'source_records.json', {'records': x.records['source']})
        save(PUBLIC/'validation_records.json', {'records': x.records['validation']})
        save(PUBLIC/'model_manifest.json', {'records': x.fit_records})
        if x.prediction_rows:
            with (PUBLIC/'predictions.csv').open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(x.prediction_rows[0])); writer.writeheader(); writer.writerows(x.prediction_rows)
        save(PUBLIC/'execution.json', {'status': status, 'failure': failure,
             'activation_plan_sha256': digest(PUBLIC/'activation_plan.json'),
             'source_trials_prepared': len(x.records['source']), 'validation_trials_prepared': len(x.records['validation']),
             'source_flagged_trials': sum(bool(r['review_flags']) for r in x.records['source']),
             'source_gate': x.gate, 'original_ratings_decoded': x.ratings_decoded,
             'original_ridge_fits': len(x.fit_records), 'prediction_rows': len(x.prediction_rows),
             'selected_sample_values_decoded': int(x.decoded_values),
             'elapsed_seconds': round(time.monotonic()-x.started, 3),
             'auxiliary_neighbor_or_confirmation_samples_decoded': 0,
             'protected_ratings_decoded': 0, 'other_rating_items_decoded': 0,
             'raw_waveforms_persisted': 0, 'research_question_changed': False})
        print(load(PUBLIC/'execution.json'), flush=True)
    if failure: raise RuntimeError('Original-data execution failed; see immutable execution receipt')


def verify(local=False):
    p, activation, source, validation = check()
    execution = load(PUBLIC/'execution.json')
    records = load(PUBLIC/'source_records.json')['records']
    if len(records) != execution['source_trials_prepared']:
        raise ValueError('Source receipt count differs')
    if execution['status'] in ('held_by_source_quality', 'completed_controls'):
        gate = source_gate(records, source)
        if execution['source_gate'] != gate or load(PUBLIC/'source_gate.json')['source_records_sha256'] != digest(PUBLIC/'source_records.json'):
            raise ValueError('Persisted source gate differs')
    if execution['status'] == 'held_by_source_quality':
        if execution['original_ratings_decoded'] or execution['original_ridge_fits'] or execution['validation_trials_prepared']:
            raise ValueError('Held source gate nevertheless opened outcomes or validation')
        if (PUBLIC/'outcome_activation.json').exists(): raise ValueError('Held gate has outcome activation')
    val_records = load(PUBLIC/'validation_records.json')['records']
    for role, rows, expected in [('source', records, source), ('validation', val_records, validation)]:
        if any((r['subject'], r['vid']) not in set(expected) for r in rows):
            raise ValueError('Protected recording identity in receipt')
        if local:
            for r in rows:
                if r['technical_passed']:
                    stem = f"{r['subject']}_clip{r['vid']:02d}"
                    for suffix, key in [('_access.json', 'access_ledger_file_sha256'), ('_quality.json', 'quality_file_sha256'), ('_features.npz', 'feature_cache_sha256')]:
                        if digest(PRIVATE/(stem+suffix)) != r[key]: raise ValueError('Private feature/QC/access cache changed')
                    with np.load(PRIVATE/(stem+'_features.npz'), allow_pickle=False) as cache:
                        if array_digest(cache['spectral']) != r['spectral_feature_sha256'] or array_digest(cache['context']) != r['context_feature_sha256']:
                            raise ValueError('Private numeric feature hash differs')
    if execution['status'] == 'completed_controls':
        comparison = load(PUBLIC/'comparison.json')
        if comparison['ridge_fits'] != 90 or len(load(PUBLIC/'model_manifest.json')['records']) != 90 or execution['original_ratings_decoded'] != 1260:
            raise ValueError('Completed fitting/target counts differ')
        if len(val_records) != 280: raise ValueError('Incomplete validation coverage')
    public_files = sorted(path for path in PUBLIC.iterdir() if path.suffix in ('.json', '.csv') and path.name not in ('verification.json', 'local_verification.json', 'analysis_verification.json'))
    receipt = {'bindings': {path.relative_to(REPO).as_posix(): digest(path) for path in public_files},
               'status': execution['status'], 'source_trials_verified': len(records),
               'source_gate_passed': bool(execution['source_gate'] and execution['source_gate']['passed']),
               'original_ratings_decoded': execution['original_ratings_decoded'],
               'original_ridge_fits': execution['original_ridge_fits'],
               'private_caches_checked': local, 'raw_data_or_ratings_reread': False,
               'confirmation_access': False, 'research_question_changed': False}
    target = PUBLIC/('local_verification.json' if local else 'verification.json')
    if target.exists():
        if load(target) != receipt: raise ValueError('Immutable verification receipt differs')
    else: save(target, receipt)
    print({k: v for k,v in receipt.items() if k != 'bindings'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('declare', 'execute', 'verify'))
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    if args.mode == 'declare': declare()
    elif args.mode == 'execute': execute()
    else: verify(args.local)
