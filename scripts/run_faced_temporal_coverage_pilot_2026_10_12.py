"""Frozen eight-source-trial signal-quality pilot; never decode a rating or fit.

Reads only the 24 declared 32-second source intervals. Whole-file SHA checks
transport recording bytes for integrity but do not decode other sample ranges.
Per-channel amplitudes/access ledgers stay private; public output is QC status,
geometry and hashes. Failure cannot expand scope or trigger retuning.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import scipy
import mne
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_temporal_coverage import SourceOnlySegmentGuard
from gruxnet.faced_source_quality import read_interval, selected_header, quality_metrics
from gruxnet.faced_proposal_preprocessing import calibrate_common_counts
from gruxnet.faced_artifact_control import artifact_trial_windows
from gruxnet.faced_inference_quality import eeg_availability
from gruxnet.faced_classical_contract import spectral_features
from scripts.audit_faced_raw import read_header

BASE = REPO / 'results/development'
PREFLIGHT = BASE / 'faced_temporal_coverage_preflight_2026-10-12'
PUBLIC = BASE / 'faced_temporal_coverage_pilot_2026-10-12'
PRIVATE = REPO.parent / 'publication_runs/faced_temporal_coverage_pilot_2026-10-12'
RAW = REPO.parent / 'Data/Data'
PLAN = PREFLIGHT / 'plan.json'
ACTIVATION = PUBLIC / 'activation_v2.json'
OLD_QC = REPO.parent / 'publication_runs/faced_artifact_control_2026-10-11'
CAP_SECONDS = 1200

def now():
    return datetime.now(timezone.utc).isoformat()

def load(path):
    return json.loads(path.read_bytes())

def sha(path, check_budget=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            if check_budget is not None:
                check_budget()
            block = stream.read(8*1024*1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def array_sha(array):
    x = np.ascontiguousarray(array)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'mne': mne.__version__, 'blas_threads': 12}

def code_bindings():
    names = ['scripts/run_faced_temporal_coverage_pilot_2026_10_12.py',
        'tests/test_faced_temporal_coverage_pilot.py', 'gruxnet/faced_temporal_coverage.py',
        'gruxnet/faced_source_quality.py', 'gruxnet/faced_proposal_preprocessing.py',
        'gruxnet/faced_source_notch.py', 'gruxnet/faced_artifact_control.py',
        'gruxnet/faced_inference_quality.py', 'gruxnet/faced_classical_contract.py',
        'scripts/audit_faced_raw.py']
    return {n:sha(REPO/n) for n in names}

def input_bindings():
    paths = [PLAN, PREFLIGHT/'receipt.json',
        BASE/'faced_raw_2026-10-11/headers.json', BASE/'faced_raw_2026-10-11/extraction.json',
        REPO/'docs/publication/FACED_Temporal_Coverage_Pilot_Protocol_2026-10-12.md',
        REPO/'paper/exploratory_evaluation_2026-10-12/build_receipt.json',
        REPO/'docs/paper_archive/2026-10-12-before-improvement/manifest.json']
    return {p.relative_to(REPO).as_posix():sha(p) for p in paths}

def validate_activation(activation, plan, actual_plan_sha):
    conditions = {'status':'active_source_only_quality_pilot', 'plan_sha256':actual_plan_sha,
        'original_rating_access':False, 'cached_target_access':False,
        'validation_signal_access':False, 'confirmation_access':False,
        'new_fits':0, 'trial_cap':8, 'segment_cap':24, 'seconds_cap':CAP_SECONDS,
        'cpu_seconds_cap':CAP_SECONDS}
    if any(activation.get(k) != v for k,v in conditions.items()):
        raise PermissionError('Pilot activation is missing, widened or inconsistent')
    if (plan.get('signal_access_activated') is not False or plan.get('new_fits') != 0
            or plan.get('confirmation_access') is not False
            or plan['pilot']['trial_count'] != 8 or plan['pilot']['segment_count'] != 24
            or len(plan['pilot']['trials']) != 8 or len(plan['pilot']['segments']) != 24
            or activation.get('counts_sample_values_cap') != plan['pilot']['counts_sample_values_planned']):
        raise PermissionError('Frozen source pilot geometry changed')

def verify_bindings(values):
    for name, expected in values.items():
        if sha(REPO/name) != expected:
            raise ValueError('Frozen input/code bytes changed: '+name)

def safe_raw_path(person, name):
    if not re.fullmatch(r'sub-\d{3}', person) or name not in ('data.bdf', 'evt.bdf'):
        raise PermissionError('Outside original FACED technical file layout')
    root = RAW.resolve()
    path = (root/person.replace('sub-', 'sub')/name).resolve()
    if not path.is_relative_to(root):
        raise PermissionError('Recording path escaped the declared data root')
    return path

def declare():
    plan = load(PLAN)
    verify_bindings(plan['input_bindings']); verify_bindings(plan['code_bindings'])
    preflight = load(PREFLIGHT/'receipt.json')
    if (preflight.get('plan_sha256') != sha(PLAN) or preflight.get('source_range_checks') != 2940
            or preflight.get('late_exact_identity_checks') != 980 or preflight.get('pilot_rows') != 24
            or preflight.get('waveform_samples_decoded') != 0 or preflight.get('fits') != 0
            or preflight.get('protected_or_validation_signal_access') is not False):
        raise PermissionError('Metadata preflight required')
    activation = {'date':'2026-10-12', 'declared_utc':now(),
        'status':'active_source_only_quality_pilot', 'plan_sha256':sha(PLAN),
        'author_authorization':'User requested improving the saved papers; prior authorization allows experiments. This same-target source QC diagnostic does not adopt another question.',
        'original_rating_access':False, 'cached_target_access':False,
        'validation_signal_access':False, 'confirmation_access':False, 'new_fits':0,
        'trial_cap':8, 'segment_cap':24, 'seconds_cap':CAP_SECONDS,
        'cpu_seconds_cap':CAP_SECONDS,
        'supersedes_before_execution':'activation.json; no source decoding occurred under v1',
        'source_prior_qc_files_read_cap':8,
        'counts_sample_values_cap':plan['pilot']['counts_sample_values_planned'],
        'code_bindings':code_bindings(), 'input_bindings':input_bindings(), 'runtime':runtime(),
        'limits':'Eight original source trials; all source QC failures retained. Pilot cannot authorize full preparation, targets, validation, fitting, manuscript changes or independent confirmation.'}
    validate_activation(activation, plan, sha(PLAN))
    write(ACTIVATION, activation)
    print(json.dumps({'declared':True, 'source_trials':8, 'source_segments':24,
                      'fits':0, 'targets':0}), flush=True)

def run():
    a, plan = load(ACTIVATION), load(PLAN)
    validate_activation(a, plan, sha(PLAN))
    verify_bindings(a['input_bindings']); verify_bindings(a['code_bindings'])
    verify_bindings(plan['input_bindings']); verify_bindings(plan['code_bindings'])
    if a['runtime'] != runtime() or mne.__version__ != '1.10.0':
        raise ValueError('Frozen runtime changed')
    if (PUBLIC/'execution_started.json').exists():
        raise PermissionError('Preserve completed/failed attempt; do not automatically retry')
    guard = SourceOnlySegmentGuard(source_people=plan['source_people'],
        validation_people=plan['validation_people_denied'], protected_people=plan['protected_people_denied'],
        source_pairs=[(r['subject'],r['vid']) for r in plan['source_records']], pilot_rows=plan['pilot']['segments'])
    records = {(r['subject'],r['vid']):r for r in plan['source_records']}
    pilot_pairs = {(r['subject'],r['vid']) for r in plan['pilot']['trials']}
    if len(pilot_pairs) != 8 or len(guard.pilot_ranges) != 24:
        raise PermissionError('Exact pilot allowlist required')
    headers = {r['subject']:r['data.bdf'] for r in load(BASE/'faced_raw_2026-10-11/headers.json')['records'] if r['subject'] in plan['source_people']}
    originals = {r['member']:r for r in load(BASE/'faced_raw_2026-10-11/extraction.json')['records']}
    started = time.monotonic(); cpu_started = time.process_time()
    snapshots = {}; rows = []; decoded = 0; prior_qc_reads = 0
    write(PUBLIC/'execution_started.json', {'started_utc':now(), 'activation_sha256':sha(ACTIVATION),
        'plan_sha256':sha(PLAN), 'raw_source_segments_cap':24, 'outcomes_accessed':0, 'fits':0})
    def budget():
        if time.monotonic()-started > CAP_SECONDS:
            raise TimeoutError('Source-only pilot time cap')
        if time.process_time()-cpu_started > CAP_SECONDS:
            raise TimeoutError('Source-only pilot CPU time cap')
    def reader(person, vid, region, interval, rate):
        nonlocal decoded, prior_qc_reads
        budget()
        if (person,vid) not in pilot_pairs:
            raise PermissionError('Non-pilot source recording')
        r = records[(person,vid)]
        if r['role'] != 'source' or r['segments'][region] != interval or r['native_rate_hz'] != rate:
            raise PermissionError('Authoritative source interval differs')
        path = safe_raw_path(person, 'data.bdf'); header = headers[person]
        if person not in snapshots:
            if read_header(path) != header:
                raise ValueError('Original technical header changed')
            for name in ('data.bdf','evt.bdf'):
                file = safe_raw_path(person,name)
                expected = originals[f'Data/{file.parent.name}/{name}']['sha256']
                if sha(file, budget) != expected:
                    raise ValueError('Original source file failed integrity')
            snapshots[person] = {'size':path.stat().st_size, 'mtime_ns':path.stat().st_mtime_ns,
                'sha256':originals[f'Data/{path.parent.name}/data.bdf']['sha256']}
        if r['original_data_sha256'] != snapshots[person]['sha256'] or r['original_header_sha256'] != header['header_sha256']:
            raise ValueError('Source lineage changed')
        budget()
        with path.open('rb') as stream:
            counts, access = read_interval(stream, header, *interval, r['channel_indices'], reference=True)
        decoded += int(counts.size)
        if decoded > a['counts_sample_values_cap'] or access['extra_boundary_samples_decoded'] != 0:
            raise PermissionError('Selective decoding cap/boundary changed')
        microvolts = calibrate_common_counts(counts,person,selected_header(header,r['channel_indices']))
        with threadpool_limits(limits=12, user_api='blas'):
            windows, artifact = artifact_trial_windows(microvolts,rate)
        quality = quality_metrics(counts,microvolts,windows,rate)
        availability = eeg_availability(quality,technical_passed=True,repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
        count_hash, window_hash = array_sha(counts), array_sha(windows)
        if region == 'late' and (count_hash != r['prior_late_count_array_sha256'] or window_hash != r['prior_late_processed_window_sha256']):
            raise ValueError('Historical late counts/windows fail exact replay')
        if region == 'late' and not availability['eeg_available']:
            raise ValueError('Previously qualified late source QC changed')
        if region == 'late':
            prior_path = OLD_QC/f'{person}_clip{vid:02d}_quality.json'
            if sha(prior_path) != r['prior_late_quality_file_sha256'] or load(prior_path) != quality:
                raise ValueError('Historical source QC numerical replay differs')
            prior_qc_reads += 1
            if prior_qc_reads > a['source_prior_qc_files_read_cap']:
                raise PermissionError('Historical source QC read cap')
        stem = f'{person}_clip{vid:02d}_{region}'
        write(PRIVATE/(stem+'_quality.json'),quality)
        write(PRIVATE/(stem+'_access.json'),access)
        write(PRIVATE/(stem+'_artifact.json'),artifact)
        if path.stat().st_size != snapshots[person]['size'] or path.stat().st_mtime_ns != snapshots[person]['mtime_ns']:
            raise ValueError('Original source modified during selective read')
        budget()
        return {'subject':person, 'vid':vid, 'role':'source', 'region':region,
            'assigned_category':r['assigned_category'], 'acquisition_group':r['acquisition_group'],
            'native_rate_hz':rate, 'sample_interval':interval, 'samples_decoded':int(counts.size),
            'eeg_available':availability['eeg_available'], 'reasons':availability['reasons'],
            'artifact_decision':artifact['decision'], 'repair_count':len(artifact['repaired_channels']),
            'count_array_sha256':count_hash, 'processed_window_sha256':window_hash,
            'spectral_feature_sha256':array_sha(spectral_features(windows)),
            'range_ledger_sha256':access['range_ledger_sha256'],
            'extra_boundary_samples_decoded':0, 'late_exact_replay_passed':region=='late',
            'quality_file_sha256':sha(PRIVATE/(stem+'_quality.json')),
            'private_access_file_sha256':sha(PRIVATE/(stem+'_access.json'))}
    try:
        for row in plan['pilot']['segments']:
            result = guard.read(subject=row['subject'],vid=row['vid'],region=row['region'],
                sample_interval=row['sample_interval'], native_rate_hz=row['native_rate_hz'], reader=reader)
            rows.append(result)
            write(PUBLIC/f'segment_{len(rows):02d}.json',result)
            print(json.dumps({'segment':len(rows), 'region':row['region'],
                'eeg_available':result['eeg_available'], 'elapsed_seconds':round(time.monotonic()-started,1)}), flush=True)
        for person,snapshot in snapshots.items():
            budget()
            if sha(safe_raw_path(person,'data.bdf'), budget) != snapshot['sha256']:
                raise ValueError('Original source recording changed after processing')
        verify_bindings(a['input_bindings']); verify_bindings(a['code_bindings'])
        budget()
        if len(rows) != 24 or decoded != a['counts_sample_values_cap'] or prior_qc_reads != 8:
            raise ValueError('Pilot complete-denominator/sample-count check failed')
        passed = all(r['eeg_available'] for r in rows)
        summary = {'date':'2026-10-12', 'status':'completed_source_quality_pilot',
            'passed':passed, 'source_quality_gate_passed':passed, 'source_trials':8,'segments':24,
            'unavailable_segments':sum(not r['eeg_available'] for r in rows),
            'by_region':{region:{'segments':8,'available':sum(r['eeg_available'] for r in rows if r['region']==region)} for region in ('early','middle','late')},
            'counts_sample_values_decoded':decoded,'elapsed_seconds':time.monotonic()-started,
            'cpu_seconds':time.process_time()-cpu_started, 'prior_source_qc_files_read':prior_qc_reads,
            'activation_sha256':sha(ACTIVATION), 'plan_sha256':sha(PLAN),
            'late_exact_replays':8, 'raw_array_persistence':False,
            'original_rating_access':False,'cached_target_access':False,'new_fits':0,
            'validation_signal_access':False,'confirmation_access':False,
            'limits':'Bounded QC feasibility only. Eight trials do not qualify all source data or EEG utility. No automatic widening, fitting, outcome decoding or manuscript change.'}
        write(PUBLIC/'summary.json',summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write(PUBLIC/'failure.json',{'status':'held_or_failed','error_type':type(error).__name__,
            'reason':str(error),'segments_completed':len(rows),'counts_decoded':decoded,
            'elapsed_seconds':time.monotonic()-started,'outcome_access':False,'new_fits':0,
            'cpu_seconds':time.process_time()-cpu_started,'prior_source_qc_files_read':prior_qc_reads,
            'confirmation_access':False,'automatic_retry':False})
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('declare','run'))
    args = parser.parse_args()
    (declare if args.action == 'declare' else run)()
