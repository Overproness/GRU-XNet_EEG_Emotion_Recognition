"""Source-only four-fit EEGNet capacity remedy; declare/freeze before activate/run."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_source_eegnet_capacity import (
    complete_cell, guard_cell, guarded_source_targets, source_offsets, capacity_targets,
    learning_flags, RATE, SEEDS, MAIN_EPOCHS, MAIN_CHECKPOINTS, CAPACITY_EPOCHS, CAPACITY_CHECKPOINTS,
    PreparationBudget, verify_reloaded_state, verify_replayed_predictions, CHECKPOINT_ATOL, CHECKPOINT_RTOL,
    cache_metadata_projection, guard_source_original_path)
from gruxnet.faced_eegnet_runner import SourceChannelScaler, FitBudget, fit_trial_model
from gruxnet.eegnet_regression import EEGNetRegression
from gruxnet.faced_classical_contract import context_features, predict_ridge, participant_film_mae
from gruxnet.faced_classical_runner import estimate_priors
from gruxnet.faced_eegnet_bridge import array_sha256
from gruxnet.faced_source_quality import read_interval, selected_header, quality_metrics
from gruxnet.faced_proposal_preprocessing_v2 import calibrate_common_counts
from gruxnet.faced_artifact_control import artifact_trial_windows
from gruxnet.faced_inference_quality import eeg_availability
from scripts.audit_faced_raw import read_header
from scripts.qualify_faced_saved_states_2026_10_14 import preservation as paper_preservation
from scripts.qualify_faced_source_export_2026_10_14 import preservation as claim_preservation

BASE = REPO / 'results/development'
PUBLIC = BASE / 'faced_stronger_baseline_2026-10-14'
PRIVATE = REPO.parent / 'publication_runs/faced_stronger_baseline_2026-10-14'
RAW = REPO.parent / 'Data/Data'
OLD_QC = REPO.parent / 'publication_runs/faced_artifact_control_2026-10-11'
PROTOCOL = REPO / 'docs/publication/FACED_Stronger_Baseline_Qualification_2026-10-14.md'
PLAN = PUBLIC / 'plan_v3.json'
PREPARATION_BUDGET = None
INPUTS = [
    'results/development/faced_coverage_selection_preflight_2026-10-13/plan.json',
    'results/development/faced_coverage_missing_2026-10-13/source_policy_plan.json',
    'results/development/faced_development_protocol_2026-10-11/plan.json',
    'results/development/faced_artifact_control_2026-10-11/source_records.json',
    'results/development/faced_artifact_control_2026-10-11/validation_records.json',
    'results/development/faced_backup_2026-10-10/participant_reservation.json',
    'results/development/faced_codebooks_2026-10-10/audit.json',
    'results/development/faced_raw_2026-10-11/headers.json',
    'results/development/faced_raw_2026-10-11/extraction.json',
    'results/development/faced_inference_quality_2026-10-11/predictions.csv',
    'results/development/eegnet_author_audit_2026-10-06/download_manifest.json',
    'results/development/eegnet_author_audit_2026-10-06/port_verification.json']
CODES = ['gruxnet/faced_source_eegnet_capacity.py', 'scripts/run_faced_stronger_baseline_2026_10_14.py',
         'tests/test_faced_source_eegnet_capacity.py', 'gruxnet/faced_eegnet_runner.py',
         'gruxnet/eegnet_regression.py', 'gruxnet/eegnet_control.py', 'gruxnet/faced_classical_contract.py',
         'gruxnet/faced_classical_runner.py', 'gruxnet/faced_source_quality.py',
         'gruxnet/faced_proposal_preprocessing_v2.py', 'gruxnet/faced_proposal_preprocessing.py',
         'gruxnet/faced_source_notch.py', 'gruxnet/faced_artifact_control.py',
         'gruxnet/faced_inference_quality.py', 'gruxnet/faced_eegnet_bridge.py', 'scripts/audit_faced_raw.py',
         'scripts/qualify_faced_saved_states_2026_10_14.py', 'scripts/qualify_faced_source_export_2026_10_14.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            if PREPARATION_BUDGET is not None:
                PREPARATION_BUDGET.check()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            if PREPARATION_BUDGET is not None:
                PREPARATION_BUDGET.check()
    return h.hexdigest()


def save(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metadata_bundle():
    reservation = load(BASE / 'faced_backup_2026-10-10/participant_reservation.json')
    source_people = reservation['development_source']
    validation_people = reservation['development_validation']
    protected = reservation['confirmation']
    if (len(source_people), len(validation_people), len(protected)) != (70, 20, 33):
        raise PermissionError('Changed authoritative reservation')
    if len(set(source_people + validation_people + protected)) != 123:
        raise PermissionError('Participant role overlap')
    programme = load(BASE / 'faced_development_protocol_2026-10-11/plan.json')
    catalogue = {r['clip']: r for r in load(BASE / 'faced_codebooks_2026-10-10/audit.json')['stimuli']}
    source_records = load(BASE / 'faced_artifact_control_2026-10-11/source_records.json')['records']
    validation_records = load(BASE / 'faced_artifact_control_2026-10-11/validation_records.json')['records']
    metadata, records = {}, {}
    for row in source_records + validation_records:
        pair = row['subject'], row['vid']
        role = 'source' if pair[0] in source_people else 'validation' if pair[0] in validation_people else None
        if role is None or row['role'] != role or pair in metadata or row['technical_passed'] is not True:
            raise PermissionError('Changed recording identity/technical role')
        if row['native_rate_hz'] not in (250, 1000) or row['sample_interval'][1] - row['sample_interval'][0] != 32 * row['native_rate_hz']:
            raise PermissionError('Changed selected native interval')
        metadata[pair] = {**catalogue[pair[1]], 'subject': pair[0], 'vid': pair[1], 'role': role,
                          'family': catalogue[pair[1]]['source_film_sha256'], 'trial': row['trial'],
                          'acquisition_group': row['acquisition_group']}
        records[pair] = row
    expected = {(r['subject'], r['vid']) for r in programme['input_access']['future_development_pairs']}
    if set(metadata) != expected or len(source_records) != 980 or len(validation_records) != 280:
        raise PermissionError('Authoritative metadata denominator changed')
    policy = load(BASE / 'faced_coverage_missing_2026-10-13/source_policy_plan.json')
    cells = load(BASE / 'faced_coverage_selection_preflight_2026-10-13/plan.json')['cells']
    cell = complete_cell(cells, policy['shared_missing_pairs'])
    train, held, tiny = guard_cell(cell, metadata, source_people, policy['shared_missing_pairs'])
    if cell['cell'] != 6 or cell['person_fold'] != 3 or cell['material_assignment'] != 0:
        raise PermissionError('Changed metadata-qualified complete cell')
    selected = sorted(set(train) | set(held))
    for pair in selected:
        row = records[pair]
        if len(row['channel_indices']) != 30 or len(set(row['channel_indices'])) != 30 or row['repair_capacity_exceeded']:
            raise PermissionError('Source interval/channel/repair geometry changed')
        if array_sha256(context_features(metadata[pair])) != row['context_feature_sha256']:
            raise ValueError('Source context metadata bytes changed')
    return programme, metadata, records, cell, train, held, tiny, selected


def bindings():
    return {name: digest(REPO / name) for name in INPUTS + CODES + [PROTOCOL.relative_to(REPO).as_posix()]}


def preservation_bindings():
    paper = paper_preservation(); claims = claim_preservation(); preserved = {}
    for receipt in claims:
        name, field = receipt['receipt'], receipt['field']
        preserved[name] = digest(REPO / name)
        for relative, expected in load(REPO / name)[field].items():
            if digest(REPO / relative) != expected:
                raise PermissionError('Preserved manuscript/claim dependency changed')
            if relative in preserved and preserved[relative] != expected:
                raise PermissionError('Inconsistent preserved binding')
            preserved[relative] = expected
    archive = REPO / 'docs/paper_archive/2026-10-12-before-improvement'
    for record in load(archive / 'manifest.json')['files']:
        preserved[record['source']] = record['sha256']
        preserved[(archive / record['archive']).relative_to(REPO).as_posix()] = record['sha256']
    preserved[(archive / 'manifest.json').relative_to(REPO).as_posix()] = digest(archive / 'manifest.json')
    return preserved, {'paper_and_archive': paper, 'old_claim_packets': claims}


def declare():
    programme, metadata, records, cell, train, held, tiny, selected = metadata_bundle()
    originals = {r['member']: r for r in load(BASE / 'faced_raw_2026-10-11/extraction.json')['records']}
    headers = {r['subject']: r['data.bdf'] for r in load(BASE / 'faced_raw_2026-10-11/headers.json')['records']}
    people = sorted({p[0] for p in selected})
    files = [{'subject': person, 'name': name, 'member': f'Data/{person.replace("sub-", "sub")}/{name}',
              'bytes': originals[f'Data/{person.replace("sub-", "sub")}/{name}']['bytes'],
              'sha256': originals[f'Data/{person.replace("sub-", "sub")}/{name}']['sha256']}
             for person in people for name in ('data.bdf', 'evt.bdf')]
    sample_values = sum((records[p]['sample_interval'][1] - records[p]['sample_interval'][0]) * 30 for p in selected)
    opaque_bytes = sum(r['bytes'] * (2 if r['name'] == 'data.bdf' else 1) for r in files)
    header_bytes = sum(headers[p]['header_bytes'] for p in people)
    with (BASE / 'faced_inference_quality_2026-10-11/predictions.csv').open(newline='', encoding='utf-8') as stream:
        cached = list(csv.DictReader(stream))
    target_calls = sum((r['subject'], int(r['vid'])) in set(selected) for r in cached)
    projection = cache_metadata_projection(cached)
    preserved, evidence = preservation_bindings()
    plan = {'date': '2026-10-14', 'status': 'prepared_source_only_not_activated', 'declared_utc': now(),
            'source_cell': cell, 'source_pairs': selected, 'tiny_pairs': tiny,
            'source_records': [records[p] for p in selected], 'source_participant_folds': programme['source_participant_oof_folds'],
            'target_cache_rows': 27720, 'target_scope': 532, 'target_conversion_calls': target_calls,
            'target_metadata_projection_sha256': projection,
            'validation_target_values_converted': 0,
            'source_cache_waveforms_available': False, 'original_access': '532 exact source late ranges only; prior numerical QC/count/window bytes replay',
            'neural_fits': 4, 'auxiliary_context_fits': 5, 'main_epochs': MAIN_EPOCHS,
            'main_checkpoints': MAIN_CHECKPOINTS, 'capacity_epochs': CAPACITY_EPOCHS,
            'capacity_checkpoints': CAPACITY_CHECKPOINTS, 'rate': RATE, 'seeds': SEEDS,
            'fit_seconds_cap': 7200, 'preparation_seconds_cap': 7200, 'preparation_cpu_seconds_cap': 7200,
            'window_bytes': len(selected) * 7 * 30 * 512 * 4, 'window_bytes_cap': 300000000,
            'cuda_reserved_bytes_cap': 5500000000, 'bindings': bindings(),
            'preservation_bindings': preserved, 'preservation_evidence': evidence,
            'supersedes_unexecuted_v1_sha256': digest(PUBLIC / 'plan_v1_unexecuted.json'),
            'supersedes_unexecuted_v2_sha256': digest(PUBLIC / 'plan_v2_unexecuted.json'),
            'original_files': files, 'read_budget': {'sample_values_decoded': sample_values,
                'selected_signal_bytes': sample_values * 3, 'data_header_bytes': header_bytes,
                'opaque_original_hash_bytes': opaque_bytes,
                'total_original_bytes': sample_values * 3 + header_bytes + opaque_bytes,
                'integrity_chunk_bytes': 1024 * 1024},
            'checkpoint_replay': {'states': 10, 'cpu_only': True, 'atol_original_scale': CHECKPOINT_ATOL,
                'rtol': CHECKPOINT_RTOL, 'main_source_pairs': [train[0], train[-1]],
                'main_held_pairs': [held[0], held[-1]], 'capacity_source_pairs': [tiny[0], tiny[-1]],
                'scope': 'Independent tensor-only reload/strict geometry/finite/norm and forward replay through qualified model using same transient source windows; not independent architecture reimplementation'},
            'validation_access': False, 'protected_access': False, 'research_question_changed': False,
            'new_confirmation_candidate': False, 'raw_waveform_persistence': False,
            'interpretation': 'Adaptive reused-source optimization/capacity diagnostic, no winner selection or independent utility confirmation'}
    save(PLAN, plan, exclusive=True)
    print({'plan_ready': True, 'selected_cell': 6, 'source_window_target_pairs': 532, 'neural_fit_cap': 4}, flush=True)


def check_plan():
    plan = load(PLAN)
    if plan['bindings'] != bindings() or plan['status'] != 'prepared_source_only_not_activated':
        raise PermissionError('Frozen source declaration changed')
    for relative, expected in plan['preservation_bindings'].items():
        if digest(REPO / relative) != expected:
            raise PermissionError('Preserved paper/claim binding changed')
    claim_preservation(); paper_preservation()
    return plan


def activate():
    plan = check_plan()
    save(PUBLIC / 'activation.json', {'date': '2026-10-14', 'status': 'active_source_only_diagnostic',
        'activated_utc': now(), 'plan_sha256': digest(PLAN), 'bindings': plan['bindings'],
        'validation_access': False, 'protected_access': False, 'research_question_changed': False,
        'new_confirmation_candidate': False}, exclusive=True)


def execute():
    global PREPARATION_BUDGET
    plan = check_plan(); activation = load(PUBLIC / 'activation.json')
    if activation['status'] != 'active_source_only_diagnostic' or activation['plan_sha256'] != digest(PLAN) or activation['bindings'] != bindings():
        raise PermissionError('Separately frozen activation required')
    if any(activation[k] is not False for k in ('validation_access', 'protected_access', 'research_question_changed', 'new_confirmation_candidate')):
        raise PermissionError('Changed source-only scope')
    programme, metadata, records, cell, train, held, tiny, selected = metadata_bundle()
    if plan['source_pairs'] != [list(p) for p in selected] or plan['source_records'] != [records[p] for p in selected]:
        raise PermissionError('Changed exact selected raw identity ledger')
    if not torch.cuda.is_available():
        raise PermissionError('Qualified 6 GB CUDA device required')
    torch.set_num_threads(2)
    torch.cuda.init(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); windows = {}; receipts = []; authenticated = {}
    PREPARATION_BUDGET = PreparationBudget(plan['preparation_seconds_cap'], plan['preparation_cpu_seconds_cap'])
    original_read_bytes = {'opaque_original_hash_bytes': 0, 'data_header_bytes': 0, 'selected_signal_bytes': 0}
    states = []; prediction_rows = []; histories = []; budget = FitBudget(seconds_cap=7200, fit_cap=4, reserved_bytes_cap=5500000000)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    save(PUBLIC / 'execution_started.json', {'started_utc': now(), 'plan_sha256': digest(PLAN),
        'protected_access': False, 'validation_access': False}, exclusive=True)
    try:
        headers = {r['subject']: r['data.bdf'] for r in load(BASE / 'faced_raw_2026-10-11/headers.json')['records']}
        originals = {r['member']: r for r in load(BASE / 'faced_raw_2026-10-11/extraction.json')['records']}
        allowed_files = {(r['subject'], r['name']) for r in plan['original_files']}
        safe_paths = {(person, name): guard_source_original_path(REPO.parent, RAW, person, name, allowed_files)
                      for person, name in sorted(allowed_files)}
        for index, pair in enumerate(selected):
            PREPARATION_BUDGET.check()
            row = records[pair]; person, vid = pair; path = safe_paths[person, 'data.bdf']; directory = path.parent
            if person not in authenticated:
                if read_header(path) != headers[person]:
                    raise ValueError('Authenticated source header changed')
                original_read_bytes['data_header_bytes'] += headers[person]['header_bytes']
                for name in ('data.bdf', 'evt.bdf'):
                    original = originals[f'Data/{directory.name}/{name}']; expected_hash = original['sha256']
                    if safe_paths[person, name].stat().st_size != original['bytes']:
                        raise ValueError('Original recording size differs before hashing')
                    if digest(safe_paths[person, name]) != expected_hash:
                        raise ValueError('Authenticated source original changed')
                    original_read_bytes['opaque_original_hash_bytes'] += original['bytes']
                authenticated[person] = {'sha256': originals[f'Data/{directory.name}/data.bdf']['sha256'],
                                         'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            if row['original_data_sha256'] != authenticated[person]['sha256']:
                raise ValueError('Selected source lineage differs from original')
            start, stop = row['sample_interval']; indices = row['channel_indices']; header = headers[person]
            with path.open('rb') as stream:
                counts, access = read_interval(stream, header, start, stop, indices, True)
            PREPARATION_BUDGET.check()
            original_read_bytes['selected_signal_bytes'] += access['signal_bytes_decoded']
            if array_sha256(counts) != row['count_array_sha256'] or access['range_ledger_sha256'] != row['range_ledger_sha256']:
                raise ValueError('Selected source count/access bytes changed')
            values = calibrate_common_counts(counts, person, selected_header(header, indices))
            with threadpool_limits(limits=12, user_api='blas'):
                trial_windows, artifact = artifact_trial_windows(values, row['native_rate_hz'])
            quality = quality_metrics(counts, values, trial_windows, row['native_rate_hz'])
            old = OLD_QC / f'{person}_clip{vid:02d}_quality.json'
            if digest(old) != row['quality_file_sha256'] or load(old) != quality:
                raise ValueError('Prior full numerical source QC differs')
            decision = eeg_availability(quality, technical_passed=True, repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
            if not decision['eeg_available'] or array_sha256(trial_windows) != row['processed_window_sha256']:
                raise ValueError('Prior processed source window/quality decision changed')
            if path.stat().st_size != authenticated[person]['size'] or path.stat().st_mtime_ns != authenticated[person]['mtime_ns']:
                raise ValueError('Source original changed during selective access')
            trial_windows.flags.writeable = False; windows[pair] = trial_windows
            if sum(x.nbytes for x in windows.values()) > plan['window_bytes_cap']:
                raise MemoryError('Transient physical waveform cap exceeded')
            receipts.append({'subject': person, 'vid': vid, 'role': 'source', 'sample_interval': row['sample_interval'],
                'channel_indices': indices, 'samples_decoded': int(counts.size), 'count_array_sha256': row['count_array_sha256'],
                'processed_window_sha256': row['processed_window_sha256'], 'range_ledger_sha256': access['range_ledger_sha256'],
                'quality_file_sha256': row['quality_file_sha256'], 'full_numerical_qc_replayed': True})
            if index % 16 == 0:
                print({'phase': 'source_replay', 'trials': index + 1, 'seconds': round(time.monotonic() - started, 1)}, flush=True)
        for person, expected in authenticated.items():
            if digest(safe_paths[person, 'data.bdf']) != expected['sha256']:
                raise ValueError('Source original changed after selected access')
            original_read_bytes['opaque_original_hash_bytes'] += originals[f'Data/{person.replace("sub-", "sub")}/data.bdf']['bytes']
        preparation_clocks = PREPARATION_BUDGET.receipt()
        for key, actual in original_read_bytes.items():
            if actual != plan['read_budget'][key]:
                raise PermissionError('Original source byte ledger differs from fixed read budget')
        save(PUBLIC / 'source_gate.json', {'passed': True, 'pairs': len(windows), 'receipts': receipts,
            'waveform_bytes': sum(x.nbytes for x in windows.values()), 'original_rating_decode': 0,
            'validation_signals_decoded': 0, 'protected_signals_decoded': 0, 'fits': 0, 'targets_converted': 0,
            'original_read_bytes': original_read_bytes, 'preparation_clocks': preparation_clocks}, exclusive=True)
        PREPARATION_BUDGET = None
        cache = BASE / 'faced_inference_quality_2026-10-11/predictions.csv'
        if digest(cache) != plan['bindings'][cache.relative_to(REPO).as_posix()]:
            raise ValueError('Canonical source target cache changed')
        with cache.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 27720:
            raise PermissionError('Canonical cache metadata denominator changed')
        if cache_metadata_projection(rows) != plan['target_metadata_projection_sha256']:
            raise PermissionError('Canonical cache complete metadata projection changed')
        for row in rows:
            if row['arm'] == 'source_oof' and row['subject'] not in programme['source_participant_oof_folds'][int(row['oof_fold'])]:
                raise PermissionError('Canonical source OOF recipient person changed')
        targets, conversion_receipt = guarded_source_targets(rows, metadata, selected, return_receipt=True)
        if conversion_receipt['selected_target_conversion_calls_including_duplicates'] != plan['target_conversion_calls']:
            raise PermissionError('Selected source target conversion denominator changed')
        save(PUBLIC / 'target_receipt.json', {'source_gate_sha256': digest(PUBLIC / 'source_gate.json'),
            'canonical_metadata_rows_checked_before_target_conversion': 27720, 'target_pairs_converted': 532,
            'target_source_sha256': digest(cache), 'new_original_ratings_decoded': 0,
            'validation_target_values_converted': 0, 'protected_outcomes_opened': False,
            'exact_conversion_receipt': conversion_receipt}, exclusive=True)
        decisions = {p: {'eeg_available': True} for p in selected}
        features = {p: context_features(metadata[p]) for p in selected}
        with threadpool_limits(limits=2, user_api='blas'):
            context, oof, ledger = source_offsets(train, programme['source_participant_oof_folds'], features, targets)
            priors = estimate_priors(train, targets, metadata)
        scaler = SourceChannelScaler.fit(train, windows)
        full_training_offsets = dict(zip(train, predict_ridge(context, train, features)))
        held_offsets = dict(zip(held, predict_ridge(context, held, features)))
        save(PRIVATE / 'source_preparation.json', {'source_pairs': train, 'channel_mean': scaler.mean.tolist(),
            'channel_sd': scaler.sd.tolist(), 'inner_context_ledger': ledger, 'context_mean': context['scaler'].mean_.tolist(),
            'context_scale': context['scaler'].scale_.tolist(), 'context_coefficient': context['estimator'].coef_.tolist(),
            'context_intercept': float(context['estimator'].intercept_), 'priors': priors,
            'source_oof_offsets': [[list(p), float(oof[p])] for p in train]}, exclusive=True)
        event_path = PUBLIC / 'events.jsonl'
        checkpoint_replays = []

        def retain(event, label, source_pairs, held_pairs, fitted_targets, fitted_scaler, fitted_offsets):
            kind = event['type']
            with event_path.open('a', encoding='utf-8') as stream:
                if kind in ('epoch', 'failure'):
                    stream.write(json.dumps({'utc': now(), 'fit': label, **event}, allow_nan=False) + '\n')
                elif kind == 'checkpoint':
                    epoch = event['epoch']; state_path = PRIVATE / f'{label}_epoch{epoch}.pt'
                    if state_path.exists():
                        raise PermissionError('Checkpoint overwrite forbidden')
                    torch.save(event['state_dict'], state_path)
                    record = {'fit': label, 'epoch': epoch, 'path': state_path.name, 'sha256': digest(state_path),
                              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
                    states.append(record)
                    replay_pairs = [source_pairs[0], source_pairs[-1]] + ([] if not held_pairs else [held_pairs[0], held_pairs[-1]])
                    reloaded = EEGNetRegression().cpu()
                    reloaded.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=True), strict=True)
                    verify_reloaded_state(reloaded); reloaded.eval()
                    replay_input = torch.as_tensor(np.stack([fitted_scaler.transform(windows[p]) for p in replay_pairs]))
                    expected_values, offset_values = [], []
                    for p in replay_pairs:
                        arm = 'source_fit' if p in source_pairs else 'held_source'
                        index_in_arm = (source_pairs if arm == 'source_fit' else held_pairs).index(p)
                        expected_values.append(float(event[arm + '_raw'][index_in_arm]))
                        offset_values.append(float(fitted_offsets[arm][p]))
                    with torch.no_grad():
                        replay_values = 7 * reloaded(replay_input).numpy() + np.asarray(offset_values)
                    difference = verify_replayed_predictions(expected_values, replay_values)
                    checkpoint_replays.append({**record, 'pairs': replay_pairs, 'maximum_absolute_error_original_scale': difference,
                                               'strict_tensor_geometry_loaded': True, 'finite_and_constraints_passed': True})
                    budget.check('cuda')
                    stream.write(json.dumps({'utc': now(), **record}, allow_nan=False) + '\n')
                    for arm, pairs in (('source_fit', source_pairs), ('held_source', held_pairs)):
                        if not pairs:
                            continue
                        values = event[arm + '_raw']
                        for pair, value in zip(pairs, values):
                            prediction_rows.append({'fit': label, 'epoch': epoch, 'arm': arm, 'subject': pair[0],
                                'vid': pair[1], 'family': metadata[pair]['family'], 'target': float(fitted_targets[pair]),
                                'prediction_raw': float(value)})
                    if 'source_inference_raw' in event:
                        for pair, value in zip(source_pairs, event['source_inference_raw']):
                            prediction_rows.append({'fit': label, 'epoch': epoch, 'arm': 'source_inference', 'subject': pair[0],
                                'vid': pair[1], 'family': metadata[pair]['family'], 'target': float(fitted_targets[pair]),
                                'prediction_raw': float(value)})
            if kind == 'checkpoint' or kind == 'epoch' and event['epoch'] % 10 == 0:
                print({'phase': 'source_neural', 'fit': label, 'event': kind, 'epoch': event['epoch'],
                       'charged_seconds': round(budget.elapsed + (0 if budget.started is None else time.monotonic() - budget.started), 1)}, flush=True)
        for label, permuted in (('capacity_real', False), ('capacity_permuted', True)):
            tiny_targets, order, median = capacity_targets(tiny, targets, permuted=permuted)
            tiny_scaler = SourceChannelScaler.fit(tiny, windows)
            tiny_offsets = {'source_fit': {p: median for p in tiny}}
            save(PRIVATE / (label + '_source.json'), {'pairs': tiny, 'targets': [float(tiny_targets[p]) for p in tiny],
                'permutation': order, 'median': median, 'channel_mean': tiny_scaler.mean.tolist(), 'channel_sd': tiny_scaler.sd.tolist()}, exclusive=True)
            fit = fit_trial_model(tiny, {'source_fit': list(tiny)}, windows, tiny_targets, tiny_offsets,
                tiny_scaler, decisions, rate=RATE, seed=SEEDS[0], device='cuda', budget=budget,
                epochs=CAPACITY_EPOCHS, checkpoints=CAPACITY_CHECKPOINTS,
                event=lambda e, tag=label: retain(e, tag, tiny, [], tiny_targets, tiny_scaler, tiny_offsets))
            histories.append({'fit': label, 'history': fit['history'], 'starting_control_mse': fit['starting_control_mse'],
                              'final_source_mse': fit['checkpoints'][200]['training_target_source_mse'],
                              'capacity_screen_met': fit['starting_control_mse'] > 0 and fit['checkpoints'][200]['training_target_source_mse'] <= .5 * fit['starting_control_mse']})
        fits = []
        for seed in SEEDS:
            label = f'context_residual_seed{seed}'
            fit = fit_trial_model(train, {'source_fit': list(train), 'held_source': list(held)}, windows, targets,
                {'source_fit': oof, 'held_source': held_offsets}, scaler, decisions, rate=RATE, seed=seed,
                device='cuda', budget=budget, epochs=MAIN_EPOCHS, checkpoints=MAIN_CHECKPOINTS,
                source_inference_offsets=full_training_offsets,
                event=lambda e, tag=label: retain(e, tag, train, held, targets, scaler, {'source_fit': oof, 'held_source': held_offsets}))
            fits.append(fit); histories.append({'fit': label, 'history': fit['history'], 'starting_control_mse': fit['starting_control_mse']})
        metrics = []
        for epoch in MAIN_CHECKPOINTS:
            raw = np.stack([f['checkpoints'][epoch]['held_source_raw'] for f in fits]).mean(axis=0)
            ensemble = np.clip(raw, 0, 7)
            method_mae = participant_film_mae(metric_rows(held, targets, ensemble, metadata))
            ctx_mae = participant_film_mae(metric_rows(held, targets, [held_offsets[p] for p in held], metadata))
            cat_values = [priors['category'][metadata[p]['assigned_category']] for p in held]
            cat_mae = participant_film_mae(metric_rows(held, targets, cat_values, metadata))
            metrics.append({'epoch': epoch, 'ensemble_held_mae': method_mae, 'context_held_mae': ctx_mae,
                            'category_held_mae': cat_mae, 'gain_over_context': ctx_mae - method_mae,
                            'gain_over_category': cat_mae - method_mae})
            for p, value in zip(held, ensemble):
                prediction_rows.append({'fit': 'two_seed_ensemble', 'epoch': epoch, 'arm': 'held_source',
                    'subject': p[0], 'vid': p[1], 'family': metadata[p]['family'], 'target': float(targets[p]), 'prediction_raw': float(value)})
        seed_maes = [participant_film_mae(metric_rows(held, targets, np.clip(f['checkpoints'][360]['held_source_raw'], 0, 7), metadata)) for f in fits]
        flags = learning_flags(fits[0]['starting_control_mse'], [f['checkpoints'][240]['training_target_source_mse'] for f in fits],
            [f['checkpoints'][360]['training_target_source_mse'] for f in fits], seed_maes)
        if len(checkpoint_replays) != 10:
            raise PermissionError('All ten saved checkpoint reloads must qualify')
        save(PUBLIC / 'checkpoint_reload_replay.json', {'passed': True, 'states_checked': 10, 'records': checkpoint_replays,
            'atol_original_scale': CHECKPOINT_ATOL, 'rtol': CHECKPOINT_RTOL,
            'scope': 'Independent CPU tensor-only reload and fixed-subset forward using qualified backbone and same transient source windows; no new fit or architecture reimplementation'}, exclusive=True)
        final_preservation = {'papers': paper_preservation(), 'claims': claim_preservation()}
        with (PUBLIC / 'predictions.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
        save(PUBLIC / 'histories.json', histories, exclusive=True)
        save(PUBLIC / 'comparison.json', {'status': 'completed_source_only_diagnostic', 'metrics': metrics,
            'primary_epoch': 360, 'learning_flags': flags, 'seed_held_maes': seed_maes,
            'source_mse_by_checkpoint': [[f['checkpoints'][e]['training_target_source_mse'] for e in MAIN_CHECKPOINTS] for f in fits],
            'source_offset_only_mse': fits[0]['starting_control_mse'], 'capacity_controls': histories[:2],
            'new_confirmation_candidate': False, 'protected_access': False, 'validation_access': False}, exclusive=True)
        save(PUBLIC / 'execution.json', {'status': 'complete', 'neural_fits': budget.fits,
            'charged_fit_seconds': budget.elapsed, 'total_wall_seconds': time.monotonic() - started,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__),
                        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'torch_cpu_threads': torch.get_num_threads()},
            'private_state_records': states, 'prediction_rows': len(prediction_rows),
            'preservation': final_preservation, 'original_read_bytes': original_read_bytes,
            'protected_access': False, 'validation_access': False, 'question_changed': False}, exclusive=True)
    except Exception as error:
        save(PUBLIC / 'execution.json', {'status': 'held_or_failed', 'error': type(error).__name__ + ': ' + str(error),
            'neural_fits_begun': budget.fits, 'charged_fit_seconds': budget.elapsed,
            'source_trials_replayed': len(windows), 'source_receipts': receipts, 'private_state_records': states,
            'protected_access': False, 'validation_access': False}, exclusive=True)
        raise

def metric_rows(pairs, targets, predictions, metadata):
    return [{'subject': p[0], 'vid': p[1], 'family': metadata[p]['family'],
             'target': float(targets[p]), 'prediction': float(v)} for p, v in zip(pairs, predictions)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['declare', 'activate', 'run'])
    args = parser.parse_args()
    {'declare': declare, 'activate': activate, 'run': execute}[args.stage]()
