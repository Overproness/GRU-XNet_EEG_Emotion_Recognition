"""Independent source-fixture forward replay; never fits or changes old packets.

Declaration reads identities and opaque file hashes only. Execution requires a
new bound activation. Original EEG is decoded only for declared source fixtures;
the validation pairing diagnostic is a separate cached-output transformation.
"""
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
import scipy
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO))
from gruxnet.faced_eegnet_bridge import array_sha256
from gruxnet.faced_classical_contract import context_features
from gruxnet.faced_source_quality import read_interval, selected_header, quality_metrics
from gruxnet.faced_proposal_preprocessing_v2 import calibrate_common_counts
from gruxnet.faced_artifact_control import artifact_trial_windows
from gruxnet.faced_inference_quality import eeg_availability
from scripts.audit_faced_raw import read_header
from scripts.run_faced_eegnet_diagnostic_v2 import metadata_bundle
from scripts.verify_faced_eegnet_states import state_geometry

BASE = REPO / 'results/development'
OLD = BASE / 'faced_eegnet_diagnostic_v2_2026-10-11'
ARTIFACT = BASE / 'faced_artifact_control_2026-10-11'
PRIVATE = WORKSPACE / 'publication_runs/faced_eegnet_diagnostic_v2_2026-10-11'
ARTIFACT_PRIVATE = WORKSPACE / 'publication_runs/faced_artifact_control_2026-10-11'
RAW = WORKSPACE / 'Data/Data'
SEEDS = (20261101, 20261102)
RATE = .003
EPOCH = 120
TOLERANCE = 1e-5
SALT = 'FACED-source-checkpoint-replay-2026-10-12-v1'
PREPARATION_CAP = 7200
FORWARD_CAP = 1800


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def under(path, root):
    resolved = Path(path).resolve()
    resolved.relative_to(Path(root).resolve())
    return resolved


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'preprocessing_blas_threads': 12,
            'torch_cpu_threads': 2}


def fixture_pairs(programme, metadata, count=4):
    """Outcome-free fixture selection, one per acquisition then salted fill."""
    if count != 4:
        raise PermissionError('Exactly four declared source fixtures per rotation')
    result = {}
    for manifest in programme['rotation_manifests']:
        rotation = int(manifest['rotation'])
        pairs = [(r['subject'], int(r['vid'])) for r in manifest['arms']['source_fit']]
        if len(pairs) != 560 or len(set(pairs)) != 560:
            raise PermissionError('Changed complete source arm')
        if any(p not in metadata or metadata[p]['role'] != 'source' for p in pairs):
            raise PermissionError('Source fixture selection encountered protected/validation identity')
        def order(pair):
            raw = f'{SALT}|r{rotation}|{pair[0]}|{pair[1]}'.encode()
            return hashlib.sha256(raw).hexdigest()
        ordered = sorted(pairs, key=order)
        groups = sorted({metadata[p]['acquisition_group'] for p in ordered})
        chosen = [next(p for p in ordered if metadata[p]['acquisition_group'] == g) for g in groups]
        for p in ordered:
            if len(chosen) >= count:
                break
            if p not in chosen:
                chosen.append(p)
        if len(chosen) != count or len(set(chosen)) != count:
            raise PermissionError('Acquisition fixture coverage exceeds fixed cap')
        result[rotation] = chosen
    if set(result) != {0, 1, 2}:
        raise PermissionError('Three exact rotations required')
    return result


def source_context_records(programme, fixtures, manifest):
    """Select existing full and necessary OOF context states without their values."""
    selected = []
    for rotation in programme['rotation_manifests']:
        k = int(rotation['rotation'])
        full = [(r['subject'], int(r['vid'])) for r in rotation['arms']['source_fit']]
        folds = programme['source_participant_oof_folds']
        needed = {None} | {i for i, people in enumerate(folds)
                           if any(p[0] in people for p in fixtures[k])}
        for fold in sorted(needed, key=lambda value: -1 if value is None else value):
            expected = set(full) if fold is None else {p for p in full if p[0] not in folds[fold]}
            found = [r for r in manifest['records'] if r['rotation'] == k
                     and r['family'] == 'context_only' and r['alpha'] == {0: .1, 1: 100., 2: 100.}[k]
                     and r['oof_fold'] == fold]
            if len(found) != 1:
                raise PermissionError('Exactly one previously fitted context state required')
            r = found[0]
            pairs = [(x['subject'], int(x['vid'])) for x in r['source_pairs']]
            if len(pairs) != len(expected) or set(pairs) != expected:
                raise PermissionError('Context fitted source identities changed')
            source_identity = hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()
            if source_identity != r['source_identity_sha256']:
                raise ValueError('Context fitted source identity hash changed')
            path = ARTIFACT_PRIVATE / f"model_{r['index']:03d}.npz"
            if digest(path) != r['private_state_sha256']:
                raise ValueError('Existing context state opaque hash changed')
            selected.append({'rotation': k, 'oof_fold': fold, 'path': path.relative_to(WORKSPACE).as_posix(),
                             'sha256': r['private_state_sha256'], 'source_identity_sha256': source_identity})
    return selected


def input_paths():
    names = [OLD / n for n in ('activation.json', 'execution.json', 'fit_records.json',
             'selection.json', 'source_normalizers.json', 'checkpoint_predictions.csv',
             'feature_manifest.json', 'source_gate.json', 'completion/verification.json')]
    names += [ARTIFACT / n for n in ('model_manifest.json', 'source_records.json', 'validation_records.json')]
    names += [BASE / 'faced_inference_quality_2026-10-11/signal_audit.json',
              BASE / 'faced_development_protocol_2026-10-11/plan.json',
              BASE / 'faced_backup_2026-10-10/participant_reservation.json',
              BASE / 'faced_codebooks_2026-10-10/audit.json',
              BASE / 'faced_raw_2026-10-11/headers.json', BASE / 'faced_raw_2026-10-11/extraction.json']
    names += [REPO / name for name in ('gruxnet/faced_eegnet_bridge.py', 'gruxnet/faced_eegnet_runner.py',
              'gruxnet/eegnet_control.py', 'gruxnet/eegnet_regression.py',
              'gruxnet/faced_classical_contract.py', 'gruxnet/faced_source_quality.py',
              'gruxnet/faced_proposal_preprocessing.py', 'gruxnet/faced_proposal_preprocessing_v2.py',
              'gruxnet/faced_source_notch.py', 'gruxnet/faced_artifact_control.py',
              'gruxnet/faced_inference_quality.py', 'scripts/audit_faced_raw.py',
              'scripts/run_faced_eegnet_diagnostic_v2.py', 'scripts/verify_faced_eegnet_states.py')]
    return names


def declare(activation, output):
    activation = under(activation, REPO)
    output = under(output, BASE)
    programme, records, metadata, decisions = metadata_bundle()
    fixtures = fixture_pairs(programme, metadata)
    contexts = source_context_records(programme, fixtures, load(ARTIFACT / 'model_manifest.json'))
    selected = []
    for k in range(3):
        for seed in SEEDS:
            path = PRIVATE / f'r{k}_context_residual_lr{RATE}_s{seed}_e{EPOCH}.pt'
            selected.append({'rotation': k, 'seed': seed, 'learning_rate': RATE, 'epoch': EPOCH,
                             'path': path.relative_to(WORKSPACE).as_posix(), 'sha256': digest(path)})
    quality_bindings = {}
    for pair in sorted({p for pairs in fixtures.values() for p in pairs}):
        r = records[pair]
        if r['role'] != 'source' or not decisions[pair]['eeg_available']:
            raise PermissionError('Source fixture quality changed')
        for suffix, field in (('quality', 'quality_file_sha256'), ('artifact', 'artifact_file_sha256')):
            path = ARTIFACT_PRIVATE / f'{pair[0]}_clip{pair[1]:02d}_{suffix}.json'
            if digest(path) != r[field]:
                raise ValueError('Opaque source fixture technical cache changed')
            quality_bindings[path.relative_to(WORKSPACE).as_posix()] = r[field]
    plan = {'date': '2026-10-12', 'scope': 'Four outcome-independent source fixtures per rotation; six selected residual checkpoints only',
            'activation_path': activation.relative_to(REPO).as_posix(),
            'script_sha256': digest(Path(__file__)), 'runtime': runtime(), 'fixture_salt': SALT,
            'fixture_ids': {str(k): [{'subject': p, 'vid': v} for p, v in pairs] for k, pairs in fixtures.items()},
            'bindings': {p.relative_to(REPO).as_posix(): digest(p) for p in input_paths()},
            'selected_states': selected, 'context_states': contexts, 'quality_bindings': quality_bindings,
            'prediction_tolerance': TOLERANCE, 'preparation_seconds_cap': PREPARATION_CAP,
            'forward_seconds_cap': FORWARD_CAP, 'source_inference_offset': 'Full-source clipped context; separately reproduce OOF training offset source_fit',
            'original_rating_decoding': 0, 'new_fits': 0, 'validation_forward_replay_performed': False,
            'confirmation_access': False, 'research_question_changed': False}
    save_new(output / 'checkpoint_replay_plan.json', plan)
    print(json.dumps({'checkpoint_replay_declared': True, 'fixtures_per_rotation': 4, 'selected_states': 6,
                      'targets_or_predictions_converted': 0, 'original_signal_decoded': 0}), flush=True)


def check_activation(activation, plan, plan_path):
    a = load(activation)
    if a.get('status') not in ('active_development_diagnosis', 'active_development_diagnostic'):
        raise PermissionError('New diagnosis activation required before numeric access')
    if a.get('confirmation_access') is not False or a.get('research_question_changed') is not False:
        raise PermissionError('Protected/RQ access prohibited')
    if a.get('checkpoint_replay_plan_sha256') != digest(plan_path):
        raise PermissionError('Replay declaration not bound by new activation')
    script_name = Path(__file__).relative_to(REPO).as_posix()
    bindings = {**a.get('bindings', {}), **a.get('code_bindings', {})}
    if bindings.get(script_name) != digest(Path(__file__)) or plan['script_sha256'] != digest(Path(__file__)):
        raise PermissionError('Replay executable not bound before new execution')
    if plan['runtime'] != runtime() or plan['prediction_tolerance'] != TOLERANCE:
        raise ValueError('Declared runtime/tolerance changed')
    for relative, expected in plan['bindings'].items():
        if digest(under(REPO / relative, REPO)) != expected:
            raise ValueError('Prior immutable dependency changed: ' + relative)
    for r in plan['selected_states'] + plan['context_states']:
        if digest(under(WORKSPACE / r['path'], WORKSPACE)) != r['sha256']:
            raise ValueError('Bound private fitted state changed')
    for relative, expected in plan['quality_bindings'].items():
        if digest(under(WORKSPACE / relative, WORKSPACE)) != expected:
            raise ValueError('Bound source fixture QC cache changed')
    return a


def independent_forward(state, trials):
    """Direct inference operators; no producing model/runner forward is called."""
    if trials.ndim != 4 or trials.shape[1:] != (7, 30, 512) or not torch.isfinite(trials).all():
        raise ValueError('Invalid normalized trial input')
    x = trials.reshape(-1, 30, 512).unsqueeze(1)
    x = F.conv2d(F.pad(x, (31, 32)), state['temporal.weight'])
    def bn(values, name):
        return F.batch_norm(values, state[name + '.running_mean'], state[name + '.running_var'],
                            state[name + '.weight'], state[name + '.bias'], training=False, eps=1e-3)
    x = bn(x, 'bn1')
    x = F.conv2d(x, state['spatial.weight'], groups=8)
    x = F.avg_pool2d(F.elu(bn(x, 'bn2')), (1, 4))
    x = F.conv2d(F.pad(x, (7, 8)), state['depthwise.weight'], groups=16)
    x = F.conv2d(x, state['pointwise.weight'])
    x = F.avg_pool2d(F.elu(bn(x, 'bn3')), (1, 8))
    flattened = x.permute(0, 2, 3, 1).reshape(len(x), -1)
    value = F.linear(flattened, state['head.weight'], state['head.bias']).squeeze(-1)
    return value.reshape(len(trials), 7).mean(dim=1)


def existing_context_prediction(path, vectors):
    with np.load(path, allow_pickle=False) as state:
        if set(state.files) != {'mean', 'scale', 'variance', 'coef', 'intercept'}:
            raise ValueError('Unexpected saved contextual state schema')
        mean, scale, coef = (np.asarray(state[n], dtype=np.float64) for n in ('mean', 'scale', 'coef'))
        intercept = float(state['intercept'])
    if mean.shape != (13,) or scale.shape != (13,) or coef.shape != (13,) or not np.isfinite(np.r_[mean, scale, coef, intercept]).all() or np.any(scale <= 0):
        raise ValueError('Invalid full-context fitted parameters')
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 13 or not np.isfinite(x).all():
        raise ValueError('Invalid recipient context vectors')
    return np.clip(((x - mean) / scale) @ coef + intercept, 0, 7)


def guarded_fixture_predictions(path, fixtures, metadata):
    """Skip nonfixtures and nonselected rows before converting predictions."""
    lookup = {}
    source = path if hasattr(path, 'read') else Path(path).open(newline='', encoding='utf-8')
    with source as stream:
        for row in csv.DictReader(stream):
            if row['readout'] != 'context_residual' or row['learning_rate'] != str(RATE) or row['epoch'] != str(EPOCH) or row['arm'] not in ('source_fit', 'source_inference'):
                continue
            k = int(row['rotation'])
            p = row['subject'], int(row['vid'])
            if k not in fixtures or p not in fixtures[k]:
                continue
            if p not in metadata or metadata[p]['role'] != 'source' or row['family'] != metadata[p]['family'] or row['acquisition_group'] != metadata[p]['acquisition_group']:
                raise PermissionError('Fixture identity/role guard before prediction conversion')
            if row['eeg_available'] != 'True' or row['fallback'] != 'False' or int(row['seed']) not in SEEDS:
                raise PermissionError('Changed source fixture availability or seed')
            key = k, int(row['seed']), row['arm'], p
            if key in lookup:
                raise PermissionError('Duplicated selected source fixture prediction')
            value = float(row['prediction_raw'])
            if not np.isfinite(value):
                raise ValueError('Invalid cached fixture prediction')
            lookup[key] = value
    expected = {(k, seed, arm, p) for k, pairs in fixtures.items() for seed in SEEDS
                for arm in ('source_fit', 'source_inference') for p in pairs}
    if set(lookup) != expected:
        raise PermissionError('Complete selected source fixture predictions required')
    return lookup


def run(activation, output, device):
    activation = under(activation, REPO)
    output = under(output, BASE)
    destination = output / 'checkpoint_replay.json'
    if destination.exists() or (output / 'checkpoint_replay_started.json').exists():
        raise PermissionError('No overwrite or automatic resume of a replay attempt')
    plan_path = output / 'checkpoint_replay_plan.json'
    plan = load(plan_path)
    if plan['activation_path'] != activation.relative_to(REPO).as_posix():
        raise PermissionError('New activation identity changed')
    check_activation(activation, plan, plan_path)
    save_new(output / 'checkpoint_replay_started.json', {'activation_sha256': digest(activation),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'new_fits': 0})
    started = time.monotonic()
    receipt = {'passed': False, 'source_fixture_replay_passed': False, 'activation_sha256': digest(activation),
               'plan_sha256': digest(plan_path), 'fixture_ids': plan['fixture_ids'],
               'selected_states': plan['selected_states'], 'checkpoint_csv_sha256': plan['bindings'][(OLD / 'checkpoint_predictions.csv').relative_to(REPO).as_posix()],
               'normalization_bindings': {(OLD / 'source_normalizers.json').relative_to(REPO).as_posix(): plan['bindings'][(OLD / 'source_normalizers.json').relative_to(REPO).as_posix()]},
               'context_state_bindings': plan['context_states'], 'fixtures': [], 'prediction_checks': [],
               'targets_converted': 0, 'new_fits': 0, 'original_rating_decoding': 0,
               'validation_original_samples_decoded': 0, 'protected_original_samples_decoded': 0,
               'validation_forward_replay_performed': False, 'confirmation_access': False,
               'research_question_changed': False, 'raw_waveforms_persisted': False,
               'scope': 'Only selected source fixtures forward replayed; validation donors use guarded cached residual corrections, not full validation forward replay'}
    try:
        programme, records, metadata, decisions = metadata_bundle()
        fixtures = fixture_pairs(programme, metadata)
        if {str(k): [{'subject': p, 'vid': v} for p, v in pairs] for k, pairs in fixtures.items()} != plan['fixture_ids']:
            raise PermissionError('Outcome-independent source fixture identities changed')
        unique = sorted({p for values in fixtures.values() for p in values})
        headers = {r['subject']: r['data.bdf'] for r in load(BASE / 'faced_raw_2026-10-11/headers.json')['records']}
        original = {r['member']: r for r in load(BASE / 'faced_raw_2026-10-11/extraction.json')['records']}
        authenticated = {}
        windows = {}
        def preparation_budget():
            if time.monotonic() - started > PREPARATION_CAP:
                raise TimeoutError('Declared source fixture preparation cap')
        for p in unique:
            preparation_budget()
            r = records[p]
            if r['role'] != 'source' or not decisions[p]['eeg_available']:
                raise PermissionError('No validation/protected/unavailable fixture decoding')
            directory = RAW / p[0].replace('sub-', 'sub')
            path = directory / 'data.bdf'
            if p[0] not in authenticated:
                if read_header(path) != headers[p[0]]:
                    raise ValueError('Original source BDF header changed')
                for name in ('data.bdf', 'evt.bdf'):
                    if digest(directory / name) != original[f'Data/{directory.name}/{name}']['sha256']:
                        raise ValueError('Original source recording or opaque companion changed')
                authenticated[p[0]] = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                                       'sha256': original[f'Data/{directory.name}/data.bdf']['sha256']}
            if r['original_data_sha256'] != authenticated[p[0]]['sha256']:
                raise ValueError('Source fixture raw lineage changed')
            with path.open('rb') as stream:
                counts, access = read_interval(stream, headers[p[0]], *r['sample_interval'], r['channel_indices'], True)
            if array_sha256(counts) != r['count_array_sha256'] or access['range_ledger_sha256'] != r['range_ledger_sha256']:
                raise ValueError('Selected source counts/access ledger changed')
            values = calibrate_common_counts(counts, p[0], selected_header(headers[p[0]], r['channel_indices']))
            with threadpool_limits(limits=12, user_api='blas'):
                trial, artifact = artifact_trial_windows(values, r['native_rate_hz'])
            q = quality_metrics(counts, values, trial, r['native_rate_hz'])
            stem = f'{p[0]}_clip{p[1]:02d}'
            if q != load(ARTIFACT_PRIVATE / f'{stem}_quality.json') or artifact != load(ARTIFACT_PRIVATE / f'{stem}_artifact.json'):
                raise ValueError('Exact source numerical QC/artifact replay changed')
            decision = eeg_availability(q, technical_passed=True, repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
            if decision != {key: decisions[p][key] for key in ('policy_id', 'eeg_available', 'reasons')}:
                raise ValueError('Source fixture availability changed')
            if array_sha256(trial) != r['processed_window_sha256']:
                raise ValueError('Source processed-window bytes changed')
            trial.flags.writeable = False
            windows[p] = trial
            receipt['fixtures'].append({'subject': p[0], 'vid': p[1], 'role': 'source',
                   'acquisition_group': r['acquisition_group'], 'sample_values_decoded': int(counts.size),
                   'count_array_sha256': r['count_array_sha256'], 'range_ledger_sha256': r['range_ledger_sha256'],
                   'processed_window_sha256': r['processed_window_sha256'], 'exact_qc_and_artifact_replayed': True})
            del counts, values
        for person, expected in authenticated.items():
            preparation_budget()
            path = RAW / person.replace('sub-', 'sub') / 'data.bdf'
            if path.stat().st_size != expected['size'] or path.stat().st_mtime_ns != expected['mtime_ns'] or digest(path) != expected['sha256']:
                raise ValueError('Source original changed after selective decoding')
        preparation_budget()
        receipt['preparation_seconds'] = time.monotonic() - started
        # Only source fixture predictions are converted, after source signal replay.
        observed = guarded_fixture_predictions(OLD / 'checkpoint_predictions.csv', fixtures, metadata)
        normalizers = {r['rotation']: r for r in load(OLD / 'source_normalizers.json')['rotations']}
        fits = load(OLD / 'fit_records.json')['fits']
        selection = load(OLD / 'selection.json')['selection']['context_residual']
        if selection['learning_rate'] != RATE or selection['epoch'] != EPOCH:
            raise PermissionError('Original selected residual schedule changed')
        torch.set_num_threads(2)
        requested = torch.device(device)
        if requested.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Declared CUDA device unavailable')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        forward_started = time.monotonic()
        for k, pairs in fixtures.items():
            norm = normalizers[k]
            full = [(r['subject'], int(r['vid'])) for r in programme['rotation_manifests'][k]['arms']['source_fit']]
            if [tuple(p) for p in norm['source_pairs']] != full or norm['context_alpha'] != {0: .1, 1: 100., 2: 100.}[k]:
                raise PermissionError('Source-only normalization identities changed')
            mean, sd = np.asarray(norm['channel_mean_uv'], dtype=float), np.asarray(norm['channel_sd_uv'], dtype=float)
            if mean.shape != (30,) or sd.shape != (30,) or not np.isfinite(np.r_[mean, sd]).all() or np.any(sd < 1e-6):
                raise ValueError('Invalid source channel normalization')
            tensors = np.stack([((windows[p].astype(np.float64) - mean[None, :, None]) / sd[None, :, None]).astype(np.float32) for p in pairs])
            full_state = next(r for r in plan['context_states'] if r['rotation'] == k and r['oof_fold'] is None)
            context = np.stack([context_features(metadata[p]) for p in pairs])
            offsets_full = existing_context_prediction(WORKSPACE / full_state['path'], context)
            offsets_oof = []
            for p in pairs:
                fold = next(i for i, people in enumerate(programme['source_participant_oof_folds']) if p[0] in people)
                state = next(r for r in plan['context_states'] if r['rotation'] == k and r['oof_fold'] == fold)
                offsets_oof.append(existing_context_prediction(WORKSPACE / state['path'], [context_features(metadata[p])])[0])
            offsets_oof = np.asarray(offsets_oof, dtype=np.float64)
            for selected in (r for r in plan['selected_states'] if r['rotation'] == k):
                if time.monotonic() - forward_started > FORWARD_CAP:
                    raise TimeoutError('Declared checkpoint forward/inference cap')
                matching = [r for r in fits if r['rotation'] == k and r['readout'] == 'context_residual'
                            and r['learning_rate'] == RATE and r['seed'] == selected['seed']]
                if len(matching) != 1 or matching[0]['status'] != 'completed':
                    raise PermissionError('One completed selected checkpoint fit required')
                cp = matching[0]['checkpoints'][str(EPOCH)]
                if cp['state_dict_sha256'] != selected['sha256'] or cp['private_state_file'] != Path(selected['path']).name:
                    raise ValueError('Selected checkpoint differs from fit ledger')
                loaded = torch.load(WORKSPACE / selected['path'], map_location='cpu', weights_only=True)
                state_geometry(loaded, EPOCH)
                state = {key: value.to(requested) for key, value in loaded.items()}
                with torch.no_grad():
                    normalized = independent_forward(state, torch.as_tensor(tensors, device=requested))
                    # Match original float32 OOF addition, followed by the exact
                    # float64 full-source substitution used in source_inference.
                    training = (7 * normalized + torch.as_tensor(offsets_oof, dtype=normalized.dtype, device=requested)).cpu().numpy().astype(np.float64)
                inference = training - offsets_oof + offsets_full
                for i, p in enumerate(pairs):
                    for arm, value in (('source_fit', training[i]), ('source_inference', inference[i])):
                        expected = observed[k, selected['seed'], arm, p]
                        difference = abs(float(value) - expected)
                        receipt['prediction_checks'].append({'rotation': k, 'seed': selected['seed'], 'arm': arm,
                           'subject': p[0], 'vid': p[1], 'cached_raw_prediction': expected,
                           'replayed_raw_prediction': float(value), 'absolute_difference': difference})
                        if not np.isfinite(value) or difference > TOLERANCE:
                            raise ValueError('Independent raw source forward replay exceeds fixed tolerance')
                del state, loaded
        if requested.type == 'cuda':
            torch.cuda.synchronize(requested)
        if time.monotonic() - forward_started > FORWARD_CAP:
            raise TimeoutError('Declared checkpoint forward/inference cap')
        check_activation(activation, plan, plan_path)
        receipt.update(passed=True, source_fixture_replay_passed=True,
                       forward_seconds=time.monotonic() - forward_started,
                       selected_checkpoints_forward_replayed=6, source_fixture_checks=48,
                       prediction_tolerance=TOLERANCE,
                       maximum_absolute_prediction_difference=max(r['absolute_difference'] for r in receipt['prediction_checks']),
                       normalized_input_sha256_by_rotation={str(k): array_sha256(np.stack([((windows[p].astype(np.float64) - np.asarray(normalizers[k]['channel_mean_uv'])[None, :, None]) / np.asarray(normalizers[k]['channel_sd_uv'])[None, :, None]).astype(np.float32) for p in pairs])) for k, pairs in fixtures.items()},
                       device=str(requested), runtime=runtime())
    except Exception as error:
        receipt['failure'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        save_new(destination, receipt)
        print(json.dumps({'source_fixture_replay_passed': receipt['source_fixture_replay_passed'],
                          'validation_forward_replay_performed': False, 'elapsed_seconds': receipt['elapsed_seconds'],
                          'failure': receipt.get('failure')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--declare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--activation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.declare:
        declare(args.activation, args.output)
    else:
        run(args.activation, args.output, args.device)


if __name__ == '__main__':
    main()
