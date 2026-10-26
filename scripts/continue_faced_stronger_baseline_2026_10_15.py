"""Fresh source-only continuation: explicit CUDA index, unchanged scientific recipe."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from scripts import run_faced_stronger_baseline_2026_10_14 as previous
from gruxnet.faced_eegnet_runner import SourceChannelScaler, FitBudget, fit_trial_model
from gruxnet.eegnet_regression import EEGNetRegression
from gruxnet.faced_source_eegnet_capacity import verify_reloaded_state, verify_replayed_predictions

OLD_PUBLIC = previous.PUBLIC
OLD_PLAN = previous.PLAN
PUBLIC = previous.BASE / 'faced_stronger_baseline_continuation_2026-10-15'
PRIVATE = REPO.parent / 'publication_runs/faced_stronger_baseline_continuation_2026-10-15'
PLAN = PUBLIC / 'plan.json'
PROTOCOL = REPO / 'docs/publication/FACED_Stronger_Baseline_Continuation_2026-10-15.md'
ORIGINAL_FIT = fit_trial_model
EXTRA_NAMES = [
    'scripts/continue_faced_stronger_baseline_2026_10_15.py',
    'tests/test_faced_stronger_baseline_continuation.py',
    PROTOCOL.relative_to(REPO).as_posix(),
    'docs/publication/FACED_Stronger_Baseline_Held_Attempt_2026-10-15.md',
    *[(OLD_PUBLIC / name).relative_to(REPO).as_posix() for name in (
        'plan_v3.json', 'activation.json', 'source_gate.json', 'target_receipt.json',
        'events.jsonl', 'execution.json', 'held_attempt_audit_2026-10-15.json')]]


def explicit_index_fit(*args, fit_function=None, **kwargs):
    if str(kwargs.get('device')) not in ('cuda', 'cuda:0'):
        raise PermissionError('Continuation permits only the first declared CUDA device')
    kwargs['device'] = 'cuda:0'
    return (ORIGINAL_FIT if fit_function is None else fit_function)(*args, **kwargs)


def verify_previous_hold():
    old = previous.load(OLD_PLAN)
    for relative, expected in old['bindings'].items():
        if previous.digest(REPO / relative) != expected:
            raise PermissionError('Frozen failed-attempt code/input changed')
    receipt = previous.load(OLD_PUBLIC / 'execution.json')
    events = [__import__('json').loads(line) for line in (OLD_PUBLIC / 'events.jsonl').read_text().splitlines()]
    if (receipt['status'] != 'held_or_failed' or receipt['neural_fits_begun'] != 1
            or receipt['private_state_records'] != [] or 'specified index' not in receipt['error']
            or len(events) != 1 or events[0]['history'] != []):
        raise PermissionError('Require the exact preserved pre-optimizer device hold')
    if previous.load(OLD_PUBLIC / 'held_attempt_audit_2026-10-15.json')['passed'] is not True:
        raise PermissionError('Held-attempt audit required')
    return old


def code_bindings():
    old = verify_previous_hold()
    names = list(old['bindings']) + EXTRA_NAMES
    return {name: previous.digest(REPO / name) for name in names}


def declare():
    old = verify_previous_hold()
    previous.paper_preservation(); previous.claim_preservation()
    plan = deepcopy(old)
    plan.update({'date': '2026-10-15', 'declared_utc': previous.now(),
        'status': 'prepared_source_only_not_activated', 'bindings': code_bindings(),
        'continuation_of': OLD_PUBLIC.relative_to(REPO).as_posix(),
        'previous_failed_plan_sha256': previous.digest(OLD_PLAN),
        'implementation_change': 'Only pass explicit cuda:0 through a strict fit adapter; old files/code unchanged; fresh output directories',
        'manufactured_smoke': {'fits': 1, 'optimizer_steps': 2, 'epochs': 2, 'checkpoints': [1, 2],
            'source_trials': 4, 'held_trials': 2, 'seconds_cap': 300, 'cuda_reserved_bytes_cap': 5500000000,
            'original_reads': 0, 'cached_target_reads': 0, 'passed_required_before_real_run': True}})
    previous.save(PLAN, plan, exclusive=True)
    print({'continuation_declared': True, 'source_pairs': 532, 'real_neural_fit_cap': 4,
           'synthetic_steps_cap': 2, 'validation_access': False, 'protected_access': False}, flush=True)


def configure():
    previous.PUBLIC, previous.PRIVATE, previous.PLAN = PUBLIC, PRIVATE, PLAN
    previous.PROTOCOL = PROTOCOL
    previous.bindings = code_bindings
    previous.fit_trial_model = explicit_index_fit


def smoke():
    configure(); previous.check_plan()
    previous.save(PUBLIC / 'smoke_started.json', {'started_utc': previous.now(),
        'plan_sha256': previous.digest(PLAN), 'manufactured_inputs_only': True}, exclusive=True)
    started = time.monotonic(); records = []
    budget = FitBudget(seconds_cap=300, fit_cap=1, reserved_bytes_cap=5500000000)
    try:
        if not torch.cuda.is_available():
            raise PermissionError('Declared CUDA smoke requires first GPU')
        torch.set_num_threads(2)
        rng = np.random.default_rng(20261015)
        source = [('manufactured-source', i) for i in range(4)]
        held = [('manufactured-held', i) for i in range(2)]
        windows = {p: rng.normal(size=(7, 30, 512)).astype('float32') for p in source + held}
        targets = dict(zip(source + held, (2., 3., 4., 5., 1., 6.)))
        scaler = SourceChannelScaler.fit(source, windows)
        offsets = {'source_fit': {p: 3.5 for p in source}, 'held_source': {p: 3.5 for p in held}}
        directory = PRIVATE / 'manufactured_smoke'; directory.mkdir(parents=True, exist_ok=True)

        def event(item):
            if item['type'] != 'checkpoint':
                return
            path = directory / f'epoch{item["epoch"]}.pt'
            if path.exists():
                raise PermissionError('Manufactured checkpoint overwrite forbidden')
            torch.save(item['state_dict'], path)
            model = EEGNetRegression().cpu()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            verify_reloaded_state(model); model.eval()
            pairs = [source[0], source[-1], held[0], held[-1]]
            with torch.no_grad():
                replay = 7 * model(torch.as_tensor(np.stack([scaler.transform(windows[p]) for p in pairs]))).numpy() + 3.5
            expected = [item['source_fit_raw'][0], item['source_fit_raw'][-1], item['held_source_raw'][0], item['held_source_raw'][-1]]
            error = verify_replayed_predictions(expected, replay)
            records.append({'epoch': item['epoch'], 'sha256': previous.digest(path),
                            'maximum_reload_error': error, 'geometry_finite_constraints_passed': True})

        result = explicit_index_fit(source, {'source_fit': source, 'held_source': held}, windows, targets,
            offsets, scaler, {p: {'eeg_available': True} for p in source + held}, rate=.001, seed=20261101,
            device='cuda', budget=budget, epochs=2, checkpoints=(1, 2), event=event)
        if len(result['history']) != 2 or len(records) != 2 or budget.fits != 1:
            raise PermissionError('Manufactured full optimizer/checkpoint path incomplete')
        previous.save(PUBLIC / 'smoke.json', {'passed': True, 'manufactured_only': True, 'optimizer_steps': 2,
            'checkpoint_records': records, 'wall_seconds': time.monotonic() - started,
            'charged_seconds': budget.elapsed, 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_signal_reads': 0, 'cached_target_reads': 0, 'protected_access': False}, exclusive=True)
        print({'manufactured_cuda_smoke_passed': True, 'optimizer_steps': 2, 'checkpoints': 2}, flush=True)
    except Exception as error:
        previous.save(PUBLIC / 'smoke.json', {'passed': False, 'status': 'held_or_failed',
            'error': type(error).__name__ + ': ' + str(error), 'checkpoint_records': records,
            'source_signal_reads': 0, 'cached_target_reads': 0, 'protected_access': False}, exclusive=True)
        raise


def activate():
    configure(); previous.check_plan()
    result = previous.load(PUBLIC / 'smoke.json')
    if result.get('passed') is not True or result.get('optimizer_steps') != 2 or len(result.get('checkpoint_records', [])) != 2:
        raise PermissionError('Frozen manufactured CUDA smoke must pass before source activation')
    previous.activate()


def execute():
    configure(); previous.check_plan()
    if previous.load(PUBLIC / 'smoke.json').get('passed') is not True:
        raise PermissionError('Manufactured smoke must pass before original source access')
    previous.execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['declare', 'smoke', 'activate', 'run'])
    args = parser.parse_args(); {'declare': declare, 'smoke': smoke, 'activate': activate, 'run': execute}[args.stage]()
