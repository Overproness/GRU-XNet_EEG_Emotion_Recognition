"""Separately frozen representative-batch smoke; preserve prior failed smoke."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import continue_faced_stronger_baseline_2026_10_15 as prior
from gruxnet.faced_eegnet_runner import SourceChannelScaler, FitBudget
from gruxnet.eegnet_regression import EEGNetRegression
from gruxnet.faced_source_eegnet_capacity import verify_reloaded_state, verify_replayed_predictions

PUBLIC, PRIVATE = prior.PUBLIC, prior.PRIVATE
PLAN = PUBLIC / 'plan_v2.json'
PROTOCOL = REPO / 'docs/publication/FACED_Representative_Batch_Smoke_2026-10-15.md'
NEW_NAMES = ['scripts/qualify_faced_batch8_cuda_2026_10_15.py',
    'tests/test_faced_batch8_cuda.py', PROTOCOL.relative_to(REPO).as_posix(),
    *[(PUBLIC / n).relative_to(REPO).as_posix() for n in ('plan.json', 'smoke_started.json', 'smoke.json', 'smoke_hold_audit_2026-10-15.json')],
    'docs/publication/FACED_Stronger_Baseline_Smoke_Hold_2026-10-15.md']


def fabricated_inputs():
    rng = np.random.default_rng(20261015)
    source = [('manufactured-source', i) for i in range(8)]
    held = [('manufactured-held', i) for i in range(4)]
    windows = {p: rng.normal(size=(7, 30, 512)).astype('float32') for p in source + held}
    targets = dict(zip(source + held, (0., 1., 2., 3., 4., 5., 6., 7., 1., 2., 5., 6.)))
    offsets = {'source_fit': {p: 3.5 for p in source}, 'held_source': {p: 3.5 for p in held}}
    return source, held, windows, targets, offsets


def code_bindings():
    prior.verify_previous_hold()
    previous_plan = prior.previous.load(prior.PLAN)
    for relative, expected in previous_plan['bindings'].items():
        if prior.previous.digest(REPO / relative) != expected:
            raise PermissionError('Frozen initial continuation changed')
    return {name: prior.previous.digest(REPO / name) for name in list(previous_plan['bindings']) + NEW_NAMES}


def declare():
    failed = prior.previous.load(PUBLIC / 'smoke.json')
    if failed.get('passed') is not False or failed.get('checkpoint_records') != [] or 'CUDNN_STATUS_INTERNAL_ERROR' not in failed['error']:
        raise PermissionError('Exact failed manufactured smoke must remain intact')
    if (PUBLIC / 'activation.json').exists() or (PUBLIC / 'execution_started.json').exists():
        raise PermissionError('No original-data continuation may have started')
    plan = deepcopy(prior.previous.load(prior.PLAN))
    plan.update({'declared_utc': prior.previous.now(), 'bindings': code_bindings(),
        'previous_unexecuted_source_plan_sha256': prior.previous.digest(prior.PLAN),
        'representative_smoke_reason': 'Test real training batch8 and held-inference remainder4; batch geometry is an unproven cause hypothesis, not a scientific model choice',
        'smoke_restart_barrier': 'Old four-source failed smoke is preserved; only new smoke_batch8.json under this separately reviewed plan can permit a new source activation',
        'manufactured_smoke': {'fits': 1, 'optimizer_steps': 2, 'epochs': 2, 'checkpoints': [1, 2],
            'source_trials': 8, 'held_trials': 4, 'seconds_cap': 300,
            'cuda_reserved_bytes_cap': 5500000000, 'original_reads': 0, 'cached_target_reads': 0}})
    prior.previous.save(PLAN, plan, exclusive=True)
    print({'representative_batch_smoke_declared': True, 'synthetic_steps': 2, 'source_activation': False}, flush=True)


def configure():
    prior.configure()
    prior.previous.PLAN, prior.previous.PROTOCOL = PLAN, PROTOCOL
    prior.previous.bindings = code_bindings


def smoke():
    configure(); prior.previous.check_plan()
    prior.previous.save(PUBLIC / 'smoke_batch8_started.json', {'started_utc': prior.previous.now(),
        'plan_sha256': prior.previous.digest(PLAN), 'manufactured_only': True}, exclusive=True)
    budget = FitBudget(seconds_cap=300, fit_cap=1, reserved_bytes_cap=5500000000)
    started = time.monotonic(); records = []
    try:
        if not torch.cuda.is_available():
            raise PermissionError('Representative smoke requires the declared first CUDA device')
        torch.set_num_threads(2)
        source, held, windows, targets, offsets = fabricated_inputs()
        scaler = SourceChannelScaler.fit(source, windows)
        directory = PRIVATE / 'manufactured_smoke_batch8'; directory.mkdir(parents=True, exist_ok=True)

        def event(item):
            if item['type'] != 'checkpoint':
                return
            path = directory / f'epoch{item["epoch"]}.pt'
            if path.exists():
                raise PermissionError('Representative manufactured checkpoint overwrite forbidden')
            torch.save(item['state_dict'], path)
            model = EEGNetRegression().cpu()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            verify_reloaded_state(model); model.eval()
            pairs = [source[0], source[-1], held[0], held[-1]]
            with torch.no_grad():
                replay = 7 * model(torch.as_tensor(np.stack([scaler.transform(windows[p]) for p in pairs]))).numpy() + 3.5
            expected = [item['source_fit_raw'][0], item['source_fit_raw'][-1], item['held_source_raw'][0], item['held_source_raw'][-1]]
            records.append({'epoch': item['epoch'], 'sha256': prior.previous.digest(path),
                'maximum_reload_error': verify_replayed_predictions(expected, replay),
                'geometry_finite_constraints_passed': True})

        result = prior.explicit_index_fit(source, {'source_fit': source, 'held_source': held}, windows, targets,
            offsets, scaler, {p: {'eeg_available': True} for p in source + held}, rate=.001, seed=20261101,
            device='cuda', budget=budget, epochs=2, checkpoints=(1, 2), event=event)
        if len(result['history']) != 2 or len(records) != 2 or budget.fits != 1:
            raise PermissionError('Representative optimizer/checkpoint path incomplete')
        prior.previous.save(PUBLIC / 'smoke_batch8.json', {'passed': True, 'manufactured_only': True,
            'source_trials': 8, 'held_trials': 4, 'optimizer_steps': 2, 'checkpoint_records': records,
            'wall_seconds': time.monotonic() - started, 'charged_seconds': budget.elapsed,
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_signal_reads': 0, 'cached_target_reads': 0, 'protected_access': False}, exclusive=True)
        print({'representative_cuda_smoke_passed': True, 'optimizer_steps': 2, 'checkpoints': 2}, flush=True)
    except Exception as error:
        prior.previous.save(PUBLIC / 'smoke_batch8.json', {'passed': False, 'status': 'held_or_failed',
            'error': type(error).__name__ + ': ' + str(error), 'checkpoint_records': records,
            'source_signal_reads': 0, 'cached_target_reads': 0, 'protected_access': False}, exclusive=True)
        raise


def require_new_smoke():
    result = prior.previous.load(PUBLIC / 'smoke_batch8.json')
    if result.get('passed') is not True or result.get('optimizer_steps') != 2 or len(result.get('checkpoint_records', [])) != 2:
        raise PermissionError('New representative-batch smoke must pass; old failure cannot be reused as permission')


def activate():
    configure(); prior.previous.check_plan(); require_new_smoke(); prior.previous.activate()


def execute():
    configure(); prior.previous.check_plan(); require_new_smoke(); prior.previous.execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['declare', 'smoke', 'activate', 'run'])
    args = parser.parse_args(); {'declare': declare, 'smoke': smoke, 'activate': activate, 'run': execute}[args.stage]()
