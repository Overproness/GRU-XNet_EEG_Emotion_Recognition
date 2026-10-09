"""Declare, prepare and run the bounded source-only preprocessing diagnostic."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.preprocessing_diagnostic import (STUDY, declare, prepare, validate, sha,
    stamp, atomic, case_id, check_cache, fit_classical, fit_neural, selection_key)
from gruxnet.eegnet_control import EEGNetControl
from gruxnet.train import seed_everything
from scripts.audit_preprocessing_diagnostic import replay_classical, replay_neural
from scripts.export_preprocessing_diagnostic import export


def publish(output, message, push=False):
    export(output)
    if not push:
        return
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True)
    if git('branch', '--show-current').stdout.strip() != 'main':
        raise ValueError('Publisher requires main')
    if git('remote', 'get-url', 'origin').stdout.strip() != 'https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition.git':
        raise ValueError('Unexpected publication destination')
    owned = f'results/development/{STUDY}'
    staged = git('diff', '--cached', '--name-only').stdout.splitlines()
    if any(not (p == owned or p.startswith(owned+'/')) for p in staged):
        raise ValueError('Unrelated staged changes; stop publication')
    git('add', '--', owned)
    if git('diff', '--cached', '--name-only').stdout.strip():
        git('commit', '-m', message)
        git('push', 'origin', 'main')


def config(output):
    atomic(output/'config.json', {'plan_sha256': sha(output/'plan.json'),
        'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(),
        'input_bindings': {d: sha(output/'inputs'/d/'prepared.json') for d in ('deap', 'seediv')}})


def summary(output):
    result = {'development_only': True, 'research_question_change_approved': False,
              'neural': [], 'classical': []}
    grouped = {}
    for folder in sorted((output/'fits').glob('*')):
        if not (folder/'verification.json').exists():
            continue
        record = json.loads((folder/'record.json').read_text())
        job = record['job']
        key = (job['dataset'], job['group'], json.dumps(job['recipe'], sort_keys=True))
        grouped.setdefault(key, []).extend([{**c, 'lr': job['lr'], 'fit': folder.name} for c in record['candidates']])
    for key, candidates in grouped.items():
        result['neural'].append({'dataset': key[0], 'group': key[1], 'recipe': json.loads(key[2]),
                                 'selected': min(candidates, key=selection_key), 'candidates': candidates})
    for folder in sorted((output/'classical').glob('*')):
        if (folder/'verification.json').exists():
            result['classical'].append(json.loads((folder/'record.json').read_text()))
    atomic(output/'summary.json', result)
    return result


def run(root, limit=None, push=False):
    output = root/STUDY
    plan = validate(output, root)
    declaration = json.loads((output/'config.json').read_text())
    if declaration['plan_sha256'] != sha(output/'plan.json'):
        raise ValueError('Changed config')
    for name, actual in (('torch', torch.__version__), ('numpy', np.__version__), ('python', sys.version),
                         ('cuda', torch.version.cuda), ('cudnn', torch.backends.cudnn.version()), ('gpu', torch.cuda.get_device_name())):
        if declaration[name] != actual:
            raise ValueError('Changed declared environment')
    for dataset in ('deap', 'seediv'):
        cache = output/'inputs'/dataset
        check_cache(cache)
        if sha(cache/'prepared.json') != declaration['input_bindings'][dataset]:
            raise ValueError('Changed declared inputs')
    completed, new = 0, 0
    started = time.perf_counter()
    for job in plan['jobs']:
        validate(output, root)
        folder = output/'classical'/f'{job["dataset"].lower()}_g{job["group"]}'
        fit_classical(job['dataset'], job['group'], output)
        if not (folder/'verification.json').exists():
            replay_classical(output, folder)
        folder = output/'fits'/case_id(job)
        existed = (folder/'verification.json').exists()
        fit_neural(job, output)
        if not existed:
            replay_neural(output, folder)
            new += 1
        completed += 1
        progress = {'state': 'running', 'updated_utc': stamp(), 'pid': os.getpid(),
                    'neural_completed': completed, 'neural_total': len(plan['jobs']),
                    'latest_case': folder.name, 'worker_elapsed_seconds': time.perf_counter()-started,
                    'research_question_change_approved': False}
        atomic(output/'progress.json', progress)
        print(json.dumps(progress), flush=True)
        if new and new % 6 == 0:
            summary(output)
            publish(output, f'Checkpoint preprocessing diagnostic {completed}/72 verified trajectories', push)
        if limit and new >= limit:
            summary(output)
            publish(output, f'Checkpoint preprocessing diagnostic {completed}/72 verified trajectories', push)
            return
    result = summary(output)
    if len(result['neural']) != 36 or len(result['classical']) != 4:
        raise ValueError('Incomplete panel/recipe coverage')
    atomic(output/'verification.json', {'passed': True, 'complete': True, 'neural_trajectories': 72,
        'candidate_states_replayed': 288, 'classical_candidates_refitted': 80,
        'plan_sha256': sha(output/'plan.json'), 'test_predictions_generated': 0,
        'scope': 'Source-only saved-state/scaler/probability/metric replay and independent classical refits; no convergence, novel method, fresh test cohort or first-party signal authentication established.'})
    progress.update(state='complete', updated_utc=stamp())
    atomic(output/'progress.json', progress)
    findings(output, result)
    publish(output, 'Complete verified source-only preprocessing diagnostic', push)


def findings(output, result):
    lines = ['# Source-only preprocessing diagnostic', '',
             'All declared fits and source-only candidate replays are complete. These small reused validation panels are development evidence; no test score or paper pivot is claimed.', '',
             '| Dataset | Group | EEGNet recipe | Selected LR/step/BN | Training BA | Unseen-validation BA/loss | Familiar-validation BA/loss |',
             '|---|---:|---|---|---:|---|---|']
    for row in result['neural']:
        c = row['selected']; m = c['metrics']; r = row['recipe']
        label = f'{r["montage"]}, {r["normalization"]}, {r["seconds"]}s, baseline={r["baseline"]}'
        lines.append(f'| {row["dataset"]} | {row["group"]} | {label} | {c["lr"]}/{c["step"]}/{c["normalization"]} | {m["train"]["balanced_accuracy"]:.2%} | {m["validation_unseen"]["balanced_accuracy"]:.2%}/{m["validation_unseen"]["balanced_log_loss"]:.4f} | {m["validation_familiar"]["balanced_accuracy"]:.2%}/{m["validation_familiar"]["balanced_log_loss"]:.4f} |')
    lines += ['', 'Complete classical comparisons and every candidate/curve are retained in [summary.json](summary.json) and the per-fit records.', '',
              'One initialization, fixed first folds/session/rotation, small reused validation panels and offline full-trial context limit interpretation. Montage changes parameter count; short-input models change head size and consumed training samples. Baseline controls are explicitly local adaptations. Fresh-model replay does not prove optimization convergence or authenticate first-party DEAP signals. A main-question change still requires a concrete proposal, author approval and a fresh manuscript archive.']
    (output/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def pilot(root):
    seed_everything(42)
    results = []
    for channels in (14, 32, 62):
        for seconds in (4, 40):
            model = EEGNetControl(3, channels=channels, samples=128*seconds).cuda()
            x = torch.randn(12, channels, seconds*128, device='cuda')
            y = torch.arange(12, device='cuda') % 3
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); start = time.perf_counter()
            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(x), y).backward()
                optimizer.step(); model.constrain()
            torch.cuda.synchronize()
            results.append({'channels': channels, 'seconds': seconds,
                'update_seconds': (time.perf_counter()-start)/10,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                'parameters': sum(p.numel() for p in model.parameters())})
            del model, x, y, optimizer
            torch.cuda.empty_cache()
    atomic(root/'preprocessing_resource_pilot_2026-10-09.json', {'synthetic_only': True, 'results': results})
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pilot', 'plan', 'prepare', 'run', 'export'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    parser.add_argument('--data-root', type=Path, default=REPO.parent/'emotion-recognition-eeg-datasets')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(); output = root/STUDY
    try:
        if args.command == 'pilot': pilot(root)
        elif args.command == 'plan':
            output = declare(root); export(output)
            print(json.dumps({'declared': 72, 'plan_sha256': sha(output/'plan.json')}), flush=True)
        elif args.command == 'prepare':
            prepare(output, root, args.data_root.resolve()); config(output); export(output)
        elif args.command == 'export': export(output)
        else: run(root, args.limit, args.push)
    except Exception as error:
        if args.command == 'run' and output.exists():
            atomic(output/f'FAILURE_{time.time_ns()}.json', {'failed_utc': stamp(),
                'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
            atomic(output/'progress.json', {'state': 'failed', 'updated_utc': stamp(), 'error': str(error)})
        raise
