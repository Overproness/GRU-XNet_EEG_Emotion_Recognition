"""Declare and run the authorized matched embedding/dropout diagnostic."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_conditioning import (
    STUDY, declare, validate, progress, export, fit, identifier, sha, atomic, stamp,
    resource_preflight, prepare_normalizer, normalizer_id, is_anchor,
)
from scripts.audit_cbramod_conditioning import audit_normalizer, audit_case, finish, read_record
from gruxnet.train import seed_everything


def publish(output, message):
    validate(output.parent, output); export(output)
    prefix = f'results/development/{STUDY}'
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() != 'main':
        raise ValueError('Unexpected publication branch')
    if subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=REPO, text=True).strip() != 'https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition.git':
        raise ValueError('Unexpected publication destination')
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).splitlines()
    if any(not n.startswith(prefix+'/') for n in staged):
        raise ValueError('Unrelated staged publication path')
    subprocess.run(['git', 'add', '--', prefix], cwd=REPO, check=True)
    subprocess.run(['git', 'commit', '-m', message], cwd=REPO, check=True)
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pilot', 'declare', 'run', 'export'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); output = root/STUDY
    seed_everything(42)
    if args.action == 'pilot':
        print(json.dumps(resource_preflight(root))); return
    if args.action == 'declare':
        declare(root); export(output)
        print(json.dumps({'declared': STUDY, 'plan_sha256': sha(output/'plan.json')})); return
    if args.action == 'export':
        export(output); return
    plan = validate(root, output)
    if (output/'verification.json').exists():
        proof = json.loads((output/'verification.json').read_text())
        if not proof['complete'] or proof['plan_sha256'] != sha(output/'plan.json') or proof['summary_sha256'] != sha(output/'summary.json'):
            raise ValueError('Changed completed conditioning study')
        for job in plan['jobs']:
            folder, record = read_record(root, output, job)
            case = json.loads((folder/'verification.json').read_text())
            if not case['complete'] or case['record_sha256'] != sha(folder/'record.json'):
                raise ValueError('Changed completed conditioning case')
        print(json.dumps({'complete': True, 'action': 'Checked completed artifacts without overwriting records'})); return
    lock = output/'worker.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    try:
        config = {'python': platform.python_version(), 'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
            'plan_sha256': sha(output/'plan.json'), 'outer_test_inferences': 0, 'precision': 'FP32CUDAexplicitMATH', 'inference_batch': 16}
        if (output/'config.json').exists():
            if json.loads((output/'config.json').read_text()) != config:
                raise ValueError('Changed conditioning environment')
        else:
            atomic(output/'config.json', config)
        for job in plan['normalizer_jobs']:
            prepare_normalizer(root, output, job)
            folder = output/'normalizers'/normalizer_id(job)
            if not (folder/'verification.json').exists():
                audit_normalizer(root, output, job)
            progress(output, 'preparing_normalizers')
            print(json.dumps({'phase': 'normalizers', 'verified': len(list((output/'normalizers').glob('*/verification.json'))), 'total': 8, 'job': job}), flush=True)
        if args.push:
            publish(output, 'Verify eight training-only CBraMod conditioners')
        else:
            export(output)
        ordered = [j for j in plan['jobs'] if is_anchor(j)] + [j for j in plan['jobs'] if not is_anchor(j)]
        for job in ordered:
            fit(root, output, job)
            folder = output/'runs'/identifier(job)
            if not (folder/'verification.json').exists():
                audit_case(root, output, job)
            progress(output, 'running_controls')
            state = json.loads((output/'progress.json').read_text())
            print(json.dumps({'phase': 'controls', 'verified': state['conditions_completed'], 'total': 64,
                'new_verified': state['new_trajectories_completed'], 'job': job}), flush=True)
            milestone = (is_anchor(job) and state['anchors_completed'] == 16) or \
                (not is_anchor(job) and state['new_trajectories_completed'] % 4 == 0)
            if milestone:
                message = f'Verify CBraMod conditioning {state["conditions_completed"]} of 64 controls'
                if args.push:
                    publish(output, message)
                else:
                    export(output)
        result = finish(root, output); progress(output, 'complete')
        if args.push:
            publish(output, 'Complete matched CBraMod embedding and dropout controls')
        else:
            export(output)
        print(json.dumps(result), flush=True)
    except Exception as error:
        atomic(output/f'FAILURE_{stamp().replace(":", "-")}.json',
            {'exception': repr(error), 'traceback': traceback.format_exc(), 'created_utc': stamp()})
        export(output); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
