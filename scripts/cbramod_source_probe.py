"""Declare, prepare and complete the small authorized frozen representation study."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import numpy as np
import scipy
import sklearn
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_probe import (STUDY, pilot, declare, validate, prepare,
    extract, fit_job, export, sha, atomic, stamp)
from gruxnet.train import seed_everything
from scripts.audit_cbramod_source_probe import audit_head, finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pilot', 'declare', 'run', 'export'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    parser.add_argument('--data-root', type=Path, default=REPO.parent/'emotion-recognition-eeg-datasets')
    args = parser.parse_args()
    root = args.root.resolve(); output = root/STUDY
    seed_everything(42)
    if args.action == 'pilot':
        print(json.dumps(pilot(root))); return
    if args.action == 'declare':
        declare(root); export(root, output)
        print(json.dumps({'declared': STUDY, 'plan_sha256': sha(output/'plan.json')})); return
    if args.action == 'export':
        export(root, output); return
    plan = validate(root, output)
    lock = output/'worker.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    try:
        config = {'python': platform.python_version(), 'torch': torch.__version__,
                  'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                  'plan_sha256': sha(output/'plan.json'), 'device': torch.cuda.get_device_name(),
                  'outer_test_inferences': 0}
        if (output/'config.json').exists():
            if json.loads((output/'config.json').read_text()) != config:
                raise ValueError('Changed worker environment')
        else:
            atomic(output/'config.json', config)
        prepare(root, output, args.data_root.resolve())
        extract(root, output)
        for job in plan['jobs']:
            folder = output/'fits'/f'{job["dataset"].lower()}_g{job["group"]}_{job["model"]}'
            fit_job(root, output, job)
            if not (folder/'verification.json').exists():
                audit_head(root, output, job)
            count = len(list((output/'fits').glob('*/verification.json')))
            atomic(output/'progress.json', {'state': 'fitting', 'heads_completed': count,
                'heads_total': len(plan['jobs']), 'updated_utc': stamp(), 'outer_test_inferences': 0,
                'research_question_change_approved': False})
            print(json.dumps({'verified_heads': count, 'total': len(plan['jobs']), 'job': job}), flush=True)
            export(root, output)
        result = finish(root, output, args.data_root.resolve())
        atomic(output/'progress.json', {'state': 'complete', 'heads_completed': len(plan['jobs']),
            'heads_total': len(plan['jobs']), 'updated_utc': stamp(), 'outer_test_inferences': 0,
            'research_question_change_approved': False})
        export(root, output)
        print(json.dumps(result), flush=True)
    except Exception as error:
        atomic(output/f'FAILURE_{stamp().replace(":", "-")}.json',
               {'exception': repr(error), 'traceback': traceback.format_exc(), 'created_utc': stamp()})
        export(root, output)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
