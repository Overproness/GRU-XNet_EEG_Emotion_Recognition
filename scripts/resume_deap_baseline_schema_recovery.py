"""Declared schema-only verification recovery; all fitting source bytes stay frozen."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time
import traceback
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.deap_baseline_folds_v2 import STUDY, MODELS, cell_id, load_inputs, fit_model, fit_prior
from gruxnet.preprocessing_diagnostic_v2 import atomic, sha, stamp
from gruxnet.train import seed_everything
from scripts.deap_baseline_folds_v2 import validate, environment, publish
from scripts.audit_deap_baseline_schema_recovery import audit_model, audit_complete, certificate_valid
from scripts.export_deap_baseline_folds_v2 import export

RECOVERY = 'schema_recovery_declaration.json'
RECOVERY_SOURCES = ('scripts/audit_deap_baseline_schema_recovery.py', 'scripts/resume_deap_baseline_schema_recovery.py')


def declare(root):
    output = root/STUDY; plan = validate(root, output)
    if (output/RECOVERY).exists(): raise FileExistsError('Preserve recovery declaration')
    records = list((output/'cells').glob('*/*/record.json'))
    selections = list((output/'cells').glob('*/*/selection.json'))
    failures = list(output.glob('FAILURE_*.json'))
    if len(records) != 9 or len(selections) != 9 or len(failures) != 1:
        raise ValueError('Unexpected recovery starting state')
    note = {'created_utc': stamp(), 'plan_sha256': sha(output/'plan.json'),
        'reason': 'Verifier expected p0/p1 in older context probabilities; actual archived schema is p_0/p_1. Nine first-cell logistic models already sealed, eight verified. Direct corrected-schema comparison confirms old/new context probabilities differ by0. Frozen fitting/selection/data/analysis sources are unchanged; only the supplementary verifier reads the correct older column names. No scientific choice or experimental grid changes.',
        'source_sha256': {name: sha(REPO/name) for name in RECOVERY_SOURCES},
        'preserved_artifacts': {path.relative_to(output).as_posix(): sha(path) for path in (*records, *selections, *failures)},
        'resume': 'Existing sealed record files are audited without refitting or rewriting predictions. New cases use the original frozen v2 fit functions and exact grid/configuration. All raw/calibrated contextual matches remain required with original tolerance1e-12.',
        'development_only': True, 'research_question_change_approved': False}
    atomic(output/RECOVERY, note)
    public = REPO/'results/development'/STUDY
    public.mkdir(parents=True, exist_ok=True)
    (public/RECOVERY).write_bytes((output/RECOVERY).read_bytes())
    for failure in failures: (public/failure.name).write_bytes(failure.read_bytes())
    atomic(public/'schema_recovery_manifest.json', {'files': [
        {'file': name, 'sha256': sha(public/name)} for name in (RECOVERY, *(p.name for p in failures))],
        'scope': 'Supplementary schema-only recovery declaration and preserved original verifier failure; separate from the original frozen study exporter.'})
    print(json.dumps({'declared_schema_recovery': True, 'source_files': len(RECOVERY_SOURCES),
        'preserved_records_and_selections': len(records)+len(selections)}), flush=True)


def guard(root, output):
    plan = validate(root, output); recovery = json.loads((output/RECOVERY).read_text())
    if recovery['plan_sha256'] != sha(output/'plan.json'): raise ValueError('Changed parent declaration')
    for name, checksum in recovery['source_sha256'].items():
        if sha(REPO/name) != checksum: raise ValueError('Changed declared recovery source')
    for name, checksum in recovery['preserved_artifacts'].items():
        if sha(output/name) != checksum: raise ValueError('Changed pre-recovery fitted/selected artifacts')
    return plan


def run(root, limit=None, push=False):
    output = root/STUDY; plan = guard(root, output)
    config = json.loads((output/'config.json').read_text())
    if config['plan_sha256'] != sha(output/'plan.json') or config['environment'] != environment():
        raise ValueError('Changed frozen fitting environment')
    if config['prepared_sha256'] != sha(output/'inputs/prepared.json'): raise ValueError('Changed input binding')
    base, features = load_inputs(output); seed_everything(42)
    started = time.perf_counter(); newly = 0; completed = 0
    for job in plan['jobs']:
        guard(root, output); fresh = False
        for model in MODELS:
            folder = output/'cells'/cell_id(job)/model
            if (folder/'verification.json').exists(): certificate_valid(folder, output)
            else:
                if not (folder/'record.json').exists():
                    if model == 'prior': fit_prior(output, base, job)
                    else: fit_model(output, base, features, job, model)
                audit_model(output, root, base, features, job, model); fresh = True
        completed += 1; newly += int(fresh)
        progress = {'state': 'running', 'updated_utc': stamp(), 'worker_pid': os.getpid(),
            'cells_completed': completed, 'cells_total': 160, 'candidates_verified': completed*36,
            'last_cell': cell_id(job), 'new_cells_this_process': newly,
            'elapsed_seconds_this_process': time.perf_counter()-started,
            'schema_recovery_sha256': sha(output/RECOVERY), 'research_question_change_approved': False}
        atomic(output/'progress.json', progress); print(json.dumps(progress), flush=True)
        if push and fresh and newly % 10 == 0: publish(output, f'Checkpoint DEAP baseline controls: {completed}/160 cells verified')
        if limit is not None and newly >= limit:
            progress['state'] = 'bounded_run_finished'; atomic(output/'progress.json', progress)
            export(output)
            if push: publish(output, f'Verify DEAP baseline controls after schema recovery: {completed}/160')
            return
    verification = audit_complete(output)
    from scripts.analyze_deap_baseline_folds_v2 import analyze
    analyze(output); guard(root, output)
    progress['state'] = 'complete'; progress['updated_utc'] = stamp()
    progress['elapsed_seconds_this_process'] = time.perf_counter()-started
    atomic(output/'progress.json', progress); export(output)
    if push: publish(output, 'Complete and verify matched full-fold DEAP baseline and context controls')
    print(json.dumps(verification), flush=True)


def exclusive(root, limit=None, push=False):
    with (root/STUDY/'worker.lock').open('a+b') as lock:
        if lock.tell() == 0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, limit, push)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__); parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    parser.add_argument('--limit', type=int); parser.add_argument('--push', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    try:
        if args.command == 'plan': declare(root)
        else: exclusive(root, args.limit, args.push)
    except Exception as error:
        atomic(root/STUDY/f'RECOVERY_FAILURE_{time.time_ns()}.json', {'created_utc': stamp(),
            'error': str(error), 'traceback': traceback.format_exc()})
        raise
