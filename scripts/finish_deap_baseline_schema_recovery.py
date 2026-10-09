"""Declare and finish the same fixed analysis with an explicit legacy-column adapter."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.deap_baseline_folds_v2 import STUDY
from gruxnet.preprocessing_diagnostic_v2 import atomic, sha, stamp
from scripts.resume_deap_baseline_schema_recovery import guard
from scripts.audit_deap_baseline_schema_recovery import audit_complete
from scripts.analyze_deap_baseline_schema_recovery import analyze
from scripts.deap_baseline_folds_v2 import publish
from scripts.export_deap_baseline_folds_v2 import export

DECLARATION = 'analysis_schema_recovery_declaration.json'
SOURCES = ('scripts/analyze_deap_baseline_schema_recovery.py', 'scripts/finish_deap_baseline_schema_recovery.py')


def declare(root):
    output = root/STUDY; guard(root, output)
    if (output/DECLARATION).exists(): raise FileExistsError('Preserve supplementary analysis declaration')
    note = {'created_utc': stamp(), 'plan_sha256': sha(output/'plan.json'),
        'verification_recovery_sha256': sha(output/'schema_recovery_declaration.json'),
        'source_sha256': {name: sha(REPO/name) for name in SOURCES},
        'reason': 'The previously checked shared bootstrap also expects p_0/p_1. Add explicit column renaming before its call; original fixed analysis source and all fitting/selection bytes stay unchanged. No aggregate analysis was run before this correction. Group/model/contrast/metric/interval definitions are exactly those originally declared.',
        'execution': 'Resume original fixed fitting grid with declared schema-recovery runner and an explicit cell limit that returns before original analysis. After all160cells verify, finish with this declared schema-adapted analysis. No extra fitting, outcome-based choice, dropped comparison or altered cohort.',
        'development_only': True, 'research_question_change_approved': False}
    atomic(output/DECLARATION, note)
    public = REPO/'results/development'/STUDY
    (public/DECLARATION).write_bytes((output/DECLARATION).read_bytes())
    atomic(public/'analysis_schema_recovery_manifest.json', {'files': [{'file': DECLARATION, 'sha256': sha(public/DECLARATION)}],
        'scope': 'Supplementary analysis schema adapter; separate from original frozen exporter.'})
    print(json.dumps({'declared_analysis_schema_adapter': True}), flush=True)


def finish(root, push=False):
    output = root/STUDY; guard(root, output)
    declaration = json.loads((output/DECLARATION).read_text())
    if declaration['plan_sha256'] != sha(output/'plan.json'): raise ValueError('Changed analysis declaration')
    if declaration['verification_recovery_sha256'] != sha(output/'schema_recovery_declaration.json'):
        raise ValueError('Changed verifier recovery')
    for name, checksum in declaration['source_sha256'].items():
        if sha(REPO/name) != checksum: raise ValueError('Changed declared analysis adapter')
    audit_complete(output); analyze(output); guard(root, output)
    progress = json.loads((output/'progress.json').read_text())
    progress.update(state='complete', updated_utc=stamp(), cells_completed=160,
        analysis_schema_recovery_sha256=sha(output/DECLARATION))
    atomic(output/'progress.json', progress); export(output)
    if push: publish(output, 'Complete and independently verify matched full-fold DEAP baseline/context findings')
    print(json.dumps({'complete': True, 'cells': 160, 'candidate_fits': 5760}), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__); parser.add_argument('command', choices=('plan', 'finish'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs'); parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan': declare(args.root.resolve())
    else: finish(args.root.resolve(), args.push)
