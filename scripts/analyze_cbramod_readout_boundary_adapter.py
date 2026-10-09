"""Declared correction: validation roles can share held-out people, not trials."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_learning import sha, read, write

STUDY = 'cbramod_readout_2026-10-09'
PUBLIC = REPO/'results/development'/STUDY
ORIGINAL = REPO/'scripts/analyze_cbramod_readout.py'
DECLARATION = 'analysis_boundary_adapter_declaration.json'
BEFORE = "if set(roles[a][name]) & set(roles[b][name]):"
AFTER = "if (name == 'trial_id' or a == 'train') and set(roles[a][name]) & set(roles[b][name]):"


def effective_source():
    source = ORIGINAL.read_text(encoding='utf-8')
    if source.count(BEFORE) != 1:
        raise ValueError('Boundary correction must replace exactly one guard')
    return source.replace(BEFORE, AFTER)


def declare():
    private = REPO.parent/'publication_runs'/STUDY
    if (private/DECLARATION).exists() or (PUBLIC/DECLARATION).exists():
        raise FileExistsError('Preserve boundary correction declaration')
    progress = read(private/'progress.json')
    if progress['new_trajectories_completed'] or any(p.name != 'pooled_linear' and 'pooled_linear' not in p.name for p in (private/'runs').iterdir()):
        raise ValueError('Declare correction before any new trajectory begins')
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(Path(__file__)), 'original_analysis_sha256': sha(ORIGINAL),
        'effective_analysis_sha256': hashlib.sha256(effective_source().encode('utf-8')).hexdigest(),
        'fit_plan_sha256': sha(private/'plan.json'), 'progress_at_declaration': progress,
        'original_guard': BEFORE, 'corrected_guard': AFTER,
        'scope': 'Correct an overly strict public-analysis subject guard. Training people are disjoint from both validation roles; validation roles deliberately share held-out people with disjoint trials/material roles. Trial exclusions remain pairwise. Original sixty bound files, data, fits, metrics, selection, contrasts and tolerances unchanged. Declared before new task trajectories; no original-label or manuscript/research-question change.'}
    write(private/DECLARATION, record)
    shutil.copyfile(private/DECLARATION, PUBLIC/DECLARATION)
    manifest = read(PUBLIC/'export_manifest.json')
    manifest['files'].append({'file': DECLARATION, 'sha256': sha(PUBLIC/DECLARATION)})
    write(PUBLIC/'export_manifest.json', manifest)
    print(json.dumps({'declared': True, 'new_trajectories_completed': 0,
        'declaration_sha256': sha(PUBLIC/DECLARATION)}))


def api():
    declaration = read(PUBLIC/DECLARATION)
    source = effective_source()
    if declaration['source_sha256'] != sha(Path(__file__)) or declaration['original_analysis_sha256'] != sha(ORIGINAL):
        raise ValueError('Changed boundary adapter/source binding')
    if declaration['effective_analysis_sha256'] != hashlib.sha256(source.encode('utf-8')).hexdigest() or declaration['fit_plan_sha256'] != sha(PUBLIC/'plan.json'):
        raise ValueError('Changed effective analysis or fit declaration')
    namespace = {'__name__': 'declared_readout_boundary_analysis', '__file__': str(Path(__file__).resolve())}
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('declare', 'generate', 'verify'))
    args = parser.parse_args()
    if args.action == 'declare':
        declare()
    else:
        api()[args.action](PUBLIC)
