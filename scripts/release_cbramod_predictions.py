"""Release only the explicitly authorized predictions and anonymous ID metadata."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_probe import STUDY, validate, sha, atomic, stamp, ROLES


def main():
    root = REPO.parent/'publication_runs'; output = root/STUDY
    plan = validate(root, output)
    if not json.loads((output/'verification.json').read_text())['complete']:
        raise ValueError('Completed verified study required')
    public = REPO/'results/development'/STUDY
    approval = {'recorded_utc': stamp(), 'user_authorization':
        'User answered "yea sure thing" to "May I also publish the trial-level predictions and anonymous participant IDs?"',
        'scope': 'Verified selected-head probability tables and their anonymous participant/trial/video references, labels, fit/verification records. No raw EEG, embeddings, fitted coefficients, or per-trial amplitude features.',
        'destination': 'https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition',
        'research_question_change_approved': False}
    approval_path = public/'prediction_release_authorization.json'
    if approval_path.exists(): approval = json.loads(approval_path.read_text())
    else: atomic(approval_path, approval)
    records = []; maximum = 0.; sets = 0
    for job in plan['jobs']:
        name = f'{job["dataset"].lower()}_g{job["group"]}_{job["model"]}'
        source = output/'fits'/name; target = public/'fits'/name
        target.mkdir(parents=True, exist_ok=True)
        record = json.loads((source/'record.json').read_text())
        proof = json.loads((source/'verification.json').read_text())
        if not proof['complete'] or proof['record_sha256'] != sha(source/'record.json'):
            raise ValueError('Stale fit proof')
        for file in ('record.json', 'verification.json', *[f'{role}.csv' for role in ROLES]):
            shutil.copyfile(source/file, target/file)
            if sha(source/file) != sha(target/file): raise ValueError('Release bytes changed')
            records.append({'file': (target/file).relative_to(public).as_posix(), 'sha256': sha(target/file)})
        for role in ROLES:
            rows = pd.read_csv(target/f'{role}.csv')
            allowed = {'trial_id', 'subject_id', 'material_key', 'original_label', 'label', 'role', 'p0', 'p1', 'p2'}
            if set(rows.columns)-allowed: raise ValueError('Unapproved export field')
            y = rows.label.to_numpy(dtype=int); p = rows[[f'p{c}' for c in range(y.max()+1)]].to_numpy()
            if not np.allclose(p.sum(1), 1., atol=1e-12): raise ValueError('Invalid released probabilities')
            values = {'n': len(y), 'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(1))),
                'balanced_log_loss': float(np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
                                                      weights=compute_sample_weight('balanced', y)))}
            for key, value in values.items():
                error = abs(value-record['metrics'][role][key]); maximum = max(maximum, error)
                if error > 2e-11: raise ValueError('Released metric mismatch')
            sets += 1
    for file in ('config.json', 'summary.json', 'verification.json', 'progress.json'):
        shutil.copyfile(output/file, public/file)
        records.append({'file': file, 'sha256': sha(public/file)})
    result = {'complete': True, 'selected_heads': 24, 'released_prediction_tables': 72,
        'independent_public_metric_sets': sets, 'max_metric_abs': maximum,
        'authorization_sha256': sha(approval_path), 'source_plan_sha256': sha(output/'plan.json'),
        'release_source_sha256': sha(Path(__file__)), 'files': records,
        'excluded': 'Raw EEG, embeddings, coefficients, per-trial amplitude statistics and input-source tables.'}
    atomic(public/'prediction_release_verification.json', result)
    # Preserve the original declaration snapshots, with their original names and
    # internal hash relationships, before replacing the current export ledger.
    initial = public/'initial_snapshot'; initial.mkdir(exist_ok=True)
    for name in ('plan.json', 'progress.json', 'export_manifest.json'):
        path = (public/name).relative_to(REPO).as_posix()
        original = subprocess.check_output(['git', 'show', f'd95f16245:{path}'], cwd=REPO)
        (initial/name).write_bytes(original)
    note = '# Authorized prediction release\n\nThe author explicitly approved publishing trial-level predictions and anonymous participant IDs on 9 October 2026. All 72 selected train/validation probability tables are now released with [independent checks](prediction_release_verification.json) and [authorization scope](prediction_release_authorization.json).\n\nThe earlier automatic-review rejection and local withholding remain recorded as historical events. Raw EEG, embeddings, coefficients and per-trial amplitude features remain local. The [aggregate bundle](../cbramod_source_probe_aggregate_2026-10-09/summary.json) is unchanged. No new experiment, participant, test inference or research-question change follows from this release.\n'
    (public/'WITHHELD_NOTE.md').write_text(note, encoding='utf-8')
    current = [{'file': p.relative_to(public).as_posix(), 'sha256': sha(p)}
               for p in public.rglob('*') if p.is_file() and p != public/'export_manifest.json']
    atomic(public/'export_manifest.json', {'files': current,
        'scope': 'User explicitly approved selected trial probabilities and anonymous IDs; current bytes checked. Original declaration snapshots retained under initial_snapshot.',
        'excluded': 'Raw EEG, embeddings, coefficients, per-trial amplitude data and full input-source tables.'})
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__': main()
