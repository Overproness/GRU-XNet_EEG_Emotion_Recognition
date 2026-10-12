"""Freeze, execute and verify the separately declared development diagnosis."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet import faced_development_diagnosis as diagnosis

PUBLIC = REPO / 'results/development/faced_target_context_pairing_2026-10-12'
PRIVATE = REPO.parent / 'publication_runs/faced_target_context_pairing_2026-10-12'
PROTOCOL = 'docs/publication/FACED_Target_Context_Pairing_Protocol_2026-10-12.md'

def load(path):
    return json.loads(Path(path).read_bytes())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def save_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')

def paper_bindings():
    result = {}
    for relative in ('paper/current_progress_2026-10-12/build_receipt.json',
                     'paper/submission_note_2026-10-12/build_receipt.json'):
        receipt = load(REPO / relative)
        result[relative] = digest(REPO / relative)
        for path, expected in receipt['bindings'].items():
            if digest(REPO / path) != expected:
                raise ValueError('Existing paper binding changed: ' + path)
            result[path] = expected
    return result

def verify_bindings(activation):
    for relative, expected in activation['bindings'].items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or digest(path) != expected:
            raise ValueError('Frozen input changed: ' + relative)
    if digest(REPO.parent / 'report.tex') != activation['original_report_sha256']:
        raise ValueError('Original report changed')
    if digest(PUBLIC / 'checkpoint_replay_plan.json') != activation['checkpoint_replay_plan_sha256']:
        raise ValueError('Replay plan changed')

def declare():
    if (PUBLIC / 'activation.json').exists():
        raise PermissionError('Preserve existing activation; no overwrite')
    plan = load(PUBLIC / 'checkpoint_replay_plan.json')
    if plan['confirmation_access'] is not False or plan['new_fits'] != 0:
        raise PermissionError('Changed checkpoint replay scope')
    programme, records, metadata, decisions = diagnosis.metadata_bundle()
    folds = {}
    for rotation in programme['rotation_manifests']:
        pairs = [(r['subject'], int(r['vid'])) for r in rotation['arms']['source_fit']]
        folds[str(rotation['rotation'])] = diagnosis.crossed_source_folds(
            pairs, metadata, programme['source_participant_oof_folds'])
    bindings = dict(plan['bindings'])
    for path in list((REPO / 'gruxnet').glob('faced*.py')) + list((REPO / 'gruxnet').glob('eegnet*.py')):
        bindings[path.relative_to(REPO).as_posix()] = digest(path)
    names = [PROTOCOL, 'gruxnet/faced_development_diagnosis.py',
             'scripts/run_faced_development_diagnosis_2026_10_12.py',
             'scripts/replay_faced_diagnostic_checkpoints_2026_10_12.py',
             'tests/test_faced_development_diagnosis.py',
             'tests/test_faced_checkpoint_replay_2026_10_12.py',
             'results/development/faced_inference_quality_2026-10-11/predictions.csv',
             'results/development/faced_inference_quality_2026-10-11/comparison.json',
             'results/development/faced_eegnet_diagnostic_v2_2026-10-11/ensemble_predictions.csv',
             'results/development/faced_eegnet_diagnostic_v2_2026-10-11/checkpoint_state_audit.json']
    for relative in names:
        bindings[relative] = digest(REPO / relative)
    for relative in diagnosis.required_binding_paths():
        bindings[relative] = digest(REPO / relative)
    bindings.update(paper_bindings())
    activation = {
        'status': 'active_development_diagnosis',
        'declared_utc': datetime.now(timezone.utc).isoformat(),
        'prior_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'scope': 'Previously permitted 70 source/20 validation people and 14 development title families only',
        'bindings': bindings,
        'checkpoint_replay_plan_sha256': digest(PUBLIC / 'checkpoint_replay_plan.json'),
        'original_report_sha256': digest(REPO.parent / 'report.tex'),
        'source_crossed_folds': folds,
        'material_salt': diagnosis.MATERIAL_SALT,
        'material_hash_format': '{salt}/{vid}',
        'swap_seeds': list(diagnosis.SWAP_SEEDS),
        'alpha_selection': 'Smallest source-crossed MAE, then smaller alpha',
        'new_fit_cap': 612, 'new_neural_fit_cap': 0,
        'source_prior_constructions_cap': 63,
        'time_cap_seconds': 1800,
        'source_replay_preparation_cap_seconds': 7200,
        'source_replay_forward_cap_seconds': 1800,
        'swapped_prediction_row_cap': 107520,
        'new_original_rating_decoding': 0,
        'original_data_access': False,
        'original_data_access_note': 'This main runner has no raw access. The separately bound replay stage reads only its declared source EEG intervals.',
        'validation_forward_replay_performed': False,
        'neural_scope': 'Independent 24 source-fixture forward comparisons plus authenticated cached validation correction swaps; no whole-validation forward or whole-recording coverage claim',
        'confirmation_access': False, 'research_question_changed': False,
        'manuscript_changed': False, 'independent_confirmation': False,
        'no_automatic_retry_or_expansion': True,
    }
    verify_bindings(activation)
    save_new(PUBLIC / 'activation.json', activation)
    print(json.dumps({'declared': True, 'new_numerical_analysis': 0, 'new_fits': 0,
                      'protected_access': False, 'bindings': len(bindings)}), flush=True)

def run():
    activation_path = PUBLIC / 'activation.json'
    activation = load(activation_path)
    verify_bindings(activation)
    gate = load(PUBLIC / 'checkpoint_replay.json')
    plan = load(PUBLIC / 'checkpoint_replay_plan.json')
    if gate.get('activation_sha256') != digest(activation_path) or gate.get('plan_sha256') != digest(PUBLIC / 'checkpoint_replay_plan.json'):
        raise PermissionError('Checkpoint gate does not bind this activation/plan')
    if gate.get('selected_states') != plan['selected_states']:
        raise PermissionError('Checkpoint gate selected states changed')
    if gate.get('fixture_ids') != plan['fixture_ids'] or gate.get('context_state_bindings') != plan['context_states']:
        raise PermissionError('Checkpoint gate fixture/context state bindings changed')
    if (gate.get('passed') is not True or gate.get('source_fixture_replay_passed') is not True
        or gate.get('selected_checkpoints_forward_replayed') != 6
        or gate.get('source_fixture_checks') != 48
        or gate.get('prediction_tolerance') != plan['prediction_tolerance']
        or not 0 <= gate.get('maximum_absolute_prediction_difference', float('inf')) <= plan['prediction_tolerance']
        or gate.get('confirmation_access') is not False
        or gate.get('research_question_changed') is not False
        or gate.get('validation_forward_replay_performed') is not False):
        raise PermissionError('Incomplete or out-of-scope source checkpoint replay gate')
    expected_norm = {k: v for k, v in plan['bindings'].items() if k.endswith('/source_normalizers.json')}
    if gate.get('normalization_bindings') != expected_norm:
        raise PermissionError('Checkpoint gate normalization bindings changed')
    diagnosis.run(PUBLIC, PRIVATE, activation, neural_replay_gate=gate)
    verify_bindings(activation)

def verify():
    activation = load(PUBLIC / 'activation.json')
    verify_bindings(activation)
    execution = load(PUBLIC / 'execution.json')
    gate = load(PUBLIC / 'checkpoint_replay.json')
    if execution['status'] != 'completed_development_diagnosis' or not gate['passed']:
        raise ValueError('Incomplete diagnostic or failed replay')
    if (execution['new_ridge_fits'] != 612 or execution['new_neural_fits'] != 0
        or execution['swapped_prediction_rows'] != 107520 or execution['donor_rows'] != 53760
        or execution['cv_prediction_rows'] != 21840 or execution['prediction_rows'] != 5880):
        raise ValueError('Declared execution counts differ')
    if execution['confirmation_access'] is not False or execution['research_question_changed'] is not False:
        raise PermissionError('Changed research scope')
    outputs = {p.relative_to(REPO).as_posix(): digest(p) for p in PUBLIC.iterdir()
               if p.is_file() and p.name != 'verification.json'}
    save_new(PUBLIC / 'verification.json', {
        'passed': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_input_bindings_checked': len(activation['bindings']),
        'source_fixture_replay_passed': True,
        'validation_forward_replay_performed': False,
        'manuscript_bindings_preserved': True, 'original_report_unchanged': True,
        'confirmation_access': False, 'research_question_changed': False,
        'new_ridge_fits': 612, 'new_neural_fits': 0,
        'bindings': outputs,
        'limits': 'Integrity receipt; separate public numerical audit follows. Reused development outcomes are not independent confirmation.'})
    print(json.dumps({'verification_passed': True, 'bound_outputs': len(outputs)}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--declare', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    (declare if args.declare else run if args.run else verify)()
