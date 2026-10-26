"""Freeze and verify category-supported source-person context qualification.

No work occurs on import. This wrapper never opens original EEG/ratings or
changes the completed preceding diagnostic or manuscript packages.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet import faced_category_supported_context as context
from scripts.run_faced_development_diagnosis_2026_10_12 import paper_bindings

BASE = REPO / 'results/development'
PREVIOUS = BASE / 'faced_target_context_pairing_2026-10-12'
PUBLIC = BASE / 'faced_category_supported_context_2026-10-12'
PRIVATE = REPO.parent / 'publication_runs/faced_category_supported_context_2026-10-12'
PROTOCOL = 'docs/publication/FACED_Category_Supported_Context_Protocol_2026-10-12.md'
DESIGN = 'docs/publication/FACED_Category_Supported_Context_Design_2026-10-12.md'
STATUS = 'active_category_supported_context'


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_hash_bindings(bindings):
    for relative, expected in bindings.items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO.resolve()) or digest(path) != expected:
            raise ValueError('Frozen qualification input changed: ' + relative)


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'blas_threads': 2}


def previous_packet_gate():
    verification = load(PREVIOUS / 'verification.json')
    audit = load(PREVIOUS / 'independent_numerical_audit.json')
    execution = load(PREVIOUS / 'execution.json')
    activation = load(PREVIOUS / 'activation.json')
    if verification.get('passed') is not True or audit.get('passed') is not True:
        raise PermissionError('Completed preceding integrity and independent numerical audit required')
    for receipt in (verification, audit, execution, activation):
        if receipt.get('confirmation_access') is not False or receipt.get('research_question_changed') is not False:
            raise PermissionError('Preceding protected/research scope changed')
    if execution.get('status') != 'completed_development_diagnosis' or execution.get('new_ridge_fits') != 612 or execution.get('new_neural_fits') != 0:
        raise PermissionError('Complete preceding diagnostic required')
    check_hash_bindings(activation['bindings'])
    check_hash_bindings(verification['bindings'])
    check_hash_bindings(audit['input_bindings'])
    if digest(REPO.parent / 'report.tex') != activation['original_report_sha256']:
        raise ValueError('Historical report changed since preceding diagnostic')
    if audit.get('cv_prediction_rows_checked') != 21840 or audit.get('validation_prediction_rows_checked') != 5880:
        raise PermissionError('Preceding audited source/validation coverage changed')
    return {p.relative_to(REPO).as_posix(): digest(p) for p in PREVIOUS.iterdir() if p.is_file()}


def source_fold_plan():
    programme, records, metadata, decisions = context.metadata_bundle()
    folds, support = {}, {}
    for manifest in programme['rotation_manifests']:
        k = str(manifest['rotation'])
        pairs = [(r['subject'], int(r['vid'])) for r in manifest['arms']['source_fit']]
        folds[k] = context.person_folds(pairs, metadata, programme['source_participant_oof_folds'])
        categories = {}
        for category in context.CATEGORIES:
            members = [p for p in pairs if metadata[p]['assigned_category'] == category]
            titles = {metadata[p]['family'] for p in members}
            categories[category] = {'source_rows': len(members), 'source_title_families': len(titles)}
        if any(r != {'source_rows': 70, 'source_title_families': 1} for r in categories.values()):
            raise PermissionError('One-source-title/category design support changed')
        support[k] = categories
    return json.loads(json.dumps(folds)), support


def verify_activation(activation):
    if activation.get('status') != STATUS:
        raise PermissionError('Separately frozen category-supported activation required')
    exact = {'new_fit_cap': 156, 'new_neural_fit_cap': 0, 'source_prior_constructions_cap': 18,
             'time_cap_seconds': 900, 'new_original_rating_decoding': 0,
             'original_data_access': False, 'confirmation_access': False,
             'research_question_changed': False, 'manuscript_changed': False,
             'independent_confirmation': False}
    if any(activation.get(key) != value for key, value in exact.items()):
        raise PermissionError('Qualification scope, counts or caps changed')
    if activation.get('runtime') != runtime():
        raise ValueError('Frozen qualification runtime changed')
    if not set(context.required_binding_paths()) <= set(activation['bindings']):
        raise PermissionError('Missing source/validation or implementation dependency')
    check_hash_bindings(activation['bindings'])
    check_hash_bindings(activation['prior_packet_bindings'])
    if digest(REPO.parent / 'report.tex') != activation['original_report_sha256']:
        raise ValueError('Historical manuscript changed')
    folds, support = source_fold_plan()
    if folds != activation['source_person_folds'] or support != activation['source_category_support']:
        raise PermissionError('Source person/category selection identities changed before outcome conversion')


def declare():
    if (PUBLIC / 'activation.json').exists():
        raise PermissionError('Preserve existing qualification declaration; no overwrite')
    prior_bindings = previous_packet_gate()
    folds, support = source_fold_plan()
    bindings = dict(prior_bindings)
    names = context.required_binding_paths() + [PROTOCOL, DESIGN,
        'scripts/run_faced_category_supported_context_2026_10_12.py',
        'scripts/run_faced_development_diagnosis_2026_10_12.py',
        'scripts/audit_faced_development_diagnosis_2026_10_12.py',
        'tests/test_faced_category_supported_context.py']
    for relative in names:
        bindings[relative] = digest(REPO / relative)
    bindings.update(paper_bindings())
    activation = {'status': STATUS, 'date': '2026-10-12',
        'declared_utc': datetime.now(timezone.utc).isoformat(),
        'prior_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'scope': 'Previously permitted 70 source and 20 validation participants; same 14 development title families',
        'reason_for_separate_stage': 'Earlier crossed source-title folds also removed their assigned categories; this source-person qualification retains every source category and evaluates a different selection task.',
        'bindings': bindings, 'prior_packet_bindings': prior_bindings,
        'original_report_sha256': digest(REPO.parent / 'report.tex'), 'runtime': runtime(),
        'source_person_folds': folds, 'source_category_support': support,
        'context_alphas': list(context.ALPHAS),
        'source_selection': 'Five fixed source participant folds; all eight source titles/categories remain represented. No development-validation selection.',
        'source_selection_generalization': 'New source participants on familiar source titles; not unseen-title validation',
        'alpha_tie_rule': 'Participant/title-equal MAE, then smaller alpha',
        'comparator_tie_order': list(context.COMPARATOR_PRIORITY),
        'new_fit_cap': 156, 'new_neural_fit_cap': 0,
        'source_prior_constructions_cap': 18, 'time_cap_seconds': 900,
        'expected_cv_prediction_rows': 21840, 'expected_validation_prediction_rows': 5880,
        'eeg_anchors': list(context.ANCHORS), 'eeg_anchor_recipe_changed': False,
        'original_data_access': False, 'new_original_rating_decoding': 0,
        'new_eeg_feature_or_state_access': False,
        'confirmation_access': False, 'research_question_changed': False,
        'manuscript_changed': False, 'independent_confirmation': False,
        'no_automatic_retry_or_expansion': True}
    verify_activation(activation)
    save_new(PUBLIC / 'activation.json', activation)
    print(json.dumps({'declared': True, 'new_fits': 0, 'new_numerical_analysis': 0,
                      'source_person_folds': 15, 'bindings': len(bindings),
                      'confirmation_access': False}), flush=True)


def run():
    path = PUBLIC / 'activation.json'
    activation = load(path)
    verify_activation(activation)
    activation_sha = digest(path)
    try:
        context.run(PUBLIC, PRIVATE, activation)
    except BaseException as error:
        if not (PUBLIC / 'execution.json').exists():
            save_new(PUBLIC / 'execution.json', {'status': 'setup_failed',
                'error_type': type(error).__name__, 'error': str(error),
                'activation_sha256': activation_sha, 'new_ridge_fits': 0, 'new_eeg_fits': 0,
                'original_files_opened': 0, 'new_original_rating_decoding': 0,
                'confirmation_access': False, 'research_question_changed': False})
        raise
    if digest(path) != activation_sha:
        raise ValueError('Qualification activation changed during execution')
    verify_activation(activation)


def row_count(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return sum(1 for _ in csv.DictReader(stream))


def verify():
    activation = load(PUBLIC / 'activation.json')
    verify_activation(activation)
    execution = load(PUBLIC / 'execution.json')
    models = load(PUBLIC / 'model_manifest.json')
    priors = load(PUBLIC / 'prior_manifest.json')
    selection = load(PUBLIC / 'source_selection.json')
    if execution.get('status') != 'completed_category_supported_context':
        raise PermissionError('Complete qualification required')
    expected = {'new_ridge_fits': 156, 'source_prior_constructions': 18,
                'cv_prediction_rows': 21840, 'prediction_rows': 5880,
                'new_eeg_fits': 0, 'private_eeg_arrays_opened': 0, 'original_files_opened': 0,
                'new_original_rating_decoding': 0, 'confirmation_access': False,
                'research_question_changed': False}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError('Qualification completed counts/scope differ')
    if models.get('new_ridge_fits') != 156 or models.get('new_eeg_fits') != 0 or len(models['fits']) != 156:
        raise ValueError('Incomplete bounded fitted-context manifest')
    if priors.get('source_prior_constructions') != 18 or len(priors['constructions']) != 18:
        raise ValueError('Incomplete source-prior construction manifest')
    if selection.get('source_person_folds') != activation['source_person_folds'] or selection.get('validation_selection') is not False or selection.get('independent_title_validation') is not False:
        raise PermissionError('Context selection data scope changed')
    if row_count(PUBLIC / 'cv_predictions.csv') != 21840 or row_count(PUBLIC / 'predictions.csv') != 5880:
        raise ValueError('Qualification saved row counts differ')
    outputs = {p.relative_to(REPO).as_posix(): digest(p) for p in PUBLIC.iterdir()
               if p.is_file() and p.name != 'verification.json'}
    save_new(PUBLIC / 'verification.json', {'passed': True,
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'activation_sha256': digest(PUBLIC / 'activation.json'),
        'frozen_input_bindings_checked': len(activation['bindings']),
        'prior_packet_bindings_preserved': len(activation['prior_packet_bindings']),
        'manuscript_bindings_preserved': True, 'original_report_unchanged': True,
        'new_ridge_fits': 156, 'source_prior_constructions': 18,
        'cv_prediction_rows': 21840, 'validation_prediction_rows': 5880,
        'new_eeg_fits': 0, 'new_original_rating_decoding': 0,
        'original_data_access': False, 'confirmation_access': False, 'research_question_changed': False,
        'bindings': outputs,
        'limits': 'Integrity/counts receipt. Separate public numerical audit required; person-held source selection uses familiar source titles and all validation outcomes remain adaptive development.'})
    print(json.dumps({'verification_passed': True, 'bound_outputs': len(outputs),
                      'preserved_prior_files': len(activation['prior_packet_bindings'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--declare', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    (declare if args.declare else run if args.run else verify)()
