"""Prepare metadata-only inactive confirmation and authorize synthetic precision."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_confirmation_contract import metadata_builder, canonical_sha256, validate_contract
from scripts.run_faced_development_diagnosis_2026_10_12 import paper_bindings
from scripts.faced_confirmation_precision_2026_10_12 import scientific_conditions, metadata_projection

OUT = REPO / 'results/development/faced_confirmation_freeze_2026-10-12'
PROTOCOL = 'docs/publication/FACED_Independent_Confirmation_Protocol_2026-10-12.md'
QUESTION = ('On the reserved FACED catalogue roster, does the specified spectral-plus-context Ridge recipe '
            'improve retrospective valence prediction over source category medians, and does a matched '
            'context-only control support attributing a difference to adding the spectral representation?')
CLAIM = ('In reused FACED development panels, source-selected category medians are competitive with '
         'the two tested late-interval EEG predictors; their gains and person/film pairing effects are '
         'inconsistent or uncertain. Category support changes contextual comparator qualification.')

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

def verify_map(bindings):
    for relative, expected in bindings.items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or digest(path) != expected:
            raise PermissionError('Prior/frozen evidence changed: ' + relative)

def prepare():
    if any((OUT / n).exists() for n in ('activation.json', 'contract.json', 'selected_claim.json')):
        raise PermissionError('Preserve an existing preparation; no overwrite')
    prior_path = REPO / 'results/development/faced_category_supported_context_2026-10-12/findings_completion.json'
    prior = load(prior_path)
    if prior.get('passed') is not True or prior.get('confirmation_access') is not False or prior.get('research_question_changed') is not False:
        raise PermissionError('Passing, scope-preserving development completion required')
    verify_map(prior['bindings'])
    precision_plan = load(OUT / 'precision_plan.json')
    verify_map(precision_plan['metadata_bindings'])
    verify_map(precision_plan['code_bindings'])
    if precision_plan['conditions'] != scientific_conditions() or precision_plan['metadata_projection'] != metadata_projection():
        raise PermissionError('Synthetic precision assumptions or metadata changed')
    contract = metadata_builder(REPO)
    contract['candidate_question'] = QUESTION
    contract['resource_bounds'] = {
        'new_source_ridge_fits_maximum': 2, 'source_median_constructions': 1,
        'source_cache_qualification_seconds': 3600, 'source_fit_and_replay_seconds': 900,
        'protected_signal_preparation_seconds': 14400, 'protected_inference_seconds': 900,
        'protected_scoring_seconds': 900, 'automatic_retry': False,
        'source_or_protected_execution_authorized_now': False,
    }
    contract['backend_status'] = ('Guarded pure analysis/information helpers only; actual extraction, final '
                                  'source states and export-byte qualification are not implemented or launched')
    validate_contract(contract)
    claim = {'status': 'selected_for_author_review_not_adopted', 'supported_development_claim': CLAIM,
             'candidate_confirmation_question': QUESTION, 'manuscript_research_question_changed': False,
             'independent_confirmation': False, 'general_eeg_information_null_claimed': False,
             'equivalence_claimed': False, 'causal_emotion_claimed': False,
             'primary_contrast_scope': 'Whole-recipe utility', 'mandatory_secondary': 'Matched-context Ridge100',
             'original_empirical_continuation_gate_passed': False,
             'reason_for_hold': 'No qualifying reproducible incremental-utility lead; uncertain development and small catalogue roster',
             'author_question_approval': False, 'confirmation_access_approval': False}
    bindings = dict(contract['bindings']) | dict(contract['policy_code_bindings'])
    paths = [PROTOCOL, 'gruxnet/faced_confirmation_contract.py',
             'tests/test_faced_confirmation_contract.py', 'scripts/prepare_faced_confirmation_2026_10_12.py',
             'scripts/run_faced_development_diagnosis_2026_10_12.py',
             'gruxnet/faced_source_quality.py', 'gruxnet/faced_proposal_preprocessing.py',
             'gruxnet/faced_source_notch.py', 'gruxnet/faced_classical_contract.py',
             'gruxnet/faced_artifact_control.py', 'gruxnet/faced_proposal_preprocessing_v2.py',
             prior_path.relative_to(REPO).as_posix()]
    for relative in paths:
        bindings[relative] = digest(REPO / relative)
    bindings.update(precision_plan['code_bindings'])
    bindings.update(paper_bindings())
    save_new(OUT / 'contract.json', contract)
    save_new(OUT / 'selected_claim.json', claim)
    bindings[(OUT / 'contract.json').relative_to(REPO).as_posix()] = digest(OUT / 'contract.json')
    bindings[(OUT / 'selected_claim.json').relative_to(REPO).as_posix()] = digest(OUT / 'selected_claim.json')
    activation = {
        'status': 'prepared_development_only_confirmation_protocol', 'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'prior_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'confirmation_protocol_status': 'frozen_not_activated',
        'contract_semantic_sha256': canonical_sha256(contract), 'bindings': bindings,
        'prior_completed_evidence_bindings': prior['bindings'],
        'precision_plan_sha256': digest(OUT / 'precision_plan.json'),
        'precision_simulation_authorized': True, 'synthetic_matrix_cap': 4096,
        'synthetic_bootstrap_mean_cap': 2097152, 'synthetic_time_cap_seconds': 300,
        'empirical_precision_gate_passed': False, 'original_empirical_continuation_gate_passed': False,
        'author_question_approval': False, 'confirmation_access_approval': False,
        'confirmation_access': False, 'research_question_changed': False, 'manuscripts_changed': False,
        'original_report_sha256': digest(REPO.parent / 'report.tex'),
        'new_empirical_fits_authorized': 0, 'original_data_access_authorized': False,
        'limitations': 'Frozen inactive protocol and synthetic planning sensitivity; no end-to-end final backend or confirmed scientific claim',
    }
    verify_map(bindings)
    save_new(OUT / 'activation.json', activation)
    print(json.dumps({'prepared': True, 'confirmation_status': 'frozen_not_activated',
                      'synthetic_simulation_authorized': True, 'real_value_access': 0,
                      'empirical_fits': 0, 'bindings': len(bindings)}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.parse_args()
    prepare()
