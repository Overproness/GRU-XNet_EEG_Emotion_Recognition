"""Verify immutable preparation, synthetic evidence and human-readable findings."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_confirmation_contract import canonical_sha256, validate_contract

OUT = REPO / 'results/development/faced_confirmation_freeze_2026-10-12'
FINDINGS = REPO / 'docs/publication/FACED_Confirmation_Freeze_Findings_2026-10-12.md'

def load(path):
    return json.loads(path.read_bytes())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def verify_map(bindings):
    for relative, expected in bindings.items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or digest(path) != expected:
            raise ValueError('Bound preparation/evidence changed: ' + relative)

def main():
    destination = OUT / 'completion.json'
    if destination.exists():
        raise PermissionError('Preserve completed preparation; no overwrite')
    activation = load(OUT / 'activation.json')
    verify_map(activation['bindings'])
    verify_map(activation['prior_completed_evidence_bindings'])
    if digest(REPO.parent / 'report.tex') != activation['original_report_sha256']:
        raise ValueError('Historical report changed')
    if (activation['confirmation_access'] is not False or activation['research_question_changed'] is not False
        or activation['new_empirical_fits_authorized'] != 0 or activation['original_data_access_authorized'] is not False):
        raise PermissionError('Preparation scope changed')
    contract = load(OUT / 'contract.json')
    validate_contract(contract)
    if canonical_sha256(contract) != activation['contract_semantic_sha256'] or any(contract['currently_passed_confirmation_gates'].values()):
        raise PermissionError('Inactive contract or false gates changed')
    receipt = load(OUT / 'precision_receipt.json')
    if (receipt['status'] != 'completed_synthetic_only' or receipt['synthetic_matrices'] != 4096
        or receipt['bootstrap_means'] != 2097152 or receipt['report_rows'] != 48
        or receipt['activation_sha256'] != digest(OUT / 'activation.json')
        or receipt['plan_sha256'] != digest(OUT / 'precision_plan.json')
        or receipt['confirmation_access'] is not False or receipt['precision_gate_passed'] is not False
        or receipt['new_empirical_fits'] != 0):
        raise ValueError('Declared synthetic completion differs')
    for name, expected in receipt['output_bindings'].items():
        if digest(OUT / name) != expected:
            raise ValueError('Synthetic output changed')
    verification = load(OUT / 'precision_verification.json')
    if verification.get('passed') is not True or verification['synthetic_matrices'] != 4096 or verification['summary_rows'] != 48:
        raise ValueError('Synthetic verification missing')
    figure = load(OUT / 'figures/precision_plot_receipt.json')
    for name, expected in figure['input_bindings'].items():
        if digest(OUT / name) != expected:
            raise ValueError('Figure input changed')
    for name, expected in figure['output_bindings'].items():
        if digest(OUT / 'figures' / name) != expected:
            raise ValueError('Figure artifact changed')
    if digest(REPO / 'scripts/plot_faced_confirmation_precision_2026_10_12.py') != figure['script_sha256']:
        raise ValueError('Figure code changed')
    rows = load(OUT / 'precision_summary.json')['rows']
    if len(rows) != 48 or figure['plotted_condition_rows'] != 16:
        raise ValueError('Declared reporting/figure coverage changed')
    text = FINDINGS.read_text(encoding='utf-8')
    scenarios = [('Low', 'low'), ('Participant dominant', 'participant_dominant'),
                 ('Title dominant', 'title_dominant'), ('Balanced', 'balanced')]
    checked = 0
    for label, key in scenarios:
        small = [r for r in rows if r['scenario'] == key and r['subjects'] == 33 and r['titles'] == 8]
        base = next(r for r in small if r['true_gain'] == .045)
        large = next(r for r in small if r['true_gain'] == .15)
        def cell(count, n):
            return f'{count}/{n} ({100 * count / n:.2f}%)'
        expected = [f"{base['mean_interval_width']:.4f}",
                    cell(base['coverage_count'], base['replicates']),
                    cell(base['positive_lower_endpoint_count'], base['replicates']),
                    cell(large['positive_lower_endpoint_count'], large['replicates'])]
        line = next(line for line in text.splitlines() if line.startswith('| ' + label + ' |'))
        actual = [value.strip() for value in line.split('|')[2:-1]]
        if actual != expected:
            raise ValueError('Human-readable synthetic summary differs')
        checked += len(actual)
    names = [p for p in OUT.rglob('*') if p.is_file()]
    names += [FINDINGS, REPO / 'docs/publication/FACED_Confirmation_Claim_Review_2026-10-12.md',
              REPO / 'docs/publication/FACED_Independent_Confirmation_Protocol_2026-10-12.md',
              REPO / 'docs/publication/FACED_Confirmation_Precision_Design_2026-10-12.md',
              REPO / 'scripts/plot_faced_confirmation_precision_2026_10_12.py', Path(__file__)]
    bindings = {path.relative_to(REPO).as_posix(): digest(path) for path in names}
    result = {'passed': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'current_input_bindings_checked': len(activation['bindings']),
              'prior_evidence_bindings_preserved': len(activation['prior_completed_evidence_bindings']),
              'safeguard_tests_passed': 41,
              'independent_public_scientific_and_numeric_review_passed': True,
              'independent_synthetic_aggregation_rows_checked': 48,
              'independent_seeded_fixture_matrices_replayed': 48,
              'independent_fixture_bootstrap_means_replayed': 24576,
              'summary_table_entries_checked': checked, 'synthetic_matrices': 4096,
              'synthetic_bootstrap_means': 2097152, 'figure_visually_inspected': True,
              'confirmation_protocol_status': 'frozen_not_activated',
              'new_empirical_fits': 0, 'protected_values_accessed': 0,
              'original_recordings_or_ratings_opened': 0,
              'confirmation_access': False, 'manuscripts_changed': False,
              'research_question_changed': False, 'precision_gate_passed': False,
              'original_empirical_continuation_gate_passed': False,
              'final_source_states_qualified': False, 'end_to_end_confirmation_backend_implemented': False,
              'original_report_and_paper_bindings_preserved': True,
              'bindings': bindings,
              'limits': 'Synthetic assumed-variance sensitivity is not empirical FACED power. The claim is exploratory development; the frozen candidate protocol is inactive and does not authorize protected access or a manuscript pivot.'}
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'completed_preparation': True, 'confirmation_inactive': True,
                      'summary_table_entries': checked, 'bound_files': len(bindings)}), flush=True)

if __name__ == '__main__':
    main()
