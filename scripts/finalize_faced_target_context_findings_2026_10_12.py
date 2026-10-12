"""Bind completed public diagnostic evidence and verify the findings tables."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'results/development'
FIRST = BASE / 'faced_target_context_pairing_2026-10-12'
SECOND = BASE / 'faced_category_supported_context_2026-10-12'
DOC = REPO / 'docs/publication/FACED_Target_Context_Pairing_Findings_2026-10-12.md'

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
            raise ValueError('Completed scientific binding changed: ' + relative)

def main():
    destination = SECOND / 'findings_completion.json'
    if destination.exists():
        raise PermissionError('Preserve the completed findings receipt')
    counts = []
    for packet in (FIRST, SECOND):
        activation = load(packet / 'activation.json')
        verify_map(activation['bindings'])
        if digest(REPO.parent / 'report.tex') != activation['original_report_sha256']:
            raise ValueError('Original historical report changed')
        for filename in ('verification.json', 'independent_numerical_audit.json'):
            receipt = load(packet / filename)
            if receipt.get('passed') is not True or receipt.get('confirmation_access') is not False or receipt.get('research_question_changed') is not False:
                raise PermissionError('Passing scope-preserving receipt required')
            verify_map(receipt.get('bindings', receipt.get('input_bindings', {})))
        counts.append(load(packet / 'execution.json')['new_ridge_fits'])
    if counts != [612, 156]:
        raise ValueError('Completed bounded fit counts changed')
    text = DOC.read_text(encoding='utf-8')
    comparisons = load(SECOND / 'comparison.json')['panels']
    unseen = [r for r in comparisons if r['arm'] == 'validation_unseen']
    keys = ('category_median', 'joint_spectral', 'eegnet_residual')
    means = {k: sum(r['metrics'][k]['participant_title_mae'] for r in unseen) / 3 for k in keys}
    expected = []
    for r in unseen:
        m = r['metrics']
        expected.append([m['category_median']['participant_title_mae'], m['joint_spectral']['participant_title_mae'],
                         r['eeg_vs_source_selected_context']['joint_spectral']['gain'], m['eegnet_residual']['participant_title_mae'],
                         r['eeg_vs_source_selected_context']['eegnet_residual']['gain']])
    expected.append([means['category_median'], means['joint_spectral'], means['category_median'] - means['joint_spectral'],
                     means['eegnet_residual'], means['category_median'] - means['eegnet_residual']])
    rows = [line for line in text.splitlines() if line.startswith('| Rotation ') or line.startswith('| Descriptive mean across rotations |')]
    if len(rows) != 4:
        raise ValueError('Findings MAE table changed')
    cells = 0
    for line, values in zip(rows, expected):
        actual = [float(v.strip()) for v in line.split('|')[2:-1]]
        if actual != [round(v, 4) for v in values]:
            raise ValueError('Findings MAE table disagrees with audited public values')
        cells += len(actual)
    pairings = load(FIRST / 'comparison.json')['pairing_panels']
    for prefix, anchor, scheme in (
        ('| Spectral: same film/acquisition, other person |', 'joint_spectral', 'person_same_film_acquisition'),
        ('| EEGNet: same film/acquisition, other person |', 'eegnet_residual', 'person_same_film_acquisition'),
        ('| Spectral: same person/coarse valence, other film |', 'joint_spectral', 'film_same_person_coarse_valence'),
        ('| EEGNet: same person/coarse valence, other film |', 'eegnet_residual', 'film_same_person_coarse_valence')):
        line = next(line for line in text.splitlines() if line.startswith(prefix))
        actual = [float(v.strip()) for v in line.split('|')[2:-1]]
        values = [next(r for r in pairings if r['rotation'] == k and r['arm'] == 'validation_unseen' and r['anchor'] == anchor and r['scheme'] == scheme)['mean_full_delta'] for k in range(3)]
        if actual != [round(v, 4) for v in values]:
            raise ValueError('Findings pairing table disagrees with audited public values')
        cells += 3
    names = [DOC, REPO / 'docs/publication/FACED_Contribution_Decision_Proposal_2026-10-12.md',
             REPO / 'docs/publication/FACED_Affordable_Venue_Routes_2026-10-12.md',
             REPO / 'docs/publication/FACED_Diagnosis_Independent_Audit_2026-10-12.md',
             REPO / 'docs/publication/FACED_Category_Supported_Context_Independent_Audit_2026-10-12.md',
             REPO / 'scripts/audit_faced_development_diagnosis_2026_10_12.py',
             REPO / 'scripts/audit_faced_category_supported_context_2026_10_12.py', Path(__file__)]
    names += [p for packet in (FIRST, SECOND) for p in packet.iterdir() if p.is_file()]
    bindings = {p.relative_to(REPO).as_posix(): digest(p) for p in names}
    receipt = {'passed': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
               'both_independent_public_audits_passed': True, 'new_ridge_fits_in_two_stages': 768,
               'new_neural_fits': 0, 'new_fits_or_original_data_access_by_finalizer': 0,
               'source_checkpoint_forward_comparisons': 24, 'source_offset_convention_checks': 48,
               'validation_forward_replay_performed': False,
               'findings_table_numeric_cells_checked': cells,
               'original_report_and_paper_bindings_preserved': True,
               'source_only_eeg_replay_occurred_in_first_stage': True,
               'new_original_rating_decoding': 0, 'confirmation_access': False,
               'research_question_changed': False, 'manuscripts_changed': False,
               'contribution_proposal_status': 'For author review; not adopted or activated',
               'mandatory_author_fee_ceiling_pkr': 10000,
               'bindings': bindings,
               'limits': 'Reused development; source category/title confounding; no independent confirmation or equivalence claim. Public audits do not independently refit Ridge or forward-replay validation EEG.'}
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'finalized': True, 'numeric_table_cells': cells, 'bound_files': len(bindings),
                      'confirmation_access': False}), flush=True)

if __name__ == '__main__':
    main()
