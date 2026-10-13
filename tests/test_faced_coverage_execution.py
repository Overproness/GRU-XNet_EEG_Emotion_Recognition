"""Target role/conversion gates; manufactured rows only, no data I/O."""
import pytest
from scripts.run_faced_coverage_selection_2026_10_13 import source_targets_from_rows, qualify_source_ledger

class FloatTrap:
    def __float__(self):
        raise AssertionError('Forbidden target conversion')

def metadata():
    return {('source',2):{'role':'source','family':'family','acquisition_group':'group'}}

def row(**changes):
    result = {'subject':'source','vid':'2','arm':'source_oof','family':'family',
              'acquisition_group':'group','target':'3.5'}
    result.update(changes)
    return result

def test_failed_qc_rejects_before_rows_or_float():
    with pytest.raises(PermissionError):
        source_targets_from_rows([row(target=FloatTrap())],metadata(),False)

@pytest.mark.parametrize('subject',['validation','protected','unknown'])
def test_other_roles_ignored_before_vid_and_target_conversion(subject):
    result,receipt = source_targets_from_rows([row(subject=subject,vid=FloatTrap(),target=FloatTrap()),row()],metadata(),True)
    assert result == {('source',2):3.5}
    assert receipt['rows_ignored_before_target_conversion'] == 1

@pytest.mark.parametrize('changes',[{'family':'wrong'}, {'acquisition_group':'wrong'}, {'vid':'3'}])
def test_wrong_source_identity_checked_before_target(changes):
    with pytest.raises(PermissionError):
        source_targets_from_rows([row(target=FloatTrap(),**changes)],metadata(),True)

def test_conflicting_cached_source_labels_hold():
    with pytest.raises(ValueError):
        source_targets_from_rows([row(),row(target='4')],metadata(),True)

@pytest.mark.parametrize('target',['nan','inf','-0.1','7.1'])
def test_invalid_valence_holds(target):
    with pytest.raises(ValueError):
        source_targets_from_rows([row(target=target)],metadata(),True)

def test_consistent_duplicate_targets_preserved():
    values,receipt = source_targets_from_rows([row(),row()],metadata(),True)
    assert values == {('source',2):3.5} and receipt['source_rows_agreeing'] == 2

def test_non_source_arm_skipped_before_float():
    values,receipt = source_targets_from_rows([row(arm='validation_unseen',target=FloatTrap()),row()],metadata(),True)
    assert values == {('source',2):3.5} and receipt['rows_ignored_before_target_conversion'] == 1

def ledger_fixture():
    pairs = [(f'person-{p:03d}',v) for p in range(70) for v in range(14)]
    rows = [{'subject':p,'vid':v,'role':'source','region':r,'eeg_available':True,'reasons':[]}
            for p,v in pairs for r in ('early','middle','late')]
    gate = {'passed':True,'unavailable_regions':0,'activation_sha256':'activation',
            'source_regions_sha256':'ledger','source_regions':2940,'source_trials':980,
            'cached_targets_converted':0,'new_fits':0}
    audit = {'passed':True,'source_quality_gate_passed':True,
             'source_regions_sha256':'ledger','source_gate_sha256':'gate'}
    return rows,pairs,gate,audit

@pytest.mark.parametrize('violation',['unavailable_region','flagged_region','duplicate_region',
    'incomplete_rectangle','audit_other_gate','audit_false_quality','gate_other_activation'])
def test_region_and_audit_tampering_holds_before_target_preflight(violation):
    rows,pairs,gate,audit = ledger_fixture()
    if violation == 'unavailable_region': rows[0]['eeg_available'] = False
    if violation == 'flagged_region': rows[0]['reasons'] = ['acquisition_failure']
    if violation == 'duplicate_region': rows[-1] = rows[0]
    if violation == 'incomplete_rectangle': rows.pop()
    if violation == 'audit_other_gate': audit['source_gate_sha256'] = 'changed'
    if violation == 'audit_false_quality': audit['source_quality_gate_passed'] = False
    if violation == 'gate_other_activation': gate['activation_sha256'] = 'changed'
    with pytest.raises(PermissionError):
        qualify_source_ledger(rows,pairs,gate,audit,'activation','ledger','gate')

def test_complete_audited_rectangle_allows_source_preflight():
    rows,pairs,gate,audit = ledger_fixture()
    qualified = qualify_source_ledger(rows,pairs,gate,audit,'activation','ledger','gate')
    assert len(qualified) == 2940
