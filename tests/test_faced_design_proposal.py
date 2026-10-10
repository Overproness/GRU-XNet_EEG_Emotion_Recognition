import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('design',Path(__file__).resolve().parents[1]/'scripts/prepare_faced_design_proposal.py')
design=importlib.util.module_from_spec(spec);spec.loader.exec_module(design)


def rows():
    return design.load(design.INPUTS['codebook'])['stimuli']


def test_families_never_cross_roles_and_all_confirm_categories_supported():
    rotations,info=design.rotations(rows())
    for rotation in rotations:
        roles=rotation['families']
        ids=[r['family'] for values in roles.values() for r in values]
        assert len(ids)==len(set(ids))==22
        assert set(rotation['assigned_category_support']['development_source'])==set(rotation['assigned_category_support']['confirmation'])
        assert len(rotation['assigned_category_support']['confirmation'])==8
        assert len(rotation['assigned_category_support']['development_validation'])==6
        clip_roles=rotation['clips']
        assert set().union(*(set(v) for v in clip_roles.values()))==set(range(1,29))-{13,14,15,16,22}
        assert next(k for k,v in clip_roles.items() if 7 in v)==next(k for k,v in clip_roles.items() if 8 in v)
    assert info['possible_development_rotations']==64
    assert rotations[0]['clips']['confirmation']==rotations[1]['clips']['confirmation']==rotations[2]['clips']['confirmation']


def test_metadata_row_order_cannot_change_allocation():
    assert design.rotations(rows())==design.rotations(list(reversed(rows())))


def test_identity_gate_excludes_other_people_materials_and_all_confirmation():
    rotation=design.rotations(rows())[0][0]
    reservation=design.load(design.INPUTS['participant_reservation'])
    source=reservation['development_source'][0]
    validation=reservation['development_validation'][0]
    protected=reservation['confirmation'][0]
    train=rotation['clips']['development_source'][0]
    tune=rotation['clips']['development_validation'][0]
    confirm=rotation['clips']['confirmation'][0]
    assert design.eligibility(source,train,reservation,rotation,'source_training')
    assert not design.eligibility(source,confirm,reservation,rotation,'source_training')
    assert not design.eligibility(source,tune,reservation,rotation,'source_training')
    assert not design.eligibility(validation,train,reservation,rotation,'source_training')
    assert design.eligibility(validation,tune,reservation,rotation,'selection')
    assert not design.eligibility(validation,train,reservation,rotation,'selection')
    assert not design.eligibility(protected,train,reservation,rotation,'locked_development_refit')
    assert not design.eligibility(protected,confirm,reservation,rotation,'confirmation')
    assert not design.eligibility(source,train,reservation,rotation,'proposal')


def test_removed_excluded_rows_cannot_change_source_population():
    rotation=design.rotations(rows())[0][0]
    reservation=design.load(design.INPUTS['participant_reservation'])
    universe=[(s,v) for group in ('development_source','development_validation','confirmation') for s in reservation[group] for v in range(1,29)]
    permitted=[r for r in universe if design.eligibility(*r,reservation,rotation,'source_training')]
    reduced=[r for r in universe if r[0] not in reservation['confirmation'] and r[1] not in rotation['clips']['confirmation']]
    assert permitted==[r for r in reduced if design.eligibility(*r,reservation,rotation,'source_training')]


def test_shared_source_title_with_conflicting_labels_fails_closed():
    modified=rows()
    modified[7]=dict(modified[7],assigned_category='Changed')
    with pytest.raises(ValueError,match='incompatible'):
        design.rotations(modified)
