"""Manufactured CSV roles and fold-mean cache fixtures only; no data files."""
import numpy as np
import pytest
from scripts.run_faced_coverage_missing_2026_10_13 import source_targets_from_rows, mean_fixtures, array_sha

class NumericTrap:
    def __float__(self): raise AssertionError('Forbidden outcome interpretation')
    def __int__(self): raise AssertionError('Forbidden title interpretation')

def metadata():
    return {('source',2):{'role':'source','family':'family','acquisition_group':'group'}}

def row(**changes):
    r = {'subject':'source','vid':'2','arm':'source_oof','family':'family',
         'acquisition_group':'group','target':'3.5'}
    r.update(changes); return r

def test_qualification_required_before_target_iteration():
    def trap(): raise AssertionError('Rows must not be fetched')
    with pytest.raises(PermissionError): source_targets_from_rows(trap,metadata(),False)

@pytest.mark.parametrize('changes',[{'subject':'protected'}, {'subject':'validation'},
    {'subject':'unknown'}, {'arm':'validation_unseen'}, {'arm':'wrong'}])
def test_nonsource_skip_before_both_numeric_columns(changes):
    values,receipt = source_targets_from_rows([row(vid=NumericTrap(),target=NumericTrap(),**changes),row()],metadata(),True)
    assert values == {('source',2):3.5} and receipt['rows_ignored_before_target_conversion'] == 1

@pytest.mark.parametrize('changes',[{'vid':'3'},{'family':'wrong'},{'acquisition_group':'wrong'}])
def test_source_join_rejects_before_target_conversion(changes):
    with pytest.raises(PermissionError): source_targets_from_rows([row(target=NumericTrap(),**changes)],metadata(),True)

@pytest.mark.parametrize('target',['nan','inf','-1','8'])
def test_invalid_original_scale_stops(target):
    with pytest.raises(ValueError): source_targets_from_rows([row(target=target)],metadata(),True)

def test_conflicting_targets_not_imputed():
    with pytest.raises(ValueError): source_targets_from_rows([row(),row(target='4')],metadata(),True)

def test_fold_fixtures_exclude_extreme_held_and_missing_donors():
    pairs = [('source',1),('source',2),('held',3)]
    values = np.stack([np.full((3,150),2.),np.full((3,150),4.),np.full((3,150),1e9)])
    training = [('source',1),('source',2),('missing',4)]
    f = mean_fixtures(values,pairs,[{'cell':0,'training':training}])['cells'][0]
    assert f['donor_count'] == 2 and f['imputed_training_pairs'] == [('missing',4)]
    assert f['mean_sha256'] == {m:array_sha(np.full(150,3.)) for m in ('late_joint','distributed_joint')}

def test_mean_fixture_without_training_donors_stops():
    with pytest.raises(PermissionError):
        mean_fixtures(np.ones((1,3,150)),[('held',1)],[{'cell':0,'training':[('missing',2)]}])
