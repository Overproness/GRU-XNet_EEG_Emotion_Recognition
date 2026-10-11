import json
from pathlib import Path
import numpy as np
import pytest
from gruxnet.faced_classical_contract import (
    ALPHAS,role_pairs,spectral_features,context_features,fit_source_ridge,
    predict_ridge,participant_film_mae,select_alpha)

REPO=Path(__file__).resolve().parents[1]


def test_all_three_original_geometry_contracts_exclude_confirmation():
    r=json.loads((REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json').read_text())
    d=json.loads((REPO/'results/development/faced_design_proposal_2026-10-11/recommended_proposal.json').read_text())
    union=set()
    for rotation in d['rotations']:
        roles=role_pairs(r,rotation)
        assert {k:len(v) for k,v in roles.items()}=={'source_fit':560,'validation_unseen':120,'validation_familiar':160}
        for pairs in roles.values():
            assert not {p for p,v in pairs}&set(r['confirmation'])
            assert not {v for p,v in pairs}&set(rotation['clips']['confirmation'])
            union.update(pairs)
    assert len(union)==1260


def test_role_overlap_fails_before_source_preparation():
    r={'development_source':['a'],'development_validation':['b'],'confirmation':['c']}
    d={'clips':{'development_source':[1],'development_validation':[2],'confirmation':[1]}}
    with pytest.raises(ValueError,match='Film'):role_pairs(r,d)


def test_spectral_power_respects_physical_amplitude_and_uses_all_windows():
    rng=np.random.default_rng(42);values=rng.normal(size=(7,30,512))
    a=spectral_features(values);b=spectral_features(values*10)
    assert a.shape==(150,)
    np.testing.assert_allclose(b-a,2,atol=1e-12)
    changed=values.copy();changed[6]*=10
    np.testing.assert_allclose(spectral_features(changed)-a,2/7,atol=1e-12)


def test_context_uses_declared_metadata_without_video_id_or_outcome():
    m={'assigned_category':'Joy','acquisition_group':'modern_1000','catalogue_duration_seconds':60,'trial':4}
    a=context_features(m);m.update({'vid':999,'valence':float('nan')})
    np.testing.assert_array_equal(a,context_features(m));assert a.shape==(13,)


def test_source_fit_is_invariant_to_excluded_feature_and_target_changes():
    pairs=[('source',i) for i in range(6)]
    features={p:np.asarray([i,i%2],float) for i,p in enumerate(pairs)}
    targets={p:float(i) for i,p in enumerate(pairs)}
    features[('protected',1)]=np.asarray([1e15,float('nan')]);targets[('protected',1)]=float('nan')
    a=fit_source_ridge(pairs,features,targets,1.)
    features[('protected',1)]=np.asarray([-1e30,0]);targets[('protected',1)]=7
    b=fit_source_ridge(pairs,features,targets,1.)
    np.testing.assert_array_equal(a['scaler'].mean_,b['scaler'].mean_)
    np.testing.assert_array_equal(a['estimator'].coef_,b['estimator'].coef_)
    np.testing.assert_array_equal(a['scaler'].mean_,np.stack([features[p] for p in pairs]).mean(axis=0))


def test_synthetic_source_signal_learns_and_inference_does_not_refit_scaler():
    pairs=[('source',i) for i in range(30)]
    features={p:np.asarray([i/5],float) for i,p in enumerate(pairs)}
    targets={p:float(i/5) for i,p in enumerate(pairs)}
    model=fit_source_ridge(pairs,features,targets,.1)
    predicted=predict_ridge(model,pairs,features)
    assert np.mean(np.abs(predicted-np.asarray(list(targets.values()))))<.01
    before=model['scaler'].mean_.copy();features[('eval',1)]=np.asarray([1000.])
    assert predict_ridge(model,[('eval',1)],features)[0]==7
    np.testing.assert_array_equal(before,model['scaler'].mean_)


def test_film_equal_metric_does_not_count_two_clips_as_two_films():
    rows=[{'subject':p,'vid':vid,'family':fam,'target':0,'prediction':pred}
          for p in ('a','b') for vid,fam,pred in ((1,'same',0),(2,'same',2),(3,'other',5))]
    assert participant_film_mae(rows)==3
    with pytest.raises(ValueError,match='Duplicate'):participant_film_mae(rows+[rows[0]])


def test_selection_uses_only_declared_unseen_metrics_and_stable_ties():
    rows=[{'alpha':alpha,'unseen_mae':1,'unseen_mse':2,'familiar_mae':-i} for i,alpha in enumerate(ALPHAS)]
    assert select_alpha(rows)==0
    rows[2]['unseen_mse']=1;assert select_alpha(rows)==2
    rows[3]['unseen_mae']=.9;assert select_alpha(rows)==3
