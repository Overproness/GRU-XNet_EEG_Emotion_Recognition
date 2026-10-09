"""Trainability, paired sampling, source selection and complete low-C integration."""
from pathlib import Path
from uuid import uuid4
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from gruxnet.cbramod_adaptation import Adaptation,sampling,fit_linear,ALL_C,NEW_C,HEAD_RATE,optimizer_for
from gruxnet.cbramod_probe import fit_estimator


def toy(classes=2):
    rng=np.random.default_rng(10)
    table=pd.DataFrame({'trial_id':[f'T{i}' for i in range(36)],'subject_id':[f'S{i//6}' for i in range(36)],
        'material_key':[f'V{i%6}' for i in range(36)],'original_label':np.arange(36)%classes,'label':np.arange(36)%classes})
    idx={'train':np.arange(24),'validation_unseen':np.arange(24,30),'validation_familiar':np.arange(30,36)}
    return table,idx,rng.normal(size=(36,5)).astype(np.float32)


@pytest.mark.parametrize('classes',(2,3))
def test_sampling_is_exactly_balanced_paired_and_excludes_validation(classes):
    table,idx,_=toy(classes)
    a,w,s=sampling(table,idx['train']);b,v,t=sampling(table,idx['train'])
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(w,v);assert s==t
    assert a.shape==(200,6) and w.min()>=0 and w.max()<4
    assert not set(a.ravel())&set(np.r_[idx['validation_unseen'],idx['validation_familiar']])
    for batch in a:assert np.bincount(table.label.to_numpy()[batch]).tolist()==[6//classes]*classes
    changed=table.copy();changed.loc[24:,'label']=(changed.loc[24:,'label']+1)%classes
    c,z,u=sampling(changed,idx['train']);np.testing.assert_array_equal(a,c);assert s==u


def test_paired_heads_and_frozen_encoder_update_permission():
    torch.manual_seed(19);backbone=nn.Linear(200,200)
    import copy
    a=Adaptation(copy.deepcopy(backbone),2,False);b=Adaptation(copy.deepcopy(backbone),2,True)
    # Adaptation expects channel/patch axes; a small linear backbone preserves them.
    x=torch.randn(6,3,10,200);y=torch.arange(6)%2
    np.testing.assert_array_equal(a.head.weight.detach(),b.head.weight.detach())
    np.testing.assert_array_equal(a(x).detach(),b(x).detach())
    before=copy.deepcopy(a.backbone.state_dict())
    for model in(a,b):
        optimizer=optimizer_for(model,1e-4);optimizer.zero_grad()
        nn.functional.cross_entropy(model(x),y,label_smoothing=.1).backward();optimizer.step()
    for name,value in before.items():torch.testing.assert_close(a.backbone.state_dict()[name],value,rtol=0,atol=0)
    assert not torch.equal(a.backbone.weight,b.backbone.weight)
    assert HEAD_RATE==.001*(6/256)**.5


@pytest.mark.parametrize('classes',(2,3))
def test_expanded_fit_reuses_old_bytes_and_independently_refits_all_new_candidates(classes,monkeypatch):
    import gruxnet.cbramod_adaptation as implementation
    import scripts.audit_cbramod_adaptation as audit
    root=Path(__file__).resolve().parents[2]/'publication_runs'/f'adaptation_test_{uuid4().hex}'
    output=root/'new';output.mkdir(parents=True);(output/'plan.json').write_text('{}\n')
    table,idx,x=toy(classes);job={'dataset':'DEAP','group':1,'model':'pretrained_average'}
    old=root/implementation.PREDECESSOR/'fits'/implementation.linear_id(job);old.mkdir(parents=True)
    candidates=[]
    for k,C in enumerate(ALL_C[len(NEW_C):]):
        scaler,model,p,metrics=fit_estimator(x,table,idx,C)
        np.savez(old/f'candidate{k}.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,
            intercept=model.intercept_,classes=model.classes_,**p)
        candidates.append({'id':k,'C':C,'metrics':metrics,'parameters_sha256':implementation.sha(old/f'candidate{k}.npz')})
    (old/'record.json').write_text(json.dumps({'candidates':candidates}))
    def panel(*args,**kwargs):return table,idx,x
    monkeypatch.setattr(implementation,'old_panel',panel);monkeypatch.setattr(audit,'old_panel',panel)
    fit_linear(root,output,job);proof=audit.audit_linear(root,output,job)
    assert proof['complete'] and proof['new_candidate_refits']==4 and proof['reused_candidates_exact']==4
    assert proof['metric_sets']==24
