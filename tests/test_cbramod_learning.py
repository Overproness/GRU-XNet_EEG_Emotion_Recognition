"""Training-only boundaries, counterfactual labels, attention dropout and head replay."""
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytest
from gruxnet.cbramod_learning import tiny_definition,head_data,disable_capacity_dropout,fit_head,identifier
from scripts.audit_cbramod_learning import reconstruct_tiny,audit_head


def toy(classes):
    rng=np.random.default_rng(71)
    table=pd.DataFrame({'trial_id':[f'T{i}' for i in range(60)],'subject_id':[f'S{i//6}' for i in range(60)],
        'material_key':[f'V{i%6}' for i in range(60)],'original_label':np.arange(60)%classes,'label':np.arange(60)%classes})
    idx={'train':np.arange(36),'validation_unseen':np.arange(36,48),'validation_familiar':np.arange(48,60)}
    features=rng.normal(0,1,(60,200)).astype(np.float32)
    features[:,-1]=1.
    return table,idx,features


@pytest.mark.parametrize('classes',(2,3))
def test_tiny_membership_windows_and_permutation_are_training_only(classes):
    table,idx,_=toy(classes);before=table.copy(deep=True)
    real,rows,windows,definition=tiny_definition(table,idx['train'],'real')
    changed,p_rows,p_windows,p_def=tiny_definition(table,idx['train'],'permuted')
    np.testing.assert_array_equal(rows,p_rows);np.testing.assert_array_equal(windows,p_windows)
    assert len(set(rows))==12 and set(rows)<=set(idx['train'])
    assert not np.array_equal(real.label,changed.label)
    np.testing.assert_array_equal(np.bincount(changed.label),[12//classes]*classes)
    pd.testing.assert_frame_equal(table,before)
    independent,i_rows,i_windows,i_def=reconstruct_tiny(table,idx['train'],'permuted')
    pd.testing.assert_frame_equal(changed,independent);assert i_def==p_def
    modified=table.copy();modified.loc[36:,'label']=(modified.loc[36:,'label']+1)%classes
    _,newrows,newwindows,newdef=tiny_definition(modified,idx['train'],'real')
    np.testing.assert_array_equal(rows,newrows);np.testing.assert_array_equal(windows,newwindows);assert definition==newdef


def test_capacity_disables_attention_probability_dropout_and_module_dropout():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.attention=nn.MultiheadAttention(12,3,dropout=.8,batch_first=True)
            self.dropout=nn.Dropout(.9)
        def forward(self,x):return self.dropout(self.attention(x,x,x,need_weights=False)[0])
    model=Toy().train();x=torch.randn(4,7,12)
    assert disable_capacity_dropout(model)==2
    assert model.attention.dropout==0. and model.dropout.p==0.
    torch.testing.assert_close(model(x),model(x),rtol=0,atol=0)
    model(x).sum().backward();assert model.attention.in_proj_weight.grad is not None


@pytest.mark.parametrize('classes',(2,3))
def test_cached_scaler_is_fitted_on_training_observations_only(classes,monkeypatch):
    import gruxnet.cbramod_learning as implementation
    table,idx,x=toy(classes)
    monkeypatch.setattr(implementation,'old_panel',lambda *a,**k:(table,idx,x))
    job={'dataset':'DEAP','group':1,'pretrained':True,'scaled':True}
    _,_,z,mean,scale=head_data(Path('.'),job)
    np.testing.assert_allclose(z[idx['train']].mean(0),0,atol=1e-14)
    assert scale[-1]==1.
    x[36:]=1e12
    _,_,second,mean2,scale2=head_data(Path('.'),job)
    np.testing.assert_array_equal(mean,mean2);np.testing.assert_array_equal(scale,scale2)
    np.testing.assert_array_equal(z[:36],second[:36])


@pytest.mark.parametrize('classes',(2,3))
def test_complete_head_fit_and_independent_fp64_coefficient_replay(classes,monkeypatch):
    import gruxnet.cbramod_learning as implementation
    import scripts.audit_cbramod_learning as audit
    table,idx,x=toy(classes)
    monkeypatch.setattr(implementation,'old_panel',lambda *a,**k:(table,idx,x))
    monkeypatch.setattr(audit,'old_panel',lambda *a,**k:(table,idx,x))
    monkeypatch.setitem(implementation.STEPS,'head',(0,3,6,9))
    root=Path(__file__).resolve().parents[2]/'publication_runs'/f'cbramod_learning_test_{uuid4().hex}'
    root.mkdir(parents=True);(root/'plan.json').write_text('{}\n')
    job={'dataset':'DEAP','group':1,'pretrained':True,'scaled':True,'head_rate':.001}
    fit_head(root,root,job);proof=audit_head(root,root,job)
    assert proof['states_checked']==4 and proof['metric_sets']==12 and proof['complete']
    assert proof['max_probability_abs']<1e-12
    # The saved state is sealed: changing a coefficient must fail before replay.
    path=root/'head'/identifier('head',job)/'checkpoint9.npz'
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):audit_head(root,root,job)
