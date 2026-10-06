import copy
import numpy as np
import pandas as pd
import pytest
import torch
from gruxnet.heldout_tuning import (partitions,jobs_and_feasibility,choose,normalizer,
    population,make_model,keep_checkpoint,source_priors)
from scripts.analyze_heldout_tuning import repeated_pairs
from test_session_controls import cohort
from test_grouped_material_controls import deap_cohort


def test_full_grid_covers_each_trial_and_validation_source_roles():
    jobs,contexts,checks,retained=jobs_and_feasibility({'SEEDIV':cohort(),'DEAP':deap_cohort()})
    assert len(jobs)==2040 and len(contexts)==len(checks)==340
    assert retained==1392  # 1360 EEGNet states +32 fixed GRU sentinels.
    assert all(c['counts']['validation_familiar']>0 for c in checks)
    for dataset,base in (('SEEDIV',cohort()),('DEAP',deap_cohort())):
        job=next(j for j in jobs if j['dataset']==dataset and j['arm']=='unexposed')
        table,idx=partitions(base,job)
        val_people=set(table.iloc[idx['validation_unseen']].subject_id)
        assert set(table.iloc[idx['validation_familiar']].subject_id)==val_people
        assert set(table.iloc[idx['validation_familiar']].material_key)<=set(table.iloc[idx['train']].material_key)
        assert not set(table.iloc[idx['test']].material_key)&set(table.iloc[idx['validation_familiar']].material_key)


def test_equal_panel_proper_loss_selection_stable_ties_and_not_test_score():
    def candidate(name,loss1,loss2,ba1=.5,ba2=.5):
        return {'id':name,'validation':{p:{'binary':{'balanced_log_loss':loss,'balanced_accuracy':ba}} for p,loss,ba in
                (('validation_unseen',loss1,ba1),('validation_familiar',loss2,ba2))},'test':{'balanced_accuracy':1.}}
    a=candidate('a',.2,.8,1.,1.); b=candidate('b',.4,.5,.1,.1)
    assert choose([a,b],'DEAP')['id']=='b'
    a['test']['balanced_accuracy']=0
    assert choose([a,b],'DEAP')['id']=='b'
    c=copy.deepcopy(b); c['id']='c'
    assert choose([b,c],'DEAP')['id']=='b'
    c['validation']['validation_unseen']['binary']['balanced_accuracy']=.2
    assert choose([b,c],'DEAP')['id']=='c'


def test_source_statistics_and_participant_excluded_priors_ignore_target_changes():
    base=deap_cohort(); job=dict(dataset='DEAP',group=1,initialization=42,session=1,rotation=0,fold=0,model='eegnet',arm='unexposed')
    table,idx=partitions(base,job)
    rng=np.random.default_rng(5); x=rng.normal(size=(len(base),2,12)).astype(np.float32)
    a=normalizer(x,idx['train'],True); q=source_priors(table,idx,2)
    x[idx['test']]=1e6
    for first,second in zip(a,normalizer(x,idx['train'],True)): np.testing.assert_array_equal(first,second)
    changed=table.copy(); changed.loc[idx['test'],'label']=1-changed.loc[idx['test'],'label']
    for role,p in q.items(): np.testing.assert_array_equal(p,source_priors(changed,idx,2)[role])
    person=table.iloc[idx['train'][0]].subject_id
    changed=table.copy(); selected=idx['train'][table.iloc[idx['train']].subject_id.eq(person).to_numpy()]
    changed.loc[selected,'label']=1-changed.loc[selected,'label']
    own=np.flatnonzero(table.iloc[idx['train']].subject_id.eq(person).to_numpy())
    np.testing.assert_array_equal(q['train'][own],source_priors(changed,idx,2)['train'][own])


def test_context_population_calibration_matches_eeg_only_and_preserves_parameters_rng():
    # Short raw length gives the same convolution/BN operators at low test cost.
    from gruxnet.eegnet_control import EEGNetControl
    torch.manual_seed(17)
    plain=EEGNetControl(2,samples=64,channels=3)
    contextual=EEGNetControl(2,samples=64,channels=3,context=True)
    contextual.load_state_dict(plain.state_dict())
    x=torch.randn(13,3,64)
    parameters={k:v.detach().clone() for k,v in contextual.named_parameters()}
    rng=torch.get_rng_state().clone()
    population(plain,x); population(contextual,x)
    assert contextual.context is True and torch.equal(rng,torch.get_rng_state())
    for name,p in contextual.named_parameters(): assert torch.equal(parameters[name],p)
    for name,value in plain.state_dict().items(): assert torch.equal(value,contextual.state_dict()[name])


def test_donors_stay_in_same_group_initialization_and_fit_cell():
    rows=pd.DataFrame({'subject_id':['A','B']*4,'material_key':['v']*8,'group':[1]*4+[2]*4,
                       'seed':[42,42,91,91]*2,'source_session':1,'material_rotation':0,'fold':0})
    recipient,donor,base,eligible=repeated_pairs(rows)
    assert len(eligible)==8 and len(recipient)==8
    assert (rows.iloc[recipient].group.to_numpy()==rows.iloc[donor].group.to_numpy()).all()
    assert (rows.iloc[recipient].seed.to_numpy()==rows.iloc[donor].seed.to_numpy()).all()
    np.testing.assert_array_equal(base,np.ones(8))


def test_context_seals_before_test_and_independent_refit_rejects_corruption(pdf_workspace,monkeypatch):
    tmp_path=pdf_workspace
    import gruxnet.heldout_tuning as study
    from scripts.audit_heldout_tuning import replay_case,certificate_valid
    from gruxnet.data import write_json
    base=deap_cohort(); job=dict(dataset='DEAP',group=1,session=1,rotation=0,fold=0,arm='exposed')
    write_json(tmp_path/'config.json',{'fixture':True})
    _,idx=partitions(base,dict(job,model='context_logistic',initialization=42))
    folder=tmp_path/'contexts'/study.context_id(job)
    original=study.prior; accessed=[]
    def checked_prior(table,train,tested,classes,crossfit=False):
        if np.array_equal(tested,idx['test']):
            assert (folder/'selection.json').is_file()
            accessed.append('sealed_test')
        return original(table,train,tested,classes,crossfit)
    monkeypatch.setattr(study,'prior',checked_prior)
    study.fit_context(job,base,tmp_path)
    assert accessed==['sealed_test']
    proof=replay_case(folder,tmp_path,base,context=True)
    assert proof['passed'] and proof['maximum_probability_error']<1e-8
    certificate_valid(folder,tmp_path,True)
    with (folder/'predictions_prior_test.csv').open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError,match='Changed evidence'):
        certificate_valid(folder,tmp_path,True)


def test_neural_seal_replay_and_declared_deletion_with_small_fixture(pdf_workspace,monkeypatch):
    tmp_path=pdf_workspace
    import gruxnet.heldout_tuning as study
    import scripts.audit_heldout_tuning as auditor
    from gruxnet.data import write_json
    rows=pd.DataFrame({'trial_id':[f't{i}' for i in range(10)],'subject_id':[f'p{i}' for i in range(10)],
        'material_key':[f'v{i}' for i in range(10)],'label':[0,1]*5,'original_label':[3.,7.]*5,
        'session':1,'material_rank':range(10),'training_class':[0,1]*5})
    idx={'train':np.arange(4),'validation_unseen':np.arange(4,6),'validation_familiar':np.arange(6,8),'test':np.arange(8,10)}
    job=dict(dataset='DEAP',model='gru',group=1,initialization=42,session=1,rotation=1,fold=1,arm='unexposed')
    folder=tmp_path/'fits'/study.case_id(job)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head=torch.nn.Linear(16,2)
        def forward(self,x,log_prior=None):
            if x.abs().max()>100: assert (folder/'selection.json').exists()
            return self.head(x.reshape(len(x),-1))
    for module in (study,auditor):
        monkeypatch.setattr(module,'make_model',lambda name,classes:Tiny())
        monkeypatch.setattr(module,'partitions',lambda base,job:(rows,idx))
        monkeypatch.setattr(module,'population',lambda model,source:{})
        monkeypatch.setattr(module,'RATES',(.001,)); monkeypatch.setattr(module,'STEPS',(2,4))
    monkeypatch.setattr(study,'prefix_guard',lambda *args:None)
    x=np.random.default_rng(6).normal(size=(10,2,1,8)).astype(np.float32); x[8:]=999
    write_json(tmp_path/'config.json',{'fixture':True})
    record=study.fit_neural(job,x,x,rows,tmp_path,tmp_path,device='cpu')
    assert record['checkpoint_retained'] is False and (folder/'selected.pt').is_file()
    proof=auditor.replay_case(folder,tmp_path,rows,x,x,device='cpu')
    assert proof['passed'] and proof['maximum_probability_error']<1e-6
    assert not (folder/'selected.pt').exists()
    auditor.certificate_valid(folder,tmp_path)
    # Matching completed records resume without retraining deleted weights.
    assert study.fit_neural(job,x,x,rows,tmp_path,tmp_path,device='cpu')==record
