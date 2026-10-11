"""Meaningful regression/access tests using only synthetic or public technical inputs."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gruxnet.eegnet_control import EEGNetControl
from gruxnet.eegnet_regression import EEGNetRegression, original_scale_prediction
from gruxnet import faced_eegnet_runner as runner

REPO=Path(__file__).resolve().parents[1]
PUBLIC=REPO/'results/development'


class RecordingMap(dict):
    def __init__(self,items):super().__init__(items);self.read=[]
    def __getitem__(self,key):self.read.append(key);return super().__getitem__(key)


def public_bundle():
    programme=json.loads((PUBLIC/'faced_development_protocol_2026-10-11/plan.json').read_bytes())
    signal=json.loads((PUBLIC/'faced_inference_quality_2026-10-11/signal_audit.json').read_bytes())
    decisions={(r['subject'],r['vid']):r for r in signal['trial_decisions']}
    films=json.loads((PUBLIC/'faced_backup_2026-10-10/stimulus_metadata.json').read_bytes())['records']
    films={r['clip']:r for r in films}
    metadata={p:{'family':films[p[1]]['source_film_sha256'],'role':d['role']} for p,d in decisions.items()}
    return programme,metadata,decisions


class EEGNetPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_backbone_is_original_and_scalar_count(self):
        torch.manual_seed(23);original=EEGNetControl(classes=2,samples=512,channels=30,dropout=.5,context=False)
        torch.manual_seed(23);model=EEGNetRegression()
        self.assertEqual(sum(p.numel() for p in model.parameters()),1841)
        for key,value in model.state_dict().items():
            if not key.startswith('head.'):self.assertTrue(torch.equal(value,original.state_dict()[key]),key)
        self.assertEqual(model.head.out_features,1)
        self.assertEqual(float(model.head.bias.detach()),0.)
        self.assertLessEqual(float(model.head.weight.detach().abs().max()),math.sqrt(6/257))
        original.eval();model.eval()
        features=[]
        hook=original.head.register_forward_pre_hook(lambda module,args:features.append(args[0].detach()))
        x=torch.randn(14,30,512);original(x);hook.remove()
        expected=features[0]@model.head.weight.T+model.head.bias
        torch.testing.assert_close(model.forward_windows(x),expected[:,0],rtol=0,atol=0)

    def test_trial_mean_and_gradient_weights(self):
        model=EEGNetRegression().eval()
        leaf=torch.arange(14,dtype=torch.float32,requires_grad=True)
        with patch.object(model,'forward_windows',return_value=leaf):
            prediction=model(torch.zeros(2,7,30,512))
            torch.testing.assert_close(prediction,torch.tensor([3.,10.]))
            prediction.sum().backward()
        torch.testing.assert_close(leaf.grad,torch.full((14,),1/7))
        with self.assertRaises(ValueError):model(torch.zeros(7,30,512))
        torch.testing.assert_close(original_scale_prediction(torch.tensor([-1.,2.]),torch.tensor([2.,3.])),torch.tensor([-5.,17.]))

    def test_backward_bn_and_constraints(self):
        model=EEGNetRegression().train();optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        old=model.bn1.running_mean.clone()
        loss=((model(torch.randn(8,7,30,512))-.2)**2).mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step();model.constrain()
        self.assertFalse(torch.equal(old,model.bn1.running_mean))
        self.assertLessEqual(float(model.spatial.weight.detach().square().sum(dim=2).sqrt().max()),1+1e-6)
        self.assertLessEqual(float(model.head.weight.detach().norm(dim=1).max()),.25+1e-6)
        old=model.bn1.running_mean.clone();model.eval()
        with torch.no_grad():model(torch.randn(1,7,30,512))
        self.assertTrue(torch.equal(old,model.bn1.running_mean))

    def test_source_scaler_matches_independent_axes(self):
        rng=np.random.default_rng(9)
        pairs=[('source',1),('source',2)]
        windows=RecordingMap({p:rng.normal(size=(7,30,512))*(np.arange(30)+1)[None,:,None]+p[1] for p in pairs})
        windows['held',1]=np.full((7,30,512),1e12)
        scaler=runner.SourceChannelScaler.fit(pairs,windows)
        joined=np.concatenate([dict.__getitem__(windows,p).transpose(1,0,2).reshape(30,-1) for p in pairs],axis=1)
        np.testing.assert_allclose(scaler.mean,joined.mean(1),atol=1e-13)
        np.testing.assert_allclose(scaler.sd,joined.std(1),atol=1e-13)
        self.assertEqual(windows.read,pairs)
        self.assertEqual(scaler.transform(dict.__getitem__(windows,pairs[0])).dtype,np.float32)
        constant=runner.SourceChannelScaler.fit([pairs[0]],{pairs[0]:np.ones((7,30,512))})
        np.testing.assert_array_equal(constant.sd,np.full(30,1e-6))

    def test_oof_excludes_entire_recipient_person(self):
        pairs=[(str(person),clip) for person in range(10) for clip in (1,2)]
        folds=[[str(p),str(p+1)] for p in range(0,10,2)]
        features=RecordingMap({p:np.array([int(p[0]),p[1],1.]) for p in pairs})
        targets=RecordingMap({p:float(int(p[0])%7) for p in pairs})
        full,oof,ledger=runner.source_context(pairs,folds,features,targets,1.)
        self.assertEqual(set(oof),set(pairs));self.assertEqual(full['source_pairs'],pairs)
        for entry in ledger:
            self.assertFalse({p[0] for p in entry['source_pairs']}&{p[0] for p in entry['held_pairs']})
            self.assertEqual(len(entry['held_pairs']),4)
        self.assertEqual(set(targets.read),set(pairs))
        with self.assertRaises(PermissionError):runner.source_context(pairs,folds[:-1],features,targets,1.)

    def test_public_technical_rectangles_pass(self):
        programme,metadata,decisions=public_bundle()
        self.assertEqual(len(runner.guard_bundle(programme,metadata,decisions)),1260)

    def test_role_duplicate_and_source_failure_guards(self):
        programme,metadata,decisions=public_bundle()
        bad=copy.deepcopy(programme);bad['rotation_manifests'].reverse()
        with self.assertRaises(PermissionError):runner.guard_bundle(bad,metadata,decisions)
        bad=copy.deepcopy(programme);bad['input_access']['future_development_pairs'].append(bad['input_access']['future_development_pairs'][0])
        with self.assertRaises(PermissionError):runner.guard_bundle(bad,metadata,decisions)
        pair=next(p for p,m in metadata.items() if m['role']=='source')
        wrong=copy.deepcopy(decisions);wrong[pair]['eeg_available']=False;wrong[pair]['reasons']=['rail']
        with self.assertRaises(PermissionError):runner.guard_bundle(programme,metadata,wrong)
        wrong=copy.deepcopy(metadata);wrong[pair]['role']='confirmation'
        with self.assertRaises(PermissionError):runner.guard_bundle(programme,wrong,decisions)

    def test_missing_activation_stops_before_any_mapping(self):
        poison=RecordingMap({})
        with self.assertRaises(PermissionError):
            runner.run_diagnostic({},poison,poison,poison,poison,poison,{},activation=None,expected_plan_sha256='fixed',code_bindings={},device='cuda')
        self.assertEqual(poison.read,[])
        with self.assertRaises(PermissionError):runner.require_activation({'status':'active_development_diagnostic','preflight_plan_sha256':'fixed','code_bindings':{},'confirmation_access':True,'research_question_changed':False},'fixed',{})

    def test_strong_controls_require_full_identity_panel(self):
        programme,_,_=public_bundle()
        controls={r['rotation']:{name:{(p['subject'],p['vid']):3.5 for p in r['arms']['validation_unseen']} for name in runner.STRONG_CONTROLS} for r in programme['rotation_manifests']}
        runner.guard_controls(programme,controls)
        wrong=copy.deepcopy(controls);del wrong[1]['context_kernel']
        with self.assertRaises(PermissionError):runner.guard_controls(programme,wrong)
        wrong=copy.deepcopy(controls);wrong[2]['context_only']['protected',7]=3.
        with self.assertRaises(PermissionError):runner.guard_controls(programme,wrong)

    def test_fallback_never_fetches_failed_eeg(self):
        pairs=[('person',1),('person',2)]
        windows=RecordingMap({pairs[0]:np.zeros((7,30,512))})
        scaler=runner.SourceChannelScaler(np.zeros(30),np.ones(30),tuple(pairs[:1]))
        decisions={pairs[0]:{'eeg_available':True},pairs[1]:{'eeg_available':False}}
        class Constant(torch.nn.Module):
            def forward(self,x):return torch.ones(len(x),device=x.device)*.5
        prediction=runner.predict_trials(Constant(),pairs,windows,scaler,{pairs[0]:2.,pairs[1]:6.},decisions,'cpu')
        np.testing.assert_array_equal(prediction,[5.5,6.]);self.assertEqual(windows.read,pairs[:1])

    def test_ensemble_clips_after_averaging_and_ties(self):
        pairs=[('p',1)];targets={pairs[0]:3.};metadata={pairs[0]:{'family':'f'}}
        records=[{'learning_rate':rate,'seed':seed,'checkpoints':{e:{'validation_unseen_raw':np.array([-2. if seed==runner.SEEDS[0] else 8.])} for e in runner.CHECKPOINTS}} for rate in runner.RATES for seed in runner.SEEDS]
        choice,_=runner.select_schedule(records,pairs,targets,metadata)
        self.assertEqual(choice,{'learning_rate':.0003,'epoch':30,'mae':0.})
        with self.assertRaises(PermissionError):runner.select_schedule(records[:-1],pairs,targets,metadata)
        records[0]['checkpoints'][60]['validation_unseen_raw'][0]=np.nan
        with self.assertRaises(ValueError):runner.select_schedule(records,pairs,targets,metadata)

    def test_learning_flags_and_exact_boundaries(self):
        self.assertEqual(runner.training_mse_flags(10,[8,8],[8,8],[1,1]),[])
        self.assertIn('insufficient_source_learning',runner.training_mse_flags(10,[9,9],[8.0001,8],[1,1]))
        self.assertNotIn('budget_limited_convergence',runner.training_mse_flags(10,[5,5],[4.5,4.5],[1,1]))
        self.assertIn('budget_limited_convergence',runner.training_mse_flags(10,[5,5],[4.49,4.5],[1,1]))
        self.assertIn('validation_seed_divergence',runner.training_mse_flags(10,[5,5],[5,5],[1,1.151]))
        self.assertIn('source_seed_divergence',runner.training_mse_flags(10,[2,2],[1,2.001],[1,1]))
        self.assertEqual(runner.training_mse_flags(10,[1,np.nan],[1,1],[1,1]),['invalid_or_nonfinite_learning_metrics'])

    def test_budget_counts_failed_attempts_and_elapsed(self):
        ticks=[0.];budget=runner.FitBudget(seconds_cap=10,fit_cap=1,clock=lambda:ticks[0])
        budget.begin();ticks[0]=3.;budget.end()
        self.assertEqual((budget.fits,budget.elapsed),(1,3.))
        with self.assertRaises(PermissionError):budget.begin()
        budget=runner.FitBudget(seconds_cap=10,clock=lambda:ticks[0]);budget.begin();ticks[0]=13.
        with self.assertRaises(TimeoutError):budget.check()
        budget.end();self.assertEqual(budget.elapsed,10.)

    def test_synthetic_fit_replays_and_separates_source_offsets(self):
        rng=np.random.default_rng(4);source=[('s',i) for i in range(8)];held=[('v',0)]
        windows={p:rng.normal(size=(7,30,512)) for p in source+held}
        targets={p:float(i%7) for i,p in enumerate(source+held)}
        arms={'source_fit':source,'validation_unseen':held,'validation_familiar':held}
        offsets={a:{p:2. for p in pairs} for a,pairs in arms.items()}
        decisions={p:{'eeg_available':True} for p in windows}
        scaler=runner.SourceChannelScaler.fit(source,windows);events=[]
        options=dict(rate=.001,seed=runner.SEEDS[0],device='cpu',epochs=2,checkpoints=(1,2),source_inference_offsets={p:3. for p in source})
        first=runner.fit_trial_model(source,arms,windows,targets,offsets,scaler,decisions,budget=runner.FitBudget(),event=events.append,**options)
        second=runner.fit_trial_model(source,arms,windows,targets,offsets,scaler,decisions,budget=runner.FitBudget(),**options)
        for epoch in (1,2):
            a=first['checkpoints'][epoch];b=second['checkpoints'][epoch]
            np.testing.assert_array_equal(a['source_fit_raw'],b['source_fit_raw'])
            np.testing.assert_allclose(a['source_inference_raw'],a['source_fit_raw']+1,atol=1e-14)
            self.assertAlmostEqual(a['training_target_source_mse'],np.mean((runner.valid_targets(source,targets)-a['source_fit_raw'])**2))
        self.assertEqual(sum(e['type']=='checkpoint' for e in events),2)

    def test_twenty_fit_transfer_orchestration_on_tagged_fixture(self):
        programme,metadata,decisions=public_bundle()
        pairs=sorted(metadata)
        # Public identities, exclusively fabricated features/targets/predictions.
        rng=np.random.default_rng(31)
        features={p:rng.normal(size=13) for p in pairs};targets={p:3. for p in pairs}
        for p in pairs:metadata[p].update(assigned_category='Joy',assigned_valence='Positive',vid=p[1])
        controls={r['rotation']:{name:{(p['subject'],p['vid']):3.4 for p in r['arms']['validation_unseen']} for name in runner.STRONG_CONTROLS} for r in programme['rotation_manifests']}
        calls=[];events=[];budget=runner.FitBudget()
        def fake_fit(source,arms,windows,targets,offsets,scaler,decisions,**kwargs):
            calls.append((kwargs['rate'],kwargs['seed']));kwargs['budget'].begin();kwargs['budget'].end()
            checkpoints={}
            for epoch in runner.CHECKPOINTS:
                error=.05 if kwargs['rate']==.001 and epoch==60 else .3
                checkpoints[epoch]={arm+'_raw':np.full(len(pairs),3+error) for arm,pairs in arms.items()}
                checkpoints[epoch].update(training_target_source_mse=5.,source_inference_raw=np.full(len(source),3+error))
            return {'learning_rate':kwargs['rate'],'seed':kwargs['seed'],'starting_control_mse':10.,'checkpoints':checkpoints}
        activation={'status':'active_development_diagnostic','preflight_plan_sha256':'synthetic','code_bindings':{},'confirmation_access':False,'research_question_changed':False}
        with patch.object(runner.SourceChannelScaler,'fit',return_value=runner.SourceChannelScaler(np.zeros(30),np.ones(30),())),patch.object(runner,'fit_trial_model',side_effect=fake_fit):
            result=runner.run_diagnostic(programme,metadata,decisions,RecordingMap({}),targets,features,controls,activation=activation,expected_plan_sha256='synthetic',code_bindings={},device='cuda',budget=budget,event=events.append)
        self.assertEqual(len(calls),20);self.assertEqual(len(result['rotations']),6)
        self.assertTrue(all(result['engineering_leads'].values()))
        self.assertEqual(calls[12:],[(.001,seed) for _ in range(4) for seed in runner.SEEDS])
        self.assertTrue(all(s['epoch']==60 for s in result['selection'].values()))
        self.assertEqual(sum(e['type']=='selection' for e in events),2)

    def test_numerical_failure_records_attempt_and_history(self):
        p=('synthetic',1);windows={p:np.zeros((7,30,512))};targets={p:3.}
        arms={'source_fit':[p]};offsets={'source_fit':{p:3.}}
        scaler=runner.SourceChannelScaler.fit([p],windows);budget=runner.FitBudget();events=[]
        with patch.object(EEGNetRegression,'forward',return_value=torch.tensor([float('nan')],requires_grad=True)):
            with self.assertRaises(FloatingPointError):
                runner.fit_trial_model([p],arms,windows,targets,offsets,scaler,{p:{'eeg_available':True}},rate=.001,seed=runner.SEEDS[0],device='cpu',budget=budget,event=events.append,epochs=1,checkpoints=(1,))
        self.assertEqual(budget.fits,1);self.assertIsNone(budget.started)
        self.assertEqual(events[-1]['type'],'failure');self.assertEqual(events[-1]['history'],[])


if __name__=='__main__':unittest.main()
