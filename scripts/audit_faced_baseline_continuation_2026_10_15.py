"""Declare/run an independent, bounded source-result audit. No fit or raw I/O."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.faced_baseline_result_audit import checked_panel,audit_ridge,independent_flags
from scripts.run_faced_stronger_baseline_2026_10_14 import metadata_bundle

OUT=REPO/'results/development/faced_stronger_baseline_continuation_2026-10-15'
PRIVATE=REPO.parent/'publication_runs/faced_stronger_baseline_continuation_2026-10-15'
AUDIT=REPO/'results/development/faced_baseline_independent_audit_2026-10-15'
PLAN=AUDIT/'plan.json'
SEEDS=(20261101,20261102)
EPOCHS=(120,240,360)
MAIN=tuple('context_residual_seed'+str(seed) for seed in SEEDS)
TINY=('capacity_real','capacity_permuted')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,cap=8*1024**2):
    with Path(path).open('rb') as stream: payload=stream.read(cap+1)
    if len(payload)>cap: raise PermissionError('Audit input exceeds bounded size')
    return json.loads(payload)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
def close(a,b):
    if not np.allclose(a,b,atol=1e-10,rtol=1e-10): raise ValueError('Independent numerical comparison differs')

def identities():
    _,meta,_,cell,train,held,tiny,selected=metadata_bundle()
    expected=set()
    for fit in TINY:
        expected|={(fit,e,'source_fit',*p) for e in (100,200) for p in tiny}
    for fit in MAIN:
        for arm,pairs in (('source_fit',train),('source_inference',train),('held_source',held)):
            expected|={(fit,e,arm,*p) for e in EPOCHS for p in pairs}
    expected|={('two_seed_ensemble',e,'held_source',*p) for e in EPOCHS for p in held}
    if len(expected)!=6260 or cell['cell']!=6: raise PermissionError('Fixed audit rectangle changed')
    return meta,train,held,tiny,selected,expected

def declare():
    _,train,held,tiny,selected,expected=identities()
    names=['gruxnet/faced_baseline_result_audit.py','tests/test_faced_baseline_result_audit.py',
        'scripts/audit_faced_baseline_continuation_2026_10_15.py','gruxnet/eegnet_regression.py',
        'gruxnet/eegnet_control.py','scripts/run_faced_stronger_baseline_2026_10_14.py',
        'gruxnet/faced_source_eegnet_capacity.py',
        'scripts/continue_faced_stronger_baseline_2026_10_15.py',
        'results/development/faced_stronger_baseline_continuation_2026-10-15/plan.json']
    states=[f'{fit}_epoch{e}.pt' for fit in TINY for e in (100,200)]
    states += [f'{fit}_epoch{e}.pt' for fit in MAIN for e in EPOCHS]
    save(PLAN,{'date':'2026-10-15','source_only':True,'bindings':{n:sha(REPO/n) for n in names},
        'public_prediction_rows':len(expected),'source_pairs':[list(p) for p in selected],
        'private_json_files':['source_preparation.json','capacity_real_source.json','capacity_permuted_source.json'],
        'private_tensor_files':states,'state_scalars_cap_each':2500,'private_state_bytes_cap_each':2*1024**2,
        'source_training_trials':len(train),'held_source_trials':len(held),'tiny_trials':len(tiny),
        'protected_or_validation_access':False,'old_target_cache_access':False,'raw_or_waveform_access':False,
        'new_fits':0,'new_model_selection':False,
        'limitations':'Independent saved-state geometry/equations and public prediction/metric algebra; does not independently reacquire waveforms or replay all neural forwards.'})
    print({'audit_declared':True,'public_prediction_rows':6260,'new_fits':0})

def run():
    plan=load(PLAN)
    for relative,digest in plan['bindings'].items():
        if sha(REPO/relative)!=digest: raise PermissionError('Frozen audit code changed')
    execution=load(OUT/'execution.json')
    if (execution['status']!='complete' or execution['neural_fits']!=4 or execution['prediction_rows']!=6260
            or execution['charged_fit_seconds']>7200
            or execution['protected_access'] is not False or execution['validation_access'] is not False):
        raise PermissionError('Complete source-only execution required before audit')
    meta,train,held,tiny,selected,expected=identities()
    if plan['source_pairs']!=[list(p) for p in selected]: raise PermissionError('Source audit roster changed')
    with (OUT/'predictions.csv').open(newline='',encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    panel=checked_panel(rows,expected,{p:meta[p]['family'] for p in selected})
    save(AUDIT/'started.json',{'plan_sha256':sha(PLAN),'source_only':True,'new_fits':0,
                             'execution_sha256':sha(OUT/'execution.json')})
    try:
        targets={}
        for (fit,e,arm,person,vid),(target,prediction) in panel.items():
            if fit=='capacity_permuted': continue
            pair=person,vid
            if pair in targets and targets[pair]!=target: raise ValueError('True source target disagreement')
            targets[pair]=target
        if set(targets)!=set(selected): raise PermissionError('Incomplete true source outcome rectangle')
        # Only the three named source-only JSON outputs, never an arbitrary path.
        prep=load(PRIVATE/'source_preparation.json')
        if prep['source_pairs']!=[list(p) for p in train]: raise PermissionError('Scaler/source participant identity changed')
        mean=np.asarray(prep['channel_mean']); sd=np.asarray(prep['channel_sd'])
        if mean.shape!=(30,) or sd.shape!=(30,) or not np.isfinite([mean,sd]).all() or np.any(sd<1e-6):
            raise ValueError('Invalid physical-channel scaler')
        cats=['Amusement','Anger','Disgust','Fear','Inspiration','Joy','Sadness','Tenderness']
        acqs=['legacy_1000','legacy_250','modern_1000']
        features={p:np.array([float(meta[p]['assigned_category']==c) for c in cats]
            +[float(meta[p]['acquisition_group']==a) for a in acqs]
            +[float(meta[p]['catalogue_duration_seconds']),float(meta[p]['trial'])]) for p in selected}
        matrix=lambda pairs:np.stack([features[p] for p in pairs])
        y=lambda pairs:np.array([targets[p] for p in pairs])
        residuals=[audit_ridge(matrix(train),y(train),prep['context_mean'],prep['context_scale'],
                               prep['context_coefficient'],prep['context_intercept'])]
        def ridge_values(state,pairs,prefix=''):
            return np.clip((matrix(pairs)-np.asarray(state[prefix+'mean']))/np.asarray(state[prefix+'scale'])
                           @np.asarray(state[prefix+'coefficient'])+state[prefix+'intercept'],0,7)
        context={p:float(v) for p,v in zip(selected,ridge_values(prep,selected,'context_'))}
        oof={}; seen_people=set()
        if len(prep['inner_context_ledger'])!=4: raise PermissionError('Four inner source folds required')
        for state in prep['inner_context_ledger']:
            source=[tuple(p) for p in state['training']]; inner=[tuple(p) for p in state['held']]
            people={p[0] for p in inner}
            if (len(source)!=336 or len(inner)!=112 or set(source)&set(inner) or set(source)|set(inner)!=set(train)
                    or {p[0] for p in source}&people or seen_people&people):
                raise PermissionError('Source-only participant OOF separation failed')
            seen_people|=people
            if hashlib.sha256(json.dumps(source,separators=(',',':')).encode()).hexdigest()!=state['source_identity_sha256']:
                raise PermissionError('Inner source identity byte digest differs')
            residuals.append(audit_ridge(matrix(source),y(source),state['mean'],state['scale'],state['coefficient'],state['intercept']))
            oof.update(zip(inner,ridge_values(state,inner)))
        if set(oof)!=set(train): raise PermissionError('Incomplete source OOF offsets')
        saved_oof={tuple(p):v for p,v in prep['source_oof_offsets']}
        if set(saved_oof)!=set(oof): raise PermissionError('Saved OOF recipient identity failed')
        close([saved_oof[p] for p in train],[oof[p] for p in train])
        prior={'global':float(np.median(y(train)))}
        for field,key in (('assigned_valence','coarse'),('assigned_category','category'),('vid','video')):
            groups={}
            for p in train: groups.setdefault(str(p[1] if field=='vid' else meta[p][field]),[]).append(targets[p])
            prior[key]={g:float(np.median(values)) for g,values in groups.items()}
        if prior!=prep['priors']: raise ValueError('Training-only source priors differ')
        for fit in TINY:
            item=load(PRIVATE/(fit+'_source.json'))
            if item['pairs']!=[list(p) for p in tiny]: raise PermissionError('Tiny source participant set differs')
            order=np.arange(32) if fit=='capacity_real' else np.random.default_rng(20261014).permutation(32)
            close(item['targets'],y(tiny)[order]); close(item['median'],np.median(y(tiny)))
            if item['permutation']!=order.tolist(): raise ValueError('Fixed capacity permutation differs')
            for i,p in enumerate(tiny):
                for e in (100,200): close(panel[fit,e,'source_fit',*p][0],y(tiny)[order][i])
        # All ten existing source states: bounded tensor-only geometry/norm audit.
        import torch
        from gruxnet.eegnet_regression import EEGNetRegression
        records=execution['private_state_records']
        if len(records)!=10 or {r['path'] for r in records}!=set(plan['private_tensor_files']):
            raise PermissionError('Exact ten-state filename rectangle required')
        tensor_count=0
        for record in records:
            path=PRIVATE/record['path']
            if path.stat().st_size>plan['private_state_bytes_cap_each'] or sha(path)!=record['sha256']:
                raise PermissionError('Source checkpoint bytes differ')
            state=torch.load(path,map_location='cpu',weights_only=True)
            model=EEGNetRegression(); model.load_state_dict(state,strict=True)
            count=sum(value.numel() for value in state.values()); tensor_count+=count
            if count>plan['state_scalars_cap_each'] or any(not torch.isfinite(v).all() for v in state.values()):
                raise ValueError('Invalid finite checkpoint geometry')
            for key,dims,bound in (('spatial.weight',(2,),1.),('head.weight',(1,),.25)):
                if torch.linalg.vector_norm(state[key],dim=dims).max()>bound+1e-6: raise ValueError('Checkpoint norm constraint failed')
        replay=load(OUT/'checkpoint_reload_replay.json')
        if replay['states_checked']!=10 or not replay['passed'] or {r['sha256'] for r in replay['records']}!={r['sha256'] for r in records}:
            raise PermissionError('All-checkpoint forward replay receipts missing')
        comparison=load(OUT/'comparison.json'); histories=load(OUT/'histories.json')
        history={item['fit']:item for item in histories}
        if set(history)!=set(TINY+MAIN): raise PermissionError('Four complete learning histories required')
        for fit,item in history.items():
            epochs=200 if fit in TINY else 360
            if [r['epoch'] for r in item['history']]!=list(range(1,epochs+1)): raise PermissionError('Incomplete fixed training schedule')
        mse=lambda fit,e:float(np.mean([(targets[p]-panel[fit,e,'source_fit',*p][1])**2 for p in train]))
        baseline=float(np.mean((y(train)-np.array([oof[p] for p in train]))**2))
        close(comparison['source_offset_only_mse'],baseline)
        close(comparison['source_mse_by_checkpoint'],[[mse(fit,e) for e in EPOCHS] for fit in MAIN])
        for fit in MAIN:
            close(history[fit]['starting_control_mse'],baseline)
            for e in EPOCHS:
                close([panel[fit,e,'source_inference',*p][1] for p in train],
                      [panel[fit,e,'source_fit',*p][1]-oof[p]+context[p] for p in train])
        held_y=y(held); ctx_mae=float(np.mean(np.abs(held_y-np.array([context[p] for p in held]))))
        cat_mae=float(np.mean(np.abs(held_y-np.array([prior['category'][meta[p]['assigned_category']] for p in held]))))
        metrics=[]
        for e in EPOCHS:
            values=np.array([[panel[fit,e,'held_source',*p][1] for p in held] for fit in MAIN])
            ensemble=np.clip(values.mean(axis=0),0,7)
            close(ensemble,[panel['two_seed_ensemble',e,'held_source',*p][1] for p in held])
            mae=float(np.mean(np.abs(held_y-ensemble)))
            metric=dict(epoch=e,ensemble_held_mae=mae,context_held_mae=ctx_mae,category_held_mae=cat_mae,
                        gain_over_context=ctx_mae-mae,gain_over_category=cat_mae-mae)
            stored=next(r for r in comparison['metrics'] if r['epoch']==e)
            for k,v in metric.items(): close(stored[k],v)
            metrics.append(metric)
        maes=[float(np.mean(np.abs(held_y-np.clip([panel[fit,360,'held_source',*p][1] for p in held],0,7)))) for fit in MAIN]
        flags=independent_flags(baseline,[mse(f,240) for f in MAIN],[mse(f,360) for f in MAIN],maes)
        if flags!=comparison['learning_flags'] or comparison['primary_epoch']!=360: raise ValueError('Fixed endpoint/learning flags differ')
        close(comparison['seed_held_maes'],maes)
        for fit in TINY:
            yy=np.array([panel[fit,200,'source_fit',*p][0] for p in tiny])
            pp=np.array([panel[fit,200,'source_fit',*p][1] for p in tiny])
            base=float(np.mean((yy-np.median(yy))**2)); final=float(np.mean((yy-pp)**2))
            close(history[fit]['starting_control_mse'],base); close(history[fit]['final_source_mse'],final)
            if history[fit]['capacity_screen_met']!=(base>0 and final<=.5*base): raise ValueError('Capacity flag differs')
        gate=load(OUT/'source_gate.json'); receipt=load(OUT/'target_receipt.json')
        if (not gate['passed'] or gate['pairs']!=532 or gate['waveform_bytes']!=228802560
                or gate['original_read_bytes']!={'opaque_original_hash_bytes':38720146041,'data_header_bytes':591360,'selected_signal_bytes':1044000000}
                or gate['preparation_clocks']['wall_seconds']>7200 or gate['preparation_clocks']['cpu_seconds']>7200
                or receipt['target_pairs_converted']!=532
                or receipt['validation_target_values_converted']!=0 or receipt['protected_outcomes_opened'] is not False
                or receipt['exact_conversion_receipt']['selected_target_conversion_calls_including_duplicates']!=6468):
            raise PermissionError('Source-only replay/target receipts differ')
        from scripts.qualify_faced_source_export_2026_10_14 import preservation
        preserved=preservation()
        result={'date':'2026-10-15','passed':True,'independent_audit':True,'plan_sha256':sha(PLAN),
            'prediction_rows':6260,'source_unique_targets':532,'ridge_equations_checked':5,
            'max_normalized_ridge_equation_residual':max(residuals),'checkpoint_states_checked':10,
            'checkpoint_tensor_scalar_values':tensor_count,'metrics':metrics,'learning_flags':flags,
            'source_scaler_identity_and_finite_geometry_checked':True,'source_waveform_scaler_moments_independently_recomputed':False,
            'all_neural_forwards_independently_replayed':False,'producer_fixed_subset_reload_receipts_checked':True,
            'raw_recordings_or_waveforms_read':False,'old_target_cache_read':False,'new_fits':0,
            'protected_or_validation_numeric_access':False,'papers_changed':False,'preservation':preserved,
            'private_output_file_hashes':{n:sha(PRIVATE/n) for n in plan['private_json_files']+plan['private_tensor_files']},
            'public_result_hashes':{n:sha(OUT/n) for n in ('execution.json','comparison.json','histories.json','predictions.csv','source_gate.json','target_receipt.json','checkpoint_reload_replay.json')}}
        save(AUDIT/'completion.json',result)
        print({'independent_audit_passed':True,'predictions':6260,'states':10,'new_fits':0})
    except Exception as error:
        save(AUDIT/'failure.json',{'held':True,'error':type(error).__name__+': '+str(error),'new_fits':0,'protected_access':False})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('declare','run'))
    (declare if parser.parse_args().action=='declare' else run)()
