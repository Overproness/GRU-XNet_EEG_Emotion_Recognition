"""Independent coefficient links, batch-changed checkpoint replay and source selection."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.attention import SDPBackend,sdpa_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import softmax
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.cbramod_learning import (PRIOR,PREDECESSOR,STEPS,ROLES,identifier,sha,atomic,stamp,
    source_data,sampling,new_model,old_panel,state_digest,validate)


def measures(y,p):
    return {'n':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,p.argmax(1))),
        'balanced_log_loss':float(np.average(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),
                                            weights=compute_sample_weight('balanced',y)))}


def check_exposure(record,train,draws,windows,steps,kind):
    expected=[]
    for step in steps:
        if not step:continue
        stream=draws[:step].reshape(-1);counts={int(i):0 for i in train}
        for i in stream:counts[int(i)]+=1
        item={'step':step,'training_observations':len(train),'draws':6*step,
            'unique_observations_seen':sum(v>0 for v in counts.values()),
            'minimum_draws_per_observation':min(counts.values()),'maximum_draws_per_observation':max(counts.values())}
        if kind=='long':
            pairs=np.column_stack((stream,windows[:step].reshape(-1)))
            item.update({'unique_observation_window_pairs':len(np.unique(pairs,axis=0)),
                         'available_observation_window_pairs':4*len(train)})
        expected.append(item)
    if expected!=record['exposure']:raise ValueError('Training exposure mismatch')


def read_record(output,kind,job):
    folder=output/kind/identifier(kind,job);record=json.loads((folder/'record.json').read_text())
    if record['job']!=job or record['plan_sha256']!=sha(output/'plan.json'):raise ValueError('Wrong learning record binding')
    if [i['step'] for i in record['candidates']]!=list(STEPS[kind]):raise ValueError('Missing/incorrect checkpoints')
    for n,s in record['artifact_sha256'].items():
        if sha(folder/n)!=s:raise ValueError('Learning artifact changed')
    return folder,record


def compare_csv(folder,item,role,table,rows,p,tolerance):
    frame=pd.read_csv(folder/f'step{item["step"]}_{role}.csv')
    expected=table.iloc[rows][['trial_id','subject_id','material_key','original_label','label']].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[list(expected.columns)],expected)
    if not(frame.role==role).all():raise ValueError('Wrong prediction role')
    saved=frame[[f'p{c}' for c in range(p.shape[1])]].to_numpy(dtype=float)
    np.testing.assert_allclose(saved.sum(1),1.,atol=2e-12,rtol=0)
    np.testing.assert_allclose(saved,p,atol=tolerance,rtol=0)
    direct=measures(expected.label.to_numpy(),saved);replayed=measures(expected.label.to_numpy(),p)
    maximum=0.
    for name in direct:
        if abs(direct[name]-item['metrics'][role][name])>2e-11:raise ValueError('Published metric mismatch')
        error=abs(replayed[name]-item['metrics'][role][name]);maximum=max(maximum,error)
        if error>tolerance:raise ValueError('Replay metric mismatch')
    return float(np.max(np.abs(saved-p))),maximum


def audit_head(root,output,job):
    folder,record=read_record(output,'head',job)
    table,idx,x=old_panel(root,root/PREDECESSOR,job['dataset'],job['group'],
                         'pretrained_average' if job['pretrained'] else 'random42_average')
    x=np.asarray(x,dtype=np.float64)
    if job['scaled']:
        scaler=StandardScaler().fit(x[idx['train']]);mean=scaler.mean_;scale=scaler.scale_
    else:mean=np.zeros(200);scale=np.ones(200)
    observations,windows,signature=sampling(table,idx['train'],updates=STEPS['head'][-1])
    if signature!=record['sampling_digest']:raise ValueError('Changed head draws')
    check_exposure(record,idx['train'],observations,windows,STEPS['head'],'head')
    torch.manual_seed(4242);initial=nn.Linear(200,int(table.label.max()+1)).double()
    if state_digest(initial.state_dict())!=record['initial_head_digest']:raise ValueError('Changed initial head')
    maximum=0.;metric_max=0.;sets=0
    for item in record['candidates']:
        if sha(folder/item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('Head coefficient binding changed')
        with np.load(folder/item['checkpoint'],allow_pickle=False) as state:
            np.testing.assert_array_equal(state['mean'],mean);np.testing.assert_array_equal(state['scale'],scale)
            loaded={'weight':torch.tensor(state['weight']),'bias':torch.tensor(state['bias'])}
            if state_digest(loaded)!=item['state_digest']:raise ValueError('Head state digest mismatch')
            if item['step']==0:
                for name,value in loaded.items():torch.testing.assert_close(value,initial.state_dict()[name],rtol=0,atol=0)
            elif item['state_digest']==record['initial_head_digest']:raise ValueError('Head never updated')
            # Direct NumPy coefficient link, independent of the training forward.
            p=softmax(((x-mean)/scale)@state['weight'].T+state['bias'],axis=1)
        for role,rows in idx.items():
            a,b=compare_csv(folder,item,role,table,rows,p[rows],2e-11)
            maximum=max(maximum,a);metric_max=max(metric_max,b);sets+=1
    proof={'complete':True,'record_sha256':sha(folder/'record.json'),'states_checked':len(record['candidates']),
        'metric_sets':sets,'max_probability_abs':maximum,'max_metric_abs':metric_max,'created_utc':stamp()}
    atomic(folder/'verification.json',proof);return proof


def reconstruct_tiny(table,train,target):
    y=table.label.to_numpy(dtype=int);classes=int(y.max()+1);rng=np.random.default_rng(20261012)
    chosen=[]
    for c in range(classes):chosen.extend(rng.choice(train[y[train]==c],12//classes,replace=False).tolist())
    chosen=np.asarray(chosen)[rng.permutation(12)];windows=np.random.default_rng(20261013).integers(0,4,size=12)
    labels=y[chosen].copy()
    if target=='permuted':labels=labels[np.random.default_rng(20261014).permutation(12)]
    frame=table.iloc[chosen].copy().reset_index(drop=True);frame['label']=labels
    definition={'trial_ids':frame.trial_id.tolist(),'windows':windows.tolist(),'target_labels':labels.tolist(),
                'original_task_labels':y[chosen].tolist(),'source_only':True}
    if not set(chosen)<=set(train) or len(set(chosen))!=12:raise ValueError('Invalid source-only capacity subset')
    np.testing.assert_array_equal(np.bincount(labels),np.full(classes,12//classes))
    if target=='permuted' and np.array_equal(labels,y[chosen]):raise ValueError('Identity permutation')
    return frame,chosen,windows,definition


def replay(model,data,rows,kind):
    x=data[rows] if kind=='tiny' else data[rows].reshape(len(rows)*4,data.shape[2],10,200)
    batch=3 if kind=='tiny' else 8;parts=[];model.eval()
    with sdpa_kernel(SDPBackend.MATH),torch.inference_mode():
        for start in range(0,len(x),batch):
            parts.append(model(torch.from_numpy(np.array(x[start:start+batch],copy=True)).to('cuda')).cpu().numpy())
    logits=np.concatenate(parts).astype(np.float64)
    if kind=='long':logits=np.sum(logits.reshape(len(rows),4,-1),axis=1)/4
    return softmax(logits,axis=1)


def audit_neural(root,output,kind,job):
    folder,record=read_record(output,kind,job);table,idx,data=source_data(root,job)
    if kind=='tiny':
        table,rows,windows,definition=reconstruct_tiny(table,idx['train'],job['target'])
        if definition!=record['tiny_definition']:raise ValueError('Changed capacity membership/window/target')
        data=data[rows,windows];idx={'capacity_train':np.arange(12)};train=idx['capacity_train']
    else:train=idx['train']
    observations,windows,signature=sampling(table,train,updates=STEPS[kind][-1])
    if signature!=record['sampling_digest']:raise ValueError('Changed neural sampling')
    check_exposure(record,train,observations,windows,STEPS[kind],kind)
    if kind=='long':
        prior,oldwindows,_=sampling(table,train,updates=200)
        np.testing.assert_array_equal(observations[:200],prior);np.testing.assert_array_equal(windows[:200],oldwindows)
    model=new_model(root,int(table.label.max()+1),job['pretrained'],kind=='tiny' or job['trainable'])
    if kind=='tiny':
        count=0
        for module in model.modules():
            if isinstance(module,nn.Dropout):module.p=0.;count+=1
            elif isinstance(module,nn.MultiheadAttention):module.dropout=0.;count+=1
        if count!=record['disabled_dropout_modules']:raise ValueError('Capacity dropout configuration mismatch')
    if state_digest(model.backbone.state_dict())!=record['initial_encoder_digest'] or \
            state_digest(model.head.state_dict())!=record['initial_head_digest']:raise ValueError('Initial state mismatch')
    maximum=0.;metric_max=0.;sets=0
    for item in record['candidates']:
        if sha(folder/item['checkpoint'])!=item['checkpoint_sha256']:raise ValueError('Neural checkpoint bytes changed')
        state=torch.load(folder/item['checkpoint'],map_location='cpu',weights_only=True)
        if state_digest(state)!=item['state_digest']:raise ValueError('Neural state binding mismatch')
        model.load_state_dict(state,strict=True)
        encoder=state_digest(model.backbone.state_dict());head=state_digest(model.head.state_dict())
        if encoder!=item['encoder_digest'] or head!=item['head_digest']:raise ValueError('Module digest mismatch')
        if item['step']==0:
            if encoder!=record['initial_encoder_digest'] or head!=record['initial_head_digest']:raise ValueError('Wrong initial checkpoint')
        else:
            if kind=='long' and not job['trainable']:
                if encoder!=record['initial_encoder_digest']:raise ValueError('Frozen encoder changed')
            elif encoder==record['initial_encoder_digest']:raise ValueError('Trainable encoder never updated')
            if head==record['initial_head_digest']:raise ValueError('Neural head never updated')
        for role,rows in idx.items():
            p=replay(model,data,rows,kind);a,b=compare_csv(folder,item,role,table,rows,p,2e-6)
            maximum=max(maximum,a);metric_max=max(metric_max,b);sets+=1
        if state_digest(model.state_dict())!=item['state_digest']:raise ValueError('Replay mutated state')
    if kind=='tiny':
        values=record['candidates'][-1]['metrics']['capacity_train']
        if record['capacity_pass']!=(values['balanced_accuracy']>=.95 and values['balanced_log_loss']<=.1):
            raise ValueError('Capacity threshold changed')
    proof={'complete':True,'record_sha256':sha(folder/'record.json'),'states_checked':len(record['candidates']),
        'metric_sets':sets,'max_probability_abs':maximum,'max_metric_abs':metric_max,'created_utc':stamp()}
    atomic(folder/'verification.json',proof);return proof


def select(items):
    return min(range(len(items)),key=lambda k:(sum(items[k]['metrics'][r]['balanced_log_loss'] for r in ROLES[1:])/2,
        -sum(items[k]['metrics'][r]['balanced_accuracy'] for r in ROLES[1:])/2,k))


def finish(root,output):
    plan=validate(root,output,deep_inputs=True);proofs=[];allrecords={};initializations={};choices={'head':{},'long':{}}
    for kind in STEPS:
        allrecords[kind]=[]
        for job in plan['jobs'][kind]:
            folder,record=read_record(output,kind,job);proof=json.loads((folder/'verification.json').read_text())
            if not proof['complete'] or proof['record_sha256']!=sha(folder/'record.json'):raise ValueError('Stale/missing case proof')
            proofs.append(proof);allrecords[kind].append(record)
            pair=(kind,job['dataset'],job['group'])
            initializations.setdefault(pair,[]).append(record)
            if kind=='head':key=(job['dataset'],job['group'],job['pretrained'],job['scaled'])
            elif kind=='long':key=(job['dataset'],job['group'],job['pretrained'],job['trainable'])
            else:continue
            choices[kind].setdefault(key,[])
            choices[kind][key].extend({**item,'trajectory':folder.name,'job':job}
                                      for item in record['candidates'] if item['step']>0)
    for (kind,_,_),records in initializations.items():
        if len({r['initial_head_digest'] for r in records})!=1:raise ValueError('Unmatched paired head initializations')
        if kind!='tiny' and len({r['sampling_digest'] for r in records})!=1:raise ValueError('Unmatched paired sampling')
        if kind!='head':
            for pretrained in (True,False):
                subset=[r for r in records if r['job']['pretrained']==pretrained]
                if len({r['initial_encoder_digest'] for r in subset})!=1:raise ValueError('Unmatched paired encoder initializations')
            if kind=='tiny':
                if len({tuple(r['tiny_definition']['trial_ids']) for r in records})!=1 or \
                        len({tuple(r['tiny_definition']['windows']) for r in records})!=1:
                    raise ValueError('Unmatched capacity subsets/windows')
    selected={}
    for kind,conditions in choices.items():
        selected[kind]=[]
        for key,items in conditions.items():
            if len(items)!=(9 if kind=='head' else 3):raise ValueError('Missing source-selection candidates')
            winner=items[select(items)]
            selected[kind].append({'condition':list(key),'selected':winner,'candidates':items})
    if len(selected['head'])!=16 or len(selected['long'])!=16:raise ValueError('Incomplete selected conditions')
    result={'development_only':True,'outer_test_inferences':0,'research_question_change_approved':False,
        'selected':selected,'capacity':[{'job':r['job'],'capacity_pass':r['capacity_pass'],
            'candidates':r['candidates']} for r in allrecords['tiny']]}
    atomic(output/'summary.json',result)
    proof={'complete':True,'plan_sha256':sha(output/'plan.json'),'summary_sha256':sha(output/'summary.json'),
        'head_trajectories':48,'tiny_trajectories':16,'long_trajectories':16,
        'states_checked':sum(p['states_checked'] for p in proofs),'metric_sets':sum(p['metric_sets'] for p in proofs),
        'head_selected_conditions':16,'long_selected_conditions':16,
        'maximum_probability_abs':max(p['max_probability_abs'] for p in proofs),
        'maximum_metric_abs':max(p['max_metric_abs'] for p in proofs),'created_utc':stamp()}
    atomic(output/'verification.json',proof);return proof
