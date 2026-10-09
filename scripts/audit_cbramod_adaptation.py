"""Independent expanded head refits and batch-changed neural checkpoint replay."""
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.cbramod_adaptation import (PREDECESSOR, ROLES, ALL_C, NEW_C, STEPS,
    sha, atomic, stamp, old_panel, source_data, sampling, new_model, linear_id,
    neural_id, state_digest, validate)


def measures(y,p):
    return {'n':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,p.argmax(1))),
        'balanced_log_loss':float(np.average(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),
                                            weights=compute_sample_weight('balanced',y)))}


def csv_check(path, table, rows, p, tolerance):
    saved=pd.read_csv(path)
    expected=table.iloc[rows][['trial_id','subject_id','material_key','label']].reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[list(expected.columns)],expected)
    np.testing.assert_allclose(saved[[f'p{c}' for c in range(p.shape[1])]],p,atol=tolerance,rtol=0)


def audit_linear(root,output,job):
    folder=output/'linear'/linear_id(job);record=json.loads((folder/'record.json').read_text())
    if record['job']!=job or record['plan_sha256']!=sha(output/'plan.json'):raise ValueError('Wrong linear binding')
    old=root/PREDECESSOR/'fits'/linear_id(job)
    if record['predecessor_record_sha256']!=sha(old/'record.json'):raise ValueError('Changed original record')
    for name,checksum in record['artifact_sha256'].items():
        if sha(folder/name)!=checksum:raise ValueError('Changed linear artifact')
    table,idx,x=old_panel(root,root/PREDECESSOR,**job);y=table.label.to_numpy(dtype=int)
    keys=[];maximum=0.;probability_max=0.
    for k,C in enumerate(ALL_C):
        item=record['candidates'][k];saved=np.load(folder/f'candidate{k}.npz',allow_pickle=False)
        if item['id']!=k or item['C']!=C or sha(folder/f'candidate{k}.npz')!=item['parameters_sha256']:
            raise ValueError('Wrong candidate ordering/binding')
        scaler=StandardScaler().fit(x[idx['train']]);z=scaler.transform(x)
        np.testing.assert_array_equal(saved['mean'],scaler.mean_);np.testing.assert_array_equal(saved['scale'],scaler.scale_)
        if k<len(NEW_C):
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model=LogisticRegression(C=C,solver='lbfgs',class_weight='balanced',
                    max_iter=4000,tol=1e-6,random_state=42).fit(z[idx['train']],y[idx['train']])
            np.testing.assert_array_equal(saved['coef'],model.coef_)
            np.testing.assert_array_equal(saved['intercept'],model.intercept_)
            np.testing.assert_array_equal(saved['classes'],model.classes_)
        else:
            original=old/f'candidate{k-len(NEW_C)}.npz'
            if sha(original)!=sha(folder/f'candidate{k}.npz'):raise ValueError('Older candidate bytes changed')
        scores={}
        for role,rows in idx.items():
            logits=z[rows]@saved['coef'].T+saved['intercept']
            p=np.column_stack((1-expit(logits[:,0]),expit(logits[:,0]))) if len(saved['classes'])==2 else softmax(logits,axis=1)
            delta=float(np.max(np.abs(p-saved[role])));probability_max=max(probability_max,delta)
            np.testing.assert_allclose(p,saved[role],atol=2e-12,rtol=0)
            scores[role]=measures(y[rows],p)
            for name in scores[role]:
                error=abs(scores[role][name]-item['metrics'][role][name]);maximum=max(maximum,error)
                if error>2e-11:raise ValueError('Independent linear metric mismatch')
            if k==record['selected_id']:csv_check(folder/f'{role}.csv',table,rows,p,2e-12)
        keys.append((sum(scores[r]['balanced_log_loss'] for r in ROLES[1:])/2,
                     -sum(scores[r]['balanced_accuracy'] for r in ROLES[1:])/2,k))
    if min(keys)[2]!=record['selected_id'] or record['selected_C']!=ALL_C[record['selected_id']]:
        raise ValueError('Independent expanded selection mismatch')
    result={'complete':True,'record_sha256':sha(folder/'record.json'),'new_candidate_refits':4,
        'reused_candidates_exact':4,'metric_sets':24,'max_metric_abs':maximum,
        'max_probability_link_abs':probability_max,'created_utc':stamp()}
    atomic(folder/'verification.json',result);return result


def replay(model,data,rows):
    samples=data[rows].reshape(len(rows)*4,data.shape[2],10,200);logits=[];model.eval()
    with sdpa_kernel(SDPBackend.MATH),torch.inference_mode():
        for start in range(0,len(samples),8):
            x=torch.from_numpy(np.array(samples[start:start+8],copy=True)).to('cuda')
            logits.append(model(x).cpu().numpy())
    values=np.concatenate(logits).reshape(len(rows),4,-1)
    average=np.sum(values.astype(np.float64),axis=1)/4
    return softmax(average,axis=1)


def audit_neural(root,output,job):
    folder=output/'neural'/neural_id(job);record=json.loads((folder/'record.json').read_text())
    if record['job']!=job or record['plan_sha256']!=sha(output/'plan.json'):raise ValueError('Wrong neural binding')
    for name,checksum in record['artifact_sha256'].items():
        if sha(folder/name)!=checksum:raise ValueError('Changed neural artifact')
    table,idx,data=source_data(root,job);y=table.label.to_numpy(dtype=int)
    observations,windows,signature=sampling(table,idx['train'])
    if signature!=record['sampling_digest']:raise ValueError('Changed sampling/window stream')
    model=new_model(root,int(y.max()+1),job['pretrained'],job['trainable'])
    if state_digest(model.backbone.state_dict())!=record['encoder_initial_digest'] or \
            state_digest(model.head.state_dict())!=record['head_initial_digest']:
        raise ValueError('Initial encoder/head state mismatch')
    maximum=0.;metric_max=0.;sets=0
    for item in record['candidates']:
        if item['step'] not in STEPS or sha(folder/item['checkpoint'])!=item['checkpoint_sha256']:
            raise ValueError('Wrong neural checkpoint')
        state=torch.load(folder/item['checkpoint'],map_location='cpu',weights_only=True)
        if state_digest(state)!=item['state_digest']:raise ValueError('Saved state digest mismatch')
        model.load_state_dict(state,strict=True)
        encoder=state_digest(model.backbone.state_dict())
        if encoder!=item['encoder_digest'] or state_digest(model.head.state_dict())!=item['head_digest']:
            raise ValueError('Module state mismatch')
        if (not job['trainable'] and encoder!=record['encoder_initial_digest']) or \
                (job['trainable'] and encoder==record['encoder_initial_digest']):
            raise ValueError('Encoder gradient/update permission mismatch')
        for role,rows in idx.items():
            p=replay(model,data,rows);csv=pd.read_csv(folder/f'step{item["step"]}_{role}.csv')
            saved=csv[[f'p{c}' for c in range(p.shape[1])]].to_numpy()
            error=float(np.max(np.abs(p-saved)));maximum=max(maximum,error)
            csv_check(folder/f'step{item["step"]}_{role}.csv',table,rows,p,2e-6)
            calculated=measures(y[rows],p)
            # Validate saved probabilities independently as well: batch variation
            # must not obscure their exact published metric calculations.
            direct=measures(y[rows],saved)
            for name in direct:
                if abs(direct[name]-item['metrics'][role][name])>2e-11:raise ValueError('Saved neural metric mismatch')
                delta=abs(calculated[name]-item['metrics'][role][name]);metric_max=max(metric_max,delta)
                if delta>2e-6:raise ValueError('Neural replay metric mismatch')
            sets+=1
        if state_digest(model.state_dict())!=item['state_digest']:raise ValueError('Inference mutated state')
    result={'complete':True,'record_sha256':sha(folder/'record.json'),'replayed_states':2,
        'metric_sets':sets,'max_probability_abs':maximum,'max_metric_abs':metric_max,'created_utc':stamp()}
    atomic(folder/'verification.json',result);return result


def finish(root,output):
    plan=validate(root,output);linear=[];neural=[];proofs=[];conditions={}
    for job in plan['regularization_jobs']:
        folder=output/'linear'/linear_id(job);record=json.loads((folder/'record.json').read_text())
        proof=json.loads((folder/'verification.json').read_text());proofs.append(proof)
        if not proof['complete'] or proof['record_sha256']!=sha(folder/'record.json'):raise ValueError('Missing/stale linear proof')
        linear.append({'job':job,'selected_C':record['selected_C'],'selected_id':record['selected_id'],'metrics':record['metrics']})
    paired={}
    for job in plan['neural_jobs']:
        folder=output/'neural'/neural_id(job);record=json.loads((folder/'record.json').read_text())
        proof=json.loads((folder/'verification.json').read_text());proofs.append(proof)
        if not proof['complete'] or proof['record_sha256']!=sha(folder/'record.json'):raise ValueError('Missing/stale neural proof')
        key=(job['dataset'],job['group']);paired.setdefault(key,[]).append(record)
        condition=(*key,job['pretrained'],job['trainable']);conditions.setdefault(condition,[])
        for item in record['candidates']:conditions[condition].append({**item,'job':job,'trajectory':neural_id(job)})
    for key,records in paired.items():
        if len({r['sampling_digest'] for r in records})!=1 or len({r['head_initial_digest'] for r in records})!=1:
            raise ValueError('Unmatched sampling/head initialization')
        for pretrained in (True,False):
            if len({r['encoder_initial_digest'] for r in records if r['job']['pretrained']==pretrained})!=1:
                raise ValueError('Unmatched encoder initialization')
    for (dataset,group,pretrained,trainable),candidates in conditions.items():
        keys=[(sum(i['metrics'][r]['balanced_log_loss'] for r in ROLES[1:])/2,
               -sum(i['metrics'][r]['balanced_accuracy'] for r in ROLES[1:])/2,k) for k,i in enumerate(candidates)]
        selected=candidates[min(keys)[2]]
        neural.append({'dataset':dataset,'group':group,'pretrained':pretrained,'trainable':trainable,
            'selected_trajectory':selected['trajectory'],'selected_step':selected['step'],
            'encoder_rate':selected['job']['encoder_rate'],'metrics':selected['metrics'],
            'selection_candidates':[{'trajectory':i['trajectory'],'step':i['step'],'metrics':i['metrics']} for i in candidates]})
    if len(linear)!=24 or len(neural)!=16:raise ValueError('Incomplete selected coverage')
    atomic(output/'summary.json',{'development_only':True,'outer_test_inferences':0,'linear':linear,'neural':neural})
    result={'complete':True,'expanded_linear_heads':24,'new_linear_refits':96,'reused_candidates_exact':96,
        'linear_metric_sets':sum(p.get('metric_sets',0) for p in proofs if 'new_candidate_refits' in p),
        'neural_trajectories':24,'neural_replayed_states':48,'selected_neural_conditions':16,
        'neural_metric_sets':sum(p.get('metric_sets',0) for p in proofs if 'replayed_states' in p),
        'max_linear_metric_abs':max(p['max_metric_abs'] for p in proofs if 'new_candidate_refits' in p),
        'max_neural_probability_abs':max(p['max_probability_abs'] for p in proofs if 'replayed_states' in p),
        'max_neural_metric_abs':max(p['max_metric_abs'] for p in proofs if 'replayed_states' in p),
        'source_pairing_panels_verified':4,'plan_sha256':sha(output/'plan.json'),
        'summary_sha256':sha(output/'summary.json'),'created_utc':stamp()}
    atomic(output/'verification.json',result);return result
