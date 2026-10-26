"""Source-only capacity, cached-feature head optimization and longer fixed schedules."""
from pathlib import Path
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from .cbramod_adaptation import (STUDY as PRIOR, PREDECESSOR, SOURCES as OLD_SOURCES,
    ROLES, HEAD_RATE, REPO, validate as old_validate, source_data, sampling,
    new_model, infer, old_panel, save_prediction, sha, atomic, stamp, state_digest)
from .cbramod_probe import check_cache,check_features
from .data import digest
from .train import seed_everything

STUDY='cbramod_learning_2026-10-09'
HEAD_RATES=(HEAD_RATE,1e-3,1e-2)
STEPS={'head':(0,200,800,2400),'tiny':(0,200,400),'long':(0,200,600,1200)}
SOURCES=tuple(dict.fromkeys((*OLD_SOURCES,'gruxnet/cbramod_learning.py',
    'scripts/cbramod_learning.py','scripts/audit_cbramod_learning.py',
    'tests/test_cbramod_learning.py')))


def jobs(kind):
    common=[{'dataset':d,'group':g,'pretrained':p} for d in ('DEAP','SEEDIV')
            for g in (1,2) for p in (True,False)]
    if kind=='head':return [{**j,'scaled':s,'head_rate':r} for j in common for s in (False,True) for r in HEAD_RATES]
    if kind=='tiny':return [{**j,'target':t} for j in common for t in ('real','permuted')]
    if kind=='long':return [{**j,'trainable':t} for j in common for t in (False,True)]
    raise ValueError('Unknown phase')


def identifier(kind,job):
    name=f'{job["dataset"].lower()}_g{job["group"]}_{"pretrained" if job["pretrained"] else "random42"}'
    if kind=='head':return name+f'_{"standardized" if job["scaled"] else "raw"}_lr{job["head_rate"]:.12g}'
    if kind=='tiny':return name+'_'+job['target']
    return name+'_'+('finetune' if job['trainable'] else 'frozen')


def declare(root):
    old_validate(root,root/PRIOR)
    if not json.loads((root/PRIOR/'verification.json').read_text())['complete']:
        raise ValueError('Verified completed predecessor required')
    for dataset in ('deap','seediv'):
        cache=root/PREDECESSOR/'inputs'/dataset;check_cache(cache);check_features(cache)
    preflight=root/'cbramod_learning_resource_2026-10-09/preflight.json'
    if not preflight.exists() or json.loads(preflight.read_text())['source_sha256']!=sha(Path(__file__)):
        raise ValueError('Matching outcome-free GPU preflight required')
    output=root/STUDY
    if output.exists():raise FileExistsError('Preserve existing learning declaration')
    output.mkdir()
    plan={'created_utc':stamp(),'development_only':True,'outer_test_inferences':0,
        'research_question_change_approved':False,
        'source_sha256':{n:sha(REPO/n) for n in SOURCES},
        'upstream_sha256':{**{f'{PRIOR}/{n}':sha(root/PRIOR/n) for n in ('plan.json','summary.json','verification.json')},
                           preflight.relative_to(root).as_posix():sha(preflight)},
        'jobs':{k:jobs(k) for k in STEPS},'steps':STEPS,'head_rates':HEAD_RATES,
        'scope':'Same native32/62-channel DEAP/SEED-IV source panels, groups1/2, session1/rotation0/fold0 unexposed. Corrected DEAP binary individual valence/existing assigned coarse3 SEED-IV labels. No outer-test inference or new cohort. Existing inputs, preprocessing/calibration/authentication limitations unchanged. No task-outcome exclusions or new label mapping.',
        'head':'48CPUFP64 trajectories:pretrained/random average200D frozen cached embeddings x raw/training-onlyStandardScaler x headLR .001sqrt(6/256)/.001/.01. Linear200->2/3, initseed4242, balanced6training observation draws with replacement, fixed2400AdamW updates, wd.05, smoothing.1, clipglobal1, CosineAnnealingLR2400 eta_min1e-6. No dropout or window variability: one deterministic four-window-averaged feature per original trial. Infer in FP64. Save0/200/800/2400weights/scalers/allsource probabilities. Select16conditions separately perpretraining/scaling across3rates*3positive steps by equal familiar/unseen balancedloss,meanBA,stablejoborder. No global recipe chosen. Objective/dtype/mode differ from neural training; scaling/rate contrasts are within this head-only experiment.',
        'tiny':'16 fulltrain encoder capacity checks: each4sourcepanels x pretrained/random x real/fixedpermuted target. Select12original training observations,6/class DEAP or4/class SEED, only from training using RNG20261012; one fixed10swindow/observation RNG20261013. Seed20261014permutes targets, preserving class counts; never modify original dataset labels. Exactly400AdamW updates, balanced6draws using shared original sampling seed streams but pools follow capacity targets. HeadLR.001, encoderLR1e-4, NOweightdecay/smoothing/dropout, clip1; constantLR/no scheduler. This deliberately easier deterministic capacity condition cannot estimate validation performance or isolate a sole optimizer change. Save0/200/400states/probabilities onlycapacity_train. Success at400requiresBA>=.95 ANDbalancedCE<=.10. Report allfailed controls; no early stop or threshold/steps adapted to outcomes.',
        'long':'16trajectories:4panels x pretrained/random x frozen/trainable. Exactly1200balanced6window updates, same observation/window prefix as prior200run. HeadLR fixed.001 forall; encoderLR1e-4 when trainable. Author-style dropout,wd.05,smoothing.1,clip1,CosineAnnealingLR1200 eta_min1e-6; FP32explicitMATH/noAMP. Save0/200/600/1200states/fullsource train/familiar/unseen probabilities. Sharedheads4242,encoderrandom42,dropout424242; pairedstreams. Select16conditions among200/600/1200 by originalequalpanel balancedloss rule. Rate fixed BEFORE anyhead/tinyfits; all16run irrespective ofdiagnostic outcomes. Old200run differs inheadLR and cosinehorizon, so cannotattribute old/newdifference solely to duration. Withinnewcheckpoints, later steps are the continuation ofoneschedule, notdifferentcosinebudgets.',
        'verification':'Independent source-onlyscalers, numpyFP64head coefficient links, allpublicmetrics/metadata/selections; everyneuralstate strictlyrestored,replayed atbatch8 vs16(long) or3 vs6(tiny), encoder/headstatebindings verified. Initialprobabilities ALSOretainedandchecked; initialpoints neverselected. Tinyselection/permutation/window membership independently reconstructed; allsamplingprefixes/initpairing checked. No independentfulloptimizerupdate replay. Sources/upstreaminputs immutable; exclusiveworkerlock; preservefailures.13oldertests plusnew meaningfulchecks beforedeclaration.',
        'publishing':'Existing explicitauthorapproval covers verified probability tables andanonymousparticipant IDs; original/permuted diagnosticlabels clearlydistinguished viajob/role. Publish JSON/CSVcode/metrics/figures only. NoEEG/embeddings/checkpoints/coefficients orpertrialamplitude data. Declare/pushbeforefits; pushcompletedheads,eachtiny4,long4andcompletiontoexistingorigin/main. Noquestionchange.'}
    atomic(output/'plan.json',plan)
    shutil.copyfile(preflight,output/'resource_preflight.json')
    progress(output,'declared');return output


def validate(root,output,deep_inputs=False):
    old_validate(root,root/PRIOR);plan=json.loads((output/'plan.json').read_text())
    for n,s in plan['source_sha256'].items():
        if sha(REPO/n)!=s:raise ValueError('Frozen learning source changed: '+n)
    for n,s in plan['upstream_sha256'].items():
        if sha(root/n)!=s:raise ValueError('Changed upstream learning evidence: '+n)
    if deep_inputs:
        for dataset in ('deap','seediv'):
            cache=root/PREDECESSOR/'inputs'/dataset;check_cache(cache);check_features(cache)
    return plan


def progress(output,state):
    atomic(output/'progress.json',{'state':state,'updated_utc':stamp(),'outer_test_inferences':0,
        'research_question_change_approved':False,
        **{f'{k}_completed':len(list((output/k).glob('*/verification.json'))) for k in STEPS},
        'head_total':48,'tiny_total':16,'long_total':16})


def head_data(root,job):
    table,idx,x=old_panel(root,root/PREDECESSOR,job['dataset'],job['group'],
                         'pretrained_average' if job['pretrained'] else 'random42_average')
    if x.shape[1]!=200:raise ValueError('Expected averaged200D embeddings')
    x=np.asarray(x,dtype=np.float64)
    scaler=StandardScaler().fit(x[idx['train']]) if job['scaled'] else None
    mean=scaler.mean_ if scaler is not None else np.zeros(x.shape[1])
    scale=scaler.scale_ if scaler is not None else np.ones(x.shape[1])
    return table,idx,(x-mean)/scale,mean,scale


def tiny_definition(table,train,target):
    original=table.label.to_numpy(dtype=int);classes=int(original.max()+1)
    rng=np.random.default_rng(20261012)
    rows=np.concatenate([rng.choice(train[original[train]==c],12//classes,replace=False) for c in range(classes)])
    rows=rng.permutation(rows);windows=np.random.default_rng(20261013).integers(0,4,size=12)
    labels=original[rows].copy()
    if target=='permuted':labels=np.random.default_rng(20261014).permutation(labels)
    elif target!='real':raise ValueError('Unknown capacity target')
    table_tiny=table.iloc[rows].copy().reset_index(drop=True);table_tiny['label']=labels
    definition={'trial_ids':table_tiny.trial_id.tolist(),'windows':windows.tolist(),'target_labels':labels.tolist(),
                'original_task_labels':original[rows].tolist(),'source_only':True}
    return table_tiny,rows,windows,definition


def scores(y,p):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    return {'n':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,p.argmax(1))),
        'balanced_log_loss':float(np.average(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),
                                            weights=compute_sample_weight('balanced',y)))}


def disable_capacity_dropout(model):
    count=0
    for module in model.modules():
        if isinstance(module,nn.Dropout):module.p=0.;count+=1
        elif isinstance(module,nn.MultiheadAttention):module.dropout=0.;count+=1
    return count


def resource_preflight(root):
    output=root/'cbramod_learning_resource_2026-10-09'
    if output.exists():raise FileExistsError('Preserve outcome-free preflight')
    output.mkdir();results=[]
    for channels,classes in ((32,2),(62,3)):
        model=new_model(root,classes,True,True).train();count=disable_capacity_dropout(model)
        x=torch.from_numpy(np.random.default_rng(channels).normal(0,.1,(6,channels,10,200)).astype(np.float32)).to('cuda')
        y=torch.arange(6,device='cuda')%classes
        before=state_digest(model.backbone.state_dict());head=state_digest(model.head.state_dict())
        optimizer=torch.optim.AdamW([{'params':model.head.parameters(),'lr':.001},
                                    {'params':model.backbone.parameters(),'lr':1e-4}],weight_decay=0.)
        seed_everything(424242);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            torch.testing.assert_close(model(x),model(x),rtol=0,atol=0)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x),y)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if not torch.isfinite(loss) or before==state_digest(model.backbone.state_dict()) or head==state_digest(model.head.state_dict()):
            raise ValueError('Synthetic full-gradient preflight failed')
        results.append({'channels':channels,'classes':classes,'disabled_dropout_modules':count,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'synthetic_loss':float(loss.item()),'deterministic_capacity_forward':True})
        del model,optimizer,x,y
    result={'synthetic_only':True,'task_fits':0,'created_utc':stamp(),'source_sha256':sha(Path(__file__)),
            'torch':torch.__version__,'device':torch.cuda.get_device_name(),'records':results}
    atomic(output/'preflight.json',result);return result


def exposure_records(train,draws,windows,steps,kind):
    position={int(v):k for k,v in enumerate(train)};records=[]
    for step in steps:
        if step==0:continue
        observed=draws[:step].ravel();counts=np.bincount([position[int(v)] for v in observed],minlength=len(train))
        record={'step':step,'training_observations':len(train),'draws':len(observed),
            'unique_observations_seen':int(np.count_nonzero(counts)),
            'minimum_draws_per_observation':int(counts.min()),'maximum_draws_per_observation':int(counts.max())}
        if kind=='long':record.update({'unique_observation_window_pairs':len(set(zip(observed,windows[:step].ravel()))),
                                       'available_observation_window_pairs':4*len(train)})
        records.append(record)
    return records


def grad_l2(parameters):
    values=[torch.sum(p.grad.detach().float()**2) for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.


def fit_head(root,output,job):
    folder=output/'head'/identifier('head',job)
    if (folder/'verification.json').exists():return
    if folder.exists():raise FileExistsError('Preserve partial head fit')
    folder.mkdir(parents=True)
    table,idx,x,mean,scale=head_data(root,job);classes=int(table.label.max()+1)
    torch.manual_seed(4242);model=nn.Linear(200,classes).double();initial=state_digest(model.state_dict())
    total=STEPS['head'][-1];samples,windows,signature=sampling(table,idx['train'],updates=total)
    xx=torch.tensor(x,dtype=torch.float64);yy=torch.tensor(table.label.to_numpy(dtype=np.int64))
    optimizer=torch.optim.AdamW(model.parameters(),lr=job['head_rate'],weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total,eta_min=1e-6)
    seed_everything(424242);torch.set_num_threads(1)
    history=[];losses=[];candidates=[];start=time.perf_counter()
    def snapshot(step):
        with torch.inference_mode():p=torch.softmax(model(xx),dim=1).numpy()
        name=f'checkpoint{step}.npz'
        np.savez(folder/name,weight=model.weight.detach().numpy(),bias=model.bias.detach().numpy(),mean=mean,scale=scale)
        measures={role:scores(table.iloc[rows].label.to_numpy(),p[rows]) for role,rows in idx.items()}
        for role,rows in idx.items():save_prediction(folder,role,table,rows,p[rows],prefix=f'step{step}_')
        return {'step':step,'metrics':measures,'checkpoint':name,'checkpoint_sha256':sha(folder/name),
                'state_digest':state_digest(model.state_dict())}
    candidates.append(snapshot(0))
    for step in range(1,total+1):
        optimizer.zero_grad(set_to_none=True);rows=samples[step-1]
        loss=nn.functional.cross_entropy(model(xx[rows]),yy[rows],label_smoothing=.1)
        if not torch.isfinite(loss):raise ValueError('Nonfinite head loss')
        loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();losses.append(float(loss.item()))
        if step%200==0:history.append({'step':step,'mean_last200_smoothed_minibatch_loss':float(np.mean(losses[-200:])),
            'gradient_L2_before_clip':float(norm),'head_lr':optimizer.param_groups[0]['lr']})
        if step in STEPS['head']:candidates.append(snapshot(step))
    atomic(folder/'history.json',history)
    seal_record(output,folder,job,{'initial_head_digest':initial,'sampling_digest':signature,
        'candidates':candidates,'seconds':time.perf_counter()-start,'dtype':'FP64CPU',
        'exposure':exposure_records(idx['train'],samples,windows,STEPS['head'],'head')})


def tiny_infer(model,data,batch=6):
    model.eval();chunks=[]
    with sdpa_kernel(SDPBackend.MATH),torch.inference_mode():
        for start in range(0,len(data),batch):chunks.append(model(torch.tensor(data[start:start+batch],device='cuda')).cpu().numpy())
    return softmax(np.concatenate(chunks).astype(np.float64),axis=1)


def neural_snapshot(model,data,table,idx,folder,step,kind):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    name=f'checkpoint{step}.pt';torch.save(state,folder/name)
    p={r:(tiny_infer(model,data[rows]) if kind=='tiny' else infer(model,data,rows)) for r,rows in idx.items()}
    measures={r:scores(table.iloc[idx[r]].label.to_numpy(),v) for r,v in p.items()}
    for r,rows in idx.items():save_prediction(folder,r,table,rows,p[r],prefix=f'step{step}_')
    return {'step':step,'checkpoint':name,'checkpoint_sha256':sha(folder/name),'state_digest':state_digest(state),
        'encoder_digest':state_digest(model.backbone.state_dict()),'head_digest':state_digest(model.head.state_dict()),'metrics':measures}


def fit_neural(root,output,kind,job):
    folder=output/kind/identifier(kind,job)
    if (folder/'verification.json').exists():return
    if folder.exists():raise FileExistsError('Preserve partial neural learning fit')
    folder.mkdir(parents=True)
    table,idx,data=source_data(root,job);definition=None
    if kind=='tiny':
        table,rows,windows,definition=tiny_definition(table,idx['train'],job['target'])
        data=data[rows,windows];idx={'capacity_train':np.arange(12)};train=idx['capacity_train']
    else:train=idx['train']
    model=new_model(root,int(table.label.max()+1),job['pretrained'],kind=='tiny' or job['trainable'])
    dropout_count=disable_capacity_dropout(model) if kind=='tiny' else 0
    initial_encoder=state_digest(model.backbone.state_dict());initial_head=state_digest(model.head.state_dict())
    final_step=STEPS[kind][-1];draws,windows,signature=sampling(table,train,updates=final_step)
    groups=[{'params':model.head.parameters(),'lr':1e-3}]
    if kind=='tiny' or job['trainable']:groups.append({'params':model.backbone.parameters(),'lr':1e-4})
    optimizer=torch.optim.AdamW(groups,weight_decay=0. if kind=='tiny' else .05)
    scheduler=None if kind=='tiny' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1200,eta_min=1e-6)
    seed_everything(424242);history=[];losses=[];candidates=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    candidates.append(neural_snapshot(model,data,table,idx,folder,0,kind))
    for step in range(1,final_step+1):
        model.train();optimizer.zero_grad(set_to_none=True);rows=draws[step-1]
        values=data[rows] if kind=='tiny' else data[rows,windows[step-1]]
        labels=torch.tensor(table.label.to_numpy(dtype=np.int64)[rows],device='cuda')
        with sdpa_kernel(SDPBackend.MATH):
            loss=nn.functional.cross_entropy(model(torch.tensor(values,device='cuda')),labels,
                                            label_smoothing=0. if kind=='tiny' else .1)
            if not torch.isfinite(loss):raise ValueError('Nonfinite neural loss')
            loss.backward()
        if step%100==0:
            head_norm=grad_l2(model.head.parameters());encoder_norm=grad_l2(model.backbone.parameters())
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:scheduler.step()
        losses.append(float(loss.item()))
        if step%100==0:history.append({'step':step,'mean_last100_minibatch_loss':float(np.mean(losses[-100:])),
            'gradient_L2_before_clip':float(norm),'head_lr':optimizer.param_groups[0]['lr'],
            'head_gradient_L2_before_clip':head_norm,'encoder_gradient_L2_before_clip':encoder_norm,
            'encoder_lr':optimizer.param_groups[-1]['lr'] if len(groups)>1 else 0.})
        if step in STEPS[kind]:
            cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
            candidates.append(neural_snapshot(model,data,table,idx,folder,step,kind))
            torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu)
    atomic(folder/'history.json',history)
    extra={'initial_encoder_digest':initial_encoder,'initial_head_digest':initial_head,'sampling_digest':signature,
        'tiny_definition':definition,'disabled_dropout_modules':dropout_count,'candidates':candidates,
        'exposure':exposure_records(train,draws,windows,STEPS[kind],kind),
        'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    if kind=='tiny':
        metric=candidates[-1]['metrics']['capacity_train']
        extra['capacity_pass']=metric['balanced_accuracy']>=.95 and metric['balanced_log_loss']<=.10
    seal_record(output,folder,job,extra)


def seal_record(output,folder,job,extra):
    atomic(folder/'record.json',{'job':job,'plan_sha256':sha(output/'plan.json'),**extra,
        'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.suffix in('.pt','.npz','.csv','.json')}})


def export(output):
    destination=REPO/'results/development'/STUDY;files=[]
    for path in output.rglob('*'):
        if not path.is_file() or path.suffix not in('.json','.csv','.md'):continue
        relative=path.relative_to(output)
        if relative.parts[0] in STEPS and not(path.parent/'verification.json').exists():continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        if sha(target)!=sha(path):raise ValueError('Publication copy mismatch')
        files.append({'file':relative.as_posix(),'sha256':sha(target)})
    atomic(destination/'export_manifest.json',{'files':files,
        'scope':'Author-approved verified trial probabilities, anonymous IDs and clearly marked capacity-target labels; noEEG/embeddings/coefficients/weights/amplitude values.'})
    return destination
