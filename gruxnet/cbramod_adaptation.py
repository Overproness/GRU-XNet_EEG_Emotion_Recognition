"""Bounded source-only head regularization and matched encoder adaptation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from .cbramod_probe import (STUDY as PREDECESSOR, PARENT, ASSETS, SOURCES as OLD_SOURCES,
    MODELS as LINEAR_MODELS, C_VALUES as OLD_C, ROLES, REPO, local_model,
    load_panel as old_panel, fit_estimator, check_cache, check_features, sha, atomic,
    stamp, metric, selection_key, validate as old_validate)
from .data import digest
from .material_controls import state_digest
from .train import seed_everything

STUDY = 'cbramod_adaptation_2026-10-09'
NEW_C = (1e-6, 1e-5, 1e-4, 1e-3)
ALL_C = (*NEW_C, *OLD_C)
STEPS = (50, 200)
RATES = (1e-4, 3e-5)
BATCH = 6
INFERENCE_BATCH = 16
HEAD_RATE = .001*(BATCH/256)**.5
SOURCES = tuple(dict.fromkeys((*OLD_SOURCES, 'gruxnet/cbramod_adaptation.py',
    'scripts/cbramod_adaptation.py', 'scripts/audit_cbramod_adaptation.py',
    'tests/test_cbramod_adaptation.py')))


class Adaptation(nn.Module):
    def __init__(self, backbone, classes, trainable):
        super().__init__()
        self.backbone = backbone.requires_grad_(trainable)
        self.trainable_encoder = trainable
        torch.manual_seed(4242)
        self.head = nn.Linear(200, classes)

    def forward(self, x):
        # Frozen and trainable encoders use the same train/eval modes, including
        # dropout. Only the encoder's gradient/update permission differs.
        if self.trainable_encoder:
            tokens = self.backbone(x)
        else:
            with torch.no_grad(): tokens = self.backbone(x)
        return self.head(tokens.mean((1, 2)))


def new_model(root, classes, pretrained, trainable):
    return Adaptation(local_model(root/ASSETS, pretrained), classes, trainable).to('cuda')


def optimizer_for(model, rate):
    groups = [{'params': model.head.parameters(), 'lr': HEAD_RATE}]
    if model.trainable_encoder: groups.append({'params': model.backbone.parameters(), 'lr': rate})
    return torch.optim.AdamW(groups, weight_decay=.05)


def resource_pilot(root):
    output = root/'cbramod_adaptation_resource_2026-10-09'
    if output.exists(): raise FileExistsError('Preserve resource pilot')
    output.mkdir()
    records = []
    for channels in (32, 62):
        model = new_model(root, 3, True, True)
        x = torch.tensor(np.random.default_rng(channels).normal(0, .1, (BATCH, channels, 10, 200)).astype(np.float32), device='cuda')
        y = torch.arange(BATCH, device='cuda') % 3
        optimizer = optimizer_for(model, RATES[0]); seed_everything(424242)
        torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            for step in range(3):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x), y, label_smoothing=.1)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                inference = model(torch.zeros((INFERENCE_BATCH, channels, 10, 200), device='cuda'))
        if not torch.isfinite(loss) or not torch.isfinite(inference).all(): raise ValueError('Nonfinite synthetic pilot')
        records.append({'channels': channels, 'training_batch': BATCH, 'inference_batch': INFERENCE_BATCH,
            'three_updates_seconds': time.perf_counter()-start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'synthetic_loss': float(loss.item())})
        del model, optimizer, x, y
    result = {'created_utc': stamp(), 'synthetic_only': True, 'records': records,
        'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
        'attention_backend': 'Explicit deterministic PyTorch MATH for forward/backward/replay',
        'task_fits': 0, 'source_sha256': sha(Path(__file__))}
    atomic(output/'feasibility.json', result)
    return result


def neural_jobs():
    return [dict(dataset=d, group=g, pretrained=p, trainable=t, encoder_rate=rate)
        for d in ('DEAP', 'SEEDIV') for g in (1, 2) for p in (True, False)
        for t in (False, True) for rate in (RATES if t else (0.,))]


def neural_id(job):
    return f'{job["dataset"].lower()}_g{job["group"]}_{"pretrained" if job["pretrained"] else "random42"}_{"finetune" if job["trainable"] else "frozen"}_lr{job["encoder_rate"]:g}'


def declare(root):
    output = root/STUDY
    if output.exists(): raise FileExistsError('Preserve adaptation declaration')
    old_validate(root, root/PREDECESSOR)
    pilot = root/'cbramod_adaptation_resource_2026-10-09/feasibility.json'
    if not pilot.exists(): raise ValueError('Synthetic backpropagation pilot required')
    output.mkdir()
    bindings = {f'{PREDECESSOR}/{n}': sha(root/PREDECESSOR/n) for n in ('plan.json', 'verification.json', 'summary.json')}
    bindings[pilot.relative_to(root).as_posix()] = sha(pilot)
    for dataset in ('deap', 'seediv'):
        cache = root/PREDECESSOR/'inputs'/dataset
        check_cache(cache); check_features(cache)
        for name in ('prepared.json', 'features.json', 'trials.csv'):
            bindings[(cache/name).relative_to(root).as_posix()] = sha(cache/name)
        for group in (1, 2):
            for name in ('trials.csv', 'panel.json'):
                p = root/PARENT/'panels'/f'{dataset}_g{group}'/name
                bindings[p.relative_to(root).as_posix()] = sha(p)
        for fit in sorted((root/PREDECESSOR/'fits').glob(f'{dataset}_*')):
            bindings[(fit/'record.json').relative_to(root).as_posix()] = sha(fit/'record.json')
            bindings[(fit/'verification.json').relative_to(root).as_posix()] = sha(fit/'verification.json')
    plan = {'created_utc': stamp(), 'development_only': True, 'research_question_change_approved': False,
        'source_sha256': {n: sha(REPO/n) for n in SOURCES}, 'upstream_sha256': bindings,
        'regularization_jobs': [dict(dataset=d, group=g, model=m) for d in ('DEAP','SEEDIV') for g in (1,2) for m in LINEAR_MODELS],
        'neural_jobs': neural_jobs(), 'C_values': ALL_C, 'new_C_values': NEW_C,
        'neural_checkpoint_updates': STEPS, 'neural_head_lr': HEAD_RATE,
        'counts': {'expanded_linear_heads': 24, 'new_linear_candidates': 96, 'new_independent_refits': 96,
                   'reused_verified_candidates': 96, 'neural_trajectories': 24, 'neural_candidates': 48, 'selected_neural_conditions': 16},
        'question': 'Does stronger head regularization resolve frozen high-dimensional overfitting, and does source-only adaptation improve pretrained/random averaged representations under matched learning exposure? This is an experiment, not an adopted new main research question.',
        'scope': 'Same4source panels as verified predecessor, session1/rotation0/fold0 unexposed groups1/2. No new preprocessing, subjects, labels, outer-test inference or target-population adaptation. One random encoder/head initialization; source panels reused and used for selection.',
        'head_regularization': 'Add C1e-6/1e-5/1e-4/1e-3 for ALL6frozen/spectral representations. Reuse4older C candidates byte-for-byte, checking source/scaler/coefficient and file bindings. Same training-only StandardScaler,balancedlbfgs,maxiter4000,tol1e-6. Choose among8Cs by equal familiar/unseen balancedlogloss,thenmeanBA,stable declared firsttie. Keepeverycandidate/model/panel;no global winning model.',
        'neural_model': 'Exact pinned official CBraMod encoder with author-supported token-average -> Linear200->2/3 head; no feature scaler or novel layers. Pair pretrained/random42 and frozen/trainable encoder, identical independentheadseed4242. Frozen and trainable both use encoder train-mode dropout during training, eval mode at prediction. Explicit PyTorchMATHattention/noAMP/no augmentation; changes include parameter updates/optimizer state, not input mode or architecture.',
        'input': 'Exactly predecessor native200Hz40s arrays, units/100,4disjoint10s windows. One deterministic uniformly drawn10swindow per sampled training observation; all4windows at inference. Probability=softmax(mean4windowlogits), not mean probabilities or windowaccuracy. Same observation/window draw sequence across allpairedconditions and rates. No trial or participant filtering based on results.',
        'training': 'Exactly200AdamWupdates,balanced6observation draws (3/classDEAP,2/classSEED) with replacement;fixed perpanel NumPystream20261009 and independently seeded window stream20261010; Torchdropoutseed424242. wd.05,clipglobalgradL2at1,label_smoothing.1. Fine encoder rates1e-4/3e-5;frozen0. HeadLR=.001sqrt(6/256)=author multi_lr formula, shared across allconditions. CosineAnnealingLR T_max200 eta_min1e-6 aftereachupdate. This is a finite small budget with partial stochastic trial exposure, not convergence or exact author epoch/split reproduction. Global clipping includes encoder gradients when trainable.',
        'selection_and_learning': 'Save initial training/validation scores; candidate checkpoints50/200with fullsource train/familiar/unseen probabilities. For each of16dataset/group/pretraining/trainability conditions choose bothcheckpoints and,forfine,bothrates using equalpanelbalancedloss,thenmeanBA,stablefirst. Initialstate is diagnostic only,notselectable. Keepallcandidates/finalstates and50update minibatchloss summaries. No outcome-dependent extension or threshold tuning.',
        'verification': 'All96new heads independentlyrefit;all96oldnpz/record bindingschecked. All48neural states restored strictly,encoder/headdigests and fullsource probabilities replayed with inferencebatch8 ratherthan16; independent weightedmetrics,participant/trial/material exclusions,all16selections andpairingchecked. Frozen encoder digests must stay unchanged,fullfine must change;sameencoder init perpretraining and samehead/observation/window/draw prefixes across paired conditions. No fulloptimizer-trajectory reproduction or first-party data/pretraining corpus authentication.',
        'publishing': 'Author explicitly approved verified trial-level probabilities and anonymous participant IDs on9Oct2026. Publish checked JSON/CSVderivedprobabilities/labels/anonymous references and code;excludeEEG,embeddings,coefficients/checkpoints and pertrialamplitude data. Commit verified regularization phase,each8newneuraltrajectories andcompletion to existingorigin/main;no forcepush.',
        'decision': 'Report allgroups/conditions andloss-primary comparisons;small reused panels permit learning diagnosis only. Stronger regularization or existingencoder finetuning is notnovelty. No fullheldoutgrid or questionchange is adopted without evidence/proposal;actualpaperpivot needs explicitauthorapproval andfresharchive.'}
    atomic(output/'plan.json', plan)
    atomic(output/'progress.json', {'state':'declared','linear_heads_completed':0,'linear_heads_total':24,
        'neural_trajectories_completed':0,'neural_trajectories_total':24,'updated_utc':stamp(),
        'research_question_change_approved':False,'outer_test_inferences':0})
    return output


def validate(root, output):
    plan = json.loads((output/'plan.json').read_text())
    old_validate(root, root/PREDECESSOR)
    for name, checksum in plan['source_sha256'].items():
        if sha(REPO/name) != checksum: raise ValueError(f'Changed adaptation source {name}')
    for name, checksum in plan['upstream_sha256'].items():
        if sha(root/name) != checksum: raise ValueError(f'Changed adaptation predecessor {name}')
    return plan


def linear_id(job): return f'{job["dataset"].lower()}_g{job["group"]}_{job["model"]}'


def fit_linear(root, output, job):
    folder = output/'linear'/linear_id(job)
    if (folder/'verification.json').exists(): return
    if folder.exists(): raise FileExistsError('Preserve partial expanded head')
    folder.mkdir(parents=True)
    table, idx, x = old_panel(root, root/PREDECESSOR, **job)
    old = root/PREDECESSOR/'fits'/linear_id(job)
    original = json.loads((old/'record.json').read_text())
    candidates=[]
    for k, C in enumerate(ALL_C):
        if k < len(NEW_C):
            scaler, model, p, measures = fit_estimator(x,table,idx,C)
            np.savez(folder/f'candidate{k}.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,
                intercept=model.intercept_,classes=model.classes_,**p)
            candidate={'id':k,'C':C,'metrics':measures,'iterations':model.n_iter_.tolist(),'reused':False}
        else:
            j=k-len(NEW_C); source=old/f'candidate{j}.npz'
            if sha(source)!=original['candidates'][j]['parameters_sha256']: raise ValueError('Changed old candidate')
            import shutil
            shutil.copyfile(source,folder/f'candidate{k}.npz')
            candidate={**original['candidates'][j],'id':k,'reused':True,'original_id':j}
        candidate['parameters_sha256']=sha(folder/f'candidate{k}.npz');candidates.append(candidate)
    selected=min(candidates,key=selection_key)
    pack=np.load(folder/f'candidate{selected["id"]}.npz',allow_pickle=False)
    for role in ROLES: save_prediction(folder,role,table,idx[role],pack[role])
    atomic(folder/'record.json',{'job':job,'plan_sha256':sha(output/'plan.json'),
        'predecessor_record_sha256':sha(old/'record.json'),'candidates':candidates,
        'selected_id':selected['id'],'selected_C':selected['C'],'metrics':selected['metrics'],
        'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.suffix in('.npz','.csv')}})


def save_prediction(folder, role, table, rows, probabilities, prefix=''):
    frame=table.iloc[rows][['trial_id','subject_id','material_key','original_label','label']].copy()
    frame['role']=role
    for c in range(probabilities.shape[1]):frame[f'p{c}']=probabilities[:,c]
    frame.to_csv(folder/f'{prefix}{role}.csv',index=False)


def source_data(root, job):
    table,idx,_=old_panel(root,root/PREDECESSOR,job['dataset'],job['group'],'band_absolute')
    cache=root/PREDECESSOR/'inputs'/job['dataset'].lower()
    union=pd.read_csv(cache/'trials.csv');positions=dict(zip(union.trial_id,union.index))
    raw=np.load(cache/'native200.npy',mmap_mode='r',allow_pickle=False)[[positions[t] for t in table.trial_id]]
    n,c,_=raw.shape
    data=(raw/np.float32(100.)).reshape(n,c,4,10,200).transpose(0,2,1,3,4).copy()
    return table,idx,data


def sampling(table, train, updates=200):
    y=table.label.to_numpy(dtype=int);classes=int(y.max()+1)
    if BATCH%classes:raise ValueError('Unbalanced declared batch')
    rng=np.random.default_rng(20261009);window_rng=np.random.default_rng(20261010)
    pools=[train[y[train]==c] for c in range(classes)]
    observations=np.array([rng.permutation(np.concatenate([rng.choice(pool,BATCH//classes,replace=True) for pool in pools])) for _ in range(updates)])
    windows=window_rng.integers(0,4,size=observations.shape)
    return observations,windows,digest({'observations':observations.tolist(),'windows':windows.tolist()})


def infer(model, data, rows, batch=INFERENCE_BATCH):
    c=data.shape[2];samples=data[rows].reshape(len(rows)*4,c,10,200)
    chunks=[];model.eval()
    with sdpa_kernel(SDPBackend.MATH),torch.inference_mode():
        for start in range(0,len(samples),batch):
            logits=model(torch.tensor(samples[start:start+batch],device='cuda')).cpu().numpy()
            if not np.isfinite(logits).all():raise ValueError('Nonfinite logits')
            chunks.append(logits)
    logits=np.concatenate(chunks).reshape(len(rows),4,-1).mean(1,dtype=np.float64)
    from scipy.special import softmax
    return softmax(logits,axis=1)


def candidate_state(model, data, table, idx, folder, step):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    name=f'checkpoint{step}.pt';torch.save(state,folder/name)
    probabilities={r:infer(model,data,rows) for r,rows in idx.items()}
    measures={r:metric(table.iloc[idx[r]].label.to_numpy(),p) for r,p in probabilities.items()}
    for role in ROLES:save_prediction(folder,role,table,idx[role],probabilities[role],prefix=f'step{step}_')
    return {'id':step,'step':step,'metrics':measures,'checkpoint':name,'checkpoint_sha256':sha(folder/name),
        'state_digest':state_digest(state),'encoder_digest':state_digest(model.backbone.state_dict()),
        'head_digest':state_digest(model.head.state_dict())}


def fit_neural(root, output, job):
    folder=output/'neural'/neural_id(job)
    if (folder/'verification.json').exists():return
    if folder.exists():raise FileExistsError('Preserve partial neural trajectory')
    folder.mkdir(parents=True)
    table,idx,data=source_data(root,job);classes=int(table.label.max()+1)
    model=new_model(root,classes,job['pretrained'],job['trainable'])
    encoder_initial=state_digest(model.backbone.state_dict());head_initial=state_digest(model.head.state_dict())
    sampled,windows,signature=sampling(table,idx['train'])
    initial={role:metric(table.iloc[rows].label.to_numpy(),infer(model,data,rows)) for role,rows in idx.items()}
    optimizer=optimizer_for(model,job['encoder_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200,eta_min=1e-6)
    seed_everything(424242);history=[];losses=[];candidates=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,201):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=torch.tensor(data[sampled[step-1],windows[step-1]],device='cuda')
        labels=torch.tensor(table.label.to_numpy(dtype=np.int64)[sampled[step-1]],device='cuda')
        with sdpa_kernel(SDPBackend.MATH):
            loss=nn.functional.cross_entropy(model(batch),labels,label_smoothing=.1)
            if not torch.isfinite(loss):raise ValueError('Nonfinite neural loss')
            loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();losses.append(float(loss.item()))
        if step%50==0:
            history.append({'step':step,'last50_mean_smoothed_minibatch_loss':float(np.mean(losses[-50:])),
                'last_gradient_L2_before_clip':float(norm.item()),'head_lr':optimizer.param_groups[0]['lr'],
                'encoder_lr':optimizer.param_groups[-1]['lr'] if job['trainable'] else 0.})
        if step in STEPS:
            # Evaluation consumes no random draws; nonetheless seal/restore RNG
            # to ensure checkpoint evaluation cannot perturb continuation.
            cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
            candidates.append(candidate_state(model,data,table,idx,folder,step))
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
    final_encoder=state_digest(model.backbone.state_dict())
    if (not job['trainable'] and final_encoder!=encoder_initial) or (job['trainable'] and final_encoder==encoder_initial):
        raise ValueError('Encoder trainability invariant failed')
    atomic(folder/'history.json',history)
    atomic(folder/'record.json',{'job':job,'plan_sha256':sha(output/'plan.json'),
        'encoder_initial_digest':encoder_initial,'head_initial_digest':head_initial,
        'sampling_digest':signature,'initial_metrics':initial,'candidates':candidates,
        'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'artifact_sha256':{p.name:sha(p) for p in folder.iterdir() if p.suffix in('.pt','.csv','.json')}})


def export(output):
    import shutil
    destination=REPO/'results/development'/STUDY;records=[]
    for path in output.rglob('*'):
        if not path.is_file() or path.suffix not in('.json','.csv','.md'):continue
        relative=path.relative_to(output)
        if relative.parts[0] in('linear','neural') and not(path.parent/'verification.json').exists():continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        if sha(path)!=sha(target):raise ValueError('Export byte mismatch')
        records.append({'file':relative.as_posix(),'sha256':sha(target)})
    atomic(destination/'export_manifest.json',{'files':records,
        'scope':'Author-approved verified derived probability tables and anonymous IDs; code/declarations/metrics. RawEEG/embeddings/fittedcoefficients/checkpoints excluded.'})
    return destination
