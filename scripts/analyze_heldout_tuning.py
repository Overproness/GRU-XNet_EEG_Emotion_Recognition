"""Complete-grid analysis only; no interim aggregate model ranking."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.data import digest, sha256
from gruxnet.heldout_tuning import STUDY, MODELS, ARMS, INITIALIZATIONS, case_id, context_id, atomic_json
from gruxnet.grouped_material_controls import weights, check_coverage
from gruxnet.full_context_controls_v2 import statistic
from scripts.audit_heldout_tuning import numerical_metrics
from scripts.within_video_alignment import pairs as old_pairs, scores, aggregate


def repeated_pairs(rows):
    recipient=[]; donor=[]; coefficient=[]; eligible=[]
    for _,part in rows.groupby(['group','seed'],sort=True):
        indexes = part.index.to_numpy()
        r,d,b,e = old_pairs(part.reset_index(drop=True))
        recipient.extend(indexes[r]); donor.extend(indexes[d]); coefficient.extend(b); eligible.extend(indexes[e])
    return np.array(recipient,dtype=int),np.array(donor,dtype=int),np.array(coefficient),np.array(sorted(eligible),dtype=int)


def collect(output,dataset):
    plan=json.loads((output/'plan.json').read_text()); parts={}
    for job in plan['jobs']:
        if job['dataset'] != dataset: continue
        path=output/'fits'/case_id(job)/'predictions_test.csv'
        parts.setdefault((job['model'],job['arm']),[]).append(pd.read_csv(path))
    for job in plan['contexts']:
        if job['dataset'] != dataset: continue
        for model in ('prior','context_logistic'):
            rows=pd.read_csv(output/'contexts'/context_id(job)/f'predictions_{model}_test.csv')
            for initialization in INITIALIZATIONS:
                copy=rows.copy(); copy['seed']=initialization
                parts.setdefault((model,job['arm']),[]).append(copy)
    frames={}
    ordering=['group','seed','source_session','material_rotation','fold','trial_id']
    for key,items in parts.items():
        rows=pd.concat(items,ignore_index=True).sort_values(ordering).reset_index(drop=True)
        check_coverage(rows,rows.drop_duplicates('trial_id'))
        frames[key]=rows
        rows.to_csv(output/f'predictions_{dataset.lower()}_{key[0]}_{key[1]}.csv',index=False)
    metadata=['trial_id','subject_id','material_key','label','original_label',*ordering[:-1]]
    reference=frames[(MODELS[0],ARMS[0])][metadata]
    for rows in frames.values(): pd.testing.assert_frame_equal(reference,rows[metadata],check_exact=True)
    return frames


def analyze(output,dataset):
    verification=json.loads((output/'verification.json').read_text())
    if not verification['passed'] or not verification['complete']: raise ValueError('Complete case verification required')
    frames=collect(output,dataset)
    reference=next(iter(frames.values())); subjects,materials,sw,mw=weights(reference,dataset)
    fixed=np.ones_like(mw); tasks=('coarse3','binary') if dataset=='SEEDIV' else ('binary',)
    result={'dataset':dataset,'development_only':True,'research_question_change_approved':False,
            'verification_sha256':sha256(output/'verification.json'),'models':{},'contrasts':[],
            'bootstrap':{'draws':10000,'seed':20261006,'person_weight_digest':digest(sw.tolist()),'video_weight_digest':digest(mw.tolist()),
                         'scope':'Paired fixed fitted-model percentile intervals; correctness and losses averaged within observed person/video across all four grouping/initialization combinations. Reused cohorts, unadjusted exploratory intervals; no probability ensemble or population/new-cohort confirmation.'},
            'input_sha256':{f'predictions_{dataset.lower()}_{model}_{arm}.csv':sha256(output/f'predictions_{dataset.lower()}_{model}_{arm}.csv') for model,arm in frames}}
    model_names=(*MODELS,'prior','context_logistic')
    comparisons=[(m,'unexposed',m,'exposed') for m in model_names]
    comparisons += [(a,arm,b,arm) for arm in ARMS for a,b in (('eegnet','gru'),('eegnet_context','eegnet'),('eegnet_context','prior'),('eegnet_context','context_logistic'))]
    scopes=[('combined',None,None),*[(f'group{g}_init{i}',g,i) for g in (1,2) for i in INITIALIZATIONS]]
    for scope,group,initialization in scopes:
        distributions={}; reports={}
        for (model,arm),all_rows in frames.items():
            rows=all_rows if group is None else all_rows[all_rows.group.eq(group)&all_rows.seed.eq(initialization)]
            reports.setdefault(model,{})[arm]={}; distributions[(model,arm)]={}
            for task in tasks:
                distributions[(model,arm)][task]={}
                for which in ('BA','logloss'):
                    point,crossed=statistic(rows,dataset,task,subjects,materials,sw,mw,which)
                    _,participant=statistic(rows,dataset,task,subjects,materials,sw,fixed,which)
                    reports[model][arm].setdefault(task,{})[which]={'point':point,'crossed_percentile_95':np.quantile(crossed,[.025,.975]).tolist(),'participant_percentile_95':np.quantile(participant,[.025,.975]).tolist()}
                    distributions[(model,arm)][task][which]=(point,crossed,participant)
            if group is not None: reports[model][arm]['metrics']=numerical_metrics(rows,dataset)
        result['models'][scope]=reports
        for a,aa,b,ba in comparisons:
            for task in tasks:
                for which in ('BA','logloss'):
                    pa,ca,sa=distributions[(a,aa)][task][which]; pb,cb,sb=distributions[(b,ba)][task][which]
                    result['contrasts'].append({'scope':scope,'model_a':a,'arm_a':aa,'model_b':b,'arm_b':ba,'task':task,'metric':which,
                        'difference':pa-pb,'crossed_percentile_95':np.quantile(ca-cb,[.025,.975]).tolist(),'participant_percentile_95':np.quantile(sa-sb,[.025,.975]).tolist()})
    atomic_json(output/f'comparison_{dataset.lower()}.json',result)
    # EEG/rating pairing within identical selected model, grouping, initialization and video.
    r,d,base,eligible=repeated_pairs(reference)
    ri=reference.iloc[r].subject_id.map({s:i for i,s in enumerate(subjects)}).to_numpy()
    di=reference.iloc[d].subject_id.map({s:i for i,s in enumerate(subjects)}).to_numpy()
    vi=reference.iloc[r].material_key.map({s:i for i,s in enumerate(materials)}).to_numpy()
    alignment={'dataset':dataset,'development_only':True,'research_question_change_approved':False,'eligible_rows':len(eligible),'total_rows_per_model':len(reference),
               'excluded_trials':reference.drop(eligible)[['group','seed','trial_id']].to_dict('records'),'dyadic_pairs':len(r),'models':{},'contrasts':[],
               'scope':'Average scores over every other held-out person watching the same video in the same selected cell/group/init. Recipient/donor/video paired bootstrap weights. Conditional association, not a physiological or causal mechanism.',
               'bootstrap':result['bootstrap'],'input_sha256':result['input_sha256']}
    distributions={}
    for (model,arm),rows in frames.items():
        alignment['models'].setdefault(model,{})[arm]={}
        for task in tasks:
            target,included,values,classes=scores(rows,task,dataset,r,d)
            item={}; distributions[(model,arm,task)]={}
            for which in ('BA','logloss'):
                difference=values['aligned'][which]-values['exchanged'][which]
                point,draws,valid=aggregate(difference,target,included,classes,base,ri,di,vi,sw,mw)
                if model in ('prior','context_logistic') and (abs(point)>1e-10 or np.max(np.abs(draws[valid]))>1e-10): raise ValueError('Context-only exchange invariance failed')
                if dataset=='SEEDIV' and (abs(point)>1e-10 or np.max(np.abs(draws[valid]))>1e-10): raise ValueError('Assigned video-label exchange sanity failed')
                item[which]={'aligned_minus_exchanged':point,'crossed_percentile_95':np.quantile(draws[valid],[.025,.975]).tolist(),'nonestimable_draws':int((~valid).sum())}
                distributions[(model,arm,task)][which]=(point,draws,valid)
            alignment['models'][model][arm][task]=item
    for model in MODELS:
        for baseline in ('prior','context_logistic'):
            for arm in ARMS:
                for task in tasks:
                    for which in ('BA','logloss'):
                        a,da,va=distributions[(model,arm,task)][which]; b,db,vb=distributions[(baseline,arm,task)][which]; valid=va&vb
                        alignment['contrasts'].append({'model':model,'baseline':baseline,'arm':arm,'task':task,'metric':which,'alignment_difference':a-b,
                            'crossed_percentile_95':np.quantile((da-db)[valid],[.025,.975]).tolist(),'nonestimable_draws':int((~valid).sum())})
    atomic_json(output/f'alignment_{dataset.lower()}.json',alignment)
    return result


def report(output):
    rows=['# Held-out tuning findings','', 'This development study reuses the existing participants and videos. The research question and manuscript remain unchanged.','',
          'Every neural fold independently selected learning rate, duration and BatchNorm variant using equally weighted familiar-video and unseen-video source-validation balanced log loss. Test participants were excluded from fitting and selection. Context-only calibration used the same criterion.','']
    for dataset in ('SEEDIV','DEAP'):
        result=json.loads((output/f'comparison_{dataset.lower()}.json').read_text()); task='coarse3' if dataset=='SEEDIV' else 'binary'
        rows += [f'## {dataset}', '', '| Model | Video exposure | Balanced accuracy | Balanced log loss |','|---|---|---:|---:|']
        for model,arms in result['models']['combined'].items():
            for arm,item in arms.items(): rows.append(f'| {model} | {arm} | {100*item[task]["BA"]["point"]:.2f}% | {item[task]["logloss"]["point"]:.4f} |')
        rows += ['',f'Complete paired contrasts, all four individual grouping/initialization results and conditional uncertainty are in [comparison_{dataset.lower()}.json](comparison_{dataset.lower()}.json).',
                 f'The within-video rating/EEG pairing diagnostic is in [alignment_{dataset.lower()}.json](alignment_{dataset.lower()}.json).','']
    rows += ['These results use different raw versus STFT representations for EEGNet and GRU; they cannot isolate an architectural effect. Training exposure and the familiar-validation panel change together across arms, so the exposure contrast describes that whole procedure. SEED labels are assigned by video; its known-video prior and exchange sanity check cannot establish individual physiological emotion prediction. DEAP waveform comparison to the first-party release remains outstanding.','',
             'Checkpoint replay verifies numerical output, rather than establishing optimization convergence. All candidate outcomes, including weak fits, are retained. Most selected GRU weights were verified and deleted according to the declared storage rule; retained sentinels and all EEGNet weights permit later state replay. No successful initialization, grouping or significant contrast was selected for reporting.','',
             'A contribution or research-question change still requires a concrete review with the author and explicit approval.']
    (output/'FINDINGS.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=REPO.parent/'publication_runs'/STUDY)
    args=parser.parse_args()
    for dataset in ('SEEDIV','DEAP'): analyze(args.output.resolve(),dataset)
    report(args.output.resolve())
