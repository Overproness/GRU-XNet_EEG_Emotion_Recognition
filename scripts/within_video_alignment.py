"""Predeclared no-refit EEG/rating alignment diagnostic within held-out videos."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.data import digest, sha256, write_json
from gruxnet.grouped_material_controls import probabilities, weights
from gruxnet.full_context_controls_v2 import ALL, ARMS

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent/'publication_runs'
PLAN=ROOT/'within_video_alignment_plan_2026-10-06.json'


def plan():
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'development_only':True,'research_question_change_approved':False,
            'source_sha256':sha256(Path(__file__)),
            'timing':'Supplement declared while corrected full-model fits are running, before aggregate primary results are computed; no fitting, checkpoint selection or primary-protocol change.',
            'population':'Both corpora, all6 full-study models, both arms, all primary and secondary tasks; reuse only verified selected OOF probabilities from grouping1.',
            'exchange':'For each recipient trial, average classification correctness and target log loss over EVERY other held-out participant who saw the SAME video in the SAME selected model/cell. Each donor receives1/(k-1) weight. Context/normalizer/model are identical within a cell and video, so reuse donor probabilities is exactly test-time EEG exchange with recipient context held fixed. Average scores, not probabilities or ensemble classifications. Require >=2 held-out people/video; otherwise report noneligible trials explicitly and all metrics use the same eligible cohort.',
            'controls':'Source-only raw/calibrated video priors must be invariant to exchange. SEED assigned video labels imply zero aggregate matched-minus-exchanged differences, including symmetric pair-weighted bootstrap, serving as a mathematical sanity control. DEAP individual ratings can differ within video.',
            'uncertainty':'10000 preexisting seed20261006 person/video weight draws. Dyadic weights w_recipient*w_donor*w_video/(k-1), SAME weights/eligible pairs/class denominators for aligned and exchanged metrics. Person weights apply to both recipient and donor, preserving shared-person dependence; video draws unstratifiedDEAP/native-stratifiedSEED. Report any draw losing class coverage, omit only nonestimable draws and disclose count; no resampling until significant. Conditional fixedfits/partition/donor cohort, unadjusted exploratory percentile intervals.',
            'contrasts':'Aligned minus within-video exchanged BA and balancedlogloss for ALLmodels/arms/tasks; BApositive/loglossnegative favors aligned individualEEG. Also report all EEG-model alignment differences versus both context-only controls; all context-only alignment differences must be zero.',
            'interpretation':'Association of correctly paired individual EEG and ratings conditional on video/cell, not a physiological mechanism or proof of causal emotion recognition. Exchanging participants also changes artifacts, stable participant traits and demographics. Limited held-out donors and one development grouping; no new samples and no main-question pivot.'}


def pairs(rows):
    rows=rows.reset_index(drop=True); recipient=[]; donor=[]; coefficient=[]; eligible=[]
    for _,group in rows.groupby(['source_session','material_rotation','fold','material_key'],sort=True):
        if group.subject_id.duplicated().any(): raise ValueError('Duplicate person/video within cell')
        ids=group.index.to_numpy()
        if len(ids)<2: continue
        eligible.extend(ids)
        for i in ids:
            for j in ids:
                if i==j: continue
                recipient.append(i); donor.append(j); coefficient.append(1/(len(ids)-1))
    return np.array(recipient,dtype=int),np.array(donor,dtype=int),np.array(coefficient),np.array(sorted(eligible),dtype=int)


def scores(rows,task,dataset,recipient,donor):
    p=probabilities(rows)
    if dataset=='SEEDIV' and task=='binary':
        pos=p[:,2]/np.maximum(p[:,1:].sum(1),1e-12); p=np.stack([1-pos,pos],axis=1)
        y=(rows.original_label.to_numpy()==3).astype(int)
        included=rows.original_label.to_numpy()[recipient]!=0
    else:
        y=rows.label.to_numpy(dtype=int); included=np.ones(len(recipient),dtype=bool)
    target=y[recipient]
    result={}
    for name,index in (('aligned',recipient),('exchanged',donor)):
        result[name]={'BA':(p[index].argmax(1)==target).astype(float),
                      'logloss':-np.log(np.clip(p[index,target],1e-12,1))}
    return target,included,result,p.shape[1]


def aggregate(values,target,included,classes,base,recipient_subject,donor_subject,video,sw,mw):
    points=[]; draws=np.zeros(len(sw)); valid=np.ones(len(sw),dtype=bool)
    for c in range(classes):
        keep=included & (target==c); fixed=base*keep
        if fixed.sum()==0: raise ValueError('Eligible cohort lost class')
        points.append((fixed*values).sum()/fixed.sum())
        for start in range(0,len(sw),256):
            stop=min(start+256,len(sw))
            weighted=sw[start:stop,recipient_subject]*sw[start:stop,donor_subject]*mw[start:stop,video]*fixed[None,:]
            denominator=weighted.sum(1); ok=denominator>0; valid[start:stop]&=ok
            draws[start:stop]+=np.divide(weighted@values,denominator,out=np.zeros(len(weighted)),where=ok)/classes
    return float(np.mean(points)),draws,valid


def analyze(dataset,write=True):
    if json.loads(PLAN.read_text())['source_sha256']!=sha256(Path(__file__)): raise ValueError('Changed predeclared diagnostic source')
    folder=ROOT/f'full_context_v2_{dataset.lower()}'
    if not json.loads((folder/'verification.json').read_text())['passed']: raise ValueError('Selected checkpoint replay required')
    base_rows=pd.read_csv(folder/'predictions_gru_exposed.csv')
    subjects,materials,sw,mw=weights(base_rows,dataset)
    subject_lookup={s:i for i,s in enumerate(subjects)}; video_lookup={s:i for i,s in enumerate(materials)}
    r,d,base,eligible=pairs(base_rows)
    ri=base_rows.iloc[r].subject_id.map(subject_lookup).to_numpy(); di=base_rows.iloc[d].subject_id.map(subject_lookup).to_numpy()
    vi=base_rows.iloc[r].material_key.map(video_lookup).to_numpy()
    metadata=['trial_id','subject_id','source_session','material_rotation','fold','material_key','label','original_label']
    result={'dataset':dataset,'development_only':True,'research_question_change_approved':False,'plan_sha256':sha256(PLAN),
            'source_sha256':sha256(Path(__file__)),'selected_model_verification_sha256':sha256(folder/'verification.json'),
            'eligible_trials':len(eligible),'total_trials':len(base_rows),'excluded_trial_ids':base_rows.drop(eligible).trial_id.tolist(),
            'pairs':len(r),'models':{},'contrasts':[],'input_hashes':{},
            'bootstrap':{'draws':len(sw),'seed':20261006,'person_weight_digest':digest(sw.tolist()),'video_weight_digest':digest(mw.tolist()),
                         'weight':'recipient person * donor person * video / available other donors; shared observed-cell class denominators'}}
    distributions={}
    tasks=('coarse3','binary') if dataset=='SEEDIV' else ('binary',)
    for model in ALL:
        result['models'][model]={}
        for arm in ARMS:
            path=folder/f'predictions_{model}_{arm}.csv'; rows=pd.read_csv(path)
            pd.testing.assert_frame_equal(base_rows[metadata],rows[metadata],check_exact=True)
            result['input_hashes'][path.name]=sha256(path); result['models'][model][arm]={}
            for task in tasks:
                target,included,value,classes=scores(rows,task,dataset,r,d)
                for which in ('BA','logloss'):
                    a,ad,av=aggregate(value['aligned'][which],target,included,classes,base,ri,di,vi,sw,mw)
                    b,bd,bv=aggregate(value['exchanged'][which],target,included,classes,base,ri,di,vi,sw,mw)
                    valid=av & bv; difference=ad-bd
                    if model in ('prior','context_logistic') or dataset=='SEEDIV':
                        if abs(a-b)>1e-12 or np.max(np.abs(difference[valid]))>1e-12: raise ValueError('Exchange invariant sanity check failed')
                    item={'task':task,'statistic':which,'model':model,'arm':arm,'aligned':a,'exchanged':b,'difference':a-b,
                          'crossed_dyadic_percentile_95':np.quantile(difference[valid],[.025,.975]).tolist(),
                          'nonestimable_bootstrap_draws':int((~valid).sum())}
                    result['models'][model][arm][task+'_'+which]=item; result['contrasts'].append(item)
                    distributions[model,arm,task,which]=(a-b,difference,valid)
    for model in ('gru','lstm','cbsatt_local','gru_context'):
        for control in ('prior','context_logistic'):
            for arm in ARMS:
                for task in tasks:
                    for which in ('BA','logloss'):
                        a,ad,av=distributions[model,arm,task,which]; b,bd,bv=distributions[control,arm,task,which]; valid=av & bv
                        result['contrasts'].append({'model_a':model,'model_b':control,'arm':arm,'task':task,'statistic':which,
                            'alignment_difference':a-b,'crossed_dyadic_percentile_95':np.quantile((ad-bd)[valid],[.025,.975]).tolist(),
                            'nonestimable_bootstrap_draws':int((~valid).sum())})
    if write:
        out=ROOT/'within_video_alignment'; out.mkdir(exist_ok=True)
        pair_map=pd.DataFrame({'recipient_trial':base_rows.iloc[r].trial_id.to_numpy(),'donor_trial':base_rows.iloc[d].trial_id.to_numpy(),
                              'recipient_subject':base_rows.iloc[r].subject_id.to_numpy(),'donor_subject':base_rows.iloc[d].subject_id.to_numpy(),
                              'material_key':base_rows.iloc[r].material_key.to_numpy(),'weight':base})
        pair_map.to_csv(out/f'pairs_{dataset.lower()}.csv',index=False)
        write_json(out/f'comparison_{dataset.lower()}.json',result)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__); parser.add_argument('command',choices=('plan','analyze')); args=parser.parse_args()
    if args.command=='plan':
        if PLAN.exists(): raise FileExistsError('Do not overwrite declaration')
        write_json(PLAN,plan()); print(PLAN)
    else:
        for dataset in ('SEEDIV','DEAP'):
            result=analyze(dataset); replay=analyze(dataset,write=False)
            if result!=replay: raise ValueError('Alignment analysis replay changed')
            write_json(ROOT/'within_video_alignment'/f'verification_{dataset.lower()}.json',{'passed':True,'analysis_digest':digest(result),'independent_recomputation':True})
            print(dataset,'eligible',result['eligible_trials'],'contrasts',len(result['contrasts']),flush=True)
