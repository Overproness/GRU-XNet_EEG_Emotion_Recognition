"""Independently replay cached fit equations, source identities and public metrics.
No refitting, original BDF/MAT reads, new target decoding or confirmation access.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_artifact_control import BASE,PRIVATE,PROGRAMME,load,digest,save,check
FAMILIES=('context_only','spectral_eeg_only','context_plus_spectral_eeg')
PRIORS=('global_median','coarse_valence_median','category_median','familiar_video_median')
ALPHAS=(.1,1.,10.,100.,1000.)
OUT=BASE/'control_analysis'

def independent_metrics(rows):
    grouped=defaultdict(list);people=defaultdict(list)
    errors=np.array([r['prediction']-r['target'] for r in rows])
    if len({(r['subject'],r['vid']) for r in rows})!=len(rows) or not np.isfinite(errors).all():raise ValueError('Invalid prediction geometry')
    for r,e in zip(rows,errors):grouped[r['subject'],r['family']].append(abs(float(e)))
    for (p,f),e in grouped.items():people[p].append(float(np.mean(e)))
    return {'participant_film_mae':float(np.mean([np.mean(e) for e in people.values()])),
        'trial_mae':float(np.abs(errors).mean()),'trial_mse':float(np.square(errors).mean()),
        'trials':len(rows),'participants':len(people),'films':len({r['family'] for r in rows})}

def same_metrics(a,b):
    if set(a)!=set(b) or any(abs(a[k]-b[k])>1e-11 for k in a):raise ValueError('Independent public metrics differ')

def audit():
    plan,activation,source,validation=check();execution=load(BASE/'execution.json')
    if execution['status']!='completed_controls':raise PermissionError('Completed gated controls required')
    if not load(BASE/'source_gate.json')['passed']:raise PermissionError('Source gate not passed')
    rows=[]
    with (BASE/'predictions.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            r['vid']=int(r['vid']);r['rotation']=int(r['rotation'])
            r['alpha']=float(r['alpha']) if r['alpha'] else None
            r['oof_fold']=int(r['oof_fold']) if r['oof_fold'] else None
            r['target']=float(r['target']);r['prediction']=float(r['prediction']);r['fallback']=r['fallback']=='True'
            rows.append(r)
    if len(rows)!=execution['prediction_rows']:raise ValueError('Prediction receipt count differs')
    identities=set(source+validation);targets={};features={};metadata={}
    for r in rows:
        pair=r['subject'],r['vid']
        if pair not in identities or not 0<=r['target']<=7 or not 0<=r['prediction']<=7:raise ValueError('Undeclared/nonfinite trial outcome')
        if pair in targets and targets[pair]!=r['target']:raise ValueError('Repeated target disagrees')
        targets[pair]=r['target']
    if set(targets)!=identities:raise ValueError('Original decoded target coverage differs')
    private_targets=load(PRIVATE/'targets.json')
    if {(r['subject'],r['vid']):r['valence'] for r in private_targets}!=targets:raise ValueError('Public target bytes differ from selective cache')
    catalogue={r['clip']:r for r in load(REPO/'results/development/faced_codebooks_2026-10-10/audit.json')['stimuli']}
    records=load(BASE/'source_records.json')['records']+load(BASE/'validation_records.json')['records']
    for r in records:
        pair=r['subject'],r['vid'];metadata[pair]={**catalogue[r['vid']],'acquisition_group':r['acquisition_group']}
        if not r['technical_passed']:continue
        path=PRIVATE/f"{r['subject']}_clip{r['vid']:02d}_features.npz"
        if digest(path)!=r['feature_cache_sha256']:raise ValueError('Feature cache changed')
        with np.load(path,allow_pickle=False) as f:features[pair]=(f['context'].copy(),f['spectral'].copy())
    def vector(pair,family):
        c,e=features[pair]
        return c if family=='context_only' else e if family=='spectral_eeg_only' else np.concatenate((c,e))
    comparison=load(BASE/'comparison.json');selected={};grid_metrics={}
    for rot in plan['rotation_manifests']:
        k=rot['rotation'];actual=comparison['rotations'][k]
        for family in FAMILIES:
            grid=[]
            for alpha in ALPHAS:
                byarm={}
                for arm in ('validation_unseen','validation_familiar'):
                    chosen=[r for r in rows if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,family,alpha,arm)]
                    expected={(r['subject'],r['vid']) for r in rot['arms'][arm]}
                    if {(r['subject'],r['vid']) for r in chosen}!=expected or len(chosen)!=len(expected):raise ValueError('Candidate evaluation identities differ')
                    byarm[arm]=independent_metrics(chosen)
                grid.append(byarm)
            index=min(range(5),key=lambda i:(grid[i]['validation_unseen']['participant_film_mae'],grid[i]['validation_unseen']['trial_mse'],i))
            selected[k,family]=ALPHAS[index];grid_metrics[k,family]=grid
            if actual['selected'][family]['alpha']!=ALPHAS[index]:raise ValueError('Unseen-only alpha selection does not replay')
            for i,candidate in enumerate(actual['candidates'][family]):
                same_metrics(grid[i]['validation_unseen'],candidate['unseen']);same_metrics(grid[i]['validation_familiar'],candidate['familiar'])
            same_metrics(grid[index]['validation_unseen'],actual['selected'][family]['unseen'])
            same_metrics(grid[index]['validation_familiar'],actual['selected'][family]['familiar'])
    max_scaler=max_gradient=max_prediction=0.;state_checks=[];state_paths=[]
    manifest=load(BASE/'model_manifest.json')['records']
    if len(manifest)!=90:raise ValueError('Exact fit cap differs')
    seen_jobs=set()
    for state in manifest:
        k,family,alpha,fold=state['rotation'],state['family'],state['alpha'],state['oof_fold']
        job=(k,family,alpha,fold)
        if job in seen_jobs:raise ValueError('Duplicated fit state')
        seen_jobs.add(job)
        source_pairs=[(r['subject'],r['vid']) for r in plan['rotation_manifests'][k]['arms']['source_fit']]
        if fold is not None:
            held=set(plan['source_participant_oof_folds'][fold]);source_pairs=[p for p in source_pairs if p[0] not in held]
            if alpha!=selected[k,family]:raise ValueError('OOF alpha differs from declared selection')
        declared=[(r['subject'],r['vid']) for r in state['source_pairs']]
        identity=hashlib.sha256(json.dumps(source_pairs,separators=(',',':')).encode()).hexdigest()
        if declared!=source_pairs or identity!=state['source_identity_sha256']:raise ValueError('Exact source fit identities disagree')
        path=PRIVATE/f"model_{state['index']:03d}.npz";state_paths.append(path)
        if digest(path)!=state['private_state_sha256']:raise ValueError('Fit state changed')
        with np.load(path,allow_pickle=False) as f:stored={name:f[name].copy() for name in f.files}
        x=np.stack([vector(p,family) for p in source_pairs]);y=np.array([targets[p] for p in source_pairs])
        mean=x.mean(axis=0);var=np.square(x-mean).mean(axis=0);scale=np.sqrt(var);scale[var==0]=1.
        error=max(float(np.max(np.abs(stored['mean']-mean))),float(np.max(np.abs(stored['variance']-var))),float(np.max(np.abs(stored['scale']-scale))))
        max_scaler=max(max_scaler,error)
        if error>1e-8:raise ValueError('Scaler is not source-only')
        z=(x-stored['mean'])/stored['scale'];pred=z@stored['coef']+stored['intercept']
        gradient=z.T@(pred-y)+alpha*stored['coef'];gradient_error=float(np.max(np.abs(gradient)))
        intercept_error=abs(float(np.mean(pred-y)));max_gradient=max(max_gradient,gradient_error,intercept_error)
        if gradient_error>1e-7 or intercept_error>1e-10:raise ValueError('Ridge normal equation does not hold')
        chosen=[r for r in rows if (r['rotation'],r['model'],r['alpha'],r['oof_fold'])==job]
        expected_eval={(r['subject'],r['vid']) for arm in ('validation_unseen','validation_familiar') for r in plan['rotation_manifests'][k]['arms'][arm]} if fold is None else {(r['subject'],r['vid']) for r in plan['rotation_manifests'][k]['arms']['source_fit'] if r['subject'] in held}
        if {(r['subject'],r['vid']) for r in chosen}!=expected_eval or len(chosen)!=len(expected_eval):raise ValueError('Saved state prediction identities differ')
        for r in chosen:
            pair=r['subject'],r['vid']
            if r['fallback']:continue
            prediction=float(np.clip(((vector(pair,family)-stored['mean'])/stored['scale'])@stored['coef']+stored['intercept'],0,7))
            delta=abs(prediction-r['prediction']);max_prediction=max(max_prediction,delta)
            if delta>1e-10:raise ValueError('Public prediction differs from saved state')
        state_checks.append({'rotation':k,'family':family,'alpha':alpha,'oof_fold':fold,'source_trials':len(source_pairs),
            'source_trial_mae':float(np.abs(np.clip(pred,0,7)-y).mean()),'ridge_gradient_max_abs':gradient_error})
    expected_jobs={(k,f,a,None) for k in range(3) for f in FAMILIES for a in ALPHAS}|{(k,f,selected[k,f],fold) for k in range(3) for f in FAMILIES for fold in range(5)}
    if seen_jobs!=expected_jobs:raise ValueError('Fit-job grid differs')
    # Independent source median estimations for all main/OOF prior groups.
    priors_checked=0
    for k in range(3):
        for fold in (None,0,1,2,3,4):
            pairs=[(r['subject'],r['vid']) for r in plan['rotation_manifests'][k]['arms']['source_fit']]
            if fold is not None:pairs=[p for p in pairs if p[0] not in set(plan['source_participant_oof_folds'][fold])]
            global_median=float(np.median([targets[p] for p in pairs]))
            for name in PRIORS:
                groups=defaultdict(list)
                for pair in pairs:
                    key=pair[1] if name=='familiar_video_median' else metadata[pair]['assigned_valence'] if name=='coarse_valence_median' else metadata[pair]['assigned_category'] if name=='category_median' else 'global'
                    groups[key].append(targets[pair])
                values={g:float(np.median(y)) for g,y in groups.items()}
                chosen=[r for r in rows if (r['rotation'],r['model'],r['oof_fold'])==(k,name,fold)]
                for r in chosen:
                    pair=r['subject'],r['vid']
                    key=pair[1] if name=='familiar_video_median' else metadata[pair]['assigned_valence'] if name=='coarse_valence_median' else metadata[pair]['assigned_category'] if name=='category_median' else 'global'
                    if abs(r['prediction']-values.get(key,global_median))>1e-12:raise ValueError('Source-only median prior differs')
                priors_checked+=1
                if fold is None:
                    for arm in ('validation_unseen','validation_familiar'):
                        same_metrics(independent_metrics([r for r in chosen if r['arm']==arm]),comparison['rotations'][k]['priors'][arm][name])
        for name in FAMILIES+PRIORS:
            same_metrics(independent_metrics([r for r in rows if (r['rotation'],r['model'],r['arm'])==(k,name,'source_oof')]),comparison['rotations'][k]['source_oof'][name])
    # Bootstrap array and leave-film sensitivity replay independently from public rows.
    tables=[];gains=[];prior_gains=[]
    for k in range(3):
        chosen={f:[r for r in rows if (r['rotation'],r['model'],r['arm'],r['alpha'])==(k,f,'validation_unseen',selected[k,f])] for f in FAMILIES}
        c={(r['subject'],r['vid']):r for r in chosen['context_only']};j={(r['subject'],r['vid']):r for r in chosen['context_plus_spectral_eeg']}
        grouped=defaultdict(list)
        for p,r in c.items():grouped[r['subject'],r['family']].append(abs(r['target']-r['prediction'])-abs(j[p]['target']-j[p]['prediction']))
        people=sorted({p for p,f in grouped});films=sorted({f for p,f in grouped})
        matrix=np.array([[np.mean(grouped[p,f]) for f in films] for p in people]);gain=float(matrix.mean());gains.append(gain)
        rng=np.random.default_rng(425401);pi=rng.integers(len(people),size=(4000,len(people)));fi=rng.integers(len(films),size=(4000,len(films)))
        draws=np.array([np.mean(matrix[np.ix_(a,b)]) for a,b in zip(pi,fi)])
        interval=np.quantile(draws,[.025,.975]);actual=comparison['rotations'][k]
        if abs(gain-actual['paired']['gain'])>1e-11 or not np.allclose(interval,actual['paired']['descriptive_95_percent_interval'],atol=1e-11,rtol=0):raise ValueError('Crossed-bootstrap replay differs')
        for i,film in enumerate(films):
            if abs(float(np.delete(matrix,i,axis=1).mean())-actual['paired']['leave_one_film_out_gain'][film])>1e-11:raise ValueError('Leave-film sensitivity differs')
        priormae={n:actual['priors']['validation_unseen'][n]['participant_film_mae'] for n in PRIORS[:3]};best=min(priormae,key=priormae.get)
        joint=independent_metrics(chosen['context_plus_spectral_eeg'])['participant_film_mae'];priorgain=priormae[best]-joint;prior_gains.append(priorgain)
        if best!=actual['strongest_fixed_unseen_prior'] or abs(priorgain-actual['joint_gain_over_strongest_fixed_prior'])>1e-11:raise ValueError('Strongest prior comparison differs')
        tables.append({'rotation':k,'context_mae':independent_metrics(chosen['context_only'])['participant_film_mae'],
            'spectral_eeg_mae':independent_metrics(chosen['spectral_eeg_only'])['participant_film_mae'],'joint_mae':joint,
            'joint_gain_over_context':gain,'gain_descriptive_95_percent_interval':interval.tolist(),
            'strongest_fixed_prior':best,'strongest_fixed_prior_mae':priormae[best],'joint_gain_over_strongest_prior':priorgain,
            'selected_alphas':{f:selected[k,f] for f in FAMILIES},'source_oof':actual['source_oof'],
            'selected_source_capacity':{f:next(r['source_trial_mae'] for r in state_checks if (r['rotation'],r['family'],r['alpha'],r['oof_fold'])==(k,f,selected[k,f],None)) for f in FAMILIES}})
    lead=np.mean(gains)>=.15 and sum(g>0 for g in gains)>=2 and np.mean(prior_gains)>0
    if bool(lead)!=comparison['numerical_continuation_conditions_met']:raise ValueError('Declared numerical lead conditions differ')
    return {'rotations':tables,'ridge_fit_states_verified':90,'source_prior_estimations_verified':priors_checked,
        'prediction_rows_verified':len(rows),'target_pairs_verified':len(targets),
        'maximum_source_scaler_discrepancy':max_scaler,'maximum_ridge_gradient_residual':max_gradient,
        'maximum_state_prediction_discrepancy':max_prediction,'state_source_capacity':state_checks,
        'mean_joint_gain_over_context':float(np.mean(gains)),'positive_gain_rotations':sum(g>0 for g in gains),
        'mean_joint_gain_over_strongest_fixed_prior':float(np.mean(prior_gains)),
        'numerical_continuation_conditions_met':bool(lead),'independent_confirmation':False,
        'validation_fallback_prediction_rows':sum(r['fallback'] for r in rows),
        'target_range':[min(targets.values()),max(targets.values())],
        'original_recordings_or_ratings_reread':False,'new_fits':0,'research_question_changed':False},state_paths

def main():
    summary,states=audit()
    files=[Path(__file__),BASE/'activation_plan.json',BASE/'comparison.json',BASE/'predictions.csv',BASE/'model_manifest.json',BASE/'target_access.json']
    if (OUT/'summary.json').exists():
        if load(OUT/'summary.json')!=summary:raise ValueError('Immutable numeric analysis differs')
        receipt=load(OUT/'verification.json')
        for rel,h in receipt['bindings'].items():
            if digest(REPO/rel)!=h:raise ValueError('Public numerical analysis changed')
    else:
        save(OUT/'summary.json',summary);files.append(OUT/'summary.json')
        save(OUT/'verification.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in files},
            'private_states_sha256':{p.name:digest(p) for p in states},'new_fits':0,'original_files_reread':False})
    print({k:summary[k] for k in ('ridge_fit_states_verified','source_prior_estimations_verified','prediction_rows_verified','maximum_state_prediction_discrepancy','mean_joint_gain_over_context','numerical_continuation_conditions_met')})

if __name__=='__main__':main()
