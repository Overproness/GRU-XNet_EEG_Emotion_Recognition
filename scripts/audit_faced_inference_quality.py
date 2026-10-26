"""Independent signal-rule, exact-source median and cached fallback audit.
Does not call the new policy/replay implementation, fit, or read original files.
"""
from collections import defaultdict
import csv
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_artifact_control import BASE as PREVIOUS,PRIVATE,PROGRAMME,load,digest,save
from scripts.audit_faced_artifact_controls import independent_metrics
PUBLIC=REPO/'results/development/faced_inference_quality_2026-10-11'

def read(path):
    with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('vid','rotation'):r[k]=int(r[k])
        for k in ('alpha','target','prediction'):
            r[k]=float(r[k]) if r[k] else None
        r['oof_fold']=int(r['oof_fold']) if r['oof_fold'] else None
        for k in ('previous_prediction',):
            if k in r:r[k]=float(r[k])
        for k in ('fallback','eeg_used','eeg_available','previous_fallback'):
            if k in r:r[k]=r[k]=='True'
    return rows

def key(r):return tuple(r[k] for k in ('rotation','model','arm','alpha','oof_fold','subject','vid'))

def main():
    signal=load(PUBLIC/'signal_audit.json');comparison=load(PUBLIC/'comparison.json')
    original=read(PREVIOUS/'predictions.csv');current=read(PUBLIC/'predictions.csv')
    plan=load(PROGRAMME);before=load(PREVIOUS/'comparison.json')
    oi={key(r):r for r in original};ni={key(r):r for r in current}
    if len(oi)!=27720 or len(ni)!=27720 or oi.keys()!=ni.keys():raise ValueError('Prediction identity coverage differs')
    available={};private_caches_checked=0;rule_checks=0
    for d in signal['trial_decisions']:
        pair=d['subject'],d['vid'];stem=f'{pair[0]}_clip{pair[1]:02d}'
        qp=PRIVATE/(stem+'_quality.json');ap=PRIVATE/(stem+'_artifact.json')
        if digest(qp)!=signal['private_technical_cache_hashes'][qp.name] or digest(ap)!=signal['private_technical_cache_hashes'][ap.name]:raise ValueError('Technical cache differs')
        q=load(qp);a=load(ap);private_caches_checked+=2;flags=[]
        if a['repair_capacity_exceeded']:flags.append(('all','repair_capacity_exceeded'))
        for c in q['channels']:
            tests=(('adc_rail',c['raw_rail_fraction']>.0001),('stationary_counts',c['raw_identical_adjacent_fraction']>.95),
                ('low_processed_variation',c['processed_sd_uv']<.1),('high_processed_variation',c['processed_sd_uv']>100.),
                ('high_processed_excursion',c['processed_peak_to_peak_uv']>1000.),('residual_50hz_power',c['processed_49_51_to_1_45_power_ratio']>.10))
            flags.extend((c['channel'],name) for name,bad in tests if bad);rule_checks+=6
        expected=not flags
        if expected!=d['eeg_available'] or flags!=[(x['channel'],x['reason']) for x in d['reasons']]:raise ValueError('Independent availability differs')
        available[pair]=expected
    source={(r['subject'],r['vid']) for r in load(PREVIOUS/'source_records.json')['records']}
    validation={(r['subject'],r['vid']) for r in load(PREVIOUS/'validation_records.json')['records']}
    if set(available)!=source|validation or any(not available[p] for p in source):raise ValueError('Source gate/declared signal coverage differs')
    targets={}
    for r in original:
        pair=r['subject'],r['vid']
        if pair not in available:raise PermissionError('Undeclared participant/material')
        if pair in targets and targets[pair]!=r['target']:raise ValueError('Repeated target mismatch')
        targets[pair]=r['target']
    # Compute every source/fold median independently using only permitted cached targets.
    medians={}
    for k in range(3):
        src=[(r['subject'],r['vid']) for r in plan['rotation_manifests'][k]['arms']['source_fit']]
        for fold in (None,0,1,2,3,4):
            permitted=src if fold is None else [p for p in src if p[0] not in set(plan['source_participant_oof_folds'][fold])]
            medians[k,fold]=float(np.median([targets[p] for p in permitted]))
    fallbacks=changed=0;max_error=0.
    for identity,r in ni.items():
        old=oi[identity];pair=r['subject'],r['vid'];model=r['model']
        if any(r[f]!=old[f] for f in ('target','family','acquisition_group')) or r['previous_prediction']!=old['prediction']:
            raise ValueError('Original target/metadata/prediction was changed')
        fallback=False;used=False
        if model in ('spectral_eeg_only','context_plus_spectral_eeg') and not available[pair]:
            fallback=True
            if model=='spectral_eeg_only':
                value=medians[r['rotation'],r['oof_fold']];branch='source_global_median_fallback'
            else:
                alpha=before['rotations'][r['rotation']]['selected']['context_only']['alpha']
                value=oi[(r['rotation'],'context_only',r['arm'],alpha,r['oof_fold'],*pair)]['prediction'];branch='context_fallback'
        else:
            value=old['prediction']
            branch='context_model' if model=='context_only' else 'eeg_model' if model=='spectral_eeg_only' else 'joint_model' if model=='context_plus_spectral_eeg' else 'fixed_source_prior'
            used=model in ('spectral_eeg_only','context_plus_spectral_eeg')
        error=abs(r['prediction']-value);max_error=max(max_error,error)
        if error>1e-12 or r['fallback']!=fallback or r['eeg_used']!=used or r['branch']!=branch or r['eeg_available']!=available[pair]:raise ValueError('Independent fallback branch/value differs')
        fallbacks+=fallback;changed+=r['prediction']!=old['prediction']
        if (model=='context_only' or model not in ('spectral_eeg_only','context_plus_spectral_eeg') or r['arm']=='source_oof') and r['prediction']!=old['prediction']:
            raise ValueError('Context/prior/source-OOF prediction changed')
    max_metric=0.;gains=[];tables=[]
    for k in range(3):
        c=comparison['rotations'][k]
        for family,selected in c['selected'].items():
            if selected['alpha']!=before['rotations'][k]['selected'][family]['alpha']:raise ValueError('Alpha reselected')
            for arm in ('validation_unseen','validation_familiar'):
                subset=[r for r in current if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,family,selected['alpha'],arm)]
                expected={(r['subject'],r['vid']) for r in plan['rotation_manifests'][k]['arms'][arm]}
                if {(r['subject'],r['vid']) for r in subset}!=expected or len(subset)!=len(expected):raise ValueError('Selected arm identities dropped')
                metrics=independent_metrics(subset)
                delta=max(abs(metrics[m]-selected[arm][m]) for m in metrics);max_metric=max(max_metric,delta)
        # Check every candidate/fixed-prior/OOF metric, including preserved identity coverage.
        for family,candidates in c['candidate_metrics'].items():
            for candidate in candidates:
                for arm in ('validation_unseen','validation_familiar'):
                    metrics=independent_metrics([r for r in current if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,family,candidate['alpha'],arm)])
                    max_metric=max(max_metric,max(abs(metrics[m]-candidate[arm][m]) for m in metrics))
        for arm,priors in c['priors'].items():
            for family,stored in priors.items():
                metrics=independent_metrics([r for r in current if (r['rotation'],r['model'],r['arm'])==(k,family,arm)])
                max_metric=max(max_metric,max(abs(metrics[m]-stored[m]) for m in metrics))
        for family,stored in c['source_oof'].items():
            metrics=independent_metrics([r for r in current if (r['rotation'],r['model'],r['arm'])==(k,family,'source_oof')])
            max_metric=max(max_metric,max(abs(metrics[m]-stored[m]) for m in metrics))
        chosen={f:{(r['subject'],r['vid']):r for r in current if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,f,c['selected'][f]['alpha'],'validation_unseen')} for f in ('context_only','context_plus_spectral_eeg')}
        cells=defaultdict(list)
        for p,row in chosen['context_only'].items():
            joint=chosen['context_plus_spectral_eeg'][p]
            cells[row['subject'],row['family']].append(abs(row['target']-row['prediction'])-abs(joint['target']-joint['prediction']))
        people=sorted({p for p,f in cells});films=sorted({f for p,f in cells})
        matrix=np.array([[np.mean(cells[p,f]) for f in films] for p in people]);gain=float(matrix.mean());gains.append(gain)
        rng=np.random.default_rng(425401);pi=rng.integers(len(people),size=(4000,len(people)));fi=rng.integers(len(films),size=(4000,len(films)))
        samples=np.array([matrix[np.ix_(p,f)].mean() for p,f in zip(pi,fi)]);interval=np.quantile(samples,[.025,.975])
        if abs(gain-c['paired']['gain'])>1e-12 or not np.allclose(interval,c['paired']['descriptive_95_percent_interval'],rtol=0,atol=1e-12):raise ValueError('Independent paired interval differs')
        if any(abs(np.delete(matrix,i,axis=1).mean()-c['paired']['leave_one_film_out_gain'][film])>1e-12 for i,film in enumerate(films)):raise ValueError('Independent film sensitivity differs')
        tables.append({'rotation':k,'context_mae':c['selected']['context_only']['validation_unseen']['participant_film_mae'],
            'eeg_mae':c['selected']['spectral_eeg_only']['validation_unseen']['participant_film_mae'],
            'joint_mae':c['selected']['context_plus_spectral_eeg']['validation_unseen']['participant_film_mae'],
            'gain_over_context':gain,'descriptive_interval':interval.tolist(),'coverage':c['coverage']})
    if max_metric>1e-12 or abs(float(np.mean(gains))-comparison['mean_joint_gain_over_context'])>1e-12 or fallbacks!=comparison['fallback_prediction_rows'] or changed!=comparison['changed_prediction_rows']:
        raise ValueError('Independent summary differs')
    summary={'rotations':tables,'prediction_rows_verified':27720,'availability_rules_recomputed':rule_checks,
        'private_technical_caches_verified':private_caches_checked,'source_fold_medians_verified':len(medians),
        'fallback_rows_verified':fallbacks,'changed_rows_verified':changed,'max_prediction_discrepancy':max_error,
        'max_metric_discrepancy':max_metric,'all_previous_selected_alphas_preserved':True,
        'all_source_oof_context_and_prior_predictions_preserved':True,
        'mean_joint_gain_over_context':float(np.mean(gains)),
        'source_unavailable_trials':sum(not available[p] for p in source),
        'validation_unavailable_trials':sum(not available[p] for p in validation),
        'new_fits':0,'new_rating_decoding':0,'original_files_or_waveform_features_read':False,
        'post_hoc_sensitivity':True,'independent_confirmation':False,'research_question_changed':False}
    out=PUBLIC/'independent_analysis'
    files=[Path(__file__),PUBLIC/'plan.json',PUBLIC/'signal_audit.json',PUBLIC/'comparison.json',PUBLIC/'predictions.csv',PREVIOUS/'predictions.csv',PROGRAMME]
    if (out/'summary.json').exists():
        if load(out/'summary.json')!=summary:raise ValueError('Immutable independent analysis differs')
        for rel,h in load(out/'verification.json')['bindings'].items():
            if digest(REPO/rel)!=h:raise ValueError('Independent audit artifact changed')
    else:
        save(out/'summary.json',summary);files.append(out/'summary.json')
        save(out/'verification.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in files},
            'new_policy_or_replay_function_called':False,'new_fits':0,'original_files_opened':False})
    print({k:summary[k] for k in ('prediction_rows_verified','fallback_rows_verified','max_prediction_discrepancy','mean_joint_gain_over_context','new_fits')})

if __name__=='__main__':main()
