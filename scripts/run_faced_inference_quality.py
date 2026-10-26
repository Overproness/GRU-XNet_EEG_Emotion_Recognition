"""Freeze signal-only inference policy and replay existing cached predictions.

No original files, new targets, waveform features, estimators or fitted states.
All prior alphas and source/OOF identities stay fixed; this is post-hoc sensitivity.
"""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_artifact_control import (BASE as PREVIOUS,PRIVATE,check as check_previous,
    load,digest,save,runtime,PROGRAMME)
from scripts.audit_faced_artifact_controls import independent_metrics
from gruxnet.faced_classical_runner import paired_diagnostics
from gruxnet.faced_inference_quality import (RULES,POLICY_ID,eeg_availability,
    predict_with_quality,require_source_quality)
PUBLIC=REPO/'results/development/faced_inference_quality_2026-10-11'
PROTOCOL=REPO/'docs/publication/FACED_Inference_Quality_Policy_2026-10-11.md'
FAMILIES=('context_only','spectral_eeg_only','context_plus_spectral_eeg')
PRIORS=('global_median','coarse_valence_median','category_median','familiar_video_median')


def dependencies():
    paths=[Path(__file__),REPO/'gruxnet/faced_inference_quality.py',REPO/'tests/test_faced_inference_quality.py',
        PROTOCOL,PROGRAMME,PREVIOUS/'completion/verification.json',PREVIOUS/'source_records.json',
        PREVIOUS/'validation_records.json',PREVIOUS/'predictions.csv',PREVIOUS/'comparison.json',
        PREVIOUS/'model_manifest.json',REPO/'gruxnet/faced_classical_runner.py',
        REPO/'scripts/audit_faced_artifact_controls.py',REPO/'gruxnet/faced_source_quality.py']
    return {p.relative_to(REPO).as_posix():digest(p) for p in paths}


def prior_integrity():
    check_previous()
    for rel,h in load(PREVIOUS/'completion/verification.json')['bindings'].items():
        if digest(REPO/rel)!=h:raise ValueError('Previous immutable experiment changed: '+rel)


def audit_signal():
    plan,activation,source,validation=check_previous();prior_integrity()
    records=load(PREVIOUS/'source_records.json')['records']+load(PREVIOUS/'validation_records.json')['records']
    expected=set(source+validation)
    if {(r['subject'],r['vid']) for r in records}!=expected or len(records)!=len(expected):raise PermissionError('Declared QC union changed')
    qualities={};decisions=[];diagnosis=[];cache_hashes={}
    for r in records:
        pair=r['subject'],r['vid'];role='source' if pair in set(source) else 'validation'
        if role!=r['role']:raise PermissionError('Source/validation QC identity mismatch')
        stem=f"{r['subject']}_clip{r['vid']:02d}"
        qp=PRIVATE/(stem+'_quality.json');ap=PRIVATE/(stem+'_artifact.json')
        if digest(qp)!=r['quality_file_sha256'] or digest(ap)!=r['artifact_file_sha256']:raise ValueError('Cached QC/artifact bytes changed')
        cache_hashes[qp.name]=digest(qp);cache_hashes[ap.name]=digest(ap)
        q=load(qp);a=load(ap);qualities[pair]=q
        d=eeg_availability(q,technical_passed=r['technical_passed'],repair_capacity_exceeded=r['repair_capacity_exceeded'])
        derived=[{'channel':f['channel'],'flag':f['reason']} for f in d['reasons']]
        if derived!=r['review_flags']:raise ValueError('Quality decision differs from independently frozen previous QC')
        decisions.append({'subject':pair[0],'vid':pair[1],'role':role,**d})
        if not d['eeg_available']:
            raw_details=[]
            for i,c in enumerate(q['channels']):
                if c['raw_rail_fraction']>RULES['raw_rail_fraction_above'] or c['raw_identical_adjacent_fraction']>RULES['raw_identical_adjacent_fraction_above']:
                    raw_details.append({'channel':c['channel'],'raw_rail_fraction':c['raw_rail_fraction'],
                        'raw_identical_adjacent_fraction':c['raw_identical_adjacent_fraction'],
                        'raw_sd_uv':c['raw_sd_uv'],'raw_peak_to_peak_uv':c['raw_peak_to_peak_uv'],
                        'raw_median_uv':c['raw_median_uv'],'pre_car_sd_uv':a['pre_car_sd_uv'][i],
                        'processed_sd_uv':c['processed_sd_uv'],
                        'constant_at_negative_full_scale':bool(c['raw_rail_fraction']==1 and c['raw_identical_adjacent_fraction']==1 and c['raw_sd_uv']==0 and c['raw_median_uv']==-375000)})
            diagnosis.append({'subject':pair[0],'vid':pair[1],'native_rate_hz':r['native_rate_hz'],
                'all_selected_counts_previously_scalar_checked':r['scalar_reference_checked'],
                'native_samples_per_channel':r['sample_interval'][1]-r['sample_interval'][0],
                'previous_amplitude_interpolations':a['repaired_channels'],'raw_channel_details':raw_details})
    gate=require_source_quality([r for r in records if r['role']=='source'],lambda p:qualities[p],source)
    return {'policy_id':POLICY_ID,'source_gate':gate,'trial_decisions':decisions,
        'source_eeg_unavailable_trials':sum(not r['eeg_available'] for r in decisions if r['role']=='source'),
        'validation_eeg_unavailable_trials':sum(not r['eeg_available'] for r in decisions if r['role']=='validation'),
        'flagged_diagnosis':diagnosis,'private_technical_cache_hashes':cache_hashes,
        'original_files_opened':0,'rating_values_opened':0,'waveform_features_opened':0,
        'prediction_rows_opened':0,'new_fits':0,'research_question_changed':False}


def declare():
    signal=audit_signal()
    if not signal['source_gate']['passed']:raise PermissionError('Future source gate fails; no prediction replay allowed')
    save(PUBLIC/'signal_audit.json',signal)
    save(PUBLIC/'plan.json',{'date':'2026-10-11','policy_id':POLICY_ID,'rules':RULES,
        'bindings':dependencies(),'runtime':runtime(),'signal_audit_sha256':digest(PUBLIC/'signal_audit.json'),
        'quality_handling':'Any original raw rail/stationarity, residual processed QC/capacity, or numerical unavailability makes trial EEG unavailable. No trial/person exclusion or new interpolation.',
        'inference':'Joint uses already selected context prediction; EEG-only uses exact source/OOF global median; context and fixed priors unchanged. Never call unavailable EEG predictor.',
        'source_guard':'Exact declared source coverage and all quality metrics must pass before further EEG fitting. Source faults hold fitting rather than silently dropping, zero-imputing or using inference fallback for training.',
        'access':'Previously decoded technical QC/artifact caches and authorized published prediction rows only. No original BDF/MAT or waveform-feature/state cache access; no new rating decoding.',
        'grid':'Replay all 27,720 existing prediction rows, retaining all five main candidate alphas and all selected source OOF rows. No fit, alpha reselection, new model, threshold search or participant/material reassignment.',
        'selection':'Use previous context and joint/EEG alphas unchanged even if fallback would alter candidate ordering. All outcome effects are post-hoc sensitivity on previously evaluated development data.',
        'bootstrap':{'draws':4000,'seed':425401,'scope':'Exploratory conditional on old development-selected alphas; six films and same twenty people; no independent confirmation'},
        'continuation':'No automatic fitting/encoder/confirmation launch or paper pivot from this sensitivity. Reviewable policy first; separately declare any further bounded fitting.',
        'synthetic_tests_passed':13,'original_ratings_decoded_in_this_phase':0,'new_fits':0,
        'confirmation_access':False,'research_question_changed':False,
        'primary_reference':'https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html',
        'reference_scope':'General early raw-channel QC rationale, not source of our engineering thresholds or mandatory fallback rule; docs version1.13.2, no changed MNE runtime invoked.'})
    print({'policy_frozen':POLICY_ID,'source_trials':signal['source_gate']['source_trials'],
        'source_unavailable':signal['source_eeg_unavailable_trials'],'validation_unavailable':signal['validation_eeg_unavailable_trials'],
        'prediction_rows_opened':0,'new_fits':0})


def check():
    prior_integrity();p=load(PUBLIC/'plan.json')
    if p['bindings']!=dependencies() or p['runtime']!=runtime() or p['rules']!=RULES:raise ValueError('Frozen inference policy changed')
    if digest(PUBLIC/'signal_audit.json')!=p['signal_audit_sha256']:raise ValueError('Signal-only diagnosis changed')
    return p


def read_predictions():
    rows=[]
    with (PREVIOUS/'predictions.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            r['vid']=int(r['vid']);r['rotation']=int(r['rotation'])
            r['alpha']=float(r['alpha']) if r['alpha'] else None;r['oof_fold']=int(r['oof_fold']) if r['oof_fold'] else None
            r['target']=float(r['target']);r['prediction']=float(r['prediction']);r['fallback']=r['fallback']=='True';rows.append(r)
    return rows


def replay():
    check();signal=load(PUBLIC/'signal_audit.json');decisions={(r['subject'],r['vid']):r for r in signal['trial_decisions']}
    if not signal['source_gate']['passed']:raise PermissionError('No passed source-quality gate')
    rows=read_predictions();previous=load(PREVIOUS/'comparison.json');plan=load(PROGRAMME)
    if len(rows)!=27720:raise ValueError('Previous prediction coverage differs')
    index={(r['rotation'],r['model'],r['arm'],r['alpha'],r['oof_fold'],r['subject'],r['vid']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('Duplicated previous prediction job/trial')
    result=[]
    for r in rows:
        pair=r['subject'],r['vid'];d=decisions[pair];k=r['rotation'];model=r['model']
        if model in FAMILIES:
            def context():
                alpha=previous['rotations'][k]['selected']['context_only']['alpha']
                return index[k,'context_only',r['arm'],alpha,r['oof_fold'],*pair]['prediction']
            median=index[k,'global_median',r['arm'],None,r['oof_fold'],*pair]['prediction']
            response=predict_with_quality(d,model,eeg_predictor=lambda:r['prediction'],
                context_predictor=(lambda:r['prediction']) if model=='context_only' else context,source_median=median)
        else:
            if model not in PRIORS:raise ValueError('Unknown previous model')
            response={'prediction':r['prediction'],'branch':'fixed_source_prior','fallback':False,'eeg_used':False}
        result.append({**r,'previous_prediction':r['prediction'],'previous_fallback':r['fallback'],**response,
            'eeg_available':d['eeg_available'],'quality_reasons':'|'.join(f"{q['channel']}:{q['reason']}" for q in d['reasons']),
            'policy_id':POLICY_ID})
    # Preserve all candidates and source OOF metrics without revisiting alpha selection.
    tables=[]
    for k in range(3):
        actual=previous['rotations'][k];candidate_metrics={}
        for family in FAMILIES:
            candidate_metrics[family]=[]
            for alpha in (.1,1.,10.,100.,1000.):
                candidate_metrics[family].append({'alpha':alpha,**{arm:independent_metrics([r for r in result if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,family,alpha,arm)]) for arm in ('validation_unseen','validation_familiar')}})
        selected={}
        for family in FAMILIES:
            alpha=actual['selected'][family]['alpha']
            selected[family]={'alpha':alpha,**{arm:independent_metrics([r for r in result if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,family,alpha,arm)]) for arm in ('validation_unseen','validation_familiar')}}
        chosen=lambda family:[r for r in result if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,family,actual['selected'][family]['alpha'],'validation_unseen')]
        paired=paired_diagnostics(chosen('context_only'),chosen('context_plus_spectral_eeg'))
        priors={arm:{name:independent_metrics([r for r in result if (r['rotation'],r['model'],r['arm'])==(k,name,arm)]) for name in PRIORS} for arm in ('validation_unseen','validation_familiar')}
        oof={name:independent_metrics([r for r in result if (r['rotation'],r['model'],r['arm'])==(k,name,'source_oof')]) for name in FAMILIES+PRIORS}
        coverage={}
        for arm in ('validation_unseen','validation_familiar'):
            subset=[r for r in result if (r['rotation'],r['model'],r['alpha'],r['arm'])==(k,'context_plus_spectral_eeg',actual['selected']['context_plus_spectral_eeg']['alpha'],arm)]
            coverage[arm]={'trials':len(subset),'eeg_unavailable':sum(not r['eeg_available'] for r in subset),'fallbacks':sum(r['fallback'] for r in subset),
                'participants_retained':len({r['subject'] for r in subset}),'films_retained':len({r['family'] for r in subset})}
        best=actual['strongest_fixed_unseen_prior']
        tables.append({'rotation':k,'selected':selected,'candidate_metrics':candidate_metrics,'source_oof':oof,
            'priors':priors,'paired':paired,'coverage':coverage,'previous_gain_over_context':actual['paired']['gain'],
            'strongest_fixed_prior_unchanged':best,'joint_gain_over_strongest_fixed_prior':priors['validation_unseen'][best]['participant_film_mae']-selected['context_plus_spectral_eeg']['validation_unseen']['participant_film_mae']})
    return result,{'policy_id':POLICY_ID,'rotations':tables,
        'mean_joint_gain_over_context':float(np.mean([r['paired']['gain'] for r in tables])),
        'previous_mean_joint_gain_over_context':previous['mean_joint_gain_over_context'],
        'mean_joint_gain_over_strongest_fixed_prior':float(np.mean([r['joint_gain_over_strongest_fixed_prior'] for r in tables])),
        'prediction_rows':len(result),'fallback_prediction_rows':sum(r['fallback'] for r in result),
        'changed_prediction_rows':sum(r['prediction']!=r['previous_prediction'] for r in result),
        'source_unavailable_trials':signal['source_eeg_unavailable_trials'],'validation_unavailable_trials':signal['validation_eeg_unavailable_trials'],
        'all_alphas_unchanged':True,'refits':0,'new_original_ratings_decoded':0,'original_files_or_waveform_features_opened':0,
        'post_hoc_cached_sensitivity':True,'independent_confirmation':False,'no_automatic_continuation':True,
        'confirmation_access':False,'research_question_changed':False}


def execute():
    if (PUBLIC/'execution.json').exists():raise ValueError('Preserve earlier sensitivity execution')
    rows,summary=replay()
    with (PUBLIC/'predictions.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(PUBLIC/'comparison.json',summary)
    save(PUBLIC/'execution.json',{'status':'completed_cached_sensitivity','plan_sha256':digest(PUBLIC/'plan.json'),
        'predictions_sha256':digest(PUBLIC/'predictions.csv'),'comparison_sha256':digest(PUBLIC/'comparison.json'),
        'new_fits':0,'original_files_or_waveform_features_opened':0,'new_ratings_decoded':0,'confirmation_access':False})
    print({k:summary[k] for k in ('prediction_rows','fallback_prediction_rows','mean_joint_gain_over_context','previous_mean_joint_gain_over_context','refits')})


def verify():
    rows,summary=replay();actual=load(PUBLIC/'comparison.json');execution=load(PUBLIC/'execution.json')
    if actual!=summary or digest(PUBLIC/'predictions.csv')!=execution['predictions_sha256'] or digest(PUBLIC/'comparison.json')!=execution['comparison_sha256']:
        raise ValueError('Frozen cached sensitivity changed')
    with (PUBLIC/'predictions.csv').open(encoding='utf-8',newline='') as f:
        saved=list(csv.DictReader(f))
    for a,b in zip(saved,rows):
        if a!={k:('' if v is None else str(v)) for k,v in b.items()}:raise ValueError('Prediction replay differs')
    if len(saved)!=len(rows):raise ValueError('Prediction row count differs')
    files=[PUBLIC/n for n in ('plan.json','signal_audit.json','comparison.json','predictions.csv','execution.json')]
    receipt={'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in files},
        'rows_replayed':len(rows),'new_fits':0,'all_preceding_experiment_bindings_preserved':True,
        'all_alphas_unchanged':True,'original_data_or_waveform_features_read':False,
        'confirmation_access':False,'research_question_changed':False}
    path=PUBLIC/'verification.json'
    if path.exists():
        if load(path)!=receipt:raise ValueError('Immutable sensitivity verification changed')
    else:save(path,receipt)
    print({'cached_sensitivity_verified':True,'rows_replayed':len(rows),'new_fits':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('declare','execute','verify'));args=p.parse_args()
    {'declare':declare,'execute':execute,'verify':verify}[args.mode]()
