"""Freeze and execute one bounded FACED representation/capacity experiment."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_artifact_control import (BASE as OLD, PRIVATE as OLD_PRIVATE,
    RAW, RAW_EVIDENCE, PROGRAMME, load, save, digest, array_digest, runtime,
    selected_header, read_interval, read_header, calibrate_common_counts,
    artifact_trial_windows, quality_metrics, spectral_features, context_features)
from scripts.run_faced_inference_quality import PUBLIC as QUALITY, check as quality_check, audit_signal
from scripts.audit_faced_artifact_controls import independent_metrics
from gruxnet.faced_classical_runner import paired_diagnostics
from gruxnet.faced_inference_quality import eeg_availability, require_source_quality, predict_with_quality
from gruxnet.faced_representation import (CONDITIONS, covariance_features, band_covariances,
    exact_pairs, fit_readout, predict_readout)

PUBLIC=REPO/'results/development/faced_representation_2026-10-11'
PRIVATE=REPO.parent/'publication_runs/faced_representation_2026-10-11'
PROTOCOL=REPO/'docs/publication/FACED_Representation_Capacity_Protocol_2026-10-11.md'
OLD_FAMILIES=('context_only','spectral_eeg_only','context_plus_spectral_eeg')


def bindings():
    paths=[Path(__file__),PROTOCOL,REPO/'gruxnet/faced_representation.py',
        REPO/'tests/test_faced_representation.py',REPO/'tests/test_faced_representation_runner.py',REPO/'scripts/audit_faced_representation.py',
        PROGRAMME,QUALITY/'completion/verification.json',QUALITY/'predictions.csv',QUALITY/'comparison.json',
        OLD/'source_records.json',OLD/'validation_records.json',
        REPO/'gruxnet/faced_inference_quality.py',REPO/'gruxnet/faced_artifact_control.py',
        REPO/'gruxnet/faced_classical_contract.py',REPO/'gruxnet/faced_classical_runner.py',
        REPO/'scripts/run_faced_artifact_control.py',REPO/'scripts/run_faced_inference_quality.py',
        RAW_EVIDENCE/'headers.json',RAW_EVIDENCE/'joins_numeric_projection.json',RAW_EVIDENCE/'extraction.json']
    return {p.relative_to(REPO).as_posix():digest(p) for p in paths}


def rectangles():
    records=load(OLD/'source_records.json')['records']+load(OLD/'validation_records.json')['records']
    programme=load(PROGRAMME)
    intended=[(r['subject'],r['vid']) for r in programme['input_access']['future_development_pairs']]
    exact_pairs([(r['subject'],r['vid']) for r in records],intended)
    source=[(r['subject'],r['vid']) for r in records if r['role']=='source']
    validation=[(r['subject'],r['vid']) for r in records if r['role']=='validation']
    if len(source)!=980 or len(validation)!=280:raise PermissionError('Changed role rectangles')
    return programme,records,source,validation


def source_sets(programme):
    sets={}
    for rotation in programme['rotation_manifests']:
        k=rotation['rotation'];full=[(r['subject'],r['vid']) for r in rotation['arms']['source_fit']]
        sets[f'r{k}/full']=full
        for fold,people in enumerate(programme['source_participant_oof_folds']):
            sets[f'r{k}/fold{fold}']=[p for p in full if p[0] not in people]
    if len(sets)!=18 or any(len(v)!=(560 if key.endswith('/full') else 448) for key,v in sets.items()):
        raise ValueError('Changed exact source-fold sets')
    return sets


def declare():
    quality_check();programme,records,source,validation=rectangles()
    signal=audit_signal()
    if not signal['source_gate']['passed']:raise PermissionError('Source quality holds study declaration')
    save(PUBLIC/'plan.json',{'date':'2026-10-11','bindings':bindings(),'runtime':runtime(),
        'source_sets':source_sets(programme),'conditions':CONDITIONS,'signal_audit_sha256':digest(QUALITY/'signal_audit.json'),
        'source_trials':980,'validation_trials':280,'validation_eeg_available':265,
        'normal_fits':126,'capacity_fits':12,'total_new_fit_cap':138,'prediction_rows_expected':36120,
        'time_cap_seconds':7200,'blas_threads':2,'source_only_fitting':True,'validation_selection':False,
        'capacity_permutation_seed_base':445401,'quality_policy':'faced-quality-context-fallback-v1',
        'new_original_rating_decoding':0,'confirmation_access':False,'research_question_changed':False,
        'access':'Original selective EEG only in preceding exact intervals/channels of available development trials; opaque original hashes. All source quality/representations before validation EEG/fitting. Previously authorized published development targets after source gate. No waveform persistence.',
        'interpretation':'Reused development; established representations/readouts, no independent confirmation, preregistration or new algorithm. No automatic grid expansion or manuscript pivot.'})
    print({'study_declared':True,'new_fit_cap':138,'new_original_rating_decoding':0},flush=True)


def check():
    quality_check();p=load(PUBLIC/'plan.json')
    if p['bindings']!=bindings() or p['runtime']!=runtime():raise ValueError('Frozen study implementation/runtime changed')
    programme,records,source,validation=rectangles()
    if p['source_sets']!={k:[list(v) for v in pairs] for k,pairs in source_sets(programme).items()}:
        raise ValueError('Frozen source sets changed')
    return p,programme,records,source,validation


class Study:
    def __init__(self,p,programme,records,source,validation):
        self.plan=p;self.programme=programme;self.records={ (r['subject'],r['vid']):r for r in records}
        self.source=source;self.validation=validation;self.started=time.monotonic()
        self.headers={r['subject']:r['data.bdf'] for r in load(RAW_EVIDENCE/'headers.json')['records']}
        self.files={r['member']:r for r in load(RAW_EVIDENCE/'extraction.json')['records']}
        self.catalogue={r['clip']:r for r in load(REPO/'results/development/faced_codebooks_2026-10-10/audit.json')['stimuli']}
        self.signal=audit_signal();self.decisions={(r['subject'],r['vid']):r for r in self.signal['trial_decisions']}
        self.features={b:{} for b in ('context','spectral','covariance')};self.metadata={};self.targets={}
        self.prepared=[];self.fit_records=[];self.predictions=[];self.authenticated={};self.gate=None
        self.values_decoded=0;self.baselines=[]

    def budget(self):
        if time.monotonic()-self.started>self.plan['time_cap_seconds']:raise TimeoutError('Study time cap reached')
        if len(self.fit_records)>138:raise ValueError('Study fit cap exceeded')

    def authenticate(self,subject):
        self.budget()
        if subject in self.authenticated:return
        directory=RAW/subject.replace('sub-','sub');path=directory/'data.bdf'
        if read_header(path)!=self.headers[subject]:raise ValueError('Original header changed')
        for name in ('data.bdf','evt.bdf','After_remarks.mat'):
            if digest(directory/name)!=self.files[f'Data/{directory.name}/{name}']['sha256']:
                raise ValueError('Original or opaque companion changed')
        self.authenticated[subject]={'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,
            'sha256':self.files[f'Data/{directory.name}/data.bdf']['sha256']}

    def prepare(self,requested,role):
        pairs=exact_pairs(requested,self.source if role=='source' else self.validation)
        if role not in ('source','validation'):raise PermissionError('Undeclared role')
        if role=='validation' and (not self.gate or not self.gate['passed']):raise PermissionError('Source gate required')
        for pair in pairs:
            self.budget();person,vid=pair;r=self.records[pair]
            if r['role']!=role:raise PermissionError('Role mismatch before file access')
            metadata={**self.catalogue[vid],'trial':r['trial'],'vid':vid,'family':self.catalogue[vid]['source_film_sha256'],
                      'acquisition_group':r['acquisition_group']}
            self.metadata[pair]=metadata
            context=context_features(metadata)
            if array_digest(context)!=r['context_feature_sha256']:raise ValueError('Context bytes changed')
            self.features['context'][pair]=context
            if not self.decisions[pair]['eeg_available']:
                if role=='source':raise PermissionError('Source EEG unavailable')
                self.prepared.append({'subject':person,'vid':vid,'role':role,'eeg_available':False,
                    'original_signal_or_eeg_features_opened':False,'samples_decoded':0})
                continue
            self.authenticate(person);h=self.headers[person];path=RAW/person.replace('sub-','sub')/'data.bdf'
            start,stop=r['sample_interval'];indices=r['channel_indices'];projected=selected_header(h,indices)
            with path.open('rb') as stream:counts,access=read_interval(stream,h,start,stop,indices,True)
            self.values_decoded+=counts.size
            if array_digest(counts)!=r['count_array_sha256'] or access['range_ledger_sha256']!=r['range_ledger_sha256']:
                raise ValueError('Selected counts/access differ from original qualified experiment')
            values=calibrate_common_counts(counts,person,projected)
            windows,artifact=artifact_trial_windows(values,r['native_rate_hz'])
            if array_digest(windows)!=r['processed_window_sha256']:raise ValueError('Processed windows changed')
            quality=quality_metrics(counts,values,windows,r['native_rate_hz'])
            decision=eeg_availability(quality,technical_passed=True,repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
            if not decision['eeg_available']:raise PermissionError('Current signal fails frozen quality')
            stem=f'{person}_clip{vid:02d}'
            old_features=OLD_PRIVATE/(stem+'_features.npz')
            if digest(old_features)!=r['feature_cache_sha256']:raise ValueError('Original feature cache changed')
            spectral=spectral_features(windows)
            with np.load(old_features,allow_pickle=False) as original:
                if not np.array_equal(original['spectral'],spectral) or not np.array_equal(original['context'],context):
                    raise ValueError('Original reference features differ')
            covariance,info=covariance_features(windows);matrices=band_covariances(windows)
            cache=PRIVATE/(stem+'_features.npz');cache.parent.mkdir(parents=True,exist_ok=True)
            with cache.open('xb') as stream:np.savez_compressed(stream,context=context,spectral=spectral,covariance=covariance,band_covariances=matrices)
            save(PRIVATE/(stem+'_access.json'),access)
            save(PRIVATE/(stem+'_quality.json'),quality)
            self.features['spectral'][pair]=spectral;self.features['covariance'][pair]=covariance
            self.prepared.append({'subject':person,'vid':vid,'role':role,'eeg_available':True,
                'samples_decoded':int(counts.size),'selected_count_hash_matches':True,'processed_window_hash_matches':True,
                'reference_feature_bytes_match':True,'range_ledger_sha256':access['range_ledger_sha256'],
                'private_features_sha256':digest(cache),'private_access_sha256':digest(PRIVATE/(stem+'_access.json')),
                'private_quality_sha256':digest(PRIVATE/(stem+'_quality.json')),'feature_info':info})
            auth=self.authenticated[person]
            if path.stat().st_size!=auth['size'] or path.stat().st_mtime_ns!=auth['mtime_ns']:
                raise ValueError('Original changed during selected reads')
            if len(self.prepared)%70==0:print({'phase':role,'trials_prepared':len(self.prepared),'elapsed_seconds':round(time.monotonic()-self.started,1)},flush=True)
        for person in sorted({p for p,v in pairs if self.decisions[p,v]['eeg_available']}):
            if digest(RAW/person.replace('sub-','sub')/'data.bdf')!=self.authenticated[person]['sha256']:
                raise ValueError('Original changed after selected reads')

    def source_gate(self):
        records=[self.records[p] for p in self.source]
        self.gate=require_source_quality(records,lambda p:load(PRIVATE/f'{p[0]}_clip{p[1]:02d}_quality.json'),self.source)
        if not self.gate['passed'] or len([r for r in self.prepared if r['role']=='source'])!=980:
            raise PermissionError('Complete source representation/QC gate required')
        save(PUBLIC/'source_gate.json',{**self.gate,'representations_prepared':980,'fits':0,'validation_signals_decoded':0,'new_rating_decoding':0})

    def open_cached_targets(self):
        if not self.gate or not self.gate['passed'] or not (PUBLIC/'source_gate.json').exists():raise PermissionError('Persisted source gate required')
        permitted=set(self.source+self.validation)
        with (QUALITY/'predictions.csv').open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                pair=row['subject'],int(row['vid'])
                if pair not in permitted:raise PermissionError('Protected cached target identity')
                target=float(row['target'])
                if not np.isfinite(target) or not 0<=target<=7:raise ValueError('Invalid cached target')
                if pair in self.targets and self.targets[pair]!=target:raise ValueError('Repeated cached targets differ')
                self.targets[pair]=target
                row['rotation']=int(row['rotation']);row['vid']=pair[1];row['target']=target;row['prediction']=float(row['prediction'])
                row['alpha']=float(row['alpha']) if row['alpha'] else None
                row['oof_fold']=int(row['oof_fold']) if row['oof_fold'] else None
                self.baselines.append(row)
        exact_pairs(list(self.targets),list(permitted))
        save(PUBLIC/'target_reuse.json',{'cached_targets':1260,'new_original_rating_decoding':0,
            'source_prediction_file_sha256':digest(QUALITY/'predictions.csv'),'source_gate_sha256':digest(PUBLIC/'source_gate.json')})

    def retain_fit(self,k,name,fold,key,fitted,capacity=False,permuted=False):
        self.budget();i=len(self.fit_records);path=PRIVATE/f'model_{i:03d}.npz'
        if i>=138:raise ValueError('Fit cap reached')
        with path.open('xb') as stream:
            np.savez_compressed(stream,mean=fitted['scaler'].mean_,scale=fitted['scaler'].scale_,variance=fitted['scaler'].var_,
                block_weights=fitted['block_weights'],source_z=fitted['source_z'],dual=fitted['dual'],source_y=fitted['source_y'])
        self.fit_records.append({'index':i,'rotation':k,'model':name,'oof_fold':fold,'source_set':key,
            'source_identity_sha256':fitted['source_identity_sha256'],'source_rows':len(fitted['source_pairs']),
            'kind':fitted['kind'],'blocks':fitted['blocks'],'alpha':fitted['alpha'],'gamma':fitted['gamma'],
            'median_source_squared_distance':fitted['median_source_squared_distance'],'target_mean':fitted['target_mean'],
            'training_mae':fitted['training_mae'],'training_raw_mae':fitted['training_raw_mae'],
            'normal_equation_residual':fitted['normal_equation_residual'],'private_state_sha256':digest(path),
            'capacity_only':capacity,'permuted_targets':permuted})

    def evaluated_rows(self,k,name,fold,arm,pairs,fitted,context_predictions,median,targets=None):
        available=[p for p in pairs if self.decisions[p]['eeg_available']]
        kind,blocks,alpha=CONDITIONS[name] if name in CONDITIONS else ('rbf',fitted['blocks'],1e-6)
        uses_eeg=blocks!=('context',);joint=len(blocks)==2
        prediction_pairs=available if uses_eeg else pairs
        predicted=dict(zip(prediction_pairs,predict_readout(fitted,prediction_pairs,prediction_pairs,self.features))) if prediction_pairs else {}
        result=[]
        for pair in pairs:
            d=self.decisions[pair]
            family='context_plus_spectral_eeg' if joint else ('spectral_eeg_only' if uses_eeg else 'context_only')
            response=predict_with_quality(d,family,eeg_predictor=lambda p=pair:float(predicted[p]),
                context_predictor=lambda p=pair:float(context_predictions[p] if joint else predicted[p]),source_median=median)
            result.append({'rotation':k,'model':name,'arm':arm,'oof_fold':fold,'subject':pair[0],'vid':pair[1],
                'family':self.metadata[pair]['family'],'acquisition_group':self.metadata[pair]['acquisition_group'],
                'target':float((targets or self.targets)[pair]),**response,'eeg_available':d['eeg_available']})
        return result

    def fit_all(self):
        if not self.gate or not self.gate['passed']:raise PermissionError('Source gate required before fitting')
        original=load(QUALITY/'comparison.json')
        old_index={(r['rotation'],r['model'],r['arm'],r['alpha'],r['oof_fold'],r['subject'],r['vid']):r['prediction'] for r in self.baselines}
        for rotation in self.programme['rotation_manifests']:
            k=rotation['rotation'];full=[tuple(p) for p in self.plan['source_sets'][f'r{k}/full']]
            for fold in [None,0,1,2,3,4]:
                key=f'r{k}/full' if fold is None else f'r{k}/fold{fold}'
                train=[tuple(p) for p in self.plan['source_sets'][key]]
                if fold is None:
                    arms={a:[(r['subject'],r['vid']) for r in rotation['arms'][a]] for a in ('source_fit','validation_unseen','validation_familiar')}
                else:arms={'source_oof':[p for p in full if p not in set(train)]}
                context=fit_readout(train,train,self.features,self.targets,'rbf',('context',),.1)
                self.retain_fit(k,'context_kernel',fold,key,context)
                context_predictions={a:dict(zip(pairs,predict_readout(context,pairs,pairs,self.features))) for a,pairs in arms.items()}
                median=float(np.median([self.targets[p] for p in train]))
                for name,(kind,blocks,alpha) in CONDITIONS.items():
                    self.budget()
                    fitted=context if name=='context_kernel' else fit_readout(train,train,self.features,self.targets,kind,blocks,alpha)
                    if name!='context_kernel':self.retain_fit(k,name,fold,key,fitted)
                    for arm,pairs in arms.items():
                        if kind=='linear':
                            context_alpha=original['rotations'][k]['selected']['context_only']['alpha']
                            contexts={p:old_index[k,'context_only',arm,context_alpha,fold,*p] for p in pairs if not self.decisions[p]['eeg_available']}
                        else:contexts=context_predictions[arm]
                        self.predictions.extend(self.evaluated_rows(k,name,fold,arm,pairs,fitted,contexts,median))
                print({'rotation':k,'source_fold':fold,'normal_fits_complete':len(self.fit_records),'elapsed_seconds':round(time.monotonic()-self.started,1)},flush=True)
            permutation=np.random.default_rng(445401+k).permutation(len(full))
            true=np.asarray([self.targets[p] for p in full]);permuted=dict(zip(full,true[permutation]))
            for representation in ('spectral','covariance'):
                for shuffled in (False,True):
                    self.budget();targets=permuted if shuffled else self.targets
                    name=f'capacity_{representation}_'+('permuted' if shuffled else 'real')
                    fitted=fit_readout(full,full,self.features,targets,'rbf',(representation,),1e-6)
                    self.retain_fit(k,name,None,f'r{k}/full',fitted,capacity=True,permuted=shuffled)
                    self.predictions.extend(self.evaluated_rows(k,name,None,'capacity_source_fit',full,fitted,{},float(np.median(true)),targets))
        if len(self.fit_records)!=138 or len(self.predictions)!=36120:raise ValueError('Final fit/prediction coverage differs')

    def compare(self):
        previous=load(QUALITY/'comparison.json');tables=[]
        for k in range(3):
            by_model={}
            for name,(kind,blocks,alpha) in CONDITIONS.items():
                rows=[r for r in self.predictions if r['rotation']==k and r['model']==name]
                metrics={arm:independent_metrics([r for r in rows if r['arm']==arm]) for arm in ('source_fit','source_oof','validation_unseen','validation_familiar')}
                entry={'metrics':metrics,'alpha':alpha,'kind':kind,'blocks':blocks}
                if len(blocks)==2:
                    context_name='context_kernel' if kind=='rbf' else 'context_only'
                    if kind=='rbf':context_rows=[r for r in self.predictions if r['rotation']==k and r['model']=='context_kernel' and r['arm']=='validation_unseen']
                    else:
                        selected=previous['rotations'][k]['selected']['context_only']['alpha']
                        context_rows=[r for r in self.baselines if r['rotation']==k and r['model']=='context_only' and r['arm']=='validation_unseen' and r['alpha']==selected]
                    joint_rows=[r for r in rows if r['arm']=='validation_unseen']
                    entry['matched_context']=context_name
                    entry['paired']=paired_diagnostics(context_rows,joint_rows,4000,425401)
                    entry['paired']['limits']='Exploratory: reused validation and six films; new heads fixed before this execution, original Ridge baseline previously development-selected; no independent confirmation or tuning-corrected uncertainty'
                    entry['gain_over_original_context_ridge']=previous['rotations'][k]['selected']['context_only']['validation_unseen']['participant_film_mae']-metrics['validation_unseen']['participant_film_mae']
                    prior=previous['rotations'][k]['strongest_fixed_prior_unchanged']
                    entry['fixed_prior']=prior
                    entry['gain_over_fixed_prior']=previous['rotations'][k]['priors']['validation_unseen'][prior]['participant_film_mae']-metrics['validation_unseen']['participant_film_mae']
                by_model[name]=entry
            tables.append({'rotation':k,'models':by_model})
        leads={}
        for name in ('context_covariance_linear','context_spectral_kernel','context_covariance_kernel'):
            entries=[r['models'][name] for r in tables]
            gains=[r['paired']['gain'] for r in entries]
            mean=float(np.mean(gains));ridge=float(np.mean([r['gain_over_original_context_ridge'] for r in entries]));prior=float(np.mean([r['gain_over_fixed_prior'] for r in entries]))
            leads[name]={'mean_gain_over_matched_context':mean,'mean_gain_over_original_context_ridge':ridge,
                'mean_gain_over_fixed_prior':prior,'positive_rotations':sum(g>0 for g in gains),
                'numerical_screen_passed':mean>=.15 and ridge>=.15 and sum(g>0 for g in gains)>=2 and prior>0,
                'requires_source_capacity_and_coverage_review':True,'independent_confirmation':False}
        capacity=[{k:r[k] for k in ('rotation','model','alpha','training_mae','training_raw_mae','normal_equation_residual','permuted_targets')} for r in self.fit_records if r['capacity_only']]
        return {'rotations':tables,'lead_screens':leads,'capacity_diagnostics':capacity,
            'normal_fits':126,'capacity_fits':12,'new_fits':138,'prediction_rows':36120,
            'source_trials':980,'validation_trials':280,'validation_eeg_available':265,'validation_eeg_unavailable':15,
            'new_original_rating_decoding':0,'no_validation_hyperparameter_selection':True,
            'reused_development':True,'confirmation_access':False,'research_question_changed':False}


def execute():
    p,programme,records,source,validation=check()
    if (PUBLIC/'execution_started.json').exists():raise ValueError('Preserve earlier execution; no implicit resume')
    save(PUBLIC/'execution_started.json',{'plan_sha256':digest(PUBLIC/'plan.json'),'fits':0,'user_authorized':True})
    study=Study(p,programme,records,source,validation);failure=None;status='running'
    try:
        with threadpool_limits(limits=2):
            study.prepare(source,'source');study.source_gate();study.prepare(validation,'validation')
            study.open_cached_targets();study.fit_all();study.budget()
            save(PUBLIC/'comparison.json',study.compare());status='completed_bounded_study'
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)};status='failed_or_truncated'
    finally:
        save(PUBLIC/'feature_manifest.json',{'records':study.prepared})
        save(PUBLIC/'model_manifest.json',{'records':study.fit_records})
        if study.predictions:
            with (PUBLIC/'predictions.csv').open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(study.predictions[0]));writer.writeheader();writer.writerows(study.predictions)
        save(PUBLIC/'execution.json',{'status':status,'failure':failure,'plan_sha256':digest(PUBLIC/'plan.json'),
            'source_gate':study.gate,'source_trials_prepared':sum(r['role']=='source' for r in study.prepared),
            'validation_trials_prepared':sum(r['role']=='validation' for r in study.prepared),
            'new_fits':len(study.fit_records),'prediction_rows':len(study.predictions),
            'selected_sample_values_decoded':int(study.values_decoded),'new_original_rating_decoding':0,
            'raw_waveforms_persisted':0,'auxiliary_neighbor_or_confirmation_samples_decoded':0,
            'elapsed_seconds':round(time.monotonic()-study.started,3),'confirmation_access':False,'research_question_changed':False})
        print(load(PUBLIC/'execution.json'),flush=True)
    if failure:raise RuntimeError('Bounded experiment failed; immutable receipt retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','execute'))
    {'declare':declare,'execute':execute}[parser.parse_args().mode]()
