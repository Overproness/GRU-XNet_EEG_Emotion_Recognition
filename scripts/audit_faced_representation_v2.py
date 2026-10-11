"""Independent FACED feature, source-fit, state-prediction and outcome audit.

Does not call new representation/readout/quality/prediction functions or raw readers.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import logm
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_artifact_control import load,save,digest,array_digest,BASE as OLD,PRIVATE as OLD_PRIVATE,PROGRAMME
from scripts.audit_faced_artifact_controls import independent_metrics
PUBLIC=REPO/'results/development/faced_representation_2026-10-11'
PRIVATE=REPO.parent/'publication_runs/faced_representation_2026-10-11'
QUALITY=REPO/'results/development/faced_inference_quality_2026-10-11'


def read_rows(path):
    with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for field in ('vid','rotation'):row[field]=int(row[field])
        row['oof_fold']=int(row['oof_fold']) if row['oof_fold'] else None
        for field in ('target','prediction'):row[field]=float(row[field])
    return rows


def compare_numbers(actual,expected,tolerance=1e-10):
    if set(actual)!=set(expected):raise ValueError('Metric keys differ')
    for key in actual:
        if abs(actual[key]-expected[key])>tolerance:raise ValueError('Metric differs: '+key)


def distance_squared(a,b):
    return np.maximum(np.sum(a*a,axis=1)[:,None]+np.sum(b*b,axis=1)[None,:]-2*a@b.T,0)


def median_nonduplicate_squared_distance(z):
    # Norm expansion can leave a tiny positive value for identical rows.
    # Exact row equality, not a tuned tolerance, determines duplicate pairs.
    distances=distance_squared(z,z)
    upper=np.triu_indices(len(z),1)
    _, groups=np.unique(z,axis=0,return_inverse=True)
    positive=distances[upper][groups[upper[0]]!=groups[upper[1]]]
    if not positive.size or np.any(positive<=0):
        raise ValueError('No valid positive source distances')
    return float(np.median(positive))


def audit():
    plan=load(PUBLIC/'plan.json');execution=load(PUBLIC/'execution.json')
    if execution['status']!='completed_bounded_study' or execution['new_fits']!=138:raise PermissionError('Completed bounded experiment required')
    for rel,expected in plan['bindings'].items():
        if digest(REPO/rel)!=expected:raise ValueError('Frozen dependency changed: '+rel)
    if digest(PUBLIC/'plan.json')!=execution['plan_sha256']:raise ValueError('Plan changed during execution')
    if execution['elapsed_seconds']>7200 or not load(PUBLIC/'source_gate.json')['passed']:raise ValueError('Budget/source gate failed')
    programme=load(PROGRAMME);features={b:{} for b in ('context','spectral','covariance')}
    expected_sets={}
    for rotation in programme['rotation_manifests']:
        k=rotation['rotation'];full=[(r['subject'],r['vid']) for r in rotation['arms']['source_fit']]
        expected_sets[f'r{k}/full']=[list(p) for p in full]
        for fold,people in enumerate(programme['source_participant_oof_folds']):
            expected_sets[f'r{k}/fold{fold}']=[list(p) for p in full if p[0] not in people]
    if plan['source_sets']!=expected_sets:raise PermissionError('Fit source roles differ from reserved programme')
    decisions={(r['subject'],r['vid']):r['eeg_available'] for r in load(QUALITY/'signal_audit.json')['trial_decisions']}
    originals={(r['subject'],r['vid']):r for name in ('source_records.json','validation_records.json') for r in load(OLD/name)['records']}
    records=load(PUBLIC/'feature_manifest.json')['records']
    if len(records)!=1260 or {(r['subject'],r['vid']) for r in records}!=set(originals):raise ValueError('Feature coverage differs')
    catalogue={r['clip']:r for r in load(REPO/'results/development/faced_codebooks_2026-10-10/audit.json')['stimuli']}
    categories=['Amusement','Anger','Disgust','Fear','Inspiration','Joy','Sadness','Tenderness']
    acquisitions=['legacy_1000','legacy_250','modern_1000']
    upper=np.triu_indices(30);weights=np.where(upper[0]==upper[1],1,np.sqrt(2))
    max_feature_error=0.;total_counts=0;cache_bindings={}
    for r in records:
        pair=r['subject'],r['vid'];old=originals[pair];c=catalogue[pair[1]]
        if r['eeg_available']!=decisions[pair] or r['role']!=old['role']:raise ValueError('Feature role/availability differs')
        context=np.asarray([float(c['assigned_category']==a) for a in categories]+[float(old['acquisition_group']==a) for a in acquisitions]+[float(c['catalogue_duration_seconds']),float(old['trial'])])
        if array_digest(context)!=old['context_feature_sha256']:raise ValueError('Context mapping differs')
        features['context'][pair]=context
        if not r['eeg_available']:
            if r['samples_decoded']!=0 or r['original_signal_or_eeg_features_opened']:raise ValueError('Unavailable EEG was accessed')
            if r['role']!='validation':raise PermissionError('Unavailable source admitted')
            continue
        stem=f'{pair[0]}_clip{pair[1]:02d}'
        for suffix,key in (('_features.npz','private_features_sha256'),('_access.json','private_access_sha256'),('_quality.json','private_quality_sha256')):
            path=PRIVATE/(stem+suffix)
            if digest(path)!=r[key]:raise ValueError('Private representation evidence changed')
            cache_bindings[path.name]=r[key]
        access=load(PRIVATE/(stem+'_access.json'))
        if access['range_ledger_sha256']!=old['range_ledger_sha256'] or not access['all_counts_scalar_reference_checked'] or access['extra_boundary_samples_decoded']!=0:
            raise ValueError('Selective access ledger differs')
        q=load(PRIVATE/(stem+'_quality.json'))
        for channel in q['channels']:
            if (channel['raw_rail_fraction']>.0001 or channel['raw_identical_adjacent_fraction']>.95 or
                channel['processed_sd_uv']<.1 or channel['processed_sd_uv']>100 or
                channel['processed_peak_to_peak_uv']>1000 or channel['processed_49_51_to_1_45_power_ratio']>.10):
                raise ValueError('Signal metric violates frozen quality')
        with np.load(PRIVATE/(stem+'_features.npz'),allow_pickle=False) as data:
            matrices=data['band_covariances'];covariance=data['covariance'];spectral=data['spectral']
            if matrices.shape!=(5,30,30) or covariance.shape!=(2330,):raise ValueError('Representation geometry differs')
            vectors=[];traces=[]
            for matrix in matrices:
                np.testing.assert_allclose(matrix,matrix.T,atol=1e-13)
                trace=float(np.trace(matrix));normalized=.9*matrix/trace+np.eye(30)*(.1/30)
                logged=logm(normalized)
                if np.max(np.abs(logged.imag))>1e-10 or np.linalg.eigvalsh(normalized).min()<.1/30-1e-10:
                    raise ValueError('Invalid independently recomputed covariance logarithm')
                vectors.append(logged.real[upper]*weights);traces.append(trace)
            reference=np.concatenate(vectors+[np.log10(traces)])
            error=float(np.max(np.abs(covariance-reference)));max_feature_error=max(max_feature_error,error)
            if error>1e-10:raise ValueError('Independent log-covariance differs')
            np.testing.assert_array_equal(data['context'],context)
            with np.load(OLD_PRIVATE/(stem+'_features.npz'),allow_pickle=False) as previous:
                np.testing.assert_array_equal(previous['spectral'],spectral)
            features['spectral'][pair]=spectral.copy();features['covariance'][pair]=covariance.copy()
        total_counts+=r['samples_decoded']
    if total_counts!=execution['selected_sample_values_decoded'] or sum(r['eeg_available'] for r in records)!=1245:
        raise ValueError('Selected count/available coverage differs')
    old_rows=read_rows(QUALITY/'predictions.csv');targets={}
    for row in old_rows:
        pair=row['subject'],row['vid']
        if pair not in originals or (pair in targets and targets[pair]!=row['target']):raise ValueError('Cached targets differ')
        targets[pair]=row['target']
    if len(targets)!=1260:raise ValueError('Target rectangle differs')
    previous=load(QUALITY/'comparison.json')
    old_context={}
    for row in old_rows:
        k=row['rotation'];alpha=previous['rotations'][k]['selected']['context_only']['alpha']
        if row['model']=='context_only' and float(row['alpha'])==alpha:
            old_context[k,row['arm'],row['oof_fold'],row['subject'],row['vid']]=row['prediction']
    rows=read_rows(PUBLIC/'predictions.csv');index={}
    for row in rows:
        key=row['rotation'],row['model'],row['oof_fold'],row['arm'],row['subject'],row['vid']
        if key in index:raise ValueError('Duplicated new prediction')
        index[key]=row
    models=load(PUBLIC/'model_manifest.json')['records']
    if len(models)!=138 or len(rows)!=36120:raise ValueError('Model/prediction coverage differs')
    expected_jobs={(k,name,fold) for k in range(3) for name in plan['conditions'] for fold in (None,0,1,2,3,4)}
    expected_jobs.update((k,f'capacity_{b}_{target}',None) for k in range(3) for b in ('spectral','covariance') for target in ('real','permuted'))
    if {(r['rotation'],r['model'],r['oof_fold']) for r in models}!=expected_jobs:
        raise ValueError('Declared fit jobs differ')
    fitted={};max_equation_error=0.;max_scale_error=0.
    for model in models:
        i=model['index'];k=model['rotation'];fold=model['oof_fold'];name=model['model']
        if name in plan['conditions']:
            kind,blocks,alpha=plan['conditions'][name]
            if model['capacity_only'] or model['permuted_targets']:raise ValueError('Normal fit marked as capacity')
        else:
            kind='rbf';blocks=['spectral' if '_spectral_' in name else 'covariance'];alpha=1e-6
            if not model['capacity_only'] or model['permuted_targets']!=name.endswith('_permuted'):
                raise ValueError('Capacity job/target handling differs')
        if (model['kind'],model['blocks'],model['alpha'])!=(kind,blocks,alpha):raise ValueError('Readout grid changed')
        expected_key=f'r{k}/full' if fold is None else f'r{k}/fold{fold}'
        if model['source_set']!=expected_key:raise PermissionError('Wrong source fold')
        train=[tuple(p) for p in plan['source_sets'][model['source_set']]]
        identity=hashlib.sha256(json.dumps(train,separators=(',',':')).encode()).hexdigest()
        if identity!=model['source_identity_sha256']:raise ValueError('Fit identity hash differs')
        x=np.concatenate([np.stack([features[b][p] for p in train]) for b in model['blocks']],axis=1)
        expected_y=np.asarray([targets[p] for p in train])
        if model['permuted_targets']:expected_y=expected_y[np.random.default_rng(445401+k).permutation(len(train))]
        mean=x.mean(axis=0);variance=np.mean((x-mean)**2,axis=0)
        bound=len(train)*np.finfo(float).eps*variance+(len(train)*mean*np.finfo(float).eps)**2
        scale=np.sqrt(variance);scale[variance<=bound]=1.
        block_weights=np.concatenate([np.full(features[b][train[0]].size,1/np.sqrt(features[b][train[0]].size) if model['kind']=='rbf' else 1.) for b in model['blocks']])
        state_path=PRIVATE/f'model_{i:03d}.npz'
        if digest(state_path)!=model['private_state_sha256']:raise ValueError('Private model bytes changed')
        cache_bindings[state_path.name]=model['private_state_sha256']
        with np.load(state_path,allow_pickle=False) as state:
            for field,reference in (('mean',mean),('variance',variance),('scale',scale)):
                error=float(np.max(np.abs(state[field]-reference)));max_scale_error=max(max_scale_error,error)
                np.testing.assert_allclose(state[field],reference,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(state['source_y'],expected_y)
            np.testing.assert_array_equal(state['block_weights'],block_weights)
            z=(x-state['mean'])/state['scale']*block_weights
            np.testing.assert_allclose(z,state['source_z'],atol=1e-12)
            gamma=model['gamma']
            if model['kind']=='rbf':
                distances=distance_squared(z,z)
                if abs(1/median_nonduplicate_squared_distance(z)-gamma)>1e-10:raise ValueError('Bandwidth uses different source statistics')
                kernel=np.exp(-gamma*distances)
            else:kernel=z@z.T
            dual=state['dual'].copy();offset=float(expected_y.mean())
            if offset!=model['target_mean']:raise ValueError('Source target offset differs')
            residual=float(np.max(np.abs((kernel+model['alpha']*np.eye(len(train)))@dual-(expected_y-offset))))
            max_equation_error=max(max_equation_error,residual)
            if residual>1e-6:raise ValueError('Independent source equations fail')
            raw=offset+kernel@dual
            if abs(float(np.mean(abs(raw-expected_y)))-model['training_raw_mae'])>1e-7:raise ValueError('Source capacity error differs')
            fitted[k,name,fold]={'model':model,'train':train,'mean':state['mean'].copy(),'scale':state['scale'].copy(),
                                'weights':block_weights,'z':z,'dual':dual,'gamma':gamma,'offset':offset,'source_y':expected_y}
    def predict(state,pair):
        x=np.concatenate([features[b][pair] for b in state['model']['blocks']])
        z=((x-state['mean'])/state['scale']*state['weights'])[None,:]
        kernel=np.exp(-state['gamma']*distance_squared(z,state['z'])) if state['model']['kind']=='rbf' else z@state['z'].T
        return float(np.clip(state['offset']+kernel@state['dual'],0,7)[0])
    verified=set();max_prediction_error=0.;fallbacks=0
    for state in fitted.values():
        model=state['model'];k=model['rotation'];name=model['model'];fold=model['oof_fold']
        full=[tuple(p) for p in plan['source_sets'][f'r{k}/full']]
        if model['capacity_only']:arms={'capacity_source_fit':full}
        elif fold is not None:arms={'source_oof':[p for p in full if p not in set(state['train'])]}
        else:arms={a:[(r['subject'],r['vid']) for r in programme['rotation_manifests'][k]['arms'][a]] for a in ('source_fit','validation_unseen','validation_familiar')}
        source_target=dict(zip(state['train'],state['source_y']))
        for arm,pairs in arms.items():
            for pair in pairs:
                key=k,name,fold,arm,*pair
                if key not in index:raise ValueError('Missing declared job/trial prediction')
                row=index[key];verified.add(key)
                target=source_target[pair] if model['capacity_only'] else targets[pair]
                if target!=row['target'] or row['family']!=catalogue[pair[1]]['source_film_sha256']:raise ValueError('Prediction target/material differs')
                uses_eeg=model['blocks']!=['context'];joint=len(model['blocks'])==2
                if row['eeg_available']!=str(decisions[pair]):raise ValueError('Availability differs')
                fallback=uses_eeg and not decisions[pair]
                if not fallback:
                    expected=predict(state,pair);branch='context_model' if not uses_eeg else ('joint_model' if joint else 'eeg_model')
                elif joint:
                    expected=(predict(fitted[k,'context_kernel',fold],pair) if model['kind']=='rbf' else old_context[k,arm,fold,*pair]);branch='context_fallback'
                else:expected=float(np.median(state['source_y']));branch='source_global_median_fallback'
                if row['fallback']!=str(fallback) or row['eeg_used']!=str(uses_eeg and not fallback) or row['branch']!=branch:
                    raise ValueError('Inference branch differs')
                error=abs(row['prediction']-expected);max_prediction_error=max(max_prediction_error,error)
                if error>1e-7:raise ValueError('Independent state/fallback prediction differs')
                fallbacks+=int(fallback)
    if verified!=set(index) or fallbacks!=270:raise ValueError('Extra prediction or fallback coverage differs')
    comparison=load(PUBLIC/'comparison.json');max_metric_error=0.;lead_screens={}
    for table in comparison['rotations']:
        k=table['rotation']
        for name,result in table['models'].items():
            for arm,reported in result['metrics'].items():
                selected=[r for r in rows if r['rotation']==k and r['model']==name and r['arm']==arm]
                actual=independent_metrics(selected);compare_numbers(reported,actual)
                max_metric_error=max(max_metric_error,max(abs(reported[key]-actual[key]) for key in actual))
            if 'paired' in result:
                joint=[r for r in rows if r['rotation']==k and r['model']==name and r['arm']=='validation_unseen']
                if result['kind']=='rbf':context=[r for r in rows if r['rotation']==k and r['model']=='context_kernel' and r['arm']=='validation_unseen']
                else:context=[r for r in old_rows if r['rotation']==k and r['model']=='context_only' and r['arm']=='validation_unseen' and float(r['alpha'])==previous['rotations'][k]['selected']['context_only']['alpha']]
                mapping={(r['subject'],r['family']):abs(r['target']-r['prediction']) for r in context}
                people=sorted({r['subject'] for r in joint});films=sorted({r['family'] for r in joint})
                values={(r['subject'],r['family']):mapping[r['subject'],r['family']]-abs(r['target']-r['prediction']) for r in joint}
                matrix=np.asarray([[values[p,f] for f in films] for p in people])
                rng=np.random.default_rng(425401);ps=rng.integers(20,size=(4000,20));fs=rng.integers(6,size=(4000,6))
                draws=np.asarray([matrix[p[:,None],f[None,:]].mean() for p,f in zip(ps,fs)])
                np.testing.assert_allclose(np.quantile(draws,[.025,.975]),result['paired']['descriptive_95_percent_interval'],atol=1e-12)
                if abs(float(matrix.mean())-result['paired']['gain'])>1e-12:raise ValueError('Paired gain differs')
                for i,film in enumerate(films):
                    if abs(np.delete(matrix,i,axis=1).mean()-result['paired']['leave_one_film_out_gain'][film])>1e-12:raise ValueError('Leave-film gain differs')
                for group,group_metrics in result['paired']['acquisition_metrics'].items():
                    compare_numbers(group_metrics['context'],independent_metrics([r for r in context if r['acquisition_group']==group]))
                    compare_numbers(group_metrics['joint'],independent_metrics([r for r in joint if r['acquisition_group']==group]))
                joint_mae=result['metrics']['validation_unseen']['participant_film_mae']
                ridge=previous['rotations'][k]['selected']['context_only']['validation_unseen']['participant_film_mae']-joint_mae
                prior=previous['rotations'][k]['priors']['validation_unseen'][result['fixed_prior']]['participant_film_mae']-joint_mae
                if abs(ridge-result['gain_over_original_context_ridge'])>1e-12 or abs(prior-result['gain_over_fixed_prior'])>1e-12:raise ValueError('Fixed baseline gain differs')
    for name,screen in comparison['lead_screens'].items():
        entries=[t['models'][name] for t in comparison['rotations']];gains=[e['paired']['gain'] for e in entries]
        means=[float(np.mean([e[key] for e in entries])) for key in ('gain_over_original_context_ridge','gain_over_fixed_prior')]
        expected=np.mean(gains)>=.15 and means[0]>=.15 and sum(g>0 for g in gains)>=2 and means[1]>0
        if bool(expected)!=screen['numerical_screen_passed']:raise ValueError('Continuation screen differs')
        if abs(np.mean(gains)-screen['mean_gain_over_matched_context'])>1e-12 or abs(means[0]-screen['mean_gain_over_original_context_ridge'])>1e-12 or abs(means[1]-screen['mean_gain_over_fixed_prior'])>1e-12:
            raise ValueError('Mean lead-screen values differ')
        lead_screens[name]=screen
    expected_capacity=[{key:r[key] for key in ('rotation','model','alpha','training_mae','training_raw_mae','normal_equation_residual','permuted_targets')} for r in models if r['capacity_only']]
    if comparison['capacity_diagnostics']!=expected_capacity:raise ValueError('Capacity report differs from fitted states')
    summary={'feature_trials_verified':1245,'unavailable_trials_bypassed':15,'models_verified':138,
        'prediction_rows_verified':36120,'fallback_rows_verified':270,'private_files_verified':len(cache_bindings),
        'maximum_feature_discrepancy':max_feature_error,'maximum_source_statistic_discrepancy':max_scale_error,
        'maximum_independent_equation_residual':max_equation_error,'maximum_prediction_discrepancy':max_prediction_error,
        'maximum_metric_discrepancy':max_metric_error,'lead_screens':lead_screens,
        'new_audit_fits':0,'original_data_files_opened':False,'new_original_rating_decoding':0,
        'new_representation_fit_quality_functions_called':False,'confirmation_access':False,'research_question_changed':False}
    out=PUBLIC/'independent_analysis';save(out/'summary.json',summary)
    receipt_paths=[Path(__file__)]+[PUBLIC/n for n in ('plan.json','execution.json','source_gate.json','target_reuse.json','feature_manifest.json','model_manifest.json','predictions.csv','comparison.json')]+[out/'summary.json']
    save(out/'verification.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in receipt_paths},
        'private_cache_bindings':cache_bindings,'independent_audit_passed':True})
    print({k:summary[k] for k in ('models_verified','prediction_rows_verified','maximum_prediction_discrepancy','lead_screens')},flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):audit()
