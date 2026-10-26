"""New declared cache-only source experiment with shared missing-EEG handling.

Preserve the preceding strict QC hold. Interpret only available source features
after freeze, then source cached targets only after independent qualification.
No raw/original rating reader, validation numeric input or protected access.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_coverage_selection_2026_10_13 import (
    metadata as original_metadata, now, runtime, write, sha, array_sha, check_bindings)
from scripts.audit_faced_coverage_source_qc_2026_10_13 import qualify_public_rows, preservation
from gruxnet.faced_coverage_selection import crossed_source_cells, MODELS, ALPHAS
from gruxnet.faced_coverage_missing import shared_availability, run_source_cv, fit_full_source
from gruxnet.faced_temporal_coverage import REGIONS
from gruxnet.faced_classical_contract import context_features

BASE = REPO/'results/development'
PRIOR = BASE/'faced_coverage_selection_2026-10-13'
SOURCE_PRIVATE = REPO.parent/'publication_runs/faced_coverage_selection_2026-10-13'
PUBLIC = BASE/'faced_coverage_missing_2026-10-13'
PRIVATE = REPO.parent/'publication_runs/faced_coverage_missing_2026-10-13'
ACTIVATION = PUBLIC/'activation.json'
PLAN = BASE/'faced_coverage_selection_preflight_2026-10-13/plan.json'
TARGETS = BASE/'faced_inference_quality_2026-10-11/predictions.csv'
PROTOCOL = REPO/'docs/publication/FACED_Coverage_Missing_EEG_Protocol_2026-10-13.md'
SNAPSHOT = PRIVATE/'available_source_vectors.npz'
CAP_SECONDS = 600
POLICY = 'faced-shared-three-region-training-mean-v1'
FIT_OUTPUTS = ('fitting_started.json','fitting_failure.json','cached_source_targets_receipt.json',
    'fit_records.jsonl','cv_predictions.json','comparison.json','final_source_state_qualification.json')

def identity_sha(pairs):
    import hashlib
    return hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest()

def source_evidence():
    previous,_,meta,pairs = original_metadata()
    completion = json.loads((PRIOR/'completion.json').read_bytes())
    for name,digest in completion['public_bindings'].items():
        if sha(PRIOR/name) != digest:
            raise ValueError('Preserved strict attempt changed: '+name)
    held = json.loads((BASE/'faced_coverage_confirmation_proposal_2026-10-13/proposal.json').read_bytes())
    check_bindings(held['bindings'])
    gate = json.loads((PRIOR/'source_gate.json').read_bytes())
    audit = json.loads((PRIOR/'source_qc_independent_audit.json').read_bytes())
    if (completion['status'] != 'held_no_source_finalist' or gate['passed'] is not False
            or gate['cached_targets_converted'] != 0 or gate['new_fits'] != 0
            or audit['passed'] is not True or audit['source_quality_gate_passed'] is not False
            or audit['source_gate_sha256'] != sha(PRIOR/'source_gate.json')
            or audit['source_regions_sha256'] != sha(PRIOR/'source_regions.jsonl')
            or gate['activation_sha256'] != sha(PRIOR/'activation_v2.json')
            or any((PRIOR/n).exists() for n in FIT_OUTPUTS)):
        raise PermissionError('Authenticated preserved source-only QC hold required')
    rows = [json.loads(x) for x in (PRIOR/'source_regions.jsonl').read_text().splitlines()]
    qualify_public_rows(rows,previous)
    for row in rows:
        if row['eeg_available'] is not (row['reasons'] == []):
            raise PermissionError('Engineered availability/reason decision changed')
    quality = [{**r,'technical_passed':True,'review_flags':r['reasons']} for r in rows]
    available,quality_gate = shared_availability(pairs,meta,quality)
    folds = json.loads((BASE/'faced_development_protocol_2026-10-11/plan.json').read_bytes())['source_participant_oof_folds']
    cells = crossed_source_cells(pairs,meta,folds)
    prior_plan = json.loads(PLAN.read_bytes()); check_bindings(prior_plan['bindings'])
    if json.dumps(cells,sort_keys=True) != json.dumps(prior_plan['cells'],sort_keys=True):
        raise ValueError('Crossed source selection geometry changed')
    if len(pairs) != 980 or len(rows) != 2940 or sum(available.values()) != 979:
        raise PermissionError('Declared 979 available/one shared-missing source geometry changed')
    for cell in cells:
        if not any(available[tuple(p)] for p in cell['training']):
            raise PermissionError('No eligible training EEG donor')
    preservation()
    return meta,pairs,rows,quality,available,quality_gate,cells,folds

def dependencies():
    files = {PROTOCOL,PLAN,TARGETS,REPO/'scripts/run_faced_coverage_missing_2026_10_13.py',
        REPO/'tests/test_faced_coverage_missing_execution.py',REPO/'gruxnet/faced_coverage_missing.py',
        REPO/'tests/test_faced_coverage_missing.py',
        REPO/'gruxnet/faced_eegnet_bridge.py',
        REPO/'scripts/audit_faced_coverage_missing_results_2026_10_13.py',
        REPO/'scripts/audit_faced_coverage_missing_qualification_2026_10_13.py'}
    for packet in ('activation_v2.json','completion.json'):
        value = json.loads((PRIOR/packet).read_bytes()); files.add(PRIOR/packet)
        files.update(REPO/n for n in value.get('bindings',{}))
        files.update(PRIOR/n for n in value.get('public_bindings',{}))
    held = BASE/'faced_coverage_confirmation_proposal_2026-10-13/proposal.json'
    files.add(held); files.update(REPO/n for n in json.loads(held.read_bytes())['bindings'])
    files.add(REPO/'scripts/audit_faced_coverage_source_qc_2026_10_13.py')
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(files)}

def validate_activation(a):
    fixed = {'status':'active_source_only_shared_missing_eeg','policy_id':POLICY,
        'source_trials':980,'source_regions':2940,'available_trials':979,'shared_missing_trials':1,
        'available_regional_vector_cap':2937,'available_feature_scalar_cap':440550,
        'new_raw_sample_decodes':0,'original_rating_decodes':0,'validation_numeric_access':False,
        'confirmation_access':False,'cv_fit_cap':90,'full_source_fit_cap':3,
        'qualification_wall_seconds_cap':600,'qualification_cpu_seconds_cap':600,
        'fitting_wall_seconds_cap':600,'fitting_cpu_seconds_cap':600,
        'primary_diagnostic_alpha':100.0,'independent_qualification_before_targets':True}
    if any(a.get(k) != v for k,v in fixed.items()) or a.get('runtime') != runtime():
        raise PermissionError('Missing, widened or changed source activation/runtime')
    check_bindings(a['bindings'])

def declare():
    _,pairs,_,_,available,quality_gate,cells,_ = source_evidence()
    missing = [p for p in pairs if not available[p]]
    a = {'date':'2026-10-13','declared_utc':now(),'status':'active_source_only_shared_missing_eeg',
        'policy_id':POLICY,'author_authorization':'User explicitly authorized qualifying shared label-blind missing EEG and running the matched comparison without changing quality thresholds.',
        'source_trials':980,'source_regions':2940,'available_trials':979,'shared_missing_trials':1,
        'shared_missing_pairs':missing,'available_regional_vector_cap':2937,'available_feature_scalar_cap':440550,
        'new_raw_sample_decodes':0,'original_rating_decodes':0,'validation_numeric_access':False,
        'confirmation_access':False,'cv_fit_cap':90,'full_source_fit_cap':3,
        'qualification_wall_seconds_cap':600,'qualification_cpu_seconds_cap':600,
        'fitting_wall_seconds_cap':600,'fitting_cpu_seconds_cap':600,
        'primary_diagnostic_alpha':100.0,'independent_qualification_before_targets':True,
        'alphas':list(ALPHAS),'models':list(MODELS),'runtime':runtime(),'bindings':dependencies(),
        'training_policy':'Retain all original rows/context/targets; each joint missing EEG replaced by its available-training-only mean',
        'held_policy':'All matched predictors share category-median fallback if any region fails QC',
        'quality_thresholds_changed':False,'old_strict_gate_changed':False,'source_rows_excluded':False,
        'availability_feature_added':False,'automatic_retry':False,'research_question_changed':False}
    validate_activation(a); write(ACTIVATION,a)
    write(PUBLIC/'source_policy_plan.json',{'date':'2026-10-13','policy_id':POLICY,
        'old_strict_source_gate_passed':False,'quality_gate':quality_gate,
        'source_identity_sha256':identity_sha(pairs),'available_identity_sha256':identity_sha([p for p in pairs if available[p]]),
        'shared_missing_pairs':missing,'cell_geometry_reference':PLAN.relative_to(REPO).as_posix(),
        'cell_training_missing_counts':[sum(not available[tuple(p)] for p in c['training']) for c in cells],
        'cell_held_missing_counts':[sum(not available[tuple(p)] for p in c['held']) for c in cells],
        'targets_converted':0,'new_fits':0,'confirmation_access':False})
    print(json.dumps({'declared':True,'available_trials':979,'missing_trials':1,'fit_cap':93,'protected_access':False}),flush=True)

def check():
    a = json.loads(ACTIVATION.read_bytes()); validate_activation(a)
    return a

def mean_fixtures(vectors,available_pairs,cells):
    index = {p:i for i,p in enumerate(available_pairs)}
    branches = {'late_joint':vectors[:,2,:], 'distributed_joint':vectors.mean(axis=1)}
    def one(training,cell):
        donors = [tuple(p) for p in training if tuple(p) in index]
        if not donors:
            raise PermissionError('No training donor for feature mean')
        means = {m:v[[index[p] for p in donors]].mean(axis=0) for m,v in branches.items()}
        return {'cell':cell,'eligible_train_donor_pairs':donors,'donor_count':len(donors),
            'eligible_train_donor_identity_sha256':identity_sha(donors),
            'imputed_training_pairs':[tuple(p) for p in training if tuple(p) not in index],
            'mean_sha256':{m:array_sha(v) for m,v in means.items()}}
    return {'cells':[one(c['training'],c['cell']) for c in cells],
        'full_source':one(sorted({tuple(p) for c in cells for p in c['training']}),None)}

def qualify():
    a = check(); _,pairs,rows,_,available,_,cells,_ = source_evidence()
    if (PUBLIC/'qualification_started.json').exists():
        raise PermissionError('Preserve this qualification attempt; no automatic retry')
    start = time.monotonic(); cpu = time.process_time(); vectors = []; loaded = 0
    def budget():
        if time.monotonic()-start > CAP_SECONDS or time.process_time()-cpu > CAP_SECONDS:
            raise TimeoutError('Cooperative source qualification cap')
    write(PUBLIC/'qualification_started.json',{'utc':now(),'activation_sha256':sha(ACTIVATION),
        'targets_converted':0,'fits':0,'raw_samples_decoded':0})
    PRIVATE.mkdir(parents=True,exist_ok=True)
    try:
        row_map = {(r['subject'],r['vid'],r['region']):r for r in rows}
        available_pairs = [p for p in pairs if available[p]]
        root = SOURCE_PRIVATE.resolve()
        for pair in available_pairs:
            regions = []
            for region in REGIONS:
                budget()
                file = (root/f'{pair[0]}_clip{pair[1]:02d}_{region}_features.npz').resolve()
                if not file.is_relative_to(root) or not available[pair]:
                    raise PermissionError('Source feature read outside available allowlist')
                row = row_map[(*pair,region)]
                if sha(file,budget) != row['feature_file_sha256']:
                    raise ValueError('Available original cache hash changed')
                with np.load(file,allow_pickle=False) as bundle:
                    if bundle.files != ['features']:
                        raise ValueError('Unexpected source feature archive keys')
                    values = bundle['features'].copy()
                if values.shape != (150,) or not np.isfinite(values).all() or array_sha(values) != row['spectral_feature_sha256']:
                    raise ValueError('Source feature shape/finite/hash qualification failed')
                loaded += 1
                if loaded > a['available_regional_vector_cap']:
                    raise PermissionError('Source numerical feature cap exceeded')
                regions.append(values)
            vectors.append(np.stack(regions))
        values = np.stack(vectors)
        if values.shape != (979,3,150) or values.size != a['available_feature_scalar_cap']:
            raise ValueError('Available source snapshot geometry differs')
        if SNAPSHOT.exists():
            raise FileExistsError('Preserve source snapshot')
        np.savez_compressed(SNAPSHOT,vectors=values)
        fixtures = mean_fixtures(values,available_pairs,cells)
        write(PUBLIC/'imputation_fixtures.json',fixtures)
        validate_activation(a); budget()
        receipt = {'date':'2026-10-13','passed':True,'policy_id':POLICY,
            'qualification_scope':'Complete available source cache and fold-specific label-blind mean fixtures; no new raw execution',
            'activation_sha256':sha(ACTIVATION),'prior_source_regions_sha256':sha(PRIOR/'source_regions.jsonl'),
            'available_pairs':available_pairs,'available_identity_sha256':identity_sha(available_pairs),
            'shared_missing_pairs':[p for p in pairs if not available[p]],
            'available_trials':979,'regional_vectors_decoded':loaded,'feature_scalar_values':int(values.size),
            'unavailable_trial_regional_arrays_decoded':0,'unavailable_regional_arrays_skipped':3,
            'source_snapshot_file_sha256':sha(SNAPSHOT),'source_snapshot_array_sha256':array_sha(values),
            'imputation_fixtures_sha256':sha(PUBLIC/'imputation_fixtures.json'),
            'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'old_strict_gate_passed':False,'targets_converted':0,'fits':0,'raw_samples_decoded':0,
            'original_rating_decodes':0,'validation_numeric_access':False,'confirmation_access':False,
            'preservation':preservation()}
        write(PUBLIC/'qualification.json',receipt); print(json.dumps({k:v for k,v in receipt.items() if k not in ('available_pairs',)}),flush=True)
    except Exception as error:
        write(PUBLIC/'qualification_failure.json',{'status':'held_or_failed','error':str(error),
            'vectors_loaded':loaded,'targets_converted':0,'new_fits':0,'automatic_retry':False})
        raise

def source_targets_from_rows(rows,meta,policy_qualified):
    """Authorize source numeric labels under the NEW policy, not the old QC gate."""
    if policy_qualified is not True:
        raise PermissionError('Independent masked-source qualification precedes labels')
    people = {p[0] for p in meta}; targets = {}; retained = ignored = 0
    for row in rows:
        if row.get('subject') not in people or row.get('arm') != 'source_oof':
            ignored += 1; continue
        pair = row['subject'],int(row['vid'])
        if pair not in meta or meta[pair].get('role') != 'source':
            raise PermissionError('Cached target outside source declaration')
        if row.get('family') != meta[pair]['family'] or row.get('acquisition_group') != meta[pair]['acquisition_group']:
            raise PermissionError('Source target context join differs')
        value = float(row['target'])
        if not np.isfinite(value) or not 0 <= value <= 7:
            raise ValueError('Invalid source valence')
        if pair in targets and targets[pair] != value:
            raise ValueError('Conflicting source target values')
        targets[pair] = value; retained += 1
    if set(targets) != set(meta):
        raise PermissionError('All 980 original source labels must be retained')
    return targets,{'distinct_source_targets':len(targets),'source_rows_agreeing':retained,
        'rows_ignored_before_target_conversion':ignored,'original_rating_decoding':0,
        'unavailable_trial_targets_retained':1}

def fit():
    a = check(); meta,pairs,_,quality,available,_,_,folds = source_evidence()
    qualification = json.loads((PUBLIC/'qualification.json').read_bytes())
    independent = json.loads((PUBLIC/'qualification_independent_audit.json').read_bytes())
    if (qualification['passed'] is not True or independent['passed'] is not True
            or independent['qualification_sha256'] != sha(PUBLIC/'qualification.json')
            or independent['imputation_fixtures_sha256'] != sha(PUBLIC/'imputation_fixtures.json')
            or qualification['activation_sha256'] != sha(ACTIVATION)
            or qualification['old_strict_gate_passed'] is not False
            or (PUBLIC/'qualification_failure.json').exists()):
        raise PermissionError('Independent exact masked-source qualification required before labels')
    if (PUBLIC/'fitting_started.json').exists():
        raise PermissionError('Preserve this fit attempt; no automatic retry')
    start = time.monotonic(); cpu = time.process_time(); records = []; phase = 'snapshot_preflight'; converted = False
    write(PUBLIC/'fitting_started.json',{'utc':now(),'activation_sha256':sha(ACTIVATION),
        'qualification_sha256':sha(PUBLIC/'qualification.json'),
        'independent_qualification_sha256':sha(PUBLIC/'qualification_independent_audit.json'),
        'source_targets_only':True,'old_strict_gate_changed':False})
    def budget():
        if time.monotonic()-start > CAP_SECONDS or time.process_time()-cpu > CAP_SECONDS:
            raise TimeoutError('Cooperative source fitting cap')
    fixtures = json.loads((PUBLIC/'imputation_fixtures.json').read_bytes())
    def callback(manifest,fitted):
        budget()
        if len(records) >= 93:
            raise PermissionError('93 Ridge fit cap')
        stage = manifest.get('stage','source_cv'); model = manifest['model']
        fixture = fixtures['full_source'] if stage=='full_source_state_qualification' else fixtures['cells'][manifest['cell']]
        estimator,scaler = fitted['estimator'],fitted['scaler']
        arrays = {'coefficient':estimator.coef_,'intercept':np.asarray(estimator.intercept_),
            'mean':scaler.mean_,'scale':scaler.scale_,'variance':scaler.var_,
            'alpha':np.asarray(manifest['alpha'])}
        if model != 'context_only':
            mean = fitted['eeg_imputation_mean']
            if array_sha(mean) != fixture['mean_sha256'][model]:
                raise ValueError('Actual fitted mean differs from pretarget fixture')
            arrays['eeg_imputation_mean'] = mean
        path = PRIVATE/f'fit_{len(records):03d}_{stage}_{model}_state.npz'
        if path.exists(): raise FileExistsError('Preserve fitted source state')
        np.savez_compressed(path,**arrays)
        records.append({**manifest,'private_state_file':path.name,'private_state_sha256':sha(path),
            'context_scaler_moment_hashes':{'mean':array_sha(scaler.mean_[:13]),
                'variance':array_sha(scaler.var_[:13]),'scale':array_sha(scaler.scale_[:13])}})
        with (PUBLIC/'fit_records.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(records[-1],allow_nan=False)+'\n')
        if len(records)%9 == 0 or stage != 'source_cv':
            print(json.dumps({'phase':'source_fitting','fits':len(records),'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
        budget()
    try:
        if sha(SNAPSHOT,budget) != qualification['source_snapshot_file_sha256']:
            raise ValueError('Qualified available snapshot changed')
        with np.load(SNAPSHOT,allow_pickle=False) as bundle:
            if bundle.files != ['vectors']: raise ValueError('Unexpected snapshot keys')
            vectors = bundle['vectors'].copy()
        available_pairs = [tuple(p) for p in qualification['available_pairs']]
        if (available_pairs != [p for p in pairs if available[p]] or vectors.shape != (979,3,150)
                or not np.isfinite(vectors).all() or array_sha(vectors) != qualification['source_snapshot_array_sha256']):
            raise ValueError('Qualified snapshot role/order/geometry changed')
        index = {p:i for i,p in enumerate(available_pairs)}
        def feature_loader(pair,region):
            if pair not in index or not available.get(pair) or region not in REGIONS:
                raise PermissionError('Never decode unavailable or nonsource EEG')
            return vectors[index[pair],REGIONS.index(region),:]
        phase = 'cached_source_targets'
        if sha(TARGETS,budget) != a['bindings'][TARGETS.relative_to(REPO).as_posix()]:
            raise ValueError('Canonical cached source targets changed')
        converted = True
        with TARGETS.open(newline='',encoding='utf-8') as stream:
            targets,target_receipt = source_targets_from_rows(csv.DictReader(stream),meta,True)
        write(PUBLIC/'cached_source_targets_receipt.json',target_receipt)
        phase = 'source_cv_and_full_state_qualification'
        with threadpool_limits(limits=2,user_api='blas'):
            result = run_source_cv(source_pairs=pairs,metadata=meta,person_folds=folds,
                quality_rows=quality,feature_loader=feature_loader,target_loader=lambda p:targets[p],
                on_fit=callback,check_budget=budget)
            full = fit_full_source(source_pairs=pairs,metadata=meta,quality_rows=quality,
                feature_loader=feature_loader,target_loader=lambda p:targets[p],
                candidate_metrics=result['candidate_metrics'],on_fit=callback,check_budget=budget)
        write(PUBLIC/'cv_predictions.json',{'ridge':result['cv_predictions'],'category':result['category_predictions']})
        replays = []
        for model,fitted in full['private_fitted_models'].items():
            record = next(r for r in records if r.get('stage')=='full_source_state_qualification' and r['model']==model)
            with np.load(PRIVATE/record['private_state_file'],allow_pickle=False) as saved:
                values = []
                for pair in pairs:
                    context = context_features(meta[pair])
                    if model == 'context_only': values.append(context); continue
                    eeg = vectors[index[pair],2,:] if available[pair] and model=='late_joint' else (
                        vectors[index[pair]].mean(axis=0) if available[pair] else saved['eeg_imputation_mean'])
                    values.append(np.r_[context,eeg])
                x = np.stack(values)
                manual = np.clip(((x-saved['mean'])/saved['scale'])@saved['coefficient']+saved['intercept'],0,7)
            reference = np.clip(fitted['estimator'].predict(fitted['scaler'].transform(x)),0,7)
            delta = float(np.max(np.abs(manual-reference)))
            if delta > 1e-10: raise ValueError('Saved masked-source state replay differs')
            replays.append({'model':model,'rows':980,'alpha':full['common_alpha_selection']['selected_alpha'],
                'max_prediction_discrepancy':delta,'source_identity_sha256':fitted['source_identity_sha256'],
                'private_state_sha256':record['private_state_sha256'],
                'context_scaler_moment_hashes':record['context_scaler_moment_hashes'],
                'imputation_mean_sha256':None if model=='context_only' else array_sha(fitted['eeg_imputation_mean'])})
        write(PRIVATE/'full_source_category_prior.json',full['category_prior'])
        write(PUBLIC/'final_source_state_qualification.json',{'passed':True,'models':replays,
            'private_category_prior_sha256':sha(PRIVATE/'full_source_category_prior.json'),
            'source_rows':980,'available_donors':979,'imputed_training_rows':1,
            'replay_scope':'Source regression-state algebra including training imputation; no source performance or protected fallback export claim',
            'protected_exports_qualified':False,'validation_scored':False})
        summary = {k:v for k,v in result.items() if k not in ('cv_predictions','category_predictions','fits','cells','shared_availability')}
        summary['shared_availability'] = [{'subject':p[0],'vid':p[1],'eeg_available':available[p]} for p in pairs]
        summary.update({'date':'2026-10-13','status':'completed_source_coverage_missing_selection','policy_id':POLICY,
            'cv_fit_count':90,'full_source_fit_count':3,'final_source_fits':3,'category_prior_fits':11,
            'old_strict_gate_passed':False,'quality_thresholds_changed':False,
            'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'cv_predictions_sha256':sha(PUBLIC/'cv_predictions.json'),
            'cached_targets_receipt_sha256':sha(PUBLIC/'cached_source_targets_receipt.json'),
            'validation_access':False,'protected_access':False,'independent_confirmation':False})
        validate_activation(a); source_evidence(); budget()
        if len(records) != 93: raise ValueError('Source fit count differs')
        write(PUBLIC/'comparison.json',summary)
        print(json.dumps({'status':summary['status'],'fits':93,
            'source_nomination':summary['source_nomination'],
            'fixed_alpha100':next(x for x in summary['contrasts'] if x['primary_fixed_alpha']),
            'protected_access':False}),flush=True)
    except Exception as error:
        write(PUBLIC/'fitting_failure.json',{'status':'held_or_failed','phase':phase,'error':str(error),
            'cached_source_target_conversion_may_have_started':converted,'fits_completed':len(records),
            'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'automatic_retry':False,'protected_access':False})
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('declare','qualify','fit'))
    args = parser.parse_args()
    {'declare':declare,'qualify':qualify,'fit':fit}[args.action]()
