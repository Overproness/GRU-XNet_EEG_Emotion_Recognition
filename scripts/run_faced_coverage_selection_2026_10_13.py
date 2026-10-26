"""Declared complete-source QC, then bounded source-only coverage selection.

No original rating decoder or validation/protected numeric input interpretation
is permitted. Existing public source valence is converted only after all 2,940
regions and their independent QC audit pass. Frozen historical workers, papers
and failure ledgers stay intact. Hash/CSV transport is not numeric interpretation.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_temporal_coverage_pilot_2026_10_12 import (
    array_sha, sha, load, safe_raw_path, runtime as preceding_runtime, read_header,
    read_interval, selected_header, quality_metrics, calibrate_common_counts,
    artifact_trial_windows, eeg_availability, spectral_features,
)
from gruxnet.faced_temporal_coverage import SourceOnlySegmentGuard, REGIONS
from gruxnet.faced_coverage_selection import run_source_cv, fit_full_source, source_roster, qualify_quality
from gruxnet.faced_classical_contract import context_features, CATEGORIES, ACQUISITIONS, BANDS
from gruxnet.faced_proposal_preprocessing import COMMON_CHANNELS

BASE = REPO/'results/development'
PREVIOUS = BASE/'faced_temporal_coverage_preflight_2026-10-12/plan.json'
PLAN = BASE/'faced_coverage_selection_preflight_2026-10-13/plan.json'
PUBLIC = BASE/'faced_coverage_selection_2026-10-13'
PRIVATE = REPO.parent/'publication_runs/faced_coverage_selection_2026-10-13'
OLD_QC = REPO.parent/'publication_runs/faced_artifact_control_2026-10-11'
TARGETS = BASE/'faced_inference_quality_2026-10-11/predictions.csv'
PROTOCOL = REPO/'docs/publication/FACED_Coverage_Experiment_Protocol_2026-10-13.md'
ACTIVATION = PUBLIC/'activation_v2.json'
PREP_CAP = 3600
FIT_CAP_SECONDS = 600

def now():
    return datetime.now(timezone.utc).isoformat()

def runtime():
    return {**preceding_runtime(),'sklearn':sklearn.__version__,
            'threadpoolctl':threadpoolctl.__version__,'fitting_blas_threads':2}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')

def bindings(paths):
    return {p.relative_to(REPO).as_posix():sha(p) for p in paths}

def check_bindings(values):
    for name,digest in values.items():
        if sha(REPO/name) != digest:
            raise ValueError('Frozen bytes changed: '+name)

def metadata():
    previous = load(PREVIOUS)
    catalogue = {r['clip']:r for r in load(BASE/'faced_codebooks_2026-10-10/audit.json')['stimuli']}
    old = {(r['subject'],r['vid']):r for r in load(BASE/'faced_artifact_control_2026-10-11/source_records.json')['records']}
    rows = previous['source_records']
    result = {}
    for row in rows:
        pair = row['subject'],row['vid']
        if pair not in old or old[pair]['role'] != 'source' or row['role'] != 'source':
            raise PermissionError('Authoritative source identity differs')
        result[pair] = {**catalogue[pair[1]],**row,'trial':old[pair]['trial']}
        if array_sha(context_features(result[pair])) != old[pair]['context_feature_sha256']:
            raise ValueError('Historical contextual input bytes changed')
    pairs,_ = source_roster(list(result),result)
    return previous,old,result,pairs

def declared_segments(previous):
    return [{'subject':r['subject'],'vid':r['vid'],'role':'source','region':region,
             'native_rate_hz':r['native_rate_hz'],'sample_interval':r['segments'][region]}
            for r in previous['source_records'] for region in REGIONS]

def dependencies():
    files = [PREVIOUS,PLAN,PROTOCOL,TARGETS,
        BASE/'faced_raw_2026-10-11/headers.json',BASE/'faced_raw_2026-10-11/extraction.json',
        BASE/'faced_temporal_coverage_pilot_2026-10-12/completion.json',
        REPO/'paper/exploratory_evaluation_2026-10-12/build_receipt.json']
    names = ('scripts/run_faced_coverage_selection_2026_10_13.py',
        'tests/test_faced_coverage_execution.py','gruxnet/faced_coverage_selection.py',
        'scripts/prepare_faced_coverage_selection_2026_10_13.py','tests/test_faced_coverage_selection.py',
        'scripts/run_faced_temporal_coverage_pilot_2026_10_12.py',
        'gruxnet/faced_temporal_coverage.py','gruxnet/faced_source_quality.py',
        'gruxnet/faced_proposal_preprocessing.py','gruxnet/faced_source_notch.py',
        'gruxnet/faced_artifact_control.py','gruxnet/faced_inference_quality.py',
        'gruxnet/faced_classical_contract.py','gruxnet/faced_classical_runner.py','scripts/audit_faced_raw.py')
    files.extend(REPO/n for n in names)
    for packet in (load(PREVIOUS),load(PLAN)):
        for key in ('input_bindings','code_bindings','bindings'):
            files.extend(REPO/n for n in packet.get(key,{}))
    return bindings(sorted(set(files)))

def validate_activation(a,previous,plan):
    fixed = {'status':'active_source_only_coverage_experiment','confirmation_access':False,
        'validation_signal_access':False,'original_rating_access':False,
        'source_trials':980,'source_segments':2940,'cv_fit_cap':90,'full_source_fit_cap':3,
        'preparation_wall_seconds_cap':PREP_CAP,'preparation_cpu_seconds_cap':PREP_CAP,
        'fitting_wall_seconds_cap':FIT_CAP_SECONDS,'fitting_cpu_seconds_cap':FIT_CAP_SECONDS,
        'independent_source_qc_required_before_targets':True,
        'strict_source_qc_no_exclusions':True,'primary_diagnostic_alpha':100.0}
    if any(a.get(k) != v for k,v in fixed.items()):
        raise PermissionError('Activation missing, widened or inconsistent')
    if a.get('previous_plan_sha256') != sha(PREVIOUS) or a.get('cv_plan_sha256') != sha(PLAN):
        raise PermissionError('Source or CV geometry changed')
    segments = declared_segments(previous)
    if len(segments) != 2940 or a.get('counts_sample_values_cap') != sum(30*32*r['native_rate_hz'] for r in segments):
        raise PermissionError('Native sample budget changed')
    if (plan['fit_cap'] != 90 or len(plan['cells']) != 10
            or plan['final_source_state_qualification_fit_cap'] != 3
            or plan['total_ridge_fit_cap_including_optional_state_qualification'] != 93
            or plan['alphas'] != [.1,10.,100.]):
        raise PermissionError('Source selection plan differs')

def declare():
    previous,_,_,_ = metadata(); plan = load(PLAN)
    pilot = load(BASE/'faced_temporal_coverage_pilot_2026-10-12/completion.json')
    check_bindings(pilot['bindings']); check_bindings(plan['bindings'])
    if not pilot['passed'] or pilot['availability_passes'] != 24:
        raise PermissionError('Qualified preceding pilot required')
    a = {'date':'2026-10-13','declared_utc':now(),
        'supersedes_unexecuted_declaration':'activation.json',
        'status':'active_source_only_coverage_experiment','author_authorization':'User explicitly requested the bounded matched late-versus-distributed experiment with source-only selection, followed by a fixed confirmation proposal.',
        'previous_plan_sha256':sha(PREVIOUS),'cv_plan_sha256':sha(PLAN),
        'confirmation_access':False,'validation_signal_access':False,'original_rating_access':False,
        'cached_source_target_access':'Only source rows from bound public quality CSV after complete source QC and independent audit pass',
        'source_trials':980,'source_segments':2940,'cv_fit_cap':90,'full_source_fit_cap':3,
        'counts_sample_values_cap':sum(30*32*r['native_rate_hz'] for r in declared_segments(previous)),
        'preparation_wall_seconds_cap':PREP_CAP,'preparation_cpu_seconds_cap':PREP_CAP,
        'fitting_wall_seconds_cap':FIT_CAP_SECONDS,'fitting_cpu_seconds_cap':FIT_CAP_SECONDS,
        'independent_source_qc_required_before_targets':True,'strict_source_qc_no_exclusions':True,
        'primary_diagnostic_alpha':100.0,'alphas':[.1,10.,100.],
        'secondary_selection':'Shared alpha: equal-model mean source CV MAE, then MSE, then alpha order; coverage at shared alpha MAE/MSE/tie late',
        'bindings':dependencies(),'runtime':runtime(),
        'automatic_retry':False,'research_question_changed':False,
        'limits':'Adaptive source development, six paired-category held titles per assignment; no held-title Fear/Inspiration. Any unavailable source segment holds target conversion and all fits. Confirmation proposal only; protected data stay sealed.'}
    validate_activation(a,previous,plan); write(ACTIVATION,a)
    print(json.dumps({'declared':True,'source_regions':2940,'source_cv_fit_cap':90,'protected_access':False}),flush=True)

def check():
    a = load(ACTIVATION); previous = load(PREVIOUS); plan = load(PLAN)
    validate_activation(a,previous,plan); check_bindings(a['bindings'])
    if a['runtime'] != runtime():
        raise ValueError('Frozen source runtime changed')
    return a

def prepare():
    a = check(); previous,old,meta,pairs = metadata()
    if (PUBLIC/'preparation_started.json').exists():
        raise PermissionError('Preserve this completed/failed attempt; no automatic retry')
    segments = declared_segments(previous)
    guard = SourceOnlySegmentGuard(source_people=previous['source_people'],
        validation_people=previous['validation_people_denied'],protected_people=previous['protected_people_denied'],
        source_pairs=pairs,pilot_rows=segments)
    headers = {r['subject']:r['data.bdf'] for r in load(BASE/'faced_raw_2026-10-11/headers.json')['records'] if r['subject'] in previous['source_people']}
    originals = {r['member']:r for r in load(BASE/'faced_raw_2026-10-11/extraction.json')['records']}
    start = time.monotonic(); cpu = time.process_time(); snapshots = {}; rows = []; count = 0
    write(PUBLIC/'preparation_started.json',{'utc':now(),'activation_sha256':sha(ACTIVATION),
        'original_rating_access':False,'cached_target_access':False,'new_fits':0})
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    def budget():
        if time.monotonic()-start > PREP_CAP or time.process_time()-cpu > PREP_CAP:
            raise TimeoutError('Cooperative source preparation CPU/wall cap')
    def reader(person,vid,region,interval,rate):
        nonlocal count
        budget(); pair = person,vid
        if pair not in meta or meta[pair]['role'] != 'source' or meta[pair]['segments'][region] != interval:
            raise PermissionError('Authoritative source range changed')
        r = meta[pair]; path = safe_raw_path(person,'data.bdf'); header = headers[person]
        if person not in snapshots:
            if read_header(path) != header:
                raise ValueError('Original technical header changed')
            for name in ('data.bdf','evt.bdf'):
                file = safe_raw_path(person,name)
                if sha(file,budget) != originals[f'Data/{file.parent.name}/{name}']['sha256']:
                    raise ValueError('Original source file integrity failure')
            snapshots[person] = {'sha256':originals[f'Data/{path.parent.name}/data.bdf']['sha256'],
                'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        if r['original_data_sha256'] != snapshots[person]['sha256'] or r['original_header_sha256'] != header['header_sha256']:
            raise ValueError('Source lineage changed')
        budget()
        with path.open('rb') as stream:
            counts,access = read_interval(stream,header,*interval,r['channel_indices'],reference=True)
        count += int(counts.size)
        if count > a['counts_sample_values_cap'] or access['extra_boundary_samples_decoded'] != 0:
            raise PermissionError('Decoded sample/boundary budget exceeded')
        values = calibrate_common_counts(counts,person,selected_header(header,r['channel_indices']))
        with threadpool_limits(limits=12,user_api='blas'):
            windows,artifact = artifact_trial_windows(values,rate)
        quality = quality_metrics(counts,values,windows,rate)
        decision = eeg_availability(quality,technical_passed=True,repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
        features = spectral_features(windows)
        count_hash,window_hash,feature_hash = array_sha(counts),array_sha(windows),array_sha(features)
        if region == 'late':
            if (count_hash != r['prior_late_count_array_sha256'] or window_hash != r['prior_late_processed_window_sha256']
                    or feature_hash != old[pair]['spectral_feature_sha256']):
                raise ValueError('Historical late count/window/feature replay differs')
            prior = OLD_QC/f'{person}_clip{vid:02d}_quality.json'
            if sha(prior) != r['prior_late_quality_file_sha256'] or load(prior) != quality:
                raise ValueError('Historical late QC numerical replay differs')
        stem = f'{person}_clip{vid:02d}_{region}'
        write(PRIVATE/(stem+'_quality.json'),quality); write(PRIVATE/(stem+'_access.json'),access)
        write(PRIVATE/(stem+'_artifact.json'),artifact)
        feature_path = PRIVATE/(stem+'_features.npz')
        if feature_path.exists():
            raise FileExistsError('Preserve existing feature cache')
        np.savez_compressed(feature_path,features=features)
        if path.stat().st_size != snapshots[person]['size'] or path.stat().st_mtime_ns != snapshots[person]['mtime_ns']:
            raise ValueError('Original source recording changed during read')
        budget()
        return {'subject':person,'vid':vid,'role':'source','region':region,'sample_interval':interval,
            'native_rate_hz':rate,'samples_decoded':int(counts.size),'extra_boundary_samples_decoded':0,
            'eeg_available':decision['eeg_available'],'reasons':decision['reasons'],
            'artifact_decision':artifact['decision'],'repair_count':len(artifact['repaired_channels']),
            'count_array_sha256':count_hash,'processed_window_sha256':window_hash,
            'spectral_feature_sha256':feature_hash,'feature_file_sha256':sha(feature_path),
            'range_ledger_sha256':access['range_ledger_sha256'],
            'quality_file_sha256':sha(PRIVATE/(stem+'_quality.json')),
            'access_file_sha256':sha(PRIVATE/(stem+'_access.json')),
            'late_exact_replay_passed':region=='late'}
    try:
        with (PUBLIC/'source_regions.jsonl').open('x',encoding='utf-8',newline='\n') as ledger:
            for row in segments:
                result = guard.read(subject=row['subject'],vid=row['vid'],region=row['region'],
                    sample_interval=row['sample_interval'],native_rate_hz=row['native_rate_hz'],reader=reader)
                rows.append(result); ledger.write(json.dumps(result,allow_nan=False)+'\n'); ledger.flush()
                if len(rows)%42 == 0:
                    print(json.dumps({'phase':'source_preparation','regions':len(rows),'total':2940,
                        'unavailable':sum(not r['eeg_available'] for r in rows),
                        'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
        for person,snapshot in snapshots.items():
            if sha(safe_raw_path(person,'data.bdf'),budget) != snapshot['sha256']:
                raise ValueError('Original source recording changed after preparation')
        check_bindings(a['bindings']); budget()
        if len(rows) != 2940 or count != a['counts_sample_values_cap']:
            raise ValueError('Incomplete source-region/count rectangle')
        unavailable = [r for r in rows if not r['eeg_available']]
        gate = {'date':'2026-10-13','passed':not unavailable,
            'status':'qualified_source_preparation' if not unavailable else 'held_source_quality',
            'source_trials':980,'source_regions':2940,'source_people':70,
            'unavailable_regions':len(unavailable),'unavailable_trials':len({(r['subject'],r['vid']) for r in unavailable}),
            'failures':[{k:r[k] for k in ('subject','vid','region','reasons')} for r in unavailable],
            'by_region':{region:{'regions':980,'unavailable':sum(not r['eeg_available'] for r in rows if r['region']==region)} for region in REGIONS},
            'counts_sample_values_decoded':count,'late_exact_replays':980,
            'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'activation_sha256':sha(ACTIVATION),'source_regions_sha256':sha(PUBLIC/'source_regions.jsonl'),
            'cached_targets_converted':0,'original_rating_access':False,'new_fits':0,
            'validation_signal_access':False,'confirmation_access':False,'raw_array_persistence':False}
        write(PUBLIC/'source_gate.json',gate); print(json.dumps({k:v for k,v in gate.items() if k!='failures'}),flush=True)
    except Exception as error:
        write(PUBLIC/'preparation_failure.json',{'status':'held_or_failed','reason':str(error),'error_type':type(error).__name__,
            'regions_completed':len(rows),'counts_decoded':count,'elapsed_seconds':time.monotonic()-start,
            'cpu_seconds':time.process_time()-cpu,'targets_converted':0,'new_fits':0,
            'protected_access':False,'automatic_retry':False})
        raise

def source_targets_from_rows(rows,meta,quality_passed):
    """Canonical source targets only; reject/skip role before target conversion."""
    if quality_passed is not True:
        raise PermissionError('Complete source QC before any target conversion')
    people = {p[0] for p in meta}; targets = {}; retained = 0; ignored = 0
    for row in rows:
        if row.get('subject') not in people or row.get('arm') != 'source_oof':
            ignored += 1; continue
        pair = row['subject'],int(row['vid'])
        if pair not in meta or meta[pair].get('role') != 'source':
            raise PermissionError('Cached source target identity outside declaration')
        if row.get('family') != meta[pair]['family'] or row.get('acquisition_group') != meta[pair]['acquisition_group']:
            raise PermissionError('Cached source target family/acquisition differs')
        value = float(row['target'])
        if not np.isfinite(value) or not 0 <= value <= 7:
            raise ValueError('Invalid original-scale cached source valence')
        if pair in targets and targets[pair] != value:
            raise ValueError('Conflicting canonical source targets')
        targets[pair] = value; retained += 1
    if set(targets) != set(meta):
        raise PermissionError('Complete source target rectangle required')
    return targets,{'distinct_source_targets':len(targets),'source_rows_agreeing':retained,
                    'rows_ignored_before_target_conversion':ignored,'original_rating_decoding':0}

def qualify_source_ledger(rows,pairs,gate,audit,activation_digest,ledger_digest,gate_digest):
    """Authenticate the full region-level gate before features or target values."""
    if (gate.get('passed') is not True or gate.get('unavailable_regions') != 0
            or gate.get('activation_sha256') != activation_digest
            or gate.get('source_regions_sha256') != ledger_digest
            or gate.get('source_regions') != 2940 or gate.get('source_trials') != 980
            or gate.get('cached_targets_converted') != 0 or gate.get('new_fits') != 0):
        raise PermissionError('Strict source preparation gate differs')
    if (audit.get('passed') is not True or audit.get('source_quality_gate_passed') is not True
            or audit.get('source_regions_sha256') != ledger_digest
            or audit.get('source_gate_sha256') != gate_digest):
        raise PermissionError('Independent exact source gate audit required')
    if any(r.get('eeg_available') is not True or r.get('reasons') != [] for r in rows):
        raise PermissionError('Any unavailable or flagged source region holds target conversion')
    quality_rows = [{**r,'technical_passed':True,'review_flags':r['reasons']} for r in rows]
    qualify_quality(pairs,quality_rows)
    return quality_rows

def fit():
    a = check(); previous,_,meta,pairs = metadata(); plan = load(PLAN)
    gate = load(PUBLIC/'source_gate.json')
    if gate['passed'] is not True or gate['unavailable_regions'] != 0:
        raise PermissionError('Strict complete-source QC holds targets and fitting')
    audit = load(PUBLIC/'source_qc_independent_audit.json')
    if (PUBLIC/'fitting_started.json').exists():
        raise PermissionError('Preserve this fit attempt; no automatic retry')
    rows = [json.loads(line) for line in (PUBLIC/'source_regions.jsonl').read_text(encoding='utf-8').splitlines()]
    quality_rows = qualify_source_ledger(rows,pairs,gate,audit,sha(ACTIVATION),
        sha(PUBLIC/'source_regions.jsonl'),sha(PUBLIC/'source_gate.json'))
    row_map = {(r['subject'],r['vid'],r['region']):r for r in rows}
    if len(row_map) != 2940:
        raise PermissionError('Source feature rectangle changed')
    start = time.monotonic(); cpu = time.process_time(); fit_records = []
    write(PUBLIC/'fitting_started.json',{'utc':now(),'activation_sha256':sha(ACTIVATION),
        'source_gate_sha256':sha(PUBLIC/'source_gate.json'),'independent_qc_sha256':sha(PUBLIC/'source_qc_independent_audit.json'),
        'targets_scope':'980 source only; no original rating decode or validation/protected target conversion'})
    def budget():
        if time.monotonic()-start > FIT_CAP_SECONDS or time.process_time()-cpu > FIT_CAP_SECONDS:
            raise TimeoutError('Cooperative source fitting CPU/wall cap')
    def callback(manifest,fitted):
        if len(fit_records) >= 93:
            raise PermissionError('Source Ridge fit cap')
        stage = manifest.get('stage','source_cv'); stem = f'fit_{len(fit_records):03d}_{stage}_{manifest["model"]}'
        path = PRIVATE/(stem+'_state.npz')
        if path.exists():
            raise FileExistsError('Preserve fitted state')
        estimator,scaler = fitted['estimator'],fitted['scaler']
        np.savez_compressed(path,coefficient=estimator.coef_,intercept=np.asarray(estimator.intercept_),
            mean=scaler.mean_,scale=scaler.scale_,variance=scaler.var_,alpha=np.asarray(manifest['alpha']))
        fit_records.append({**manifest,'private_state_file':path.name,'private_state_sha256':sha(path)})
        with (PUBLIC/'fit_records.jsonl').open('a',encoding='utf-8',newline='\n') as ledger:
            ledger.write(json.dumps(fit_records[-1],allow_nan=False)+'\n')
        if len(fit_records)%9 == 0 or stage != 'source_cv':
            print(json.dumps({'phase':'source_fitting','fits':len(fit_records),'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
        budget()
    phase = 'source_feature_preflight'; target_conversion_started = False
    try:
        features = {}
        for pair in pairs:
            for region in REGIONS:
                budget()
                row = row_map[(*pair,region)]
                file = PRIVATE/f'{pair[0]}_clip{pair[1]:02d}_{region}_features.npz'
                if sha(file,budget) != row['feature_file_sha256']:
                    raise ValueError('Private source feature cache changed')
                with np.load(file,allow_pickle=False) as bundle:
                    values = bundle['features'].copy()
                if values.shape != (150,) or not np.isfinite(values).all() or array_sha(values) != row['spectral_feature_sha256']:
                    raise ValueError('Source features fail qualified geometry/hash')
                features[(*pair,region)] = values
        phase = 'cached_source_target_preflight'
        if sha(TARGETS,budget) != a['bindings'][TARGETS.relative_to(REPO).as_posix()]:
            raise ValueError('Canonical public target CSV changed')
        target_conversion_started = True
        with TARGETS.open(newline='',encoding='utf-8') as stream:
            targets,target_receipt = source_targets_from_rows(csv.DictReader(stream),meta,True)
        write(PUBLIC/'cached_source_targets_receipt.json',target_receipt)
        feature_loader = lambda pair,region:features[(*pair,region)]
        target_loader = lambda pair:targets[pair]
        phase = 'source_cv_and_final_state_qualification'
        with threadpool_limits(limits=2,user_api='blas'):
            result = run_source_cv(source_pairs=pairs,metadata=meta,
                person_folds=load(BASE/'faced_development_protocol_2026-10-11/plan.json')['source_participant_oof_folds'],
                quality_rows=quality_rows,feature_loader=feature_loader,target_loader=target_loader,
                on_fit=callback,check_budget=budget)
            full = fit_full_source(source_pairs=pairs,metadata=meta,quality_rows=quality_rows,
                feature_loader=feature_loader,target_loader=target_loader,candidate_metrics=result['candidate_metrics'],
                on_fit=callback,check_budget=budget)
        write(PUBLIC/'cv_predictions.json',{'ridge':result['cv_predictions'],'category':result['category_predictions']})
        summary = {k:v for k,v in result.items() if k not in ('cv_predictions','category_predictions','fits','cells')}
        summary.update({'date':'2026-10-13','status':'completed_source_coverage_selection',
            'cv_fit_count':90,'full_source_fit_count':3,'category_prior_fits':11,
            'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'cv_predictions_sha256':sha(PUBLIC/'cv_predictions.json'),
            'cached_targets_receipt_sha256':sha(PUBLIC/'cached_source_targets_receipt.json')})
        # Full states are qualified by source-only prediction replay, not source error scores.
        state_replay = []
        for model,fitted in full['private_fitted_models'].items():
            record = next(r for r in fit_records if r.get('stage')=='full_source_state_qualification' and r['model']==model)
            context = {p:context_features(meta[p]) for p in pairs}
            if model == 'context_only':
                x = np.stack([context[p] for p in pairs])
            else:
                eeg = {p:features[(*p,'late')] if model=='late_joint' else np.mean([features[(*p,q)] for q in REGIONS],axis=0) for p in pairs}
                x = np.stack([np.r_[context[p],eeg[p]] for p in pairs])
            with np.load(PRIVATE/record['private_state_file'],allow_pickle=False) as state:
                manual = np.clip(((x-state['mean'])/state['scale'])@state['coefficient']+state['intercept'],0,7)
            reference = np.clip(fitted['estimator'].predict(fitted['scaler'].transform(x)),0,7)
            discrepancy = float(np.max(np.abs(manual-reference)))
            if discrepancy > 1e-10:
                raise ValueError('Saved source state prediction replay differs')
            state_replay.append({'model':model,'rows':980,'max_prediction_discrepancy':discrepancy,
                'source_identity_sha256':fitted['source_identity_sha256'],'private_state_sha256':record['private_state_sha256'],
                'alpha':full['common_alpha_selection']['selected_alpha']})
        write(PRIVATE/'full_source_category_prior.json',full['category_prior'])
        write(PUBLIC/'final_source_state_qualification.json',{'passed':True,'models':state_replay,
            'private_category_prior_sha256':sha(PRIVATE/'full_source_category_prior.json'),
            'source_rows':980,'protected_exports_qualified':False,'validation_scored':False})
        check_bindings(a['bindings']); budget()
        if len(fit_records) != 93:
            raise ValueError('Source fit count differs')
        write(PUBLIC/'comparison.json',summary)
        print(json.dumps({'status':summary['status'],'source_nomination':summary['source_nomination'],
            'fixed_alpha100':next(x for x in summary['contrasts'] if x['primary_fixed_alpha']),
            'fits':93,'protected_access':False}),flush=True)
    except Exception as error:
        write(PUBLIC/'fitting_failure.json',{'status':'held_or_failed','reason':str(error),'error_type':type(error).__name__,
            'phase':phase,'cached_source_target_conversion_may_have_started':target_conversion_started,
            'fits_completed':len(fit_records),'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'protected_access':False,'automatic_retry':False})
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('declare','prepare','fit'))
    args = parser.parse_args()
    {'declare':declare,'prepare':prepare,'fit':fit}[args.action]()
