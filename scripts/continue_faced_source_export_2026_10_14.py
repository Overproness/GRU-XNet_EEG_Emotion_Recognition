"""Reviewed cache-only continuation of the preserved source export hold.

Correct only the prior-schema compatibility check. Authenticate actual prior
raw-QC/access evidence and cached features; no recording or rating is reread.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.qualify_faced_source_export_2026_10_14 import (
    source_scope,preservation,audit_predictions,write,sha,load,array_sha,
    PUBLIC,PRIVATE,PRIOR,STATE_ROOT,BASE)
from gruxnet.faced_temporal_coverage import REGIONS
from gruxnet.faced_frozen_inference import infer_four_methods,export_prediction_lock,verify_prediction_lock

PLAN=PUBLIC/'continuation_plan.json'
CAP_SECONDS=600


def validate_hold(old,failure):
    if (failure.get('status')!='held_or_failed' or failure.get('reason')!='Unexpected full-source prior schema or support'
            or failure.get('raw_regions_completed')!=27 or failure.get('count_scalars_decoded')!=19440000
            or failure.get('new_fits')!=0 or failure.get('target_reads')!=0
            or failure.get('validation_access') is not False or failure.get('protected_access') is not False
            or old.get('raw_trials')!=9 or old.get('raw_regions')!=27
            or old.get('count_scalar_cap')!=19440000 or old.get('protected_access') is not False):
        raise PermissionError('Exact preserved prior-schema source hold required')


def declare():
    old=load(PUBLIC/'plan.json'); failure=load(PUBLIC/'failure.json')
    validate_hold(old,failure); preservation()
    for name,digest in old['code_and_input_bindings'].items():
        if sha(REPO/name)!=digest:
            raise ValueError('Preserved original source worker dependency differs')
    expected={(r['subject'],r['vid'],r['region']):r for r in
              [json.loads(s) for s in (BASE/'faced_coverage_selection_2026-10-13/source_regions.jsonl').read_text().splitlines()]}
    raw_receipts={}
    for row in old['pilot_regions']:
        reference=expected[row['subject'],row['vid'],row['region']]
        for kind in ('quality','access'):
            file=PRIVATE/f"{row['subject']}_clip{row['vid']:02d}_{row['region']}_{kind}.json"
            digest=sha(file)
            if digest!=reference[kind+'_file_sha256']:
                raise ValueError('Actual earlier raw replay receipt differs')
            raw_receipts[file.relative_to(REPO.parent).as_posix()]=digest
    if len(raw_receipts)!=54:
        raise ValueError('Require all27actual source QC/access receipt pairs')
    files=[
        'scripts/continue_faced_source_export_2026_10_14.py','tests/test_faced_source_export_continuation_2026_10_14.py',
        'gruxnet/faced_frozen_prior_adapter.py','tests/test_faced_frozen_prior_adapter.py',
        'results/development/faced_source_export_qualification_2026-10-14/plan.json',
        'results/development/faced_source_export_qualification_2026-10-14/execution_started.json',
        'results/development/faced_source_export_qualification_2026-10-14/failure.json']
    bindings=dict(old['code_and_input_bindings']); bindings.update({p:sha(REPO/p) for p in files})
    plan={'date':'2026-10-14','status':'declared_reviewed_cache_only_prior_schema_continuation',
          'reason':'Frozen loader expected lowercase coarse labels; original saved source prior uses Negative/Positive. Validate that exact original schema without editing any values.',
          'original_failed_plan_sha256':sha(PUBLIC/'plan.json'),'original_failure_sha256':sha(PUBLIC/'failure.json'),
          'raw_receipts':raw_receipts,'private_bindings':old['private_bindings'],'code_and_input_bindings':bindings,
          'source_identity_sha256':old['source_identity_sha256'],'source_pairs':old['source_pairs'],
          'final_state_records':old['final_state_records'],'source_snapshot_scalar_cap':440550,
          'source_trials':980,'source_prediction_rows':3920,'existing_raw_replay_regions':27,
          'new_raw_decodes':0,'new_fits':0,'target_reads':0,'original_rating_reads':0,
          'validation_access':False,'protected_access':False,'confirmation_activated':False,
          'alpha':100.0,'wall_seconds_cap':CAP_SECONDS,'cpu_seconds_cap':CAP_SECONDS,
          'new_output_directory':'source_prediction_lock_v2','automatic_retry':False,
          'interpretation':'Separate reviewed continuation; uses authenticated earlier actual raw receipts and historical available source features, not another raw replay or a changed scientific recipe.'}
    write(PLAN,plan)
    print(json.dumps({'declared':True,'raw_receipts':54,'new_raw_decodes':0,'new_fits':0,'source_rows':980}),flush=True)


def run():
    from gruxnet.faced_frozen_prior_adapter import load_frozen_bundle_v2
    plan=load(PLAN); old=load(PUBLIC/'plan.json'); validate_hold(old,load(PUBLIC/'failure.json'))
    for name,digest in plan['code_and_input_bindings'].items():
        if sha(REPO/name)!=digest:
            raise ValueError('Frozen continuation dependency differs')
    if (plan['status']!='declared_reviewed_cache_only_prior_schema_continuation' or plan['new_raw_decodes']!=0
            or plan['new_fits']!=0 or plan['protected_access'] is not False):
        raise PermissionError('Only the declared source-only cache continuation is allowed')
    if (PUBLIC/'continuation_started.json').exists():
        raise PermissionError('Preserve the completed/failed continuation; no automatic retry')
    _,meta,pairs,_,_,_=source_scope()
    if json.loads(json.dumps(pairs))!=plan['source_pairs']:
        raise PermissionError('Exact prior source roster required')
    start=time.monotonic(); cpu=time.process_time()
    def budget():
        if time.monotonic()-start>CAP_SECONDS or time.process_time()-cpu>CAP_SECONDS:
            raise TimeoutError('Cooperative cache-only continuation cap')
    write(PUBLIC/'continuation_started.json',{'plan_sha256':sha(PLAN),'new_raw_decodes':0,'new_fits':0,'target_reads':0})
    try:
        preservation()
        for group in ('raw_receipts','private_bindings'):
            for name,digest in plan[group].items():
                if sha(REPO.parent/name,budget)!=digest:
                    raise ValueError('Bound source-only receipt/cache/state changed')
        expected=[json.loads(s) for s in (BASE/'faced_coverage_selection_2026-10-13/source_regions.jsonl').read_text().splitlines()]
        regional={(r['subject'],r['vid'],r['region']):r for r in expected}
        qual=load(PRIOR/'qualification.json'); available=[tuple(p) for p in qual['available_pairs']]
        with np.load(STATE_ROOT/'available_source_vectors.npz',allow_pickle=False) as saved:
            if saved.files!=['vectors']:
                raise ValueError('Unexpected source snapshot schema')
            vectors=saved['vectors'].copy()
        if (vectors.shape!=(979,3,150) or vectors.dtype!=np.float64 or not np.isfinite(vectors).all()
                or vectors.size!=plan['source_snapshot_scalar_cap'] or array_sha(vectors)!=qual['source_snapshot_array_sha256']):
            raise ValueError('Source snapshot geometry/hash differs')
        if available!=[p for p in pairs if p!=('sub-087',20)]:
            raise PermissionError('Qualified available source order differs')
        for i,pair in enumerate(available):
            budget()
            for j,region in enumerate(REGIONS):
                if array_sha(vectors[i,j])!=regional[*pair,region]['spectral_feature_sha256']:
                    raise ValueError('Every cached region must match prior spectral lineage')
        index={p:i for i,p in enumerate(available)}; calls=[]
        def feature_loader(pair,region):
            if pair not in index:
                raise PermissionError('Unavailable source callback forbidden')
            calls.append((*pair,region)); budget()
            return vectors[index[pair],REGIONS.index(region)].copy()
        metadata_rows=[{k:meta[p][k] for k in ('subject','vid','role','family','assigned_category','acquisition_group','catalogue_duration_seconds','trial')} for p in pairs]
        quality_rows=[{**{k:r[k] for k in ('subject','vid','role','region','eeg_available')},'technical_passed':True,'review_flags':r['reasons']} for r in expected]
        bindings={r['model']:{'path':STATE_ROOT/r['private_state_file'],'sha256':r['private_state_sha256'],'record':r} for r in plan['final_state_records']}
        prior_sha=plan['private_bindings']['publication_runs/faced_coverage_missing_2026-10-13/full_source_category_prior.json']
        bundle=load_frozen_bundle_v2(bindings,STATE_ROOT/'full_source_category_prior.json',prior_sha,plan['source_identity_sha256'])
        with threadpool_limits(limits=2,user_api='blas'):
            rows=infer_four_methods(bundle,metadata_rows,quality_rows,feature_loader,mode='source')
        fixtures=load(BASE/'faced_saved_state_qualification_2026-10-14/source_prediction_fixtures.json')
        delta=audit_predictions(rows,fixtures)
        if len(calls)!=2937 or len(set(calls))!=2937 or any(p=='sub-087' and v==20 for p,v,_ in calls):
            raise ValueError('Available-only complete callback denominator differs')
        output=PUBLIC/plan['new_output_directory']
        export_prediction_lock(output,rows,metadata_rows,quality_rows,mode='source',bundle=bundle)
        verified=verify_prediction_lock(output,metadata_rows,quality_rows,mode='source',bundle=bundle)
        exported=load(output/'predictions.json'); exported_delta=audit_predictions(exported,fixtures)
        for name,digest in plan['code_and_input_bindings'].items():
            if sha(REPO/name)!=digest:
                raise ValueError('Continuation code/public inputs changed')
        for group in ('raw_receipts','private_bindings'):
            for name,digest in plan[group].items():
                if sha(REPO.parent/name,budget)!=digest:
                    raise ValueError('Source receipt/cache/state changed during continuation')
        result={'date':'2026-10-14','passed':True,'status':'source_cache_continuation_and_export_passed',
                'plan_sha256':sha(PLAN),'original_failed_attempt_preserved':True,
                'source_trials':980,'source_prediction_rows':3920,'source_feature_callbacks':len(calls),
                'source_snapshot_scalars':vectors.size,'all2937cached_region_hashes_match':True,
                'existing_actual_raw_regions':27,'actual_raw_qc_access_receipt_hashes_match':54,
                'new_raw_decodes':0,'new_fits':0,'target_reads':0,'original_rating_reads':0,
                'shared_missing_inference_trials':1,'conditional_fallback_rows':3,'missing_prior_agreement_rows':4,
                'independent_algebra_max_prediction_discrepancy':delta,
                'export_roundtrip_max_prediction_discrepancy':exported_delta,'source_prediction_lock':verified,
                'source_prior_values_or_states_changed':False,'validation_access':False,'protected_access':False,
                'protected_exports_qualified':False,'confirmation_activated':False,
                'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'preservation':preservation(),
                'limits':'Raw replay was the preserved original held attempt; this separate continuation uses authenticated caches. Source/manufactured qualification does not authorize or prove protected execution.'}
        budget(); write(PUBLIC/'continuation_completion.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('preservation','source_prediction_lock')}),flush=True)
    except Exception as error:
        write(PUBLIC/'continuation_failure.json',{'status':'held_or_failed','error_type':type(error).__name__,'reason':str(error),
                'new_raw_decodes':0,'new_fits':0,'target_reads':0,'protected_access':False,'automatic_retry':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('declare','run'))
    (declare if parser.parse_args().action=='declare' else run)()
