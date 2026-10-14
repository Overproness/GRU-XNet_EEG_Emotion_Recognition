"""Declare/run a single source-cache and manufactured executor qualification.

No original recording/header/rating is statted, opened or hashed. No validation
or reserved numerical value is interpreted. This script never creates a
production activation. Every output is target-free qualification evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import scipy
import mne
from gruxnet.faced_guarded_confirmation import (CAPS,VERSION,Budget,declaration,hash_json,
    validate_scope,manufactured_scope_replay,write_lock,verify_lock,trial_features,
    prepare_processing_runtime,process_bdf_stream,private_memory_bytes,verify_bindings)
from gruxnet.faced_frozen_inference import array_sha256
from gruxnet.faced_frozen_prior_adapter import load_frozen_bundle_v2
from scripts.qualify_faced_source_export_2026_10_14 import metadata,preservation

BASE=REPO/'results/development'
OUT=BASE/'faced_guarded_confirmation_2026-10-14'
PLAN=OUT/'plan.json'
PRIOR=BASE/'faced_coverage_missing_2026-10-13'
PRIVATE=REPO.parent/'publication_runs/faced_coverage_missing_2026-10-13'
SCOPE='results/development/faced_confirmation_backend_2026-10-14/scope.json'
PACKAGE='results/development/faced_confirmation_backend_2026-10-14/package.json'
PROTOCOL='docs/publication/FACED_Guarded_Confirmation_Qualification_Protocol_2026-10-14.md'


def load(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
def runtime():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'mne':mne.__version__,'platform':sys.platform,'numerical_blas_threads':12,
            'private_memory_counter':'Windows private bytes' if sys.platform=='win32' else 'USS'}
def fixture_module():
    spec=importlib.util.spec_from_file_location('guard_fixture',REPO/'tests/test_faced_guarded_confirmation.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def declare():
    scope=load(REPO/SCOPE); validate_scope(scope); preserved=preservation()
    final=[json.loads(s) for s in (PRIOR/'fit_records.jsonl').read_text().splitlines()
           if json.loads(s).get('stage')=='full_source_state_qualification']
    if len(final)!=3 or {r['model'] for r in final}!={'context_only','late_joint','distributed_joint'}:
        raise PermissionError('Exactly three existing source states required')
    qual=load(PRIOR/'qualification.json'); state=load(PRIOR/'final_source_state_qualification.json')
    private={r['private_state_file']:r['private_state_sha256'] for r in final}
    private['full_source_category_prior.json']=state['private_category_prior_sha256']
    private['available_source_vectors.npz']=qual['source_snapshot_file_sha256']
    names=[SCOPE,PACKAGE,PROTOCOL,'gruxnet/faced_guarded_confirmation.py','gruxnet/faced_frozen_inference.py',
        'gruxnet/faced_frozen_prior_adapter.py','gruxnet/faced_source_quality.py','gruxnet/faced_artifact_control.py',
        'gruxnet/faced_proposal_preprocessing.py','gruxnet/faced_inference_quality.py','gruxnet/faced_temporal_coverage.py',
        'gruxnet/faced_source_notch.py','gruxnet/faced_coverage_missing.py',
        'gruxnet/faced_classical_contract.py','scripts/qualify_faced_guarded_confirmation_2026_10_14.py',
        'tests/test_faced_guarded_confirmation.py','scripts/run_faced_coverage_selection_2026_10_13.py',
        'scripts/qualify_faced_source_export_2026_10_14.py',
        'results/development/faced_raw_2026-10-11/headers.json',
        'results/development/faced_temporal_coverage_preflight_2026-10-12/plan.json',
        'results/development/faced_artifact_control_2026-10-11/source_records.json',
        'results/development/faced_codebooks_2026-10-10/audit.json',
        'results/development/faced_coverage_missing_2026-10-13/qualification.json',
        'results/development/faced_coverage_missing_2026-10-13/final_source_state_qualification.json',
        'results/development/faced_coverage_missing_2026-10-13/fit_records.jsonl',
        'results/development/faced_saved_state_qualification_2026-10-14/source_prediction_fixtures.json',
        'results/development/faced_source_export_qualification_2026-10-14/continuation_completion.json']
    plan={'date':'2026-10-14','status':'declared_source_cache_manufactured_only','executor_version':VERSION,
        'public_bindings':{n:sha(REPO/n) for n in names},'private_source_bindings':private,
        'source_identity_sha256':state['models'][0]['source_identity_sha256'],'final_records':final,
        'source_snapshot_shape':[979,3,150],'source_snapshot_scalar_cap':440550,
        'source_trials':980,'source_prediction_comparisons':3920,
        'manufactured_future_roster_trials':759,'manufactured_regions':2277,'manufactured_predictions':3036,
        'manufactured_raw_streams':2,'manufactured_raw_rates_hz':[250,1000],
        'manufactured_raw_count_scalar_cap':30*32*(250+1000),
        'source_cache_replay_tolerance':1e-10,'qualification_wall_cpu_seconds_cap':300,
        'production_resource_caps':CAPS,'runtime':runtime(),'single_attempt':True,
        'protected_original_stat_open_hash_reads':0,'original_rating_reads':0,'cached_source_target_reads':0,
        'validation_numeric_reads':0,'reserved_numeric_reads':0,'new_fits':0,'papers_changed':False,
        'activation_created':False,'protected_signal_access_authorized':False,
        'protected_rating_access_authorized':False,'preservation':preserved}
    write(PLAN,plan)
    print(json.dumps({'declared':True,'source_scalars':440550,'manufactured_trials':759,'new_fits':0}))


def source_bundle(plan):
    for name,expected in plan['private_source_bindings'].items():
        # Exactly the predeclared five source files, never arbitrary/original paths.
        if '/' in name or '\\' in name or name.startswith('.') or sha(PRIVATE/name)!=expected:
            raise PermissionError('Source-only frozen private binding changed')
    bindings={r['model']:{'path':PRIVATE/r['private_state_file'],'sha256':r['private_state_sha256'],'record':r}
              for r in plan['final_records']}
    return load_frozen_bundle_v2(bindings,PRIVATE/'full_source_category_prior.json',
        plan['private_source_bindings']['full_source_category_prior.json'],plan['source_identity_sha256'])


def replay_source(plan,bundle):
    _,_,meta,pairs=metadata(); qual=load(PRIOR/'qualification.json')
    fixtures=load(BASE/'faced_saved_state_qualification_2026-10-14/source_prediction_fixtures.json')
    if len(pairs)!=980 or any(meta[p]['role']!='source' for p in pairs):
        raise PermissionError('Exact complete genuine source-role roster required')
    available=[tuple(p) for p in qual['available_pairs']]
    if len(available)!=979 or len(set(available))!=979 or set(available)!=set(pairs)-{('sub-087',20)}:
        raise PermissionError('Frozen source availability rectangle changed')
    with np.load(PRIVATE/'available_source_vectors.npz',allow_pickle=False) as data:
        if data.files!=['vectors']: raise PermissionError('Exact source snapshot archive required')
        vectors=data['vectors'].copy()
    if (vectors.shape!=tuple(plan['source_snapshot_shape']) or vectors.size!=plan['source_snapshot_scalar_cap']
            or vectors.dtype!=np.float64 or not np.isfinite(vectors).all()
            or array_sha256(vectors)!=qual['source_snapshot_array_sha256']):
        raise ValueError('Qualified source snapshot values changed')
    index={p:i for i,p in enumerate(available)}; expected={(r['subject'],r['vid']):r for r in fixtures['rows']}
    if set(expected)!=set(pairs): raise PermissionError('Independent fixture source roster changed')
    maximum=0.; comparisons=0; missing=0
    for pair in pairs:
        row=meta[pair]; prior=bundle.category_prior[row['assigned_category']]
        features=trial_features(row,vectors[index[pair]]) if pair in index else None
        if features is None: missing+=1
        for model in ('category_median','context_only','late_joint','distributed_joint'):
            prediction=prior if model=='category_median' or features is None else float(bundle.states[model].predict(features[model][None])[0])
            reference=expected[pair]['category_median'] if model=='category_median' else expected[pair]['inference_predictions'][model]
            delta=abs(prediction-float(reference)); maximum=max(maximum,delta); comparisons+=1
            if not np.isfinite(delta) or delta>plan['source_cache_replay_tolerance']:
                raise ValueError('New source inference algebra differs from independent saved fixture')
    return {'source_trials':len(pairs),'source_scalar_values':vectors.size,'prediction_comparisons':comparisons,
            'source_missing_trials':missing,'maximum_prediction_difference':maximum,'passed':True,
            'source_targets_read':0,'fits':0,'raw_original_reads':0}


def manufactured_raw(rate,fixture,scope):
    row=next(dict(r) for r in scope['rows'] if r['native_rate_hz']==rate)
    row['segments']={'early':[2*rate+13,34*rate+13],'middle':[2*rate+13,34*rate+13],'late':[2*rate+13,34*rate+13]}
    header=fixture.manufactured_headers(scope)[row['subject']]; header['record_count']=40
    # Purely fabricated counts:32channels×40seconds; actual BDF record layout.
    samples=np.arange(40*rate)/rate
    values=np.stack([np.rint(30*np.sin(2*np.pi*(7+i*.1)*samples+i*.31)
                  +10*np.sin(2*np.pi*3*samples+i*.7)) for i in range(32)]).astype(np.int32)
    unsigned=(values.astype(np.int64)&0xffffff).astype(np.uint32)
    encoded=np.stack((unsigned&255,(unsigned>>8)&255,(unsigned>>16)&255),axis=-1).astype(np.uint8)
    body=encoded.reshape(32,40,rate,3).transpose(1,0,2,3).tobytes()
    a,b=row['segments']['early']; read_calls=30*((b-1)//rate-a//rate+1)
    resource={'two_data_plus_two_event_hash_transport_bytes':0,'data_header_bytes_one_read_pass':0,
              'selected_signal_payload_bytes':90*(b-a),'selected_count_values':30*(b-a),
              'selected_channel_payload_read_calls':read_calls}
    budget=Budget(resource)
    result=process_bdf_stream(io.BytesIO(bytes(8448)+body),header,row,'early',budget,input_kind='manufactured')
    budget.finish()
    selected=values[row['channel_indices'],a:b]
    if result['receipt']['count_sha256']!=array_sha256(selected):
        raise ValueError('Fabricated signed24 decoder differs from independently generated counts')
    return {'rate_hz':rate,'manufactured':True,'raw_count_values':30*(b-a),
            'signal_payload_bytes':90*(b-a),'read_calls':read_calls,
            'count_decode_independent_reference_passed':True,'finite_features':True,
            'technical_passed':result['technical_passed'],'engineered_qc_flags':result['review_flags'],
            'receipt':result['receipt'],'resource_counts':budget.counts}


def run():
    plan=load(PLAN); verify_bindings(REPO,plan['public_bindings']); preservation()
    if plan['runtime']!=runtime() or plan['activation_created'] is not False or plan['new_fits']!=0:
        raise PermissionError('Frozen source/manufactured runtime/scope changed')
    if (OUT/'started.json').exists(): raise PermissionError('Preserve single attempt; no automatic retry')
    write(OUT/'started.json',{'plan_sha256':sha(PLAN),'protected_access':False,'new_fits':0,'source_target_reads':0})
    started=time.monotonic(); cpu=time.process_time(); phase='source_bundle'
    def check():
        if (time.monotonic()-started>plan['qualification_wall_cpu_seconds_cap']
                or time.process_time()-cpu>plan['qualification_wall_cpu_seconds_cap']):
            raise TimeoutError('Bounded qualification wall/CPU cap')
        if private_memory_bytes()>CAPS['private_memory_bytes']: raise MemoryError('Qualification private memory cap')
    try:
        bundle=source_bundle(plan); check(); phase='source_cache'
        replay=replay_source(plan,bundle); check(); write(OUT/'source_replay.json',replay)
        scope=load(REPO/SCOPE); fixture=fixture_module(); phase='manufactured_full_scope'
        contract=declaration(scope,sha(REPO/PACKAGE),bundle.fingerprint,runtime(),plan['public_bindings'])
        write(OUT/'inactive_executor_contract.json',contract)
        reader=fixture.ManufacturedReader(missing=True)
        budget=Budget(scope['resource_scope'])
        predictions,ledger=manufactured_scope_replay(scope,bundle,reader,budget=budget); check()
        lock=OUT/'manufactured_prediction_lock'
        manifest=write_lock(lock,predictions,scope,bundle,kind='manufactured_qualification',contract_sha256=hash_json(contract))
        checked,serialized=verify_lock(lock,scope,bundle,kind='manufactured_qualification',contract_sha256=hash_json(contract))
        if checked!=manifest or len(serialized)!=3036: raise ValueError('Manufactured canonical roundtrip failed')
        phase='manufactured_raw_streams'; prepare_processing_runtime(); check()
        raw=[manufactured_raw(rate,fixture,scope) for rate in plan['manufactured_raw_rates_hz']]
        check(); write(OUT/'manufactured_raw.json',raw)
        verify_bindings(REPO,plan['public_bindings']); preservation()
        result={'date':'2026-10-14','passed':True,'status':'source_and_manufactured_qualification_complete',
            'plan_sha256':sha(PLAN),'source_replay':replay,'manufactured_full_scope_trials':759,
            'manufactured_regions':len(ledger),'manufactured_predictions':len(predictions),
            'manufactured_lock':manifest,'manufactured_resource_counts':budget.counts,
            'actual_protected_transport_bytes':0,'manufactured_raw_streams':raw,
            'elapsed_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
            'process_private_memory_bytes_at_completion':private_memory_bytes(),
            'original_stat_open_hash_reads':0,'source_targets_read':0,'validation_numeric_reads':0,
            'reserved_numeric_reads':0,'original_rating_reads':0,'new_fits':0,
            'protected_signal_access_authorized':False,'protected_rating_access_authorized':False,
            'actual_protected_exports_qualified':False,'production_activation_created':False,
            'actual_protected_throughput_qualified':False,'papers_changed':False,
            'limitations':'Synthetic full-scope bytes are counters, not real transport. Two manufactured raw streams qualify numerical decoder path, not original reserved files or757 remaining raw ranges. No actual author activation/rating access exists.',
            'output_bindings':{p.relative_to(REPO).as_posix():sha(p) for p in
                (OUT/'source_replay.json',OUT/'inactive_executor_contract.json',OUT/'manufactured_raw.json',
                 lock/'predictions.json',lock/'manifest.json',lock/'LOCK.complete.json')},
            'preservation':preservation()}
        write(OUT/'completion.json',result)
        print(json.dumps({'passed':True,'source_comparisons':3920,'source_max_difference':replay['maximum_prediction_difference'],
                          'manufactured_predictions':3036,'protected_access':False,'new_fits':0}),flush=True)
    except Exception as error:
        write(OUT/'failure.json',{'held':True,'phase':phase,'error_type':type(error).__name__,'reason':str(error),
            'new_fits':0,'protected_access':False,'source_targets_read':0,'automatic_retry':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('declare','run'))
    (declare if parser.parse_args().action=='declare' else run)()
