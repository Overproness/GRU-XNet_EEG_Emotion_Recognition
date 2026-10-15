"""Declare/run outcome-free manufactured durable-attempt qualification only.

No private array, fitted state, original recording/header/rating or cached
outcome is opened. The old source/manufactured qualification remains preserved.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.faced_durable_confirmation import (DURABLE_VERSION,CODE_PATH,PRODUCTION_DIRECTORY,ENTRY_POINT,
    durable_declaration,manufactured_durable_replay)
from gruxnet.faced_guarded_confirmation import Budget,CAPS,hash_json,verify_bindings,private_memory_bytes
from scripts.qualify_faced_guarded_confirmation_2026_10_14 import runtime,preservation

OUT=REPO/'results/development/faced_durable_confirmation_2026-10-15'
OLD=REPO/'results/development/faced_guarded_confirmation_2026-10-14'
SCOPE='results/development/faced_confirmation_backend_2026-10-14/scope.json'
PROTOCOL='docs/publication/FACED_Durable_Confirmation_Qualification_Protocol_2026-10-15.md'
SCRIPT='scripts/qualify_faced_durable_confirmation_2026_10_15.py'
TEST='tests/test_faced_durable_confirmation.py'
PLAN=OUT/'plan_v2.json'


def load(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')
def fixture():
    spec=importlib.util.spec_from_file_location('durable_qualification_fixture',REPO/'tests/test_faced_guarded_confirmation.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def declare():
    prior=load(OLD/'completion.json'); core=load(OLD/'inactive_executor_contract.json')
    if (prior['passed'] is not True or prior['production_activation_created'] is not False
            or prior['actual_protected_exports_qualified'] is not False):
        raise PermissionError('Preserved source/manufactured qualification required')
    verify_bindings(REPO,core['bindings']); preserved=preservation()
    names=[CODE_PATH,TEST,SCRIPT,PROTOCOL,SCOPE,
           'gruxnet/faced_guarded_confirmation.py','tests/test_faced_guarded_confirmation.py',
           'scripts/qualify_faced_guarded_confirmation_2026_10_14.py',
           'results/development/faced_guarded_confirmation_2026-10-14/plan.json',
           'results/development/faced_guarded_confirmation_2026-10-14/completion.json',
           'results/development/faced_guarded_confirmation_2026-10-14/inactive_executor_contract.json']
    bindings={n:sha(REPO/n) for n in names}
    updated=copy.deepcopy(core); updated['bindings'].update(bindings)
    # This retains the actual qualified source bundle fingerprint from public
    # metadata. No actual bundle or its numerical arrays are loaded.
    future=durable_declaration(updated,repo=REPO,directory=PRODUCTION_DIRECTORY,
        purpose='protected',wrapper_sha256=bindings[CODE_PATH])
    write(OUT/'inactive_durable_executor_contract_v2.json',future)
    bindings[(OUT/'inactive_durable_executor_contract_v2.json').relative_to(REPO).as_posix()]=sha(OUT/'inactive_durable_executor_contract_v2.json')
    plan={'date':'2026-10-15','status':'declared_manufactured_only_not_activated','version':DURABLE_VERSION,
        'public_bindings':bindings,'preserved_core_bindings':core['bindings'],
        'core_qualification_freeze':'f4b62649c','prior_qualification_plan_sha256':sha(OLD/'plan.json'),
        'required_production_entry_point':ENTRY_POINT,'old_core_contract_final_activation_eligible':False,
        'future_signal_directory':PRODUCTION_DIRECTORY,'manufactured_full_trials':759,
        'manufactured_full_regions':2277,'manufactured_predictions':3036,
        'manufactured_failure_callbacks':1,'manufactured_restart_callbacks':0,
        'fresh_process_restart_probes':2,
        'supersedes_unexecuted_plan_sha256':sha(OUT/'plan_v1_unexecuted.json'),
        'qualification_wall_cpu_seconds_cap':300,'private_memory_bytes_cap':CAPS['private_memory_bytes'],
        'runtime':runtime(),'single_attempt':True,'new_fits':0,'private_source_array_reads':0,
        'original_file_stat_open_hash_reads':0,'original_rating_reads':0,'cached_target_conversions':0,
        'validation_numeric_reads':0,'reserved_numeric_reads':0,'activation_created':False,
        'papers_changed':False,'protected_signal_access_authorized':False,
        'protected_rating_access_authorized':False,'preservation':preserved}
    write(PLAN,plan)
    print(json.dumps({'declared':True,'manufactured_trials':759,'private_source_reads':0,'activation_created':False}),flush=True)


def probe_reservation(label):
    """A fresh Python interpreter can only inspect/reserve manufactured paths."""
    from gruxnet.faced_durable_confirmation import SignalJournal
    if label not in ('manufactured_complete','manufactured_failure'):
        raise PermissionError('Exact manufactured probe identity required')
    contracts=load(OUT/'manufactured_contracts.json'); contract=contracts[label]
    if contract['durable_attempt']['purpose']!='manufactured':
        raise PermissionError('Separate-process probe cannot address protected attempt')
    relative=(OUT/label).relative_to(REPO).as_posix()
    if contract['durable_attempt']['relative_directory']!=relative:
        raise PermissionError('Probe path differs from declared manufactured attempt')
    try:
        SignalJournal(REPO,relative,contract,load(REPO/SCOPE),purpose='manufactured')
    except FileExistsError:
        memory=private_memory_bytes()
        if memory>CAPS['private_memory_bytes']:
            raise MemoryError('Restart-probe private-memory ceiling')
        print(json.dumps({'fresh_process_restart_rejected':True,'callbacks':0,'label':label,
            'process_cpu_seconds':time.process_time(),'process_private_memory_bytes':memory}),flush=True)
    else:
        raise AssertionError('Fresh process unexpectedly created an occupied attempt directory')


def run():
    plan=load(PLAN); verify_bindings(REPO,plan['public_bindings'])
    verify_bindings(REPO,plan['preserved_core_bindings']); preservation()
    if (plan['runtime']!=runtime() or plan['activation_created'] is not False
            or plan['private_source_array_reads']!=0 or plan['new_fits']!=0):
        raise PermissionError('Frozen manufactured-only runtime/scope changed')
    write(OUT/'started.json',{'plan_sha256':sha(PLAN),'new_fits':0,'original_reads':0,'private_source_reads':0})
    started=time.monotonic(); cpu=time.process_time(); child_cpu=0.; stage='manufactured_full'
    def check():
        if (time.monotonic()-started>plan['qualification_wall_cpu_seconds_cap']
                or time.process_time()-cpu+child_cpu>plan['qualification_wall_cpu_seconds_cap']):
            raise TimeoutError('Durable qualification wall/CPU ceiling')
        if private_memory_bytes()>plan['private_memory_bytes_cap']:
            raise MemoryError('Durable qualification private-memory ceiling')
    try:
        scope=load(REPO/SCOPE); f=fixture(); bundle=f.fixture_bundle()
        core=load(OLD/'inactive_executor_contract.json')
        # Explicit synthetic qualification fork: replace only the bundle digest
        # with a newly manufactured bundle. This fork cannot authorize originals.
        core['bundle_fingerprint']=bundle.fingerprint
        core['bindings'].update(plan['public_bindings'])
        contracts={}; results={}
        for label in ('manufactured_complete','manufactured_failure'):
            relative=(OUT/label).relative_to(REPO).as_posix()
            contracts[label]=durable_declaration(core,repo=REPO,directory=relative,purpose='manufactured',
                wrapper_sha256=plan['public_bindings'][CODE_PATH])
        write(OUT/'manufactured_contracts.json',contracts)
        reader=f.ManufacturedReader(missing=True)
        completed,manifest=manufactured_durable_replay(contracts['manufactured_complete'],scope,bundle,reader,
            repo=REPO,runtime=plan['runtime'],budget=Budget(scope['resource_scope']))
        check()
        results['full']={'completed':completed,'manifest':manifest,'callbacks':reader.calls}
        fresh=f.ManufacturedReader()
        try:
            manufactured_durable_replay(copy.deepcopy(contracts['manufactured_complete']),scope,bundle,fresh,
                repo=REPO,runtime=plan['runtime'],budget=Budget(scope['resource_scope']))
        except FileExistsError:
            results['completed_restart']={'rejected':True,'callbacks':fresh.calls}
        else: raise AssertionError('Completed attempt accepted a fresh-instance restart')
        stage='manufactured_failure'; failed=f.ManufacturedReader(technical=True)
        try:
            manufactured_durable_replay(contracts['manufactured_failure'],scope,bundle,failed,
                repo=REPO,runtime=plan['runtime'],budget=Budget(scope['resource_scope']))
        except PermissionError:
            failure=load(OUT/'manufactured_failure/SIGNAL_ATTEMPT.failure.json')
            if failure['attempted_regions']!=1 or (OUT/'manufactured_failure/SIGNAL_ATTEMPT.complete.json').exists():
                raise AssertionError('Partial failure ledger/complete status differs')
            results['failure']={'expected_failure_preserved':True,'callbacks':failed.calls,'receipt':failure}
        else: raise AssertionError('Technical failure did not stop durable attempt')
        fresh=f.ManufacturedReader()
        try:
            manufactured_durable_replay(copy.deepcopy(contracts['manufactured_failure']),scope,bundle,fresh,
                repo=REPO,runtime=plan['runtime'],budget=Budget(scope['resource_scope']))
        except FileExistsError:
            results['failed_restart']={'rejected':True,'callbacks':fresh.calls}
        else: raise AssertionError('Failed attempt accepted a fresh-instance restart')
        subprocess_probes=[]
        for label in ('manufactured_complete','manufactured_failure'):
            check()
            output=subprocess.run([sys.executable,str(Path(__file__)),'probe-reservation','--label',label],
                cwd=REPO,check=True,capture_output=True,text=True,timeout=30)
            value=json.loads(output.stdout)
            if (set(value)!={'fresh_process_restart_rejected','callbacks','label','process_cpu_seconds','process_private_memory_bytes'}
                    or value['fresh_process_restart_rejected'] is not True or value['callbacks']!=0 or value['label']!=label
                    or not 0<=value['process_private_memory_bytes']<=plan['private_memory_bytes_cap']
                    or not 0<=value['process_cpu_seconds']<=30):
                raise AssertionError('Fresh-process reservation probe differs')
            child_cpu+=value['process_cpu_seconds']; subprocess_probes.append(value); check()
        results['separate_process_restart_probes']=subprocess_probes
        if (reader.calls!=2277 or failed.calls!=1
                or results['completed_restart']['callbacks']!=0 or results['failed_restart']['callbacks']!=0):
            raise AssertionError('Manufactured callback denominator/restart ordering differs')
        check(); verify_bindings(REPO,plan['public_bindings']); verify_bindings(REPO,plan['preserved_core_bindings']); preservation()
        outputs=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('completion.json','started.json','plan.json','plan_v2.json','plan_v1_unexecuted.json')]
        report={'date':'2026-10-15','passed':True,'status':'manufactured_durable_qualification_complete',
            'plan_sha256':sha(PLAN),'results':results,'elapsed_seconds':time.monotonic()-started,
            'parent_cpu_seconds':time.process_time()-cpu,'restart_child_cpu_seconds':child_cpu,
            'combined_parent_child_cpu_seconds':time.process_time()-cpu+child_cpu,
            'process_private_memory_bytes':private_memory_bytes(),
            'private_source_array_reads':0,'original_file_stat_open_hash_reads':0,'cached_target_conversions':0,
            'validation_numeric_reads':0,'reserved_numeric_reads':0,'original_rating_reads':0,'new_fits':0,
            'production_activation_created':False,'protected_signal_access_authorized':False,
            'protected_rating_access_authorized':False,'actual_protected_throughput_qualified':False,
            'old_core_contract_final_activation_eligible':False,'required_production_entry_point':ENTRY_POINT,
            'papers_changed':False,'output_bindings':{p.relative_to(REPO).as_posix():sha(p) for p in outputs},
            'preservation':preservation(),
            'limits':'Manufactured inputs and fake state only; logical resource counters, not actual original transport. Workflow safeguards cannot prevent manually bypassing Python entry points. No actual protected/rating activation, target decoder or protected lock qualified.'}
        write(OUT/'completion.json',report)
        print(json.dumps({'passed':True,'manufactured_predictions':3036,'complete_and_failed_restart_callbacks':0,
            'private_source_reads':0,'original_reads':0,'activation_created':False}),flush=True)
    except Exception as error:
        write(OUT/'failure.json',{'held':True,'stage':stage,'error_type':type(error).__name__,'reason':str(error),
            'new_fits':0,'private_source_reads':0,'original_reads':0,'automatic_retry':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('declare','run','probe-reservation'))
    parser.add_argument('--label',choices=('manufactured_complete','manufactured_failure'))
    args=parser.parse_args()
    if args.action=='probe-reservation': probe_reservation(args.label)
    else: (declare if args.action=='declare' else run)()
