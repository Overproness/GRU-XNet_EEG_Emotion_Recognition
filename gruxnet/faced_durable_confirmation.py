"""Durable signal-attempt amendment; frozen original executor stays unchanged.

No activation or original reader is created here. A failed attempt directory
remains occupied across new Python processes; retry needs a reviewed amendment.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path,PurePosixPath

from .faced_guarded_confirmation import (VERSION,OriginalBDFReader,GuardedExecutor,
    _activation,hash_json,validate_scope,verify_bindings,manufactured_scope_replay,
    write_lock,verify_lock)
from .faced_frozen_inference import canonical_bytes

DURABLE_VERSION='faced-durable-signal-attempt-v1'
CODE_PATH='gruxnet/faced_durable_confirmation.py'
PRODUCTION_DIRECTORY='results/confirmation/faced_four_method_2026-10-15'
ENTRY_POINT='gruxnet.faced_durable_confirmation.DurableGuardedExecutor.signals'


def _root_name(path):
    return os.path.normcase(os.path.abspath(path))


def durable_declaration(core_contract,*,repo,directory,purpose,wrapper_sha256):
    if purpose not in ('manufactured','protected'):
        raise PermissionError('Explicit attempt purpose required')
    contract=copy.deepcopy(core_contract)
    if contract.get('version')!=VERSION or contract.get('status')!='inactive_not_authorized':
        raise PermissionError('Unchanged inactive core contract required')
    contract['bindings'][CODE_PATH]=wrapper_sha256
    contract['required_entry_point']=ENTRY_POINT
    contract['durable_attempt']={'version':DURABLE_VERSION,'repository_root':_root_name(repo),
        'relative_directory':directory,'purpose':purpose,'single_attempt':True,
        'failure_directory_reused':False,'old_core_contract_final_activation_eligible':False}
    return contract


def _check(contract,scope,bundle,repo,runtime,*,purpose):
    if (contract.get('version')!=VERSION or contract.get('status')!='inactive_not_authorized'
            or contract.get('scope_sha256')!=hash_json(scope)
            or contract.get('bundle_fingerprint')!=bundle.fingerprint or contract.get('runtime')!=runtime
            or contract.get('required_entry_point')!=ENTRY_POINT):
        raise PermissionError('Durable contract/scope/bundle/runtime changed')
    validate_scope(scope)
    d=contract.get('durable_attempt',{})
    if (set(d)!={'version','repository_root','relative_directory','purpose','single_attempt',
                  'failure_directory_reused','old_core_contract_final_activation_eligible'}
            or d['version']!=DURABLE_VERSION or d['repository_root']!=_root_name(repo)
            or d['purpose']!=purpose or d['single_attempt'] is not True
            or d['failure_directory_reused'] is not False
            or d['old_core_contract_final_activation_eligible'] is not False
            or CODE_PATH not in contract.get('bindings',{})):
        raise PermissionError('Exact new durable-attempt contract required; old core template is insufficient')
    relative=d['relative_directory']; p=PurePosixPath(relative)
    if (not isinstance(relative,str) or p.is_absolute() or str(p)!=relative
            or '..' in p.parts or '\\' in relative or ':' in relative):
        raise PermissionError('Unsafe attempt path')
    if purpose=='protected' and relative!=PRODUCTION_DIRECTORY:
        raise PermissionError('Exact public production attempt directory required')
    if purpose=='manufactured' and not relative.startswith(('tests/_tmp/','results/development/faced_durable_confirmation_2026-10-15/')):
        raise PermissionError('Manufactured attempt must remain in its qualification scope')
    verify_bindings(repo,contract['bindings'])
    return relative


def _safe_directory(repo,relative):
    root=Path(repo).resolve(); lexical=root/relative
    # Check each existing ancestor before mkdir; junctions and symlink parents
    # cannot redirect an exact bound path even to another location inside repo.
    current=root
    for part in PurePosixPath(relative).parts:
        current=current/part
        if current.is_symlink() or (current.exists() and current.resolve()!=current):
            raise PermissionError('Attempt path contains a symlink/junction/redirected ancestor')
    if not lexical.resolve().is_relative_to(root):
        raise PermissionError('Attempt directory escapes bound repository')
    return lexical


class SignalJournal:
    def __init__(self,repo,relative,contract,scope,*,purpose):
        self.path=_safe_directory(repo,relative); self.path.parent.mkdir(parents=True,exist_ok=True)
        # Atomic directory reservation is durable even if subsequent writes fail.
        self.path.mkdir(exist_ok=False)
        self.contract=contract; self.scope=scope; self.purpose=purpose; self.attempted=0; self.completed=0
        self._write('SIGNAL_ATTEMPT.reserved.json',{'version':DURABLE_VERSION,
            'contract_sha256':hash_json(contract),'scope_sha256':hash_json(scope),
            'bundle_fingerprint':contract['bundle_fingerprint'],'runtime_sha256':hash_json(contract['runtime']),
            'relative_directory':relative,'purpose':purpose,'phase':'signal_prediction',
            'status':'durably_reserved_before_first_original_callback','target_values':0})

    def _write(self,name,value):
        with (self.path/name).open('xb') as stream:
            stream.write(canonical_bytes(value)); stream.flush(); os.fsync(stream.fileno())

    def event(self,event,row=None,region=None,**values):
        payload={'event':event,'target_values':0,**values}
        if row is not None:
            payload|={k:row[k] for k in ('subject','vid','role','arm')}; payload['region']=region
        with (self.path/'progress.jsonl').open('ab') as stream:
            stream.write(canonical_bytes(payload)); stream.flush(); os.fsync(stream.fileno())

    def failure(self,error):
        self._write('SIGNAL_ATTEMPT.failure.json',{'version':DURABLE_VERSION,'status':'held_or_failed',
            'error_type':type(error).__name__,'reason':str(error),'attempted_regions':self.attempted,
            'completed_regions':self.completed,'purpose':self.purpose,'target_values':0,'automatic_retry':False})


class _LoggingOriginal(OriginalBDFReader):
    def __init__(self,delegate,journal):
        self.delegate=delegate; self.journal=journal; self.workspace=delegate.workspace; self.headers=delegate.headers
    def authenticate_projection(self,*args): return self.delegate.authenticate_projection(*args)
    def preflight(self,*args):
        self.journal.event('preflight_started'); result=self.delegate.preflight(*args)
        self.journal.event('preflight_completed'); return result
    def region(self,row,region,budget):
        self.journal.attempted+=1; self.journal.event('region_started',row,region)
        result=self.delegate.region(row,region,budget)
        self.journal.completed+=1; self.journal.event('region_completed',row,region)
        return result
    def finish(self,*args):
        result=self.delegate.finish(*args); self.journal.event('integrity_postflight_completed'); return result


class _LoggingManufactured:
    input_kind='manufactured'
    def __init__(self,delegate,journal): self.delegate=delegate; self.journal=journal
    def preflight(self,*args):
        self.journal.event('preflight_started'); result=self.delegate.preflight(*args)
        self.journal.event('preflight_completed'); return result
    def region(self,row,region,budget):
        self.journal.attempted+=1; self.journal.event('region_started',row,region)
        result=self.delegate.region(row,region,budget)
        self.journal.completed+=1; self.journal.event('region_completed',row,region)
        return result
    def finish(self,*args):
        result=self.delegate.finish(*args); self.journal.event('integrity_postflight_completed'); return result


def _finish(journal,scope,bundle,predictions,ledger,contract,*,budget):
    budget.check()
    if journal.completed!=2277 or len(ledger)!=2277 or len(predictions)!=3036:
        raise PermissionError('Durable attempt cannot complete a partial denominator')
    kind='protected_prediction' if journal.purpose=='protected' else 'manufactured_qualification'
    lock=journal.path/'prediction_lock'
    manifest=write_lock(lock,predictions,scope,bundle,kind=kind,contract_sha256=hash_json(contract))
    budget.check()
    checked,_=verify_lock(lock,scope,bundle,kind=kind,contract_sha256=hash_json(contract))
    if checked!=manifest: raise PermissionError('Durable prediction lock mismatch')
    budget.check()
    complete={'version':DURABLE_VERSION,'status':'signal_and_prediction_stage_completed',
              'purpose':journal.purpose,'regions':2277,'predictions':3036,'target_values':0,
              'contract_sha256':hash_json(contract),'manifest_sha256':hash_json(manifest),
              'predictions_committed':False,'rating_access_authorized':False}
    journal._write('SIGNAL_ATTEMPT.complete.json',complete)
    return complete,manifest


class DurableGuardedExecutor:
    def __init__(self,contract,scope,bundle,*,repo,runtime,activation=None,receipts=None,budget=None):
        # Caller mutations cannot silently change the checked roster/contract.
        self.contract=copy.deepcopy(contract); self.scope=copy.deepcopy(scope)
        self.bundle=bundle; self.repo=repo; self.runtime=copy.deepcopy(runtime)
        self.activation=copy.deepcopy(activation); self.receipts=copy.deepcopy(receipts or {}); self.budget=budget
    def signals(self,reader):
        relative=_check(self.contract,self.scope,self.bundle,self.repo,self.runtime,purpose='protected')
        _activation(self.contract,self.scope,self.bundle,self.activation,self.receipts,self.runtime,'signal_prediction',self.repo)
        if not isinstance(reader,OriginalBDFReader): raise PermissionError('Protected original adapter required')
        # Durable reservation precedes even authenticate_projection/preflight.
        journal=SignalJournal(self.repo,relative,self.contract,self.scope,purpose='protected')
        try:
            executor=GuardedExecutor(self.contract,self.scope,self.bundle,activation=self.activation,
                receipts=self.receipts,runtime=self.runtime,root=self.repo,budget=self.budget)
            predictions,ledger=executor.signals(_LoggingOriginal(reader,journal))
            return _finish(journal,self.scope,self.bundle,predictions,ledger,self.contract,budget=executor.budget)
        except Exception as error:
            journal.failure(error); raise


def manufactured_durable_replay(contract,scope,bundle,reader,*,repo,runtime,budget):
    frozen_contract=copy.deepcopy(contract); frozen_scope=copy.deepcopy(scope)
    relative=_check(frozen_contract,frozen_scope,bundle,repo,runtime,purpose='manufactured')
    if getattr(reader,'input_kind',None)!='manufactured':
        raise PermissionError('Manufactured amendment cannot invoke actual original adapter')
    journal=SignalJournal(repo,relative,frozen_contract,frozen_scope,purpose='manufactured')
    try:
        predictions,ledger=manufactured_scope_replay(frozen_scope,bundle,_LoggingManufactured(reader,journal),budget=budget)
        return _finish(journal,frozen_scope,bundle,predictions,ledger,frozen_contract,budget=budget)
    except Exception as error:
        journal.failure(error); raise
