"""Manufactured durable-attempt tests. Original file callbacks are fabricated."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
import uuid

from gruxnet.faced_durable_confirmation import (CODE_PATH,PRODUCTION_DIRECTORY,
    durable_declaration,manufactured_durable_replay,DurableGuardedExecutor,SignalJournal,_safe_directory,_finish)
from gruxnet.faced_guarded_confirmation import declaration,hash_json,OriginalBDFReader
from gruxnet.faced_frozen_inference import sha256_bytes

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('durable_guard_fixtures',REPO/'tests/test_faced_guarded_confirmation.py')
fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.scope=json.loads((REPO/'results/development/faced_confirmation_backend_2026-10-14/scope.json').read_bytes())
        self.bundle=fixture.fixture_bundle(); self.runtime={'manufactured_fixture':True}
        temp=REPO/'tests/_tmp'; temp.mkdir(exist_ok=True)
        self.repo=temp/('durable-'+uuid.uuid4().hex); self.repo.mkdir()
        (self.repo/'gruxnet').mkdir(); data=(REPO/CODE_PATH).read_bytes()
        (self.repo/CODE_PATH).write_bytes(data)
        self.core=declaration(self.scope,'c'*64,self.bundle.fingerprint,self.runtime,{CODE_PATH:sha256_bytes(data)})
        self.relative='tests/_tmp/manufactured_attempt'
        self.contract=durable_declaration(self.core,repo=self.repo,directory=self.relative,
            purpose='manufactured',wrapper_sha256=sha256_bytes(data))

    def run_fixture(self,reader=None,contract=None):
        return manufactured_durable_replay(self.contract if contract is None else contract,self.scope,self.bundle,
            fixture.ManufacturedReader() if reader is None else reader,repo=self.repo,runtime=self.runtime,
            budget=fixture.test_budget(self.scope))

    def test_complete_manufactured_attempt_and_fresh_process_restart(self):
        complete,manifest=self.run_fixture()
        self.assertEqual((complete['regions'],complete['predictions'],complete['target_values']),(2277,3036,0))
        self.assertEqual(manifest['kind'],'manufactured_qualification')
        path=self.repo/self.relative
        self.assertTrue((path/'SIGNAL_ATTEMPT.reserved.json').is_file())
        self.assertTrue((path/'SIGNAL_ATTEMPT.complete.json').is_file())
        reader=fixture.ManufacturedReader()
        with self.assertRaises(FileExistsError): self.run_fixture(reader,copy.deepcopy(self.contract))
        self.assertEqual(reader.calls,0)

    def test_failure_is_durable_and_new_instance_cannot_restart(self):
        reader=fixture.ManufacturedReader(technical=True)
        with self.assertRaises(PermissionError): self.run_fixture(reader)
        failure=json.loads((self.repo/self.relative/'SIGNAL_ATTEMPT.failure.json').read_bytes())
        self.assertEqual(failure['attempted_regions'],1); self.assertFalse(failure['automatic_retry'])
        self.assertFalse((self.repo/self.relative/'SIGNAL_ATTEMPT.complete.json').exists())
        fresh=fixture.ManufacturedReader()
        with self.assertRaises(FileExistsError): self.run_fixture(fresh)
        self.assertEqual(fresh.calls,0)

    def test_reserved_receipt_write_failure_still_blocks_restart(self):
        reader=fixture.ManufacturedReader()
        with patch.object(SignalJournal,'_write',side_effect=OSError('manufactured write failure')):
            with self.assertRaises(OSError): self.run_fixture(reader)
        self.assertTrue((self.repo/self.relative).is_dir()); self.assertEqual(reader.calls,0)
        with self.assertRaises(FileExistsError): self.run_fixture(reader)
        self.assertEqual(reader.calls,0)

    def test_old_core_contract_not_final_activation_eligible(self):
        with self.assertRaises(PermissionError): self.run_fixture(contract=self.core)

    def test_mutated_contract_before_first_callback(self):
        for field,value in (('scope_sha256','e'*64),('bundle_fingerprint','f'*64),('runtime',{'changed':True})):
            c=copy.deepcopy(self.contract); c[field]=value; reader=fixture.ManufacturedReader()
            with self.assertRaises(PermissionError): self.run_fixture(reader,c)
            self.assertEqual(reader.calls,0)
        self.assertFalse((self.repo/self.relative).exists())

    def test_wrong_repository_or_runtime_precedes_reservation(self):
        c=copy.deepcopy(self.contract); c['durable_attempt']['repository_root']=str(self.repo.parent)
        with self.assertRaises(PermissionError): self.run_fixture(contract=c)
        self.assertFalse((self.repo/self.relative).exists())

    def test_path_escape_and_wrong_production_directory_denied(self):
        for p in ('../escape','/absolute','D:/absolute','results/confirmation/unbound'):
            c=durable_declaration(self.core,repo=self.repo,directory=p,purpose='protected',
                wrapper_sha256=self.core['bindings'][CODE_PATH])
            executor=DurableGuardedExecutor(c,self.scope,self.bundle,repo=self.repo,runtime=self.runtime)
            with self.assertRaises(PermissionError): executor.signals(OriginalBDFReader(self.repo.parent,{}))

    def test_symlink_or_junction_ancestor_denied(self):
        target=self.repo/'tests'
        with patch.object(Path,'is_symlink',new=lambda p:p==target):
            with self.assertRaises(PermissionError): _safe_directory(self.repo,self.relative)
        real_resolve=Path.resolve
        def redirected(p,*args,**kwargs):
            return self.repo/'elsewhere' if p==target else real_resolve(p,*args,**kwargs)
        target.mkdir()
        with patch.object(Path,'resolve',new=redirected):
            with self.assertRaises(PermissionError): _safe_directory(self.repo,self.relative)

    def test_original_callback_provenance_denied_by_manufactured_entry(self):
        with self.assertRaises(PermissionError): self.run_fixture(OriginalBDFReader(self.repo.parent,{}))
        self.assertFalse((self.repo/self.relative).exists())

    def test_false_author_phase_precedes_attempt_reservation(self):
        c=durable_declaration(self.core,repo=self.repo,directory=PRODUCTION_DIRECTORY,
            purpose='protected',wrapper_sha256=self.core['bindings'][CODE_PATH])
        executor=DurableGuardedExecutor(c,self.scope,self.bundle,repo=self.repo,runtime=self.runtime)
        with self.assertRaises(PermissionError): executor.signals(OriginalBDFReader(self.repo.parent,{}))
        self.assertFalse((self.repo/PRODUCTION_DIRECTORY).exists())

    def test_real_wrapper_ordering_with_fabricated_permissions_and_callbacks(self):
        # Both author barriers are intentionally fabricated in this test only.
        # The fake repository, signals and resulting lock stay under ignored
        # tests/_tmp; no real production instruction or original reader runs.
        c=durable_declaration(self.core,repo=self.repo,directory=PRODUCTION_DIRECTORY,
            purpose='protected',wrapper_sha256=self.core['bindings'][CODE_PATH])
        expected=self.repo/PRODUCTION_DIRECTORY
        fake=fixture.ManufacturedReader()
        fake.input_kind='protected_original'
        class FabricatedOriginal(OriginalBDFReader):
            def __init__(inner): inner.workspace=self.repo.parent; inner.headers={}
            def authenticate_projection(inner,*args):
                self.assertTrue((expected/'SIGNAL_ATTEMPT.reserved.json').is_file())
            def preflight(inner,*args):
                self.assertTrue((expected/'SIGNAL_ATTEMPT.reserved.json').is_file())
                return fake.preflight(*args)
            def region(inner,*args): return fake.region(*args)
            def finish(inner,*args): return fake.finish(*args)
        executor=DurableGuardedExecutor(c,self.scope,self.bundle,repo=self.repo,runtime=self.runtime,
            budget=fixture.test_budget(self.scope))
        with patch('gruxnet.faced_durable_confirmation._activation'),patch('gruxnet.faced_guarded_confirmation._activation'):
            complete,_=executor.signals(FabricatedOriginal())
        self.assertEqual(complete['predictions'],3036)
        fresh=DurableGuardedExecutor(c,self.scope,self.bundle,repo=self.repo,runtime=self.runtime,
            budget=fixture.test_budget(self.scope))
        before=fake.calls
        with patch('gruxnet.faced_durable_confirmation._activation'),patch('gruxnet.faced_guarded_confirmation._activation'):
            with self.assertRaises(FileExistsError): fresh.signals(FabricatedOriginal())
        self.assertEqual(fake.calls,before)

    def test_production_callback_failure_ordering_and_restart(self):
        c=durable_declaration(self.core,repo=self.repo,directory=PRODUCTION_DIRECTORY,
            purpose='protected',wrapper_sha256=self.core['bindings'][CODE_PATH])
        path=self.repo/PRODUCTION_DIRECTORY; calls=[]
        class FabricatedFailure(OriginalBDFReader):
            def __init__(inner): inner.workspace=self.repo.parent; inner.headers={}
            def authenticate_projection(inner,*args):
                self.assertTrue((path/'SIGNAL_ATTEMPT.reserved.json').is_file()); calls.append('projection')
                raise OSError('Fabricated projection failure; no original path')
        for index in range(2):
            executor=DurableGuardedExecutor(c,self.scope,self.bundle,repo=self.repo,runtime=self.runtime,
                budget=fixture.test_budget(self.scope))
            with patch('gruxnet.faced_durable_confirmation._activation'),patch('gruxnet.faced_guarded_confirmation._activation'):
                with self.assertRaises(OSError if index==0 else FileExistsError): executor.signals(FabricatedFailure())
        self.assertEqual(calls,['projection'])
        self.assertTrue((path/'SIGNAL_ATTEMPT.failure.json').is_file())
        self.assertFalse((path/'SIGNAL_ATTEMPT.complete.json').exists())

    def test_scope_and_contract_caller_mutations_do_not_change_snapshot(self):
        c=durable_declaration(self.core,repo=self.repo,directory=PRODUCTION_DIRECTORY,
            purpose='protected',wrapper_sha256=self.core['bindings'][CODE_PATH])
        executor=DurableGuardedExecutor(c,self.scope,self.bundle,repo=self.repo,runtime=self.runtime)
        old=hash_json(executor.scope); self.scope['rows'][-1]['role']='source'; c['scope_sha256']='f'*64
        self.assertEqual(hash_json(executor.scope),old); self.assertNotEqual(executor.contract['scope_sha256'],'f'*64)

    def test_budget_exhaustion_during_lock_prevents_completion(self):
        journal=SimpleNamespace(completed=2277,purpose='manufactured',path=self.repo,_write=Mock())
        budget=Mock(); budget.check.side_effect=[None,None,TimeoutError('Fabricated post-lock budget ceiling')]
        manifest={'fabricated':'manifest'}
        with patch('gruxnet.faced_durable_confirmation.write_lock',return_value=manifest), \
             patch('gruxnet.faced_durable_confirmation.verify_lock',return_value=(manifest,[])):
            with self.assertRaises(TimeoutError):
                _finish(journal,self.scope,self.bundle,[{}]*3036,[{}]*2277,self.contract,budget=budget)
        journal._write.assert_not_called()


if __name__=='__main__': unittest.main()
