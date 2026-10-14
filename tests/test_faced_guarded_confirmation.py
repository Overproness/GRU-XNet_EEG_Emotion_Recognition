"""Manufactured phase/range/export failures; no protected recording access."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import uuid
import unittest
from unittest.mock import patch

import numpy as np

from gruxnet.faced_frozen_inference import FrozenBundle,FrozenRidge,METHODS,array_sha256
from gruxnet.faced_classical_contract import CATEGORIES
from gruxnet.faced_guarded_confirmation import (ARMS, CAPS, READINESS, VERSION,
    Budget, GuardedExecutor, OriginalBDFReader, declaration, hash_json, validate_scope,
    manufactured_scope_replay, validate_predictions, write_lock, verify_lock,
    read_item10, verify_bindings, trial_features, process_bdf_stream,prepare_processing_runtime,
    validate_public_headers,verify_committed_lock)
from gruxnet.faced_proposal_preprocessing import COMMON_CHANNELS

REPO=Path(__file__).resolve().parents[1]
SCOPE_PATH=REPO/'results/development/faced_confirmation_backend_2026-10-14/scope.json'
TEST_TMP=REPO/'tests/_tmp'


def fixture_bundle():
    states={}
    for i,m in enumerate(METHODS[1:],1):
        n=13 if m=='context_only' else 163
        coefficient=np.zeros(n,dtype=np.float64); coefficient[0]=.2
        if n>13: coefficient[13:]=.0001*i
        states[m]=FrozenRidge(m,coefficient,np.asarray(3.,dtype=np.float64),
            np.zeros(n,dtype=np.float64),np.ones(n,dtype=np.float64),np.zeros(n,dtype=np.float64),
            None if n==13 else np.zeros(150,dtype=np.float64),str(i)*64,'a'*64)
    return FrozenBundle(states,{c:3.5 for c in CATEGORIES},'b'*64,'a'*64)


class ManufacturedReader:
    input_kind='manufactured'
    def __init__(self,*,missing=False,technical=False,nonfinite=False,bad_receipt=False):
        self.calls=0; self.missing=missing; self.technical=technical; self.nonfinite=nonfinite; self.bad_receipt=bad_receipt
    def preflight(self,scope,files,budget):
        budget.consume('integrity_bytes',budget.limits['integrity_bytes']//2)
        budget.consume('header_bytes',budget.limits['header_bytes'])
    def region(self,row,region,budget):
        self.calls+=1; a,b=row['segments'][region]; r=row['native_rate_hz']
        budget.consume('count_values',30*(b-a)); budget.consume('signal_bytes',90*(b-a))
        budget.consume('read_calls',30*((b-1)//r-a//r+1))
        value=np.full(150,row['vid']*.001+('early','middle','late').index(region)*.01,dtype=np.float64)
        if self.nonfinite and self.calls==1: value[0]=np.nan
        receipt={k:row[k] for k in ('subject','vid','role','arm')}|{'region':region,
            'sample_interval':row['segments'][region],'input_kind':self.input_kind,
            'extra_boundary_samples_decoded':0,'feature_sha256':array_sha256(value)}
        if self.bad_receipt and self.calls==1: receipt['sample_interval']=[0,1]
        return {'technical_passed':not(self.technical and self.calls==1),
                'review_flags':[{'channel':'Fp2','reason':'adc_rail'}] if self.missing and self.calls==1 else [],
                'features':value,'receipt':receipt}
    def finish(self,scope,files,budget):
        budget.consume('integrity_bytes',budget.limits['integrity_bytes']-budget.counts['integrity_bytes'])


def test_budget(scope,**kwargs):
    return Budget(scope['resource_scope'],memory=lambda:0,clock=lambda:0,cpu=lambda:0,**kwargs)


def manufactured_headers(scope):
    from gruxnet.faced_proposal_preprocessing import UNIT_REPAIR_SUBJECTS
    headers={}
    for row in scope['rows']:
        p=row['subject']; rate=row['native_rate_hz']; labels=['']*32
        for i,c in zip(row['channel_indices'],COMMON_CHANNELS): labels[i]=c
        extras=[i for i,c in enumerate(labels) if not c]
        for i,c in zip(extras,('A1','A2')): labels[i]=c
        if extras==[16,17]:
            aliases={'T7':'T3','T8':'T4','P7':'T5','P8':'T6'}
            labels=[aliases.get(c,c) for c in labels]
        unit='?V' if p in UNIT_REPAIR_SUBJECTS else 'uV'
        headers[p]={'header_bytes':8448,'sample_width':3,'continuity':'BDF',
            'record_duration_seconds':1.,'record_count':row['recording_samples']//rate,
            'sample_rates_hz':[float(rate)]*32,'header_sha256':row['original_header_sha256'],
            'columns':{'labels':labels,'units':[unit]*32,'samples_per_record':[str(rate)]*32,
                'physical_min':['-8388608']*32,'physical_max':['8388607']*32,
                'digital_min':['-8388608']*32,'digital_max':['8388607']*32}}
    return headers


def activation_fixture(scope,bundle):
    required=('gruxnet/faced_guarded_confirmation.py','gruxnet/faced_frozen_inference.py',
        'gruxnet/faced_source_quality.py','gruxnet/faced_artifact_control.py','gruxnet/faced_source_notch.py',
        'gruxnet/faced_proposal_preprocessing.py','gruxnet/faced_inference_quality.py','gruxnet/faced_classical_contract.py',
        'gruxnet/faced_temporal_coverage.py','gruxnet/faced_coverage_missing.py',
        'gruxnet/faced_frozen_prior_adapter.py','results/development/faced_raw_2026-10-11/headers.json',
        'results/development/faced_confirmation_backend_2026-10-14/scope.json',
        'results/development/faced_confirmation_backend_2026-10-14/package.json')
    from gruxnet.faced_frozen_inference import sha256_bytes
    bindings={n:sha256_bytes((REPO/n).read_bytes()) for n in required}
    contract=declaration(scope,bindings[required[-1]],bundle.fingerprint,{'manufactured_test':True},bindings)
    receipts={n:{'version':VERSION,'scope_sha256':hash_json(scope),'contract_sha256':hash_json(contract),
        'passed':True,'protected_data_accessed':False} for n in READINESS}
    activation={'version':VERSION,'phase':'signal_prediction','contract_sha256':hash_json(contract),
        'scope_sha256':hash_json(scope),'package_sha256':contract['package_sha256'],
        'bundle_fingerprint':bundle.fingerprint,'author':'Muhammad Muntazar',
        'author_instruction':'In-memory manufactured fixture only; this is not an author instruction.',
        'authorization_receipt_sha256':'c'*64,'authorization_receipt_path':'docs/publication/nonexistent-fixture.json',
        'readiness_receipts':{n:hash_json(r) for n,r in receipts.items()},'signal_access_authorized':True,
        'rating_access_authorized':False,'single_attempt':True,'new_fits':0}
    return contract,activation,receipts


class GuardedConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope=json.loads(SCOPE_PATH.read_text()); cls.bundle=fixture_bundle()
        TEST_TMP.mkdir(exist_ok=True)
        cls.predictions,cls.ledger=manufactured_scope_replay(cls.scope,cls.bundle,ManufacturedReader(missing=True),budget=test_budget(cls.scope))

    def test_exact_scope_and_real_roles_preserved(self):
        rows,files=validate_scope(self.scope)
        self.assertEqual((len(rows),len(files),len(self.predictions),len(self.ledger)),(759,66,3036,2277))
        self.assertEqual({p['role'] for p in self.predictions},{'confirmation'})
        self.assertEqual({p['arm'] for p in self.predictions},set(ARMS))
        self.assertEqual(sum(p['arm']==ARMS[0] for p in self.predictions),1188)
        self.assertEqual(sum(p['arm']==ARMS[1] for p in self.predictions),1848)

    def test_missing_region_common_category_fallback_all_methods(self):
        p=self.predictions[:4]
        self.assertEqual([r['prediction'] for r in p],[3.5]*4)
        self.assertEqual([r['common_category_fallback'] for r in p],[False,True,True,True])
        self.assertTrue(all(not r['shared_eeg_available'] for r in p))

    def test_bad_late_identity_fails_before_first_callback(self):
        scope=copy.deepcopy(self.scope); scope['rows'][-1]['role']='source'; reader=ManufacturedReader()
        with self.assertRaises(PermissionError): manufactured_scope_replay(scope,self.bundle,reader,budget=test_budget(scope))
        self.assertEqual(reader.calls,0)

    def test_changed_region_range_fails_before_callback(self):
        scope=copy.deepcopy(self.scope); scope['rows'][-1]['segments']['early'][0]+=1; reader=ManufacturedReader()
        with self.assertRaises(PermissionError): manufactured_scope_replay(scope,self.bundle,reader,budget=test_budget(scope))
        self.assertEqual(reader.calls,0)

    def test_scope_duplicate_denied(self):
        scope=copy.deepcopy(self.scope); scope['rows'][-1]=scope['rows'][0]
        with self.assertRaises(PermissionError): validate_scope(scope)

    def test_wrong_file_path_denied_before_reader(self):
        for path in ('../Data/data.bdf','Data/Data/sub000/evt.bdf','D:/Data/data.bdf'):
            scope=copy.deepcopy(self.scope); scope['files'][0]['workspace_relative_path']=path
            with self.assertRaises(PermissionError): validate_scope(scope)

    def test_changed_file_hash_disagrees_with_rows(self):
        scope=copy.deepcopy(self.scope); scope['files'][0]['sha256']='f'*64
        with self.assertRaises(PermissionError): validate_scope(scope)

    def test_resource_metadata_mutation_denied(self):
        scope=copy.deepcopy(self.scope); scope['resource_scope']['selected_count_values']+=1
        with self.assertRaises(PermissionError): validate_scope(scope)

    def test_technical_failure_not_missing(self):
        with self.assertRaises(PermissionError):
            manufactured_scope_replay(self.scope,self.bundle,ManufacturedReader(technical=True),budget=test_budget(self.scope))

    def test_nonfinite_features_fatal(self):
        with self.assertRaises(ValueError):
            manufactured_scope_replay(self.scope,self.bundle,ManufacturedReader(nonfinite=True),budget=test_budget(self.scope))

    def test_range_receipt_fails(self):
        with self.assertRaises(PermissionError):
            manufactured_scope_replay(self.scope,self.bundle,ManufacturedReader(bad_receipt=True),budget=test_budget(self.scope))

    def test_no_actual_original_reader_in_manufactured_replay(self):
        reader=OriginalBDFReader('no-original-path-opened',{})
        with self.assertRaises(PermissionError): manufactured_scope_replay(self.scope,self.bundle,reader,budget=test_budget(self.scope))

    def test_inactive_production_guard_before_callback(self):
        contract=declaration(self.scope,'c'*64,self.bundle.fingerprint,{'test':True},{})
        executor=GuardedExecutor(contract,self.scope,self.bundle,runtime={'test':True})
        reader=ManufacturedReader()
        with self.assertRaises(PermissionError): executor.signals(reader)
        self.assertEqual(reader.calls,0)

    def test_old_three_method_activation_denied(self):
        contract=declaration(self.scope,'c'*64,self.bundle.fingerprint,{}, {})
        executor=GuardedExecutor(contract,self.scope,self.bundle,activation={'signal_access_authorized':True},runtime={})
        with self.assertRaises(PermissionError): executor.signals(ManufacturedReader())

    def test_original_adapter_direct_preflight_denied(self):
        reader=OriginalBDFReader('not-opened',{})
        with patch.object(Path,'stat',side_effect=AssertionError('Original stat forbidden')):
            with self.assertRaises(PermissionError): reader.preflight(self.scope,{},test_budget(self.scope))

    def test_fabricated_legacy_and_modern_header_geometry(self):
        headers=manufactured_headers(self.scope)
        self.assertTrue(any(h['columns']['labels'][16]=='A1' for h in headers.values()))
        self.assertTrue(any(h['columns']['labels'][-2:]==['A1','A2'] for h in headers.values()))
        self.assertIs(validate_public_headers(self.scope,headers),headers)

    def test_bad_last_header_stops_before_original_path(self):
        headers=manufactured_headers(self.scope); person=sorted(headers)[-1]; headers[person]['columns']['units'][0]='mV'
        reader=OriginalBDFReader('not-opened',headers); budget=test_budget(self.scope)
        budget._production_scope_sha256=hash_json(self.scope)
        with patch.object(reader,'_path',side_effect=AssertionError('Original callback forbidden')):
            with self.assertRaises(PermissionError): reader.preflight(self.scope,validate_scope(self.scope)[1],budget)

    def test_readiness_failures_and_runtime_phase_stop_before_reader(self):
        contract,activation,receipts=activation_fixture(self.scope,self.bundle)
        for name in READINESS:
            broken=copy.deepcopy(receipts); broken[name]['passed']=False
            executor=GuardedExecutor(contract,self.scope,self.bundle,activation=activation,receipts=broken,
                runtime=contract['runtime'],root=REPO)
            reader=ManufacturedReader()
            with self.assertRaises(PermissionError): executor.signals(reader)
            self.assertEqual(reader.calls,0)
        for field,value in (('phase','item10_rating'),('author','Another Author'),('scope_sha256','e'*64)):
            broken=dict(activation); broken[field]=value
            with self.assertRaises(PermissionError):
                GuardedExecutor(contract,self.scope,self.bundle,activation=broken,receipts=receipts,
                    runtime=contract['runtime'],root=REPO).signals(ManufacturedReader())
        with self.assertRaises(PermissionError):
            GuardedExecutor(contract,self.scope,self.bundle,activation=activation,receipts=receipts,
                runtime={'changed':True},root=REPO).signals(ManufacturedReader())

    def test_fabricated_all_true_flags_are_not_author_approval(self):
        contract,activation,receipts=activation_fixture(self.scope,self.bundle)
        # No author receipt is created by the test. All true flags still stop
        # before the reader because the concrete saved instruction is absent.
        reader=ManufacturedReader()
        with self.assertRaises((PermissionError,FileNotFoundError)):
            GuardedExecutor(contract,self.scope,self.bundle,activation=activation,receipts=receipts,
                runtime=contract['runtime'],root=REPO).signals(reader)
        self.assertEqual(reader.calls,0)

    def test_counterfeit_commit_receipt_requires_actual_git_blobs(self):
        class Output:
            def __init__(self,value): self.stdout=value
        tmp=TEST_TMP/('guarded-'+uuid.uuid4().hex); tmp.mkdir(); root=tmp/'lock'
        manifest=write_lock(root,self.predictions,self.scope,self.bundle,kind='manufactured_qualification',contract_sha256='c'*64)
        with patch('gruxnet.faced_guarded_confirmation.subprocess.run',side_effect=[Output(b'2'),Output(b'{}')]):
            with self.assertRaises(PermissionError): verify_committed_lock(REPO,root,'a'*40,manifest)

    def test_memory_ceiling(self):
        with self.assertRaises(MemoryError): Budget(self.scope['resource_scope'],memory=lambda:2*1024**3+1,clock=lambda:0,cpu=lambda:0)

    def test_wall_cpu_and_checkpoint_gap_ceiling(self):
        for clock,cpu in (([0,0,10801],[0,0,0]),([0,0,0],[0,0,10801]),([0,0,6],[0,0,0])):
            b=Budget(self.scope['resource_scope'],memory=lambda:0,clock=lambda:clock.pop(0),cpu=lambda:cpu.pop(0))
            with self.assertRaises(TimeoutError): b.check()

    def test_resource_ceiling_before_read(self):
        b=test_budget(self.scope)
        with self.assertRaises(PermissionError): b.consume('signal_bytes',b.limits['signal_bytes']+1)
        self.assertEqual(b.counts['signal_bytes'],0)

    def test_incomplete_resource_denominator(self):
        with self.assertRaises(PermissionError): test_budget(self.scope).finish()

    def test_partial_prediction_denominator_rejected(self):
        with self.assertRaises(PermissionError): validate_predictions(self.predictions[:-1],self.scope,self.bundle)

    def test_target_field_and_role_mask_rejected(self):
        for field,value in (('target',3.),('role','manufactured')):
            predictions=copy.deepcopy(self.predictions); predictions[-1][field]=value
            with self.assertRaises(PermissionError): validate_predictions(predictions,self.scope,self.bundle)

    def test_state_or_fallback_mutation_denied(self):
        for field,value in (('state_sha256','e'*64),('prediction',3.1)):
            predictions=copy.deepcopy(self.predictions); predictions[0][field]=value
            with self.assertRaises(PermissionError): validate_predictions(predictions,self.scope,self.bundle)

    def test_canonical_lock_roundtrip_and_no_retry(self):
        tmp=TEST_TMP/('guarded-'+uuid.uuid4().hex); tmp.mkdir()
        if True:
            self.assertTrue(tmp.resolve().is_relative_to(TEST_TMP.resolve()))
            root=Path(tmp)/'lock'; manifest=write_lock(root,self.predictions,self.scope,self.bundle,kind='manufactured_qualification',contract_sha256='c'*64)
            loaded,rows=verify_lock(root,self.scope,self.bundle,kind='manufactured_qualification',contract_sha256='c'*64)
            self.assertEqual(loaded,manifest); self.assertEqual(len(rows),3036)
            with self.assertRaises(FileExistsError): write_lock(root,self.predictions,self.scope,self.bundle,kind='manufactured_qualification',contract_sha256='c'*64)
            with self.assertRaises(PermissionError): verify_lock(root,self.scope,self.bundle,kind='protected_prediction',contract_sha256='c'*64)

    def test_partial_or_corrupt_lock_rejected(self):
        tmp=TEST_TMP/('guarded-'+uuid.uuid4().hex); tmp.mkdir()
        if True:
            self.assertTrue(tmp.resolve().is_relative_to(TEST_TMP.resolve()))
            root=Path(tmp)/'lock'; write_lock(root,self.predictions,self.scope,self.bundle,kind='manufactured_qualification',contract_sha256='c'*64)
            (root/'LOCK.complete.json').write_bytes(b'{}\n')
            with self.assertRaises(PermissionError): verify_lock(root,self.scope,self.bundle,kind='manufactured_qualification',contract_sha256='c'*64)

    def test_rating_phase_inactive_before_reader(self):
        class Reader:
            input_kind='protected_item10_only'
            def item10(self,*args,**kwargs): raise AssertionError('No rating callback permitted')
        contract=declaration(self.scope,'c'*64,self.bundle.fingerprint,{}, {})
        with self.assertRaises(PermissionError): read_item10(contract,self.scope,self.bundle,None,{}, {},REPO,'no-lock',{}, {},Reader())

    def test_public_binding_hash_and_original_path_fail(self):
        with self.assertRaises(PermissionError): verify_bindings(REPO,{'gruxnet/faced_guarded_confirmation.py':'a'*64})
        with self.assertRaises(PermissionError): verify_bindings(REPO,{'Data/Data/sub000/data.bdf':'a'*64})

    def test_trial_features_shared_source_algebra(self):
        row=dict(self.scope['rows'][0]); regions=[np.full(150,i,dtype=np.float64) for i in (1,2,3)]
        x=trial_features(row,regions); self.assertTrue(np.array_equal(x['late_joint'][13:],regions[2]))
        self.assertTrue(np.array_equal(x['distributed_joint'][13:],regions[1]))
        row['role']='source'; self.assertTrue(np.array_equal(trial_features(row,regions)['distributed_joint'],x['distributed_joint']))


if __name__=='__main__': unittest.main()
