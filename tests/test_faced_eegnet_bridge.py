"""Original-access gates tested with fabricated windows; no BDF/MAT/target I/O."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from gruxnet.faced_eegnet_bridge import authoritative_bundle,WindowBridge,guard_cached_identity,array_sha256

BASE=Path(__file__).resolve().parents[1]/'results/development'


def fixture():
    load=lambda p:json.loads((BASE/p).read_bytes())
    programme=load('faced_development_protocol_2026-10-11/plan.json')
    reservation=load('faced_backup_2026-10-10/participant_reservation.json')
    catalogue={r['clip']:r for r in load('faced_codebooks_2026-10-10/audit.json')['stimuli']}
    records=load('faced_artifact_control_2026-10-11/source_records.json')['records']+load('faced_artifact_control_2026-10-11/validation_records.json')['records']
    decisions={(r['subject'],r['vid']):r for r in load('faced_inference_quality_2026-10-11/signal_audit.json')['trial_decisions']}
    records,metadata=authoritative_bundle(programme,reservation,catalogue,records,decisions)
    return programme,reservation,catalogue,records,metadata,decisions


class BridgeTests(unittest.TestCase):
    def synthetic_bridge(self):
        _,_,_,records,metadata,decisions=fixture();values=np.zeros((7,30,512),dtype=np.float32)
        for r in records.values():r['processed_window_sha256']=array_sha256(values)
        read=[]
        def replay(r):
            pair=r['subject'],r['vid'];read.append(pair)
            return values,{k:decisions[pair][k] for k in ('policy_id','eeg_available','reasons')},{'samples_decoded':7*30*512}
        return WindowBridge(records,metadata,decisions,replay),read

    def test_authoritative_roles_not_just_caller_geometry(self):
        programme,reservation,catalogue,records,metadata,decisions=fixture()
        bad=copy.deepcopy(reservation);bad['development_source'][0]=bad['confirmation'][0]
        with self.assertRaises(PermissionError):authoritative_bundle(programme,bad,catalogue,list(records.values()),decisions)
        bad=copy.deepcopy(list(records.values()));bad[0]['role']='confirmation'
        with self.assertRaises(PermissionError):authoritative_bundle(programme,reservation,catalogue,bad,decisions)

    def test_protected_pairs_and_pre_gate_access_never_call_backend(self):
        bridge,read=self.synthetic_bridge()
        with self.assertRaises(PermissionError):bridge.prepare(('sub-000',7))
        with self.assertRaises(PermissionError):bridge.prepare(bridge.validation[0])
        with self.assertRaises(PermissionError):bridge.require_targets_gate()
        self.assertEqual(read,[])

    def test_all_source_before_targets_then_full_fallback_denominator(self):
        bridge,read=self.synthetic_bridge();gate=bridge.prepare_source()
        self.assertTrue(gate['passed']);self.assertFalse(gate['targets_joined']);bridge.require_targets_gate()
        self.assertEqual(len(read),980);bridge.prepare_validation()
        self.assertEqual(len(read),1245);self.assertEqual(len(bridge.receipts),1260)
        failed=[p for p,d in bridge.decisions.items() if not d['eeg_available']]
        self.assertEqual(len(failed),15);self.assertFalse(set(read)&set(failed))
        self.assertFalse(bridge.windows[bridge.source[0]].flags.writeable)

    def test_changed_window_bytes_holds_targets(self):
        bridge,_=self.synthetic_bridge();bridge.records[bridge.source[0]]['processed_window_sha256']='wrong'
        with self.assertRaises(ValueError):bridge.prepare_source()
        with self.assertRaises(PermissionError):bridge.require_targets_gate()

    def test_changed_quality_holds_targets(self):
        bridge,_=self.synthetic_bridge();old=bridge.replay
        def wrong(r):
            x,d,receipt=old(r);d['eeg_available']=False;d['reasons']=[{'channel':'all','reason':'rail'}];return x,d,receipt
        bridge.replay=wrong
        with self.assertRaises(PermissionError):bridge.prepare_source()
        self.assertFalse(bridge.source_passed)

    def test_metadata_guard_precedes_numeric_outcome(self):
        _,reservation,_,_,metadata,_=fixture()
        class Row(dict):
            def __getitem__(self,key):
                if key in ('target','prediction'):raise AssertionError('Numeric outcome touched')
                return super().__getitem__(key)
        row=Row(subject=reservation['confirmation'][0],vid='2',family='unknown')
        with self.assertRaises(PermissionError):guard_cached_identity(row,metadata)

    def test_memory_cap_and_duplicate_replay(self):
        bridge,_=self.synthetic_bridge();bridge.memory_cap=1
        with self.assertRaises(MemoryError):bridge.prepare_source()
        bridge,_=self.synthetic_bridge();bridge.prepare(bridge.source[0])
        with self.assertRaises(PermissionError):bridge.prepare(bridge.source[0])


if __name__=='__main__':unittest.main()
