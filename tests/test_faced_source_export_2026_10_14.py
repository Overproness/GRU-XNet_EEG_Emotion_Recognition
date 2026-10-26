"""Failure guards for the bounded raw replay, without opening recordings."""
import unittest
from scripts.qualify_faced_source_export_2026_10_14 import audit_segment, audit_predictions
from gruxnet.faced_temporal_coverage import SourceOnlySegmentGuard


class SourceBackendGuards(unittest.TestCase):
    def test_protected_and_validation_rejected_before_reader(self):
        segment={'subject':'sub-001','vid':2,'role':'source','region':'early','native_rate_hz':250,'sample_interval':[0,8000]}
        guard=SourceOnlySegmentGuard(source_people=['sub-001'],validation_people=['sub-002'],protected_people=['sub-003'],source_pairs=[('sub-001',2)],pilot_rows=[segment])
        reads=[]
        for person in ['sub-002','sub-003','sub-004']:
            with self.assertRaises(PermissionError):
                guard.read(subject=person,vid=2,region='early',sample_interval=[0,8000],native_rate_hz=250,reader=lambda *args:reads.append(args))
        self.assertEqual(reads,[])

    def test_widened_range_rejected_before_reader(self):
        segment={'subject':'sub-001','vid':2,'role':'source','region':'early','native_rate_hz':250,'sample_interval':[0,8000]}
        guard=SourceOnlySegmentGuard(source_people=['sub-001'],validation_people=[],protected_people=[],source_pairs=[('sub-001',2)],pilot_rows=[segment])
        reads=[]
        with self.assertRaises(PermissionError):
            guard.read(subject='sub-001',vid=2,region='early',sample_interval=[0,8001],native_rate_hz=250,reader=lambda *args:reads.append(args))
        self.assertEqual(reads,[])

    def test_exact_failure_replay_is_a_pass_not_exclusion(self):
        failed={'subject':'sub-001','vid':2,'role':'source','region':'middle','eeg_available':False,'reasons':['high_processed_excursion']}
        audit_segment(failed,dict(failed))
        with self.assertRaises(ValueError):
            audit_segment({**failed,'eeg_available':True},failed)

    def test_native_range_qc_hash_and_repair_changes_hold(self):
        expected={'subject':'sub-001','vid':2,'role':'source','region':'early','sample_interval':[1,8001],
                  'native_rate_hz':250,'eeg_available':True,'reasons':[],'count_array_sha256':'a','processed_window_sha256':'b',
                  'spectral_feature_sha256':'c','range_ledger_sha256':'d','repair_count':1,'artifact_decision':'repair'}
        for key in ('sample_interval','eeg_available','spectral_feature_sha256','repair_count','range_ledger_sha256'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit_segment({**expected,key:None},expected)

    def test_independent_export_comparison_rejects_changed_value_or_duplicates(self):
        fixture={'rows':[{'subject':'sub-001','vid':2,'shared_eeg_available':False,'category_median':3.0,
                         'inference_predictions':dict(context_only=3.0,late_joint=3.0,distributed_joint=3.0)}]}
        rows=[dict(subject='sub-001',vid=2,model=m,prediction=3.0,shared_eeg_available=False)
              for m in ('category_median','context_only','late_joint','distributed_joint')]
        self.assertEqual(audit_predictions(rows,fixture),0.0)
        with self.assertRaises(ValueError):
            audit_predictions(rows[:-1]+[{**rows[-1],'prediction':2.9}],fixture)
        with self.assertRaises(ValueError):
            audit_predictions(rows[:-1]+[rows[0]],fixture)


if __name__=='__main__':
    unittest.main()
