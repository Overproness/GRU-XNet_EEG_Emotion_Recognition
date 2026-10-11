"""Access-boundary tests use only manufactured identities and no original files."""
import unittest
from unittest.mock import Mock,patch
from gruxnet.faced_classical_runner import source_gate,activate_after_source
from scripts.run_faced_artifact_control import Execution,save

class ExecutionTests(unittest.TestCase):
    def test_unplanned_identity_rejected_before_authentication(self):
        x=Execution.__new__(Execution)
        x.activation={'source_pairs':[{'subject':'sub-001','vid':11}]}
        x.authenticate=Mock(side_effect=AssertionError('No file operation allowed'))
        for pairs in ([('sub-002',11)],[('sub-001',22)],[('sub-001',11),('sub-001',11)]):
            with self.assertRaises(PermissionError):x.prepare(pairs,'source')
        x.authenticate.assert_not_called()

    def test_validation_and_targets_rejected_when_source_held(self):
        x=Execution.__new__(Execution);x.activation={'validation_pairs':[{'subject':'sub-002','vid':11}]}
        x.gate={'passed':False};x.authenticate=Mock()
        with self.assertRaises(PermissionError):x.prepare([('sub-002',11)],'validation')
        with self.assertRaises(PermissionError):x.open_targets({('sub-001',11)})
        x.authenticate.assert_not_called()

    def test_capacity_flag_cannot_open_targets_or_fit(self):
        rows=[{'subject':'sub-001','vid':11,'technical_passed':True,
            'review_flags':[{'channel':'multiple','flag':'repair_capacity_exceeded'}]}]
        loader=Mock();self.assertFalse(source_gate(rows,[('sub-001',11)])['passed'])
        with self.assertRaises(PermissionError):activate_after_source(rows,[('sub-001',11)],loader)
        loader.assert_not_called()

    def test_receipt_overwrite_rejected_before_write(self):
        path=Mock();path.name='existing.json';path.exists.return_value=True
        with self.assertRaises(ValueError):save(path,{})
        path.open.assert_not_called()

if __name__=='__main__':unittest.main()
