import unittest
import numpy as np
from gruxnet.faced_baseline_result_audit import checked_panel, audit_ridge, independent_flags

class Trap:
    def __float__(self): raise AssertionError('Forbidden premature outcome conversion')

class AuditTests(unittest.TestCase):
    def row(self):
        return dict(fit='fixed',epoch='360',arm='held_source',subject='source',vid='1',family='a',target=Trap(),prediction_raw=Trap())

    def test_last_unknown_participant_fails_before_any_outcome(self):
        a=self.row(); b=self.row(); b['subject']='reserved'
        with self.assertRaises(PermissionError):
            checked_panel([a,b],{('fixed',360,'held_source','source',1)},{('source',1):'a'})

    def test_missing_or_duplicate_before_numerics(self):
        a=self.row(); expected={('fixed',360,'held_source','source',1)}
        for rows in ([],[a,a]):
            with self.assertRaises(PermissionError): checked_panel(rows,expected,{('source',1):'a'})

    def test_ridge_known_solution_and_corrupted_state(self):
        x=np.array([[-1.,0.],[0.,0.],[1.,0.]])
        y=np.array([1.,2.,3.]); scale=np.array([np.sqrt(2/3),1.])
        coefficient=np.array([2/scale[0]/103,0.])
        self.assertLess(audit_ridge(x,y,[0.,0.],scale,coefficient,2.),1e-12)
        with self.assertRaises(ValueError): audit_ridge(x,y,[0.,0.],scale,coefficient*2,2.)
        with self.assertRaises(ValueError): audit_ridge(x,y,[1.,0.],scale,coefficient,2.)

    def test_flags_do_not_promote_improvement_to_independent_utility(self):
        self.assertEqual(independent_flags(2.,[1.,1.],[.89,.9],[1.,1.]),['budget_limited_convergence'])

if __name__=='__main__': unittest.main()
