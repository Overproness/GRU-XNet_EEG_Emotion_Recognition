import unittest
import numpy as np
from gruxnet.faced_baseline_result_audit_v2 import audit_ridge,checked_panel

class Trap:
    def __float__(self): raise AssertionError('Premature outcome conversion')

class Tests(unittest.TestCase):
    def test_invalid_scalar_parameters_fail(self):
        x=np.array([[-1.,0.],[0.,0.],[1.,0.]])
        y=np.array([1.,2.,3.]); scale=np.array([np.sqrt(2/3),1.]); coef=np.array([2/scale[0]/103,0.])
        for intercept,alpha in ((np.nan,100.),(np.inf,100.),(2.,np.nan),(2.,np.inf),(2.,0.),(2.,-1.),([2.],100.)):
            with self.assertRaises(ValueError): audit_ridge(x,y,[0.,0.],scale,coef,intercept,alpha)
        self.assertLess(audit_ridge(x,y,[0.,0.],scale,coef,2.),1e-12)

    def test_late_reserved_identity_prevents_outcome_conversion(self):
        row=dict(fit='fixed',epoch='360',arm='held_source',subject='source',vid='1',family='a',target=Trap(),prediction_raw=Trap())
        bad={**row,'subject':'reserved'}
        with self.assertRaises(PermissionError):
            checked_panel([row,bad],{('fixed',360,'held_source','source',1)},{('source',1):'a'})

if __name__=='__main__': unittest.main()
