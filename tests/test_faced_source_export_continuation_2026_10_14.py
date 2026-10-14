import unittest
from scripts.continue_faced_source_export_2026_10_14 import validate_hold


class ReviewedHold(unittest.TestCase):
    def test_only_exact_complete_original_schema_hold_is_eligible(self):
        plan=dict(raw_trials=9,raw_regions=27,count_scalar_cap=19440000,protected_access=False)
        failure=dict(status='held_or_failed',reason='Unexpected full-source prior schema or support',
                     raw_regions_completed=27,count_scalars_decoded=19440000,new_fits=0,target_reads=0,
                     validation_access=False,protected_access=False)
        validate_hold(plan,failure)
        for key,value in [('raw_regions_completed',26),('reason','different'),('new_fits',1),('target_reads',1),('protected_access',True)]:
            with self.subTest(key=key),self.assertRaises(PermissionError):
                validate_hold(plan,{**failure,key:value})


if __name__=='__main__':
    unittest.main()
