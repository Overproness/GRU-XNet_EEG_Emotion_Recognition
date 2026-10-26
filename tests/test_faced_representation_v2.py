import unittest
from unittest.mock import patch
from threadpoolctl import threadpool_info,threadpool_limits
from scripts import run_faced_representation_v2 as runner


class PreprocessingReplayTests(unittest.TestCase):
    def test_preprocessing_restores_qualified_threads_only_within_operation(self):
        def manufactured(values,rate):
            self.assertTrue(all(p['num_threads']==12 for p in threadpool_info() if p['user_api']=='blas'))
            return values,{'rate':rate}
        with threadpool_limits(limits=2):
            with patch.object(runner,'artifact_trial_windows',manufactured):
                self.assertEqual(runner.original_preprocessing_windows('dummy',250),('dummy',{'rate':250}))
            self.assertTrue(all(p['num_threads']==2 for p in threadpool_info() if p['user_api']=='blas'))

    def test_failed_preprocessing_restores_readout_threads(self):
        with threadpool_limits(limits=2):
            with patch.object(runner,'artifact_trial_windows',side_effect=ValueError('manufactured')):
                with self.assertRaises(ValueError):runner.original_preprocessing_windows('dummy',250)
            self.assertTrue(all(p['num_threads']==2 for p in threadpool_info() if p['user_api']=='blas'))


if __name__=='__main__':unittest.main()
