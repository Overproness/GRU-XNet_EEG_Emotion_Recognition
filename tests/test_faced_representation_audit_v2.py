import unittest
import numpy as np
from scipy.spatial.distance import pdist
from scripts.audit_faced_representation_v2 import median_nonduplicate_squared_distance


class IndependentBandwidthTests(unittest.TestCase):
    def test_mixed_exact_duplicates_match_direct_differences(self):
        rng=np.random.default_rng(619);z=rng.normal(size=(120,13));z[40:60]=z[:20]
        distances=pdist(z,metric='sqeuclidean')
        self.assertAlmostEqual(median_nonduplicate_squared_distance(z),float(np.median(distances[distances>0])),places=12)

    def test_distinct_high_dimensional_source_rows_match(self):
        z=np.random.default_rng(37).normal(size=(24,2330))
        self.assertAlmostEqual(median_nonduplicate_squared_distance(z),float(np.median(pdist(z,metric='sqeuclidean'))),places=10)

    def test_all_identical_source_rows_hold(self):
        with self.assertRaises(ValueError):median_nonduplicate_squared_distance(np.ones((18,13)))


if __name__=='__main__':unittest.main()
