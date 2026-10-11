import unittest
import numpy as np
from gruxnet.faced_reference_diagnostic import reference_decomposition


class ReferenceTests(unittest.TestCase):
    def test_one_extreme_channel_spreads_through_average_and_linearity_holds(self):
        rate = 250; t = np.arange(32*rate)/rate
        values = np.stack([5*np.sin(2*np.pi*(8+i/20)*t) for i in range(30)])
        values[27] += 30000*np.sin(2*np.pi*3*t)
        observed, m = reference_decomposition(values, rate, 27)
        self.assertEqual(observed.shape,(7,30,512))
        sd = m['core_median_other_channel_sd_uv']
        self.assertGreater(sd['observed'],500)
        self.assertLess(sd['remainder'],5)
        self.assertLess(abs(sd['observed']-sd['dominant_channel_contribution']),5)
        self.assertLess(m['maximum_signal_linearity_discrepancy_uv'],.01)

    def test_invalid_geometry_or_channel_fails(self):
        for values,channel in [(np.zeros((29,8000)),0),(np.zeros((30,8000)),30)]:
            with self.assertRaises(ValueError): reference_decomposition(values,250,channel)


if __name__ == '__main__': unittest.main()
