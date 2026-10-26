import unittest
import numpy as np
from gruxnet.faced_artifact_control import (artifact_trial_windows,pre_car_windows,
    detect_high_channels,interpolate_input,qualified_weights)
from gruxnet.faced_source_notch import notched_late_trial_windows


class ArtifactTests(unittest.TestCase):
    def clean(self,rate):
        t=np.arange(32*rate)/rate
        return np.stack([5*np.sin(2*np.pi*(8+i/20)*t) for i in range(30)])

    def test_no_detection_preserves_frozen_output_bytes_for_both_rates(self):
        for rate in (250,1000):
            x=self.clean(rate); original=x.copy();windows,metadata=artifact_trial_windows(x,rate)
            self.assertTrue(np.array_equal(windows,notched_late_trial_windows(x,rate)))
            self.assertTrue(np.array_equal(x,original));self.assertEqual(metadata['decision'],'unchanged')

    def test_extreme_channel_is_detected_before_reference_spread_and_repaired(self):
        x=self.clean(250);t=np.arange(8000)/250;x[27]+=30000*np.sin(2*np.pi*3*t)
        windows,m=artifact_trial_windows(x,250)
        self.assertEqual(m['detected_channels'],['Oz']);self.assertEqual(m['repaired_channels'],['Oz'])
        self.assertLess(windows.astype(float).std(axis=-1).max(),10)
        self.assertLess(m['interpolation_weight_max_discrepancy'],1e-9)

    def test_interpolation_donors_and_common_constant_are_preserved(self):
        x=self.clean(250)+25.;repaired,w,error=interpolate_input(x,[0,1])
        self.assertTrue(np.array_equal(repaired[2:],x[2:]));self.assertLess(error,1e-9)
        self.assertTrue(np.allclose(w.sum(axis=1),1,atol=1e-9))
        weights,_,_=qualified_weights((0,1,27));self.assertEqual(weights.shape,(3,27))

    def test_four_detections_hold_capacity_without_silent_repair_or_trial_drop(self):
        x=self.clean(250);t=np.arange(8000)/250;x[:4]+=30000*np.sin(2*np.pi*3*t)
        windows,m=artifact_trial_windows(x,250)
        self.assertTrue(m['repair_capacity_exceeded']);self.assertEqual(m['repaired_channels'],[])
        self.assertTrue(np.array_equal(windows,notched_late_trial_windows(x,250)))
        self.assertEqual(m['trials_or_windows_dropped'],0)

    def test_detection_uses_all_windows_and_exact_strict_thresholds(self):
        x=np.zeros((7,30,512));x[6,4,0]=1000.
        self.assertEqual(detect_high_channels(x)[0],[])
        x[6,4,0]=1001.;self.assertEqual(detect_high_channels(x)[0],[4])

    def test_temporal_detection_agrees_with_existing_CAR_order_to_float32_accuracy(self):
        x=self.clean(1000);expected=notched_late_trial_windows(x,1000)
        before=pre_car_windows(x-x.mean(axis=0,keepdims=True),1000)
        self.assertLess(np.max(np.abs(expected-before)),1e-5)

    def test_nonfinite_or_wrong_geometry_rejected(self):
        x=self.clean(250);x[0,0]=np.nan
        with self.assertRaises(ValueError):artifact_trial_windows(x,250)
        with self.assertRaises(ValueError):artifact_trial_windows(np.zeros((29,8000)),250)


if __name__=='__main__':unittest.main()
