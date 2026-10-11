import unittest
from unittest.mock import Mock
import numpy as np
from gruxnet.faced_inference_quality import (eeg_availability,predict_with_quality,
    require_source_quality,RULES,COMMON_CHANNELS)

def clean_quality():
    return {'channels':[{'channel':c,'raw_rail_fraction':0.,'raw_identical_adjacent_fraction':0.,
        'processed_sd_uv':10.,'processed_peak_to_peak_uv':100.,'processed_49_51_to_1_45_power_ratio':0.,
        'review_flags':[]} for c in COMMON_CHANNELS],
        'valid_windows':7,'windows_dropped':0,'finite_processed_values':True}

class PolicyTests(unittest.TestCase):
    def test_rail_is_unavailable_despite_plausible_processed_variation_and_empty_flag_text(self):
        q=clean_quality();q['channels'][list(COMMON_CHANNELS).index('F8')]['raw_rail_fraction']=1.
        d=eeg_availability(q);self.assertFalse(d['eeg_available'])
        self.assertEqual(d['reasons'],[{'channel':'F8','reason':'adc_rail'}])

    def test_all_six_rules_strict_boundaries(self):
        for metric,limit,side in [('raw_rail_fraction',.0001,1),('raw_identical_adjacent_fraction',.95,1),
            ('processed_sd_uv',.1,-1),('processed_sd_uv',100.,1),('processed_peak_to_peak_uv',1000.,1),
            ('processed_49_51_to_1_45_power_ratio',.10,1)]:
            q=clean_quality();q['channels'][0][metric]=limit
            self.assertTrue(eeg_availability(q)['eeg_available'])
            q['channels'][0][metric]=limit+side*1e-8
            self.assertFalse(eeg_availability(q)['eeg_available'])

    def test_geometry_and_missing_metrics_rejected(self):
        for mutate in (lambda q:q['channels'].reverse(),lambda q:q['channels'].pop(),
            lambda q:q.update(valid_windows=6),lambda q:q.update(windows_dropped=1),
            lambda q:q['channels'][0].pop('raw_rail_fraction')):
            q=clean_quality();mutate(q)
            with self.assertRaises((ValueError,KeyError)):eeg_availability(q)

    def test_nonfinite_and_out_of_range_metrics_rejected(self):
        for value in (float('nan'),float('inf'),-1.,1.01,True,'0'):
            q=clean_quality();q['channels'][0]['raw_rail_fraction']=value
            with self.assertRaises(ValueError):eeg_availability(q)

    def test_technical_failure_and_capacity_hold_do_not_become_eeg_available(self):
        self.assertFalse(eeg_availability(None,technical_passed=False)['eeg_available'])
        self.assertFalse(eeg_availability(clean_quality(),repair_capacity_exceeded=True)['eeg_available'])

    def test_joint_fallback_never_calls_poisoned_eeg_predictor(self):
        d=eeg_availability(None,technical_passed=False);bad=Mock(side_effect=AssertionError('EEG unavailable'))
        context=Mock(return_value=3.2)
        r=predict_with_quality(d,'context_plus_spectral_eeg',eeg_predictor=bad,context_predictor=context)
        self.assertEqual(r,{'prediction':3.2,'branch':'context_fallback','fallback':True,'eeg_used':False})
        bad.assert_not_called();context.assert_called_once()

    def test_eeg_only_fallback_uses_only_source_median(self):
        d=eeg_availability(None,technical_passed=False);bad=Mock(side_effect=AssertionError('No EEG/context read'))
        r=predict_with_quality(d,'spectral_eeg_only',eeg_predictor=bad,context_predictor=bad,source_median=4.)
        self.assertEqual(r['prediction'],4.);self.assertTrue(r['fallback']);bad.assert_not_called()

    def test_available_eeg_and_context_predictions_preserved(self):
        d=eeg_availability(clean_quality())
        for family in ('spectral_eeg_only','context_plus_spectral_eeg'):
            forbidden=Mock(side_effect=AssertionError('No fallback read'))
            r=predict_with_quality(d,family,eeg_predictor=lambda:2.5,context_predictor=forbidden)
            self.assertEqual(r['prediction'],2.5);self.assertTrue(r['eeg_used']);forbidden.assert_not_called()
        r=predict_with_quality(d,'context_only',context_predictor=lambda:1.5)
        self.assertEqual(r['prediction'],1.5);self.assertFalse(r['eeg_used'])

    def test_prediction_boundaries_and_inconsistent_decisions_rejected(self):
        d=eeg_availability(clean_quality())
        for v in (-1.,8.,float('nan')):
            with self.assertRaises(ValueError):predict_with_quality(d,'context_only',context_predictor=lambda:v)
        poisoned={**d,'eeg_available':False}
        with self.assertRaises(ValueError):predict_with_quality(poisoned,'context_only',context_predictor=lambda:3.)

    def test_identity_and_target_metadata_do_not_select_policy(self):
        q=clean_quality();expected=eeg_availability(q)
        q.update(subject='held-person',vid=22,target=0.,prediction=7.)
        self.assertEqual(eeg_availability(q),expected)

    def test_fitting_guard_reads_only_declared_source_quality_and_holds_on_failure(self):
        pair=('source-person',11);rows=[{'subject':pair[0],'vid':pair[1],'technical_passed':True}]
        loader=Mock(return_value=clean_quality())
        self.assertTrue(require_source_quality(rows,loader,[pair])['passed']);loader.assert_called_once_with(pair)
        q=clean_quality();q['channels'][0]['raw_identical_adjacent_fraction']=1.
        gate=require_source_quality(rows,lambda p:q,[pair]);self.assertFalse(gate['passed']);self.assertTrue(gate['fitting_held'])

    def test_source_identity_guard_precedes_quality_reads(self):
        loader=Mock(side_effect=AssertionError('Excluded quality read'))
        rows=[{'subject':'excluded','vid':11,'technical_passed':True}]
        with self.assertRaises(PermissionError):require_source_quality(rows,loader,[('source',11)])
        loader.assert_not_called()

    def test_manufactured_railed_channel_acquires_post_car_variation_but_stays_unavailable(self):
        from gruxnet.faced_source_notch import notched_late_trial_windows
        from gruxnet.faced_artifact_control import pre_car_windows
        from gruxnet.faced_source_quality import quality_metrics
        i=list(COMMON_CHANNELS).index('F8');gain=750000/16777215;offset=.5*gain
        for rate in (250,1000):
            t=np.arange(32*rate)/rate
            signal=np.stack([10*np.sin(2*np.pi*(8+j/20)*t+j/30) for j in range(30)])
            counts=np.rint((signal-offset)/gain).astype(np.int32);counts[i]=-8388608
            values=counts.astype(float)*gain+offset
            windows=notched_late_trial_windows(values,rate)
            q=quality_metrics(counts,values,windows,rate)
            self.assertGreater(q['channels'][i]['processed_sd_uv'],.1)
            self.assertLess(pre_car_windows(values,rate)[:,i].astype(float).std(),.1)
            self.assertFalse(eeg_availability(q)['eeg_available'])
            self.assertIn({'channel':'F8','reason':'stationary_counts'},eeg_availability(q)['reasons'])

if __name__=='__main__':unittest.main()
