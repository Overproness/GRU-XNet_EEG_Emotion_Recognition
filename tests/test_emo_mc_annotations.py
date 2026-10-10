"""Annotation parsing and run-boundary checks without any source EEG."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_emo_mc_pilot import tal_events
from analyze_emo_mc_annotations import pair_trials, is_subsequence, separate_initial_state_markers, numeric_identity


class AnnotationTests(unittest.TestCase):
    def test_edfplus_timekeeping_and_duration(self):
        payload=b'+0\x14\x14\x00+12.5\x150.25\x14TypeID: 5\x14\x00'
        self.assertEqual(tal_events(payload),[{'onset_s':12.5,'duration_s':0.25,'description':'TypeID: 5'}])

    def test_stimulus_pairing_does_not_cross_runs_or_next_stimulus(self):
        def event(t,code,run):return dict(onset_s=t,description=f'TypeID: {code}',run_path=run)
        events=[event(0,4,'r1'),event(5,5,'r1'),event(7,6,'r1'),event(20,4,'r1'),
                event(30,5,'r1'),event(0,4,'r2')]
        pairs=pair_trials(events,dict(ima=5,vid=6,rating=4,fade=3))
        self.assertEqual(len(pairs),3)
        self.assertIsNone(pairs[0]['rating_start_s'])
        self.assertEqual(pairs[1]['start_to_rating_s'],13)
        self.assertIsNone(pairs[2]['rating_start_s'])

    def test_missing_trial_order_is_subsequence_and_requires_order(self):
        self.assertTrue(is_subsequence(['sad4','sad8'],['sad4','sad5','sad8']))
        self.assertFalse(is_subsequence(['sad8','sad4'],['sad4','sad5','sad8']))

    def test_ambiguous_initial_state_cluster_and_single_valid_start(self):
        def event(t,code):return dict(onset_s=t,description=f'TypeID: {code}',run_path='r1')
        initial=[event(0.0017,code) for code in (1,7,6,2)]
        later=[event(1,code) for code in (1,7,6,2)]
        kept,excluded=separate_initial_state_markers(initial+later)
        self.assertEqual(excluded,initial)
        self.assertEqual(kept,later)
        self.assertEqual(separate_initial_state_markers([event(0,6)]),([event(0,6)],[]))

    def test_matlab_wrapper_and_plain_numeric_identity_are_equivalent(self):
        self.assertEqual(numeric_identity('[[38]]'),numeric_identity('38'))
        self.assertEqual(numeric_identity('38.0'),38)
        with self.assertRaises(ValueError):numeric_identity('38.5')


if __name__=='__main__':unittest.main()
