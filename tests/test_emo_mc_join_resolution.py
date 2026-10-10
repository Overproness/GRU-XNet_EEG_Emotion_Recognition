"""Protect the unopened outcome gate and exercise ambiguous timing/identity cases."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_emo_mc_historical_events import parse_events
from collect_emo_mc_identity_maps import identity_vectors
from analyze_emo_mc_join_resolution import clock_blocks, compare_events, protected_mask, video_endpoint, vid_only_identity


def mat_bytes(**variables):
    stream = io.BytesIO()
    savemat(stream, variables, do_compression=True)
    return stream.getvalue()


class OutcomeGateTests(unittest.TestCase):
    def test_rating_variable_blocks_payload_reader(self):
        data = mat_bytes(trial=np.array([[1]], dtype='int32'),
                         vid=np.array([[1]], dtype='int64'), score=np.array([[0.0]]))
        with patch('collect_emo_mc_identity_maps.loadmat', side_effect=AssertionError('Payload must remain unopened')) as reader:
            with self.assertRaisesRegex(ValueError, 'identity-only'):
                identity_vectors(data, 'ima')
            reader.assert_not_called()

    def test_floating_rating_like_schema_blocks_reader(self):
        data = mat_bytes(trial=np.array([[1]],dtype='int32'),vid=np.array([[1.0]]))
        with patch('collect_emo_mc_identity_maps.loadmat') as reader:
            with self.assertRaisesRegex(ValueError, 'dtype'):
                identity_vectors(data, 'ima')
            reader.assert_not_called()

    def test_valid_context_identity_vectors(self):
        data = mat_bytes(trial=np.array([[1,2]],dtype='int32'),vid=np.array([[22,42]],dtype='int64'))
        self.assertEqual(identity_vectors(data,'vid'),dict(trial=[1,2],vid=[22,42]))

    def test_duplicate_codes_quarantine(self):
        data = mat_bytes(trial=np.array([[1,2]],dtype='int32'),vid=np.array([[22,22]],dtype='int64'))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            identity_vectors(data,'vid')

    def test_wrong_context_codes_quarantine(self):
        data = mat_bytes(trial=np.array([[1]],dtype='int32'),vid=np.array([[22]],dtype='int64'))
        with self.assertRaisesRegex(ValueError,'material identity'):
            identity_vectors(data,'ima')

    def test_event_score_column_blocks_row_decoding(self):
        body=b'onset\tduration\tstim_type\ttrial_type\tscore\nnot-a-time\t0\t5\tima\tPRIVATE\n'
        with self.assertRaisesRegex(ValueError,'schema'):
            parse_events(body)

    def test_zero_duration_marker_is_not_trial_end(self):
        rows=parse_events(b'onset\tduration\tstim_type\ttrial_type\n1\t0\t5\tima\n')
        self.assertEqual(rows[0]['duration_s'],0)
        self.assertNotIn('trial_end_s',rows[0])

    def test_single_identity_vector_does_not_invent_explicit_ordinals(self):
        data=mat_bytes(vid=np.array([list(range(1,22))],dtype='int32'))
        result=vid_only_identity(data,'ima')
        self.assertFalse(result['trial_ordinal_supplied'])
        self.assertNotIn('trial',result)

    def test_single_vector_supplement_still_blocks_rating_variables(self):
        data=mat_bytes(vid=np.array([list(range(1,22))],dtype='int32'),score=np.array([[0.0]]))
        with patch('scipy.io.loadmat',side_effect=AssertionError('Scores must stay unopened')) as reader:
            with self.assertRaisesRegex(ValueError,'schema'):
                vid_only_identity(data,'ima')
            reader.assert_not_called()


class TimingAndReservationTests(unittest.TestCase):
    def test_clock_reset_retains_order_without_claiming_raw_run_identity(self):
        rows=[dict(onset_s=t,duration_s=0,trigger_code=5) for t in (5,8,1,4)]
        result=clock_blocks(rows)
        self.assertEqual([r['onset_s'] for r in result],[5,8,1,4])
        self.assertNotEqual(result[1]['run_path'],result[2]['run_path'])
        self.assertTrue(result[2]['run_path'].startswith('historical-clock-block-'))

    def test_nonfinite_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            clock_blocks([dict(onset_s=float('nan'),duration_s=0,trigger_code=5)])

    def test_serialization_precision_match_does_not_accept_one_sample_shift(self):
        h=[dict(onset_s=1/600,trigger_code=5)]
        a=[dict(onset_s=0.0017,description='TypeID: 5')]
        self.assertTrue(compare_events(h,a)['ordered_markers_match_at_edf_precision'])
        a[0]['onset_s']+=1/600
        self.assertFalse(compare_events(h,a)['ordered_markers_match_at_edf_precision'])

    def test_trigger_order_mismatch_rejected(self):
        h=[dict(onset_s=1,trigger_code=5),dict(onset_s=2,trigger_code=4)]
        a=[dict(onset_s=1,description='TypeID: 4'),dict(onset_s=2,description='TypeID: 5')]
        self.assertFalse(compare_events(h,a)['ordered_markers_match_at_edf_precision'])

    def test_delayed_rating_does_not_certify_video_tail(self):
        result=video_endpoint(1516.34,1764.9417,150)
        self.assertAlmostEqual(result['candidate_end_difference_s'],97.6017)
        self.assertEqual(result['overlap_between_two_30s_candidates_s'],0)
        self.assertFalse(result['actual_playback_end_verified'])
        self.assertFalse(result['fitting_allowed'])

    def test_matching_duration_still_does_not_certify_playback(self):
        result=video_endpoint(100,251,150)
        self.assertEqual(result['candidate_end_difference_s'],0)
        self.assertFalse(result['actual_playback_end_verified'])

    def test_short_video_rejected_without_stitching(self):
        with self.assertRaises(ValueError):
            video_endpoint(100,111,10)

    def test_unmapped_imagery_remains_excluded_even_when_video_known(self):
        mappings={('vid','joy4'):'vid:01'}
        result=protected_mask('ima','joy4',mappings,{'ima:01'})
        self.assertEqual(result['status'],'unknown_identity_exclude')
        self.assertIsNone(result['material_key'])
        self.assertFalse(result['fitting_allowed'])

    def test_reserved_video_retains_exclusion(self):
        result=protected_mask('vid','joy5',{('vid','joy5'):'vid:03'},{'vid:03'})
        self.assertEqual(result['status'],'reserved_exclude')
        self.assertFalse(result['fitting_allowed'])


if __name__=='__main__':
    unittest.main()
