"""Manufactured boundary and access-ledger checks; no original data."""
import copy
import hashlib
import json
import unittest
from scripts.analyze_faced_classical_source import engineering_flags, check_access


class AnalysisTests(unittest.TestCase):
    def test_threshold_boundaries_and_independent_flag_reconstruction(self):
        channel = {'raw_rail_fraction': .0001, 'raw_identical_adjacent_fraction': .95,
                   'processed_sd_uv': .1, 'processed_peak_to_peak_uv': 1000.,
                   'processed_49_51_to_1_45_power_ratio': .10, 'review_flags': ['poisoned']}
        self.assertEqual(engineering_flags(channel), [])
        self.assertEqual(engineering_flags({**channel, 'processed_sd_uv': 100.}), [])
        bad = {**channel, 'raw_rail_fraction': .0002, 'raw_identical_adjacent_fraction': .96,
               'processed_sd_uv': 101., 'processed_peak_to_peak_uv': 1001.,
               'processed_49_51_to_1_45_power_ratio': .101}
        self.assertEqual(engineering_flags(bad), ['adc_rail','stationary_counts','high_processed_variation',
                                                'high_processed_excursion','residual_50hz_power'])
        self.assertEqual(engineering_flags({**channel, 'processed_sd_uv': .099}), ['low_processed_variation'])

    def manufactured_access(self):
        header = {'header_bytes': 8192, 'columns': {'samples_per_record': ['4']*32}}
        channels = list(reversed(range(30)))
        ranges = []
        # Two partial boundary records, selected scalp channels only.
        for record,lo,hi in [(0,3,4),(1,4,6)]:
            for channel in channels:
                ranges.append({'channel_index': channel, 'start_sample': lo, 'stop_sample': hi,
                    'offset': 8192+record*384+channel*12+3*(lo-record*4), 'bytes': 3*(hi-lo)})
        sha = hashlib.sha256(json.dumps(ranges, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        row = {'sample_interval': [3,6], 'channel_indices': channels, 'range_ledger_sha256': sha, 'samples_decoded': 90}
        access = {'ranges': ranges, 'range_ledger_sha256': sha, 'sample_values_decoded': 90,
                  'signal_bytes_decoded': 270, 'extra_boundary_samples_decoded': 0,
                  'all_counts_scalar_reference_checked': True}
        return row,access,header

    def test_partial_record_offsets_reversed_channels_and_count_coverage(self):
        row, access, header = self.manufactured_access()
        self.assertEqual(check_access(row, access, header), 60)

    def test_corrupted_offsets_and_extra_samples_fail(self):
        row, access, header = self.manufactured_access()
        bad = copy.deepcopy(access); bad['ranges'][0]['offset'] += 3
        with self.assertRaises(ValueError): check_access(row, bad, header)
        bad = {**access, 'extra_boundary_samples_decoded': 1}
        with self.assertRaises(ValueError): check_access(row, bad, header)


if __name__ == '__main__': unittest.main()
