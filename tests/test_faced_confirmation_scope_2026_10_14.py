"""Manufactured mutations of public metadata; never read original files."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.prepare_faced_confirmation_scope_2026_10_14 import INPUTS, REPO, build, build_from_metadata


class ConfirmationScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values={k:json.loads((REPO/n).read_bytes()) for k,n in INPUTS.items()}

    def fresh(self):
        return copy.deepcopy(self.values)

    def test_exact_counts_and_transport_accounting(self):
        scope=build_from_metadata(self.fresh()); caps=scope['resource_scope']
        self.assertEqual((scope['trials'],scope['regions'],scope['required_prediction_count']),(759,2277,3036))
        self.assertEqual(scope['native_rate_participant_counts'],{1000:17,250:16})
        self.assertEqual(caps['selected_count_values'],1391040000)
        self.assertEqual(caps['selected_signal_payload_bytes'],4173120000)
        self.assertEqual(caps['data_file_bytes_one_hash_pass'],8961374784)
        self.assertEqual(caps['event_file_bytes_one_hash_pass'],2448132)
        self.assertEqual(caps['two_data_plus_two_event_hash_transport_bytes'],17927645832)
        self.assertEqual(caps['all_regional_float64_features_bytes'],2732400)
        self.assertEqual(len(scope['files']),66)
        self.assertEqual(scope['original_files_opened'],0)
        self.assertEqual(scope['activation_gates_cleared_by_this_metadata_scope'],0)

    def test_ranges_and_complete_prediction_denominator(self):
        scope=build_from_metadata(self.fresh())
        self.assertEqual(len({(r['subject'],r['vid']) for r in scope['rows']}),759)
        for r in scope['rows']:
            self.assertEqual(r['role'],'confirmation')
            self.assertEqual(len(set(r['channel_indices'])),30)
            for a,b in r['segments'].values():
                self.assertEqual(b-a,32*r['native_rate_hz'])
                self.assertGreaterEqual(a,r['marker_start_seconds']*r['native_rate_hz'])
                self.assertLessEqual(b,r['marker_end_seconds']*r['native_rate_hz'])

    def test_metadata_builder_has_no_original_stat_or_open(self):
        original_read=Path.read_bytes
        def safe_read(path):
            if not path.is_relative_to(REPO) or 'Data/Data' in path.as_posix() or 'publication_runs' in path.as_posix():
                raise AssertionError('Original or private reader was called')
            return original_read(path)
        with patch.object(Path,'read_bytes',safe_read),patch.object(Path,'stat',side_effect=AssertionError('No original stat')):
            self.assertEqual(build()['original_files_statted'],0)

    def test_changed_participant_role_rejected(self):
        v=self.fresh(); v['roles']['confirmation'][0]=v['roles']['development_source'][0]
        with self.assertRaises(PermissionError):build_from_metadata(v)

    def test_duplicate_join_rejected(self):
        v=self.fresh();v['joins']['records'].append(copy.deepcopy(v['joins']['records'][0]))
        with self.assertRaises(PermissionError):build_from_metadata(v)

    def test_changed_family_rejected(self):
        v=self.fresh();v['catalogue']['stimuli'][0]['source_film_sha256']='0'*64
        with self.assertRaises(PermissionError):build_from_metadata(v)

    def test_protected_activation_not_accepted(self):
        v=self.fresh();v['proposal']['protected_signal_access_authorized']=True
        with self.assertRaises(PermissionError):build_from_metadata(v)

    def test_behavior_join_ordinal_mismatch_rejected(self):
        v=self.fresh();p=v['roles']['confirmation'][0]
        r=next(x for x in v['joins']['records'] if x['subject']==p)
        r['identity_rows'][0]['trial']=29
        with self.assertRaises(PermissionError):build_from_metadata(v)

    def test_short_span_rejected(self):
        v=self.fresh();p=v['roles']['confirmation'][0]
        r=next(x for x in v['joins']['records'] if x['subject']==p)
        span=next(x for x in r['spans'] if x['vid']==1)
        span['end_seconds']=span['start_seconds']+31
        with self.assertRaises(ValueError):build_from_metadata(v)

    def test_unqualified_inventory_rejected(self):
        v=self.fresh();v['extraction']['all_members_match']=False
        with self.assertRaises(PermissionError):build_from_metadata(v)

    def test_selected_original_byte_mismatch_rejected(self):
        v=self.fresh();p=v['roles']['confirmation'][0]; member='Data/'+p.replace('sub-','sub')+'/data.bdf'
        next(x for x in v['extraction']['records'] if x['member']==member)['bytes']+=1
        with self.assertRaises(ValueError):build_from_metadata(v)

    def test_bogus_integer_header_literal_rejected(self):
        v=self.fresh();p=v['roles']['confirmation'][0]
        next(x for x in v['headers']['records'] if x['subject']==p)['data.bdf']['columns']['samples_per_record'][0]=True
        with self.assertRaises(ValueError):build_from_metadata(v)


if __name__=='__main__':unittest.main()
