"""Checks for the outcome-blind source reader and reservation boundary."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/qualify_emo_mc.py'
SPEC = importlib.util.spec_from_file_location('emo_qualification', PATH)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def synthetic_header():
    n = 2
    header = bytearray(b' ' * (256 + n * 256))
    for start, width, value in [(0, 8, '0'), (184, 8, str(len(header))),
                                (192, 44, 'EDF+C'), (236, 8, '3'),
                                (244, 8, '1'), (252, 4, str(n))]:
        header[start:start + width] = value.encode().ljust(width)
    values = [['Fp1', 'EDF Annotations'], ['', ''], ['uV', ''],
              ['-100000', '-1'], ['100000', '1'], ['-32768', '-32768'],
              ['32767', '32767'], ['', ''], ['600', '57'], ['', '']]
    pos = 256
    for (_, width), field in zip(Q.FIELDS, values):
        for value in field:
            header[pos:pos + width] = value.encode().ljust(width)
            pos += width
    return bytes(header), len(header) + 3 * (600 + 57) * 2


class QualificationTests(unittest.TestCase):
    def test_calibration_and_short_object_quarantine(self):
        body, size = synthetic_header()
        parsed = Q.parse_edf_header(body, size)
        self.assertTrue(parsed['geometry_consistent'])
        eeg = parsed['channels'][0]
        self.assertEqual(eeg['sampling_hz'], 600)
        self.assertAlmostEqual(eeg['gain_in_declared_unit'], 200000 / 65535)
        self.assertAlmostEqual(-32768 * eeg['gain_in_declared_unit'] + eeg['offset_in_declared_unit'], -100000)
        damaged = Q.parse_edf_header(body, size - 13)
        self.assertFalse(damaged['geometry_consistent'])
        self.assertEqual(damaged['declared_size_delta_bytes'], -13)

    def test_score_bytes_never_decoded(self):
        header = '\t'.join(['trial_number', 'video_name'] + [f'score_{i}' for i in range(1, 11)]).encode()
        first = header + b'\n1.0\t[[38]]\t\xff\xfe\x80\n'
        second = header + b'\n1\t[[38]]\tanything\tpoisoned\n'
        self.assertEqual(Q.identity_projection(first), Q.identity_projection(second))
        with self.assertRaises(ValueError):
            Q.identity_projection(header + b'\n1.5\t38\tunused\n')

    def test_server_ignoring_range_does_not_read_body(self):
        class Response:
            status_code = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_content(self, *args):
                raise AssertionError('Full EEG body must never be consumed')
        with patch.object(Q.requests, 'get', return_value=Response()):
            with self.assertRaisesRegex(ValueError, 'Incorrect bounded response'):
                Q.range_get('https://example.invalid/object', 0, 255, 100000)

    def test_boundary_requires_mapping_and_excludes_both_reservations(self):
        plan = {'participants': {'development_source': ['s1'], 'development_validation': ['s2']},
                'materials': {'reserved_keys': ['vid:01', 'ima:01']}}
        with self.assertRaisesRegex(ValueError, 'Unverified'):
            Q.permit_row(plan, 's1', 'vid', 2)
        self.assertFalse(Q.permit_row(plan, 's3', 'vid', 2, True))
        self.assertFalse(Q.permit_row(plan, 's1', 'ima', 1, True))
        self.assertTrue(Q.permit_row(plan, 's1', 'vid', 2, True))

    def test_film_family_is_never_divided(self):
        rows = [{'context': 'vid', 'number': 13, 'family': 'Insidious'},
                {'context': 'vid', 'number': 14, 'family': 'Insidious'},
                {'context': 'vid', 'number': 15, 'family': 'Dead Silence'}]
        self.assertEqual(Q.family_closure(rows, [13]), [13, 14])
        self.assertEqual(Q.family_closure(rows, [15]), [15])


if __name__ == '__main__':
    unittest.main()
