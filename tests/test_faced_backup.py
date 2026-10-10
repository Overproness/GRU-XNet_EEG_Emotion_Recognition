"""Protect outcome-blind metadata and byte-range boundaries."""
import importlib.util
import json
from pathlib import Path
import unittest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('faced_backup',REPO/'scripts/qualify_faced_backup.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class FakeResponse:
    def __init__(self,status,range_value,data):
        self.status_code=status;self.headers={'Content-Range':range_value};self.data=data;self.read=False
    def iter_content(self,_):
        self.read=True;yield self.data


def header(unit='uV',digital_max='8388607'):
    fixed=bytearray(b' '*256);fixed[:8]=b'\xffBIOSEMI';fixed[184:192]=b'512     ';fixed[236:244]=b'2       ';fixed[244:252]=b'1       ';fixed[252:256]=b'1   '
    values={'label':'Fp1','transducer':'','unit':unit,'physical_min':'-375000','physical_max':'375000','digital_min':'-8388607','digital_max':digital_max,'prefilter':'HP:0.01','samples':'250','reserved':''}
    return bytes(fixed)+b''.join(values[key].encode().ljust(width) for key,width in module.FIELDS)


class RangeTests(unittest.TestCase):
    def test_ignored_range_does_not_read_body(self):
        response=FakeResponse(200,None,b'EEG payload')
        with self.assertRaises(ValueError):module.checked_range(response,0,255,10000)
        self.assertFalse(response.read)
    def test_wrong_total_does_not_read_body(self):
        response=FakeResponse(206,'bytes 0-255/9999',b'x'*256)
        with self.assertRaises(ValueError):module.checked_range(response,0,255,10000)
        self.assertFalse(response.read)
    def test_exact_range(self):
        response=FakeResponse(206,'bytes 0-255/10000',b'x'*256)
        self.assertEqual(module.checked_range(response,0,255,10000),b'x'*256)
    def test_short_and_oversized_ranges(self):
        for length in (255,257):
            with self.assertRaises(ValueError):module.checked_range(FakeResponse(206,'bytes 0-255/10000',b'x'*length),0,255,10000)
    def test_encoded_response_rejected_before_read(self):
        response=FakeResponse(206,'bytes 0-255/10000',b'x'*256);response.headers['Content-Encoding']='gzip'
        with self.assertRaises(ValueError):module.checked_range(response,0,255,10000)
        self.assertFalse(response.read)


class MetadataTests(unittest.TestCase):
    def test_forbidden_outcome_paths_rejected_before_network(self):
        for path in ('sub-001/eeg/sub-001_events.tsv','participants.tsv','Data/sub001/After_remarks.mat','sub-001/sub-001_scans.tsv'):
            with self.assertRaises(ValueError):module.metadata_get(path,'unused')
    def test_ambiguous_annex_pointer_rejected(self):
        one=b'SHA256E-s1000--'+b'a'*64+b'.bdf';two=b'SHA256E-s1001--'+b'b'*64+b'.bdf'
        self.assertEqual(module.pointer(one),(1000,'a'*64))
        with self.assertRaises(ValueError):module.pointer(one+two)
    def test_calibration_and_geometry(self):
        result=module.parse_header(header(),512+2*250*3)
        self.assertTrue(result['file_geometry_matches'])
        self.assertEqual(result['channels'][0]['sampling_frequency'],250)
        self.assertAlmostEqual(result['channels'][0]['physical_units_per_count'],750000/16777214)
        self.assertFalse(module.parse_header(header(),10000)['file_geometry_matches'])
    def test_ambiguous_units_not_silently_repaired(self):
        self.assertEqual(module.parse_header(header('?V'),2012)['channels'][0]['unit'],'?V')
    def test_invalid_calibration_rejected(self):
        with self.assertRaises(ValueError):module.parse_header(header(digital_max='-8388607'),2012)
    def test_no_participant_or_date_fields_in_projection(self):
        data=bytearray(header());data[8:88]=b'PRIVATE IDENTITY'.ljust(80);data[168:176]=b'12.12.12'
        output=json.dumps(module.parse_header(bytes(data),2012))
        self.assertNotIn('PRIVATE',output);self.assertNotIn('12.12.12',output)
    def test_reserved_roles_are_disjoint_and_complete(self):
        data=json.loads((REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json').read_text())
        roles=[set(data[k]) for k in ('confirmation','development_validation','development_source')]
        self.assertEqual([len(s) for s in roles],[33,20,70])
        self.assertEqual(len(set.union(*roles)),123)
        for i in range(3):
            for j in range(i):self.assertFalse(roles[i]&roles[j])
        self.assertTrue(data['all_material_outcomes_sealed'])


if __name__=='__main__':unittest.main()
