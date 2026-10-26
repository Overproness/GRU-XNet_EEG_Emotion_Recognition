import hashlib
import importlib.util
import io
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('lineage',Path(__file__).resolve().parents[1]/'scripts/verify_faced_raw_lineage.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Source:
    def __init__(self,payload):
        self.payload=payload
    def open(self,mode):
        assert mode=='rb'
        return io.BytesIO(self.payload)


def test_header_substitution_reproduces_only_same_payload():
    original=bytearray(b' '*256)
    original[184:192]=b'256     '
    corrected=bytearray(original)
    corrected[8:88]=b'X'*80
    body=b'opaque samples'*50
    before,after,payload,diff=module.normalized_hash(Source(original+body),corrected)
    assert before==hashlib.sha256(original+body).hexdigest()
    assert after==hashlib.sha256(corrected+body).hexdigest()
    assert payload==hashlib.sha256(body).hexdigest()
    assert diff['patient_identification']==80 and sum(diff.values())==80
    assert module.normalized_hash(Source(original+body+b'changed'),corrected)[1]!=after


def test_header_substitution_rejects_geometry_change():
    original=bytearray(b' '*256)
    original[184:192]=b'256     '
    with pytest.raises(ValueError,match='geometry'):
        module.normalized_hash(Source(original),b' '*512)


def test_redirect_guard_rejects_unapproved_hosts_and_second_route():
    module.pilot.validate_redirect('https://nemar.s3.us-east-2.amazonaws.com/public-object')
    with pytest.raises(ValueError):
        module.pilot.validate_redirect('https://unexpected.example/public-object')
    with pytest.raises(ValueError):
        module.pilot.validate_redirect('http://nemar.s3.us-east-2.amazonaws.com/public-object')
