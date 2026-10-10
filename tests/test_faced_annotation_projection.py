import importlib.util
import io
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('projection',Path(__file__).resolve().parents[1]/'scripts/audit_faced_annotation_projection.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_non_numeric_text_remains_opaque():
    payload=b'+0\x14\x14\0+1\x14private text\x14\0+2\x14101\x14\0'
    events,clocks,opaque=module.project_tals(payload)
    assert events==[{'code':101,'onset_seconds':2,'duration_seconds':0}]
    assert clocks==[0] and len(opaque)==1 and opaque[0]['text_bytes']==12
    assert all('private text' not in str(x) for x in (events,clocks,opaque))


def test_changed_opaque_content_cannot_change_numeric_projection():
    first=module.project_tals(b'+1\x14secret A\x14\0+2\x1407\x14\0')
    second=module.project_tals(b'+1\x14secret B\x14\0+2\x1407\x14\0')
    assert first[:2]==second[:2] and first[2][0]['text_sha256']!=second[2][0]['text_sha256']


def test_dummy_channel_is_sought_past_without_reading():
    class CheckedStream(io.BytesIO):
        def read(self,n=-1):
            assert self.tell()>=3, 'Protected dummy channel read'
            return super().read(n)
    class Source:
        def open(self,mode):
            assert mode=='rb'
            tal=b'+1\x1407\x14\0'
            return CheckedStream(b'XYZ'+tal.ljust(300,b'\0'))
    header={'sample_width':3,'header_bytes':0,'record_count':1,
            'columns':{'labels':['Empty Event Data','BDF Annotations'],'samples_per_record':['1','100']}}
    rows,_=module.events_module.read_events(Source(),header)
    assert rows[0]['code']==7


def test_changed_channel_geometry_rejected_before_read():
    header={'sample_width':3,'columns':{'labels':['EEG','BDF Annotations'],'samples_per_record':['1','100']}}
    with pytest.raises(ValueError,match='geometry'):
        module.events_module.read_events(None,header)
