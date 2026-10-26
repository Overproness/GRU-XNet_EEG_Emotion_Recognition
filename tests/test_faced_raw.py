"""Synthetic boundary tests: no downloaded EEG or outcome values used."""
import importlib.util
from pathlib import Path
import struct
import zlib
import pytest

spec=importlib.util.spec_from_file_location('raw_audit',Path(__file__).resolve().parents[1]/'scripts/audit_faced_raw.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def tag(kind,data):
    return struct.pack('<II',kind,len(data))+data+b'\0'*((-len(data))%8)


def matrix(cls,shape,name,payload):
    return tag(14,tag(6,struct.pack('<II',cls,0))+tag(5,struct.pack('<'+'i'*len(shape),*shape))+
               tag(1,name.encode())+payload)


def mat_bytes(poison=False,duplicate=False,compressed=False):
    fields=['score','trial','vid','Accuracy','ResponseTime']
    payload=tag(5,struct.pack('<i',16))+tag(1,b''.join(f.encode().ljust(16,b'\0') for f in fields))
    for trial in range(1,29):
        for field in fields:
            if field in ('trial','vid'):
                value=1 if duplicate and field=='vid' else trial
                payload+=matrix(6,(1,1),'',tag(9,struct.pack('<d',value)))
            else:
                # A poisoned protected matrix has no valid numeric/header payload.
                payload+=tag(14,b'invalid protected bytes' if poison else matrix(6,(1,1),'',tag(9,struct.pack('<d',777)))[8:])
    body=matrix(2,(1,28),'After_remark',payload)
    if compressed:
        body=struct.pack('<II',15,len(zlib.compress(body)))+zlib.compress(body)
    return b' '*124+b'\x00\x01IM'+body


@pytest.mark.parametrize('compressed',[False,True])
def test_identity_projection_invariant_to_poisoned_protected_fields(compressed):
    rows,scope=audit.identity_projection(mat_bytes(compressed=compressed))
    poisoned,other=audit.identity_projection(mat_bytes(poison=True,compressed=compressed))
    assert rows==poisoned==[{'trial':i,'vid':i} for i in range(1,29)]
    assert scope==other and scope['skipped_protected_matrices']==84
    assert scope['rating_values_decoded']==0


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError,match='Missing/duplicate'):
        audit.identity_projection(mat_bytes(duplicate=True))


def test_truncated_mat_rejected():
    with pytest.raises(ValueError):
        audit.identity_projection(mat_bytes()[:-8])


def test_tals_preserve_clock_and_codes():
    events,clocks=audit.parse_tals(b'+0\x14\x14\0+1.25\x1407\x14\0+1.3\x150.1\x14101\x14\0')
    assert clocks==[0]
    assert [r['code'] for r in events]==[7,101]
    assert events[1]['duration_seconds']==0.1


def test_tal_narratives_fail_closed():
    with pytest.raises(ValueError,match='Non-numeric'):
        audit.parse_tals(b'+0\x14private narrative\x14\0')


def test_stray_start_cannot_create_emotion_trial():
    rows=[{'code':c,'onset_seconds':t} for c,t in ((101,0),(7,10),(101,11),(102,20))]
    spans,anomalies=audit.strict_spans(rows)
    assert spans==[{'vid':7,'start_seconds':11,'end_seconds':20,'duration_seconds':9}]
    assert anomalies==[{'event_index':0,'type':'unidentified_start'}]


def test_replaced_start_quarantines_candidate_and_differs_from_curator():
    rows=[{'code':c,'onset_seconds':t} for c,t in ((7,0),(101,1),(101,2),(102,9))]
    spans,anomalies=audit.strict_spans(rows)
    assert spans==[] and len(anomalies)==2
    assert audit.curator_spans(rows)[0]['start_seconds']==2


@pytest.mark.parametrize('name',['../outside','/absolute','Data/../../outside','Data/C:bad','Data\\sub000'])
def test_unsafe_zip_paths_rejected(name):
    with pytest.raises(ValueError):
        audit.member_target(name,Path(__file__).resolve().parent)


def test_safe_zip_path():
    root=Path(__file__).resolve().parent
    assert audit.member_target('Data/sub000/evt.bdf',root)==root/'Data/sub000/evt.bdf'
