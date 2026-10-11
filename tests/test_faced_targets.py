import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import zlib
import numpy as np
import pytest
from scipy.io import loadmat,savemat
from gruxnet.faced_targets import audit_score_storage,decode_item10,read_declared_item10

spec=importlib.util.spec_from_file_location('target_test_tags',Path(__file__).with_name('test_faced_raw.py'))
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
IDENTITIES=[{'trial':i,'vid':i} for i in range(1,29)]


def fixture(compressed=False,shape=(1,12),allowed=(3,9),poison_protected=False,
            selected=4.25,fields=None,corrupt_item=None):
    fields=fields or ['score','trial','vid','Accuracy','ResponseTime']
    payload=t.tag(5,struct.pack('<i',16))+t.tag(1,b''.join(f.encode().ljust(16,b'\0') for f in fields))
    for trial in range(1,29):
        for field in fields:
            if field in ('trial','vid'):
                payload+=t.matrix(6,(1,1),'',t.tag(9,struct.pack('<d',trial)))
            elif field=='score' and (trial in allowed or not poison_protected):
                values=[float('nan')]*12;values[9]=selected
                if corrupt_item is not None:values[corrupt_item]=float('nan')
                payload+=t.matrix(6,shape,'',t.tag(9,struct.pack('<'+'d'*12,*values)))
            else: payload+=t.tag(14,b'opaque field bytes with no valid inner header')
    body=t.matrix(2,(1,28),'After_remark',payload)
    if compressed:
        body=struct.pack('<II',15,len(zlib.compress(body)))+zlib.compress(body)
    return b' '*124+b'\x00\x01IM'+body


@pytest.mark.parametrize('compressed',[False,True])
@pytest.mark.parametrize('shape',[(1,12),(12,1)])
def test_metadata_audit_opens_no_score_values_and_skips_poisoned_protected_headers(compressed,shape):
    rows,scope=audit_score_storage(fixture(compressed,shape,poison_protected=True),[3,9],IDENTITIES)
    assert [r['vid'] for r in rows]==[3,9] and scope['score_values_decoded']==0
    assert scope['protected_score_headers_skipped']==26
    assert all(r['score_storage']['item10_byte_range']==[72,80] for r in rows)


@pytest.mark.parametrize('compressed',[False,True])
def test_only_permitted_tenth_item_is_numerically_interpreted(compressed):
    events=[];raw=fixture(compressed,poison_protected=True)
    rows,scope=decode_item10(raw,[3,9],IDENTITIES,observer=events.append)
    assert [r['valence'] for r in rows]==[4.25,4.25]
    assert events==[{'vid':v,'item_one_based':10,'bytes':[72,80]} for v in (3,9)]
    assert scope['other_rating_values_decoded']==0 and scope['item10_values_decoded']==2


@pytest.mark.parametrize('value',[0.0,7.0,3.5])
def test_documented_endpoints_and_midpoint_remain_eligible(value):
    rows,_=decode_item10(fixture(selected=value),[3,9],IDENTITIES)
    assert all(r['valence']==value for r in rows)


@pytest.mark.parametrize('value',[-.1,7.1,float('nan'),float('inf')])
def test_invalid_selected_item_stops_without_switching_items(value):
    with pytest.raises(ValueError,match='documented scale'):
        decode_item10(fixture(selected=value),[3],IDENTITIES)


def test_field_order_does_not_change_item_or_identity_selection():
    fields=['Accuracy','vid','ResponseTime','score','trial']
    a,_=decode_item10(fixture(poison_protected=True),[3,9],IDENTITIES)
    b,_=decode_item10(fixture(poison_protected=True,fields=fields),[3,9],IDENTITIES)
    assert a==b


def test_identity_mismatch_rejected_before_any_score_value_decode():
    expected=IDENTITIES[::-1];events=[]
    with pytest.raises(ValueError,match='crosswalk'):
        decode_item10(fixture(),[3],expected,observer=events.append)
    assert events==[]


def test_protected_participant_rejected_before_file_access():
    def forbidden(path):raise AssertionError('Protected file touched')
    with pytest.raises(PermissionError):
        read_declared_item10('unopened','sub-999',{('sub-001',3)},'unused',IDENTITIES,{},forbidden)


def test_changed_source_hash_rejected_before_numeric_interpretation():
    events=[]
    with pytest.raises(ValueError,match='source bytes'):
        read_declared_item10('.', 'sub-001',{('sub-001',3)},'wrong',IDENTITIES,{},
                            lambda p:fixture(),events.append)
    assert events==[]


def test_authenticated_wrapper_decodes_only_declared_pair():
    raw=fixture(poison_protected=True);rows,_=audit_score_storage(raw,[3],IDENTITIES)
    required={r['vid']:r['score_storage'] for r in rows}
    actual,scope=read_declared_item10('.','sub-001',{('sub-001',3)},hashlib.sha256(raw).hexdigest(),
                                   IDENTITIES,required,lambda p:raw)
    assert actual==[{'trial':3,'vid':3,'valence':4.25}] and scope['item10_values_decoded']==1


def test_independent_scipy_roundtrip_on_manufactured_struct():
    dtype=[(field,object) for field in ('score','trial','vid','Accuracy','ResponseTime')]
    data=np.empty((1,28),dtype=dtype)
    for i in range(28):
        data['score'][0,i]=np.arange(12,dtype=float)[None,:]/2
        data['trial'][0,i]=np.asarray([[i+1]],dtype=float)
        data['vid'][0,i]=np.asarray([[i+1]],dtype=float)
        data['Accuracy'][0,i]=np.asarray([[1]],dtype=float)
        data['ResponseTime'][0,i]=np.asarray([[.2]],dtype=float)
    for compressed in (False,True):
        buffer=io.BytesIO();savemat(buffer,{'After_remark':data},do_compression=compressed)
        raw=buffer.getvalue();selected,_=decode_item10(raw,[3,9],IDENTITIES)
        independent=loadmat(io.BytesIO(raw))['After_remark']
        assert [r['valence'] for r in selected]==[float(independent['score'][0,i][0,9]) for i in (2,8)]


def test_truncated_mat_rejected():
    with pytest.raises(ValueError):audit_score_storage(fixture()[:-8],[3],IDENTITIES)
