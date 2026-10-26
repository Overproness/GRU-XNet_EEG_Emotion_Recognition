"""Bounded MAT structural audit and exact item-10 decoding with identity guards.

Protected scores are skipped at their outer matrix tags. Only manufactured
targets may be decoded until a separate real-outcome access execution exists.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import re
import struct
import zlib
from scripts.audit_faced_raw import element, matrix_prefix, identity_scalar

MAX_BYTES=1_000_000
FIELDS={'score','trial','vid','Accuracy','ResponseTime'}
NUMERIC={6:(9,'d'),7:(7,'f'),8:(1,'b'),9:(2,'B'),10:(3,'h'),11:(4,'H'),
         12:(5,'i'),13:(6,'I'),14:(12,'q'),15:(13,'Q')}


def behavioral_rows(raw, expected_identities):
    if not 136<=len(raw)<=MAX_BYTES or raw[126:128]!=b'IM':
        raise ValueError('Require bounded little-endian Level-5 MAT')
    pos=128;matrices=[]
    while pos<len(raw):
        kind,value,pos=element(raw,pos)
        if kind==15:
            inflater=zlib.decompressobj();expanded=inflater.decompress(value,MAX_BYTES+1)
            if len(expanded)>MAX_BYTES or inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
                raise ValueError('Invalid or excessive compressed MAT')
            kind,value,end=element(expanded,0)
            if end!=len(expanded): raise ValueError('Unexpected compressed framing')
        if kind!=14: raise ValueError('Unexpected top-level MAT element')
        matrices.append(value)
    if len(matrices)!=1: raise ValueError('Expected one behavioral variable')
    data=matrices[0];cls,shape,name,pos=matrix_prefix(data)
    if cls!=2 or shape not in ((1,28),(28,1)) or name!='After_remark':
        raise ValueError('Unexpected behavioral struct geometry')
    kind,flen,pos=element(data,pos)
    if kind!=5 or len(flen)!=4: raise ValueError('Invalid field width tag')
    width=struct.unpack('<i',flen)[0];kind,names,pos=element(data,pos)
    if kind!=1 or not 1<=width<=128 or len(names)%width:
        raise ValueError('Invalid field names geometry')
    fields=[bytes(names[i:i+width]).split(b'\0')[0].decode('ascii') for i in range(0,len(names),width)]
    if len(fields)!=5 or set(fields)!=FIELDS: raise ValueError('Unexpected behavioral fields')
    rows=[]
    for _ in range(28):
        identity={};score=None
        for field in fields:
            kind,value,pos=element(data,pos)
            if kind!=14: raise ValueError('Expected outer field matrix')
            if field in ('trial','vid'): identity[field]=identity_scalar(value)
            elif field=='score': score=value  # opaque view, not numeric interpretation
            # Accuracy and ResponseTime are skipped without any inner parsing.
        rows.append((identity,score))
    identities=[r[0] for r in rows]
    if pos!=len(data) or sorted(r['trial'] for r in identities)!=list(range(1,29)) or sorted(r['vid'] for r in identities)!=list(range(1,29)):
        raise ValueError('Missing/duplicate identities or trailing fields')
    if identities!=expected_identities: raise ValueError('Identity crosswalk differs from authenticated receipt')
    return rows


def score_storage(data):
    cls,shape,name,pos=matrix_prefix(data)
    _,flags,_=element(data,0)
    flag=struct.unpack_from('<I',flags)[0]
    if flag & 0x200 or cls not in NUMERIC or shape not in ((1,12),(12,1)) or name!='':
        raise ValueError('Require a nonlogical real 12-item numeric vector')
    kind,payload,end=element(data,pos)
    expected_kind,fmt=NUMERIC[cls];width=struct.calcsize('<'+fmt)
    if kind!=expected_kind or len(payload)!=12*width or end!=len(data):
        raise ValueError('Score type/length/trailing data mismatch')
    descriptor={'matlab_class':cls,'shape':list(shape),'mi_numeric_type':kind,
                'bytes_per_item':width,'item_count':12,'item10_byte_range':[9*width,10*width],
                'matlab_item_order':'Vector linear order; item 10 is zero-based offset 9',
                'score_values_decoded':0}
    return descriptor,payload,fmt


def audit_score_storage(raw, allowed_vids, expected_identities):
    allowed=set(allowed_vids)
    if not allowed or not allowed<=set(range(1,29)): raise ValueError('Invalid permitted clip set')
    result=[];skipped=0
    for identity,opaque_score in behavioral_rows(raw,expected_identities):
        if identity['vid'] not in allowed:
            skipped+=1;continue
        descriptor,_,_=score_storage(opaque_score)
        result.append({**identity,'score_storage':descriptor})
    if {r['vid'] for r in result}!=allowed: raise ValueError('Requested storage identity missing')
    return result,{'score_values_decoded':0,'score_headers_inspected':len(result),
                   'protected_score_headers_skipped':skipped,
                   'accuracy_and_response_matrices_skipped':56}


def decode_item10(raw, allowed_vids, expected_identities, required_storage=None, observer=None):
    allowed=set(allowed_vids)
    if not allowed or not allowed<=set(range(1,29)): raise ValueError('Invalid permitted clip set')
    # Validate every identity against the receipt before decoding any rating.
    rows=behavioral_rows(raw,expected_identities);decoded=[]
    for identity,opaque_score in rows:
        if identity['vid'] not in allowed: continue
        descriptor,payload,fmt=score_storage(opaque_score)
        if required_storage is not None and descriptor!=required_storage[identity['vid']]:
            raise ValueError('Selected score storage differs from structural qualification')
        lo,hi=descriptor['item10_byte_range']
        if observer is not None: observer({'vid':identity['vid'],'item_one_based':10,'bytes':[lo,hi]})
        value=struct.unpack('<'+fmt,payload[lo:hi])[0]
        if not math.isfinite(value) or not 0<=value<=7:
            raise ValueError('Selected valence is nonfinite or outside the documented scale')
        decoded.append({**identity,'valence':float(value)})
    if {r['vid'] for r in decoded}!=allowed: raise ValueError('Requested target identity missing')
    return decoded,{'item10_values_decoded':len(decoded),'other_rating_values_decoded':0,
                    'protected_score_headers_skipped':28-len(decoded),
                    'accuracy_and_response_matrices_skipped':56}


def read_declared_item10(raw_root,subject,allowed_pairs,expected_sha256,expected_identities,
                         required_storage,reader=None,observer=None):
    """Future access wrapper; rejects protected people before any file operation."""
    allowed_vids={vid for person,vid in allowed_pairs if person==subject}
    if not allowed_vids: raise PermissionError('Participant is outside the declared target access')
    if not re.fullmatch(r'sub-\d{3}',subject): raise ValueError('Invalid participant identity')
    root=Path(raw_root).resolve();path=(root/subject.replace('sub-','sub')/'After_remarks.mat').resolve()
    if not path.is_relative_to(root): raise ValueError('Target path leaves original-data root')
    if reader is None:
        if path.stat().st_size>MAX_BYTES: raise ValueError('Behavioral file exceeds budget')
        raw=path.read_bytes()
    else: raw=reader(path)
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Behavioral source bytes changed before numerical access')
    return decode_item10(raw,allowed_vids,expected_identities,required_storage,observer)
