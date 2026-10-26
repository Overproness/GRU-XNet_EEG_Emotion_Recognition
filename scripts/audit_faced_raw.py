"""Outcome-blind local FACED extraction/header/event/identity qualification.

Opaque hashing is allowed; EEG numbers and rating values are never decoded.
MAT parsing skips score/Accuracy/ResponseTime matrices at their outer tags.
No raw payloads, private header fields or recordInformation JSON are exported.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import time
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
RAW = WORKSPACE / 'Data/Data'
PUBLIC = REPO / 'results/development/faced_raw_2026-10-11'
BACKUP = REPO / 'results/development/faced_backup_2026-10-10'
CODEBOOK = REPO / 'results/development/faced_codebooks_2026-10-10'
SUBJECTS = [f'sub{i:03d}' for i in range(123)]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    if path.exists():
        raise ValueError('Preserve frozen evidence: ' + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def bindings():
    return {str(p.relative_to(WORKSPACE)).replace('\\', '/'): digest(p) for p in (
        Path(__file__), BACKUP / 'participant_reservation.json', CODEBOOK / 'audit.json',
        CODEBOOK / 'plan.json', WORKSPACE / 'report.tex',
        REPO / 'results/development/emo_mc_qualification_2026-10-10/reservation.json')}


def declare():
    save(PUBLIC / 'plan.json', {
        'date': '2026-10-11', 'timezone': 'Asia/Karachi', 'bindings': bindings(),
        'source': 'User-supplied Data.zip and extracted Data/Data, not an independently authenticated release checksum',
        'allowed': ['ZIP member metadata and opaque streaming CRC32/SHA256 of archive and extracted bytes',
                    'All 123 technical BDF headers, ignoring identity/date fields',
                    'Original evt.bdf annotation channels only, numerical task codes and relative times',
                    'After_remarks.mat struct field names and trial/vid identity scalars only'],
        'forbidden': ['EEG numerical samples', 'score/Accuracy/ResponseTime numerical decoding',
                      'recordInformation.json interpretation', 'individual outcomes', 'fits', 'material reassignment'],
        'endpoint_rule': 'Require one video identity then start then end; flag stray/replaced starts and duplicate identities; compare separately to curator state machine',
        'duration_comparison': 'Report observed minus catalogue/documented duration; tolerance 1 second for metadata discrepancy flag, not a cropping rule',
        'time_geometry': 'Compare relative marker endpoints with data record duration; physical same-session clock synchronization remains separate',
        'zip_deletion_rule': 'Only after every non-directory member has matching safe relative path, size and streamed CRC32; preserve SHA256 receipts and exact inventory',
        'ratings_values_decoded': 0, 'waveform_samples_decoded': 0, 'models_fitted': 0,
        'research_question_changed': False, 'outcome_access_cleared': False})
    print(json.dumps({'declared': True, 'participants': 123}), flush=True)


def check_plan():
    plan = load(PUBLIC / 'plan.json')
    if plan['bindings'] != bindings():
        raise ValueError('Declared dependencies changed')
    return plan


def member_target(name, root):
    if '\\' in name or '\x00' in name:
        raise ValueError('Unsafe ZIP member')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or any(':' in x for x in p.parts):
        raise ValueError('Unsafe ZIP member')
    target = root.joinpath(*p.parts).resolve()
    target.relative_to(root.resolve())
    if not p.parts:
        raise ValueError('Empty ZIP path')
    return target


def archive_audit():
    check_plan()
    archive_path = WORKSPACE / 'Data.zip'
    extraction_root = WORKSPACE / 'Data'
    records, failures = [], []
    start = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        seen = set()
        for i, info in enumerate(members):
            target = member_target(info.filename, extraction_root)
            key = str(target).casefold()
            if key in seen:
                raise ValueError('Duplicate ZIP member after Windows path normalization')
            seen.add(key)
            if info.is_dir():
                if not target.is_dir():
                    failures.append({'member': info.filename, 'reason': 'missing_directory'})
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('ZIP symlink outside declared scope')
            row = {'member': info.filename, 'bytes': info.file_size, 'zip_crc32': f'{info.CRC:08x}'}
            if not target.is_file() or target.stat().st_size != info.file_size:
                row['passed'] = False
                failures.append({'member': info.filename, 'reason': 'missing_or_size_mismatch'})
            else:
                crc, h, payload = 0, hashlib.sha256(), hashlib.sha256()
                skip = None
                if target.name == 'data.bdf':
                    with target.open('rb') as header_file:
                        first = header_file.read(256)
                    skip = int(first[184:192])
                    if not 256 <= skip <= 33024:
                        raise ValueError('Unexpected BDF header budget')
                offset = 0
                with target.open('rb') as f:
                    while chunk := f.read(8 * 1024 * 1024):
                        crc = zlib.crc32(chunk, crc)
                        h.update(chunk)
                        if skip is not None and offset + len(chunk) > skip:
                            payload.update(chunk[max(0, skip - offset):])
                        offset += len(chunk)
                row.update({'extracted_crc32': f'{crc:08x}', 'sha256': h.hexdigest(), 'passed': crc == info.CRC})
                if skip is not None:
                    row.update({'signal_payload_sha256': payload.hexdigest(), 'header_bytes': skip})
                if not row['passed']:
                    failures.append({'member': info.filename, 'reason': 'crc_mismatch'})
            records.append(row)
            if len(records) % 20 == 0 or i == len(members)-1:
                print(json.dumps({'files_checked': len(records), 'failures': len(failures), 'elapsed_seconds': round(time.monotonic()-start)}), flush=True)
    listed = {r['member'] for r in records}
    actual = {str(p.relative_to(extraction_root)).replace('\\', '/') for p in extraction_root.rglob('*') if p.is_file()}
    extras = sorted(actual - listed)
    print(json.dumps({'phase': 'hash_archive', 'archive_bytes': archive_path.stat().st_size}), flush=True)
    report = {'plan_sha256': digest(PUBLIC/'plan.json'), 'archive_name': 'Data.zip',
              'archive_bytes': archive_path.stat().st_size, 'archive_sha256': digest(archive_path),
              'members': len(members), 'files': len(records), 'extracted_file_bytes': sum(r['bytes'] for r in records),
              'records': records, 'failures': failures, 'extra_extracted_files': extras,
              'all_members_match': not failures, 'deletion_qualified': not failures and not extras,
              'scope': 'Opaque byte hashes/CRC; no rating or EEG numerical values decoded; CRC proves extraction consistency, not author release authenticity'}
    save(PUBLIC/'extraction.json', report)
    print(json.dumps({k:report[k] for k in ('files','all_members_match','deletion_qualified','extracted_file_bytes')}), flush=True)


def read_header(path):
    with path.open('rb') as f:
        first = f.read(256)
        if first[:8] not in (b'\xffBIOSEMI', b'0       '):
            raise ValueError('Unrecognized EDF/BDF signature')
        n, size = int(first[252:256]), int(first[184:192])
        if not 1 <= n <= 128 or size != 256 + 256*n:
            raise ValueError('Invalid header geometry')
        raw = first + f.read(size-256)
    if len(raw) != size:
        raise ValueError('Truncated header')
    pos, columns = 256, {}
    for key,width in [('labels',16),('transducer',80),('units',8),('physical_min',8),('physical_max',8),
                      ('digital_min',8),('digital_max',8),('prefilter',80),('samples_per_record',8),('reserved_signals',32)]:
        columns[key] = [raw[pos+width*i:pos+width*(i+1)].decode('latin1').strip() for i in range(n)]
        pos += width*n
    width = 3 if first[:8] == b'\xffBIOSEMI' else 2
    count, duration = int(first[236:244]), float(first[244:252])
    samples = [int(x) for x in columns['samples_per_record']]
    if count < 0 or duration < 0 or any(x < 0 for x in samples):
        raise ValueError('Incomplete recording geometry')
    if path.stat().st_size != size + count*sum(samples)*width:
        raise ValueError('Header/file-size inconsistency')
    return {'header_sha256':hashlib.sha256(raw).hexdigest(), 'header_bytes':size,'sample_width':width,
            'record_count':count,'record_duration_seconds':duration,'recording_duration_seconds':count*duration,
            'continuity':first[192:236].decode('ascii').strip(), 'columns':columns,
            'sample_rates_hz':[s/duration if duration else None for s in samples]}


def parse_tals(raw):
    events, clocks = [], []
    for tal in raw.split(b'\x00'):
        if not tal:
            continue
        parts = tal.split(b'\x14')
        if len(parts) < 2 or parts[-1] != b'':
            raise ValueError('Invalid TAL termination')
        timing = parts[0].split(b'\x15')
        if len(timing) > 2:
            raise ValueError('Invalid TAL time fields')
        onset = float(timing[0])
        duration = float(timing[1]) if len(timing)==2 else 0.0
        if not math.isfinite(onset) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid annotation time')
        texts = [p for p in parts[1:] if p]
        if not texts:
            clocks.append(onset)
        for text in texts:
            # Do not decode/export arbitrary narratives or possible identifiers.
            if not re.fullmatch(rb'[0-9]{1,4}', text):
                raise ValueError('Non-numeric annotation requires a separately declared projection')
            events.append({'onset_seconds':onset,'duration_seconds':duration,'code':int(text)})
    return events, clocks


def read_events(path, header):
    cols = header['columns']
    if any(label not in ('BDF Annotations','EDF Annotations') for label in cols['labels']):
        raise ValueError('evt.bdf contains a non-annotation channel')
    lengths = [int(s)*header['sample_width'] for s in cols['samples_per_record']]
    events, clocks = [], []
    with path.open('rb') as f:
        f.seek(header['header_bytes'])
        for _ in range(header['record_count']):
            for n in lengths:
                rows, timekeeping = parse_tals(f.read(n))
                events.extend(rows)
                clocks.extend(timekeeping)
    if events != sorted(events,key=lambda r:r['onset_seconds']):
        raise ValueError('Annotation times not monotonic')
    return events, clocks


def strict_spans(events):
    spans, anomalies = [], []
    clip, start = None, None
    for i,r in enumerate(events):
        code, t = r['code'], r['onset_seconds']
        if 1 <= code <= 28:
            if clip is not None or start is not None:
                anomalies.append({'event_index':i,'type':'unclosed_identity_or_start'})
            clip, start = code, None
        elif code == 101:
            if clip is None:
                anomalies.append({'event_index':i,'type':'unidentified_start'})
            elif start is not None:
                anomalies.append({'event_index':i,'type':'replaced_start'})
                clip, start = None, None
            else:
                start = t
        elif code == 102:
            if clip is None or start is None or t <= start:
                anomalies.append({'event_index':i,'type':'unpaired_end'})
            else:
                spans.append({'vid':clip,'start_seconds':start,'end_seconds':t,'duration_seconds':t-start})
            clip, start = None, None
        elif code == 100 and (clip is not None or start is not None):
            anomalies.append({'event_index':i,'type':'experiment_start_interrupts_candidate'})
            clip, start = None, None
    if clip is not None or start is not None:
        anomalies.append({'event_index':len(events),'type':'open_candidate_at_eof'})
    return spans, anomalies


def curator_spans(events):
    # Independent implementation of pinned curator timing logic only; no ratings.
    clip, start, spans = None, None, []
    for r in events:
        code, t = r['code'], r['onset_seconds']
        if 1 <= code <= 28:
            clip = code
        elif code == 101:
            start = t
        elif code == 102 and clip is not None and start is not None:
            spans.append({'vid':clip,'start_seconds':start,'end_seconds':t,'duration_seconds':t-start})
            clip, start = None, None
    return spans


def element(data, pos, padded=True):
    if pos+8 > len(data):
        raise ValueError('Truncated MAT tag')
    tag, length = struct.unpack_from('<II', data, pos)
    small = tag >> 16
    if small:
        if small > 4:
            raise ValueError('Invalid small MAT element')
        return tag & 0xffff, memoryview(data)[pos+4:pos+4+small], pos+8
    end = pos+8+length
    if end > len(data):
        raise ValueError('Truncated MAT payload')
    return tag, memoryview(data)[pos+8:end], end + ((-length)%8 if padded and tag != 15 else 0)


def matrix_prefix(data):
    kind, flags, pos = element(data,0)
    if kind != 6 or len(flags)!=8:
        raise ValueError('Unexpected MAT flags')
    flag = struct.unpack_from('<I',flags)[0]
    if flag & 0x800:
        raise ValueError('Complex MAT arrays unsupported')
    kind,dims,pos = element(data,pos)
    if kind != 5 or len(dims)%4:
        raise ValueError('Unexpected MAT dimensions')
    shape = struct.unpack('<'+'i'*(len(dims)//4),dims)
    kind,name,pos = element(data,pos)
    if kind != 1:
        raise ValueError('Unexpected MAT name')
    return flag & 255, shape, bytes(name).decode('ascii'), pos


def identity_scalar(data):
    cls, shape, _, pos = matrix_prefix(data)
    if math.prod(shape)!=1 or cls not in (6,8,9,10,11,12,13,14,15):
        raise ValueError('Identity is not a numeric scalar')
    kind,value,end = element(data,pos)
    formats = {1:'b',2:'B',3:'h',4:'H',5:'i',6:'I',7:'f',9:'d',12:'q',13:'Q'}
    if kind not in formats or len(value)!=struct.calcsize('<'+formats[kind]) or end!=len(data):
        raise ValueError('Unexpected identity encoding')
    number = struct.unpack('<'+formats[kind],value)[0]
    if not math.isfinite(number) or int(number)!=number or not 1 <= number <= 28:
        raise ValueError('Invalid trial/video identity')
    return int(number)


def identity_projection(raw):
    if len(raw)>1000000 or len(raw)<136 or raw[126:128]!=b'IM':
        raise ValueError('Only bounded little-endian Level 5 MAT is allowed')
    pos, matrices = 128, []
    while pos < len(raw):
        kind,value,pos = element(raw,pos)
        if kind == 15:
            inflater = zlib.decompressobj()
            expanded = inflater.decompress(value,1000001)
            if len(expanded)>1000000 or inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
                raise ValueError('Compressed MAT exceeds budget or invalid stream')
            kind,value,end = element(expanded,0)
            if end != len(expanded):
                raise ValueError('Unexpected compressed MAT framing')
        if kind != 14:
            raise ValueError('Unexpected top-level MAT element')
        matrices.append(value)
    if len(matrices)!=1:
        raise ValueError('Expected one After_remark variable')
    data = matrices[0]
    cls,shape,name,pos = matrix_prefix(data)
    if cls!=2 or math.prod(shape)!=28 or name!='After_remark':
        raise ValueError('Unexpected behavioral structure')
    kind,flen,pos = element(data,pos)
    if kind!=5 or len(flen)!=4:
        raise ValueError('Unexpected field length')
    width = struct.unpack('<i',flen)[0]
    kind,names,pos = element(data,pos)
    if kind!=1 or not 1<=width<=128 or len(names)%width:
        raise ValueError('Unexpected field names')
    fields = [bytes(names[i:i+width]).split(b'\0')[0].decode('ascii') for i in range(0,len(names),width)]
    if len(fields)!=5 or set(fields)!={'score','trial','vid','Accuracy','ResponseTime'}:
        raise ValueError('Behavioral fields differ from source codebook')
    rows,skipped = [], 0
    for _ in range(28):
        row = {}
        for field in fields:
            kind,value,pos = element(data,pos)
            if kind!=14:
                raise ValueError('Expected field matrix')
            if field in ('trial','vid'):
                row[field] = identity_scalar(value)
            else:
                # No matrix_prefix or numeric unpack is performed on protected fields.
                skipped += 1
        rows.append(row)
    if pos!=len(data) or sorted(r['trial'] for r in rows)!=list(range(1,29)) or sorted(r['vid'] for r in rows)!=list(range(1,29)):
        raise ValueError('Missing/duplicate identities or trailing matrices')
    return rows, {'fields':fields,'skipped_protected_matrices':skipped,'rating_values_decoded':0}


def inspect_headers():
    check_plan()
    records=[]
    for subject in SUBJECTS:
        row={'subject':subject.replace('sub','sub-')}
        for name in ('data.bdf','evt.bdf'):
            try:
                row[name]=read_header(RAW/subject/name)
            except (ValueError,UnicodeError) as error:
                row[name]={'failed':type(error).__name__,'reason':str(error)}
        records.append(row)
    save(PUBLIC/'headers.json',{'plan_sha256':digest(PUBLIC/'plan.json'),'records':records,
        'identity_and_date_fields_exported':False,'waveform_samples_decoded':0})
    print(json.dumps({'headers':246,'failed':sum('failed' in r[n] for r in records for n in ('data.bdf','evt.bdf'))}))


def audit_joins():
    check_plan()
    headers=load(PUBLIC/'headers.json')['records']
    stimuli={r['clip']:r for r in load(CODEBOOK/'audit.json')['stimuli']}
    records=[]
    for row in headers:
        subject=row['subject']; path=RAW/subject.replace('sub-','sub')
        result={'subject':subject}
        try:
            if any('failed' in row[n] for n in ('data.bdf','evt.bdf')):
                raise ValueError('Header qualification failed')
            events,clocks=read_events(path/'evt.bdf',row['evt.bdf'])
            spans,anomalies=strict_spans(events)
            identities,scope=identity_projection((path/'After_remarks.mat').read_bytes())
            observed=[s['vid'] for s in spans]
            join={r['vid']:r['trial'] for r in identities}
            for s in spans:
                expected=83.0 if s['vid']==22 and 36<=int(subject[-3:])<=60 else stimuli[s['vid']]['catalogue_duration_seconds']
                s.update({'trial':join[s['vid']],'expected_duration_seconds':expected,
                          'duration_difference_seconds':s['duration_seconds']-expected})
            result.update({'events':events,'annotation_clocks':clocks,'spans':spans,'anomalies':anomalies,
                'code_counts':dict(Counter(str(r['code']) for r in events)), 'identity_scope':scope,
                'identity_rows':identities,'all_28_videos_once':sorted(observed)==list(range(1,29)),
                'presentation_order_matches': [join[v] for v in observed]==list(range(1,29)),
                'curator_spans_equal':curator_spans(events)==[{k:s[k] for k in ('vid','start_seconds','end_seconds','duration_seconds')} for s in spans],
                'all_endpoints_in_data_geometry':all(0<=s['start_seconds']<s['end_seconds']<=row['data.bdf']['recording_duration_seconds'] for s in spans),
                'duration_flags_over_one_second':[s['vid'] for s in spans if abs(s['duration_difference_seconds'])>1.0],
                'passed':True})
        except (ValueError,UnicodeError,struct.error,zlib.error,KeyError) as error:
            result.update({'passed':False,'failure_type':type(error).__name__,'reason':str(error)})
        records.append(result)
    save(PUBLIC/'joins.json',{'plan_sha256':digest(PUBLIC/'plan.json'),'headers_sha256':digest(PUBLIC/'headers.json'),
        'records':records,'rating_values_decoded':0,'waveform_samples_decoded':0,'models_fitted':0})
    print(json.dumps({'participants':len(records),'parsed':sum(r['passed'] for r in records),
        'failures':Counter(r.get('reason','') for r in records if not r['passed'])}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('declare','archive','headers','joins'))
    args=parser.parse_args()
    {'declare':declare,'archive':archive_audit,'headers':inspect_headers,'joins':audit_joins}[args.mode]()
