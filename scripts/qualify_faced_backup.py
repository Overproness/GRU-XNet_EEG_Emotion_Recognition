"""Outcome-blind FACED header qualification with strict byte-range limits."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import requests

REPO = Path(__file__).resolve().parents[1]
PRIVATE = REPO.parent / 'publication_runs/faced_backup_2026-10-10'
PUBLIC = REPO / 'results/development/faced_backup_2026-10-10'
PIN = '4c37c73ece79e68702de2d8afa6b9274811a7cc8'
RAW = 'https://raw.githubusercontent.com/nemarDatasets/nm000112/' + PIN + '/'
DATA = 'https://data.nemar.org/nm000112/v1.1.3/'
FIELDS = [('label',16), ('transducer',80), ('unit',8), ('physical_min',8),
          ('physical_max',8), ('digital_min',8), ('digital_max',8),
          ('prefilter',80), ('samples',8), ('reserved',32)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def metadata_get(path, blob):
    # Never fetch events.tsv, participants.tsv, scans.tsv or original rating MATs.
    allowed = ('_eeg.bdf', '_eeg.json', '_channels.tsv')
    if not path.startswith('sub-') or not path.endswith(allowed):
        raise ValueError('Forbidden source path')
    with requests.get(RAW + path, stream=True, timeout=(10,25), allow_redirects=False) as response:
        if response.status_code != 200:
            raise ValueError('Metadata did not return directly')
        chunks = bytearray()
        for chunk in response.iter_content(4096):
            chunks.extend(chunk)
            if len(chunks) > 20000:
                raise ValueError('Metadata exceeds budget')
        data = bytes(chunks)
    if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != blob:
        raise ValueError('Frozen Git metadata changed')
    return data


def pointer(data):
    matches = re.findall(rb'SHA256E-s(\d+)--([0-9a-f]{64})\.bdf', data)
    if not matches or len(set(matches)) != 1:
        raise ValueError('Invalid annex pointer')
    size, digest = matches[0]
    return int(size), digest.decode()


def checked_range(response, start, end, total):
    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
        raise ValueError('Server ignored or changed the bounded byte range')
    expected = end-start+1
    if response.headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('Encoded byte range cannot certify original offsets')
    data = bytearray()
    for chunk in response.iter_content(4096):
        data.extend(chunk)
        if len(data) > expected:
            raise ValueError('Range exceeds declared length')
    if len(data) != expected:
        raise ValueError('Truncated byte range')
    return bytes(data)


def range_get(path, start, end, total):
    with requests.get(DATA + path, headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'},
                      timeout=(10,25), stream=True, allow_redirects=False) as response:
        return checked_range(response, start, end, total)


def parse_header(data, total):
    if len(data) < 256 or data[:8] != b'\xffBIOSEMI':
        raise ValueError('Unsupported BDF signature')
    count = int(data[252:256]); size = int(data[184:192]); records = int(data[236:244]); duration = float(data[244:252])
    if not 1 <= count <= 128 or size != 256+256*count or len(data) != size or records <= 0 or duration <= 0:
        raise ValueError('Invalid BDF geometry')
    channels = [{} for _ in range(count)]
    offset = 256
    for key,width in FIELDS:
        for channel in channels:
            # Identity and absolute-date fields in the fixed header are ignored.
            value = data[offset:offset+width].decode('ascii',errors='strict').strip()
            offset += width
            if key not in ('transducer','reserved'):
                channel[key] = value
    for channel in channels:
        for key in ('physical_min','physical_max'):
            channel[key] = float(channel[key])
        for key in ('digital_min','digital_max','samples'):
            channel[key] = int(channel[key])
        if channel['digital_min'] >= channel['digital_max'] or channel['physical_min'] >= channel['physical_max'] or channel['samples'] <= 0:
            raise ValueError('Invalid channel calibration')
        channel['sampling_frequency'] = channel['samples']/duration
        channel['physical_units_per_count'] = (channel['physical_max']-channel['physical_min'])/(channel['digital_max']-channel['digital_min'])
    geometry_matches = total == size + records*sum(c['samples'] for c in channels)*3
    return {'header_bytes':size, 'signals':count, 'records':records, 'record_duration_seconds':duration,
            'recording_duration_seconds':records*duration, 'file_geometry_matches':geometry_matches, 'channels':channels}


def collect_one(subject, paths):
    prefix = f'{subject}/eeg/{subject}_task-watchingVideoClips'
    record = {'subject':subject, 'rating_values_decoded':0, 'waveform_samples_decoded':0}
    try:
        for suffix,name in (('_eeg.bdf','pointer'), ('_eeg.json','sidecar'), ('_channels.tsv','channels')):
            path = prefix+suffix
            data = metadata_get(path, paths[path]['sha'])
            target = PRIVATE/'metadata'/subject/(name+('.json' if name=='sidecar' else '.bin'))
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            record[name+'_sha256'] = sha(data)
        size,digest = pointer((PRIVATE/'metadata'/subject/'pointer.bin').read_bytes())
        record.update({'expected_object_bytes':size, 'annex_declared_object_sha256':digest})
        first = range_get(prefix+'_eeg.bdf', 0, 255, size)
        header_size = int(first[184:192])
        signals = int(first[252:256])
        if not 1 <= signals <= 128 or header_size != 256+signals*256:
            raise ValueError('Header size exceeds declared geometry')
        header = first + range_get(prefix+'_eeg.bdf', 256, header_size-1, size)
        target = PRIVATE/'headers'/(subject+'.bdf.header')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(header)
        record.update({'header_sha256':sha(header), 'header':parse_header(header,size), 'passed':True})
    except (requests.RequestException,ValueError,KeyError,UnicodeError) as error:
        record.update({'passed':False, 'failure_type':type(error).__name__, 'failure_message':str(error)[:180]})
    return record


def declare():
    tree = json.loads((PRIVATE/'source_metadata/nemar_tree.response').read_text())
    if tree.get('truncated'):
        raise ValueError('Incomplete source inventory')
    paths = {r['path']:r for r in tree['tree'] if r['type']=='blob'}
    subjects = sorted({p.split('/')[0] for p in paths if p.endswith('_eeg.bdf')})
    if subjects != [f'sub-{i:03d}' for i in range(123)]:
        raise ValueError('Unexpected FACED inventory')
    salt = 'GRU-XNet_FACED_backup_reservation_v1_2026-10-10'
    order = sorted(subjects,key=lambda s:sha((salt+':'+s).encode()))
    reservation = {'status':'frozen_before_header_collection_and_any_outcomes', 'salt':salt, 'source_revision':PIN,
                   'confirmation':sorted(order[:33]), 'development_validation':sorted(order[33:53]),
                   'development_source':sorted(order[53:]), 'material_reservation_status':'pending_verified_stimulus_family_metadata',
                   'all_material_outcomes_sealed':True, 'no_role_replacement_from_outcomes':True,
                   'scope':'Local fresh identities; no claim of unseen biological participants or content in pretrained checkpoints.'}
    plan = {'date':'2026-10-10','timezone':'Asia/Karachi','source_revision':PIN,'curator_version':'1.1.3',
            'participants':subjects,'source_tree_sha256':sha((PRIVATE/'source_metadata/nemar_tree.response').read_bytes()),
            'allowed_metadata_suffixes':['_eeg.bdf annex pointer only','_eeg.json','_channels.tsv'],
            'forbidden_downloads':['events.tsv','participant rating MAT','EEG samples','participants.tsv','scans.tsv'],
            'maximum_header_bytes_per_file':33024,'range_requests':'exact 206/Content-Range, no redirects, no encoded transfer',
            'metadata_bytes_limit_per_response':20000,'workers':4,'script_sha256':sha(Path(__file__).read_bytes()),
            'expected_full_object_hashes_not_reverified':True,'models_fitted':0,'manuscript_change':False}
    for name,value in (('plan.json',plan),('participant_reservation.json',reservation)):
        if (PUBLIC/name).exists():raise ValueError('Do not overwrite a frozen declaration')
        write(PUBLIC/name,value)
    print(json.dumps({'declared':True,'participants':len(subjects),'confirmation':33,'development_validation':20,'development_source':70,'material_reservation':'pending; every material outcome sealed'}))


def collect():
    plan = json.loads((PUBLIC/'plan.json').read_text())
    if plan['script_sha256'] != sha(Path(__file__).read_bytes()):raise ValueError('Declared collector changed')
    if (PUBLIC/'collection.json').exists():raise ValueError('Do not overwrite a prior collection')
    tree = json.loads((PRIVATE/'source_metadata/nemar_tree.response').read_text())
    paths = {r['path']:r for r in tree['tree'] if r['type']=='blob'}
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(collect_one,s,paths):s for s in plan['participants']}
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%20==0: print(f'Checked {len(records)}/123 headers',flush=True)
    write(PUBLIC/'collection.json',{'records':sorted(records,key=lambda r:r['subject']), 'raw_samples_decoded':0,
                                 'rating_values_decoded':0, 'source_revision':PIN,'plan_sha256':sha((PUBLIC/'plan.json').read_bytes()),
                                 'participant_reservation_sha256':sha((PUBLIC/'participant_reservation.json').read_bytes())})
    print(json.dumps({'completed':len(records),'passed':sum(r['passed'] for r in records),'failed':sum(not r['passed'] for r in records)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','collect'))
    args=parser.parse_args();declare() if args.mode=='declare' else collect()
