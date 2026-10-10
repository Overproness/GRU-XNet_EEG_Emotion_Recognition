"""Bounded, outcome-blind source qualification for the pinned EmoEEG-MC release.

Only EDF headers and the two identity columns of behaviour tables are interpreted.
Score columns are never decoded or retained. No signal samples are interpreted.
Downloaded third-party sources stay outside the public repository.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote

import requests

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
PRIVATE = WORK / 'publication_runs/emo_mc_qualification_2026-10-10'
PUBLIC = REPO / 'results/development/emo_mc_qualification_2026-10-10'
OLD = WORK / 'publication_runs/data_target_review_2026-10-10/source_metadata'
PIN = '83b5b7f28dea9964a5389509548219a27b755cc4'
AUTHOR_PIN = 'd59aed5f9880a19df2a7db71f91caa034b8573c7'
S3 = 'https://s3.amazonaws.com/openneuro.org/ds005540/'
RAW = 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds005540/' + PIN + '/'
FIELDS = [('label', 16), ('transducer', 80), ('unit', 8),
          ('physical_min', 8), ('physical_max', 8), ('digital_min', 8),
          ('digital_max', 8), ('prefilter', 80), ('samples', 8), ('reserved', 32)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def small_get(url, limit=200000, expected_blob=None):
    with requests.get(url, timeout=(10, 30), stream=True) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(8192):
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError('Metadata exceeds declared budget')
        body = bytes(body)
    if expected_blob and git_blob(body) != expected_blob:
        raise ValueError('Frozen Git blob mismatch')
    return body


def range_get(url, start, end, total, version=None):
    """Fail closed when a server ignores Range; never read the full EEG body."""
    params = {'versionId': version} if version else None
    with requests.get(url, params=params, headers={'Range': f'bytes={start}-{end}'},
                      timeout=(10, 30), stream=True) as response:
        expected = f'bytes {start}-{end}/{total}'
        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
            raise ValueError(f'Incorrect bounded response: {response.status_code}')
        body = bytearray()
        for chunk in response.iter_content(4096):
            body.extend(chunk)
            if len(body) > end - start + 1:
                raise ValueError('Range budget exceeded')
        if len(body) != end - start + 1:
            raise ValueError('Truncated range')
        return bytes(body), response.headers.get('x-amz-version-id')


def annex_pointer(body):
    matches = re.findall(rb'SHA256E-s(\d+)--([a-f0-9]{64})\.edf', body)
    if not matches or len(set(matches)) != 1:
        raise ValueError('Unexpected annex pointer')
    size, digest = matches[0]
    return int(size), digest.decode()


def parse_edf_header(body, file_size):
    """Parse declared EDF calibration, without patient/date fields or signal data."""
    if len(body) < 256 or body[:8].strip() != b'0':
        raise ValueError('Not a supported EDF header')
    count = int(body[252:256])
    header_bytes = int(body[184:192])
    records = int(body[236:244])
    duration = float(body[244:252])
    if count < 1 or duration <= 0 or records < 1:
        raise ValueError('Invalid record geometry')
    if header_bytes != 256 + 256 * count or len(body) != header_bytes:
        raise ValueError('Incorrect header geometry')
    channels = [{} for _ in range(count)]
    offset = 256
    for key, width in FIELDS:
        for i, channel in enumerate(channels):
            value = body[offset + i * width:offset + (i + 1) * width]
            channel[key] = value.decode('latin-1').strip()
        offset += count * width
    for channel in channels:
        channel.pop('reserved')
        for key in ('physical_min', 'physical_max'):
            channel[key] = float(channel[key])
        for key in ('digital_min', 'digital_max', 'samples'):
            channel[key] = int(channel[key])
        if channel['digital_max'] <= channel['digital_min'] or channel['samples'] <= 0:
            raise ValueError('Invalid channel calibration')
        if channel['physical_max'] <= channel['physical_min']:
            raise ValueError('Invalid physical calibration')
        channel['sampling_hz'] = channel['samples'] / duration
        channel['gain_in_declared_unit'] = (
            (channel['physical_max'] - channel['physical_min']) /
            (channel['digital_max'] - channel['digital_min']))
        channel['offset_in_declared_unit'] = (
            channel['physical_min'] - channel['digital_min'] * channel['gain_in_declared_unit'])
    record_bytes = sum(c['samples'] * 2 for c in channels)
    size_delta = file_size - (header_bytes + record_bytes * records)
    return dict(header_bytes=header_bytes, records=records, record_duration_s=duration,
                record_bytes=record_bytes, declared_size_delta_bytes=size_delta,
                geometry_consistent=(size_delta == 0),
                format=body[192:236].decode().strip(), channels=channels)


def identity_projection(body):
    """Decode exactly trial_number/video_name; no access to any score field."""
    lines = body.splitlines()
    columns = lines[0].decode('utf-8-sig').split('\t')
    expected = ['trial_number', 'video_name'] + [f'score_{i}' for i in range(1, 11)]
    if columns != expected:
        raise ValueError('Behaviour schema differs from declared schema')
    identities = []
    for line in lines[1:]:
        if not line.strip():
            continue
        # Split at two delimiters. The remaining opaque bytes are never decoded.
        first, second, opaque_scores = line.split(b'\t', 2)
        from decimal import Decimal
        trial = Decimal(first.decode('ascii'))
        if not trial.is_finite() or trial != trial.to_integral_value():
            raise ValueError('Nonintegral trial identity')
        identities.append({'trial_number': int(trial), 'video_name_field': second.decode('ascii')})
        del opaque_scores
    return columns, identities


def declared_orders(text):
    """Use the documented per-person orders including shortened context lists."""
    import ast
    full = text.split('### Full Trial Participants', 1)[1].split('### Participants with Missing Trials', 1)[0]
    out = {}
    for number, listing in re.findall(r'^\s*(\d+)\. `([^`]+)`', full, flags=re.M):
        out[f'sub-{int(number):02d}'] = {'ima': ast.literal_eval(listing), 'vid': ast.literal_eval(listing)}
    missing = text.split('### Participants with Missing Trials', 1)[1].split("## Participants' Behaviour Reports", 1)[0]
    for number, block in re.findall(r'\*\*sub(\d+)\*\*:([^*]*(?:(?!\*\*sub).)*)', missing, flags=re.S):
        lists = [ast.literal_eval(x) for x in re.findall(r'`(\[[^`]+\])`', block)]
        out[f'sub-{int(number):02d}'] = dict(zip(('ima', 'vid'), lists if len(lists) == 2 else lists * 2))
    return out


def recording(entry):
    path = entry['path']
    pointer = small_get(RAW + quote(path), expected_blob=entry['sha'])
    size, expected_sha = annex_pointer(pointer)
    url = S3 + quote(path)
    fixed, version = range_get(url, 0, 255, size)
    header_size = int(fixed[184:192])
    if header_size > 100000:
        raise ValueError('Header exceeds budget')
    header, pinned_version = range_get(url, 0, header_size - 1, size, version)
    if version and version != pinned_version:
        raise ValueError('Object version changed within header retrieval')
    PRIVATE.joinpath('headers', path).parent.mkdir(parents=True, exist_ok=True)
    PRIVATE.joinpath('headers', path).with_suffix('.header').write_bytes(header)
    return dict(path=path, participant=path.split('/')[0], git_blob=entry['sha'],
                pointer_sha256=sha(pointer), annex_sha256=expected_sha, file_bytes=size,
                url=url, s3_version=version, header_sha256=sha(header),
                full_object_sha256_verified=False, **parse_edf_header(header, size))


def behaviour(entry):
    body = small_get(RAW + entry['path'], expected_blob=entry['sha'])
    columns, ids = identity_projection(body)
    # No original score table is written to disk.
    return dict(path=entry['path'], git_blob=entry['sha'], source_sha256=sha(body),
                schema=columns, identities=ids, score_values_decoded=0)


def sidecar(entry):
    body = small_get(RAW + entry['path'], expected_blob=entry['sha'])
    return dict(path=entry['path'], git_blob=entry['sha'], source_sha256=sha(body),
                metadata=json.loads(body))


def collect():
    tree = json.loads(OLD.joinpath('emo_mc_tree.response').read_text())
    entries = [e for e in tree['tree'] if e['type'] == 'blob']
    jobs = []
    for e in entries:
        p = e['path']
        if p.startswith('sub-') and '/eeg/' in p and p.endswith('_eeg.edf'):
            jobs.append(('recordings', e, recording))
        elif p.startswith('sub-') and p.endswith('_beh.tsv') and not p.startswith('sub-01/'):
            jobs.append(('behaviour', e, behaviour))
        elif p.startswith('sub-') and p.endswith('_eeg.json'):
            jobs.append(('sidecars', e, sidecar))
    result_path = PRIVATE / 'collection.json'
    result = (json.loads(result_path.read_text()) if result_path.exists() else
              dict(release_pin=PIN, author_pin=AUTHOR_PIN, recordings=[], behaviour=[], sidecars=[], errors=[]))
    completed = {(kind, row['path']) for kind in ('recordings', 'behaviour', 'sidecars') for row in result[kind]}
    jobs = [j for j in jobs if (j[0], j[1]['path']) not in completed]
    # Previous failed attempts stay available in a separate record.
    if result['errors']:
        write_json(PRIVATE / 'initial_errors.json', result['errors'])
    result['errors'] = []
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(fn, entry): (kind, entry['path']) for kind, entry, fn in jobs}
        for i, future in enumerate(as_completed(future_map), 1):
            kind, path = future_map[future]
            try:
                result[kind].append(future.result())
            except Exception as exc:
                result['errors'].append(dict(path=path, stage=kind, error=str(exc)))
            if i % 20 == 0 or i == len(jobs):
                write_json(PRIVATE / 'collection.json', result)
                print(json.dumps({'completed': i, 'total': len(jobs), 'errors': len(result['errors'])}), flush=True)
    for key in ('recordings', 'behaviour', 'sidecars'):
        result[key].sort(key=lambda x: x['path'])
    write_json(PRIVATE / 'collection.json', result)


def inspect_collection():
    from collections import Counter
    result = json.loads((PRIVATE / 'collection.json').read_text())
    orders = declared_orders((OLD / 'emo_mc_readme_fixed.response').read_text(encoding='utf-8'))
    print('counts', {k: len(result[k]) for k in ('recordings', 'behaviour', 'sidecars', 'errors')})
    print('order subjects', len(orders))
    print('EEG calibrations', Counter((c['unit'], c['physical_min'], c['physical_max'], c['digital_min'], c['digital_max'], c['sampling_hz'])
                                    for r in result['recordings'] for c in r['channels'] if c['label'] != 'EDF Annotations'))
    for b in result['behaviour'][:2]:
        print('identity example', b['path'], b['identities'][:3], b['identities'][-3:])
    print('errors', result['errors'])


def ranked(namespace, values):
    return sorted(values, key=lambda value: sha((namespace + '|' + value).encode()))


def catalog():
    """Short metadata only: no stimulus narratives, participant reports or embeddings."""
    from openpyxl import load_workbook
    out = []
    for context, name in [('vid', 'video_description.xlsx'), ('ima', 'imagery_guidence.xlsx')]:
        book = load_workbook(PRIVATE / 'discovery' / name, read_only=True, data_only=True)
        for row in list(book.active.values)[1:]:
            number = int(row[0])
            category = row[3]
            if context == 'vid':
                source = str(row[6])
                # Site names alone do not identify the content family.
                video_id = re.search(r'(BV[a-zA-Z0-9]+|watch\?v=[a-zA-Z0-9_-]+|a\d{7,})', source)
                family = video_id.group(0) if video_id else str(row[1]).strip()
            else:
                family = f'guidance_{number:02d}'
            out.append(dict(context=context, number=number, material_key=f'{context}:{number:02d}',
                            assigned_category=category, family=family, duration_s=row[4]))
        book.close()
    return out


def family_closure(materials, selected_numbers):
    """Keep every known film family together and reserve the paired numeric guidance."""
    selected = set(selected_numbers)
    while True:
        families = {m['family'] for m in materials if m['context'] == 'vid' and m['number'] in selected}
        expanded = selected | {m['number'] for m in materials if m['context'] == 'vid' and m['family'] in families}
        if expanded == selected:
            return sorted(selected)
        selected = expanded


def permit_row(plan, participant, context, number, mapping_verified=False):
    """Fail closed until the material-key mapping is independently verified."""
    if not mapping_verified:
        raise ValueError('Unverified trial-to-material mapping; no fitting or rating access')
    if participant not in plan['participants']['development_source'] + plan['participants']['development_validation']:
        return False
    return f'{context}:{number:02d}' not in plan['materials']['reserved_keys']


def publish_metadata_and_reserve():
    """Create a one-time identity-only reservation before any outcome or signal decoding."""
    from collections import Counter
    result = json.loads((PRIVATE / 'collection.json').read_text())
    if result['errors'] or len(result['recordings']) != 103 or len(result['behaviour']) != 59:
        raise ValueError('Incomplete metadata audit')
    orders = declared_orders((OLD / 'emo_mc_readme_fixed.response').read_text(encoding='utf-8'))
    people = sorted({r['participant'] for r in result['recordings']})
    quarantine = {r['participant'] for r in result['recordings'] if r.get('geometry_consistent') is False}
    quarantine |= set(people) - set(orders)
    pool = ranked('EmoEEG-MC-2026-10-10-participants-v1', set(people) - quarantine)
    materials = catalog()
    selected = []
    for category in sorted({m['assigned_category'] for m in materials}):
        numbers = [f'{m["number"]:02d}' for m in materials if m['context'] == 'vid' and m['assigned_category'] == category]
        selected.append(int(ranked('EmoEEG-MC-2026-10-10-materials-v1|' + category, numbers)[0]))
    reserved_numbers = family_closure(materials, selected)
    reservations = [m['material_key'] for m in materials if m['number'] in reserved_numbers]
    schema_counts = Counter()
    for b in result['behaviour']:
        values = [x.get('video_name_field', x.get('material_id')) for x in b['identities']]
        schema_counts[len(set(values))] += 1
    out = {
        'release_doi': '10.18112/openneuro.ds005540.v1.0.7', 'release_git_pin': PIN,
        'author_code_pin': AUTHOR_PIN, 'collection_sha256': sha((PRIVATE / 'collection.json').read_bytes()),
        'raw_participants': people, 'recordings': len(result['recordings']),
        'recording_counts_by_participant': dict(Counter(r['participant'] for r in result['recordings'])),
        'recording_size_total_bytes': sum(r['file_bytes'] for r in result['recordings']),
        'header_sampling_hz_counts': dict(Counter(r['channels'][0]['sampling_hz'] for r in result['recordings'])),
        'declared_units': sorted({c['unit'] for r in result['recordings'] for c in r['channels'] if c['label'] != 'EDF Annotations'}),
        'gain_uV_per_count': result['recordings'][0]['channels'][0]['gain_in_declared_unit'],
        'offset_uV': result['recordings'][0]['channels'][0]['offset_in_declared_unit'],
        'behaviour_tables': len(result['behaviour']),
        'behaviour_rows': sum(len(b['identities']) for b in result['behaviour']),
        'video_name_unique_value_counts_per_table': dict(schema_counts),
        'score_columns': [f'score_{i}' for i in range(1, 11)], 'rating_values_decoded': 0,
        'waveform_samples_decoded': 0, 'models_fitted': 0,
        'quarantined_participants': sorted(quarantine),
        'missing_readme_order': sorted(set(people) - set(orders)),
        'geometry_discrepancies': [{k:r[k] for k in ('path', 'file_bytes', 'records', 'record_bytes', 'declared_size_delta_bytes')}
                                  for r in result['recordings'] if r.get('geometry_consistent') is False],
        'materials': materials,
        'fitting_ready': False,
        'open_gates': ['Raw trigger/context/material joins, including split-run and incomplete cases',
                       'Spreadsheet number to README symbol mapping is not certified',
                       'Per-recording reference is described only as common',
                       'Full EDF annex digests not yet checked; headers are partial source qualification',
                       'FACED common collection acknowledgement does not establish exact shared clip identities',
                       'Pretraining membership and overlaps with other source corpora remain unverified'],
        'recording_sources': [{k:r[k] for k in ('path','git_blob','annex_sha256','file_bytes','url','s3_version','header_sha256')}
                              for r in result['recordings']],
        'behaviour_source_hashes': [{k:b[k] for k in ('path','git_blob','source_sha256')} for b in result['behaviour']],
    }
    write_json(PUBLIC / 'qualification.json', out)
    plan = {
        'status': 'Reserved; not a model protocol or paper pivot',
        'created_utc': '2026-10-10', 'dataset_pin': PIN,
        'selection_inputs': 'Anonymous IDs, file/header integrity, documented trial-order availability, assigned stimulus categories and film families only',
        'participant_hash_namespace': 'EmoEEG-MC-2026-10-10-participants-v1',
        'material_hash_namespace': 'EmoEEG-MC-2026-10-10-materials-v1',
        'participants': {'confirmation_reserved': sorted(pool[:20]),
                         'development_validation': sorted(pool[20:30]),
                         'development_source': sorted(pool[30:]),
                         'quarantine': sorted(quarantine), 'no_raw_eeg': ['sub-01']},
        'materials': {'initial_number_selections': sorted(selected),
                      'family_closed_numbers': reserved_numbers,
                      'reserved_keys': sorted(reservations),
                      'development_keys': sorted(m['material_key'] for m in materials if m['material_key'] not in reservations),
                      'companion_rule': 'Reserve the same numeric spreadsheet entry in both contexts; these are different contents, not repeated video stimuli',
                      'mapping_gate': 'No ratings or waveform samples may be used until numeric spreadsheet entries are verified against raw trials/README symbols',
                      'unresolved_overlap_rule': 'Exclude FACED EEG/embeddings and uncertified emotion-pretrained checkpoints from any protected-material source arm'},
        'access_policy': {'all_confirmation_participant_outcomes': 'sealed',
                          'all_reserved_material_outcomes_for_all_people': 'sealed',
                          'signal_decoding': 'forbidden until joins and material mapping verified',
                          'development_rating_decoding': 'forbidden until joins and material mapping verified',
                          'currently_allowed': 'header metadata, assigned-stimulus descriptions, opaque source hashing and trial identity fields only'},
        'no_reallocation_after_outcome_access': True,
        'quarantine_release_rule': 'Retain outside existing role lists; require documented metadata resolution and a separate outcome-blind amendment',
        'manuscript_changes': False, 'new_research_question_approved': False,
        'qualification_sha256': sha((PUBLIC / 'qualification.json').read_bytes()),
        'source_sha256': { 'scripts/qualify_emo_mc.py': sha(Path(__file__).read_bytes()) },
    }
    destination = PUBLIC / 'reservation.json'
    if destination.exists() and json.loads(destination.read_text()) != plan:
        raise ValueError('Refusing to replace a frozen reservation')
    write_json(destination, plan)
    print(json.dumps({'participants': {k:len(v) for k,v in plan['participants'].items()},
                      'reserved_material_keys': len(reservations), 'fitting_ready': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'inspect', 'reserve'])
    args = parser.parse_args()
    {'collect': collect, 'inspect': inspect_collection, 'reserve': publish_metadata_and_reserve}[args.command]()
