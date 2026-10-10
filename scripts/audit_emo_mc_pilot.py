"""Authenticate one pre-reserved development person's raw object; read triggers only."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import requests
from qualify_emo_mc import PRIVATE, PUBLIC, REPO, parse_edf_header, sha, write_json

RESERVATION_COMMIT = '341696fe8'


def tal_events(payload):
    """Read EDF+ annotation text only, never ordinary signed EEG samples."""
    events = []
    for tal in payload.split(b'\x00'):
        if not tal.strip():
            continue
        parts = tal.split(b'\x14')
        timing = parts[0].split(b'\x15')
        onset = float(timing[0])
        duration = float(timing[1]) if len(timing) == 2 else 0.0
        for value in parts[1:]:
            if value:
                events.append(dict(onset_s=onset, duration_s=duration, description=value.decode('utf-8')))
    return events


def audit(download=False):
    plan_path = PUBLIC / 'reservation.json'
    committed = subprocess.check_output(['git', 'show', RESERVATION_COMMIT + ':results/development/emo_mc_qualification_2026-10-10/reservation.json'], cwd=REPO)
    if sha(plan_path.read_bytes()) != sha(committed):
        raise ValueError('Reservation changed after source qualification commit')
    plan = json.loads(committed)
    qualification = json.loads((PUBLIC / 'qualification.json').read_text())
    by_person = {}
    for source in qualification['recording_sources']:
        person = source['path'].split('/')[0]
        if person in plan['participants']['development_source']:
            by_person.setdefault(person, []).append(source)
    person = min(by_person, key=lambda p: (sum(s['file_bytes'] for s in by_person[p]), p))
    sources = by_person[person]
    if sum(s['file_bytes'] for s in sources) > 1_000_000_000:
        raise ValueError('Pilot exceeds one-gigabyte budget')
    records = []
    all_events = []
    for source in sources:
        target = PRIVATE / 'objects' / source['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if not download:
                raise ValueError('Missing pilot object; explicitly use --download')
            partial = target.with_suffix('.edf.part')
            digest = hashlib.sha256()
            count = 0
            last = time.monotonic()
            with requests.get(source['url'], params={'versionId': source['s3_version']},
                              timeout=(10, 30), stream=True) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError('Unexpected complete-object status')
                with partial.open('wb') as output:
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        if count > source['file_bytes']:
                            raise ValueError('Source exceeds pinned annex size')
                        digest.update(chunk)
                        output.write(chunk)
                        if time.monotonic() - last > 15:
                            print(json.dumps({'downloaded_bytes':count,'total_bytes':source['file_bytes']}),flush=True)
                            last = time.monotonic()
            if count != source['file_bytes'] or digest.hexdigest() != source['annex_sha256']:
                raise ValueError('Complete-object annex checksum mismatch')
            partial.rename(target)
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if target.stat().st_size != source['file_bytes'] or digest != source['annex_sha256']:
            raise ValueError('Local source binding failed')
        header_path = PRIVATE / 'headers' / source['path']
        header = header_path.with_suffix('.header').read_bytes()
        parsed = parse_edf_header(header, source['file_bytes'])
        if not parsed['geometry_consistent']:
            raise ValueError('Quarantined EDF geometry')
        sample_offset = 0
        annotations = []
        for channel in parsed['channels']:
            if channel['label'] == 'EDF Annotations':
                annotations.append((sample_offset * 2, channel['samples'] * 2))
            sample_offset += channel['samples']
        events = []
        with target.open('rb') as stream:
            for record in range(parsed['records']):
                for channel_offset, length in annotations:
                    stream.seek(parsed['header_bytes'] + record * parsed['record_bytes'] + channel_offset)
                    payload = stream.read(length)
                    if len(payload) != length:
                        raise ValueError('Truncated annotation channel')
                    events.extend(tal_events(payload))
        write_json(PRIVATE / 'pilot_events.json', events)
        all_events.extend(events)
        descriptions = Counter(event['description'] for event in events)
        records.append(dict(path=source['path'], participant=person, sha256=digest,
                            bytes=target.stat().st_size, full_annex_digest_matches=True,
                            annotations=len(events), annotation_description_counts=dict(descriptions)))
    proof = dict(reservation_commit=RESERVATION_COMMIT, reservation_sha256=sha(committed),
                 pilot_selection='Smallest total raw-object bytes among already assigned development-source people; ties by anonymous ID',
                 participant=person, objects=records, waveform_samples_decoded=0,
                 rating_values_decoded=0, models_fitted=0, research_question_changed=False,
                 audit_script_sha256=sha(Path(__file__).read_bytes()))
    write_json(PUBLIC / 'pilot_authentication.json', proof)
    print(json.dumps(proof, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    audit(args.download)
