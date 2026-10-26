"""Preserve first FACED audit and check the newly discovered second channel layout."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import requests
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import qualify_faced_backup as base
from check_faced_public_headers import get_range
from analyze_faced_backup import digest, read, write_new
PUBLIC, PRIVATE = base.PUBLIC, base.PRIVATE


def metadata_groups():
    groups = {}
    collection = read(PUBLIC / 'collection.json')
    for record in collection['records']:
        folder = PRIVATE / 'metadata' / record['subject']
        if digest(folder / 'channels.bin') != record['channels_sha256'] or digest(folder / 'sidecar.json') != record['sidecar_sha256']:
            raise ValueError('Metadata changed')
        rows = list(csv.DictReader((folder / 'channels.bin').read_text(encoding='utf-8-sig').splitlines(), delimiter='\t'))
        names = tuple(r['name'] for r in rows)
        group = groups.setdefault(names, {'subjects': [], 'sampling_frequencies': Counter()})
        group['subjects'].append(record['subject'])
        group['sampling_frequencies'][str(float(read(folder / 'sidecar.json')['SamplingFrequency']))] += 1
    return [{'channel_order': list(names), 'participants': len(group['subjects']),
             'subject_ids': group['subjects'], 'sampling_frequency_participants': dict(group['sampling_frequencies'])}
            for names, group in groups.items()]


def declare():
    groups = metadata_groups()
    if len(groups) != 2 or groups[1]['subject_ids'][0] != 'sub-061':
        raise ValueError('Second layout changed')
    plan = {'date': '2026-10-10', 'pilot': 'sub-061',
            'selection': 'First identity in the second exact channel-order group, selected from technical metadata only',
            'reason': 'Initial three public pilots all use the first of two channel layouts; no representative of the second was read',
            'source_revision': base.PIN, 'header_requests': 'Same exact bounded public NEMAR broker range policy; one HTTPS hop, ephemeral links only',
            'signed_urls_retained': 0, 'signal_samples_allowed': False, 'rating_values_allowed': False,
            'script_sha256': digest(Path(__file__)),
            'base_script_sha256': digest(REPO / 'scripts/qualify_faced_backup.py'),
            'redirect_script_sha256': digest(REPO / 'scripts/check_faced_public_headers.py'),
            'initial_qualification_sha256': digest(PUBLIC / 'qualification.json'),
            'participant_reservation_sha256': digest(PUBLIC / 'participant_reservation.json'),
            'material_roles_assigned': False, 'no_research_question_change': True}
    write_new(PUBLIC / 'channel_supplement_plan.json', plan)
    print(json.dumps({'declared': True, 'second_layout_pilot': 'sub-061', 'channel_group_counts': [g['participants'] for g in groups]}))


def check_bindings():
    plan = read(PUBLIC / 'channel_supplement_plan.json')
    bindings = [('script_sha256', Path(__file__)), ('base_script_sha256', REPO / 'scripts/qualify_faced_backup.py'),
                ('redirect_script_sha256', REPO / 'scripts/check_faced_public_headers.py'),
                ('initial_qualification_sha256', PUBLIC / 'qualification.json'),
                ('participant_reservation_sha256', PUBLIC / 'participant_reservation.json')]
    if any(plan[k] != digest(p) for k, p in bindings):
        raise ValueError('Supplement source binding changed')
    return plan


def collect():
    plan = check_bindings()
    if (PUBLIC / 'channel_supplement.json').exists():
        raise ValueError('Do not overwrite supplemental pilot')
    subject = plan['pilot']
    r = next(r for r in read(PUBLIC / 'collection.json')['records'] if r['subject'] == subject)
    record = {'subject': subject, 'signal_samples_decoded': 0, 'rating_values_decoded': 0, 'signed_urls_retained': 0}
    try:
        path = f'{subject}/eeg/{subject}_task-watchingVideoClips_eeg.bdf'
        total = r['expected_object_bytes']
        first = get_range(path, 0, 255, total)
        size, signals = int(first[184:192]), int(first[252:256])
        if not 1 <= signals <= 128 or size != 256 + 256 * signals:
            raise ValueError('Invalid bounded header geometry')
        header = first + get_range(path, 256, size - 1, total)
        target = PRIVATE / 'pilot_headers' / (subject + '.bdf.header')
        target.write_bytes(header)
        parsed = base.parse_header(header, total)
        rows = list(csv.DictReader((PRIVATE / 'metadata' / subject / 'channels.bin').read_text(encoding='utf-8-sig').splitlines(), delimiter='\t'))
        matches = [c['label'] for c in parsed['channels']] == [r['name'] for r in rows]
        record.update({'passed': True, 'header_sha256': base.sha(header), 'header': parsed,
                       'ordered_channel_labels_match_sidecar': matches})
    except (requests.RequestException, ValueError, UnicodeError) as error:
        record.update({'passed': False, 'failure_type': type(error).__name__,
                       'failure_message': str(error)[:120] if isinstance(error, ValueError) else 'public_header_request_failed'})
    result = {'record': record, 'channel_groups': metadata_groups(),
              'plan_sha256': digest(PUBLIC / 'channel_supplement_plan.json'),
              'initial_evidence_preserved': True, 'full_recording_hashes_reverified': False,
              'outcome_access_cleared': False, 'material_roles_assigned': False}
    write_new(PUBLIC / 'channel_supplement.json', result)
    print(json.dumps({'second_layout_pilot_passed': record['passed'], 'outcome_access_cleared': False}))


def verify(local):
    plan = check_bindings()
    result = read(PUBLIC / 'channel_supplement.json')
    if result['plan_sha256'] != digest(PUBLIC / 'channel_supplement_plan.json'):
        raise ValueError('Supplement plan changed')
    record = result['record']
    if record['subject'] != plan['pilot']:
        raise ValueError('Supplement identity changed')
    if sum(g['participants'] for g in result['channel_groups']) != 123:
        raise ValueError('Incomplete channel groups')
    if local:
        if metadata_groups() != result['channel_groups']:
            raise ValueError('Channel grouping failed replay')
        if record['passed']:
            path = PRIVATE / 'pilot_headers' / (record['subject'] + '.bdf.header')
            total = next(r['expected_object_bytes'] for r in read(PUBLIC / 'collection.json')['records'] if r['subject'] == record['subject'])
            if digest(path) != record['header_sha256'] or base.parse_header(path.read_bytes(), total) != record['header']:
                raise ValueError('Supplemental header failed replay')
    print(json.dumps({'passed': True, 'local_source_replay': local, 'pilot_passed': record['passed'], 'outcome_access_cleared': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['declare', 'collect', 'verify'])
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    if args.command == 'declare': declare()
    elif args.command == 'collect': collect()
    else: verify(args.local)
