"""Audit frozen FACED technical metadata without opening event/rating/sample data.

Build uses already cached technical files; verify is public-only unless --local.
No network requests, numerical EEG decoding or individual outcome access.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

REPO = Path(__file__).resolve().parents[1]
PRIVATE = REPO.parent / 'publication_runs/faced_backup_2026-10-10'
PUBLIC = REPO / 'results/development/faced_backup_2026-10-10'
sys.path.insert(0, str(REPO / 'scripts'))
from qualify_faced_backup import parse_header, pointer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, obj):
    if path.exists():
        raise ValueError('Refusing to overwrite qualification evidence: ' + path.name)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')


def stimulus_projection(rows):
    if len(rows) != 29 or len(rows[0]) != 6:
        raise ValueError('Unexpected published stimulus table')
    records = []
    for row in rows[1:]:
        index, duration, film, source, valence, emotion = row
        canonical = ' '.join(unicodedata.normalize('NFKC', film).casefold().split())
        records.append({'clip': int(index), 'duration_seconds': float(duration),
                        'assigned_valence': valence, 'assigned_emotion': emotion,
                        'source_film_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest()})
    if sorted(r['clip'] for r in records) != list(range(1, 29)):
        raise ValueError('Stimulus indices must be unique and exhaustive')
    return {'source': 'Published FACED supplement, first table; no individual ratings',
            'family_rule': 'SHA256 of NFKC, casefolded, whitespace-normalized exact source-film title; sequel/franchise overlap not adjudicated',
            'records': records}


def family_summary(records):
    by_film = defaultdict(list)
    by_class = defaultdict(set)
    for r in records:
        by_film[r['source_film_sha256']].append(r['clip'])
        by_class[r['assigned_emotion']].add(r['source_film_sha256'])
    return {'clips': len(records), 'exact_source_film_families': len(by_film),
            'repeated_film_clip_groups': sorted(sorted(v) for v in by_film.values() if len(v) > 1),
            'film_families_by_assigned_emotion': {k: len(v) for k, v in sorted(by_class.items())},
            'three_way_all_nine_class_film_disjoint_feasible': all(len(v) >= 3 for v in by_class.values()),
            'reason': 'Every class needs at least three distinct film families to occur in three film-disjoint roles. This necessary condition is violated; a split by clip would conceal shared films.'}


def local_analysis(stimuli):
    collection = read(PUBLIC / 'collection.json')
    pilots = read(PUBLIC / 'public_header_pilots.json')
    frequencies, units, types, counts, names, statuses = (Counter() for _ in range(6))
    total_bytes = 0
    sources = {}
    for r in collection['records']:
        folder = PRIVATE / 'metadata' / r['subject']
        for name, filename in [('pointer', 'pointer.bin'), ('sidecar', 'sidecar.json'), ('channels', 'channels.bin')]:
            if digest(folder / filename) != r[name + '_sha256']:
                raise ValueError('Technical metadata changed: ' + r['subject'] + '/' + name)
        sidecar = read(folder / 'sidecar.json')
        channel_rows = list(csv.DictReader((folder / 'channels.bin').read_text(encoding='utf-8-sig').splitlines(), delimiter='\t'))
        frequencies[str(float(sidecar['SamplingFrequency']))] += 1
        counts[str(len(channel_rows))] += 1
        names[tuple(c['name'] for c in channel_rows)] += 1
        units.update(c['units'] for c in channel_rows)
        types.update(c['type'] for c in channel_rows)
        statuses.update(c['status'] for c in channel_rows)
        size, checksum = pointer((folder / 'pointer.bin').read_bytes())
        if size != r['expected_object_bytes'] or checksum != r['annex_declared_object_sha256']:
            raise ValueError('Annex metadata changed')
        total_bytes += size
    pilot_summary = []
    expected_by_subject = {r['subject']: r for r in collection['records']}
    for r in pilots['records']:
        if not r['passed']:
            raise ValueError('Pilot failed')
        header_path = PRIVATE / 'pilot_headers' / (r['subject'] + '.bdf.header')
        if digest(header_path) != r['header_sha256']:
            raise ValueError('Private pilot header changed')
        h = parse_header(header_path.read_bytes(), expected_by_subject[r['subject']]['expected_object_bytes'])
        if h != r['header'] or not h['file_geometry_matches']:
            raise ValueError('Published header projection failed replay')
        sidecar = read(PRIVATE / 'metadata' / r['subject'] / 'sidecar.json')
        if any(c['sampling_frequency'] != sidecar['SamplingFrequency'] for c in h['channels']):
            raise ValueError('Header/sidecar rate disagreement')
        pilot_summary.append({'subject': r['subject'], 'signals': h['signals'],
                              'sampling_frequency': h['channels'][0]['sampling_frequency'],
                              'header_bytes': h['header_bytes'], 'geometry_matches': h['file_geometry_matches'],
                              'all_units_uV': all(c['unit'] == 'uV' for c in h['channels']),
                              'gain_uV_per_count': sorted(set(c['physical_units_per_count'] for c in h['channels']))})
    source_root = PRIVATE / 'source_metadata'
    source_files = ['faced_published_supplement.docx', 'faced_supplement_metadata_tables.json',
                    'nemar_availability.response', 'nemar_corrections.response',
                    'nemar_conversion.response', 'nemar_events_schema.response',
                    'synapse_permission_checks.json', 'nemar_tree.response', 'nemar_readme.response',
                    'faced_nature.response']
    for filename in source_files:
        sources[filename] = digest(source_root / filename)
    published_rows = read(source_root / 'faced_supplement_metadata_tables.json')[0]['rows']
    if stimulus_projection(published_rows) != stimuli:
        raise ValueError('Published film metadata projection changed')
    availability = read(source_root / 'nemar_availability.response')['completeness']
    permissions = read(source_root / 'synapse_permission_checks.json')
    corrections = [json.loads(line) for line in (source_root / 'nemar_corrections.response').read_text().splitlines()]
    scrub = [c for c in corrections if c['action'] == 'headers-scrubbed'][0]
    manuscript = REPO.parent / 'report.tex'
    return {'source_revision': collection['source_revision'], 'participants_technical_metadata_checked': len(collection['records']),
            'sampling_frequency_participants': dict(sorted(frequencies.items())),
            'channel_count_participants': dict(counts), 'channel_type_entries': dict(types),
            'channel_unit_entries': dict(units), 'channel_status_entries': dict(statuses),
            'unique_channel_orders': len(names), 'channel_order': list(names.most_common(1)[0][0]),
            'reference_sidecar': 'n/a; do not infer rereferencing from a sidecar placeholder',
            'initial_strict_no_redirect_header_failures': sum(not r['passed'] for r in collection['records']),
            'separately_declared_public_header_pilots': pilot_summary,
            'annex_pointer_declared_total_bytes': total_bytes,
            'curator_availability': availability,
            'availability_bytes_consistent_with_pointer_inventory': availability['bytes_declared'] == total_bytes and availability['bytes_present'] == total_bytes,
            'privacy_correction': {'date_utc': scrub['at'], 'counts': scrub['counts'],
                                   'curator_verification_claim': scrub['verification'], 'payload_identity_independently_verified': False},
            'original_codebook_permissions': [{'entity_id': r['entity_id'], 'can_download_anonymously': r['permissions']['canDownload'],
                                               'certification_required': r['permissions']['isCertificationRequired']} for r in permissions['records']],
            'materials': family_summary(stimuli['records']), 'source_sha256': sources,
            'manuscript_sha256': digest(manuscript), 'rating_values_decoded': 0, 'waveform_samples_decoded': 0,
            'models_fitted': 0, 'signed_urls_retained': 0, 'full_recording_hashes_reverified': False,
            'material_roles_assigned': False, 'research_question_changed': False,
            'outcome_access_cleared': False, 'status': 'bounded_metadata_phase_complete; source_codebooks_and_trial_joins_still_gated'}


def verify(local=False):
    q = read(PUBLIC / 'qualification.json')
    for filename, expected in q['public_bindings'].items():
        if digest(PUBLIC / filename) != expected:
            raise ValueError('Public binding changed: ' + filename)
    if digest(Path(__file__)) != q['analyzer_sha256']:
        raise ValueError('Analyzer changed')
    plan = read(PUBLIC / 'plan.json')
    if digest(REPO / 'scripts/qualify_faced_backup.py') != plan['script_sha256']:
        raise ValueError('Frozen initial collector changed')
    pilot_plan = read(PUBLIC / 'public_header_pilot_plan.json')
    # The pilot collector also checks its source bindings before collection.
    for key, value in pilot_plan.items():
        if key == 'script_sha256' and digest(REPO / 'scripts/check_faced_public_headers.py') != value:
            raise ValueError('Frozen pilot collector changed')
    reserve = read(PUBLIC / 'participant_reservation.json')
    groups = [reserve[k] for k in ('development_source', 'development_validation', 'confirmation')]
    if [len(g) for g in groups] != [70, 20, 33] or len(set(sum(groups, []))) != 123:
        raise ValueError('Participant roles overlap or changed')
    stimuli = read(PUBLIC / 'stimulus_metadata.json')
    if family_summary(stimuli['records']) != q['materials']:
        raise ValueError('Material family summary failed public replay')
    if local:
        replay = local_analysis(stimuli)
        if any(q[k] != v for k, v in replay.items()):
            raise ValueError('Local technical evidence failed replay')
    return {'passed': True, 'local_source_replay': local, 'participants': 123, 'pilot_headers': 3,
            'stimulus_clips': 28, 'source_film_families': 24, 'outcome_access_cleared': False}


def build():
    stimuli = stimulus_projection(read(PRIVATE / 'source_metadata/faced_supplement_metadata_tables.json')[0]['rows'])
    q = local_analysis(stimuli)
    write_new(PUBLIC / 'stimulus_metadata.json', stimuli)
    filenames = ['plan.json', 'participant_reservation.json', 'collection.json',
                 'public_header_pilot_plan.json', 'public_header_pilots.json', 'stimulus_metadata.json']
    q['public_bindings'] = {name: digest(PUBLIC / name) for name in filenames}
    q['analyzer_sha256'] = digest(Path(__file__))
    write_new(PUBLIC / 'qualification.json', q)
    proof = verify(local=True)
    proof['qualification_sha256'] = digest(PUBLIC / 'qualification.json')
    write_new(PUBLIC / 'verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'verify'])
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build() if args.command == 'build' else verify(args.local)))
