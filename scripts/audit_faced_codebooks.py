"""Audit user-supplied FACED documentation, preserving earlier evidence/outcomes.

Uses standard-library ZIP/XML extraction in the user's research environment.
Does not edit/render spreadsheets, evaluate formulas, fetch data or open ratings.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shutil
import unicodedata
from xml.etree import ElementTree as ET
import zipfile

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
PRIVATE = WORKSPACE / 'publication_runs/faced_backup_2026-10-10/source_metadata'
PUBLIC = REPO / 'results/development/faced_codebooks_2026-10-10'
BACKUP = REPO / 'results/development/faced_backup_2026-10-10'
FILES = {'Stimuli_info.xlsx': 'syn52370955', 'Task_event.xlsx': 'syn52370956',
         'DataStructureOfBehaviouralData.xlsx': 'syn52370951'}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    if path.exists():
        raise ValueError('Do not overwrite frozen evidence: ' + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def film_digest(title):
    return hashlib.sha256(normalize(title).encode('utf-8')).hexdigest()


def read_cells(path):
    if path.name not in FILES or path.stat().st_size > 100000:
        raise ValueError('Outside the three small documentation files')
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 2000000:
            raise ValueError('Expanded documentation exceeds budget')
        if any('externalLinks/' in name or name.endswith('vbaProject.bin') for name in archive.namelist()):
            raise ValueError('External workbook links or macros are outside scope')
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheets = workbook.findall('s:sheets/s:sheet', NS)
        if len(sheets) != 1 or sheets[0].attrib.get('name') != 'Sheet1' or sheets[0].attrib.get('state', 'visible') != 'visible':
            raise ValueError('Unexpected documentation sheet inventory')
        relations = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {}
        for link in relations:
            if link.attrib.get('TargetMode') == 'External':
                raise ValueError('External relationship outside scope')
            target = link.attrib['Target']
            targets[link.attrib['Id']] = posixpath.normpath('xl/' + target)
        target = targets[sheets[0].attrib[REL]]
        if not target.startswith('xl/worksheets/'):
            raise ValueError('Invalid sheet relationship')
        root = ET.fromstring(archive.read(target))
        raw_cells = root.findall('s:sheetData/s:row/s:c', NS)
        if len(raw_cells) > 600:
            raise ValueError('Documentation cell budget exceeded')
        if any(c.find('s:f', NS) is not None for c in raw_cells):
            raise ValueError('Formula evaluation is outside source extraction')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in item.findall('.//s:t', NS))
                      for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        cells = {}
        for cell in raw_cells:
            value = cell.find('s:v', NS)
            if cell.attrib.get('t') == 's':
                text = shared[int(value.text)] if value is not None else ''
            elif cell.attrib.get('t') == 'inlineStr':
                text = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
            else:
                text = value.text if value is not None else ''
            if text:
                cells[cell.attrib['r']] = text
        return cells


def decode_stimuli(cells):
    if cells.get('A1') != 'Video index' or cells.get('C1') != 'Source Film':
        raise ValueError('Unexpected stimulus codebook')
    records = []
    for row in range(2, 30):
        emotion, valence = cells[f'F{row}'], cells[f'E{row}']
        category = 'Neutral' if valence == 'Neutral' and emotion in ('/', '\\') else emotion
        records.append({'clip': int(cells[f'A{row}']), 'catalogue_duration_seconds': float(cells[f'B{row}']),
                        'source_film_sha256': film_digest(cells[f'C{row}']), 'assigned_valence': valence,
                        'literal_assigned_emotion': emotion, 'assigned_category': category})
    if [r['clip'] for r in records] != list(range(1, 29)):
        raise ValueError('Incomplete or duplicate clip catalogue')
    # Explicit source exception; do not generalize from guessed duration distributions.
    note = cells.get('A35', '')
    if not all(token in note for token in ('video 22', 'sub036-sub060', '83-s', 'beginning')):
        raise ValueError('Expected clip-version exception not documented')
    exception = {'clip': 22, 'subjects': [f'sub-{i:03d}' for i in range(36, 61)],
                 'documented_duration_seconds': 83.0, 'catalogue_duration_seconds': 76.0,
                 'change': 'Additional seconds at beginning', 'source_cell': 'Sheet1!A35',
                 'independent_media_hashes_checked': False}
    return records, exception


def decode_events(cells, stimuli):
    semantic_cells = {'A3': 'Experiment start', 'B3': '100', 'A5': 'Video clip start',
                      'B5': '101', 'A6': 'Video clip end', 'B6': '102'}
    if any(cells.get(k, '').strip() != v for k, v in semantic_cells.items()):
        raise ValueError('Trigger semantics changed')
    by_clip = {r['clip']: r for r in stimuli}
    for row in range(13, 41):
        clip = int(cells[f'A{row}'])
        if cells[f'B{row}'] != by_clip[clip]['assigned_category'] or cells[f'C{row}'] != by_clip[clip]['assigned_valence']:
            raise ValueError('Event/stimulus assignment contradiction')
    note = cells.get('A8', '')
    if not all(token in note for token in ('sub29', 'sub45', 'sub49', 'sub59', 'sub60')) or '101' not in cells.get('A9', '') or '102' not in cells.get('A9', ''):
        raise ValueError('Unrelated-task exception changed')
    return {'trigger_roles': {'100': 'Experiment start', '1..28': 'Video identity',
                              '101': 'Video start', '102': 'Video end'},
            'catalogue_category_crosschecks': 28, 'experiment_start_scope': 'Experiment, not assumed block start',
            'unrelated_task_subjects': [f'sub-{i:03d}' for i in (29, 45, 49, 59, 60)],
            'unrelated_task_exception': '101 can occur without a following 102; never count every 101 as an emotion trial',
            'exception_cells': ['Sheet1!A8', 'Sheet1!A9'],
            'raw_trigger_sequences_opened': False}


def decode_behaviour(cells):
    fields = [cells.get(f'A{row}') for row in (2, 4, 5, 6, 7)]
    if fields != ['score', 'trial', 'vid', 'Accuracy', 'ResponseTime'] or cells.get('C2') != '0-7':
        raise ValueError('Unexpected behavioural field schema')
    order = re.findall(r'"([^\"]+)"', cells.get('C3', ''))
    if len(order) != 12 or len(set(order)) != 12 or order[9] != 'valence':
        raise ValueError('Rating item order changed')
    if 'Presentation orders' not in cells.get('B4', '') or 'Video indexes' not in cells.get('B5', ''):
        raise ValueError('Presentation order and clip identity not distinguished')
    note = cells.get('B13', '')
    if not all(token in note for token in ('0 indicated', '7 indicated', 'very negative', 'very positive', 'not at all', 'very much')):
        raise ValueError('Rating anchors unavailable')
    return {'fields': fields, 'item_order': order, 'scale': [0.0, 7.0],
            'valence_item_one_based': 10, 'arousal_item_one_based': 9,
            'valence_anchors': {'0': 'Very negative', '7': 'Very positive'},
            'other_item_anchors': {'0': 'Not at all', '7': 'Very much'},
            'trial_semantics': 'Presentation ordinal, 1..28', 'vid_semantics': 'Catalogue clip identity, 1..28',
            'arithmetic_fields': ['Accuracy', 'ResponseTime'],
            'item_order_cell': 'Sheet1!C3', 'anchor_cell': 'Sheet1!B13',
            'classification_threshold_selected': False, 'participant_values_opened': False}


def validate_identity_projection(rows):
    """Validate a future identity-only projection; never accept a score payload."""
    if len(rows) != 28 or any(set(row) != {'trial', 'vid'} for row in rows):
        raise ValueError('Only the complete trial/vid identity projection is allowed')
    expected = set(range(1, 29))
    if any(type(row[k]) is not int for row in rows for k in ('trial', 'vid')):
        raise ValueError('Identity fields must be exact integers')
    if {row['trial'] for row in rows} != expected or {row['vid'] for row in rows} != expected:
        raise ValueError('Duplicate, missing or invalid trial/video identities')
    # Presentation sequence can differ from catalogue order; never overwrite vid with row position.
    return {row['vid']: row['trial'] for row in rows}


def declare():
    source_records = []
    for name, entity in FILES.items():
        source = WORKSPACE / name
        target = PRIVATE / 'downloaded_codebooks' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(source):
            raise ValueError('Private source snapshot already differs')
        if not target.exists(): shutil.copyfile(source, target)
        read_cells(target)
        source_records.append({'name': name, 'expected_synapse_entity': entity, 'bytes': source.stat().st_size,
                               'sha256': sha(source), 'md5': hashlib.md5(source.read_bytes()).hexdigest(),
                               'provenance': 'User supplied local download; no anonymous release checksum available'})
    bindings = {'backup_qualification': sha(BACKUP / 'qualification.json'),
                'backup_reservation': sha(BACKUP / 'participant_reservation.json'),
                'channel_supplement': sha(BACKUP / 'channel_supplement.json'),
                'emo_reservation': sha(REPO / 'results/development/emo_mc_qualification_2026-10-10/reservation.json'),
                'manuscript': sha(WORKSPACE / 'report.tex')}
    plan = {'date': '2026-10-10', 'scope': 'Original documentation only, no participant matrices, event tables or samples',
            'initial_documentation_inspection_already_performed': True, 'source_files': source_records,
            'analyzer_sha256': sha(Path(__file__)), 'prior_bindings': bindings,
            'public_metadata_receipt_sha256': sha(PRIVATE / 'downloaded_codebook_release_check.json'),
            'curator_conversion_sha256': sha(PRIVATE / 'nemar_conversion.response'),
            'prior_stimulus_projection_sha256': sha(BACKUP / 'stimulus_metadata.json'),
            'models_fitted': 0, 'research_question_changed': False, 'outcome_access_cleared': False}
    save(PUBLIC / 'plan.json', plan)
    print(json.dumps({'declared': True, 'documentation_files': 3, 'outcome_access_cleared': False}))


def check_plan(local):
    plan = load(PUBLIC / 'plan.json')
    if sha(Path(__file__)) != plan['analyzer_sha256']:
        raise ValueError('Frozen codebook analyzer changed')
    paths = {'backup_qualification': BACKUP / 'qualification.json', 'backup_reservation': BACKUP / 'participant_reservation.json',
             'channel_supplement': BACKUP / 'channel_supplement.json',
             'emo_reservation': REPO / 'results/development/emo_mc_qualification_2026-10-10/reservation.json'}
    if any(sha(path) != plan['prior_bindings'][key] for key, path in paths.items()):
        raise ValueError('Protected prior evidence changed')
    if local:
        if sha(WORKSPACE / 'report.tex') != plan['prior_bindings']['manuscript']:
            raise ValueError('Manuscript changed')
        for r in plan['source_files']:
            if sha(PRIVATE / 'downloaded_codebooks' / r['name']) != r['sha256']:
                raise ValueError('Frozen documentation bytes changed')
        if sha(PRIVATE / 'nemar_conversion.response') != plan['curator_conversion_sha256'] or sha(PRIVATE / 'downloaded_codebook_release_check.json') != plan['public_metadata_receipt_sha256']:
            raise ValueError('Frozen source reference changed')
    return plan


def analysis():
    plan = check_plan(local=True)
    stimuli, exception = decode_stimuli(read_cells(PRIVATE / 'downloaded_codebooks/Stimuli_info.xlsx'))
    events = decode_events(read_cells(PRIVATE / 'downloaded_codebooks/Task_event.xlsx'), stimuli)
    behaviour = decode_behaviour(read_cells(PRIVATE / 'downloaded_codebooks/DataStructureOfBehaviouralData.xlsx'))
    tree = ast.parse((PRIVATE / 'nemar_conversion.response').read_text(encoding='utf-8'))
    rating_lists = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RATING_KEYS' for t in node.targets)]
    if len(rating_lists) != 1:
        raise ValueError('Curator rating list ambiguous')
    behaviour['curator_item_order_matches'] = [k.casefold() for k in rating_lists[0]] == behaviour['item_order']
    if not behaviour['curator_item_order_matches']:
        raise ValueError('Curator rating order contradicts documentation')
    if sha(BACKUP / 'stimulus_metadata.json') != plan['prior_stimulus_projection_sha256']:
        raise ValueError('Frozen published stimulus projection changed')
    old = {r['clip']: r for r in load(BACKUP / 'stimulus_metadata.json')['records']}
    discrepancies = []
    for r in stimuli:
        p = old[r['clip']]
        if (r['catalogue_duration_seconds'], r['source_film_sha256'], r['assigned_valence']) != (p['duration_seconds'], p['source_film_sha256'], p['assigned_valence']):
            raise ValueError('Original catalogue contradicts published metadata')
        if r['literal_assigned_emotion'] != p['assigned_emotion']:
            if r['assigned_category'] != 'Neutral' or p['assigned_emotion'] != '/':
                raise ValueError('Unexpected assigned-emotion discrepancy')
            discrepancies.append({'clip': r['clip'], 'original_literal': r['literal_assigned_emotion'], 'published_literal': p['assigned_emotion'], 'category_from_task_codebook': 'Neutral'})
    receipt = load(PRIVATE / 'downloaded_codebook_release_check.json')
    return {'source_files': plan['source_files'], 'source_metadata_receipt': receipt,
            'independent_release_payload_checksums_authenticated': False,
            'stimuli': stimuli, 'published_catalogue_checks': 28, 'neutral_literal_differences': discrepancies,
            'clip_version_exception': exception, 'event_codebook': events, 'behaviour_codebook': behaviour,
            'minimum_exact_source_film_families': len({r['source_film_sha256'] for r in stimuli}),
            'gates_resolved': ['Original catalogue matches published clip identities/durations/film families',
                               'Trigger-role meanings documented; exceptions identified',
                               'Rating item order, scale, anchors and trial/vid distinction documented'],
            'gates_remaining': ['Actual recording/event/behaviour identity joins and endpoint audit',
                                'Full recording digest and original-to-curator waveform provenance',
                                'Remaining header coverage and montage/unit processing adapter',
                                'Content-version and pretrained overlap audit',
                                'Feasible target/material design and approved research question'],
            'real_participant_identity_rows_opened': 0, 'real_raw_triggers_opened': 0,
            'individual_ratings_opened': 0, 'eeg_samples_opened': 0, 'models_fitted': 0,
            'material_roles_assigned': False, 'research_question_changed': False, 'outcome_access_cleared': False}


def build():
    result = analysis()
    result['plan_sha256'] = sha(PUBLIC / 'plan.json')
    save(PUBLIC / 'audit.json', result)
    proof = verify(local=True)
    proof['audit_sha256'] = sha(PUBLIC / 'audit.json')
    save(PUBLIC / 'verification.json', proof)
    return proof


def verify(local=False):
    check_plan(local)
    audit = load(PUBLIC / 'audit.json')
    if audit['plan_sha256'] != sha(PUBLIC / 'plan.json'):
        raise ValueError('Public plan binding changed')
    if len(audit['stimuli']) != 28 or audit['behaviour_codebook']['valence_item_one_based'] != 10:
        raise ValueError('Invalid public semantics projection')
    if local:
        replay = analysis()
        if any(audit[k] != value for k, value in replay.items()):
            raise ValueError('Local documentation audit replay failed')
    return {'passed': True, 'local_source_replay': local, 'documentation_files': 3,
            'catalogue_crosschecks': 28, 'rating_items': 12, 'documented_exception_subjects': 5,
            'longer_clip_subjects': 25, 'original_outcomes_opened': False,
            'outcome_access_cleared': False, 'research_question_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare', 'build', 'verify'])
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    if args.command == 'declare': declare()
    else: print(json.dumps(build() if args.command == 'build' else verify(args.local)))
