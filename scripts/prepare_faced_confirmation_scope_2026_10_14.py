"""Declare inactive FACED byte/range scope from previously published metadata.

Never inspect, stat, open or decode any original/private file. The old contract
supplies identities only; its historical three-model activation is not reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_confirmation_contract import validate_contract, canonical_sha256
from gruxnet.faced_proposal_preprocessing import common_indices, COMMON_CHANNELS
from gruxnet.faced_temporal_coverage import segment_bounds, REGIONS

PUBLIC = REPO / 'results/development/faced_confirmation_backend_2026-10-14'
INPUTS = {
    'contract': 'results/development/faced_confirmation_freeze_2026-10-12/contract.json',
    'roles': 'results/development/faced_backup_2026-10-10/participant_reservation.json',
    'headers': 'results/development/faced_raw_2026-10-11/headers.json',
    'joins': 'results/development/faced_raw_2026-10-11/joins_numeric_projection.json',
    'extraction': 'results/development/faced_raw_2026-10-11/extraction.json',
    'catalogue': 'results/development/faced_codebooks_2026-10-10/audit.json',
    'proposal': 'results/development/faced_coverage_missing_confirmation_proposal_2026-10-13/proposal.json',
}
MODELS = ('category_median', 'context_only', 'late_joint', 'distributed_joint')


def _integer(value, name):
    if type(value) is not int:
        raise ValueError('Require exact integer ' + name)
    return value


def _unique(records, key, label):
    result = {}
    for row in records:
        identity = row[key]
        if identity in result:
            raise PermissionError('Duplicate ' + label)
        result[identity] = row
    return result


def build_from_metadata(values):
    contract, roles = values['contract'], values['roles']
    validate_contract(contract)
    for old, new in (('source', 'development_source'), ('validation', 'development_validation'),
                     ('confirmation', 'confirmation')):
        if sorted(roles[new]) != contract['participant_roles'][old]:
            raise PermissionError('Frozen role identity mismatch')
    people = tuple(contract['participant_roles']['confirmation'])
    if any(not isinstance(p, str) or not re.fullmatch(r'sub-\d{3}', p) for p in people):
        raise PermissionError('Invalid recording participant identity')
    proposal = values['proposal']
    if (proposal.get('protected_signal_access_authorized') is not False
            or proposal.get('protected_rating_access_authorized') is not False
            or proposal.get('research_question_changed') is not False):
        raise PermissionError('Existing proposal must remain inactive')
    clips = contract['material_scope']
    if (proposal['roles_and_materials']['primary_clips'] != clips['confirmation_clips']
            or proposal['roles_and_materials']['familiar_secondary_clips'] != clips['development_clips']):
        raise PermissionError('New proposal and frozen material identities disagree')
    if values['headers'].get('waveform_samples_decoded') != 0:
        raise PermissionError('Require original outcome-blind technical header projection')
    if (values['joins'].get('rating_values_decoded') != 0
            or values['joins'].get('waveform_samples_decoded') != 0
            or values['joins'].get('models_fitted') != 0):
        raise PermissionError('Require prior identity/timing-only join projection')
    if values['extraction'].get('all_members_match') is not True:
        raise PermissionError('Require previously qualified extraction inventory')
    headers = _unique(values['headers']['records'], 'subject', 'header subject')
    joins = _unique(values['joins']['records'], 'subject', 'join subject')
    inventory = _unique(values['extraction']['records'], 'member', 'extraction member')
    catalogue = _unique(values['catalogue']['stimuli'], 'clip', 'catalogue clip')
    people_all = set(sum((contract['participant_roles'][r] for r in ('source','validation','confirmation')), []))
    if set(headers) != people_all or set(joins) != people_all:
        raise PermissionError('Prior technical projections require all123 exact participants')
    # Validate every arm's identity/family/category before interpreting ranges.
    cat_old = {r['vid']:r for r in contract['catalogue']}
    for vid in cat_old:
        c, old = catalogue[vid], cat_old[vid]
        if (c['source_film_sha256'] != old['family'] or c['assigned_category'] != old['assigned_category']
                or c['catalogue_duration_seconds'] != old['catalogue_duration_seconds']):
            raise PermissionError('Frozen catalogue metadata changed')
    files, rows, rates = [], [], Counter()
    for person in people:
        header = headers[person]['data.bdf']; join = joins[person]
        if not (join.get('passed') is True and join.get('all_28_videos_once') is True
                and join.get('presentation_order_matches') is True
                and join.get('all_endpoints_in_data_geometry') is True):
            raise PermissionError('Original participant joins are not qualified')
        spans = _unique(join['spans'], 'vid', 'trial span identity')
        identities = _unique(join['identity_rows'], 'vid', 'behavior identity')
        if set(spans) != set(range(1,29)) or set(identities) != set(spans):
            raise PermissionError('Original28trial identity projection changed')
        for vid in spans:
            if (_integer(spans[vid]['trial'], 'trial ordinal') != identities[vid]['trial']
                    or not 1 <= spans[vid]['trial'] <= 28):
                raise PermissionError('Marker and behavioral ordinal mismatch')
        size = _integer(header['header_bytes'], 'header bytes')
        count = _integer(header['record_count'], 'record count')
        if size != 8448 or header['sample_width'] != 3 or header['record_duration_seconds'] != 1:
            raise ValueError('Qualified original BDF geometry changed')
        literals = header['columns']['samples_per_record']
        if any(isinstance(x,bool) or not (type(x) is int or isinstance(x,str) and x.isascii() and x.isdecimal()) for x in literals):
            raise ValueError('Require literal original record sample counts')
        samples = [int(x) for x in literals]
        if len(samples) != 32 or len(set(samples)) != 1 or samples[0] not in (250,1000):
            raise ValueError('Qualified32channel rate changed')
        rate = samples[0]
        if set(header['sample_rates_hz']) != {float(rate)} or count <= 0:
            raise ValueError('Qualified rate/record count changed')
        indices = common_indices(header['columns']['labels'])
        rates[rate] += 1
        acquisition = ('legacy_' if int(person[-3:]) < 61 else 'modern_') + str(rate)
        if acquisition not in ('legacy_1000','legacy_250','modern_1000'):
            raise ValueError('Unsupported acquisition group')
        for name in ('data.bdf','evt.bdf'):
            member = f'Data/{person.replace("sub-","sub")}/{name}'
            original = inventory[member]
            if original.get('passed') is not True or not re.fullmatch('[a-f0-9]{64}', original['sha256']):
                raise PermissionError('Original file inventory is unqualified')
            file_bytes = _integer(original['bytes'], 'original file bytes')
            if name == 'data.bdf' and file_bytes != size + count*sum(samples)*3:
                raise ValueError('Qualified BDF file byte geometry changed')
            files.append({'subject':person,'name':name,'role':'confirmation',
                          'workspace_relative_path':f'Data/{member}', 'extraction_member':member,
                          'bytes':file_bytes,'sha256':original['sha256'],
                          'data_header_sha256':header['header_sha256'] if name=='data.bdf' else None,
                          'allowed_use':'Opaque integrity hashing only; signal decoding only declared data.bdf ranges after separate activation'})
        for arm in ('confirmation_unseen','confirmation_familiar'):
            expected = [r for r in contract['arms'][arm]['pairs'] if r['subject']==person]
            for identity in expected:
                vid = _integer(identity['vid'], 'video identity'); span = spans[vid]; cat = catalogue[vid]
                regions = segment_bounds(span['start_seconds'],span['end_seconds'],rate,count*rate)
                rows.append({'subject':person,'vid':vid,'trial':span['trial'],'role':'confirmation','arm':arm,
                    'family':cat['source_film_sha256'],'assigned_category':cat['assigned_category'],
                    'catalogue_duration_seconds':cat['catalogue_duration_seconds'],
                    'acquisition_group':acquisition,'native_rate_hz':rate,'recording_samples':count*rate,
                    'marker_start_seconds':span['start_seconds'],'marker_end_seconds':span['end_seconds'],
                    'channel_indices':indices,'common_channels':list(COMMON_CHANNELS),'segments':regions,
                    'original_data_sha256':inventory[f'Data/{person.replace("sub-","sub")}/data.bdf']['sha256'],
                    'original_header_sha256':header['header_sha256']})
    expected = {(r['subject'],r['vid']) for a in ('confirmation_unseen','confirmation_familiar') for r in contract['arms'][a]['pairs']}
    actual = {(r['subject'],r['vid']) for r in rows}
    if len(rows)!=759 or len(actual)!=759 or actual!=expected or len(files)!=66:
        raise PermissionError('Complete protected759trial/66file scope required')
    count_values=sum(30*sum(b-a for a,b in r['segments'].values()) for r in rows)
    data_bytes=sum(f['bytes'] for f in files if f['name']=='data.bdf')
    evt_bytes=sum(f['bytes'] for f in files if f['name']=='evt.bdf')
    range_reads=sum(30*sum((b-1)//r['native_rate_hz']-a//r['native_rate_hz']+1 for a,b in r['segments'].values()) for r in rows)
    return {'date':'2026-10-14','timezone':'Asia/Karachi','scope_version':'faced-inactive-confirmation-three-regions-four-models-v1',
        'status':'metadata_only_confirmation_not_activated','original_files_opened':0,'original_files_statted':0,
        'private_arrays_opened':0,'ratings_decoded':0,'waveform_samples_decoded':0,'new_fits':0,
        'protected_signal_access_authorized':False,'protected_rating_access_authorized':False,
        'old_contract_used_for_identities_only':True,'old_three_method_activation_reused':False,
        'contract_semantic_sha256':canonical_sha256(contract),'participant_count':33,
        'primary_trials':297,'primary_family_cells':264,'primary_families':8,'familiar_trials':462,
        'trials':759,'regions':2277,'models':list(MODELS),'required_prediction_count':3036,
        'native_rate_participant_counts':dict(rates),'files':files,'rows':rows,
        'resource_scope':{'selected_count_values':count_values,'selected_signal_payload_bytes':count_values*3,
            'overlap_counted_per_decode':True,'extra_boundary_samples_decoded':0,
            'selected_channel_payload_read_calls':range_reads,'data_file_bytes_one_hash_pass':data_bytes,
            'event_file_bytes_one_hash_pass':evt_bytes,'two_data_plus_one_event_hash_transport_bytes':2*data_bytes+evt_bytes,
            'two_data_plus_two_event_hash_transport_bytes':2*(data_bytes+evt_bytes),
            'data_header_bytes_one_read_pass':33*8448,'all_processed_float32_windows_bytes':2277*7*30*512*4,
            'all_regional_float64_features_bytes':2277*150*8,
            'maximum_single_counts_int32_bytes':30*32*1000*4,
            'maximum_single_calibrated_float64_bytes':30*32*1000*8,
            'behavior_file_bytes_authorized':0,'event_annotation_decodes_authorized':0},
        'future_phases':['separately approved signal/prediction activation only',
            'all3036 complete prediction-only exports byte/value locked and committed',
            'separate rating activation requires independent lock qualification before item10conversion'],
        'activation_gates_cleared_by_this_metadata_scope':0,
        'quality_policy':'Unchanged fixed engineering rules; common all-three-region category fallback; technical/nonfinite failures fatal',
        'interpretation':'Exact fixed catalogue and source recipes; no certified precision or EEG lead, original footage independence unresolved'}


def build(repo=REPO):
    repo=Path(repo)
    payloads={k:(repo/n).read_bytes() for k,n in INPUTS.items()}
    values={k:json.loads(b) for k,b in payloads.items()}
    scope=build_from_metadata(values)
    scope['input_bindings']={INPUTS[k]:hashlib.sha256(b).hexdigest() for k,b in payloads.items()}
    names=('scripts/prepare_faced_confirmation_scope_2026_10_14.py',
           'tests/test_faced_confirmation_scope_2026_10_14.py',
           'gruxnet/faced_confirmation_contract.py','gruxnet/faced_temporal_coverage.py',
           'gruxnet/faced_proposal_preprocessing.py')
    scope['code_bindings']={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names}
    return scope


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',required=True)
    args=parser.parse_args()
    scope=build(); PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PUBLIC/'scope.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(scope,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':scope['status'],'trials':scope['trials'],'regions':scope['regions'],
                     'prediction_count':scope['required_prediction_count'],'original_files_opened':0}))


if __name__=='__main__':
    main()
