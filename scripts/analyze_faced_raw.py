"""Summarize and replay FACED raw qualification without opening outcomes."""
import argparse
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


base=module('raw_analysis_base',Path(__file__).with_name('audit_faced_raw.py'))
projection=module('raw_analysis_projection',Path(__file__).with_name('audit_faced_annotation_projection.py'))
PUBLIC=base.PUBLIC
FILES=['plan.json','extraction.json','archive_deletion.json','headers.json','joins.json',
       'annotation_channel_plan.json','joins_annotations.json','numeric_projection_plan.json',
       'joins_numeric_projection.json','lineage_plan.json','lineage.json']
SCRIPTS=['audit_faced_raw.py','audit_faced_raw_events.py','audit_faced_annotation_projection.py',
         'verify_faced_raw_lineage.py','analyze_faced_raw.py']


def summary():
    base.check_plan()
    extraction=base.load(PUBLIC/'extraction.json')
    deletion=base.load(PUBLIC/'archive_deletion.json')
    headers=base.load(PUBLIC/'headers.json')['records']
    joins=base.load(PUBLIC/'joins_numeric_projection.json')['records']
    lineage=base.load(PUBLIC/'lineage.json')
    if not extraction['all_members_match'] or len(extraction['records'])!=492:
        raise ValueError('Incomplete extraction')
    expected={f'Data/sub{i:03d}/{name}' for i in range(123) for name in ('data.bdf','evt.bdf','After_remarks.mat','recordInformation.json')}
    if {r['member'] for r in extraction['records']}!=expected:
        raise ValueError('Unexpected participant/file inventory')
    if deletion['archive_sha256']!=extraction['archive_sha256'] or deletion['extraction_receipt_sha256']!=base.digest(PUBLIC/'extraction.json'):
        raise ValueError('Deletion receipt binding mismatch')
    if [r['subject'] for r in headers]!=[f'sub-{i:03d}' for i in range(123)] or [r['subject'] for r in joins]!=[r['subject'] for r in headers]:
        raise ValueError('Participant inventory/order mismatch')
    by_subject={r['subject']:r for r in headers}
    gains=set();units=Counter();rates=Counter();prefilters=Counter();bad_unit_subjects=[]
    orders=Counter()
    for row in headers:
        data=row['data.bdf']; c=data['columns']
        if data['header_bytes']!=8448 or len(c['labels'])!=32 or data['record_duration_seconds']!=1 or data['sample_width']!=3:
            raise ValueError('Signal header geometry changed')
        if len(set(c['units']))!=1 or len(set(data['sample_rates_hz']))!=1 or len(set(c['prefilter']))!=1:
            raise ValueError('Unexpected mixed per-recording acquisition settings')
        unit=c['units'][0];units[unit]+=1;rates[str(data['sample_rates_hz'][0])]+=1;prefilters[c['prefilter'][0]]+=1
        orders[tuple(c['labels'])]+=1
        if unit=='?V':
            bad_unit_subjects.append(row['subject'])
        for pmin,pmax,dmin,dmax in zip(c['physical_min'],c['physical_max'],c['digital_min'],c['digital_max']):
            if float(pmin)>=float(pmax) or int(dmin)>=int(dmax):
                raise ValueError('Invalid physical calibration')
            gains.add((float(pmax)-float(pmin))/(int(dmax)-int(dmin)))
    if bad_unit_subjects!=[f'sub-{i:03d}' for i in [4,5,*range(8,36),58,59,60]]:
        raise ValueError('Corrupted-unit participant inventory changed')
    flags=Counter();differences=[];anomalies=[];opaque=[]
    for row in joins:
        if not row['passed']:
            raise ValueError('Unparsed participant')
        events=row['events']; spans=row['spans']
        reconstructed,observed_anomalies=base.strict_spans(events)
        slim=[{k:s[k] for k in ('vid','start_seconds','end_seconds','duration_seconds')} for s in spans]
        if reconstructed!=slim or observed_anomalies!=row['anomalies'] or base.curator_spans(events)!=slim:
            raise ValueError('Timing replay mismatch')
        if sorted(s['vid'] for s in spans)!=list(range(1,29)):
            raise ValueError('Incomplete/duplicate trial identity')
        if [s['trial'] for s in spans]!=list(range(1,29)):
            raise ValueError('Presentation join mismatch')
        identities=row['identity_rows']
        identity_map={r['vid']:r['trial'] for r in identities}
        if len(identity_map)!=28 or sorted(identity_map.values())!=list(range(1,29)) or any(s['trial']!=identity_map[s['vid']] for s in spans):
            raise ValueError('Behavior/event identity mismatch')
        limit=by_subject[row['subject']]['data.bdf']['recording_duration_seconds']
        if not all(0<=s['start_seconds']<s['end_seconds']<=limit for s in spans):
            raise ValueError('Endpoint exceeds signal geometry')
        if any(spans[i]['end_seconds']>spans[i+1]['start_seconds'] for i in range(27)):
            raise ValueError('Video spans overlap')
        for key in ('all_28_videos_once','presentation_order_matches','curator_spans_equal','all_endpoints_in_data_geometry'):
            flags[key]+=bool(row[key])
        if row['identity_scope']['rating_values_decoded']!=0 or row['identity_scope']['skipped_protected_matrices']!=84:
            raise ValueError('Protected field boundary changed')
        differences.extend(s['duration_difference_seconds'] for s in spans)
        if row['duration_flags_over_one_second']!=[s['vid'] for s in spans if abs(s['duration_difference_seconds'])>1]:
            raise ValueError('Duration flag replay mismatch')
        if row['anomalies']:
            anomalies.append({'subject':row['subject'],'events':row['anomalies']})
        if row['opaque_annotations']:
            opaque.append({'subject':row['subject'],'count':len(row['opaque_annotations']),
                          'inside_video_spans':row['opaque_annotations_inside_video_spans']})
    if lineage['direct_full_file_matches']!=62 or len(lineage['records'])!=61:
        raise ValueError('Unexpected lineage inventory')
    original_files={r['member']:r for r in extraction['records']}
    curator={r['subject']:r for r in base.load(base.BACKUP/'collection.json')['records']}
    for row in lineage['records']:
        original=original_files[f"Data/{row['subject'].replace('sub-','sub')}/data.bdf"]
        if row['original_sha256']!=original['sha256'] or row['local_signal_payload_sha256']!=original['signal_payload_sha256']:
            raise ValueError('Local lineage input mismatch')
        if row['passed'] and row['normalized_full_file_sha256']!=curator[row['subject']]['annex_declared_object_sha256']:
            raise ValueError('Curator digest mismatch')
    direct=sum(original_files[f"Data/{s.replace('sub-','sub')}/data.bdf"]['sha256']==c['annex_declared_object_sha256'] for s,c in curator.items())
    corrected=sum(r['passed'] for r in lineage['records'])
    reservation=base.load(base.BACKUP/'participant_reservation.json')
    return {
        'date':'2026-10-11','participants':123,'original_video_trials':sum(len(r['spans']) for r in joins),
        'verified_extracted_files':extraction['files'],'extracted_file_bytes':extraction['extracted_file_bytes'],
        'archive_deleted':deletion['deleted'],'archive_bytes_removed':deletion['bytes_removed'],
        'technical_headers':246,'sampling_frequency_participants':dict(rates),'unit_literal_participants':dict(units),
        'corrupted_unit_subjects':bad_unit_subjects,'header_physical_units_per_count':sorted(gains),
        'original_prefilter_literals_participants':dict(prefilters),
        'channel_orders':[{'labels':list(order),'participants':count} for order,count in orders.items()],
        'trial_join_checks_passed_participants':dict(flags),
        'duration_difference_seconds_range':[min(differences),max(differences)],
        'duration_flags_over_one_second':sum(len(r['duration_flags_over_one_second']) for r in joins),
        'longer_clip22_version_participants':25,'observed_stray_start_subjects':[r['subject'] for r in anomalies],
        'stray_start_records':anomalies,'documented_exception_without_stray_numeric_start':['sub-049'],
        'opaque_annotation_subjects':opaque,'opaque_annotation_total':sum(r['count'] for r in opaque),
        'opaque_annotations_inside_video_spans':sum(r['inside_video_spans'] for r in opaque),
        'behavioral_identity_scalars_decoded':123*28*2,'protected_behavioral_matrices_skipped':123*28*3,
        'direct_curator_whole_file_matches':direct,'curator_matches_after_only_header_substitution':corrected,
        'curator_signal_payload_lineage_qualified_participants':direct+corrected,
        'curator_pinned_revision':'4c37c73ece79e68702de2d8afa6b9274811a7cc8',
        'independent_author_release_checksum_authenticated':False,
        'participant_roles_preserved':{key:len(reservation[key]) for key in ('development_source','development_validation','confirmation')},
        'numeric_trial_joins_qualified':all(n==123 for n in flags.values()),
        'rating_values_decoded':0,'waveform_samples_decoded':0,'models_fitted':0,
        'research_question_changed':False,'fitting_ready':False,
        'remaining_gates':['Feasible target/material-family design and reservation before outcome access',
            'Explicit channel/reference/preprocessing policy and clip-version content alignment',
            'Cross-corpus stimulus and pretraining membership qualification',
            'Separately declared development outcome access; sealed confirmation',
            'Author approval of evidence-backed proposal before any paper question change'],
        'scope_limits':['Header-substituted hashes prove opaque sample-byte equivalence to pinned curator objects, not signal quality',
            'Endpoints are original task-marker times; no measured display/audio onset latency or physical shared-clock calibration',
            'Non-numeric annotation texts remain opaque; their purpose was not identified',
            'Stimulus media/version hashes and exact content alignment remain unauthenticated']}


def build():
    value=summary()
    base.save(PUBLIC/'summary.json',value)
    bound={f'results/development/faced_raw_2026-10-11/{name}':base.digest(PUBLIC/name) for name in FILES+['summary.json']}
    bound.update({f'scripts/{name}':base.digest(base.REPO/'scripts'/name) for name in SCRIPTS})
    base.save(PUBLIC/'verification.json',{'bindings':bound,'rating_values_decoded':0,'waveform_samples_decoded':0,
        'models_fitted':0,'research_question_changed':False,'fitting_ready':False})
    print({k:value[k] for k in ('participants','original_video_trials','curator_signal_payload_lineage_qualified_participants','fitting_ready')})


def verify(local=False):
    receipt=base.load(PUBLIC/'verification.json')
    if any(base.digest(base.REPO/name)!=expected for name,expected in receipt['bindings'].items()):
        raise ValueError('Frozen artifact or source changed')
    if summary()!=base.load(PUBLIC/'summary.json'):
        raise ValueError('Summary replay mismatch')
    count=0
    if local:
        headers=base.load(PUBLIC/'headers.json')['records']
        joins={r['subject']:r for r in base.load(PUBLIC/'joins_numeric_projection.json')['records']}
        files={r['member']:r for r in base.load(PUBLIC/'extraction.json')['records']}
        for row in headers:
            root=base.RAW/row['subject'].replace('sub-','sub')
            for name in ('data.bdf','evt.bdf'):
                if base.read_header(root/name)!=row[name]:
                    raise ValueError('Local header replay mismatch')
            for name in ('evt.bdf','After_remarks.mat'):
                key=f"Data/{root.name}/{name}"
                if base.digest(root/name)!=files[key]['sha256']:
                    raise ValueError('Local marker/identity file changed')
            identities,scope=base.identity_projection((root/'After_remarks.mat').read_bytes())
            if identities!=joins[row['subject']]['identity_rows'] or scope!=joins[row['subject']]['identity_scope']:
                raise ValueError('Local identity replay mismatch')
            event_rows=[];opaque=[];clocks=[]
            with (root/'evt.bdf').open('rb') as f:
                f.seek(row['evt.bdf']['header_bytes'])
                for _ in range(row['evt.bdf']['record_count']):
                    f.seek(3,1)
                    events,times,other=projection.project_tals(f.read(300))
                    event_rows.extend(events);clocks.extend(times);opaque.extend(other)
            j=joins[row['subject']]
            if event_rows!=j['events'] or opaque!=j['opaque_annotations'] or clocks!=j['annotation_clocks']:
                raise ValueError('Local marker projection replay mismatch')
            count+=1
        if (base.WORKSPACE/'Data.zip').exists():
            raise ValueError('Expected authorized redundant archive removal')
    print({'verified':True,'bound_files':len(receipt['bindings']),'local_participants_replayed':count,
           'signal_bodies_rehashed_this_replay':False,'fitting_ready':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('build','verify'));parser.add_argument('--local',action='store_true')
    args=parser.parse_args()
    build() if args.mode=='build' else verify(args.local)
