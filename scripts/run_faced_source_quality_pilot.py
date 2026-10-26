"""Declared eight-trial source-only FACED quality pilot; no behavioral decoding."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

REPO=Path(__file__).resolve().parents[1]
WORKSPACE=REPO.parent
sys.path.insert(0,str(REPO))
import numpy as np
import scipy
from gruxnet.faced_source_quality import (
    QUALITY_RULES, decode_signed24, read_interval, selected_header, quality_metrics)
from gruxnet.faced_proposal_preprocessing_v2 import (
    COMMON_CHANNELS, common_indices, calibrate_common_counts, late_segment_bounds, late_trial_windows)

PUBLIC=REPO/'results/development/faced_source_quality_2026-10-11'
PRIVATE=WORKSPACE/'publication_runs/faced_source_quality_2026-10-11'
RAW=WORKSPACE/'Data/Data'
DESIGN=REPO/'results/development/faced_design_proposal_2026-10-11'
QUALIFIED=REPO/'results/development/faced_raw_2026-10-11'


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def save(path,value):
    if path.exists(): raise ValueError('Preserve previous receipt: '+path.name)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def module(path):
    spec=importlib.util.spec_from_file_location('source_quality_header',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def dependencies():
    paths=[Path(__file__),REPO/'gruxnet/faced_source_quality.py',
        REPO/'tests/test_faced_source_quality.py',REPO/'gruxnet/faced_proposal_preprocessing.py',
        REPO/'gruxnet/faced_proposal_preprocessing_v2.py',REPO/'scripts/audit_faced_raw.py',
        DESIGN/'source_quality_pilot_candidate.json',DESIGN/'recommended_proposal.json',
        DESIGN/'packet_verification.json',QUALIFIED/'verification.json',
        REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json',
        REPO/'results/development/emo_mc_qualification_2026-10-10/reservation.json',
        WORKSPACE/'report.tex',PUBLIC/'decoder_reference.json']
    return {p.relative_to(WORKSPACE).as_posix():digest(p) for p in paths}


def prior_bindings():
    for path in (DESIGN/'packet_verification.json',QUALIFIED/'verification.json'):
        for relative,expected in load(path)['bindings'].items():
            if digest(REPO/relative)!=expected: raise ValueError('Prior qualification binding changed')


def candidate():
    prior_bindings()
    c=load(DESIGN/'source_quality_pilot_candidate.json')
    roles=load(REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json')
    p=load(DESIGN/'recommended_proposal.json')
    clips=set(p['rotations'][0]['clips']['development_source'])
    if c['recommended_proposal_sha256']!=digest(DESIGN/'recommended_proposal.json'):
        raise ValueError('Proposal changed')
    rows=c['trial_identities']
    if len(rows)!=8 or len({(r['subject'],r['vid']) for r in rows})!=8:
        raise ValueError('Expected exactly eight distinct trials')
    if any(r['subject'] not in roles['development_source'] or r['vid'] not in clips for r in rows):
        raise ValueError('Protected identity encountered before raw access')
    return c


def synthetic():
    """Independent library check uses manufactured bytes only."""
    import pyedflib
    values=np.asarray([
        [-8388608,-65537,-1,0,1,255,65536,8388607,18,-18,999,-999],
        [21,-21,8388607,-8388608,0,-1,2,3,4,5,6,7]],dtype=np.int32)
    field=lambda v,n:str(v).ljust(n).encode('ascii')
    fixed=(b'\xffBIOSEMI'+field('X',80)+field('X',80)+field('01.01.01',8)
        +field('00.00.00',8)+field(768,8)+field('24BIT',44)+field(3,8)+field(1,8)+field(2,4))
    columns=[(['C3','C4'],16),(['',''],80),(['uV','uV'],8),([-375000]*2,8),
        ([375000]*2,8),([-8388608]*2,8),([8388607]*2,8),(['',''],80),([4,4],8),(['',''],32)]
    header=fixed+b''.join(field(v,n) for vals,n in columns for v in vals)
    body=b''.join(int(v).to_bytes(3,'little',signed=True) for r in range(3)
        for ch in range(2) for v in values[ch,r*4:(r+1)*4])
    PRIVATE.mkdir(parents=True,exist_ok=True)
    path=PRIVATE/'synthetic_reference.bdf'
    if path.exists(): raise ValueError('Preserve previous manufactured fixture')
    path.write_bytes(header+body)
    h={'sample_width':3,'continuity':'24BIT','header_bytes':768,'record_count':3,
       'columns':{'samples_per_record':['4','4']}}
    with path.open('rb') as f: actual,access=read_interval(f,h,3,10,[1,0],True)
    with pyedflib.EdfReader(str(path)) as external:
        alternate=np.stack([external.readSignal(ch,3,7,digital=True) for ch in (1,0)])
        physical=np.stack([external.readSignal(ch,3,7) for ch in (1,0)])
    expected=values[[1,0],3:10]
    scale=750000/16777215
    independent_physical=(expected.astype(float)+.5)*scale
    if not np.array_equal(actual,expected) or not np.array_equal(actual,alternate):
        raise ValueError('Independent library signed/order/interval check failed')
    error=float(np.max(np.abs(physical-independent_physical)))
    if error>1e-8: raise ValueError('Independent library calibration check failed')
    save(PUBLIC/'decoder_reference.json',{
        'manufactured_bytes_only':True,'fixture_sha256':digest(path),
        'independent_library':'pyedflib','version':pyedflib.__version__,
        'digital_values_compared':int(actual.size),'digital_mismatches':0,
        'physical_max_absolute_error_uv':error,'original_files_opened':0,
        'sample_geometry':'Two channels, three four-sample records; partial-record interval 3:10, reversed channels',
        'edge_values_include':'Signed 24-bit endpoints, negatives, zero and byte-boundary values',
        'format_source':'https://www.biosemi.com/faq/file_format.htm',
        'calibration_source':'https://www.edfplus.info/specs/guidelines.html',
        'independent_reader_documentation':'https://pyedflib.readthedocs.io/en/latest/ref/edfreader.html'})
    print({'synthetic_reference_passed':True,'original_files_opened':0})


def declare():
    c=candidate()
    save(PUBLIC/'plan.json',{
        'date':'2026-10-11','timezone':'Asia/Karachi','bindings':dependencies(),
        'authorization':'User explicitly requested the proposed eight-trial source-only decoder/quality pilot; all ratings sealed',
        'trial_identities':c['trial_identities'],'quality_rules':QUALITY_RULES,
        'allowed':['Opaque complete-file/header/body hashes of four original data.bdf files',
            'Opaque hashes only of four evt.bdf and four After_remarks.mat companions',
            'Technical original headers, ignoring identity/date strings',
            'Numerical samples from 30 common scalp channels in exactly eight declared 32-second intervals',
            'Fixed trial-local preprocessing and descriptive per-channel/window quality summaries'],
        'forbidden':['Behavioral MAT decoding and any individual ratings, accuracy or response times',
            'recordInformation.json interpretation','Numerical auxiliary-channel or neighbor-trial samples',
            'Validation/confirmation people or materials','Model fitting, supervised evaluation or paper-question change'],
        'interval':'floor(video_end*native_rate)-32*native_rate .. floor(video_end*native_rate), half-open',
        'core':'Approximately end-30 .. end-2; seven nonoverlapping four-second windows',
        'decoder_validation':'Every decoded real count checked against int.from_bytes; manufactured fixture separately checked by PyEDFlib',
        'calibration_check':'Original physical/digital extrema formula versus independently expressed gain*count+offset; atol 1e-8 uV',
        'technical_failure_rule':'Hash/header/bounds/decoding/calibration/finite/geometry failures stop further decoding and supervised activation',
        'quality_rule':'Any engineering review flag holds wider sample/outcome access pending documented source-only diagnosis; do not exclude flagged trials',
        'publication':'Only anonymous IDs, declared intervals, hashes and descriptive quality aggregates; no sample arrays or private headers',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'ratings_to_open':0,'models_to_fit':0,'research_question_changed':False})
    print({'declared':True,'source_trials':8,'ratings_to_open':0},flush=True)


def check():
    candidate();plan=load(PUBLIC/'plan.json')
    if plan['bindings']!=dependencies() or plan['quality_rules']!=QUALITY_RULES:
        raise ValueError('Pilot declaration dependencies changed')
    if plan['trial_identities']!=candidate()['trial_identities']:
        raise ValueError('Pilot identities changed')
    return plan


def run():
    plan=check()
    if (PUBLIC/'records.json').exists(): raise ValueError('Preserve completed pilot')
    header_reader=module(REPO/'scripts/audit_faced_raw.py')
    headers={r['subject']:r['data.bdf'] for r in load(QUALIFIED/'headers.json')['records']}
    joins={r['subject']:r for r in load(QUALIFIED/'joins_numeric_projection.json')['records']}
    files={r['member']:r for r in load(QUALIFIED/'extraction.json')['records']}
    authenticated={};rows=[];began=time.monotonic()
    for ident in plan['trial_identities']:
        subject,vid=ident['subject'],ident['vid'];h=headers[subject]
        root=RAW/subject.replace('sub-','sub');path=root/'data.bdf'
        if subject not in authenticated:
            technical=header_reader.read_header(path)
            if technical!=h: raise ValueError('Original technical header changed')
            if digest(path)!=files[f'Data/{root.name}/data.bdf']['sha256']:
                raise ValueError('Original signal bytes changed before decoding')
            for name in ('evt.bdf','After_remarks.mat'):
                if digest(root/name)!=files[f'Data/{root.name}/{name}']['sha256']:
                    raise ValueError('Companion identity/marker bytes changed')
            authenticated[subject]={'original_sha256':files[f'Data/{root.name}/data.bdf']['sha256'],
                'signal_payload_sha256':files[f'Data/{root.name}/data.bdf']['signal_payload_sha256'],
                'file_stat':(path.stat().st_size,path.stat().st_mtime_ns)}
            print({'authenticated_source_file':subject,'elapsed_seconds':round(time.monotonic()-began,2)},flush=True)
        spans=[s for s in joins[subject]['spans'] if s['vid']==vid]
        if len(spans)!=1 or not joins[subject]['passed']: raise ValueError('Unqualified identity join')
        span=spans[0];rate=int(h['sample_rates_hz'][0])
        start,stop=late_segment_bounds(span['start_seconds'],span['end_seconds'],rate,
                                     h['record_count']*int(h['columns']['samples_per_record'][0]))
        indices=common_indices(h['columns']['labels'])
        with path.open('rb') as f: counts,access=read_interval(f,h,start,stop,indices,True)
        projected=selected_header(h,indices)
        values=calibrate_common_counts(counts,subject,projected)
        cols=projected['columns'];dmin=np.asarray(cols['digital_min'],float)[:,None]
        pmin=np.asarray(cols['physical_min'],float)[:,None]
        gain=(np.asarray(cols['physical_max'],float)[:,None]-pmin)/(np.asarray(cols['digital_max'],float)[:,None]-dmin)
        independent=counts.astype(float)*gain+(pmin-dmin*gain)
        error=float(np.max(np.abs(values-independent)))
        if error>1e-8: raise ValueError('Independent physical calibration differs')
        windows=late_trial_windows(values,rate)
        quality=quality_metrics(counts,values,windows,rate,plan['quality_rules'])
        if (path.stat().st_size,path.stat().st_mtime_ns)!=authenticated[subject]['file_stat']:
            raise ValueError('Original changed during pilot')
        save(PRIVATE/f'{subject}_clip{vid:02d}_access.json',access)
        row={'subject':subject,'vid':vid,'trial':span['trial'],'native_rate_hz':rate,
            'acquisition_group':('legacy_' if int(subject[-3:])<61 else 'modern_')+str(rate),
            'sample_interval':[start,stop],'raw_interval_seconds':[start/rate,stop/rate],
            'video_marker_interval_seconds':[span['start_seconds'],span['end_seconds']],
            'end_marker_rounding_seconds':span['end_seconds']-stop/rate,
            'source_file':{k:v for k,v in authenticated[subject].items() if k!='file_stat'},
            'original_header_sha256':h['header_sha256'],'original_unit_literal':h['columns']['units'][0],
            'decoded_channel_indices':indices,'canonical_channels':COMMON_CHANNELS,
            'decoded_values':int(counts.size),'signal_bytes_decoded':int(counts.size*3),
            'access_range_count':len(access['ranges']),'access_range_ledger_sha256':access['range_ledger_sha256'],
            'every_real_count_scalar_reference_checked':True,'extra_boundary_samples_decoded':0,
            'calibration_max_absolute_error_uv':error,'output_shape':list(windows.shape),
            'quality':quality,'ratings_decoded':0,'technical_passed':True}
        rows.append(row)
        print({'source_trial':[subject,vid],'technical_passed':True,'quality_flagged_channels':len(quality['flagged_channels'])},flush=True)
    # Rehash originals after numerical access; this remains opaque byte processing.
    for subject,a in authenticated.items():
        if digest(RAW/subject.replace('sub-','sub')/'data.bdf')!=a['original_sha256']:
            raise ValueError('Original signal bytes changed after decoding')
    save(PUBLIC/'records.json',{'plan_sha256':digest(PUBLIC/'plan.json'),'records':rows,
        'elapsed_seconds':round(time.monotonic()-began,3),'ratings_decoded':0,'models_fitted':0,
        'auxiliary_or_neighbor_samples_decoded':0,'research_question_changed':False})
    flagged=[{'subject':r['subject'],'vid':r['vid'],'channels':r['quality']['flagged_channels']}
             for r in rows if r['quality']['flagged_channels']]
    summary={'source_trials':len(rows),'source_participants':len(authenticated),
        'source_clips':[11,27],'acquisition_groups':sorted({r['acquisition_group'] for r in rows}),
        'technical_passed':all(r['technical_passed'] for r in rows),
        'real_count_values_decoded_and_scalar_checked':sum(r['decoded_values'] for r in rows),
        'processed_windows':sum(r['quality']['valid_windows'] for r in rows),
        'trials_or_windows_dropped':0,'flagged_trials':flagged,
        'quality_review_required':bool(flagged),'supervised_access_cleared':False,
        'ratings_decoded':0,'models_fitted':0,'research_question_changed':False,
        'limits':['Eight source trials are not representative population QC or model-performance evidence',
            'No affected ?V-unit participant was selected; documented repair remains synthetic/lineage-qualified only',
            'No independent physical display/EEG synchronization, artifact-free or neural-origin claim',
            'Individual target-vector geometry and selective outcome decoder remain unopened']}
    save(PUBLIC/'summary.json',summary)
    paths=[PUBLIC/name for name in ('plan.json','decoder_reference.json','records.json','summary.json')]
    save(PUBLIC/'verification.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in paths},
        'script_sha256':digest(Path(__file__)),'local_access_ledgers':8,'ratings_decoded':0,
        'models_fitted':0,'research_question_changed':False})
    print({k:summary[k] for k in ('source_trials','technical_passed','quality_review_required','real_count_values_decoded_and_scalar_checked')},flush=True)


def verify(local=False):
    plan=check();receipt=load(PUBLIC/'verification.json')
    if receipt['script_sha256']!=digest(Path(__file__)) or any(digest(REPO/p)!=sha for p,sha in receipt['bindings'].items()):
        raise ValueError('Completed pilot binding changed')
    rows=load(PUBLIC/'records.json')['records']
    if [(r['subject'],r['vid']) for r in rows]!=[(r['subject'],r['vid']) for r in plan['trial_identities']]:
        raise ValueError('Completed pilot crosses identity boundary')
    for row in rows:
        if row['output_shape']!=[7,30,512] or row['ratings_decoded'] or row['extra_boundary_samples_decoded']:
            raise ValueError('Pilot geometry or seal changed')
        if local:
            ledger=load(PRIVATE/f"{row['subject']}_clip{row['vid']:02d}_access.json")
            ranges=ledger['ranges'];start,stop=row['sample_interval']
            key=hashlib.sha256(json.dumps(ranges,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if key!=row['access_range_ledger_sha256'] or any(
                not (start<=r['start_sample']<r['stop_sample']<=stop)
                or r['channel_index'] not in row['decoded_channel_indices'] for r in ranges):
                raise ValueError('Selective access ledger crosses declared range')
            if sum(r['bytes'] for r in ranges)!=row['signal_bytes_decoded']:
                raise ValueError('Access byte accounting differs')
    print({'verified':True,'source_trials':len(rows),'local_access_ledgers_checked':len(rows) if local else 0,
           'ratings_decoded':0,'models_fitted':0,'supervised_access_cleared':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('synthetic','declare','run','verify'))
    parser.add_argument('--local',action='store_true');args=parser.parse_args()
    if args.mode=='verify': verify(args.local)
    else: {'synthetic':synthetic,'declare':declare,'run':run}[args.mode]()
