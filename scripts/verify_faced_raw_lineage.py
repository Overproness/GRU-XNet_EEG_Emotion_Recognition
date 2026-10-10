"""Match opaque original signal bytes to pinned curator objects using headers only.

One public NEMAR redirect is reused from the previously audited header collector.
No signal payload is downloaded, numeric sample decoded or access URL retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
from pathlib import Path
import requests


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


raw=module('raw_lineage_base',Path(__file__).with_name('audit_faced_raw.py'))
pilot=module('curator_pilot',Path(__file__).with_name('check_faced_public_headers.py'))
PUBLIC=raw.PUBLIC
PRIVATE=raw.WORKSPACE/'publication_runs/faced_raw_2026-10-11/curator_headers'


def declare():
    raw.check_plan()
    source=raw.load(PUBLIC/'extraction.json')
    curator=raw.load(raw.BACKUP/'collection.json')['records']
    files={r['member']:r for r in source['records']}
    subjects=[r['subject'] for r in curator if files[f"Data/{r['subject'].replace('sub-','sub')}/data.bdf"]['sha256']!=r['annex_declared_object_sha256']]
    raw.save(PUBLIC/'lineage_plan.json',{
        'date':'2026-10-11','script_sha256':raw.digest(Path(__file__)),
        'pilot_collector_sha256':raw.digest(Path(pilot.__file__)),
        'base_collector_sha256':raw.digest(Path(pilot.base.__file__)),
        'extraction_sha256':raw.digest(PUBLIC/'extraction.json'),
        'curator_collection_sha256':raw.digest(raw.BACKUP/'collection.json'),
        'subjects':subjects,'initial_whole_file_matches':123-len(subjects),
        'reason':'Original whole-file hashes differ for 61 first-layout subjects; test whether substituting only pinned curator header bytes reproduces each full curator digest',
        'route':'One ephemeral HTTPS public broker redirect to nemar.s3.us-east-2.amazonaws.com; exact header ranges and expected totals only',
        'private_storage':'Bounded original curator headers only, never access links; no private header fields decoded/exported',
        'maximum_header_bytes':33024,'workers':4,'signal_download_bytes':0,
        'hash_scope':'Stream local original file, substitute curator header in memory and hash body as opaque bytes; additionally require unchanged complete original hash',
        'authentication_scope':'Pinned curator object digest equivalence; not an independent author release checksum or a signal quality test',
        'rating_values_decoded':0,'waveform_samples_decoded':0,'models_fitted':0,'research_question_changed':False})


def normalized_hash(path,header):
    original,normalized,payload=hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
    with path.open('rb') as f:
        first=f.read(256)
        size=int(first[184:192])
        if size!=len(header):
            raise ValueError('Header replacement cannot change geometry')
        old=first+f.read(size-256)
        original.update(old);normalized.update(header)
        while chunk:=f.read(8*1024*1024):
            original.update(chunk);normalized.update(chunk);payload.update(chunk)
    # Export field-level counts, never private bytes or strings.
    sections=[('version',0,8),('patient_identification',8,88),('recording_identification',88,168),
              ('start_date',168,176),('start_time',176,184),('fixed_technical',184,256),
              ('channel_technical',256,size)]
    differences={name:sum(a!=b for a,b in zip(old[lo:hi],header[lo:hi])) for name,lo,hi in sections}
    return original.hexdigest(),normalized.hexdigest(),payload.hexdigest(),differences


def collect():
    raw.check_plan()
    plan=raw.load(PUBLIC/'lineage_plan.json')
    required={
        'script_sha256':Path(__file__),'pilot_collector_sha256':Path(pilot.__file__),
        'base_collector_sha256':Path(pilot.base.__file__),
        'extraction_sha256':PUBLIC/'extraction.json','curator_collection_sha256':raw.BACKUP/'collection.json'}
    if any(plan[k]!=raw.digest(p) for k,p in required.items()):
        raise ValueError('Lineage declaration changed')
    curator={r['subject']:r for r in raw.load(raw.BACKUP/'collection.json')['records']}
    files={r['member']:r for r in raw.load(PUBLIC/'extraction.json')['records']}
    PRIVATE.mkdir(parents=True,exist_ok=True)
    def check(subject):
        target=PRIVATE/(subject+'.header')
        c=curator[subject];total=c['expected_object_bytes']
        record={'subject':subject,'expected_curator_sha256':c['annex_declared_object_sha256']}
        try:
            if target.exists():
                raise ValueError('Preserve previous private header; do not overwrite')
            path=f'{subject}/eeg/{subject}_task-watchingVideoClips_eeg.bdf'
            first=pilot.get_range(path,0,255,total)
            size,count=int(first[184:192]),int(first[252:256])
            if not 1<=count<=128 or size!=256+256*count:
                raise ValueError('Invalid header byte budget')
            header=first+pilot.get_range(path,256,size-1,total)
            technical=pilot.base.parse_header(header,total)
            target.write_bytes(header)
            original,normalized,payload,diff=normalized_hash(raw.RAW/subject.replace('sub-','sub')/'data.bdf',header)
            source=files[f"Data/{subject.replace('sub-','sub')}/data.bdf"]
            passed=(original==source['sha256'] and payload==source['signal_payload_sha256'] and normalized==c['annex_declared_object_sha256'])
            record.update({'passed':passed,'original_sha256':original,'normalized_full_file_sha256':normalized,
                'local_signal_payload_sha256':payload,'curator_header_sha256':hashlib.sha256(header).hexdigest(),
                'header':technical,'changed_header_byte_counts':diff})
        except (requests.RequestException,ValueError,UnicodeError) as error:
            # Never serialize request exception messages, which could contain access URLs.
            record.update({'passed':False,'failure_type':type(error).__name__,
                           'reason':str(error) if isinstance(error,ValueError) else 'public_header_request_failed'})
        return record
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(check,plan['subjects']):
            records.append(row)
            print({'checked':len(records),'subject':row['subject'],'passed':row['passed']},flush=True)
    raw.save(PUBLIC/'lineage.json',{'plan_sha256':raw.digest(PUBLIC/'lineage_plan.json'),
        'records':records,'direct_full_file_matches':plan['initial_whole_file_matches'],
        'header_substitution_matches':sum(r['passed'] for r in records),
        'signal_payloads_downloaded':0,'waveform_samples_decoded':0,'rating_values_decoded':0,
        'signed_urls_retained':0,'independent_author_release_checksum':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('declare','collect'))
    args=parser.parse_args()
    declare() if args.mode=='declare' else collect()
