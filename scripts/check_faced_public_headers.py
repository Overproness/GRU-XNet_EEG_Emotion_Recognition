"""Three public NEMAR header pilots; ephemeral redirect locations are never saved."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit
import requests

REPO=Path(__file__).resolve().parents[1]
PRIVATE=REPO.parent/'publication_runs/faced_backup_2026-10-10'
PUBLIC=REPO/'results/development/faced_backup_2026-10-10'
spec=importlib.util.spec_from_file_location('faced_base',REPO/'scripts/qualify_faced_backup.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PILOTS=['sub-000','sub-004','sub-036']


def validate_redirect(url):
    parts=urlsplit(url)
    if parts.scheme!='https' or parts.hostname!='nemar.s3.us-east-2.amazonaws.com' or parts.username or parts.password or parts.fragment:
        raise ValueError('Public broker changed destination; do not follow')


def get_range(path,start,end,total):
    headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
    with requests.get(base.DATA+path,headers=headers,stream=True,allow_redirects=False,timeout=(10,25)) as response:
        if response.status_code==206:return base.checked_range(response,start,end,total)
        if response.status_code!=302:raise ValueError('Expected one public-broker redirect')
        location=response.headers.get('Location','')
        validate_redirect(location)
    # Location exists only in memory; it is never put in logs or artifacts.
    with requests.get(location,headers=headers,stream=True,allow_redirects=False,timeout=(10,25)) as response:
        return base.checked_range(response,start,end,total)


def declare():
    result={'date':'2026-10-10','timezone':'Asia/Karachi','reason':'Original no-redirect header collection stopped before body reads at the public NEMAR broker. Preserve all 123 failures.',
            'pilots':PILOTS,'allowed_source':'public anonymous NEMAR broker for the pinned BDF paths only',
            'redirect':'one HTTPS hop to nemar.s3.us-east-2.amazonaws.com, ephemeral memory only; no redirect response body is read',
            'forbidden':['Synapse file downloads','stored or printed signed URLs','rating tables','event tables','signal samples','second redirects'],
            'exact_range_validation':True,'maximum_header_bytes_per_file':33024,
            'script_sha256':base.sha(Path(__file__).read_bytes()),'base_script_sha256':base.sha((REPO/'scripts/qualify_faced_backup.py').read_bytes()),
            'participant_reservation_sha256':base.sha((PUBLIC/'participant_reservation.json').read_bytes()),'old_collection_sha256':base.sha((PUBLIC/'collection.json').read_bytes())}
    target=PUBLIC/'public_header_pilot_plan.json'
    if target.exists():raise ValueError('Do not overwrite a declared pilot')
    base.write(target,result);print(json.dumps({'declared':True,'pilots':PILOTS,'signed_urls_retained':0}))


def collect():
    plan=json.loads((PUBLIC/'public_header_pilot_plan.json').read_text())
    if plan['script_sha256']!=base.sha(Path(__file__).read_bytes()):raise ValueError('Pilot collector changed')
    if (PUBLIC/'public_header_pilots.json').exists():raise ValueError('Preserve prior pilot result')
    old={r['subject']:r for r in json.loads((PUBLIC/'collection.json').read_text())['records']}
    records=[]
    for subject in PILOTS:
        record={'subject':subject,'rating_values_decoded':0,'waveform_samples_decoded':0,'signed_urls_retained':0}
        try:
            path=f'{subject}/eeg/{subject}_task-watchingVideoClips_eeg.bdf'
            total=old[subject]['expected_object_bytes']
            first=get_range(path,0,255,total)
            size=int(first[184:192]);count=int(first[252:256])
            if not 1<=count<=128 or size!=256+256*count:raise ValueError('Invalid header byte budget')
            header=first+get_range(path,256,size-1,total)
            target=PRIVATE/'pilot_headers'/(subject+'.bdf.header');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(header)
            record.update({'passed':True,'header_sha256':base.sha(header),'header':base.parse_header(header,total)})
        except (requests.RequestException,ValueError,UnicodeError) as error:
            # RequestException strings may contain transient signed locations: never log them.
            record.update({'passed':False,'failure_type':type(error).__name__,
                           'failure_message':str(error)[:120] if isinstance(error,ValueError) else 'public_header_request_failed'})
        records.append(record)
    base.write(PUBLIC/'public_header_pilots.json',{'records':records,'plan_sha256':base.sha((PUBLIC/'public_header_pilot_plan.json').read_bytes()),'full_object_digests_reverified':False})
    print(json.dumps({'pilots':len(records),'passed':sum(r['passed'] for r in records),'failed':sum(not r['passed'] for r in records),'signed_urls_retained':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','collect'))
    args=parser.parse_args();declare() if args.mode=='declare' else collect()
