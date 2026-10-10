"""Outcome-blind complete-case trigger/material qualification; no EEG decoding."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import argparse
import requests

from qualify_emo_mc import PRIVATE, PUBLIC, OLD, sha, write_json, declared_orders, parse_edf_header
from audit_emo_mc_pilot import tal_events, RESERVATION_COMMIT


def selection():
    collection = json.loads((PRIVATE / 'collection.json').read_text())
    plan = json.loads((PUBLIC / 'reservation.json').read_text())
    orders = declared_orders((OLD / 'emo_mc_readme_fixed.response').read_text(encoding='utf-8'))
    candidates = {}
    for behaviour in collection['behaviour']:
        person = behaviour['path'].split('/')[0]
        if person not in plan['participants']['development_source']:
            continue
        if any(len(orders[person][context]) != 21 for context in ('ima','vid')):
            continue
        ids = [r.get('video_name_field',r.get('material_id')) for r in behaviour['identities']]
        if len(ids) != 42 or len(set(ids)) != 42:
            continue
        sources = [r for r in collection['recordings'] if r['participant'] == person]
        candidates[person] = sources
    person = min(candidates, key=lambda p:(sum(r['file_bytes'] for r in candidates[p]),p))
    sources = candidates[person]
    if sum(r['file_bytes'] for r in sources) > 1_000_000_000:
        raise ValueError('Complete-case pilot exceeds budget')
    return collection, orders, person, sources


def declare():
    _, _, person, sources = selection()
    plan = dict(reservation_commit=RESERVATION_COMMIT, reservation_sha256=sha((PUBLIC/'reservation.json').read_bytes()),
                purpose='Verify trigger, identity and material joins only; no signal samples or ratings decoded',
                selection='Minimum total EDF bytes among development-source people with two documented 21-trial contexts and 42 unique video_name codes; ties by anonymous ID',
                participant=person, files=[r['path'] for r in sources], total_bytes=sum(r['file_bytes'] for r in sources),
                source_sha256=sha(Path(__file__).read_bytes()))
    write_json(PUBLIC/'complete_pilot_declaration.json',plan)
    print(json.dumps(plan))


def run():
    declaration=json.loads((PUBLIC/'complete_pilot_declaration.json').read_text())
    if declaration['source_sha256'] != sha(Path(__file__).read_bytes()):
        raise ValueError('Complete-pilot declaration source changed')
    if declaration['reservation_sha256'] != sha((PUBLIC/'reservation.json').read_bytes()):
        raise ValueError('Reservation changed')
    collection, orders, person, sources=selection()
    if person != declaration['participant']:
        raise ValueError('Pilot selection changed')
    objects=[]; events=[]
    # Relative annotation clocks remain separate for different runs.
    for source in sorted(sources,key=lambda r:r['path']):
        target=PRIVATE/'objects'/source['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            partial=target.with_suffix('.edf.part');digest=hashlib.sha256();count=0;last=time.monotonic()
            with requests.get(source['url'],params={'versionId':source['s3_version']},timeout=(10,30),stream=True) as response:
                response.raise_for_status()
                if response.status_code != 200: raise ValueError('Unexpected object status')
                with partial.open('wb') as out:
                    for chunk in response.iter_content(1024*1024):
                        count+=len(chunk)
                        if count>source['file_bytes']:raise ValueError('Object exceeds source size')
                        out.write(chunk);digest.update(chunk)
                        if time.monotonic()-last>15:
                            print(json.dumps({'path':source['path'],'bytes':count,'total':source['file_bytes']}),flush=True);last=time.monotonic()
            if count!=source['file_bytes'] or digest.hexdigest()!=source['annex_sha256']:raise ValueError('Annex checksum mismatch')
            partial.rename(target)
        with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=source['annex_sha256']:raise ValueError('Local object changed')
        header=PRIVATE.joinpath('headers',source['path']).with_suffix('.header').read_bytes()
        parsed=parse_edf_header(header,source['file_bytes'])
        if not parsed['geometry_consistent']:raise ValueError('Invalid EDF geometry')
        offset=0;annotation_channels=[]
        for c in parsed['channels']:
            if c['label']=='EDF Annotations':annotation_channels.append((offset*2,c['samples']*2))
            offset+=c['samples']
        run_events=[]
        with target.open('rb') as stream:
            for record in range(parsed['records']):
                for start,length in annotation_channels:
                    stream.seek(parsed['header_bytes']+record*parsed['record_bytes']+start)
                    run_events+=tal_events(stream.read(length))
        for e in run_events:e['run_path']=source['path']
        events+=run_events
        objects.append(dict(path=source['path'],bytes=source['file_bytes'],sha256=digest,full_annex_digest_matches=True,
                            annotation_description_counts=dict(Counter(e['description'] for e in run_events))))
    write_json(PRIVATE/'complete_pilot_events.json',events)
    write_json(PUBLIC/'complete_pilot_authentication.json',dict(participant=person,objects=objects,
                declaration_sha256=sha((PUBLIC/'complete_pilot_declaration.json').read_bytes()),
                rating_values_decoded=0,waveform_samples_decoded=0,models_fitted=0,research_question_changed=False))
    print(json.dumps({'participant':person,'objects_verified':len(objects),'annotation_counts':[x['annotation_description_counts'] for x in objects]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['declare','run'])
    args=parser.parse_args();declare() if args.command=='declare' else run()
