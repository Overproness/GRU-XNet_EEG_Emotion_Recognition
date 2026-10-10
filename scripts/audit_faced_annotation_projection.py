"""Numerical event projection retaining hashes, never text, of other annotations."""
import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import re

spec=importlib.util.spec_from_file_location('faced_events',Path(__file__).with_name('audit_faced_raw_events.py'))
events_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(events_module)
base=events_module.base
PUBLIC=base.PUBLIC


def declare():
    base.check_plan()
    base.save(PUBLIC/'numeric_projection_plan.json',{
        'date':'2026-10-11','script_sha256':base.digest(Path(__file__)),
        'event_reader_sha256':base.digest(Path(events_module.__file__)),
        'prior_annotation_joins_sha256':base.digest(PUBLIC/'joins_annotations.json'),
        'reason':'Ten recordings have non-numeric annotation bytes; preserve initial guard failures',
        'projection':'Parse numeric task codes exactly as before. For other annotations retain only byte length, SHA256 and relative time; never decode or export their text',
        'additional_checks':'Count opaque annotations inside each candidate video span; unknown annotation purpose remains unresolved',
        'dummy_channel_samples_decoded':0,'rating_values_decoded':0,'waveform_samples_decoded':0,
        'outcome_access_cleared':False,'research_question_changed':False})


def project_tals(raw):
    events,clocks,opaque=[],[],[]
    for tal in raw.split(b'\0'):
        if not tal:
            continue
        parts=tal.split(b'\x14')
        if len(parts)<2 or parts[-1]!=b'':
            raise ValueError('Invalid TAL termination')
        timing=parts[0].split(b'\x15')
        if len(timing)>2:
            raise ValueError('Invalid TAL time fields')
        onset=float(timing[0]);duration=float(timing[1]) if len(timing)==2 else 0.0
        if not math.isfinite(onset) or not math.isfinite(duration) or duration<0:
            raise ValueError('Invalid annotation time')
        texts=[p for p in parts[1:] if p]
        if not texts:
            clocks.append(onset)
        for text in texts:
            if re.fullmatch(rb'[0-9]{1,4}',text):
                events.append({'onset_seconds':onset,'duration_seconds':duration,'code':int(text)})
            else:
                opaque.append({'onset_seconds':onset,'duration_seconds':duration,
                               'text_bytes':len(text),'text_sha256':hashlib.sha256(text).hexdigest()})
    return events,clocks,opaque


def collect():
    base.check_plan()
    plan=base.load(PUBLIC/'numeric_projection_plan.json')
    if plan['script_sha256']!=base.digest(Path(__file__)) or plan['event_reader_sha256']!=base.digest(Path(events_module.__file__)):
        raise ValueError('Projection declaration changed')
    by_subject={}
    current=[]
    def parse(raw):
        rows,clocks,opaque=project_tals(raw)
        current.extend(opaque)
        return rows,clocks
    def read(path,header):
        current.clear()
        rows,clocks=events_module.read_events(path,header)
        by_subject[path.parent.name.replace('sub','sub-')]=list(current)
        return rows,clocks
    original_save=base.save
    def save(path,value):
        if path.name!='joins.json':
            raise ValueError('Unexpected projection output')
        value['numeric_projection_plan_sha256']=base.digest(PUBLIC/'numeric_projection_plan.json')
        for row in value['records']:
            other=by_subject.get(row['subject'],[])
            row['opaque_annotations']=other
            row['opaque_annotations_inside_video_spans']=sum(any(s['start_seconds']<=r['onset_seconds']<s['end_seconds'] for s in row.get('spans',[])) for r in other)
        original_save(PUBLIC/'joins_numeric_projection.json',value)
    base.parse_tals=parse
    base.read_events=read
    base.save=save
    base.audit_joins()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('declare','collect'))
    args=parser.parse_args()
    declare() if args.mode=='declare' else collect()
