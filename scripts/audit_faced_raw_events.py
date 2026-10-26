"""Bounded annotation-only continuation: seek past FACED's empty event channel."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('faced_raw',Path(__file__).with_name('audit_faced_raw.py'))
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PUBLIC=base.PUBLIC


def declare():
    base.check_plan()
    base.save(PUBLIC/'annotation_channel_plan.json',{
        'date':'2026-10-11','script_sha256':base.digest(Path(__file__)),
        'initial_joins_sha256':base.digest(PUBLIC/'joins.json'),
        'headers_sha256':base.digest(PUBLIC/'headers.json'),
        'initial_failure':'All evt.bdf files contain Empty Event Data plus BDF Annotations; original all-annotation guard refused before reading either channel',
        'allowed_change':'Validate exact two labels/sample counts, seek past the three-byte empty channel in every record; parse BDF Annotations only',
        'dummy_channel_samples_decoded':0,'rating_values_decoded':0,'waveform_samples_decoded':0,
        'outcome_access_cleared':False,'research_question_changed':False})


def read_events(path,header):
    c=header['columns']
    if header['sample_width']!=3 or c['labels']!=['Empty Event Data','BDF Annotations'] or c['samples_per_record']!=['1','100']:
        raise ValueError('Event channel geometry changed')
    events,clocks=[],[]
    with path.open('rb') as f:
        f.seek(header['header_bytes'])
        for _ in range(header['record_count']):
            f.seek(3,1)
            rows,times=base.parse_tals(f.read(300))
            events.extend(rows);clocks.extend(times)
    if events!=sorted(events,key=lambda r:r['onset_seconds']):
        raise ValueError('Event times not monotonic')
    return events,clocks


def collect():
    base.check_plan()
    plan=base.load(PUBLIC/'annotation_channel_plan.json')
    if plan['script_sha256']!=base.digest(Path(__file__)) or plan['headers_sha256']!=base.digest(PUBLIC/'headers.json'):
        raise ValueError('Annotation declaration changed')
    # Reuse audited join logic in memory, preserve original failed collection bytes.
    original_save=base.save
    def save(path,value):
        if path.name!='joins.json':
            raise ValueError('Unexpected continuation output')
        value['annotation_channel_plan_sha256']=base.digest(PUBLIC/'annotation_channel_plan.json')
        original_save(PUBLIC/'joins_annotations.json',value)
    base.read_events=read_events
    base.save=save
    base.audit_joins()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('declare','collect'))
    args=parser.parse_args()
    declare() if args.mode=='declare' else collect()
