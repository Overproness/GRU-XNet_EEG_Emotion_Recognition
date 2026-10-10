"""Technical metadata consistency checks, with all calendar timestamps kept private."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from qualify_emo_mc import PRIVATE, PUBLIC, parse_edf_header, sha, write_json


def analyze():
    collection=json.loads((PRIVATE/'collection.json').read_text())
    sidecars={s['path'].split('/')[0]:s for s in collection['sidecars']}
    records=defaultdict(list);mismatches=[]
    for r in collection['recordings']:
        side=sidecars[r['participant']]['metadata']
        if r['channels'][0]['sampling_hz']!=side['SamplingFrequency']:
            mismatches.append(dict(path=r['path'],edf_hz=r['channels'][0]['sampling_hz'],sidecar_hz=side['SamplingFrequency']))
        records[r['participant']].append(r)
    clocks=[]
    for person,runs in sorted(records.items()):
        if len(runs)<2:continue
        previous=None
        for r in sorted(runs,key=lambda r:r['path']):
            body=PRIVATE.joinpath('headers',r['path']).with_suffix('.header').read_bytes()
            start=datetime.strptime((body[168:176]+b' '+body[176:184]).decode(),'%d.%m.%y %H.%M.%S')
            duration=r['records']*r['record_duration_s']
            if previous:
                delta=(start-previous[0]).total_seconds()
                clocks.append(dict(participant=person,start_to_start_s=delta,
                                   preceding_declared_duration_s=previous[1],header_gap_s=delta-previous[1]))
            previous=(start,duration)
    out=dict(sidecar_sampling_discrepancies=mismatches,split_participant_clock_checks=clocks,
             overlapping_header_intervals=sum(c['header_gap_s']<0 for c in clocks),
             interpretation='Header-clock overlap does not prove duplicate signal data. Do not concatenate runs or certify chronological annotation joins from filenames/header clocks alone.',
             edf_format_counts=dict(Counter(r['format'] for r in collection['recordings'])),
             eeg_channel_orders=len({tuple(c['label'] for c in r['channels'] if c['label']!='EDF Annotations') for r in collection['recordings']}),
             reference_descriptions=sorted({s['metadata'].get('EEGReference') for s in collection['sidecars']}),
             prefilter_header_nonempty_eeg_fields=sum(bool(c['prefilter']) for r in collection['recordings'] for c in r['channels'] if c['label']!='EDF Annotations'),
             score_values_decoded=0,waveform_samples_decoded=0,models_fitted=0,
             script_sha256=sha(Path(__file__).read_bytes()))
    write_json(PUBLIC/'technical_consistency.json',out)
    print(json.dumps({'split_clock_checks':len(clocks),'overlaps':out['overlapping_header_intervals'],'sampling_mismatches':mismatches}))


if __name__=='__main__':analyze()
