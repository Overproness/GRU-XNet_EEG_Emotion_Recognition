"""Compare authenticated trigger identities to README and behaviour identity fields.

No EEG amplitudes, ratings, learned preprocessing or model outcomes are read.
Duration-based number/symbol correspondence remains an explicitly inferred map.
"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re

from qualify_emo_mc import PRIVATE, PUBLIC, OLD, declared_orders, catalog, sha, write_json


def trigger_mappings(text):
    result={}
    for number, line in re.findall(r'sub-(\d+):([^\n]+)',text):
        entries=re.findall(r'(vid|ima|rating|fade|fede)[-\s]+(\d+)',line)
        result[f'sub-{int(number):02d}']={('fade' if key=='fede' else key):int(value) for key,value in entries}
    return result


def pair_trials(events, mapping):
    out=[]
    for i,event in enumerate(events):
        match=re.fullmatch(r'TypeID:\s*(\d+)',event['description'])
        if not match:continue
        code=int(match.group(1))
        context=next((c for c in ('ima','vid') if mapping[c]==code),None)
        if context is None:continue
        run=event.get('run_path','single_run')
        rating=None; fades=[]
        for later in events[i+1:]:
            if later.get('run_path','single_run')!=run:break
            code_match=re.fullmatch(r'TypeID:\s*(\d+)',later['description'])
            if not code_match:continue
            next_code=int(code_match.group(1))
            if next_code in (mapping['ima'],mapping['vid']):break
            if next_code==mapping['fade']:fades.append(later['onset_s'])
            if next_code==mapping['rating']:
                rating=later;break
        out.append(dict(context=context,run_path=run,start_s=event['onset_s'],
                        rating_start_s=rating['onset_s'] if rating else None,
                        start_to_rating_s=rating['onset_s']-event['onset_s'] if rating else None,
                        fade_events=fades))
    return out


def separate_initial_state_markers(events):
    """Quarantine simultaneous multi-state annotations at an EDF run's origin.

    This preserves single trial markers at time zero and never filters by ratings,
    category balance, trial duration or a desired number of trials.
    """
    groups={}
    for event in events:
        if 0<=event['onset_s']<=0.005:
            key=(event.get('run_path','single_run'),event['onset_s'])
            groups.setdefault(key,[]).append(event)
    ambiguous={key for key,items in groups.items() if len({e['description'] for e in items})>=3}
    excluded=[];kept=[]
    for event in events:
        key=(event.get('run_path','single_run'),event['onset_s'])
        (excluded if key in ambiguous else kept).append(event)
    return kept,excluded


def is_subsequence(expected,observed):
    position=0
    for value in observed:
        if position<len(expected) and expected[position]==value:position+=1
    return position==len(expected)


def numeric_identity(value):
    from decimal import Decimal
    match=re.fullmatch(r'(?:\[\[\s*(\d+(?:\.0+)?)\s*\]\]|(\d+(?:\.0+)?))',value)
    if not match:raise ValueError('Unsupported identity representation')
    return int(Decimal(next(group for group in match.groups() if group is not None)))


def analyze():
    from scipy.optimize import linear_sum_assignment
    text=(OLD/'emo_mc_readme_fixed.response').read_text(encoding='utf-8')
    orders=declared_orders(text);mappings=trigger_mappings(text)
    collection=json.loads((PRIVATE/'collection.json').read_text())
    trials_by_person={};summaries=[]
    for filename,authentication in [('pilot_events.json','pilot_authentication.json'),
                                    ('complete_pilot_events.json','complete_pilot_authentication.json')]:
        person=json.loads((PUBLIC/authentication).read_text())['participant']
        events=json.loads((PRIVATE/filename).read_text())
        raw_starts=pair_trials(events,mappings[person])
        events,excluded=separate_initial_state_markers(events)
        trials=pair_trials(events,mappings[person]);trials_by_person[person]=trials
        contexts=Counter(t['context'] for t in trials)
        summary=dict(participant=person,raw_trial_counts=dict(contexts),
                     unfiltered_start_counts=dict(Counter(t['context'] for t in raw_starts)),
                     quarantined_initial_state_annotations=len(excluded),
                     documented_trial_counts={c:len(orders[person][c]) for c in ('ima','vid')},
                     behaviour_rows=next(len(b['identities']) for b in collection['behaviour'] if b['path'].startswith(person+'/')),
                     unmatched_trial_starts=sum(t['rating_start_s'] is None for t in trials),
                     short_before_rating_guard_counts={c:sum(t['start_to_rating_s'] is not None and t['start_to_rating_s']-1<30 for t in trials if t['context']==c) for c in ('ima','vid')},
                     fade_markers_in_imagery_trials=sum(len(t['fade_events']) for t in trials if t['context']=='ima'))
        summary['context_counts_match_documentation']=summary['raw_trial_counts']==summary['documented_trial_counts']
        summaries.append(summary)
        counters=Counter();counts_match=summary['context_counts_match_documentation']
        for t in trials:
            ordinal=counters[t['context']];counters[t['context']]+=1
            t['readme_symbol']=orders[person][t['context']][ordinal] if counts_match else None
        write_json(PRIVATE/f'{person}_trial_identity_pairs.json',trials)
    complete=json.loads((PUBLIC/'complete_pilot_authentication.json').read_text())['participant']
    trials=trials_by_person[complete]
    b=next(x for x in collection['behaviour'] if x['path'].startswith(complete+'/'))
    lookup={}
    if len(trials)==len(b['identities'])==42 and all(t['readme_symbol'] is not None for t in trials):
        for row,trial in zip(b['identities'],trials):
            lookup[numeric_identity(row.get('video_name_field',row.get('material_id')))]=(trial['context'],trial['readme_symbol'])
    code_checks=[]
    for behaviour in collection['behaviour']:
        person=behaviour['path'].split('/')[0]
        values=[row.get('video_name_field',row.get('material_id')) for row in behaviour['identities']]
        if len(set(values))<=1 or person not in orders:continue
        mapped=[lookup.get(numeric_identity(value)) for value in values]
        contexts={c:[m[1] for m in mapped if m is not None and m[0]==c] for c in ('ima','vid')}
        code_checks.append(dict(participant=person,all_codes_in_lookup=all(m is not None for m in mapped),
                               exact_order_match={c:contexts[c]==orders[person][c] for c in ('ima','vid')},
                               documented_order_is_subsequence={c:is_subsequence(orders[person][c],contexts[c]) for c in ('ima','vid')}))
    # Durations are a metadata consistency check, not a certificate of the materials' content.
    materials=catalog();symbol_numbers=[]
    prefix={'sadness':'sad','disgust':'dis','fear':'fear','neutral':'neu','joy':'joy','tenderness':'ten','inspiration':'ins'}
    for category,pref in prefix.items():
        native=[t for t in trials if t['context']=='vid' and str(t['readme_symbol']).startswith(pref) and t['start_to_rating_s'] is not None]
        described=[m for m in materials if m['context']=='vid' and m['assigned_category']==category]
        if len(native)!=3:continue
        cost=[[abs((t['start_to_rating_s']-1)-m['duration_s']) for m in described] for t in native]
        ti,mi=linear_sum_assignment(cost)
        for i,j in zip(ti,mi):
            symbol_numbers.append(dict(readme_symbol=native[i]['readme_symbol'],spreadsheet_number=described[j]['number'],
                                       duration_difference_after_1s_guard_s=cost[i][j],status='Inferred from one video pilot; not content-authenticated or an imagery mapping'))
    out=dict(pilots=summaries,complete_case_code_lookup_entries=len(lookup),
             coded_behaviour_consistency=code_checks,
             code_join_status='Hypothesis based on ordinal alignment to authenticated triggers and README; confirm independently before accessing targets',
             video_number_mapping_hypotheses=symbol_numbers,
             imagery_number_mapping_verified=False, rating_values_decoded=0,waveform_samples_decoded=0,
             models_fitted=0,fitting_ready=False,
             analysis_script_sha256=sha(Path(__file__).read_bytes()))
    write_json(PUBLIC/'annotation_join_checks.json',out)
    print(json.dumps({'pilot_summaries':summaries,'code_consistency_tables':len(code_checks),
                      'subsequence_failures':sum(not all(c['documented_order_is_subsequence'].values()) for c in code_checks),
                      'video_duration_hypotheses':len(symbol_numbers),'fitting_ready':False}))


if __name__=='__main__':analyze()
