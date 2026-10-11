"""Source-only diagnosis of exact interpolation replay; no targets or fitting."""
import argparse
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_artifact_control import (load,save,digest,array_digest,BASE as OLD,PRIVATE as OLD_PRIVATE,
    RAW,RAW_EVIDENCE,read_header,read_interval,selected_header,calibrate_common_counts,spectral_features)
from gruxnet.faced_artifact_control import artifact_trial_windows,qualified_weights,geometry
from gruxnet.faced_source_notch import notched_late_trial_windows
BASE=REPO/'results/development/faced_representation_2026-10-11'
PUBLIC=BASE/'replay_diagnosis'


def pools():return [{k:r[k] for k in ('user_api','internal_api','num_threads','prefix','version')} for r in threadpool_info()]


def declare():
    records=[r for r in load(OLD/'source_records.json')['records'] if r['artifact_decision']=='fixed_spherical_interpolation_before_CAR']
    failed=load(BASE/'execution.json')
    if len(records)!=8 or failed['new_fits'] or failed['validation_trials_prepared'] or failed['failure']['message']!='Processed windows changed':
        raise PermissionError('Expected source-only replay hold not present')
    paths=[Path(__file__),BASE/'plan.json',BASE/'execution.json',BASE/'feature_manifest.json',
        OLD/'source_records.json',OLD/'activation_plan.json',REPO/'gruxnet/faced_artifact_control.py',
        REPO/'gruxnet/faced_source_notch.py',RAW_EVIDENCE/'headers.json',RAW_EVIDENCE/'extraction.json']
    save(PUBLIC/'plan.json',{'date':'2026-10-11','bindings':{p.relative_to(REPO).as_posix():digest(p) for p in paths},
        'source_pairs':[[r['subject'],r['vid']] for r in records],
        'thread_pools':pools(),'scope':'Exactly eight previously declared interpolated source intervals, counts scalar checked once per trial; compare capped2/default preprocessing and prior literal interpolation weights. No new algorithms, targets, validation, fitting or outcome-based choice.',
        'no_implicit_resume':True,'confirmation_access':False,'research_question_changed':False})
    print({'source_only_diagnosis_declared':True,'trials':8,'fits':0})


def execute():
    p=load(PUBLIC/'plan.json')
    for rel,h in p['bindings'].items():
        if digest(REPO/rel)!=h:raise ValueError('Bound diagnosis dependency changed')
    if pools()!=p['thread_pools']:raise ValueError('Default thread runtime changed')
    originals={(r['subject'],r['vid']):r for r in load(OLD/'source_records.json')['records']}
    headers={r['subject']:r['data.bdf'] for r in load(RAW_EVIDENCE/'headers.json')['records']}
    rows=[];authenticated=set();values_decoded=0
    for pair in p['source_pairs']:
        person,vid=pair;r=originals[person,vid];h=headers[person]
        if r['role']!='source' or r['artifact_decision']!='fixed_spherical_interpolation_before_CAR':raise PermissionError('Forbidden diagnosis identity')
        path=RAW/person.replace('sub-','sub')/'data.bdf'
        if person not in authenticated:
            if read_header(path)!=h or digest(path)!=r['original_data_sha256']:raise ValueError('Original changed')
            authenticated.add(person)
        with path.open('rb') as stream:counts,access=read_interval(stream,h,*r['sample_interval'],r['channel_indices'],True)
        values_decoded+=counts.size
        if array_digest(counts)!=r['count_array_sha256'] or access['range_ledger_sha256']!=r['range_ledger_sha256']:raise ValueError('Original interval/counts changed')
        values=calibrate_common_counts(counts,person,selected_header(h,r['channel_indices']))
        original_artifact=load(OLD_PRIVATE/f'{person}_clip{vid:02d}_artifact.json')
        if digest(OLD_PRIVATE/f'{person}_clip{vid:02d}_artifact.json')!=r['artifact_file_sha256']:raise ValueError('Prior artifact changed')
        geometry();variants={};windows_by_variant={}
        for name,limit in (('capped_two',2),('original_default',None)):
            qualified_weights.cache_clear()
            with threadpool_limits(limits=limit):
                windows,artifact=artifact_trial_windows(values,r['native_rate_hz'])
            windows_by_variant[name]=windows
            variants[name]={'window_sha256':array_digest(windows),'matches_prior_window_bytes':array_digest(windows)==r['processed_window_sha256'],
                'detected_channels_match':artifact['detected_channels']==r['detected_channels'],
                'maximum_weight_difference':float(np.max(np.abs(np.asarray(artifact['interpolation_weights'])-original_artifact['interpolation_weights']))),
                'spectral_hash_matches':array_digest(spectral_features(windows))==r['spectral_feature_sha256']}
        bad=original_artifact['detected_indices'];good=[i for i in range(30) if i not in bad]
        with threadpool_limits(limits=2):
            repaired=values.copy();repaired[bad]=np.asarray(original_artifact['interpolation_weights'])@values[good]
            windows=notched_late_trial_windows(repaired,r['native_rate_hz'])
        variants['prior_literal_weights_capped_two']={'window_sha256':array_digest(windows),
            'matches_prior_window_bytes':array_digest(windows)==r['processed_window_sha256'],
            'spectral_hash_matches':array_digest(spectral_features(windows))==r['spectral_feature_sha256']}
        a=windows_by_variant['capped_two'];b=windows_by_variant['original_default']
        rows.append({'subject':person,'vid':vid,'samples_decoded':int(counts.size),'variants':variants,
            'capped_vs_default_differing_float32_values':int(np.count_nonzero(a!=b)),
            'capped_vs_default_maximum_absolute_difference_uv':float(np.max(np.abs(a.astype(float)-b.astype(float))))})
    for person in authenticated:
        record=next(r for r in originals.values() if r['subject']==person)
        if digest(RAW/person.replace('sub-','sub')/'data.bdf')!=record['original_data_sha256']:raise ValueError('Original changed after diagnosis')
    save(PUBLIC/'comparison.json',{'records':rows,'source_trials':8,'selected_sample_values_decoded':int(values_decoded),
        'ratings_decoded':0,'validation_trials_decoded':0,'fits':0,'confirmation_access':False,'research_question_changed':False})
    print({'source_trials':8,'all_default_match':all(r['variants']['original_default']['matches_prior_window_bytes'] for r in rows),
        'all_literal_two_match':all(r['variants']['prior_literal_weights_capped_two']['matches_prior_window_bytes'] for r in rows),
        'capped_two_mismatches':sum(not r['variants']['capped_two']['matches_prior_window_bytes'] for r in rows),'fits':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('declare','execute'))
    {'declare':declare,'execute':execute}[p.parse_args().mode]()
