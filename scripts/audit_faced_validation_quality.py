"""Replay all declared validation QC/access caches without original files or targets."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_artifact_control import BASE,PRIVATE,RAW_EVIDENCE,check,load,digest,save
from scripts.analyze_faced_classical_source import engineering_flags,check_access
from gruxnet.faced_artifact_control import RULES,COMMON_CHANNELS,qualified_weights

def main():
    plan,activation,source,expected=check();rows=load(BASE/'validation_records.json')['records']
    if len(rows)!=280 or {(r['subject'],r['vid']) for r in rows}!=set(expected):raise ValueError('Incomplete/changed validation geometry')
    headers={r['subject']:r['data.bdf'] for r in load(RAW_EVIDENCE/'headers.json')['records']}
    repaired=[];flagged=[];range_count=0;cache_count=0;rules=Counter()
    for r in rows:
        if not r['technical_passed']:raise ValueError('Numerically unavailable validation needs a separate diagnosis')
        stem=f"{r['subject']}_clip{r['vid']:02d}"
        for suffix,key in (('_access.json','access_ledger_file_sha256'),('_quality.json','quality_file_sha256'),('_artifact.json','artifact_file_sha256'),('_features.npz','feature_cache_sha256')):
            if digest(PRIVATE/(stem+suffix))!=r[key]:raise ValueError('Validation cache changed')
            cache_count+=1
        range_count+=check_access(r,load(PRIVATE/(stem+'_access.json')),headers[r['subject']])
        quality=load(PRIVATE/(stem+'_quality.json'));artifact=load(PRIVATE/(stem+'_artifact.json'))
        bad=[i for i,(s,p) in enumerate(zip(artifact['pre_car_sd_uv'],artifact['pre_car_peak_to_peak_uv']))
            if s>RULES['pre_car_core_sd_above_uv'] or p>RULES['pre_car_core_peak_to_peak_above_uv']]
        if bad!=artifact['detected_indices'] or [COMMON_CHANNELS[i] for i in bad]!=r['detected_channels']:raise ValueError('Validation fixed detector differs')
        if len(bad)>3:raise ValueError('Capacity hold needs separate analysis')
        if bad:
            w,_,_=qualified_weights(tuple(bad))
            if not np.array_equal(w,np.asarray(artifact['interpolation_weights'])):raise ValueError('Validation repair weights differ')
            repaired.append({'subject':r['subject'],'vid':r['vid'],'channels':r['repaired_channels']})
        derived=[];details=[]
        for c in quality['channels']:
            f=engineering_flags(c)
            if f!=c['review_flags']:raise ValueError('Validation channel flags differ')
            for flag in f:
                rules[flag]+=1;derived.append({'channel':c['channel'],'flag':flag})
                details.append({'channel':c['channel'],'flag':flag,'raw_rail_fraction':c['raw_rail_fraction'],
                    'raw_identical_adjacent_fraction':c['raw_identical_adjacent_fraction'],
                    'processed_sd_uv':c['processed_sd_uv'],'processed_peak_to_peak_uv':c['processed_peak_to_peak_uv']})
        if derived!=r['review_flags']:raise ValueError('Public validation flags differ')
        if details:flagged.append({'subject':r['subject'],'vid':r['vid'],'details':details})
    receipt={'validation_trials':280,'participants':20,'windows_retained':1960,'trials_or_windows_dropped':0,
        'flagged_trials':len(flagged),'flagged_participants':sorted({r['subject'] for r in flagged}),
        'rule_event_counts':dict(rules),'flagged_trial_details':flagged,'interpolated_trial_details':repaired,
        'interpolated_trials':len(repaired),'channel_trial_interpolations':sum(len(r['channels']) for r in repaired),
        'selected_ranges_reconstructed':range_count,'private_cache_objects_verified':cache_count,
        'numerical_eeg_unavailable_trials':0,'quality_flags_used_to_drop_or_retune':False,
        'original_data_or_rating_values_reread':False,'new_fits':0,'research_question_changed':False}
    out=BASE/'validation_analysis';files=[Path(__file__),BASE/'validation_records.json',BASE/'activation_plan.json']
    if (out/'summary.json').exists():
        if load(out/'summary.json')!=receipt:raise ValueError('Immutable validation analysis differs')
        for rel,h in load(out/'verification.json')['bindings'].items():
            if digest(REPO/rel)!=h:raise ValueError('Validation analysis bindings changed')
    else:
        save(out/'summary.json',receipt);files.append(out/'summary.json')
        save(out/'verification.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in files},
            'quality_rules_recomputed':280*30*8,'private_caches_verified':cache_count,'new_fits':0,'original_files_opened':False})
    print({k:receipt[k] for k in ('validation_trials','flagged_trials','flagged_participants','rule_event_counts','interpolated_trials','selected_ranges_reconstructed')})

if __name__=='__main__':main()
