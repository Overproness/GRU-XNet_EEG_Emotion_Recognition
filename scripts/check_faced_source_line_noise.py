"""One separately declared, source-only notch check after line-band observation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from scipy.signal import welch
from run_faced_source_quality_pilot import (
    PUBLIC,PRIVATE,RAW,QUALIFIED,load,save,digest,module,check as check_base,verify as verify_base)
from gruxnet.faced_source_quality import read_interval, selected_header, quality_metrics
from gruxnet.faced_proposal_preprocessing_v2 import (
    common_indices,calibrate_common_counts,late_segment_bounds,late_trial_windows)
from gruxnet.faced_source_notch import notched_late_trial_windows


def bindings():
    paths=[Path(__file__),REPO/'gruxnet/faced_source_notch.py',REPO/'tests/test_faced_source_notch.py',
           PUBLIC/'plan.json',PUBLIC/'records.json',PUBLIC/'summary.json',PUBLIC/'verification.json']
    return {p.relative_to(REPO).as_posix():digest(p) for p in paths}


def declare():
    verify_base(local=True)
    save(PUBLIC/'line_noise_plan.json',{
        'date':'2026-10-11','bindings':bindings(),'trial_identities':check_base()['trial_identities'],
        'reason':'After the fixed pilot, Pz in sub-098 has processed 49-51/1-45 Hz power ratios 11.342 and 18.493; the initial review flags omit this ratio',
        'status':'Source-quality follow-up after observed pilot values, not preregistered confirmation or original QC criterion',
        'fixed_control':'Native-rate iirnotch(50 Hz,Q=30), forward/backward odd-padding on the same 32-second trial-local scalp segment, then unchanged base preprocessing',
        'settings_searched':0,'additional_people_or_materials':0,'ratings_to_open':0,
        'allowed':'Reauthenticate and decode only the same eight 30-scalp-channel intervals; no behavioral numeric decoding',
        'forbidden':'Numerical auxiliary/neighbor samples, ratings, validation/confirmation EEG, fitting, material reassignment and paper-question change',
        'checks':['Replay every original processed output byte hash and QC aggregate',
            'Report native 40-60 Hz spectral peak and before/after line ratios on the SAME 28-second processed observation',
            'Report 1-30 Hz power changes, all original engineering flags and every trial'],
        'diagnostic_acceptance':'Notched 49-51/1-45 Hz ratio <=0.10 on each channel; no new original engineering flags; per-channel 1-30 Hz power change <=0.05 where original power >1e-6 uV^2',
        'acceptance_scope':'New engineering review requirements for this source follow-up, not psychometric validity or target-performance criteria',
        'primary_proposal_changed':False,'outcome_access_cleared':False,
        'scipy_documentation':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html'})
    print({'line_control_declared':True,'same_source_trials':8,'ratings_to_open':0})


def check():
    verify_base();plan=load(PUBLIC/'line_noise_plan.json')
    if plan['bindings']!=bindings() or plan['trial_identities']!=check_base()['trial_identities']:
        raise ValueError('Follow-up dependencies or permitted identities changed')
    return plan


def run():
    plan=check()
    if (PUBLIC/'line_noise_records.json').exists(): raise ValueError('Preserve line follow-up')
    header_reader=module(REPO/'scripts/audit_faced_raw.py')
    headers={r['subject']:r['data.bdf'] for r in load(QUALIFIED/'headers.json')['records']}
    files={r['member']:r for r in load(QUALIFIED/'extraction.json')['records']}
    original=load(PUBLIC/'records.json')['records'];results=[];seen=set();began=time.monotonic()
    for r,ident in zip(original,plan['trial_identities']):
        subject,vid=r['subject'],r['vid'];h=headers[subject]
        if ident!={'subject':subject,'vid':vid}: raise ValueError('Identity boundary changed')
        path=RAW/subject.replace('sub-','sub')/'data.bdf'
        if subject not in seen:
            if header_reader.read_header(path)!=h or digest(path)!=r['source_file']['original_sha256']:
                raise ValueError('Original signal/header changed')
            for name in ('evt.bdf','After_remarks.mat'):
                if digest(path.parent/name)!=files[f'Data/{path.parent.name}/{name}']['sha256']:
                    raise ValueError('Original identity/marker companion changed')
            seen.add(subject)
        indices=common_indices(h['columns']['labels']);rate=r['native_rate_hz'];start,stop=r['sample_interval']
        if [start,stop]!=list(late_segment_bounds(*r['video_marker_interval_seconds'],rate,
                                               h['record_count']*int(h['columns']['samples_per_record'][0]))):
            raise ValueError('Interval changed')
        with path.open('rb') as f: counts,access=read_interval(f,h,start,stop,indices,True)
        if access['range_ledger_sha256']!=r['access_range_ledger_sha256']:
            raise ValueError('Numerical access expanded')
        values=calibrate_common_counts(counts,subject,selected_header(h,indices))
        base=late_trial_windows(values,rate)
        if quality_metrics(counts,values,base,rate)!=r['quality']:
            raise ValueError('Original QC/processing replay differs')
        notched=notched_late_trial_windows(values,rate)
        q=quality_metrics(counts,values,notched,rate)
        core0=base.transpose(1,0,2).reshape(30,-1)
        core1=notched.transpose(1,0,2).reshape(30,-1)
        f,p0=welch(core0.astype(float),fs=128,nperseg=512,noverlap=256,axis=-1)
        _,p1=welch(core1.astype(float),fs=128,nperseg=512,noverlap=256,axis=-1)
        low=(f>=1)&(f<30);power0=p0[:,low].sum(axis=1)*.25;power1=p1[:,low].sum(axis=1)*.25
        relative=np.abs(power1-power0)/np.maximum(power0,1e-30)
        nf,npower=welch(values-values.mean(axis=0,keepdims=True),fs=rate,nperseg=4*rate,noverlap=2*rate,axis=-1)
        peak=nf[(nf>=40)&(nf<=60)][np.argmax(npower[:,(nf>=40)&(nf<=60)],axis=1)]
        channels=[]
        for i,(before,after) in enumerate(zip(r['quality']['channels'],q['channels'])):
            channels.append({'channel':before['channel'],'native_40_60_peak_hz':float(peak[i]),
                'original_processed_line_ratio':before['processed_49_51_to_1_45_power_ratio'],
                'notched_processed_line_ratio':after['processed_49_51_to_1_45_power_ratio'],
                'original_1_30_power_uv2':float(power0[i]),'notched_1_30_power_uv2':float(power1[i]),
                'relative_1_30_power_change':float(relative[i]),
                'original_sd_uv':before['processed_sd_uv'],'notched_sd_uv':after['processed_sd_uv'],
                'notched_review_flags':after['review_flags']})
        passed=(max(c['notched_processed_line_ratio'] for c in channels)<=.10
                and not q['flagged_channels'] and all(c['relative_1_30_power_change']<=.05
                for c in channels if c['original_1_30_power_uv2']>1e-6))
        results.append({'subject':subject,'vid':vid,'sample_interval':[start,stop],
            'range_ledger_sha256':access['range_ledger_sha256'],'decoded_values':int(counts.size),
            'original_processed_array_sha256':hashlib.sha256(base.tobytes()).hexdigest(),
            'notched_processed_array_sha256':hashlib.sha256(notched.tobytes()).hexdigest(),
            'channels':channels,'engineering_diagnostic_passed':passed,
            'original_quality_replayed_exactly':True,'valid_windows':7,'ratings_decoded':0})
        print({'same_source_trial':[subject,vid],'notch_engineering_check_passed':passed,
               'max_notched_line_ratio':max(c['notched_processed_line_ratio'] for c in channels)},flush=True)
    for subject in seen:
        r=next(r for r in original if r['subject']==subject)
        if digest(RAW/subject.replace('sub-','sub')/'data.bdf')!=r['source_file']['original_sha256']:
            raise ValueError('Original changed during repeated pilot')
    save(PUBLIC/'line_noise_records.json',{'plan_sha256':digest(PUBLIC/'line_noise_plan.json'),
        'records':results,'elapsed_seconds':round(time.monotonic()-began,3),
        'additional_people_or_materials':0,'repeated_real_count_values_decoded':sum(r['decoded_values'] for r in results),
        'ratings_decoded':0,'models_fitted':0,'primary_proposal_changed':False})
    save(PUBLIC/'line_noise_verification.json',{'bindings':{
        p.relative_to(REPO).as_posix():digest(p) for p in
        (Path(__file__),PUBLIC/'line_noise_plan.json',PUBLIC/'line_noise_records.json')},
        'same_source_trials':8,'all_engineering_diagnostics_passed':all(r['engineering_diagnostic_passed'] for r in results),
        'settings_searched':0,'ratings_decoded':0,'models_fitted':0,
        'primary_proposal_changed':False,'outcome_access_cleared':False})


def verify():
    check();receipt=load(PUBLIC/'line_noise_verification.json')
    if any(digest(REPO/p)!=expected for p,expected in receipt['bindings'].items()):
        raise ValueError('Follow-up receipt changed')
    original=load(PUBLIC/'records.json')['records'];follow=load(PUBLIC/'line_noise_records.json')['records']
    if len(follow)!=8 or any((a['subject'],a['vid'],a['sample_interval'],a['access_range_ledger_sha256'])!=
                            (b['subject'],b['vid'],b['sample_interval'],b['range_ledger_sha256'])
                            for a,b in zip(original,follow)):
        raise ValueError('Follow-up expanded the source identity/sample boundary')
    print({'line_followup_verified':True,'same_source_trials':8,
           'all_engineering_diagnostics_passed':receipt['all_engineering_diagnostics_passed'],'ratings_decoded':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','run','verify'))
    args=parser.parse_args();{'declare':declare,'run':run,'verify':verify}[args.mode]()
