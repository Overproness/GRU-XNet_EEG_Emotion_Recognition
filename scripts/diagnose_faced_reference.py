"""Separately frozen three-trial source reference diagnosis, with ratings sealed."""
import argparse
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
import numpy as np
from gruxnet.faced_reference_diagnostic import reference_decomposition
from gruxnet.faced_source_quality import read_interval, selected_header
from gruxnet.faced_proposal_preprocessing import COMMON_CHANNELS, common_indices, calibrate_common_counts
from scripts.run_faced_classical import PUBLIC, PRIVATE, RAW, RAW_EVIDENCE, check, load, digest, save, array_digest
from scripts.audit_faced_raw import read_header

OUTPUT = PUBLIC/'reference_diagnosis'
LOCAL = PRIVATE/'reference_diagnosis'


def selection():
    check(); gate = load(PUBLIC/'source_gate.json'); execution = load(PUBLIC/'execution.json')
    if gate['passed'] or execution['status'] != 'held_by_source_quality' or execution['original_ratings_decoded'] or execution['original_ridge_fits']:
        raise PermissionError('This diagnosis is only for the outcome-sealed source hold')
    rows = load(PUBLIC/'source_records.json')['records']
    flagged = sorted([r for r in rows if r['review_flags']], key=lambda r: (-r['maximum_processed_peak_to_peak_uv'],r['subject'],r['vid']))[:2]
    if len(flagged) != 2: raise ValueError('Need two separately recorded flagged sources')
    controls = sorted([r for r in rows if r['subject'] == flagged[0]['subject'] and not r['review_flags']], key=lambda r:r['vid'])
    if not controls: raise ValueError('No matched unflagged source control; no role replacement')
    result = []
    for role, row in [('flagged_highest_excursion',flagged[0]),('flagged_second_excursion',flagged[1]),('same_person_unflagged_control',controls[0])]:
        stem = f"{row['subject']}_clip{row['vid']:02d}"
        quality_path = PRIVATE/(stem+'_quality.json')
        if digest(quality_path) != row['quality_file_sha256']: raise ValueError('Source quality cache changed')
        quality = load(quality_path)
        dominant = int(np.argmax([c['raw_sd_uv'] for c in quality['channels']]))
        result.append({'subject':row['subject'],'vid':row['vid'],'selection_role':role,
                       'dominant_channel_index':dominant,'dominant_channel':COMMON_CHANNELS[dominant],
                       'source_record':row})
    return result


def bindings():
    paths = [Path(__file__), REPO/'gruxnet/faced_reference_diagnostic.py', REPO/'tests/test_faced_reference_diagnostic.py',
             PUBLIC/'activation_plan.json',PUBLIC/'execution.json', PUBLIC/'source_records.json',PUBLIC/'source_gate.json',
             PUBLIC/'source_analysis/verification.json']
    return {p.relative_to(REPO).as_posix():digest(p) for p in paths}


def declare():
    save(OUTPUT/'plan.json', {'date':'2026-10-11','bindings':bindings(),'trials':selection(),
        'selection':'Two flagged source trials with highest processed channel peak-to-peak; lowest-vid unflagged source trial in the highest-excursion participant. Channel = largest original native raw SD, before any new reads.',
        'allowed':'Re-decode only the same three qualified 30-scalp late32s intervals, authenticate exact prior counts/windows; decompose one channel through the unchanged linear reference/filter.',
        'counterfactual':'Contribution of isolated dominant channel plus contribution of all others; not a cleaned trial, dropped channel, interpolation, exclusion or alternative modeling input.',
        'report':'Anonymous seven-window SD and whole-core SD aggregates, numerical linearity discrepancy; no raw waveform arrays or waveform plots.',
        'time_cap_seconds':300,'original_ratings_to_open':0,'models_to_fit':0,
        'validation_confirmation_access':False,'source_gate_remains_held':True,'research_question_changed':False})
    print({'source_only_diagnosis_declared':True,'source_trials':3,'original_ratings_to_open':0})


def run():
    check(); plan = load(OUTPUT/'plan.json')
    if plan['bindings'] != bindings() or plan['trials'] != selection(): raise ValueError('Source diagnosis declaration changed')
    if (OUTPUT/'records.json').exists(): raise ValueError('Preserve earlier diagnosis')
    headers = {r['subject']:r['data.bdf'] for r in load(RAW_EVIDENCE/'headers.json')['records']}
    began = time.monotonic(); records=[]; authenticated={}
    for trial in plan['trials']:
        if time.monotonic()-began > plan['time_cap_seconds']: raise TimeoutError('Source-only diagnosis cap reached')
        subject,vid=trial['subject'],trial['vid']; row=trial['source_record']; h=headers[subject]
        path=RAW/subject.replace('sub-','sub')/'data.bdf'
        if subject not in authenticated:
            if read_header(path)!=h or digest(path)!=row['original_data_sha256']: raise ValueError('Source bytes/header changed')
            authenticated[subject]=row['original_data_sha256']
        channels=common_indices(h['columns']['labels']); start,stop=row['sample_interval']
        with path.open('rb') as stream: counts,access=read_interval(stream,h,start,stop,channels,True)
        if array_digest(counts)!=row['count_array_sha256'] or access['range_ledger_sha256']!=row['range_ledger_sha256']:
            raise ValueError('Repeated source counts/access differ from frozen execution')
        values=calibrate_common_counts(counts,subject,selected_header(h,channels))
        observed,metrics=reference_decomposition(values,row['native_rate_hz'],trial['dominant_channel_index'])
        if array_digest(observed)!=row['processed_window_sha256']: raise ValueError('Original preprocessing replay differs')
        save(LOCAL/f'{subject}_clip{vid:02d}_access.json',access)
        records.append({'subject':subject,'vid':vid,'selection_role':trial['selection_role'],
            'dominant_channel':trial['dominant_channel'],'sample_interval':row['sample_interval'],
            'repeated_sample_values_decoded_and_scalar_checked':int(counts.size),
            'counts_and_processed_windows_match_frozen_execution':True,
            'access_file_sha256':digest(LOCAL/f'{subject}_clip{vid:02d}_access.json'), **metrics})
    for subject,expected in authenticated.items():
        if digest(RAW/subject.replace('sub-','sub')/'data.bdf')!=expected: raise ValueError('Source bytes changed after diagnosis')
    save(OUTPUT/'records.json',{'records':records,'elapsed_seconds':round(time.monotonic()-began,3),
        'original_ratings_decoded':0,'models_fitted':0,'raw_arrays_persisted':0,
        'source_gate_remains_held':True,'research_question_changed':False})
    # Only window/channel aggregate SD is plotted; no EEG waveform is published.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),constrained_layout=True)
    for ax,row in zip(axes,records):
        data=row['window_median_other_channel_sd_uv']
        for label,color in [('observed','#245b8f'),('dominant_channel_contribution','#b74a36'),('remainder','#38805a')]:
            ax.plot(np.arange(1,8),data[label],marker='o',label=label.replace('_',' '),color=color)
        ax.set_title(f"{row['subject']}, clip {row['vid']}, dominant native-SD channel {row['dominant_channel']}")
        ax.set_ylabel('Median SD, other29 (µV)'); ax.set_xticks(np.arange(1,8)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[-1].set_xlabel('Four-second window in the declared late core')
    fig.suptitle('Source-only reference decomposition — diagnostic counterfactual, no trial repair',fontsize=12)
    fig.savefig(OUTPUT/'window_sd_diagnosis.png',dpi=170);fig.savefig(OUTPUT/'window_sd_diagnosis.svg');plt.close(fig)
    files=[OUTPUT/name for name in ('plan.json','records.json','window_sd_diagnosis.png','window_sd_diagnosis.svg')]
    save(OUTPUT/'verification.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in files},
        'source_intervals_replayed':3,'protected_outcomes_opened':0,'source_gate_remains_held':True})
    print({'source_only_diagnosis_completed':True,'records':[(r['subject'],r['vid'],r['dominant_channel'],r['core_median_other_channel_sd_uv']) for r in records]})


def verify():
    check(); plan=load(OUTPUT/'plan.json')
    if plan['bindings']!=bindings() or plan['trials']!=selection(): raise ValueError('Diagnosis implementation/identity changed')
    for p,expected in load(OUTPUT/'verification.json')['bindings'].items():
        if digest(REPO/p)!=expected: raise ValueError('Diagnosis packet changed')
    rows=load(OUTPUT/'records.json')['records']
    for r in rows:
        if digest(LOCAL/f"{r['subject']}_clip{r['vid']:02d}_access.json")!=r['access_file_sha256']: raise ValueError('Diagnosis access ledger changed')
    print({'source_diagnosis_packet_verified':True,'raw_or_rating_reread':False,'source_gate_remains_held':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','run','verify'))
    mode=parser.parse_args().mode
    if mode=='declare':declare()
    elif mode=='run':run()
    else:verify()
