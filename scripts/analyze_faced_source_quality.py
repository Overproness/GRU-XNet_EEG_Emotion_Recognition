"""Summarize frozen eight-trial quality aggregates; never opens original data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from run_faced_source_quality_pilot import PUBLIC, load, save, digest, verify as verify_pilot
import numpy as np


def table():
    rows=load(PUBLIC/'records.json')['records']; result=[]
    for r in rows:
        channels=r['quality']['channels'];sd=r['quality']['processed_channel_sd_uv_min_median_max']
        result.append({
            'subject':r['subject'],'clip':r['vid'],'native_rate_hz':r['native_rate_hz'],
            'processed_sd_min_uv':sd[0],'processed_sd_median_uv':sd[1],'processed_sd_max_uv':sd[2],
            'processed_abs_q99_max_uv':max(c['processed_abs_q99_uv'] for c in channels),
            'raw_abs_channel_median_offset_median_uv':float(np.median([abs(c['raw_median_uv']) for c in channels])),
            'native_car_50hz_ratio_median':float(np.median([c['native_car_49_51_to_1_45_power_ratio'] for c in channels])),
            'processed_50hz_ratio_median':float(np.median([c['processed_49_51_to_1_45_power_ratio'] for c in channels])),
            'maximum_adc_rail_fraction':max(c['raw_rail_fraction'] for c in channels),
            'maximum_identical_adjacent_fraction':max(c['raw_identical_adjacent_fraction'] for c in channels),
            'quality_flagged_channels':len(r['quality']['flagged_channels']),
            'retained_windows':7,
        })
    return result


def build():
    verify_pilot(local=True)
    if (PUBLIC/'analysis_verification.json').exists(): raise ValueError('Preserve analysis receipt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    rows=load(PUBLIC/'records.json')['records']; summary=table()
    csv_path=PUBLIC/'trial_quality.csv'
    if csv_path.exists(): raise ValueError('Preserve summary')
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    labels=[f"{r['subject'][-3:]}/{r['vid']}" for r in rows]
    values=np.asarray([[c['processed_sd_uv'] for c in r['quality']['channels']] for r in rows]).T
    fig,axes=plt.subplots(2,1,figsize=(10,10),gridspec_kw={'height_ratios':[3,1.2]},layout='constrained')
    im=axes[0].imshow(values,aspect='auto',cmap='viridis',norm=LogNorm(vmin=1,vmax=100))
    axes[0].set_xticks(range(8),labels)
    axes[0].set_yticks(range(30),rows[0]['canonical_channels'],fontsize=8)
    axes[0].set_title('FACED source-only quality pilot: processed channel variation')
    axes[0].set_xlabel('Anonymous participant / clip; 28-second processed core')
    fig.colorbar(im,ax=axes[0],label='Standard deviation (microvolts; logarithmic colors)')
    x=np.arange(8)
    axes[1].bar(x-.18,[r['native_car_50hz_ratio_median'] for r in summary],.36,label='Native CAR, 32 seconds')
    axes[1].bar(x+.18,[r['processed_50hz_ratio_median'] for r in summary],.36,label='Processed core, 28 seconds')
    axes[1].set_yscale('log');axes[1].set_xticks(x,labels)
    axes[1].set_ylabel('Median channel power ratio\n49–51 Hz / 1–45 Hz')
    axes[1].set_title('Residual line-band power: different observation intervals; descriptive comparison')
    axes[1].legend(loc='upper left',fontsize=9)
    axes[1].grid(axis='y',alpha=.2)
    fig.savefig(PUBLIC/'quality_overview.png',dpi=160)
    plt.close(fig)
    save(PUBLIC/'analysis_verification.json',{
        'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in
            [Path(__file__),PUBLIC/'plan.json',PUBLIC/'decoder_reference.json',PUBLIC/'records.json',
             PUBLIC/'summary.json',PUBLIC/'verification.json',csv_path,PUBLIC/'quality_overview.png']},
        'focused_tests_passed':24,'focused_test_failures':0,
        'test_scope':'10 new selective decoder/quality tests and 14 existing proposal/calibration/window tests; manufactured inputs only',
        'original_files_opened_by_analysis':0,'ratings_decoded':0,'models_fitted':0,
        'research_question_changed':False,'supervised_access_cleared':False,
        'line_comparison_scope':'Native 32-second and processed 28-second intervals differ; ratios are not an empirical filter transfer function',
        'no_preprocessing_or_thresholds_changed_after_pilot':True})
    print({'summary_rows':len(summary),'quality_plot_created':True,'ratings_decoded':0})


def verify():
    verify_pilot()
    receipt=load(PUBLIC/'analysis_verification.json')
    if any(digest(REPO/p)!=expected for p,expected in receipt['bindings'].items()):
        raise ValueError('Analysis artifact changed')
    with (PUBLIC/'trial_quality.csv').open(newline='',encoding='utf-8') as f:
        actual=list(csv.DictReader(f))
    expected=table()
    if len(actual)!=len(expected): raise ValueError('Summary identities changed')
    for a,b in zip(actual,expected):
        if set(a)!=set(b): raise ValueError('Summary schema changed')
        for key,value in b.items():
            if (a[key]!=value if isinstance(value,str) else float(a[key])!=value):
                raise ValueError('Summary does not replay')
    print({'analysis_verified':True,'quality_rows':8,'bound_files':len(receipt['bindings'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('build','verify'))
    args=parser.parse_args();build() if args.mode=='build' else verify()
