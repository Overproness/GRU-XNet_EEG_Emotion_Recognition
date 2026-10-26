"""Plot already audited aggregates only; never reads original EEG/ratings."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_faced_artifact_control import BASE,load,digest,save

def main():
    source=load(BASE/'source_analysis/summary.json');control=load(BASE/'control_analysis/summary.json')
    if len(source['interpolation_details'])!=8:raise ValueError('Plot layout needs separate review for changed repair count')
    out=BASE/'figures_v2';out.mkdir(exist_ok=True)
    rows=source['interpolation_details'];x=np.arange(len(rows));labels=[f"{r['subject'][-3:]}/{r['vid']}\n{','.join(r['repaired_channels'])}" for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,key,threshold,label in zip(axes,('max_sd_uv','max_peak_to_peak_uv'),(100,1000),('Maximum channel SD (µV)','Maximum channel peak-to-peak (µV)')):
        ax.bar(x-.19,[r['before_'+key] for r in rows],.38,label='Original pipeline',color='#c55c47')
        ax.bar(x+.19,[r['after_'+key] for r in rows],.38,label='Declared interpolation',color='#26769c')
        ax.axhline(threshold,ls='--',color='#555555',lw=1,label='Frozen review threshold')
        ax.set_yscale('log');ax.set_xticks(x,labels,fontsize=8);ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('FACED source-only control: eight flagged trials, all retained\nProcessed 28-second core; passing thresholds does not establish artifact freedom',fontsize=11)
    for ext in ('png','svg'):fig.savefig(out/f'source_amplitude_comparison.{ext}',dpi=170)
    plt.close(fig)
    rows=control['rotations'];fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained',gridspec_kw={'height_ratios':[1.5,1]})
    x=np.arange(3);colors=('#33799f','#cc9c39','#48a079','#82758a')
    for i,(key,label) in enumerate((('context_mae','Context'),('spectral_eeg_mae','Spectral EEG'),('joint_mae','Context + EEG'),('strongest_fixed_prior_mae','Strongest fixed source prior'))):
        axes[0].bar(x+(i-1.5)*.18,[r[key] for r in rows],.18,label=label,color=colors[i])
    axes[0].set_ylim(bottom=0);axes[0].set_ylabel('Participant/film-equal MAE (0–7 valence)')
    axes[0].legend(fontsize=9,loc='upper left',bbox_to_anchor=(1.01,1))
    axes[0].set_xticks(x,[f'Rotation {i}' for i in range(3)]);axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
    gains=np.array([r['joint_gain_over_context'] for r in rows]);intervals=np.array([r['gain_descriptive_95_percent_interval'] for r in rows])
    for i in range(3):
        axes[1].plot([i,i],intervals[i],color=colors[2],lw=2)
        axes[1].scatter(i,gains[i],color=colors[2],s=40,zorder=3)
    axes[1].axhline(0,color='#555555',lw=1);axes[1].axhline(.15,color='#555555',lw=1,ls='--',label='Declared mean-gain lead threshold')
    axes[1].set_xticks(x,[f'Rotation {i}' for i in range(3)]);axes[1].set_ylabel('Context MAE − joint MAE\nPositive values favor EEG');axes[1].grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8);axes[1].set_title('Descriptive crossed-bootstrap intervals; conditional on development tuning',fontsize=10)
    fig.suptitle('FACED development: unseen films and held-out participants\nSame 20 validation people; six films per rotation; no independent confirmation',fontsize=11)
    for ext in ('png','svg'):fig.savefig(out/f'context_eeg_comparison.{ext}',dpi=170)
    plt.close(fig)
    files=[Path(__file__),BASE/'source_analysis/summary.json',BASE/'control_analysis/summary.json',*sorted(out.glob('*.png')),*sorted(out.glob('*.svg'))]
    save(out/'verification.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in files},
        'figure_scope':'Previously audited aggregate QC and exploratory continuous-valence comparisons; no waveforms',
        'original_data_reread':False,'research_question_changed':False})
    print({'plots_created':4,'original_data_opened':False})

if __name__=='__main__':main()
