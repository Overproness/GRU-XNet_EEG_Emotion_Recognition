"""Plot completed fixed FACED diagnostics; no features, state selection or fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
PUBLIC=REPO/'results/development/faced_representation_v2_2026-10-11'
load=lambda name:json.loads((PUBLIC/name).read_text(encoding='utf-8'))
if load('execution.json')['status']!='completed_bounded_study':raise ValueError('Completed experiment required')
comparison=load('comparison.json')
names=['covariance_linear','context_covariance_linear','context_kernel','spectral_kernel',
       'context_spectral_kernel','covariance_kernel','context_covariance_kernel']
labels=['Covariance, Ridge','Context + covariance, Ridge','Context, RBF','Band power, RBF',
        'Context + band power, RBF','Covariance, RBF','Context + covariance, RBF']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(left,right)=plt.subplots(1,2,figsize=(16,7.4),gridspec_kw={'width_ratios':[1.15,1]})
y=np.arange(len(names));width=.23;coordinates={}
colors=['#2563a3','#da8636','#538b57']
for j,(arm,label) in enumerate([('source_fit','Source training'),('source_oof','Held-out source participants'),('validation_unseen','Validation: unseen films')]):
    values=[float(np.mean([r['models'][name]['metrics'][arm]['participant_film_mae'] for r in comparison['rotations']])) for name in names]
    left.barh(y+(j-1)*width,values,height=width,color=colors[j],label=label)
    coordinates[arm]=values
left.set_yticks(y,labels);left.invert_yaxis();left.set_xlabel('Mean participant/film-equal MAE across rotations')
left.set_title('Source fit and generalization',loc='left',fontweight='bold');left.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=1,frameon=False)
left.grid(axis='x',alpha=.18);left.set_axisbelow(True)
joint_names=['context_covariance_linear','context_spectral_kernel','context_covariance_kernel']
joint_labels=['Context + covariance, Ridge','Context + band power, RBF','Context + covariance, RBF']
paired=[]
for i,name in enumerate(joint_names):
    for k,table in enumerate(comparison['rotations']):
        entry=table['models'][name]['paired'];gain=entry['gain'];low,high=entry['descriptive_95_percent_interval'];position=i*4+k
        right.errorbar(gain,position,xerr=np.asarray([[gain-low],[high-gain]]),fmt='o',capsize=3,color=colors[k],label=f'Rotation {k}' if i==0 else None)
        paired.append({'model':name,'rotation':k,'gain':gain,'interval':[low,high]})
right.set_yticks([1,5,9],joint_labels);right.invert_yaxis();right.axvline(0,color='#555555',linewidth=1)
right.axvline(.15,color='#777777',linestyle='--',linewidth=1,label='Earlier lead threshold')
right.set_xlabel('MAE gain over matched context (positive is better)')
right.set_title('Exploratory EEG gain and descriptive intervals',loc='left',fontweight='bold')
right.grid(axis='x',alpha=.18);right.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,frameon=False)
fig.suptitle('FACED: fixed representations, source-only fits and quality fallback',fontsize=16,fontweight='bold')
fig.text(.5,.01,'Reused development: 3 rotations, the same 20 validation people and 6 unseen films per rotation. All 280 validation trials retained; 15 bypass EEG.',ha='center',fontsize=10)
fig.subplots_adjust(left=.20,right=.985,top=.89,bottom=.23,wspace=.62)
out=PUBLIC/'figures';out.mkdir(exist_ok=True)
for ext in ('png','svg'):
    path=out/f'representation_comparison.{ext}'
    if path.exists():raise ValueError('Preserve prior figure')
    fig.savefig(path,dpi=170,bbox_inches='tight')
with (out/'coordinates.json').open('x',encoding='utf-8') as stream:
    json.dump({'models':names,'error_coordinates':coordinates,'paired_coordinates':paired,
               'source':'comparison.json; no new fitting or selection'},stream,indent=2);stream.write('\n')
plt.close(fig)
print({'figure_written':True,'metric_coordinates':21,'paired_gain_intervals':9})
