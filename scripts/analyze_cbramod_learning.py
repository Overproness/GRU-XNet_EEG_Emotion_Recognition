"""Public-only reanalysis of complete source learning controls and descriptive contrasts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.analyze_cbramod_adaptation import collect as previous_collect
STUDY='cbramod_learning_2026-10-09'
PRIOR='cbramod_adaptation_2026-10-09'
OLDER='cbramod_source_probe_2026-10-09'
ROLES=('train','validation_unseen','validation_familiar')
METRICS=('balanced_accuracy','balanced_log_loss')
EXPECTED_STEPS={'head':[0,200,800,2400],'tiny':[0,200,400],'long':[0,200,600,1200]}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(65536),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def criterion(items):
    return min(range(len(items)),key=lambda k:(sum(items[k]['metrics'][r]['balanced_log_loss'] for r in ROLES[1:])/2,
        -sum(items[k]['metrics'][r]['balanced_accuracy'] for r in ROLES[1:])/2,k))


def reconstruct_exposure(table,record,kind):
    steps=EXPECTED_STEPS[kind];rng=np.random.default_rng(20261009);window_rng=np.random.default_rng(20261010)
    y=table.label.to_numpy(dtype=int);classes=int(y.max()+1);pools=[np.flatnonzero(y==c) for c in range(classes)]
    draws=np.array([rng.permutation(np.concatenate([rng.choice(p,6//classes,replace=True) for p in pools]))
                    for _ in range(steps[-1])])
    windows=window_rng.integers(0,4,size=draws.shape);items=[]
    for step in steps[1:]:
        counts=np.bincount(draws[:step].reshape(-1),minlength=len(table))
        item={'step':step,'training_observations':len(table),'draws':6*step,
            'unique_observations_seen':int(np.count_nonzero(counts)),
            'minimum_draws_per_observation':int(counts.min()),'maximum_draws_per_observation':int(counts.max())}
        if kind=='long':
            pairs=np.column_stack((draws[:step].reshape(-1),windows[:step].reshape(-1)))
            item.update({'unique_observation_window_pairs':len(np.unique(pairs,axis=0)),
                         'available_observation_window_pairs':4*len(table)})
        items.append(item)
    if items!=record['exposure']:raise ValueError('Public reconstructed exposure mismatch')
    return items


def collect(public):
    proof=read(public/'verification.json');summary=read(public/'summary.json');plan=read(public/'plan.json')
    if not proof['complete'] or sha(public/'summary.json')!=proof['summary_sha256'] or sha(public/'plan.json')!=proof['plan_sha256']:
        raise ValueError('Complete verified source-learning study required')
    if not summary['development_only'] or summary['outer_test_inferences']!=0 or summary['research_question_change_approved']:
        raise ValueError('Unexpected study scope')
    for name,value in plan['source_sha256'].items():
        if sha(REPO/name)!=value:raise ValueError('Changed frozen source: '+name)
    panels={};rows=[];records={};maximum=0.;exposure=[]
    for kind in EXPECTED_STEPS:
        records[kind]=[]
        for folder in sorted((public/kind).iterdir()):
            record=read(folder/'record.json');audit=read(folder/'verification.json');job=record['job']
            if not audit['complete'] or sha(folder/'record.json')!=audit['record_sha256']:raise ValueError('Stale case verification')
            if record['plan_sha256']!=sha(public/'plan.json'):raise ValueError('Wrong case declaration')
            if [i['step'] for i in record['candidates']]!=EXPECTED_STEPS[kind]:raise ValueError('Missing checkpoint')
            records[kind].append({**record,'trajectory':folder.name});classes=2 if job['dataset']=='DEAP' else 3
            for item in record['candidates']:
                roles=['capacity_train'] if kind=='tiny' else ROLES
                for role in roles:
                    name=f'step{item["step"]}_{role}.csv'
                    if sha(folder/name)!=record['artifact_sha256'][name]:raise ValueError('Changed probability bytes')
                    frame=pd.read_csv(folder/name)
                    allowed={'trial_id','subject_id','material_key','original_label','label','role',*[f'p{c}' for c in range(classes)]}
                    if set(frame.columns)!=allowed or frame.trial_id.duplicated().any() or not(frame.role==role).all():
                        raise ValueError('Unexpected public probability schema')
                    y=frame.label.to_numpy(dtype=int);p=frame[[f'p{c}' for c in range(classes)]].to_numpy(dtype=float)
                    if set(y)!=set(range(classes)) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
                        raise ValueError('Invalid labels/probabilities')
                    np.testing.assert_allclose(p.sum(1),1.,rtol=0,atol=2e-12)
                    scores={'n':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,p.argmax(1))),
                        'balanced_log_loss':float(np.average(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),
                                                            weights=compute_sample_weight('balanced',y)))}
                    for metric,value in scores.items():
                        discrepancy=abs(value-item['metrics'][role][metric]);maximum=max(maximum,discrepancy)
                        if discrepancy>2e-11:raise ValueError('Independent public metric mismatch')
                    key=(job['dataset'],job['group'],role) if kind!='tiny' else (job['dataset'],job['group'],role,job['target'])
                    metadata=frame[['trial_id','subject_id','material_key','original_label','label']]
                    if key in panels:pd.testing.assert_frame_equal(metadata,panels[key])
                    else:panels[key]=metadata
                    rows.append({'kind':kind,**job,'trajectory':folder.name,'step':item['step'],'role':role,**scores,
                                 'predicted_classes':int(len(np.unique(p.argmax(1))))})
            train=panels[(job['dataset'],job['group'],'capacity_train',job['target'])] if kind=='tiny' else panels[(job['dataset'],job['group'],'train')]
            for item in reconstruct_exposure(train,record,kind):exposure.append({'kind':kind,**job,'trajectory':folder.name,**item})
    if [len(records[k]) for k in EXPECTED_STEPS]!=[48,16,16] or len(rows)!=816:raise ValueError('Incomplete grid')
    for dataset in ('DEAP','SEEDIV'):
        for group in (1,2):
            train,unseen,familiar=[panels[(dataset,group,r)] for r in ROLES]
            if set(train.subject_id)&(set(unseen.subject_id)|set(familiar.subject_id)):raise ValueError('Participant overlap')
            if set(train.trial_id)&(set(unseen.trial_id)|set(familiar.trial_id)) or set(unseen.trial_id)&set(familiar.trial_id):
                raise ValueError('Trial overlap')
            if set(train.material_key)&set(unseen.material_key) or not set(familiar.material_key)<=set(train.material_key):
                raise ValueError('Material boundary mismatch')
            # Reconstruct tiny choices from ordered public source training rows.
            y=train.label.to_numpy(dtype=int);classes=int(y.max()+1);rng=np.random.default_rng(20261012)
            chosen=np.concatenate([rng.choice(np.flatnonzero(y==c),12//classes,replace=False) for c in range(classes)])
            chosen=rng.permutation(chosen);windows=np.random.default_rng(20261013).integers(0,4,12)
            for target in ('real','permuted'):
                tiny=panels[(dataset,group,'capacity_train',target)];expected=train.iloc[chosen].reset_index(drop=True).copy()
                if target=='permuted':expected['label']=np.random.default_rng(20261014).permutation(expected.label.to_numpy())
                pd.testing.assert_frame_equal(tiny,expected)
                for record in records['tiny']:
                    job=record['job']
                    if (job['dataset'],job['group'],job['target'])==(dataset,group,target):
                        definition=record['tiny_definition']
                        if definition['windows']!=windows.tolist() or definition['original_task_labels']!=y[chosen].tolist():
                            raise ValueError('Capacity window/original-label mismatch')
                        if definition['trial_ids']!=tiny.trial_id.tolist() or definition['target_labels']!=tiny.label.tolist():
                            raise ValueError('Capacity target metadata mismatch')
    byname={r['trajectory']:r for kind in records for r in records[kind]};selected=[]
    for kind in ('head','long'):
        conditions=summary['selected'][kind]
        if len(conditions)!=16:raise ValueError('Missing selected conditions')
        seen=set()
        for condition in conditions:
            key=tuple(condition['condition'])
            if key in seen:raise ValueError('Duplicate selected condition')
            seen.add(key);choices=[]
            for job in plan['jobs'][kind]:
                actual=(job['dataset'],job['group'],job['pretrained'],job['scaled'] if kind=='head' else job['trainable'])
                if actual!=key:continue
                found=[r for r in records[kind] if r['job']==job]
                if len(found)!=1:raise ValueError('Wrong declared job coverage')
                record=found[0]
                choices.extend({**item,'trajectory':record['trajectory'],'job':job} for item in record['candidates'] if item['step']>0)
            if condition['candidates']!=choices or condition['selected']!=choices[criterion(choices)]:
                raise ValueError('Independent source selection mismatch')
            winner=condition['selected']
            selected.extend({**r,'selection_kind':kind} for r in rows if r['kind']==kind and r['trajectory']==winner['trajectory'] and r['step']==winner['step'])
    capacities=[]
    for record in records['tiny']:
        values=record['candidates'][-1]['metrics']['capacity_train']
        passed=values['balanced_accuracy']>=.95 and values['balanced_log_loss']<=.10
        if record['capacity_pass']!=passed:raise ValueError('Capacity threshold mismatch')
        job=record['job'];capacities.append({**job,'capacity_pass':passed,**values})
        item=[i for i in summary['capacity'] if i['job']==job]
        if len(item)!=1 or item[0]['capacity_pass']!=passed or item[0]['candidates']!=record['candidates']:
            raise ValueError('Capacity summary mismatch')
    if len(selected)!=96 or len(capacities)!=16:raise ValueError('Missing selected/capacity outcomes')
    # Public metadata/state certificates additionally check the paired records.
    for kind in EXPECTED_STEPS:
        for dataset in ('DEAP','SEEDIV'):
            for group in (1,2):
                part=[r for r in records[kind] if (r['job']['dataset'],r['job']['group'])==(dataset,group)]
                if len({r['initial_head_digest'] for r in part})!=1:raise ValueError('Unmatched heads')
                if kind!='tiny' and len({r['sampling_digest'] for r in part})!=1:raise ValueError('Unmatched sampling certificates')
                if kind!='head':
                    for pretrained in (True,False):
                        if len({r['initial_encoder_digest'] for r in part if r['job']['pretrained']==pretrained})!=1:
                            raise ValueError('Unmatched encoders')
    _,_,prior,_,oldmaximum=previous_collect(REPO/'results/development'/PRIOR,REPO/'results/development'/OLDER)
    for role in ROLES:
        for dataset in ('DEAP','SEEDIV'):
            for group in (1,2):
                folder=REPO/'results/development'/PRIOR/'linear'/f'{dataset.lower()}_g{group}_pretrained_average'
                before=pd.read_csv(folder/f'{role}.csv')
                pd.testing.assert_frame_equal(before[['trial_id','subject_id','material_key','original_label','label']],panels[(dataset,group,role)])
    return pd.DataFrame(rows),pd.DataFrame(selected),pd.DataFrame(capacities),pd.DataFrame(exposure),prior,max(maximum,oldmaximum)


def contrasts(rows,selected,prior):
    points=[]
    def add(a,b,name):
        for metric in METRICS:points.append({'comparison':name,'dataset':a.dataset,'group':int(a.group),'role':a.role,
            'model':a.trajectory,'reference':b.trajectory,'metric':metric,'difference':float(a[metric]-b[metric]),'population_interval':None})
    for dataset in ('DEAP','SEEDIV'):
        for group in (1,2):
            for role in ROLES[1:]:
                panel=selected[(selected.dataset==dataset)&(selected.group==group)&(selected.role==role)]
                old=prior[(prior.dataset==dataset)&(prior.group==group)&(prior.role==role)].set_index('model')
                for pretrained in (True,False):
                    name='pretrained' if pretrained else 'random42'
                    heads=panel[(panel.kind=='head')&(panel.pretrained==pretrained)].copy()
                    heads['scaled']=heads['scaled'].astype(bool);heads=heads.set_index('scaled')
                    add(heads.loc[True],heads.loc[False],'head_standardized_minus_raw')
                    for scale in (False,True):add(heads.loc[scale],old.loc[name+'_average'],'head_minus_solved_averaged_logistic')
                    neural=panel[(panel.kind=='long')&(panel.pretrained==pretrained)].copy()
                    neural['trainable']=neural['trainable'].astype(bool);neural=neural.set_index('trainable')
                    add(neural.loc[True],neural.loc[False],'long_finetune_minus_frozen')
                    for trainable in (False,True):
                        a=neural.loc[trainable];oldname=name+('_finetune' if trainable else '_frozen')
                        add(a,old.loc[oldname],'long_selected_minus_prior_small_neural')
                        trajectory=rows[(rows.kind=='long')&(rows.trajectory==a.trajectory)&(rows.role==role)].set_index('step')
                        add(trajectory.loc[1200],trajectory.loc[200],'long_final1200_minus_same_schedule200')
                        for reference in ('band_absolute','band_relative'):add(a,old.loc[reference],'long_selected_minus_spectral')
                for scaled in (False,True):
                    heads=panel[(panel.kind=='head')&(panel.scaled==scaled)].set_index('pretrained')
                    add(heads.loc[True],heads.loc[False],'head_pretrained_minus_random')
                for trainable in (False,True):
                    neural=panel[(panel.kind=='long')&(panel.trainable==trainable)].set_index('pretrained')
                    add(neural.loc[True],neural.loc[False],'long_pretrained_minus_random')
    if len(points)!=448:raise ValueError('Incomplete declared descriptive contrasts')
    return points


def generate(public):
    output=REPO.parent/'publication_runs'/STUDY/'postfit_analysis'
    if output.exists():raise FileExistsError('Preserve post-fit analysis')
    if not read(public/'verification.json')['complete']:raise ValueError('Full grid required')
    output.mkdir(parents=True);shutil.copyfile(public/'export_manifest.json',output/'input_export_manifest.json')
    bindings={p.relative_to(REPO).as_posix():sha(p) for folder in
        (public,REPO/'results/development'/PRIOR,REPO/'results/development'/OLDER)
        for p in folder.rglob('*') if p.is_file() and p.name!='export_manifest.json'}
    write(output/'declaration.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'postfit_descriptive':True,
        'analysis_source_sha256':sha(Path(__file__)),'previous_analyzer_sha256':sha(REPO/'scripts/analyze_cbramod_adaptation.py'),
        'public_input_sha256':bindings,'input_manifest_sha256':sha(output/'input_export_manifest.json'),
        'definition':'Recompute all 816 new and 288 prior public probability metric sets, 32 source selections, 16 capacity outcomes, 20 matched metadata sets, four source boundaries, tiny training-only subsets and all 224 aggregate exposure points. All 448 descriptive loss/BA contrasts: 32 head scaling, 64 head versus logistic, 32 long fine-tuned versus frozen, 64 long versus small neural, 64 same-schedule 1200 versus 200, 128 long versus spectral, 32 head and 32 long pretraining. All groups/settings and failed capacity controls retained. Four scientific plots: capacity outcomes, pretrained/random head curves and longer neural curves. No new fit, outer test, global recipe, confidence interval or approved paper pivot.'})
    rows,selected,capacity,exposure,prior,maximum=collect(public)
    for name,frame in (('all_metrics.csv',rows),('selected_metrics.csv',selected),('capacity_outcomes.csv',capacity),
                       ('training_exposure.csv',exposure),('prior_selected_metrics.csv',prior)):
        frame.to_csv(output/name,index=False)
    write(output/'contrasts.json',{'development_only':True,'contrasts':contrasts(rows,selected,prior)})
    figures(output,rows,capacity)
    proof={'complete':True,'new_probability_metric_sets':816,'prior_probability_metric_sets':288,'selected_metric_sets':96,
        'capacity_outcomes':16,'exposure_points':len(exposure),'descriptive_contrast_points':448,
        'maximum_abs_metric_discrepancy':maximum,'declaration_sha256':sha(output/'declaration.json'),
        'artifact_sha256':{p.name:sha(p) for p in output.iterdir() if p.name not in ('declaration.json','verification.json')}}
    write(output/'verification.json',proof);destination=public/'postfit_analysis';destination.mkdir()
    for p in output.iterdir():shutil.copyfile(p,destination/p.name)
    manifest=read(public/'export_manifest.json')
    for item in manifest['files']:
        if sha(public/item['file'])!=item['sha256']:raise ValueError('Changed finished export')
    manifest['files'] += [{'file':p.relative_to(public).as_posix(),'sha256':sha(p)} for p in sorted(destination.iterdir())]
    manifest['postfit_analysis_source_sha256']=sha(Path(__file__));write(public/'export_manifest.json',manifest)
    print(json.dumps(proof))


def verify(public):
    folder=public/'postfit_analysis';proof=read(folder/'verification.json');declaration=read(folder/'declaration.json')
    if not proof['complete'] or sha(folder/'declaration.json')!=proof['declaration_sha256']:raise ValueError('Changed post-fit declaration')
    if sha(Path(__file__))!=declaration['analysis_source_sha256'] or sha(REPO/'scripts/analyze_cbramod_adaptation.py')!=declaration['previous_analyzer_sha256']:
        raise ValueError('Changed analysis sources')
    if sha(folder/'input_export_manifest.json')!=declaration['input_manifest_sha256']:raise ValueError('Changed input snapshot')
    for name,value in declaration['public_input_sha256'].items():
        if sha(REPO/name)!=value:raise ValueError('Changed public analysis input')
    for name,value in proof['artifact_sha256'].items():
        if sha(folder/name)!=value:raise ValueError('Changed post-fit artifacts')
    manifest=read(public/'export_manifest.json');seen=set()
    for item in manifest['files']:
        if item['file'] in seen or sha(public/item['file'])!=item['sha256']:raise ValueError('Invalid canonical export manifest')
        seen.add(item['file'])
    if seen!={p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file() and p!=public/'export_manifest.json'}:
        raise ValueError('Incomplete canonical export')
    rows,selected,capacity,exposure,prior,maximum=collect(public)
    for name,frame in (('all_metrics.csv',rows),('selected_metrics.csv',selected),('capacity_outcomes.csv',capacity),
                       ('training_exposure.csv',exposure),('prior_selected_metrics.csv',prior)):
        pd.testing.assert_frame_equal(pd.read_csv(folder/name),frame,rtol=0,atol=2e-11,check_exact=False)
    if read(folder/'contrasts.json')['contrasts']!=contrasts(rows,selected,prior):raise ValueError('Descriptive point mismatch')
    print(json.dumps({'passed':True,'new_metric_sets':len(rows),'prior_metric_sets':288,'source_selected_conditions':32,
        'capacity_outcomes':len(capacity),'exposure_points':len(exposure),'descriptive_contrast_points':448,
        'maximum_abs_metric_discrepancy':maximum,'scope':'Public probability metrics, source-only selections/metadata, capacity labels, exposure and descriptive points; no full optimizer replay or untouched test interpretation.'}))


def figures(output,rows,capacity):
    os.environ.setdefault('MPLCONFIGDIR',str(REPO.parent/'publication_runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(output/f'{name}.{ext}',dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for i,dataset in enumerate(('DEAP','SEEDIV')):
        for j,metric in enumerate(METRICS):
            ax=axes[i,j]
            for group,color,marker in ((1,'#2463aa','o'),(2,'#bd4c25','s')):
                part=capacity[(capacity.dataset==dataset)&(capacity.group==group)].set_index(['pretrained','target'])
                values=[part.loc[(p,t),metric]*(100 if j==0 else 1) for p in (True,False) for t in ('real','permuted')]
                ax.plot(values,np.arange(4)+(group-1.5)*.12,linestyle='none',marker=marker,color=color,label=f'Grouping {group}')
            ax.axvline(95 if j==0 else .1,color='#777777',linestyle='--',label='Capacity criterion')
            ax.set_yticks(range(4),['Pretrained: real','Pretrained: permuted','Random: real','Random: permuted'] if j==0 else ['']*4)
            ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(dataset+': fixed 12-trial capacity set, update400')
            ax.set_xlabel('Balanced accuracy (%)' if j==0 else 'Balanced cross entropy (log scale)')
            if j==0:ax.set_xlim(0,103)
            else:ax.set_xscale('log')
    axes[0,1].legend(fontsize=8);fig.suptitle('Capacity checks: training trials only, including permuted labels',fontsize=13)
    save(fig,'capacity_outcomes')
    colors={0.001*(6/256)**.5:'#2463aa',.001:'#bd4c25',.01:'#348344'}
    panels=(('DEAP',1),('DEAP',2),('SEEDIV',1),('SEEDIV',2))
    for pretrained in (True,False):
        fig,axes=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
        part=rows[(rows.kind=='head')&(rows.pretrained==pretrained)]
        for i,(dataset,group) in enumerate(panels):
            panel=part[(part.dataset==dataset)&(part.group==group)]
            for (scaled,rate),trajectory in panel.groupby(['scaled','head_rate']):
                for j in range(2):
                    scores=trajectory[trajectory.role=='train'].set_index('step').balanced_log_loss if j==0 else \
                        trajectory[trajectory.role!='train'].groupby('step').balanced_log_loss.mean()
                    axes[i,j].plot(scores.index,scores.values,color=colors[min(colors,key=lambda r:abs(r-float(rate)))],linestyle='-' if scaled else '--',marker='o',markersize=3,
                        label=('Standardized' if scaled else 'Raw')+f', LR={rate:.4g}')
            for j in range(2):
                axes[i,j].axhline(np.log(2 if dataset=='DEAP' else 3),color='#777777',linestyle=':',linewidth=.8)
                axes[i,j].set_title(f'{dataset}, group {group}: '+('training' if j==0 else 'equal familiar/unseen validation'))
                axes[i,j].set_xlabel('Head optimizer updates');axes[i,j].set_ylabel('Balanced log loss');axes[i,j].grid(alpha=.2)
        h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=8)
        prefix='pretrained' if pretrained else 'random';fig.suptitle(f'{prefix.capitalize()} cached features: all head scaling/rate trajectories',fontsize=13)
        save(fig,'head_'+prefix+'_learning')
    fig,axes=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
    for i,(dataset,group) in enumerate(panels):
        part=rows[(rows.kind=='long')&(rows.dataset==dataset)&(rows.group==group)]
        for (pretrained,trainable),trajectory in part.groupby(['pretrained','trainable']):
            for j in range(2):
                scores=trajectory[trajectory.role=='train'].set_index('step').balanced_log_loss if j==0 else \
                    trajectory[trajectory.role!='train'].groupby('step').balanced_log_loss.mean()
                axes[i,j].plot(scores.index,scores.values,color='#2463aa' if pretrained else '#bd4c25',
                    linestyle='-' if trainable else '--',marker='o',markersize=4,
                    label=('Pretrained' if pretrained else 'Random')+(': fine-tuned' if trainable else ': frozen'))
        for j in range(2):
            axes[i,j].axhline(np.log(2 if dataset=='DEAP' else 3),color='#777777',linestyle=':',linewidth=.8)
            axes[i,j].set_title(f'{dataset}, group {group}: '+('training' if j==0 else 'equal familiar/unseen validation'))
            axes[i,j].set_xlabel('Optimizer updates');axes[i,j].set_ylabel('Balanced log loss');axes[i,j].grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('All 16 longer neural trajectories: source panels only; head LR fixed before fitting',fontsize=13)
    save(fig,'long_source_learning')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('generate','verify'))
    args=parser.parse_args();public=REPO/'results/development'/STUDY
    (generate if args.action=='generate' else verify)(public)
