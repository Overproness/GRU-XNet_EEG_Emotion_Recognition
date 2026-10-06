"""Declare/run the resumable full-grid study; optional authorized Git milestones."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.data import sha256
from gruxnet.heldout_tuning import (STUDY,SOURCES,MODELS,RATES,STEPS,NORMS,INITIALIZATIONS,
    jobs_and_feasibility,case_id,context_id,fit_neural,fit_context,atomic_json,stamp)
from gruxnet.learning_controls import raw_inputs
from gruxnet.full_context_controls_v2 import load
from scripts.audit_heldout_tuning import replay_case,certificate_valid,audit


def declare(root):
    output=root/STUDY
    if output.exists(): raise FileExistsError('Do not replace a frozen study')
    upstream=root/'learning_controls_2026-10-06'
    old_plan=json.loads((upstream/'plan.json').read_text())
    for name,checksum in old_plan['source_sha256'].items():
        if sha256(REPO/name)!=checksum: raise ValueError('Previously frozen source changed')
    for study in ('learning_controls_2026-10-06','source_bn_diagnostic_2026-10-06','memo_bn_check_2026-10-06'):
        check=json.loads((root/study/'verification.json').read_text())
        if not check.get('passed',check.get('complete',False)): raise ValueError('Upstream verification required')
    bases={d:pd.read_csv(root/f'cache_full_context_{d.lower()}/trials.csv') for d in ('SEEDIV','DEAP')}
    jobs,contexts,feasibility,retained=jobs_and_feasibility(bases)
    plan={'created_utc':stamp(),'development_only':True,'research_question_change_approved':False,
          'phase':'Complete repeated outer held-out predictions with independently fold-local source tuning',
          'jobs':jobs,'contexts':contexts,'source_sha256':{f:sha256(REPO/f) for f in SOURCES},
          'upstream_binding':{f:sha256(root/f) for f in ('learning_controls_2026-10-06/plan.json','learning_controls_2026-10-06/verification.json','eegnet_author_audit_2026-10-06/port_verification.json','source_bn_diagnostic_2026-10-06/verification.json','memo_bn_check_2026-10-06/verification.json')},
          'counts':{'selected_neural_cases':len(jobs),'full_neural_trajectories':len(jobs)*len(RATES),'source_validation_candidates':len(jobs)*len(RATES)*len(STEPS)*len(NORMS),
                    'selected_context_calibrators':len(contexts),'context_C_refits':len(contexts)*4,'retained_selected_checkpoints':retained,'expected_outer_prediction_rows':93760},
          'known_development_evidence':'Prior 680 full-width neural fits, repeated feature/transformer studies, all96 source learning fits and posthoc normalization diagnostics have been inspected. This is a prospective new grid on REUSED cohorts, not a pristine test/new participant confirmation. No test outcome used to set a fold-specific recipe.',
          'replication':'Both existing independent fixed participant/video groupings, all session/rotation/folds, both arms, two base initializations42/91. All combinations retained. Init=base+1000fold+10000rotation, stochastic RNG=init+1000000. No seed/group selection or probability ensembling.',
          'validation':'Same validation people across arms. Two disjoint source-validation clip panels: (1) original unseen-video validation panel; (2) those validation people on videos actually present in source TRAINING. Both panels require all target classes. Unexposed arm panels exclude outer-test videos; target people excluded in all arms. Equal .5/.5 panel mean class-balanced log loss determines selection, mean panel BA breaks ties, then first declared candidate. Both panels separately retained. Applies to every neural and logistic model. Familiar panel differs between arms, so exposure contrast includes validation/selection regime and is not a causal clip-identity intervention.',
          'optimization':'Two constant LR .001/.0003 trajectories, each1200 balanced12 AdamWupdates,wd.01,clip1,dropout.5,FP32/noAMP/scheduler/augmentation. Durations200/600/1200 come from SAME uninterrupted trajectory, not separate restarts. Same native-class/person/rank draws, EEGNet maxnorm applied every update. Population calibration/evaluation does not change training RNG or optimizer trajectory; exact EMA state restored before continuing. This is finite-budget tuning, not proof of convergence.',
          'normalization':'Training-only raw channel or STFT channel/frequency mean/std floor1e-6. Candidate EMA versus equal-weight full-source population BatchNorm (three sequential eval passes, float64 moments, no dropout/learned updates, variancefloor1e-12). No validation/test data enters source moments. Preprocessing/raw prefix/label cohorts unchanged. EEGNet raw versus GRU STFT means comparison is not architecture-only.',
          'context':'EEGNet plus fixed source-only log video prior, no extra parameters. For TRAINING rows exclude ALL receiving participant labels before per-video/global Laplace1 prior. Validation/test priors use only actual training labels. Raw prior and independently trained logistic context-only calibration(C .01/.1/1/10,balanced,training-onlyscaler,maxiter2000,tol1e-6) retained; same two-panel selection criterion. Deterministic context controls logically repeated across both neural initializations, without treating them as independent fits.',
          'seal':'All candidate source probabilities, metrics, selected-state/draw/split hashes and checkpoint SHA sealed in selection.json BEFORE outer-test features are sent to inference. Test inference on a fresh model from the sealed state; immediate independent selected-state replay and all-candidate metric reconstruction required before acceptance/export.',
          'retention':'All EEGNet/EEGNet-context selected states retained. GRU sentinels fixed at rotation0/fold0 in EVERY session/group/init/arm retained. All other GRU selected.pt stored temporarily, independently replayed and source-population moments reconstructed when selected, then only this exact newly generated file deleted. SHA/state digest/certificate and all predictions remain. Later raw state replay for nonretained GRU requires refitting; no claim of archival replay for deleted weights.',
          'prefix_guard':'64 exact original200-update source-state guards: session1/rotation0/fold0,GRU/EEGNet,both datasets/groups/inits/arms/LRs. Source validation expansion cannot change optimization. Any mismatch stops the job before interpreting that test; no successful-fold substitution.',
          'evaluation':'All1080 SEED coarse3 primary,810 conditionalbinary secondary;1264 DEAP individual-valence binary. Once-per-trial OOF for each model/arm/group/init. Report ALL four combinations and combined mean correctness/loss per observed person/video, not fold BA averages or probability ensemble. Paired10000seed20261006 participant-only and crossed person/video percentiles, fixed fits/reused cohort/unadjusted. All model exposure contrasts; EEGNet minusGRU; EEGNet-context minusEEGNet/prior/calibratedcontext botharms/alltasks.',
          'alignment':'Within each identical model/cell/group/init/video, exchange every other held-out participant EEG while retaining recipient context. Average correctness/loss over donors, not probabilities. Dyadic recipient*donor*video bootstrap weights,10000existingdraws. All5models/botharms/tasks, allalignment-vs-context comparisons. Report ineligible rows/nonestimable draws. Context invariance and SEED assigned-label zero aggregate alignment are required sanity checks. Conditional association, not causal physiological mechanism.',
          'stop_resume':'Fail on nonfinite gradients/loss, missing class/split coverage, changed data/source/config, nonconvergent logistic refit or numerical replay failure; preserve failure file and stop. Resume only matching sealed study/cases. No outer-test aggregate comparisons computed before complete verification. Optional Git milestone export contains only declared compact artifacts, excludes EEG/weights, stages only owned evidence paths; failures stop publication rather than force push.',
          'gate':'No manuscript or research-question change. Author approval and then-current paper archive required before adopting a new contribution.'}
    output.mkdir(parents=True); atomic_json(output/'plan.json',plan); atomic_json(output/'feasibility.json',feasibility)
    config={'plan_sha256':sha256(output/'plan.json'),'device':'cuda','device_name':torch.cuda.get_device_name(),
            'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'numpy':np.__version__,'python':sys.version,'input_bindings':{d:json.loads((root/f'cache_full_context_{d.lower()}/prepared.json').read_text())['fingerprint'] for d in bases},
            'feasibility_sha256':sha256(output/'feasibility.json')}
    atomic_json(output/'config.json',config)
    atomic_json(output/'progress.json',{'state':'declared','updated_utc':stamp(),'selected_neural_completed':0,
                                      'selected_neural_total':2040,'config_sha256':sha256(output/'config.json'),
                                      'research_question_change_approved':False})
    print(json.dumps({'declared':plan['counts'],'plan_sha256':sha256(output/'plan.json')}),flush=True)


def validate(root,output):
    plan=json.loads((output/'plan.json').read_text()); config=json.loads((output/'config.json').read_text())
    if sha256(output/'plan.json')!=config['plan_sha256'] or sha256(output/'feasibility.json')!=config['feasibility_sha256']: raise ValueError('Changed declaration/feasibility')
    for name,checksum in plan['source_sha256'].items():
        if sha256(REPO/name)!=checksum: raise ValueError(f'Changed frozen source {name}')
    for name,checksum in plan['upstream_binding'].items():
        if sha256(root/name)!=checksum: raise ValueError('Changed verified upstream evidence')
    for key,actual in (('torch',torch.__version__),('cuda',torch.version.cuda),('cudnn',torch.backends.cudnn.version()),('numpy',np.__version__),('python',sys.version),('device_name',torch.cuda.get_device_name())):
        if config[key]!=actual: raise ValueError('Changed declared fitting environment')
    return plan,config


def publish(output,message):
    from scripts.export_heldout_tuning import export
    export(output)
    # Only this study's exported artifacts and progress note are owned by worker.
    paths=[f'results/development/{STUDY}',f'docs/publication/GRU-XNet_Heldout_Tuning_Status_2026-10-06.md']
    def git(*args):
        return subprocess.run(['git',*args],cwd=REPO,text=True,capture_output=True,check=True)
    if git('branch','--show-current').stdout.strip()!='main': raise ValueError('Publisher requires declared main branch')
    if git('remote','get-url','origin').stdout.strip()!='https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition.git': raise ValueError('Unexpected publisher destination')
    staged=git('diff','--cached','--name-only').stdout.splitlines()
    if any(not any(name==p or name.startswith(p+'/') for p in paths) for name in staged): raise ValueError('Unrelated staged changes; stop instead of committing them')
    git('add','--',*paths)
    if not git('diff','--cached','--name-only').stdout.strip(): return
    git('commit','-m',message); git('push','origin','main')


def run(root,limit=None,push=False):
    output=root/STUDY; plan,config=validate(root,output)
    data={}; completed=0; newly=0; contexts_done=set(); started=time.perf_counter()
    for job in plan['jobs']:
        validate(root,output)
        dataset=job['dataset']
        if dataset not in data:
            data.clear()  # At most one corpus of waveforms/STFT in RAM.
            x,base,info=load(root/f'cache_full_context_{dataset.lower()}')
            if info['fingerprint']!=config['input_bindings'][dataset]: raise ValueError('Changed input binding')
            wave,binding=raw_inputs(dataset,root,base)
            binding_path=output/f'waveform_binding_{dataset.lower()}.json'
            if binding_path.exists() and json.loads(binding_path.read_text())!=binding: raise ValueError('Raw waveform binding changed')
            atomic_json(binding_path,binding); data[dataset]=(x,wave,base)
        x,wave,base=data[dataset]
        cj={k:v for k,v in job.items() if k not in ('model','initialization')}; cid=context_id(cj)
        cfolder=output/'contexts'/cid
        if cid not in contexts_done:
            if (cfolder/'certificate.json').exists(): certificate_valid(cfolder,output,True)
            else: fit_context(cj,base,output); replay_case(cfolder,output,base,context=True)
            contexts_done.add(cid)
        folder=output/'fits'/case_id(job)
        if (folder/'certificate.json').exists(): certificate_valid(folder,output)
        else:
            if not (folder/'record.json').exists(): fit_neural(job,x,wave,base,output,root)
            replay_case(folder,output,base,x,wave)
            newly+=1
        completed+=1
        progress={'state':'running','updated_utc':stamp(),'worker_pid':os.getpid(),'selected_neural_completed':completed,
                  'selected_neural_total':len(plan['jobs']),'context_cases_checked_this_process':len(contexts_done),
                  'new_cases_this_process':newly,'last_case':case_id(job),'elapsed_seconds_this_process':time.perf_counter()-started,
                  'config_sha256':sha256(output/'config.json'),'research_question_change_approved':False}
        atomic_json(output/'progress.json',progress)
        print(json.dumps(progress),flush=True)
        if push and newly and newly%40==0:
            publish(output,f'Checkpoint held-out tuning: {completed}/2040 selected neural cases verified')
        if limit is not None and newly>=limit:
            progress['state']='bounded_run_finished'; atomic_json(output/'progress.json',progress)
            return
    result=audit(output)
    from scripts.analyze_heldout_tuning import analyze,report
    for dataset in ('SEEDIV','DEAP'): analyze(output,dataset)
    report(output); result=audit(output)
    progress['state']='complete'; progress['updated_utc']=stamp(); atomic_json(output/'progress.json',progress)
    if push: publish(output,'Complete and verify fold-tuned held-out and EEGNet-context study')
    print(json.dumps(result),flush=True)


def exclusive_run(root,limit=None,push=False):
    # OS lock releases automatically after a crash; prevents two GPU writers.
    output=root/STUDY
    with (output/'worker.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(root,limit,push)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__); parser.add_argument('command',choices=('plan','run'))
    parser.add_argument('--root',type=Path,default=REPO.parent/'publication_runs'); parser.add_argument('--limit',type=int)
    parser.add_argument('--push-milestones',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='plan': declare(args.root.resolve())
        else: exclusive_run(args.root.resolve(),args.limit,args.push_milestones)
    except Exception as error:
        output=args.root.resolve()/STUDY
        if output.exists(): atomic_json(output/'FAILURE.json',{'created_utc':stamp(),'worker_pid':os.getpid(),'error':str(error),'traceback':traceback.format_exc()})
        raise
