"""Declare and run the authorized small regularization/fine-tuning comparison."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.cbramod_adaptation import (STUDY, resource_pilot, declare, validate,
    fit_linear, fit_neural, neural_id, linear_id, export, sha, atomic, stamp)
from scripts.audit_cbramod_adaptation import audit_linear, audit_neural, finish
from gruxnet.train import seed_everything


def publish(output,message):
    validate(output.parent,output)
    export(output)
    path=(REPO/'results/development'/STUDY).relative_to(REPO).as_posix()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()
    if branch!='main' or remote!='https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition.git':
        raise ValueError('Unexpected publication branch/destination')
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
    if any(not name.startswith(path+'/') for name in staged):raise ValueError('Unrelated staged change')
    subprocess.run(['git','add','--',path],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
    subprocess.run(['git','push','origin','main'],cwd=REPO,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('pilot','declare','run','export'))
    parser.add_argument('--root',type=Path,default=REPO.parent/'publication_runs')
    parser.add_argument('--push',action='store_true')
    args=parser.parse_args();root=args.root.resolve();output=root/STUDY;seed_everything(42)
    if args.action=='pilot':print(json.dumps(resource_pilot(root)));return
    if args.action=='declare':declare(root);export(output);print(json.dumps({'declared':STUDY,'plan_sha256':sha(output/'plan.json')}));return
    if args.action=='export':export(output);return
    plan=validate(root,output)
    lock=output/'worker.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    try:
        config={'python':platform.python_version(),'torch':torch.__version__,'device':torch.cuda.get_device_name(),
                'plan_sha256':sha(output/'plan.json'),'outer_test_inferences':0}
        if (output/'config.json').exists():
            if json.loads((output/'config.json').read_text())!=config:raise ValueError('Changed environment')
        else:atomic(output/'config.json',config)
        for job in plan['regularization_jobs']:
            fit_linear(root,output,job)
            folder=output/'linear'/linear_id(job)
            if not(folder/'verification.json').exists():audit_linear(root,output,job)
            count=len(list((output/'linear').glob('*/verification.json')))
            atomic(output/'progress.json',{'state':'regularization','linear_heads_completed':count,'linear_heads_total':24,
                'neural_trajectories_completed':len(list((output/'neural').glob('*/verification.json'))),
                'neural_trajectories_total':24,'updated_utc':stamp(),'outer_test_inferences':0,
                'research_question_change_approved':False})
            print(json.dumps({'verified_expanded_heads':count,'total':24}),flush=True)
        if args.push:publish(output,'Verify all expanded CBraMod head regularization controls')
        else:export(output)
        for job in plan['neural_jobs']:
            fit_neural(root,output,job)
            folder=output/'neural'/neural_id(job)
            if not(folder/'verification.json').exists():audit_neural(root,output,job)
            count=len(list((output/'neural').glob('*/verification.json')))
            atomic(output/'progress.json',{'state':'fine_tuning','linear_heads_completed':24,'linear_heads_total':24,
                'neural_trajectories_completed':count,'neural_trajectories_total':24,'updated_utc':stamp(),
                'outer_test_inferences':0,'research_question_change_approved':False})
            print(json.dumps({'verified_neural_trajectories':count,'total':24,'job':job}),flush=True)
            if args.push and count in(8,16):publish(output,f'Verify {count} matched CBraMod adaptation trajectories')
            elif count%4==0:export(output)
        result=finish(root,output)
        atomic(output/'progress.json',{'state':'complete','linear_heads_completed':24,'linear_heads_total':24,
            'neural_trajectories_completed':24,'neural_trajectories_total':24,'updated_utc':stamp(),
            'outer_test_inferences':0,'research_question_change_approved':False})
        if args.push:publish(output,'Complete and verify matched CBraMod regularization and adaptation controls')
        else:export(output)
        print(json.dumps(result),flush=True)
    except Exception as error:
        atomic(output/f'FAILURE_{stamp().replace(":","-")}.json',{'exception':repr(error),
            'traceback':traceback.format_exc(),'created_utc':stamp()})
        export(output);raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
