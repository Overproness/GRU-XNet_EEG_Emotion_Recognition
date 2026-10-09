"""Declare/run the authorized source-only CBraMod learning diagnostic."""
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
from gruxnet.cbramod_learning import (STUDY,STEPS,declare,validate,progress,export,
    fit_head,fit_neural,identifier,sha,atomic,stamp,resource_preflight)
from scripts.audit_cbramod_learning import audit_head,audit_neural,finish,read_record
from gruxnet.train import seed_everything


def publish(output,message):
    validate(output.parent,output);export(output)
    prefix=(REPO/'results/development'/STUDY).relative_to(REPO).as_posix()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()
    if branch!='main' or remote!='https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition.git':
        raise ValueError('Unexpected branch/destination')
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
    if any(not name.startswith(prefix+'/') for name in staged):raise ValueError('Unrelated staged path')
    subprocess.run(['git','add','--',prefix],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
    subprocess.run(['git','push','origin','main'],cwd=REPO,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('pilot','declare','run','export'))
    parser.add_argument('--root',type=Path,default=REPO.parent/'publication_runs')
    parser.add_argument('--push',action='store_true')
    args=parser.parse_args();root=args.root.resolve();output=root/STUDY;seed_everything(42)
    if args.action=='pilot':print(json.dumps(resource_preflight(root)));return
    if args.action=='declare':
        declare(root);export(output);print(json.dumps({'declared':STUDY,'plan_sha256':sha(output/'plan.json')}));return
    if args.action=='export':export(output);return
    plan=validate(root,output)
    if (output/'verification.json').exists():
        proof=json.loads((output/'verification.json').read_text())
        if not proof['complete'] or proof['plan_sha256']!=sha(output/'plan.json') or proof['summary_sha256']!=sha(output/'summary.json'):
            raise ValueError('Changed completed study')
        for kind in STEPS:
            for job in plan['jobs'][kind]:
                folder,record=read_record(output,kind,job);case=json.loads((folder/'verification.json').read_text())
                if not case['complete'] or case['record_sha256']!=sha(folder/'record.json'):raise ValueError('Changed completed case')
        print(json.dumps({'complete':True,'action':'Checked completed study without overwriting records'}));return
    lock=output/'worker.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    try:
        config={'python':platform.python_version(),'torch':torch.__version__,'device':torch.cuda.get_device_name(),
                'plan_sha256':sha(output/'plan.json'),'outer_test_inferences':0,'head_dtype':'FP64CPU1thread',
                'neural_dtype':'FP32CUDAexplicitMATH','seed_function_cpu_threads':4}
        if (output/'config.json').exists():
            if json.loads((output/'config.json').read_text())!=config:raise ValueError('Changed environment')
        else:atomic(output/'config.json',config)
        for kind in STEPS:
            for job in plan['jobs'][kind]:
                if kind=='head':fit_head(root,output,job)
                else:fit_neural(root,output,kind,job)
                folder=output/kind/identifier(kind,job)
                if not(folder/'verification.json').exists():
                    if kind=='head':audit_head(root,output,job)
                    else:audit_neural(root,output,kind,job)
                count=len(list((output/kind).glob('*/verification.json')))
                progress(output,'running_'+kind)
                print(json.dumps({'phase':kind,'verified':count,'total':len(plan['jobs'][kind]),'job':job}),flush=True)
                checkpoint=(kind=='head' and count==48) or (kind!='head' and count%4==0)
                if args.push and checkpoint:publish(output,f'Verify CBraMod source learning {kind} milestone {count}')
                elif checkpoint:export(output)
        result=finish(root,output);progress(output,'complete')
        if args.push:publish(output,'Complete verified CBraMod capacity and longer source learning controls')
        else:export(output)
        print(json.dumps(result),flush=True)
    except Exception as error:
        atomic(output/f'FAILURE_{stamp().replace(":","-")}.json',{'exception':repr(error),
            'traceback':traceback.format_exc(),'created_utc':stamp()})
        export(output);raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
