"""Freeze/verify synthetic EEGNet checks. This CLI has no dataset-fitting mode."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.eegnet_regression import EEGNetRegression
from gruxnet import faced_eegnet_runner as runner

PACKET=REPO/'results/development/faced_eegnet_preflight_2026-10-11'
BINDINGS=PACKET/'implementation_bindings_v2.json'
IMPLEMENTATION=['gruxnet/eegnet_regression.py','gruxnet/faced_eegnet_runner.py',
                'scripts/run_faced_eegnet_preflight.py','scripts/audit_faced_closest_method.py',
                'tests/test_faced_eegnet_preflight.py']
HISTORICAL=['faced_representation_v2_2026-10-11','contribution_decision_2026-10-11','faced_scope_precision_2026-10-11']


def load(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def preserve_inputs():
    plan=load(PACKET/'plan.json')
    for path,expected in plan['bindings'].items():
        if sha(REPO/path)!=expected:raise ValueError('Frozen input changed: '+path)
    if sha(PACKET/'declaration.md')!=plan['declaration_sha256']:raise ValueError('Declaration changed')
    count=0
    for name in HISTORICAL:
        receipt=load(REPO/'results/development'/name/'completion/verification.json')
        for path,expected in receipt['bindings'].items():
            if sha(REPO/path)!=expected:raise ValueError('Previous scientific artifact changed: '+path)
            count+=1
    if count!=75:raise ValueError('Changed historical binding coverage')
    return count


def freeze():
    preserve_inputs()
    binding={name:sha(REPO/name) for name in IMPLEMENTATION}
    value={'status':'frozen_synthetic_preflight_only','plan_sha256':sha(PACKET/'plan.json'),
           'code_bindings':binding,'original_dataset_access':False,'diagnostic_activated':False,
           'confirmation_access':False,'research_question_changed':False}
    path=BINDINGS
    if path.exists() and load(path)!=value:raise PermissionError('Refuse to overwrite a frozen implementation')
    save(path,value);print({'synthetic_preflight_frozen':True,'code_files':len(binding)},flush=True)


def check():
    count=preserve_inputs();binding=load(BINDINGS)
    if binding['plan_sha256']!=sha(PACKET/'plan.json') or binding['code_bindings']!={name:sha(REPO/name) for name in IMPLEMENTATION}:
        raise ValueError('Frozen preflight implementation changed')
    return count,binding


def run(require_cuda=True):
    count,binding=check();torch.set_num_threads(2)
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA required for hardware qualification')
    started=time.monotonic();rng=np.random.default_rng(87141)
    source=[('SYNTHETIC_SOURCE',i) for i in range(8)];held=[('SYNTHETIC_EVALUATION',i) for i in range(2)]
    pairs=source+held;windows={p:rng.normal(size=(7,30,512)).astype(np.float64) for p in pairs}
    targets={p:float(rng.uniform(0,7)) for p in pairs}
    arms={'source_fit':source,'validation_unseen':held,'validation_familiar':held}
    decisions={p:{'eeg_available':True} for p in pairs}
    decisions[held[1]]['eeg_available']=False;del windows[held[1]]
    scaler=runner.SourceChannelScaler.fit(source,windows)
    rows=[]
    devices=['cpu']+(['cuda:0'] if require_cuda else [])
    # One two-epoch CPU loop plus every declared optimizer/readout/seed GPU combination.
    for device in devices:
        combinations=[('context_residual',.001,runner.SEEDS[0])] if device=='cpu' else [(r,lr,seed) for r in runner.READOUTS for lr in runner.RATES for seed in runner.SEEDS]
        for readout,rate,seed in combinations:
            offsets={arm:{p:(3.5 if readout=='eeg_only' else 2.5+.1*p[1]) for p in members} for arm,members in arms.items()}
            budget=runner.FitBudget();events=[];begin=time.monotonic()
            fit=runner.fit_trial_model(source,arms,windows,targets,offsets,scaler,decisions,rate=rate,seed=seed,device=device,budget=budget,event=events.append,epochs=2,checkpoints=(1,2),source_inference_offsets=offsets['source_fit'])
            peak=0 if device=='cpu' else torch.cuda.max_memory_reserved(device)
            allocated=0 if device=='cpu' else torch.cuda.max_memory_allocated(device)
            if peak>5500000000:raise MemoryError('Synthetic measured batch exceeds fixed memory cap')
            cp=fit['checkpoints'][2]
            if cp['validation_unseen_raw'][1]!=offsets['validation_unseen'][held[1]]:raise ValueError('Fallback fetched or changed')
            rows.append({'device':device,'readout':readout,'learning_rate':rate,'seed':seed,
                         'synthetic_epochs':2,'optimizer_steps':2,'trial_batch':8,'windows_per_trial':7,
                         'forward_backward_checkpoint_finite':True,'fallback_without_signal':True,
                         'peak_reserved_bytes':peak,'peak_allocated_bytes':allocated,'charged_fit_seconds':budget.elapsed,
                         'end_to_end_seconds':time.monotonic()-begin})
            print({'synthetic_preflight':device,'readout':readout,'rate':rate,'seed':seed,'peak_reserved_MB':round(peak/1e6,2)},flush=True)
    gpu=torch.cuda.get_device_properties(0) if require_cuda else None
    result={'status':'synthetic_hardware_preflight_passed' if gpu else 'cpu_only_not_hardware_qualified',
            'implementation_bindings_sha256':sha(BINDINGS),
            'preceding_scientific_bindings_verified':count,'frozen_input_bindings_verified':13,
            'parameter_count':sum(p.numel() for p in EEGNetRegression().parameters()),
            'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                       'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                       'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
                       'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                       'gpu_name':None if gpu is None else gpu.name,'gpu_total_bytes':None if gpu is None else gpu.total_memory},
            'checks':rows,'synthetic_optimizer_steps':sum(r['optimizer_steps'] for r in rows),
            'maximum_cuda_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows),
            'elapsed_seconds':time.monotonic()-started,'original_dataset_fits':0,'original_recordings_opened':0,
            'original_rating_files_opened':0,'new_original_targets_decoded':0,'confirmation_outcomes_opened':0,
            'diagnostic_activated':False,'research_question_changed':False,
            'scope_limit':'Implementation/finite execution/memory qualification only; no original-data convergence, prediction quality or training-time estimate.'}
    preserve_inputs();save(PACKET/'synthetic_preflight.json',result)
    print({'status':result['status'],'original_dataset_fits':0,'maximum_reserved_bytes':result['maximum_cuda_reserved_bytes']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','check','run']);parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='check':print({'historical_bindings_verified':check()[0]})
    else:run(require_cuda=not args.cpu_only)
