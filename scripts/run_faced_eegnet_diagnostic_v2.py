"""Fresh v2 continuation: exclude declared shuffled capacity targets, preserve fixed science."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.faced_eegnet_bridge import authoritative_bundle,WindowBridge,guard_cached_identity,array_sha256
from gruxnet.faced_cached_targets import cached_original_target
from gruxnet.faced_eegnet_runner import run_diagnostic,FitBudget,STRONG_CONTROLS
from gruxnet.faced_classical_contract import context_features
from gruxnet.faced_source_quality import read_interval,selected_header,quality_metrics
from gruxnet.faced_proposal_preprocessing_v2 import calibrate_common_counts
from gruxnet.faced_artifact_control import artifact_trial_windows
from gruxnet.faced_inference_quality import eeg_availability
from scripts.audit_faced_raw import read_header
from scripts.qualify_faced_author_preprocessing import installed_sources
from scripts.verify_faced_eegnet_preflight import verify as preflight_verify

BASE=REPO/'results/development'
PUBLIC=BASE/'faced_eegnet_diagnostic_v2_2026-10-11'
PRIVATE=REPO.parent/'publication_runs/faced_eegnet_diagnostic_v2_2026-10-11'
RAW=REPO.parent/'Data/Data'
PREFLIGHT=BASE/'faced_eegnet_preflight_2026-10-11'
QUALITY=BASE/'faced_inference_quality_2026-10-11'
REPRESENTATION=BASE/'faced_representation_v2_2026-10-11'
ARTIFACT=BASE/'faced_artifact_control_2026-10-11'
OLD_PRIVATE=REPO.parent/'publication_runs/faced_artifact_control_2026-10-11'
PROGRAMME=BASE/'faced_development_protocol_2026-10-11/plan.json'
PROTOCOL=REPO/'docs/publication/FACED_EEGNet_Diagnostic_Continuation_2026-10-11.md'
CHECKPOINT_COLUMNS=['rotation','readout','learning_rate','seed','epoch','arm','subject','vid','family','acquisition_group','target','prediction_raw','eeg_available','fallback']
ENSEMBLE_COLUMNS=['rotation','readout','selected_learning_rate','selected_epoch','arm','subject','vid','family','acquisition_group','target','prediction','eeg_available','fallback']


def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(path.read_bytes())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def save(path,value,*,exclusive=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)
    else:path.write_text(text,encoding='utf-8')


def code_bindings():
    original=load(PREFLIGHT/'implementation_bindings_v2.json')['code_bindings']
    names=list(original)+['gruxnet/faced_eegnet_bridge.py','scripts/run_faced_eegnet_diagnostic.py',
                         'tests/test_faced_eegnet_bridge.py','scripts/audit_faced_eegnet_diagnostic.py',
                         'tests/test_faced_eegnet_diagnostic_audit.py','scripts/run_faced_eegnet_diagnostic_v2.py',
                         'gruxnet/faced_cached_targets.py','tests/test_faced_cached_targets.py',
                         'scripts/audit_faced_eegnet_diagnostic_v2.py','tests/test_faced_eegnet_diagnostic_audit_v2.py',
                         'scripts/audit_faced_eegnet_held_attempt.py','scripts/qualify_faced_cached_targets.py']
    return {name:digest(REPO/name) for name in names}


def input_bindings():
    names=[PROGRAMME,PROTOCOL,PREFLIGHT/'completion/verification.json',PREFLIGHT/'plan.json',
           QUALITY/'signal_audit.json',QUALITY/'comparison.json',QUALITY/'predictions.csv',
           REPRESENTATION/'predictions.csv',REPRESENTATION/'comparison.json',REPRESENTATION/'plan.json',
           BASE/'faced_eegnet_diagnostic_2026-10-11/activation.json',BASE/'faced_eegnet_diagnostic_2026-10-11/execution.json',
           BASE/'faced_eegnet_diagnostic_2026-10-11/source_gate.json',BASE/'faced_eegnet_diagnostic_2026-10-11/feature_manifest.json',
           BASE/'faced_eegnet_diagnostic_2026-10-11/cached_target_diagnosis.json',BASE/'faced_eegnet_diagnostic_2026-10-11/held_attempt_audit.json',
           ARTIFACT/'source_records.json',ARTIFACT/'validation_records.json',
           BASE/'faced_backup_2026-10-10/participant_reservation.json',BASE/'faced_backup_2026-10-10/stimulus_metadata.json',
           BASE/'faced_codebooks_2026-10-10/audit.json',BASE/'faced_raw_2026-10-11/headers.json',BASE/'faced_raw_2026-10-11/extraction.json']
    names += [REPO/name for name in ('gruxnet/faced_source_quality.py','gruxnet/faced_proposal_preprocessing.py',
              'gruxnet/faced_proposal_preprocessing_v2.py','gruxnet/faced_source_notch.py','gruxnet/faced_artifact_control.py',
              'gruxnet/faced_inference_quality.py','gruxnet/faced_classical_contract.py','gruxnet/faced_classical_runner.py',
              'scripts/audit_faced_raw.py','scripts/qualify_faced_author_preprocessing.py')]
    return {p.relative_to(REPO).as_posix():digest(p) for p in names}


def runtime():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'sklearn':sklearn.__version__,'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'mne_installed_sources':installed_sources(),'preprocessing_blas_threads':12,'readout_blas_threads':2,'torch_cpu_threads':2}


def metadata_bundle():
    p=load(PROGRAMME);roles=load(BASE/'faced_backup_2026-10-10/participant_reservation.json')
    catalog={r['clip']:r for r in load(BASE/'faced_codebooks_2026-10-10/audit.json')['stimuli']}
    rows=load(ARTIFACT/'source_records.json')['records']+load(ARTIFACT/'validation_records.json')['records']
    decisions={(r['subject'],r['vid']):r for r in load(QUALITY/'signal_audit.json')['trial_decisions']}
    records,metadata=authoritative_bundle(p,roles,catalog,rows,decisions)
    return p,records,metadata,decisions


def declare():
    preflight_verify();metadata_bundle()
    held=load(BASE/'faced_eegnet_diagnostic_2026-10-11/execution.json')
    audit=load(BASE/'faced_eegnet_diagnostic_2026-10-11/held_attempt_audit.json')
    if held['status']!='held_or_failed' or held['new_neural_fits']!=0 or audit['passed'] is not True:
        raise PermissionError('Require preserved independently audited zero-fit attempt')
    for k,name in enumerate(('faced_representation_v2_2026-10-11','contribution_decision_2026-10-11','faced_scope_precision_2026-10-11','faced_eegnet_preflight_2026-10-11')):
        for relative,expected in load(BASE/name/'completion/verification.json')['bindings'].items():
            if digest(REPO/relative)!=expected:raise ValueError('Previous immutable scientific bytes changed')
    activation={'date':'2026-10-11','declared_utc':now(),'status':'active_development_diagnostic',
                'preflight_plan_sha256':digest(PREFLIGHT/'plan.json'),'code_bindings':code_bindings(),
                'bindings':input_bindings(),'runtime':runtime(),'confirmation_access':False,'research_question_changed':False,
                'continuation_of':'faced_eegnet_diagnostic_2026-10-11',
                'cache_target_rule':'Canonical quality targets; every real representation target agrees exactly; 3360 explicit shuffled source-capacity rows ignored before numeric conversion. Fixed kernel alpha from bound representation plan.',
                'neural_fit_cap':20,'auxiliary_context_fits':18,'epochs':120,'checkpoint_epochs':[30,60,120],
                'fit_seconds_cap':21600,'preparation_seconds_cap':7200,'memory_windows_bytes_cap':2000000000,
                'cuda_reserved_bytes_cap':5500000000,'new_original_rating_decoding':0,'raw_waveform_persistence':False,
                'access':'Only authoritative 980 source + 265 available validation pairs, previous exact native channels/intervals. All source byte/QC replay and post-read whole-file authentication before validation EEG or cached targets. Fifteen inference-failure trials retain frozen fallback; no original MAT or protected outcomes. All targets/controls reused from authenticated public development CSVs, identity/family guarded before numeric conversion.',
                'interpretation':'Reused adaptive development. No independent confirmation, manuscript pivot, algorithm novelty claim, schedule expansion or rescue fit.'}
    save(PUBLIC/'activation.json',activation,exclusive=True)
    print({'diagnostic_declared':True,'neural_fit_cap':20,'new_rating_decoding':0},flush=True)


def check():
    a=load(PUBLIC/'activation.json');preflight_verify()
    if a['code_bindings']!=code_bindings() or a['bindings']!=input_bindings() or a['runtime']!=runtime():
        raise ValueError('Frozen bridge/input/runtime changed')
    if a['preflight_plan_sha256']!=digest(PREFLIGHT/'plan.json') or a['confirmation_access'] or a['research_question_changed']:
        raise PermissionError('Activation contract changed')
    return a


class Execution:
    def __init__(self,activation):
        self.activation=activation;self.programme,self.records,self.metadata,self.decisions=metadata_bundle()
        self.started=time.monotonic();self.preparing=True;self.authenticated={};self.sample_values=0;self.private_files=[]
        self.headers={r['subject']:r['data.bdf'] for r in load(BASE/'faced_raw_2026-10-11/headers.json')['records']}
        self.originals={r['member']:r for r in load(BASE/'faced_raw_2026-10-11/extraction.json')['records']}
        self.bridge=WindowBridge(self.records,self.metadata,self.decisions,self.replay,memory_cap=2000000000)
        self.targets={};self.controls={k:{n:{} for n in STRONG_CONTROLS} for k in range(3)};self.oof={k:{} for k in range(3)}
        self.context={p:context_features(m) for p,m in self.metadata.items()}
        for p,x in self.context.items():
            if array_sha256(x)!=self.records[p]['context_feature_sha256']:raise ValueError('Context input bytes changed')
        self.fit_records=[];self.normalizers=[];self.selections={};self.current=None;self.rows_checkpoint=0;self.rows_ensemble=0
        PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
        self.checkpoint_file=(PUBLIC/'checkpoint_predictions.csv').open('x',newline='',encoding='utf-8')
        self.ensemble_file=(PUBLIC/'ensemble_predictions.csv').open('x',newline='',encoding='utf-8')
        self.checkpoint_writer=csv.DictWriter(self.checkpoint_file,fieldnames=CHECKPOINT_COLUMNS);self.checkpoint_writer.writeheader()
        self.ensemble_writer=csv.DictWriter(self.ensemble_file,fieldnames=ENSEMBLE_COLUMNS);self.ensemble_writer.writeheader()
        self.event_file=(PUBLIC/'events.jsonl').open('x',encoding='utf-8')

    def preparation_budget(self):
        if self.preparing and time.monotonic()-self.started>7200:raise TimeoutError('Fixed source/validation preparation cap')

    def log(self,kind,**values):
        self.event_file.write(json.dumps({'utc':now(),'elapsed_seconds':time.monotonic()-self.started,'type':kind,**values},allow_nan=False)+'\n');self.event_file.flush()

    def authenticate(self,person):
        self.preparation_budget()
        if person in self.authenticated:return
        directory=RAW/person.replace('sub-','sub');path=directory/'data.bdf'
        if read_header(path)!=self.headers[person]:raise ValueError('Original BDF header changed')
        for name in ('data.bdf','evt.bdf'):
            expected=self.originals[f'Data/{directory.name}/{name}']['sha256']
            if digest(directory/name)!=expected:raise ValueError('Original recording/opaque event companion changed')
        self.authenticated[person]={'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,
                                  'sha256':self.originals[f'Data/{directory.name}/data.bdf']['sha256']}

    def replay(self,r):
        self.preparation_budget();person,vid=r['subject'],r['vid'];self.authenticate(person)
        header=self.headers[person];path=RAW/person.replace('sub-','sub')/'data.bdf'
        if r['original_data_sha256']!=self.authenticated[person]['sha256']:raise ValueError('Record/raw lineage mismatch')
        start,stop=r['sample_interval'];indices=r['channel_indices']
        with path.open('rb') as stream:counts,access=read_interval(stream,header,start,stop,indices,True)
        if array_sha256(counts)!=r['count_array_sha256'] or access['range_ledger_sha256']!=r['range_ledger_sha256']:
            raise ValueError('Original selected counts/access bytes changed')
        self.sample_values+=int(counts.size)
        values=calibrate_common_counts(counts,person,selected_header(header,indices))
        with threadpool_limits(limits=12,user_api='blas'):
            windows,artifact=artifact_trial_windows(values,r['native_rate_hz'])
        quality=quality_metrics(counts,values,windows,r['native_rate_hz'])
        old=OLD_PRIVATE/f'{person}_clip{vid:02d}_quality.json'
        if digest(old)!=r['quality_file_sha256'] or load(old)!=quality:raise ValueError('Full numerical QC replay differs')
        decision=eeg_availability(quality,technical_passed=True,repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
        stem=f'{person}_clip{vid:02d}'
        save(PRIVATE/(stem+'_quality.json'),quality,exclusive=True);save(PRIVATE/(stem+'_access.json'),access,exclusive=True)
        if path.stat().st_size!=self.authenticated[person]['size'] or path.stat().st_mtime_ns!=self.authenticated[person]['mtime_ns']:
            raise ValueError('Original modified during selective replay')
        count=len(self.bridge.receipts)+1
        if count%70==0:print({'phase':r['role']+'_replay','trials':count,'elapsed_seconds':round(time.monotonic()-self.started,1)},flush=True)
        return windows,decision,{'samples_decoded':int(counts.size),'original_signal_decoded':True,'count_array_sha256':r['count_array_sha256'],
                                'count_hash_matches':True,'range_ledger_sha256':access['range_ledger_sha256'],
                                'qc_numeric_bytes_equal':True,'quality_file_sha256':digest(PRIVATE/(stem+'_quality.json')),
                                'private_access_sha256':digest(PRIVATE/(stem+'_access.json'))}

    def authenticate_after(self,role):
        people=sorted({p[0] for p,m in self.metadata.items() if m['role']==role and self.decisions[p]['eeg_available']})
        for person in people:
            if digest(RAW/person.replace('sub-','sub')/'data.bdf')!=self.authenticated[person]['sha256']:
                raise ValueError('Original changed after selective replay')

    def open_cached(self):
        self.bridge.require_targets_gate();gate=PUBLIC/'source_gate.json'
        if not gate.exists() or not load(gate)['passed']:raise PermissionError('Persisted authenticated source gate required')
        started_utc=now()
        selected={0:.1,1:100.,2:100.};control_rows=[];ignored=set()
        if load(REPRESENTATION/'plan.json')['conditions']['context_kernel']!=['rbf',['context'],.1]:
            raise PermissionError('Frozen context kernel configuration changed')
        for source in (QUALITY/'predictions.csv',REPRESENTATION/'predictions.csv'):
            expected=self.activation['bindings'][source.relative_to(REPO).as_posix()]
            if digest(source)!=expected:raise ValueError('Frozen public cache changed')
            with source.open(newline='',encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    origin='quality' if source.parent==QUALITY else 'representation'
                    p,y=cached_original_target(row,self.metadata,self.programme,origin)
                    if y is None:
                        key=int(row['rotation']),row['model'],p
                        if key in ignored:raise PermissionError('Duplicate declared shuffled capacity row')
                        ignored.add(key);continue
                    if origin=='representation' and p not in self.targets:raise PermissionError('Quality canonical target required')
                    if p in self.targets and y!=self.targets[p]:raise ValueError('Cached original target inconsistent')
                    self.targets[p]=y;k=int(row['rotation']);model=row['model'];arm=row['arm']
                    if source.parent==QUALITY and model=='context_only' and arm=='source_oof' and float(row['alpha'])==selected[k]:
                        if p in self.oof[k]:raise ValueError('Duplicate cached source OOF offset')
                        fold=int(row['oof_fold'])
                        if p[0] not in self.programme['source_participant_oof_folds'][fold]:raise PermissionError('Cached residual recipient fold mismatch')
                        self.oof[k][p]=float(row['prediction'])
                    retain=(arm=='validation_unseen' and ((source.parent==QUALITY and model in STRONG_CONTROLS[:-1] and (model!='context_only' or float(row['alpha'])==selected[k])) or (source.parent==REPRESENTATION and model=='context_kernel')))
                    if retain:
                        pairs={(r['subject'],r['vid']) for r in self.programme['rotation_manifests'][k]['arms']['validation_unseen']}
                        if p not in pairs or p in self.controls[k][model]:raise PermissionError('Duplicate/wrong cached control rectangle')
                        value=float(row['prediction'])
                        if not np.isfinite(value) or not 0<=value<=7:raise ValueError('Invalid cached context prediction')
                        self.controls[k][model][p]=value
                        control_rows.append({'rotation':k,'model':model,'subject':p[0],'vid':p[1],'family':row['family'],'target':y,'prediction':value})
        if len(ignored)!=3360:raise PermissionError('Changed exact shuffled capacity rectangle')
        if set(self.targets)!=set(self.metadata) or len(control_rows)!=1800:raise PermissionError('Complete cached target/control coverage required')
        for k in range(3):
            source={(r['subject'],r['vid']) for r in self.programme['rotation_manifests'][k]['arms']['source_fit']}
            if set(self.oof[k])!=source or not np.isfinite(list(self.oof[k].values())).all():raise PermissionError('Incomplete source OOF offsets')
        with (PUBLIC/'controls.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(control_rows[0]));writer.writeheader();writer.writerows(control_rows)
        save(PUBLIC/'targets_reuse.json',{'started_utc':started_utc,'completed_utc':now(),'source_gate_sha256':digest(gate),'cached_targets':1260,'canonical_target_source':'faced_inference_quality_2026-10-11/predictions.csv',
            'ignored_permuted_capacity_rows':len(ignored),'all_other_cached_targets_agree_exactly':True,
            'original_rating_files_opened':False,'protected_outcomes_opened':False,'new_original_rating_decoding':0,
            'source_files':{p.relative_to(REPO).as_posix():digest(p) for p in (QUALITY/'predictions.csv',REPRESENTATION/'predictions.csv')}},exclusive=True)
        save(PUBLIC/'control_reuse.json',{'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in (QUALITY/'predictions.csv',REPRESENTATION/'predictions.csv')},
            'controls_csv_sha256':digest(PUBLIC/'controls.csv'),'filters':{'arm':'validation_unseen','quality_models':list(STRONG_CONTROLS[:-1]),
            'selected_context_alpha_by_rotation':selected,'representation_model':'context_kernel','representation_fixed_alpha':.1},
            'rows':1800,'exact_selected_cache_projection_verified':True,'kernel_selected_rows_exactly_reused':True},exclusive=True)
        self.log('cached_targets_and_controls_joined',cached_targets=1260,source_gate_sha256=digest(gate))

    def event(self,item):
        kind=item['type']
        if kind=='source_prepared':
            data={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in item.items()}
            self.normalizers.append(data);save(PUBLIC/'source_normalizers.json',{'rotations':self.normalizers})
        elif kind=='fit_begin':
            if self.current is not None:self.current['status']='completed';self.current['completed_utc']=now()
            k=item['rotation'];source=[(r['subject'],r['vid']) for r in self.programme['rotation_manifests'][k]['arms']['source_fit']]
            median=float(np.median([self.targets[p] for p in source]))
            offset=[median if item['readout']=='eeg_only' else self.oof[k][p] for p in source]
            baseline=float(np.mean((np.array([self.targets[p] for p in source])-offset)**2))
            self.current={**{k:v for k,v in item.items() if k!='type'},'status':'running','started_utc':now(),'history':[],'checkpoints':{},'starting_control_mse':baseline}
            self.fit_records.append(self.current);save(PUBLIC/'fit_records.json',{'fits':self.fit_records})
            print({'phase':'fit_begin','fit':len(self.fit_records),**item},flush=True)
        elif kind=='epoch':
            self.current['history'].append(item['history'])
            if item['epoch']%10==0:save(PUBLIC/'fit_records.json',{'fits':self.fit_records})
        elif kind=='checkpoint':
            k=self.current['rotation'];stem=f"r{k}_{self.current['readout']}_lr{self.current['learning_rate']}_s{self.current['seed']}_e{item['epoch']}"
            path=PRIVATE/(stem+'.pt')
            with path.open('xb') as stream:torch.save(item['state_dict'],stream)
            restored=torch.load(path,map_location='cpu',weights_only=True)
            if set(restored)!=set(item['state_dict']) or any(not torch.equal(restored[k],v) for k,v in item['state_dict'].items()):
                raise ValueError('Serialized checkpoint tensors differ')
            peak=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
            allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            self.current['peak_cuda_reserved_bytes']=max(peak,self.current.get('peak_cuda_reserved_bytes',0))
            self.current['peak_cuda_allocated_bytes']=max(allocated,self.current.get('peak_cuda_allocated_bytes',0))
            cp={'training_target_source_mse':item['training_target_source_mse'],'source_inference_mse':item['source_inference_mse'],'state_dict_sha256':digest(path),'private_state_file':path.name,
                'serialized_checkpoint_tensors_match':True,'peak_cuda_reserved_bytes':peak,'peak_cuda_allocated_bytes':allocated}
            self.current['checkpoints'][str(item['epoch'])]=cp
            for arm in ('source_fit','source_inference','validation_unseen','validation_familiar'):
                original_arm='source_fit' if arm=='source_inference' else arm
                pairs=[(r['subject'],r['vid']) for r in self.programme['rotation_manifests'][k]['arms'][original_arm]]
                values=item[arm+'_raw']
                if len(values)!=len(pairs) or not np.isfinite(values).all():raise ValueError('Invalid complete checkpoint predictions')
                for p,value in zip(pairs,values):
                    self.checkpoint_writer.writerow({'rotation':k,'readout':self.current['readout'],'learning_rate':self.current['learning_rate'],'seed':self.current['seed'],
                        'epoch':item['epoch'],'arm':arm,'subject':p[0],'vid':p[1],'family':self.metadata[p]['family'],'acquisition_group':self.metadata[p]['acquisition_group'],
                        'target':self.targets[p],'prediction_raw':float(value),'eeg_available':self.decisions[p]['eeg_available'],'fallback':not self.decisions[p]['eeg_available']})
                    self.rows_checkpoint+=1
            self.checkpoint_file.flush();save(PUBLIC/'fit_records.json',{'fits':self.fit_records})
            print({'phase':'checkpoint','fit':len(self.fit_records),'epoch':item['epoch'],'source_mse':cp['training_target_source_mse'],'elapsed_seconds':round(time.monotonic()-self.started,1)},flush=True)
        elif kind=='selection':self.selections[item['readout']]=item['chosen'];save(PUBLIC/'selection.json',{'selection':self.selections,'events':[item]})
        elif kind in ('ensemble_predictions','source_inference_predictions'):
            k=item['rotation'];readout=item['readout'];choice=self.selections[readout]
            arm='source_inference' if kind=='source_inference_predictions' else item['arm']
            for row in item['rows']:
                p=row['subject'],row['vid']
                self.ensemble_writer.writerow({'rotation':k,'readout':readout,'selected_learning_rate':choice['learning_rate'],'selected_epoch':choice['epoch'],
                    'arm':arm,**row,'acquisition_group':self.metadata[p]['acquisition_group'],'eeg_available':self.decisions[p]['eeg_available'],'fallback':not self.decisions[p]['eeg_available']})
                self.rows_ensemble+=1
            self.ensemble_file.flush()
        elif kind=='failure':
            if self.current is not None:self.current['status']='failed';self.current['failure']=item['error']
        simple={k:v for k,v in item.items() if k not in ('type','state_dict','rows','source_pairs','residual_oof_ledger','channel_mean_uv','channel_sd_uv') and not isinstance(v,np.ndarray)}
        self.log(kind,**simple)
        save(PUBLIC/'progress.json',{'updated_utc':now(),'phase':kind,'fits_started':len(self.fit_records),
            'current_fit':None if self.current is None else {k:self.current[k] for k in ('rotation','readout','learning_rate','seed','status')},
            'current_epoch':None if self.current is None else len(self.current['history']),'checkpoint_rows':self.rows_checkpoint,'ensemble_rows':self.rows_ensemble,
            'elapsed_seconds':time.monotonic()-self.started,'confirmation_access':False,'research_question_changed':False})

    def execute(self):
        result=None;failure=None;budget=FitBudget()
        try:
            self.log('source_replay_started')
            gate=self.bridge.prepare_source();self.authenticate_after('source');gate['completed_utc']=now()
            save(PUBLIC/'source_gate.json',gate,exclusive=True);self.log('source_gate_passed',source_trials=980)
            self.bridge.prepare_validation();self.authenticate_after('validation')
            save(PUBLIC/'feature_manifest.json',{'records':self.bridge.receipts,'source_trials':980,'validation_trials':280,'validation_eeg_available':265,
                'validation_eeg_unavailable':15,'windows_in_memory':1245,'window_bytes_in_memory':self.bridge.bytes,'raw_waveforms_persisted':False},exclusive=True)
            self.open_cached();self.preparing=False
            with threadpool_limits(limits=2,user_api='blas'):
                result=run_diagnostic(self.programme,self.metadata,self.decisions,self.bridge.windows,self.targets,self.context,self.controls,
                    activation=self.activation,expected_plan_sha256=digest(PREFLIGHT/'plan.json'),code_bindings=code_bindings(),device='cuda:0',event=self.event,budget=budget)
            self.current['status']='completed';self.current['completed_utc']=now();save(PUBLIC/'fit_records.json',{'fits':self.fit_records})
            if self.rows_checkpoint!=84000 or self.rows_ensemble!=8400:raise ValueError('Incomplete fixed prediction denominator')
            self.log('diagnostic_completed',fits=20)
        except Exception as error:
            failure=type(error).__name__+': '+str(error)
            if self.current is not None and self.current['status']=='running':self.current['status']='failed';self.current['failure']=failure
            save(PUBLIC/'fit_records.json',{'fits':self.fit_records})
            save(PUBLIC/'incomplete_feature_manifest.json',{'records':self.bridge.receipts,'source_passed':self.bridge.source_passed,'cached_targets_joined':len(self.targets)},exclusive=True)
            self.log('execution_failed',failure=failure)
        finally:
            # Cover a failed CUDA synchronization that prevented core cleanup.
            budget.end()
            self.checkpoint_file.close();self.ensemble_file.close();self.event_file.close()
            execution={**(result or {}),'status':'completed' if failure is None else 'held_or_failed','failure':failure,
                'activation_sha256':digest(PUBLIC/'activation.json'),'completed_utc':now(),'new_neural_fits':budget.fits,
                'completed_neural_fits':sum(r['status']=='completed' for r in self.fit_records),'charged_fit_seconds':budget.elapsed,
                'source_trials_prepared':sum(r['role']=='source' for r in self.bridge.receipts),'validation_trials_prepared':sum(r['role']=='validation' for r in self.bridge.receipts),
                'selected_sample_values_decoded':self.sample_values,'checkpoint_rows':self.rows_checkpoint,'ensemble_rows':self.rows_ensemble,
                'new_original_rating_decoding':0,'original_rating_files_opened':False,'auxiliary_neighbor_or_confirmation_samples_decoded':0,
                'raw_waveforms_persisted':False,'peak_cuda_reserved_bytes':max([r.get('peak_cuda_reserved_bytes',0) for r in self.fit_records]+[torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0]),
                'runtime':{**runtime(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
                           'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
                'elapsed_seconds':time.monotonic()-self.started,'confirmation_access':False,'research_question_changed':False,
                'valid_contribution_selection':False,'scope':'Adaptive development diagnostic only; no independent confirmation or manuscript adoption.'}
            save(PUBLIC/'execution.json',execution,exclusive=True)
            print({'status':execution['status'],'fits':budget.fits,'failure':failure,'elapsed_seconds':execution['elapsed_seconds']},flush=True)
        if failure:raise RuntimeError(failure)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['declare','check','execute']);args=parser.parse_args()
    if args.action=='declare':declare()
    elif args.action=='check':check();print({'frozen_activation_verified':True})
    else:
        if (PUBLIC/'execution.json').exists() or (PUBLIC/'checkpoint_predictions.csv').exists():raise PermissionError('Refuse rerun or partial-panel resume')
        activation=check();torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('Qualified CUDA device required')
        try:study=Execution(activation)
        except Exception as error:
            save(PUBLIC/'execution.json',{'status':'setup_failed','failure':type(error).__name__+': '+str(error),
                'new_neural_fits':0,'original_rating_files_opened':False,'new_original_rating_decoding':0,
                'source_trials_prepared':0,'validation_trials_prepared':0,'confirmation_access':False,'research_question_changed':False},exclusive=True)
            raise
        study.execute()


if __name__=='__main__':main()
