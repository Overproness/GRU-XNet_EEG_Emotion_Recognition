"""Four-method FACED phase guard. No activation is supplied by this module.

Qualification uses manufactured streams or previously qualified source arrays;
the future production adapter preserves confirmation roles and both arms. An
activation document is an explicit author instruction, not a scientific gate.
Python checks are workflow safeguards rather than an OS security boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from types import MappingProxyType

import numpy as np

from .faced_classical_contract import ACQUISITIONS, CATEGORIES
from .faced_coverage_missing import ENGINEERED_MISSING_REASONS
from .faced_frozen_inference import (FrozenBundle, ALPHA, METHODS, POLICY,
    canonical_bytes, sha256_bytes, array_sha256)
from .faced_proposal_preprocessing import COMMON_CHANNELS, common_indices
from .faced_temporal_coverage import REGIONS, segment_bounds

VERSION = 'faced-guarded-four-method-confirmation-v1'
ARMS = ('confirmation_unseen', 'confirmation_familiar')
READINESS = ('source_states', 'source_export', 'executor_qualification',
             'claim_baseline_scope', 'precision_limitations', 'independent_preaccess')
CAPS = {'wall_seconds': 10800, 'cpu_seconds': 10800,
        'private_memory_bytes': 2 * 1024**3, 'checkpoint_gap_seconds': 5,
        'integrity_chunk_bytes': 1024**2}
_SHA = re.compile('[0-9a-f]{64}\\Z')
_PRED_KEYS = {'subject','vid','role','arm','family','assigned_category','acquisition_group',
             'model','alpha','prediction','shared_eeg_available','common_category_fallback',
             'routing','qc_sha256','input_sha256','state_sha256'}
_SCOPE_ROW_KEYS={'subject','vid','trial','role','arm','family','assigned_category',
    'catalogue_duration_seconds','acquisition_group','native_rate_hz','recording_samples',
    'marker_start_seconds','marker_end_seconds','channel_indices','common_channels','segments',
    'original_data_sha256','original_header_sha256'}


def hash_json(value):
    return sha256_bytes(canonical_bytes(value))


def _hash(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError('Expected lowercase SHA256')
    return value


def _finite(value, label):
    if isinstance(value, (bool, str)) or not isinstance(value, (int,float,np.number)):
        raise ValueError('Require numeric '+label)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite '+label+' is fatal')
    return value


def _relative(value):
    if not isinstance(value, str) or '\\' in value or ':' in value:
        raise PermissionError('Require a relative slash path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts or str(p) != value:
        raise PermissionError('Unsafe relative path')
    return value


def verify_bindings(root, bindings, *, cap=8*1024**2):
    """Read only declared public/code receipts; never an original recording."""
    root = Path(root).resolve()
    for name, expected in bindings.items():
        _relative(name); _hash(expected)
        if not name.startswith(('gruxnet/','scripts/','tests/','docs/','results/development/')):
            raise PermissionError('Original/private path is not a public binding')
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise PermissionError('Escaping binding path')
        with path.open('rb') as stream:
            payload = stream.read(cap+1)
        if len(payload)>cap or sha256_bytes(payload)!=expected:
            raise PermissionError('Public byte binding changed: '+name)


def validate_scope(scope):
    """Complete identity/range/file barrier; this performs no file operation."""
    if (scope.get('scope_version') != 'faced-inactive-confirmation-three-regions-four-models-v1'
            or scope.get('models') != list(METHODS) or scope.get('trials') != 759
            or scope.get('regions') != 2277 or scope.get('required_prediction_count') != 3036
            or scope.get('participant_count') != 33):
        raise PermissionError('Exact inactive four-method scope required')
    rows = scope.get('rows'); files = scope.get('files')
    if not isinstance(rows,list) or len(rows)!=759 or not isinstance(files,list) or len(files)!=66:
        raise PermissionError('Incomplete exact roster/files')
    # All identities and roles precede range/duration numerical interpretation.
    pairs=set(); people=set(); titles={}; arm_pairs={a:set() for a in ARMS}
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=_SCOPE_ROW_KEYS or row.get('role')!='confirmation'
                or row.get('arm') not in ARMS or not isinstance(row.get('subject'),str)
                or not re.fullmatch('sub-[0-9]{3}',row['subject'])
                or type(row.get('vid')) is not int):
            raise PermissionError('Malformed exact confirmation identity/role/arm')
        p=(row['subject'],row['vid'])
        if p in pairs:
            raise PermissionError('Duplicate trial')
        pairs.add(p); people.add(p[0]); arm_pairs[row['arm']].add(p)
        if (not isinstance(row.get('family'),str) or not _SHA.fullmatch(row['family'])
                or row.get('assigned_category') not in CATEGORIES
                or row.get('acquisition_group') not in ACQUISITIONS):
            raise PermissionError('Malformed family/context catalogue')
        definition=(row['family'],row['assigned_category'])
        if p[1] in titles and titles[p[1]]!=definition:
            raise PermissionError('Catalogue identity varies across people')
        titles[p[1]]=definition
    dev={2,3,4,6,9,11,12,17,18,20,23,25,27,28}; unseen={1,5,7,8,10,19,21,24,26}
    if (len(people)!=33 or arm_pairs[ARMS[0]]!={(p,v) for p in people for v in unseen}
            or arm_pairs[ARMS[1]]!={(p,v) for p in people for v in dev}
            or len({titles[v][0] for v in unseen})!=8 or titles[7][0]!=titles[8][0]
            or len({titles[v][0] for v in dev})!=14
            or {titles[v][0] for v in dev}&{titles[v][0] for v in unseen}):
        raise PermissionError('Exact participant/material rectangle changed')
    file_index={}
    for f in files:
        key=(f.get('subject'),f.get('name'))
        if (key in file_index or key[0] not in people or key[1] not in ('data.bdf','evt.bdf')
                or f.get('role')!='confirmation' or type(f.get('bytes')) is not int or f['bytes']<=0):
            raise PermissionError('Malformed file inventory')
        expected='Data/Data/'+key[0].replace('sub-','sub')+'/'+key[1]
        if _relative(f.get('workspace_relative_path'))!=expected:
            raise PermissionError('Exact recording filepath differs')
        _hash(f.get('sha256'))
        if key[1]=='data.bdf': _hash(f.get('data_header_sha256'))
        file_index[key]=f
    if set(file_index)!={(p,n) for p in people for n in ('data.bdf','evt.bdf')}:
        raise PermissionError('Incomplete66file roster')
    rates={}; ordinals=set(); counts=reads=0
    for row in rows:
        rate=row.get('native_rate_hz'); total=row.get('recording_samples')
        if type(rate) is not int or rate not in (250,1000) or type(total) is not int or total<=0:
            raise ValueError('Malformed native geometry')
        if row['subject'] in rates and rates[row['subject']]!=rate:
            raise PermissionError('Participant native rate varies')
        rates[row['subject']]=rate
        if (type(row.get('trial')) is not int or not 1<=row['trial']<=28
                or (row['subject'],row['trial']) in ordinals):
            raise PermissionError('Invalid or duplicate presentation ordinal')
        ordinals.add((row['subject'],row['trial']))
        if _finite(row['catalogue_duration_seconds'],'duration')<=0:
            raise ValueError('Nonpositive catalogue duration')
        indices=row.get('channel_indices')
        if (not isinstance(indices,list) or len(indices)!=30 or len(set(indices))!=30
                or any(type(i) is not int or not 0<=i<32 for i in indices)
                or row.get('common_channels')!=list(COMMON_CHANNELS)
                or row.get('segments')!=segment_bounds(row['marker_start_seconds'],row['marker_end_seconds'],rate,total)):
            raise PermissionError('Exact selected channel/range declaration changed')
        original=file_index[row['subject'],'data.bdf']
        if (row.get('original_data_sha256')!=original['sha256']
                or row.get('original_header_sha256')!=original['data_header_sha256']
                or original['bytes']!=8448+total*32*3):
            raise PermissionError('Trial/file integrity geometry differs')
        counts+=30*sum(b-a for a,b in row['segments'].values())
        reads+=30*sum((b-1)//rate-a//rate+1 for a,b in row['segments'].values())
    r=scope.get('resource_scope',{})
    data=sum(f['bytes'] for f in files if f['name']=='data.bdf')
    events=sum(f['bytes'] for f in files if f['name']=='evt.bdf')
    if (r.get('selected_count_values')!=counts or r.get('selected_signal_payload_bytes')!=3*counts
            or r.get('selected_channel_payload_read_calls')!=reads
            or r.get('two_data_plus_two_event_hash_transport_bytes')!=2*(data+events)
            or r.get('data_header_bytes_one_read_pass')!=33*8448
            or r.get('behavior_file_bytes_authorized')!=0 or r.get('event_annotation_decodes_authorized')!=0):
        raise PermissionError('Exact resource metadata differs')
    return tuple(rows),file_index


def declaration(scope, package_sha256, bundle_fingerprint, runtime, bindings):
    validate_scope(scope); _hash(package_sha256); _hash(bundle_fingerprint)
    for name,digest in bindings.items(): _relative(name); _hash(digest)
    return {'version':VERSION,'status':'inactive_not_authorized','scope_sha256':hash_json(scope),
            'package_sha256':package_sha256,'bundle_fingerprint':bundle_fingerprint,
            'runtime':runtime,'bindings':dict(bindings),'resource_caps':dict(CAPS),
            'author':'Muhammad Muntazar','signal_access_authorized':False,
            'rating_access_authorized':False,'new_fits':0,'research_question_changed':False}


class Budget:
    """Cooperative check; a long foreign-library call is checked after return."""
    def __init__(self, resource_scope, *, clock=time.monotonic, cpu=time.process_time,
                 memory=None, caps=None):
        self.clock=clock; self.cpu=cpu; self.memory=memory or private_memory_bytes
        self.caps=dict(CAPS if caps is None else caps)
        self.start=clock(); self.cpu_start=cpu(); self.last=self.start
        self.counts={'integrity_bytes':0,'header_bytes':0,'signal_bytes':0,'count_values':0,'read_calls':0}
        self.limits={'integrity_bytes':resource_scope['two_data_plus_two_event_hash_transport_bytes'],
            'header_bytes':resource_scope['data_header_bytes_one_read_pass'],
            'signal_bytes':resource_scope['selected_signal_payload_bytes'],
            'count_values':resource_scope['selected_count_values'],
            'read_calls':resource_scope['selected_channel_payload_read_calls']}
        self.check()

    def check(self):
        now=self.clock(); consumed=self.cpu()-self.cpu_start
        if (now<self.last or consumed<0 or now-self.start>self.caps['wall_seconds']
                or consumed>self.caps['cpu_seconds'] or now-self.last>self.caps['checkpoint_gap_seconds']):
            raise TimeoutError('Wall/CPU/cooperative checkpoint ceiling exceeded')
        mem=self.memory()
        if type(mem) is not int or mem<0 or mem>self.caps['private_memory_bytes']:
            raise MemoryError('Process-private-memory ceiling exceeded')
        self.last=now

    def consume(self, name, amount):
        self.check()
        if name not in self.counts or type(amount) is not int or amount<0 or self.counts[name]+amount>self.limits[name]:
            raise PermissionError('Exact byte/count/read cap exceeded')
        self.counts[name]+=amount

    def finish(self):
        self.check()
        if self.counts!=self.limits:
            raise PermissionError('Incomplete exact resource denominator')


def private_memory_bytes():
    import psutil
    info=psutil.Process().memory_info()
    if os.name=='nt':
        if not hasattr(info,'private'):
            raise RuntimeError('Windows private-byte counter unavailable')
        return int(info.private)
    full=psutil.Process().memory_full_info()
    if not hasattr(full,'uss'):
        raise RuntimeError('Qualified private-memory counter unavailable')
    return int(full.uss)


def _activation(contract, scope, bundle, activation, receipts, runtime, phase, root):
    """No original reader may run until every explicit binding/receipt passes."""
    validate_scope(scope)
    if (contract.get('version')!=VERSION or contract.get('status')!='inactive_not_authorized'
            or contract.get('signal_access_authorized') is not False
            or contract.get('rating_access_authorized') is not False
            or contract.get('scope_sha256')!=hash_json(scope)
            or not isinstance(bundle,FrozenBundle) or contract.get('bundle_fingerprint')!=bundle.fingerprint
            or contract.get('runtime')!=runtime or contract.get('resource_caps')!=CAPS):
        raise PermissionError('Inactive contract/scope/bundle/runtime changed')
    expected={'version','phase','contract_sha256','scope_sha256','package_sha256','bundle_fingerprint',
              'author','author_instruction','authorization_receipt_sha256','readiness_receipts',
              'authorization_receipt_path','signal_access_authorized','rating_access_authorized','single_attempt','new_fits'}
    if (not isinstance(activation,dict) or set(activation)!=expected or activation['version']!=VERSION
            or activation['phase']!=phase or activation['contract_sha256']!=hash_json(contract)
            or activation['scope_sha256']!=contract['scope_sha256']
            or activation['package_sha256']!=contract['package_sha256']
            or activation['bundle_fingerprint']!=bundle.fingerprint
            or activation['author']!='Muhammad Muntazar' or not isinstance(activation['author_instruction'],str)
            or not activation['author_instruction'].strip() or activation['single_attempt'] is not True
            or activation['new_fits']!=0 or type(activation['new_fits']) is not int
            or activation['signal_access_authorized'] is not (phase=='signal_prediction')
            or activation['rating_access_authorized'] is not (phase=='item10_rating')):
        raise PermissionError('Separate exact author/phase activation required')
    if set(receipts)!=set(READINESS) or set(activation['readiness_receipts'])!=set(READINESS):
        raise PermissionError('Complete new readiness receipts required')
    for name in READINESS:
        r=receipts[name]
        if (not isinstance(r,dict) or r.get('version')!=VERSION or r.get('scope_sha256')!=contract['scope_sha256']
                or r.get('contract_sha256')!=hash_json(contract) or r.get('passed') is not True
                or r.get('protected_data_accessed') is not False
                or hash_json(r)!=activation['readiness_receipts'][name]):
            raise PermissionError('Bound scope-specific readiness receipt missing/failed')
    _hash(activation['authorization_receipt_sha256'])
    if root is None:
        raise PermissionError('Bound public/code files must be verified before original callback')
    required = {'gruxnet/faced_guarded_confirmation.py',
                'gruxnet/faced_frozen_inference.py','gruxnet/faced_frozen_prior_adapter.py',
                'gruxnet/faced_source_quality.py','gruxnet/faced_artifact_control.py',
                'gruxnet/faced_source_notch.py','gruxnet/faced_proposal_preprocessing.py',
                'gruxnet/faced_inference_quality.py','gruxnet/faced_classical_contract.py',
                'gruxnet/faced_temporal_coverage.py','gruxnet/faced_coverage_missing.py',
                'results/development/faced_raw_2026-10-11/headers.json',
                'results/development/faced_confirmation_backend_2026-10-14/scope.json',
                'results/development/faced_confirmation_backend_2026-10-14/package.json'}
    if not required.issubset(contract['bindings']):
        raise PermissionError('Concrete executor/state/scope/package byte bindings required')
    verify_bindings(root,contract['bindings'])
    auth_path=activation['authorization_receipt_path']; _relative(auth_path)
    if not auth_path.startswith('docs/publication/'):
        raise PermissionError('Explicit saved public author instruction receipt required')
    verify_bindings(root,{auth_path:activation['authorization_receipt_sha256']})
    auth=json.loads((Path(root)/auth_path).read_bytes())
    if (not isinstance(auth,dict) or auth.get('version')!=VERSION or auth.get('author')!='Muhammad Muntazar'
            or auth.get('phase')!=phase or auth.get('contract_sha256')!=hash_json(contract)
            or auth.get('author_instruction')!=activation['author_instruction']
            or auth.get('explicit_author_access_approval') is not True):
        raise PermissionError('Saved author instruction does not approve this exact access phase')


class GuardedExecutor:
    """Production signal phase. The constructor does not touch originals."""
    def __init__(self, contract, scope, bundle, *, activation=None, receipts=None,
                 runtime=None, root=None, budget=None):
        self.contract=contract; self.scope=scope; self.bundle=bundle
        self.activation=activation; self.receipts=receipts or {}; self.runtime=runtime; self.root=root
        self.budget=budget; self._used=False

    def signals(self, reader):
        _activation(self.contract,self.scope,self.bundle,self.activation,self.receipts,self.runtime,
                    'signal_prediction',self.root)
        if self._used or not isinstance(reader,OriginalBDFReader) or getattr(reader,'input_kind',None)!='protected_original':
            raise PermissionError('Single production attempt and protected signal adapter required')
        expected_workspace=os.path.normcase(os.path.abspath(Path(self.root).parent))
        if os.path.normcase(os.path.abspath(reader.workspace))!=expected_workspace:
            raise PermissionError('Original reader workspace differs from the bound repository parent')
        reader.authenticate_projection(self.root,self.scope)
        self._used=True
        if self.budget is None: self.budget=Budget(self.scope['resource_scope'])
        self.budget._production_scope_sha256=hash_json(self.scope)
        return _signal_inference(self.scope,self.bundle,reader,self.budget,'protected_original')


def manufactured_scope_replay(scope,bundle,reader,*,budget):
    """Test exact future identities without changing their role to manufactured.

    The supplied features/streams must be fabricated. This returns a
    qualification export, which cannot satisfy a production rating gate.
    """
    if getattr(reader,'input_kind',None)!='manufactured':
        raise PermissionError('Manufactured replay cannot invoke source/protected original adapters')
    return _signal_inference(scope,bundle,reader,budget,'manufactured')


def _signal_inference(scope,bundle,reader,budget,input_kind):
    rows,files=validate_scope(scope)
    if not isinstance(bundle,FrozenBundle): raise TypeError('Require frozen bundle')
    budget.check()
    reader.preflight(scope,files,budget)
    features={}; quality={}; ledger=[]
    for row in rows:
        regions=[]; reasons=[]; qc=[]
        for region in REGIONS:
            budget.check()
            result=reader.region(row,region,budget)
            budget.check()
            if (not isinstance(result,dict) or set(result)!={'technical_passed','review_flags','features','receipt'}
                    or result['technical_passed'] is not True or not isinstance(result['review_flags'],list)):
                raise PermissionError('Technical/numerical failures are fatal, not engineered missing')
            for flag in result['review_flags']:
                if (not isinstance(flag,dict) or set(flag)!={'channel','reason'}
                        or flag['channel'] not in tuple(COMMON_CHANNELS)+('all',)
                        or flag['reason'] not in ENGINEERED_MISSING_REASONS):
                    raise PermissionError('Unknown/nonfinite QC failure cannot become missing')
            value=np.asarray(result['features'])
            if value.dtype!=np.float64 or value.shape!=(150,) or not np.isfinite(value).all():
                raise ValueError('Regional features must be finite float64/150')
            receipt=result['receipt']
            if (not isinstance(receipt,dict) or receipt.get('subject')!=row['subject'] or receipt.get('vid')!=row['vid']
                    or receipt.get('role')!='confirmation' or receipt.get('arm')!=row['arm']
                    or receipt.get('region')!=region or receipt.get('sample_interval')!=row['segments'][region]
                    or receipt.get('input_kind')!=input_kind or receipt.get('extra_boundary_samples_decoded')!=0
                    or receipt.get('feature_sha256')!=array_sha256(value)):
                raise PermissionError('Exact role/arm/range/feature receipt required')
            regions.append(value.copy()); reasons.extend(result['review_flags']); qc.append(receipt); ledger.append(receipt)
        pair=(row['subject'],row['vid']); quality[pair]=(not reasons,hash_json(qc))
        if not reasons:
            features[pair]=trial_features(row,regions)
    reader.finish(scope,files,budget); budget.finish()
    result=[]
    for row in rows:
        budget.check(); pair=(row['subject'],row['vid']); available,qc_hash=quality[pair]
        prior=bundle.category_prior[row['assigned_category']]
        for model in METHODS:
            fallback=model!='category_median' and not available
            value=prior if model=='category_median' or fallback else float(bundle.states[model].predict(features[pair][model][None])[0])
            result.append({k:row[k] for k in ('subject','vid','role','arm','family','assigned_category','acquisition_group')}|
                {'model':model,'alpha':None if model=='category_median' else ALPHA,'prediction':value,
                 'shared_eeg_available':available,'common_category_fallback':fallback,
                 'routing':'category_prior' if model=='category_median' else ('common_category_fallback' if fallback else 'frozen_ridge'),
                 'qc_sha256':qc_hash,'input_sha256':None if model=='category_median' or fallback else array_sha256(features[pair][model]),
                 'state_sha256':bundle.prior_sha256 if model=='category_median' else bundle.states[model].state_sha256})
    validate_predictions(result,scope,bundle)
    return result,ledger


def trial_features(row, regions):
    """Same algebra for source and protected rows; never rewrite their role."""
    if row.get('role') not in ('source','confirmation'):
        raise PermissionError('Explicit source or confirmation metadata required')
    if len(regions)!=3:
        raise ValueError('Exactly three regions required')
    regional=[np.asarray(v) for v in regions]
    if any(v.dtype!=np.float64 or v.shape!=(150,) or not np.isfinite(v).all() for v in regional):
        raise ValueError('Finite float64 regional vectors required')
    if row['assigned_category'] not in CATEGORIES or row['acquisition_group'] not in ACQUISITIONS:
        raise PermissionError('Qualified context fields required')
    context=np.asarray([float(row['assigned_category']==c) for c in CATEGORIES]
        +[float(row['acquisition_group']==a) for a in ACQUISITIONS]
        +[_finite(row['catalogue_duration_seconds'],'duration'),row['trial']],dtype=np.float64)
    distributed=np.stack(regional).mean(axis=0)
    if not np.isfinite(distributed).all(): raise ValueError('Nonfinite distributed feature')
    return {'context_only':context,'late_joint':np.r_[context,regional[2]],
            'distributed_joint':np.r_[context,distributed]}


def validate_predictions(predictions,scope,bundle):
    rows,_=validate_scope(scope); lookup={(r['subject'],r['vid']):r for r in rows}; seen=set(); grouped={}
    if not isinstance(predictions,list) or len(predictions)!=3036: raise PermissionError('Complete3036predictions required')
    # Entire schema/identity pass precedes any prediction conversion.
    for p in predictions:
        if not isinstance(p,dict) or set(p)!=_PRED_KEYS: raise PermissionError('Exact target-free prediction schema required')
        pair=p['subject'],p['vid']; key=(*pair,p['model'])
        if pair not in lookup or key in seen or p['model'] not in METHODS: raise PermissionError('Duplicate/undeclared prediction')
        seen.add(key); metadata=lookup[pair]
        if any(p[k]!=metadata[k] for k in ('role','arm','family','assigned_category','acquisition_group')):
            raise PermissionError('Prediction metadata changed or protected role masked')
        grouped.setdefault(pair,[]).append(p)
    if seen!={(*pair,m) for pair in lookup for m in METHODS}: raise PermissionError('Incomplete denominator')
    for pair, ps in grouped.items():
        available=ps[0]['shared_eeg_available']; qhash=ps[0]['qc_sha256']; _hash(qhash)
        prior=bundle.category_prior[lookup[pair]['assigned_category']]
        for p in ps:
            m=p['model']; value=_finite(p['prediction'],'prediction')
            if not 0<=value<=7 or type(p['shared_eeg_available']) is not bool or p['shared_eeg_available'] is not available or p['qc_sha256']!=qhash:
                raise PermissionError('Invalid value/shared QC')
            fallback=m!='category_median' and not available
            if type(p['common_category_fallback']) is not bool or p['common_category_fallback'] is not fallback:
                raise PermissionError('Shared fallback changed')
            if p['alpha']!=(None if m=='category_median' else ALPHA): raise PermissionError('Changed alpha')
            expected=bundle.prior_sha256 if m=='category_median' else bundle.states[m].state_sha256
            if p['state_sha256']!=expected: raise PermissionError('State hash changed')
            if m=='category_median' or fallback:
                if value!=prior or p['input_sha256'] is not None: raise PermissionError('Category fallback mismatch')
            else: _hash(p['input_sha256'])
            route='category_prior' if m=='category_median' else ('common_category_fallback' if fallback else 'frozen_ridge')
            if p['routing']!=route: raise PermissionError('Prediction routing changed')
    return sorted(predictions,key=lambda p:(p['arm'],p['subject'],p['vid'],METHODS.index(p['model'])))


def write_lock(directory,predictions,scope,bundle,*,kind,contract_sha256):
    if kind not in ('manufactured_qualification','protected_prediction'): raise PermissionError('Explicit lock provenance required')
    _hash(contract_sha256); canonical=validate_predictions(predictions,scope,bundle)
    root=Path(directory)
    reservation=root.parent/('.'+root.name+'.reservation')
    # Atomic exclusive reservation preserves a failed attempt and prevents retry.
    with reservation.open('xb') as stream: stream.write(canonical_bytes({'kind':kind,'contract_sha256':contract_sha256}))
    root.mkdir(exist_ok=False)
    payload=canonical_bytes(canonical)
    manifest={'version':VERSION,'kind':kind,'contract_sha256':contract_sha256,'scope_sha256':hash_json(scope),
              'bundle_fingerprint':bundle.fingerprint,'predictions':3036,'trials':759,'regions':2277,
              'arms':{a:sum(p['arm']==a for p in canonical) for a in ARMS},'target_values':0,
              'prediction_bytes':len(payload),'prediction_sha256':sha256_bytes(payload)}
    for name,data in (('predictions.json',payload),('manifest.json',canonical_bytes(manifest))):
        with (root/name).open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    complete={'manifest_sha256':hash_json(manifest),'prediction_sha256':manifest['prediction_sha256']}
    with (root/'LOCK.complete.json').open('xb') as stream: stream.write(canonical_bytes(complete)); stream.flush(); os.fsync(stream.fileno())
    return manifest


def verify_lock(directory,scope,bundle,*,kind,contract_sha256):
    root=Path(directory)
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()}!={'predictions.json','manifest.json','LOCK.complete.json'}:
        raise PermissionError('Complete exact lock directory required')
    parts={}
    for name in ('predictions.json','manifest.json','LOCK.complete.json'):
        path=root/name
        if path.is_symlink(): raise PermissionError('Lock symlink denied')
        with path.open('rb') as stream: payload=stream.read(8*1024**2+1)
        if len(payload)>8*1024**2: raise PermissionError('Lock exceeds size cap')
        value=json.loads(payload)
        if canonical_bytes(value)!=payload: raise PermissionError('Noncanonical lock bytes')
        parts[name]=(value,payload)
    predictions,payload=parts['predictions.json']; manifest,_=parts['manifest.json']; complete,_=parts['LOCK.complete.json']
    expected={'version':VERSION,'kind':kind,'contract_sha256':contract_sha256,'scope_sha256':hash_json(scope),
              'bundle_fingerprint':bundle.fingerprint,'predictions':3036,'trials':759,'regions':2277,
              'arms':{a:sum(p['arm']==a for p in predictions) for a in ARMS},'target_values':0,
              'prediction_bytes':len(payload),'prediction_sha256':sha256_bytes(payload)}
    if manifest!=expected or complete!={'manifest_sha256':hash_json(manifest),'prediction_sha256':manifest['prediction_sha256']}:
        raise PermissionError('Lock manifest/hash mismatch')
    if canonical_bytes(validate_predictions(predictions,scope,bundle))!=payload: raise PermissionError('Lock identity/order mismatch')
    return manifest,predictions


def read_item10(contract,scope,bundle,activation,receipts,runtime,root,lock_directory,
                lock_qualification,commit_receipt,reader):
    """Future separately authorized rating callback; no supplied activation here.

    Commit receipt must be produced by an independently qualified Git verifier,
    never by this exporter. Every lock and entire target-free roster passes first.
    """
    _activation(contract,scope,bundle,activation,receipts,runtime,'item10_rating',root)
    manifest,predictions=verify_lock(lock_directory,scope,bundle,kind='protected_prediction',contract_sha256=hash_json(contract))
    if (lock_qualification.get('passed') is not True or lock_qualification.get('independent') is not True
            or lock_qualification.get('manifest_sha256')!=hash_json(manifest)
            or commit_receipt.get('verified_git_blob_bytes') is not True
            or not isinstance(commit_receipt.get('commit'),str) or not re.fullmatch('[0-9a-f]{40}',commit_receipt['commit'])
            or commit_receipt.get('prediction_sha256')!=manifest['prediction_sha256']
            or commit_receipt.get('manifest_sha256')!=hash_json(manifest)
            or getattr(reader,'input_kind',None)!='protected_item10_only'):
        raise PermissionError('Independent complete committed prediction lock required before rating callback')
    checked=verify_committed_lock(root,lock_directory,commit_receipt['commit'],manifest)
    if commit_receipt!=checked:
        raise PermissionError('Caller commit receipt differs from actual Git blob bytes')
    rows,_=validate_scope(scope); targets=[]
    for row in rows:
        value=reader.item10(dict(row),item_one_based=10)
        value=_finite(value,'item10 rating')
        if not 0<=value<=7: raise ValueError('Invalid original0-7 item10 target; no imputation/drop')
        targets.append({'subject':row['subject'],'vid':row['vid'],'role':row['role'],'arm':row['arm'],'target':value})
    return targets,predictions


def verify_committed_lock(repo,lock_directory,commit,manifest):
    """Verify actual local Git blobs; a caller-provided flag is insufficient."""
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):
        raise PermissionError('Exact Git commit SHA required')
    root=Path(repo).resolve(); directory=Path(lock_directory).resolve()
    if not directory.is_relative_to(root): raise PermissionError('Protected lock must be within bound repository')
    relative=directory.relative_to(root).as_posix(); _relative(relative)
    for name in ('predictions.json','manifest.json','LOCK.complete.json'):
        path=relative+'/'+name
        # Inspect size first; the complete prediction blob is bounded to8MiB.
        spec=commit+':'+path
        size=subprocess.run(['git','cat-file','-s',spec],cwd=root,check=True,capture_output=True,timeout=30)
        if not size.stdout.strip().isdigit() or int(size.stdout)>8*1024**2:
            raise PermissionError('Committed lock blob exceeds size cap')
        output=subprocess.run(['git','show',spec],cwd=root,check=True,capture_output=True,timeout=30).stdout
        local=(directory/name).read_bytes()
        if output!=local: raise PermissionError('Committed lock bytes differ from live complete lock')
    return {'commit':commit,'verified_git_blob_bytes':True,'prediction_sha256':manifest['prediction_sha256'],
            'manifest_sha256':hash_json(manifest)}


class _MeteredStream:
    def __init__(self,stream,budget): self.stream=stream; self.budget=budget
    def seek(self,*args): self.budget.check(); return self.stream.seek(*args)
    def read(self,n):
        self.budget.consume('read_calls',1); self.budget.consume('signal_bytes',n)
        payload=self.stream.read(n); self.budget.check()
        if len(payload)!=n: raise ValueError('Truncated selective BDF bytes')
        return payload


class OriginalBDFReader:
    """Future protected signal adapter: opaque integrity only for evt.bdf.

    Constructor performs no stat/open/resolve operation on any original path.
    Exact public technical headers must be separately bound by the contract.
    """
    input_kind='protected_original'
    def __init__(self,workspace,headers):
        self.workspace=Path(workspace); self.headers=headers; self._preflight=False; self._finished=False
        self._files={}; self._snapshots={}; self._seen=set()

    def authenticate_projection(self,repo,scope):
        # The executor has already byte-verified this public file. An arbitrary
        # caller dictionary with a copied header_sha256 must not substitute a
        # changed calibration or channel projection.
        packet=json.loads((Path(repo)/'results/development/faced_raw_2026-10-11/headers.json').read_bytes())
        people={r['subject'] for r in scope['rows']}
        records=[r for r in packet['records'] if r['subject'] in people]
        if len(records)!=33 or len({r['subject'] for r in records})!=33:
            raise PermissionError('Complete bound protected header projection required')
        bound={r['subject']:r['data.bdf'] for r in records}
        if self.headers!=bound: raise PermissionError('Caller header projection differs from actual bound public bytes')
        validate_public_headers(scope,self.headers)

    def _path(self,f):
        # Path checks occur only after the production executor activation barrier.
        p=self.workspace/_relative(f['workspace_relative_path']); root=self.workspace.resolve()
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            raise PermissionError('Original is absent or escapes declared workspace')
        return p

    def _integrity(self,f,budget):
        p=self._path(f); stat=p.stat()
        if stat.st_size!=f['bytes']: raise ValueError('Original byte size differs')
        digest=hashlib.sha256(); total=0
        with p.open('rb') as stream:
            while total<f['bytes']:
                n=min(CAPS['integrity_chunk_bytes'],f['bytes']-total)
                budget.consume('integrity_bytes',n); payload=stream.read(n); budget.check()
                if len(payload)!=n: raise ValueError('Original integrity read truncated')
                digest.update(payload); total+=n
            if stream.read(1): raise ValueError('Original grew beyond inventory')
        if digest.hexdigest()!=f['sha256']: raise ValueError('Original integrity SHA differs')
        if (stat.st_size,stat.st_mtime_ns)!=(p.stat().st_size,p.stat().st_mtime_ns): raise ValueError('Original changed while hashing')
        return p,(stat.st_size,stat.st_mtime_ns)

    def preflight(self,scope,files,budget):
        if getattr(budget,'_production_scope_sha256',None)!=hash_json(scope):
            raise PermissionError('Original adapter requires the executor signal-phase permit')
        if self._preflight: raise PermissionError('Original adapter single attempt')
        self._preflight=True; self._files=files
        validate_public_headers(scope,self.headers)
        for key,f in sorted(files.items()):
            p,snapshot=self._integrity(f,budget); self._snapshots[key]=snapshot
            if key[1]=='data.bdf':
                budget.consume('header_bytes',8448)
                with p.open('rb') as stream: payload=stream.read(8448)
                budget.check()
                if len(payload)!=8448 or sha256_bytes(payload)!=f['data_header_sha256']:
                    raise ValueError('Exact raw header bytes differ')

    def region(self,row,region,budget):
        if not self._preflight or self._finished or region not in REGIONS: raise PermissionError('Original read outside signal phase')
        key=(row['subject'],row['vid'],region)
        if key in self._seen: raise PermissionError('Duplicate original regional read')
        self._seen.add(key); h=self.headers[row['subject']]
        f=self._files[row['subject'],'data.bdf']; p=self._path(f)
        if (p.stat().st_size,p.stat().st_mtime_ns)!=self._snapshots[row['subject'],'data.bdf']:
            raise ValueError('Original changed before selective read')
        with p.open('rb') as stream:
            return process_bdf_stream(stream,h,row,region,budget,input_kind=self.input_kind)

    def finish(self,scope,files,budget):
        if self._finished or len(self._seen)!=2277: raise PermissionError('Incomplete original region denominator')
        for key,f in sorted(files.items()):
            _,snapshot=self._integrity(f,budget)
            if snapshot!=self._snapshots[key]: raise ValueError('Original changed after processing')
        self._finished=True


def prepare_processing_runtime():
    """Import/preinitialize numerical code before the cooperative timer starts."""
    from .faced_artifact_control import artifact_trial_windows
    from .faced_classical_contract import spectral_features
    from .faced_inference_quality import eeg_availability
    from threadpoolctl import threadpool_limits


def validate_public_headers(scope,headers):
    """Validate all header projections before any original stat/open callback."""
    rows,files=validate_scope(scope); by_person={}
    for row in rows:
        if row['subject'] in by_person:
            first=by_person[row['subject']]
            if any(row[k]!=first[k] for k in ('channel_indices','native_rate_hz','recording_samples')):
                raise PermissionError('Participant channel/recording geometry varies')
        by_person[row['subject']]=row
    if set(headers)!=set(by_person): raise PermissionError('Exact33 public technical headers required')
    from .faced_proposal_preprocessing import UNIT_REPAIR_SUBJECTS
    for person,row in by_person.items():
        h=headers[person]; columns=h.get('columns',{})
        if (h.get('header_bytes')!=8448 or h.get('sample_width')!=3 or h.get('continuity') not in ('BDF','24BIT')
                or len(columns.get('labels',[]))!=32 or common_indices(columns['labels'])!=row['channel_indices']
                or h.get('header_sha256')!=files[person,'data.bdf']['data_header_sha256']
                or h.get('record_duration_seconds')!=1 or type(h.get('record_count')) is not int
                or h['record_count']*row['native_rate_hz']!=row['recording_samples']
                or len(h.get('sample_rates_hz',[]))!=32 or set(h['sample_rates_hz'])!={float(row['native_rate_hz'])}):
            raise PermissionError('Bound technical header projection differs')
        expected_units='?V' if person in UNIT_REPAIR_SUBJECTS else 'uV'
        # Auxiliary channel units may differ; only the selected scalp channels
        # are calibrated/qualified by the frozen adapter.
        indices=row['channel_indices']
        if any(len(columns.get(k,[]))!=32 for k in ('units','physical_min','physical_max','digital_min','digital_max','samples_per_record')):
            raise PermissionError('Incomplete32channel header columns')
        if {columns['units'][i] for i in indices}!={expected_units}:
            raise PermissionError('Qualified scalp unit literal changed')
        sizes=columns['samples_per_record']
        if any(isinstance(v,bool) or not(type(v) is int or isinstance(v,str) and v.isascii() and v.isdecimal()) for v in sizes):
            raise ValueError('Require literal native samples per record')
        if {int(v) for v in sizes}!={row['native_rate_hz']}:
            raise PermissionError('Header native rate differs')
        for lo,hi in (('physical_min','physical_max'),('digital_min','digital_max')):
            low=np.asarray(columns[lo],dtype=np.float64); high=np.asarray(columns[hi],dtype=np.float64)
            if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high<=low):
                raise ValueError('Invalid finite header calibration')
    return headers


def process_bdf_stream(stream,header,row,region,budget,*,input_kind):
    """Selective decoder/QC/feature path shared by manufactured and production.

    A stream is already open; this function never opens a path. Production use
    additionally requires the executor phase permit and prior file integrity.
    """
    if input_kind not in ('manufactured','protected_original') or region not in REGIONS:
        raise PermissionError('Explicit signal input kind and selected region required')
    if input_kind=='protected_original' and not getattr(budget,'_production_scope_sha256',None):
        raise PermissionError('Production stream requires signal-phase permit')
    if row.get('role')!='confirmation' or row.get('arm') not in ARMS:
        raise PermissionError('Preserve confirmation role and arm')
    from .faced_source_quality import read_interval,selected_header,quality_metrics
    from .faced_proposal_preprocessing import calibrate_common_counts
    from .faced_artifact_control import artifact_trial_windows
    from .faced_classical_contract import spectral_features
    from .faced_inference_quality import eeg_availability
    from threadpoolctl import threadpool_limits
    budget.check()
    a,b=row['segments'][region]
    budget.consume('count_values',30*(b-a))
    counts,ledger=read_interval(_MeteredStream(stream,budget),header,*row['segments'][region],row['channel_indices'],reference=False)
    if counts.shape!=(30,32*row['native_rate_hz']) or ledger['extra_boundary_samples_decoded']!=0:
        raise ValueError('Selected decode geometry differs')
    uv=calibrate_common_counts(counts,row['subject'],selected_header(header,row['channel_indices']))
    budget.check()
    with threadpool_limits(limits=12,user_api='blas'):
        windows,artifact=artifact_trial_windows(uv,row['native_rate_hz'])
    budget.check()
    if not np.isfinite(windows).all(): raise ValueError('Nonfinite processed signal fatal')
    quality=quality_metrics(counts,uv,windows,row['native_rate_hz'])
    decision=eeg_availability(quality,technical_passed=True,repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
    features=spectral_features(windows); budget.check()
    receipt={k:row[k] for k in ('subject','vid','role','arm')}|{'region':region,'sample_interval':row['segments'][region],
        'input_kind':input_kind,'extra_boundary_samples_decoded':0,'feature_sha256':array_sha256(features),
        'count_sha256':array_sha256(counts),'windows_sha256':array_sha256(windows),
        'range_ledger_sha256':ledger['range_ledger_sha256'],'quality_sha256':hash_json(quality),
        'artifact_sha256':hash_json(artifact)}
    return {'technical_passed':True,'review_flags':decision['reasons'],'features':features,'receipt':receipt}
