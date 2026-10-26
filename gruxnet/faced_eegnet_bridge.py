"""Authoritative identity/source gate for original FACED development windows.

No file reader lives here. The frozen execution bridge supplies a replay backend.
"""
import hashlib
import numpy as np

from .faced_eegnet_runner import guard_bundle, window_array


def array_sha256(array):
    x=np.ascontiguousarray(array)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def authoritative_bundle(programme,reservation,catalogue,records,decisions):
    source_people=set(reservation['development_source']);validation_people=set(reservation['development_validation'])
    protected=set(reservation['confirmation'])
    if len(source_people)!=70 or len(validation_people)!=20 or len(protected)!=33 or source_people&validation_people or (source_people|validation_people)&protected:
        raise PermissionError('Authoritative participant roles changed')
    expected={(r['subject'],r['vid']) for r in programme['input_access']['future_development_pairs']}
    clips={p[1] for p in expected}
    if clips!={2,3,4,6,9,11,12,17,18,20,23,25,27,28} or expected!={(p,v) for p in source_people|validation_people for v in clips}:
        raise PermissionError('Protected/excluded/nonrectangular development population')
    record_map={(r['subject'],r['vid']):r for r in records}
    if len(records)!=1260 or set(record_map)!=expected:raise PermissionError('Record identities changed')
    metadata={}
    for pair,r in record_map.items():
        role='source' if pair[0] in source_people else 'validation'
        if r['role']!=role or r['technical_passed'] is not True:
            raise PermissionError('Record technical role failed')
        if r['native_rate_hz'] not in (250,1000) or r['sample_interval'][1]-r['sample_interval'][0]!=32*r['native_rate_hz'] or len(r['channel_indices'])!=30 or len(set(r['channel_indices']))!=30:
            raise PermissionError('Qualified native interval/channel geometry changed')
        metadata[pair]={**catalogue[pair[1]],'subject':pair[0],'vid':pair[1],'trial':r['trial'],'role':role,
                        'family':catalogue[pair[1]]['source_film_sha256'],'acquisition_group':r['acquisition_group']}
    guard_bundle(programme,metadata,decisions)
    for rotation in programme['rotation_manifests']:
        for name,rows in rotation['arms'].items():
            intended=source_people if name=='source_fit' else validation_people
            if {r['subject'] for r in rows}!=intended:raise PermissionError('Rotation participant reservation changed')
    return record_map,metadata


def guard_cached_identity(row,metadata):
    pair=row['subject'],int(row['vid'])
    if pair not in metadata or row['family']!=metadata[pair]['family']:
        raise PermissionError('Protected/excluded cached outcome identity before numeric conversion')
    return pair


class WindowBridge:
    def __init__(self,records,metadata,decisions,replay,*,memory_cap=2000000000):
        self.records=records;self.metadata=metadata;self.decisions=decisions;self.replay=replay
        self.source=sorted(p for p,m in metadata.items() if m['role']=='source')
        self.validation=sorted(p for p,m in metadata.items() if m['role']=='validation')
        self.windows={};self.receipts=[];self.source_passed=False;self.memory_cap=memory_cap;self.bytes=0

    def prepare(self,pair):
        if pair not in self.records or pair not in self.metadata:raise PermissionError('Undeclared original window identity')
        role=self.metadata[pair]['role']
        if role=='validation' and not self.source_passed:raise PermissionError('Source QC required before validation EEG')
        if role not in ('source','validation'):raise PermissionError('Protected original role')
        if any((r['subject'],r['vid'])==pair for r in self.receipts):raise PermissionError('Duplicate original window replay')
        if not self.decisions[pair]['eeg_available']:
            if role=='source':raise PermissionError('Unavailable source holds experiment')
            self.receipts.append({'subject':pair[0],'vid':pair[1],'role':role,'eeg_available':False,
                                  'samples_decoded':0,'original_signal_decoded':False,'fallback':True})
            return
        values,decision,receipt=self.replay(self.records[pair])
        values=window_array(values)
        if decision!={k:self.decisions[pair][k] for k in ('policy_id','eeg_available','reasons')}:
            raise PermissionError('Fresh replay quality differs from fixed availability')
        if not decision['eeg_available']:raise PermissionError('Fresh source/inference numerical QC failed')
        digest=array_sha256(values)
        if digest!=self.records[pair]['processed_window_sha256']:raise ValueError('Original processed-window bytes changed')
        if self.bytes+values.nbytes>self.memory_cap:raise MemoryError('Declared in-memory window cap')
        values.flags.writeable=False;self.windows[pair]=values;self.bytes+=values.nbytes
        self.receipts.append({**receipt,'subject':pair[0],'vid':pair[1],'role':role,'eeg_available':True,
                              'processed_window_sha256':digest,'processed_window_hash_matches':True,'quality_replayed':True,'fallback':False})

    def prepare_source(self):
        for pair in self.source:self.prepare(pair)
        if len(self.source)!=980 or len(self.windows)!=980 or len(self.receipts)!=980:
            raise PermissionError('Complete authoritative source union required')
        self.source_passed=True
        return {'passed':True,'source_trials':980,'source_failures':0,'targets_joined':False,
                'validation_signals_decoded':0,'fits':0,'window_bytes_in_memory':self.bytes}

    def prepare_validation(self):
        if not self.source_passed:raise PermissionError('Source gate required')
        for pair in self.validation:self.prepare(pair)
        if len(self.receipts)!=1260 or len(self.windows)!=1245:
            raise PermissionError('Changed complete availability rectangle')

    def require_targets_gate(self):
        if not self.source_passed or len(self.receipts)<980:
            raise PermissionError('Source waveform/QC gate before cached outcomes')
