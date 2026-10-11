"""Qualify supplied preprocessing bytes; isolate one reviewed method on fake arrays.

Never import the authors' module, preload BDF, deserialize pickle or open ratings.
"""
import ast
import contextlib
import hashlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_faced_classical import digest,load,save
PUBLIC=REPO/'results/development/faced_artifact_control_2026-10-11'
WORKSPACE=REPO.parent

def isolated_bad_method(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Preprocessing')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='bad_channels_interpolate')
    # Only this already reviewed method is compiled, not module imports/top-level code.
    if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)) for n in ast.walk(method)):
        raise ValueError('Unexpected isolated method operations')
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'isolated-author-method','exec'),namespace)
    return namespace['bad_channels_interpolate'],[method.lineno,method.end_lineno]

def probe(source):
    method,lines=isolated_bad_method(source);records=[]
    for channel in (0,27):
        x=np.ones((1,32,7501));x[0,channel,:]=10000.
        calls=[]
        raw=SimpleNamespace(get_data=lambda:x,info={'bads':[]},interpolate_bads=lambda:calls.append(True))
        fake=SimpleNamespace(raw=raw,montage_index=dict(enumerate(f'ch{i}' for i in range(32))))
        with contextlib.redirect_stdout(io.StringIO()):method(fake,thresh1=3,proportion=.3)
        squeezed=np.squeeze(x);value=np.where(np.abs(squeezed)>3*np.median(np.abs(squeezed)),0,1)[0]
        bad=np.argwhere(np.mean(1-value,axis=0)>.3)
        expected=np.flatnonzero((np.abs(squeezed)>3*np.median(np.abs(squeezed))).mean(axis=1)>.3).tolist()
        if expected!=[channel] or calls or raw.info['bads']:raise ValueError('Synthetic source-code diagnosis changed')
        records.append({'manufactured_bad_channel_index':channel,'input_shape':list(x.shape),
            'paper_interpretation_detected_channel_indices':expected,
            'author_value_shape':list(value.shape),'author_bad_chns_shape':list(bad.shape),
            'author_bad_chns_size':int(bad.size),'interpolation_calls':len(calls),'marked_bad_channels':raw.info['bads']})
    return {'isolated_method_lines':lines,'manufactured_probes':records,
        'conclusion':'For tested epoch-shaped inputs, selecting first channel then reducing the time axis produces scalar argwhere with size zero; interpolation is not invoked even with a sustained extreme channel.',
        'scope':'Supplied code behavior on manufactured arrays under current NumPy. Does not establish code actually used for published processed data or validate a repair.'}

def build():
    archive=WORKSPACE/'Code.zip';root=WORKSPACE/'Code'
    members=['Code/Preprocessing/'+n for n in ('Preprocessing.py','main.py','torch_win.yml','torch_ubuntu.yml')]
    files={};payloads={}
    with zipfile.ZipFile(archive) as z:
        for name in members:
            payload=z.read(name);local=root/name
            if local.read_bytes()!=payload:raise ValueError('Extracted author code differs from supplied archive')
            files[name]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'extracted_matches_archive':True}
            payloads[name]=payload
    diagnosis=probe(payloads[members[0]].decode('utf-8'))
    save(PUBLIC/'supplied_code_qualification.json',{
        'date':'2026-10-11','public_entity_id':'syn61318284','archive_name':'Code.zip',
        'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,'qualified_members':files,
        'provenance':'User-supplied download; exact inspected archive/extraction bound. No independently authenticated Synapse archive checksum available.',
        'source_available':True,'source_statically_qualified':True,'author_pipeline_reproduced':False,
        'diagnosis':diagnosis,'author_execution':{'epoch':'Last 30 seconds before event102, default Epochs parameters otherwise implicit',
            'channels':32,'downsample_hz':250,'bandpass_hz':[.05,47.],'filter':'MNE default FIR',
            'bad_detection':'Global absolute median, first-channel indexing, collapsed time-axis reduction in supplied method',
            'ocular_ica':'Per-epoch FastICA; max_iter auto, no random_state; Fp1/Fp2 EOG proxies',
            'reference':'Average after attempted interpolation and ICA','units':'Whole-record mean-amplitude heuristic with direct internal-data multiplication',
            'input_access':'Full BDF preload and whole After_remarks.mat loader; incompatible with our sealed-material/selective-access contract',
            'declared_runtime':'Python3.10.4, MNE1.2.1; two differing NumPy/SciPy environment files'},
        'decision':'Retain already declared transparent source-only interpolation control. Do not run original main.py or silently fix and call it an author reproduction.',
        'author_code_redistributed':False,'original_eeg_or_ratings_opened':False,
        'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in [Path(__file__),REPO/'tests/test_faced_supplied_code.py']}})
    save(PUBLIC/'pre_pilot_addendum.json',{'activation_plan_sha256':digest(PUBLIC/'activation_plan.json'),
        'supplied_code_qualification_sha256':digest(PUBLIC/'supplied_code_qualification.json'),
        'previous_anonymous_source_inventory_receipt_preserved':True,'algorithm_or_trial_access_changed':False,
        'original_ratings_decoded':0,'pilot_started':False,'author_source_arrived_after_declaration':True})
    print({'author_archive_and_extraction_qualified':True,'synthetic_interpolation_calls':0,'original_data_opened':False})

if __name__=='__main__':build()
