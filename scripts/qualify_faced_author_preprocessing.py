"""Persist public source metadata and installed interpolation provenance; no EEG/rating access."""
import hashlib
import inspect
import importlib
import json
from pathlib import Path
import sys
import requests
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import mne
from mne.bem import fit_sphere_to_headshape
from mne.channels import interpolation
from gruxnet.faced_artifact_control import geometry,qualified_weights
from scripts.run_faced_classical import digest,save

PUBLIC=REPO/'results/development/faced_artifact_control_2026-10-11'
API='https://repo-prod.prod.sagebase.org/repo/v1'

def installed_sources():
    root=Path(mne.__file__).parent
    paths={Path(inspect.getfile(interpolation)),Path(importlib.import_module('mne.bem').__file__),
           root/'channels/data/montages/standard_1020.elc'}
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(paths)}

def build():
    s=requests.Session();s.trust_env=False
    entity=s.get(API+'/entity/syn61318284',timeout=45);entity.raise_for_status();e=entity.json()
    permissions=s.get(API+'/entity/syn61318284/permissions',timeout=45);permissions.raise_for_status();p=permissions.json()
    if e['name']!='Code.zip' or e['parentId']!='syn50614194':raise ValueError('Author code inventory changed')
    paper=s.get('https://pmc.ncbi.nlm.nih.gov/articles/PMC10600242/',timeout=45);paper.raise_for_status()
    # Retain only public identifiers/permission booleans and hashes, not temporary download URLs.
    receipt={'date':'2026-10-11','public_entity':{k:e.get(k) for k in ('id','name','parentId','versionNumber','dataFileHandleId')},
        'public_entity_response_sha256':hashlib.sha256(entity.content).hexdigest(),
        'anonymous_permissions':{k:p.get(k) for k in ('canDownload','isCertifiedUser','canCertifiedUserEdit')},
        'permission_response_sha256':hashlib.sha256(permissions.content).hexdigest(),
        'paper_url':paper.url,'paper_response_sha256':hashlib.sha256(paper.content).hexdigest(),
        'restricted_code_download_attempted':False,'author_code_bytes_available':False,
        'author_preprocessing_qualified':False,'reason':'Original Code.zip identified; anonymous canDownload is false; exact code unavailable locally at declaration.',
        'paper_documented_processing':{'sample_rate_hz':250,'bandpass_hz':[.05,47.],
            'outlier_description':'Samples exceeding three times median absolute signal; electrodes with over 30% outliers interpolated.',
            'ocular':'ICA with Fp1/Fp2 reference proxies','reference':'CAR after artifact handling',
            'author_runtime':'MNE 1.2.1; Python 3.10; paper description does not resolve all execution details'},
        'control_reproduces_author_pipeline':False,'control_uses_ICA':False,
        'installed_mne_version':mne.__version__,'installed_mne_source_hashes':installed_sources(),
        'mne_pinned_source_url':'https://github.com/mne-tools/mne-python/blob/v1.10.0/mne/channels/interpolation.py',
        'geometry':geometry(),'synthetic_weight_discrepancies':{str(b):qualified_weights(b)[2] for b in ((0,),(0,1),(27,),(0,1,27))},
        'original_eeg_samples_decoded':0,'original_ratings_decoded':0,
        'source_links':['https://www.synapse.org/Synapse:syn61318284','https://pmc.ncbi.nlm.nih.gov/articles/PMC10600242/'],
        'bindings':{p.relative_to(REPO).as_posix():digest(p) for p in [Path(__file__),REPO/'gruxnet/faced_artifact_control.py',REPO/'tests/test_faced_artifact_control.py']}}
    if p.get('canDownload') is not False:raise ValueError('Permission state requires separate qualification')
    save(PUBLIC/'author_code_qualification.json',receipt)
    print({'author_code_identified':e['id'],'author_code_bytes_qualified':False,'public_mne_operator_qualified':True,'original_data_opened':False})

if __name__=='__main__':build()
