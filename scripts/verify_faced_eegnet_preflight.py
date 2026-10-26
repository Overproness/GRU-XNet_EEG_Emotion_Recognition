"""Independent receipt checks and public-only CPU portability; no original-data I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
PACKET=REPO/'results/development/faced_eegnet_preflight_2026-10-11'
sys.path.insert(0,str(REPO))
from scripts.run_faced_eegnet_preflight import check, HISTORICAL, IMPLEMENTATION


def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify():
    count,binding=check();receipt=load(PACKET/'synthetic_preflight.json')
    if receipt['status']!='synthetic_hardware_preflight_passed' or receipt['parameter_count']!=1841:
        raise ValueError('Missing qualified regression/GPU receipt')
    if receipt['implementation_bindings_sha256']!=sha(PACKET/'implementation_bindings_v2.json'):
        raise ValueError('Preflight receipt code mismatch')
    rows=receipt['checks'];gpu=[r for r in rows if r['device']=='cuda:0']
    expected={(readout,rate,seed) for readout in ('eeg_only','context_residual') for rate in (.0003,.001,.003) for seed in (20261101,20261102)}
    if len(rows)!=13 or len(gpu)!=12 or {(r['readout'],r['learning_rate'],r['seed']) for r in gpu}!=expected:
        raise ValueError('Incomplete synthetic optimizer panel')
    if any(not r['forward_backward_checkpoint_finite'] or not r['fallback_without_signal'] or r['trial_batch']!=8 or r['windows_per_trial']!=7 or r['optimizer_steps']!=2 for r in rows):
        raise ValueError('Changed/missing preflight contract')
    peak=max(r['peak_reserved_bytes'] for r in gpu)
    if peak!=receipt['maximum_cuda_reserved_bytes'] or not 0<peak<=5500000000:
        raise ValueError('Memory ceiling or receipt inconsistency')
    for field in ('original_dataset_fits','original_recordings_opened','original_rating_files_opened','new_original_targets_decoded','confirmation_outcomes_opened'):
        if receipt[field]!=0:raise ValueError('Unexpected original-data access')
    if receipt['diagnostic_activated'] or receipt['research_question_changed']:raise ValueError('Unexpected activation/pivot')
    archive=REPO/'docs/paper_archive/2026-10-05-pre-exploration'
    manuscript_checks=[(REPO.parent/'report.tex','06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0'),
                       (archive/'report.tex','06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0'),
                       (archive/'DL-Report.pdf','921f07f271bc4dadbabaf62ede384c543681b9fe0aec02c490cf7d76ef9de101')]
    for path,expected_sha in manuscript_checks:
        if sha(path)!=expected_sha:raise ValueError('Working/archive manuscript changed')
    closest=load(PACKET/'closest_method_audit.json')
    if closest['source_code_executed'] or closest['author_figures_or_numeric_outcomes_opened'] or closest['distinction_establishes_novel_contribution']:
        raise ValueError('Closest-work scope overstatement')
    return {'preceding_scientific_bindings_verified':count,'input_bindings_verified':13,'new_code_bindings_verified':len(binding['code_bindings']),
            'synthetic_gpu_combinations':12,'synthetic_optimizer_steps_successful':26,'peak_reserved_bytes':peak,
            'working_manuscript_archived_source_pdf_unchanged':True,'original_dataset_fits':0,
            'diagnostic_activated':False,'confirmation_access':False,'research_question_changed':False}


def portable():
    paths=set(IMPLEMENTATION)|{'gruxnet/__init__.py','gruxnet/faced_proposal_preprocessing.py','gruxnet/faced_source_quality.py'}
    paths|=set(load(PACKET/'plan.json')['bindings'])
    for name in HISTORICAL:
        rel=f'results/development/{name}/completion/verification.json'
        paths.add(rel);paths|=set(load(REPO/rel)['bindings'])
    paths|={p.relative_to(REPO).as_posix() for p in PACKET.iterdir() if p.is_file() and p.suffix in ('.json','.md')}
    # Short ordinary directory avoids this Windows workspace's deep-temp ACL/path issue.
    target=REPO.parent/'_enp_portable';target.mkdir(exist_ok=True)
    for rel in sorted(paths):
        original=(REPO/rel).resolve()
        if not original.is_relative_to(REPO) or '.git' in original.parts or original.suffix not in ('.json','.md','.py','.png','.svg','.csv'):
            raise PermissionError('Nonpublic portable file')
        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(original.read_bytes())
        if sha(dest)!=sha(original):raise ValueError('Public-copy byte mismatch')
    commands=[['-m','unittest','discover','-s','tests','-p','test_faced_eegnet_preflight.py','-v'],
              ['-m','scripts.run_faced_eegnet_preflight','check'],
              ['-m','scripts.run_faced_eegnet_preflight','run','--cpu-only']]
    checks=[]
    for command in commands:
        result=subprocess.run([sys.executable,*command],cwd=target,capture_output=True,text=True,timeout=90)
        checks.append({'arguments':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:raise RuntimeError('Portable preflight failed: '+result.stdout+result.stderr)
    result={'copied_public_files':len(paths),'files_sha256':{r:sha(target/r) for r in sorted(paths)},
            'checks':checks,'private_original_data_or_cache_copied':False,'manuscript_or_archive_copied':False,
            'original_file_access':0,'cpu_only_hardware_result_does_not_replace_gpu_receipt':True}
    save(PACKET/'portability_check.json',result)
    return len(paths)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--portable',action='store_true');args=parser.parse_args()
    result=verify()
    if args.portable:result['portable_public_files']=portable()
    save(PACKET/'independent_audit.json',result);print(result)
