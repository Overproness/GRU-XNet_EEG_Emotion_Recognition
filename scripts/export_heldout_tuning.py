"""Atomic, study-specific publication export; no EEG, tensors or checkpoint files."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet.data import sha256
from gruxnet.heldout_tuning import STUDY,case_id,context_id,atomic_json
from scripts.audit_heldout_tuning import inspect_case,audit


def export(output):
    destination=REPO/'results/development'/STUDY
    plan=json.loads((output/'plan.json').read_text()); paths=[]
    # Always export the declaration, including before any fit is accepted.
    for name in ('plan.json','config.json','feasibility.json','progress.json','waveform_binding_seediv.json','waveform_binding_deap.json',
                 'verification.json','public_verification.json','comparison_seediv.json','comparison_deap.json','alignment_seediv.json','alignment_deap.json','FINDINGS.md'):
        if (output/name).is_file(): paths.append(output/name)
    for context,jobs,kind in ((False,plan['jobs'],'fits'),(True,plan['contexts'],'contexts')):
        for job in jobs:
            folder=output/kind/(context_id(job) if context else case_id(job))
            if not (folder/'certificate.json').exists(): continue
            record,selection=inspect_case(folder,output,context)
            names={'record.json','selection.json','certificate.json',*record['artifact_sha256'],*selection['artifact_sha256']}
            if any(not (n.endswith('.csv') or n in ('record.json','selection.json','certificate.json')) for n in names): raise ValueError('Unexpected export type')
            paths.extend(folder/n for n in sorted(names))
    for dataset in ('seediv','deap'):
        for model in ('eegnet','eegnet_context','gru','prior','context_logistic'):
            for arm in ('exposed','unexposed'):
                path=output/f'predictions_{dataset}_{model}_{arm}.csv'
                if path.is_file(): paths.append(path)
    manifest=[]
    for src in paths:
        if src.stat().st_size>2_000_000: raise ValueError('Review artifact exceeds declared size bound')
        relative=src.relative_to(output); dst=destination/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        temporary=dst.with_name(dst.name+'.partial'); temporary.write_bytes(src.read_bytes()); temporary.replace(dst)
        manifest.append({'path':relative.as_posix(),'sha256':sha256(src)})
    # Complete study has >60k files; bounded manifest shards avoid giant JSONs.
    shards=[]
    for number,start in enumerate(range(0,len(manifest),1000)):
        name=f'export_manifest_{number:03}.json'
        atomic_json(destination/name,manifest[start:start+1000])
        shards.append({'path':name,'sha256':sha256(destination/name),'files':len(manifest[start:start+1000])})
    atomic_json(destination/'export_manifest.json',{'shards':shards,'total_files':len(manifest)})
    check=audit(destination,partial=True,public=True)
    p=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {'state':'declared','selected_neural_completed':0}
    note=REPO/'docs/publication/GRU-XNet_Heldout_Tuning_Status_2026-10-06.md'
    note.parent.mkdir(parents=True,exist_ok=True)
    text=f'''# Held-out tuning status

State: **{p['state']}**. Verified selected neural cases: **{p['selected_neural_completed']}/2040**. Updated: {p.get('updated_utc',plan['created_utc'])}.

The declared grid contains 4080 uninterrupted 1200-update trajectories and 24,480 source-validation candidates. GRU, EEGNet and EEGNet-plus-context independently choose their learning rate, duration and normalization in each outer fold. Two groupings, two initializations, all participant/video folds and both familiar/unseen exposure arms are retained. Context-only priors and calibrated controls use the same source partitions and two-panel selection criterion.

Source validation equally weights familiar and unseen videos from held-out validation participants. Outer test participants enter inference only after selection is sealed. This repeats development on existing cohorts; it does not add independent participants or establish convergence. Different raw/STFT representations prevent an architecture-only interpretation. The manuscript and research question are unchanged.

The [machine-readable protocol](../../results/development/{STUDY}/plan.json), [feasibility checks](../../results/development/{STUDY}/feasibility.json), [progress](../../results/development/{STUDY}/progress.json), and [portable verification](../../results/development/{STUDY}/public_verification.json) are exported with complete accepted candidate probabilities and selection/replay records. No raw EEG or model tensors are published.

All selected EEGNet states and fixed GRU sentinels are retained locally. Other selected GRU states are independently replayed before deletion, with their SHA and immediate verification certificate preserved. Such deleted weights require refitting for later raw replay. Outer aggregate findings are generated only when all cases pass verification; an in-progress export does not contain complete findings.

Worker: `D:\\DL_Frameworks\\envs\\pytorch\\python.exe scripts/heldout_tuning.py run --push-milestones`. Matching declarations and certificates permit resumption. Git checkpoints occur after every 40 newly accepted neural cases and completion. A failure stops the worker and records a local FAILURE.json; no successful split or seed is substituted.
'''
    if p['state']=='complete': text+=f'\n[Complete findings](../../results/development/{STUDY}/FINDINGS.md).\n'
    note.write_text(text,encoding='utf-8')
    return {'exported_files':len(paths),'portable_verified_cases':check['verified_cases']}


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=REPO.parent/'publication_runs'/STUDY)
    args=parser.parse_args(); print(json.dumps(export(args.output.resolve())))
