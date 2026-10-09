"""Post-fit export recovery with bounded checksum buffers; frozen fits unchanged.

The original worker and a fresh original exporter exhausted memory in the
4-MiB checksum read allocation. This adapter computes the identical SHA256
with 64-KiB reads, then uses the original numerical auditor and exporter.
It changes no experiment source bytes, candidates, predictions or selections.
"""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import gruxnet.data as data
import gruxnet.heldout_tuning as study
import scripts.audit_heldout_tuning as auditor
import scripts.export_heldout_tuning as exporter


def bounded_sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        buffer=bytearray(65536)
        view=memoryview(buffer)
        while True:
            length=stream.readinto(buffer)
            if not length: break
            result.update(view[:length])
    return result.hexdigest()


def recover(output):
    # Check semantic equivalence against the original helper on bounded inputs.
    for path in (output/'config.json',output/'verification.json',REPO/'gruxnet/data.py'):
        if bounded_sha256(path)!=data.sha256(path): raise ValueError('Checksum adapter differs from original')
    for module in (data,study,auditor,exporter): module.sha256=bounded_sha256
    verification=json.loads((output/'verification.json').read_text())
    if not verification['passed'] or not verification['complete'] or verification['verified_cases']!=2380:
        raise ValueError('Completed verified experiment required; this command performs no refitting')
    plan=json.loads((output/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        if bounded_sha256(REPO/name)!=expected: raise ValueError('Frozen experiment source changed')
    destination=REPO/'results/development'/study.STUDY
    # The original worker finished copying the complete snapshot before failing
    # in its numerical audit. Verify that snapshot against both its manifest
    # and local originals, avoiding another source audit and 70k file copies.
    index=json.loads((destination/'export_manifest.json').read_text()); files=0
    for shard in index['shards']:
        path=destination/shard['path']
        if bounded_sha256(path)!=shard['sha256']: raise ValueError('Manifest shard changed')
        entries=json.loads(path.read_text())
        if len(entries)!=shard['files']: raise ValueError('Manifest count changed')
        for entry in entries:
            relative=Path(entry['path'])
            if relative.drive or relative.root or '..' in relative.parts: raise ValueError('Unexpected export path')
            path=destination/relative
            if bounded_sha256(path)!=entry['sha256']: raise ValueError('Export bytes changed')
            if bounded_sha256(output/relative)!=entry['sha256']: raise ValueError('Local originals differ from complete snapshot')
            files+=1
    if files!=index['total_files']: raise ValueError('Incomplete export manifest')
    public=auditor.audit(destination,partial=False,public=True)
    if not public['passed'] or not public['complete'] or public['verified_cases']!=2380:
        raise ValueError('Recovery did not complete public verification')
    progress=json.loads((output/'progress.json').read_text())
    if progress['state']!='complete' or progress['selected_neural_completed']!=2040: raise ValueError('Unexpected completion state')
    note=REPO/'docs/publication/GRU-XNet_Heldout_Tuning_Status_2026-10-06.md'
    lines=note.read_text(encoding='utf-8').splitlines()
    lines[2]=f"State: **complete**. Verified selected neural cases: **2040/2040**; context cells: **340/340**. Study completed: {progress['updated_utc']}. Final publication verification recovered: {study.stamp()}."
    lines += ['',f'[Complete findings](../../results/development/{study.STUDY}/FINDINGS.md).',
              '', 'The original final export ran out of memory during checksum verification after all fits and analysis completed. A separate recovery command verified the existing complete snapshot against local originals and recomputed all public candidate/selection metrics using 64-KiB checksum buffers. Frozen experiment source files and results remain unchanged. The original failure is preserved locally; [publication recovery](../../results/development/'+study.STUDY+'/publication_recovery.json) records its hash and the recovered verification.']
    note.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    recovery={'passed':True,'created_utc':study.stamp(),'purpose':'Post-fit publication recovery only',
              'experiment_sources_unchanged':True,'checksum_buffer_bytes':65536,
              'checksum_semantics':'Same SHA256 bytes; only read allocation/chunk size changed',
              'recovery_source_sha256':bounded_sha256(Path(__file__)),
              'plan_sha256':bounded_sha256(output/'plan.json'),
              'local_verification_sha256':bounded_sha256(output/'verification.json'),
              'public_verification_sha256':bounded_sha256(destination/'public_verification.json'),
              'exported_files_checked':files,'complete_verified_cases':public['verified_cases'],
              'original_failure_sha256':bounded_sha256(output/'FAILURE.json') if (output/'FAILURE.json').exists() else None}
    study.atomic_json(output/'publication_recovery.json',recovery)
    study.atomic_json(destination/'publication_recovery.json',recovery)
    print(json.dumps({'recovery':recovery}),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO.parent/'publication_runs'/study.STUDY)
    args=parser.parse_args(); recover(args.output.resolve())
