"""Frozen source-only extraction/QC/inference/export qualification; no targets.

Replay the historical eight-trial pilot and the known source QC failure. The
remaining source rows use the authenticated available-source feature snapshot.
No validation/protected reader, rating decoder, fitting or retry is provided.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_faced_coverage_selection_2026_10_13 import metadata
from scripts.run_faced_temporal_coverage_pilot_2026_10_12 import (
    sha, array_sha, load, safe_raw_path, runtime, read_header,
    read_interval, selected_header, quality_metrics, calibrate_common_counts,
    artifact_trial_windows, eeg_availability, spectral_features)
from gruxnet.faced_temporal_coverage import SourceOnlySegmentGuard, REGIONS

BASE = REPO/'results/development'
PUBLIC = BASE/'faced_source_export_qualification_2026-10-14'
PRIVATE = REPO.parent/'publication_runs/faced_source_export_qualification_2026-10-14'
PRIOR = BASE/'faced_coverage_missing_2026-10-13'
STATE_ROOT = REPO.parent/'publication_runs/faced_coverage_missing_2026-10-13'
PLAN = PUBLIC/'plan.json'
CAP_SECONDS = 1200


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def bindings_check(bindings):
    for name, expected in bindings.items():
        if sha(REPO/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)


def preservation():
    packets = [
        ('paper/exploratory_evaluation_2026-10-12/build_receipt.json', 'bindings'),
        ('results/development/faced_confirmation_freeze_2026-10-12/activation.json', 'bindings'),
        ('results/development/faced_confirmation_freeze_2026-10-12/activation.json', 'prior_completed_evidence_bindings'),
        ('results/development/faced_coverage_confirmation_proposal_2026-10-13/proposal.json', 'bindings'),
        ('results/development/faced_coverage_missing_confirmation_proposal_2026-10-13/proposal.json', 'bindings'),
        ('results/development/faced_coverage_missing_2026-10-13/activation.json', 'bindings'),
        ('results/development/faced_benchmark_claim_2026-10-13/decision.json', 'bindings')]
    result = []
    for name, field in packets:
        packet = load(REPO/name); bindings_check(packet[field])
        result.append({'receipt':name, 'field':field, 'bindings':len(packet[field]), 'sha256':sha(REPO/name)})
    gates = load(BASE/'faced_coverage_missing_confirmation_proposal_2026-10-13/proposal.json')['currently_passed_activation_gates']
    if len(gates) != 10 or any(gates.values()):
        raise PermissionError('Preserved future confirmation must remain inactive')
    return result


def source_scope():
    previous, _, meta, pairs = metadata()
    pilot = [(r['subject'],r['vid']) for r in previous['pilot']['trials']]
    missing = ('sub-087',20)
    if len(pilot) != 8 or missing in pilot or missing not in meta:
        raise PermissionError('Exact historical pilot and failure fixture required')
    selected = pilot+[missing]
    segments = [{'subject':p,'vid':v,'role':'source','region':region,
                 'native_rate_hz':meta[p,v]['native_rate_hz'],
                 'sample_interval':meta[p,v]['segments'][region]}
                for p,v in selected for region in REGIONS]
    guard = SourceOnlySegmentGuard(source_people=previous['source_people'],
        validation_people=previous['validation_people_denied'],protected_people=previous['protected_people_denied'],
        source_pairs=pairs,pilot_rows=segments)
    return previous,meta,pairs,selected,segments,guard


def declare():
    previous,meta,pairs,pilot,segments,_ = source_scope()
    preservation()
    records = [json.loads(s) for s in (PRIOR/'fit_records.jsonl').read_text().splitlines()]
    final = [r for r in records if r.get('stage') == 'full_source_state_qualification']
    if len(final) != 3 or {r['model'] for r in final} != {'context_only','late_joint','distributed_joint'}:
        raise PermissionError('Exactly three existing final states required')
    private_bindings = {str((STATE_ROOT/r['private_state_file']).relative_to(REPO.parent)).replace('\\','/'):r['private_state_sha256'] for r in final}
    qual = load(PRIOR/'qualification.json'); state = load(PRIOR/'final_source_state_qualification.json')
    private_bindings['publication_runs/faced_coverage_missing_2026-10-13/available_source_vectors.npz'] = qual['source_snapshot_file_sha256']
    private_bindings['publication_runs/faced_coverage_missing_2026-10-13/full_source_category_prior.json'] = state['private_category_prior_sha256']
    for name,expected in private_bindings.items():
        if sha(REPO.parent/name) != expected:
            raise ValueError('Source-only file integrity changed')
    originals={r['member']:r for r in load(BASE/'faced_raw_2026-10-11/extraction.json')['records']}
    original_files=[{'subject':p,'name':name, **{k:originals[f'Data/{p.replace("sub-","sub")}/{name}'][k] for k in ('bytes','sha256')}}
                    for p in sorted({p for p,_ in pilot}) for name in ('data.bdf','evt.bdf')]
    public_names=[
        'scripts/qualify_faced_source_export_2026_10_14.py','tests/test_faced_source_export_2026_10_14.py',
        'gruxnet/faced_frozen_inference.py','tests/test_faced_frozen_inference.py',
        'scripts/run_faced_coverage_selection_2026_10_13.py','scripts/run_faced_temporal_coverage_pilot_2026_10_12.py',
        'scripts/audit_faced_raw.py','gruxnet/faced_temporal_coverage.py','gruxnet/faced_source_quality.py',
        'gruxnet/faced_proposal_preprocessing.py','gruxnet/faced_source_notch.py','gruxnet/faced_artifact_control.py',
        'gruxnet/faced_inference_quality.py','gruxnet/faced_classical_contract.py',
        'results/development/faced_temporal_coverage_preflight_2026-10-12/plan.json',
        'results/development/faced_artifact_control_2026-10-11/source_records.json',
        'results/development/faced_raw_2026-10-11/headers.json','results/development/faced_raw_2026-10-11/extraction.json',
        'results/development/faced_codebooks_2026-10-10/audit.json',
        'results/development/faced_coverage_selection_2026-10-13/source_regions.jsonl',
        'results/development/faced_coverage_missing_2026-10-13/qualification.json',
        'results/development/faced_coverage_missing_2026-10-13/final_source_state_qualification.json',
        'results/development/faced_coverage_missing_2026-10-13/fit_records.jsonl',
        'results/development/faced_coverage_missing_confirmation_proposal_2026-10-13/proposal.json',
        'results/development/faced_saved_state_qualification_2026-10-14/completion.json',
        'results/development/faced_saved_state_qualification_2026-10-14/source_prediction_fixtures.json',
        'results/development/faced_benchmark_claim_2026-10-13/decision.json']
    plan={'date':'2026-10-14','status':'declared_source_only_backend_qualification',
        'author_authorization':'User approved independently verifying existing fitted states and inference/export using source/manufactured inputs, and preparing a confirmation package before reserved access.',
        'source_people':previous['source_people'],'validation_people_denied':previous['validation_people_denied'],
        'protected_people_denied':previous['protected_people_denied'],
        'source_pairs':pairs,'pilot_pairs':pilot,'pilot_regions':segments,'raw_trials':9,'raw_regions':27,
        'pilot_selection':'Preserved deterministic eight-category/acquisition pilot plus previously documented sub-087/20 shared-missing failure; no outcome selection.',
        'source_metadata_sha256':hashlib.sha256(json.dumps([meta[p] for p in pairs],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'count_scalar_cap':sum(30*32*r['native_rate_hz'] for r in segments),
        'selected_signal_payload_bytes_cap':sum(90*32*r['native_rate_hz'] for r in segments),
        'original_files':original_files,
        'whole_original_integrity_bytes_cap':sum(r['bytes']*(2 if r['name']=='data.bdf' else 1) for r in original_files),
        'source_snapshot_scalar_cap':979*3*150,'source_inference_trials':980,'prediction_rows':3920,
        'private_bindings':private_bindings,'final_state_records':final,
        'source_identity_sha256':state['models'][0]['source_identity_sha256'],
        'code_and_input_bindings':{n:sha(REPO/n) for n in public_names},
        'runtime':runtime(),'wall_seconds_cap':CAP_SECONDS,'cpu_seconds_cap':CAP_SECONDS,
        'new_fits':0,'source_targets_converted':0,'original_rating_reads':0,'validation_access':False,
        'protected_access':False,'confirmation_activated':False,'papers_changed':False,
        'automatic_retry':False,'scope':'Existing source states and source feature/extraction replay only; source export is not a protected prediction lock.'}
    write(PLAN,plan)
    print(json.dumps({'declared':True,'raw_trials':9,'raw_regions':27,'scalar_cap':plan['count_scalar_cap'],'source_export_rows':3920}),flush=True)


def audit_segment(actual, expected):
    keys=('subject','vid','role','region','sample_interval','native_rate_hz','eeg_available','reasons',
          'count_array_sha256','processed_window_sha256','spectral_feature_sha256','range_ledger_sha256',
          'repair_count','artifact_decision')
    differences=[k for k in keys if actual.get(k)!=expected.get(k)]
    if differences:
        raise ValueError('Actual source region differs from preserved replay: '+','.join(differences))


def audit_predictions(rows, fixtures):
    """Compare every exported value with independently computed source algebra."""
    source={(r['subject'],r['vid']):r for r in fixtures['rows']}
    if len(source)!=len(fixtures['rows']) or len(rows)!=4*len(source):
        raise ValueError('Independent prediction fixture denominator differs')
    seen=set(); largest=0.0
    for row in rows:
        pair=row['subject'],row['vid']; key=(*pair,row['model'])
        if pair not in source or key in seen:
            raise ValueError('Unexpected or duplicate independent prediction identity')
        seen.add(key); expected=source[pair]
        if row['shared_eeg_available'] is not expected['shared_eeg_available']:
            raise ValueError('Independent shared availability differs')
        if row['model']=='category_median':
            number=expected['category_median']
        elif row['model'] in ('context_only','late_joint','distributed_joint'):
            number=expected['inference_predictions'][row['model']]
        else:
            raise ValueError('Unexpected independently qualified method')
        delta=abs(float(row['prediction'])-float(number)); largest=max(largest,delta)
        if not np.isfinite(delta) or delta>1e-10:
            raise ValueError('Exported prediction differs from independent source algebra')
    return largest


def run():
    from gruxnet.faced_frozen_inference import load_frozen_bundle, infer_four_methods, export_prediction_lock, verify_prediction_lock
    plan=load(PLAN); bindings_check(plan['code_and_input_bindings']); preservation()
    if plan['runtime'] != runtime() or plan['protected_access'] is not False or plan['new_fits'] != 0:
        raise PermissionError('Source-only runtime/scope changed')
    previous,meta,pairs,pilot,segments,guard=source_scope()
    if json.loads(json.dumps(pairs))!=plan['source_pairs'] or segments!=plan['pilot_regions']:
        raise PermissionError('Frozen source roster changed')
    metadata_sha=hashlib.sha256(json.dumps([meta[p] for p in pairs],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if metadata_sha != plan['source_metadata_sha256']:
        raise ValueError('Source metadata changed')
    if (PUBLIC/'execution_started.json').exists():
        raise PermissionError('Preserve existing attempt; declare a reviewed revision before retry')
    write(PUBLIC/'execution_started.json',{'plan_sha256':sha(PLAN),'utc':datetime.now(timezone.utc).isoformat(),'target_reads':0,'fits':0})
    started=time.monotonic(); cpu=time.process_time(); decoded=0; integrity_bytes=0; segments_out=[]; snapshots={}; raw_vectors={}
    PRIVATE.mkdir(parents=True,exist_ok=True)
    def budget():
        if time.monotonic()-started>CAP_SECONDS or time.process_time()-cpu>CAP_SECONDS:
            raise TimeoutError('Cooperative source backend qualification cap')
    expected_rows=[json.loads(s) for s in (BASE/'faced_coverage_selection_2026-10-13/source_regions.jsonl').read_text().splitlines()]
    expected={(r['subject'],r['vid'],r['region']):r for r in expected_rows}
    headers={r['subject']:r['data.bdf'] for r in load(BASE/'faced_raw_2026-10-11/headers.json')['records'] if r['subject'] in previous['source_people']}
    originals={(r['subject'],r['name']):r for r in plan['original_files']}
    def reader(person,vid,region,interval,rate):
        nonlocal decoded,integrity_bytes
        budget(); record=meta[person,vid]; path=safe_raw_path(person,'data.bdf'); header=headers[person]
        if person not in snapshots:
            if read_header(path)!=header:
                raise ValueError('Source original header differs')
            for name in ('data.bdf','evt.bdf'):
                file=safe_raw_path(person,name); original=originals[person,name]
                if file.stat().st_size!=original['bytes'] or sha(file,budget)!=original['sha256']:
                    raise ValueError('Original source integrity failed')
                integrity_bytes+=original['bytes']
            snapshots[person]=(path.stat().st_size,path.stat().st_mtime_ns)
        with path.open('rb') as stream:
            counts,access=read_interval(stream,header,*interval,record['channel_indices'],reference=True)
        decoded+=int(counts.size)
        if decoded>plan['count_scalar_cap'] or access['extra_boundary_samples_decoded']!=0:
            raise PermissionError('Source decoded sample cap exceeded')
        uv=calibrate_common_counts(counts,person,selected_header(header,record['channel_indices']))
        with threadpool_limits(limits=12,user_api='blas'):
            windows,artifact=artifact_trial_windows(uv,rate)
        quality=quality_metrics(counts,uv,windows,rate)
        decision=eeg_availability(quality,technical_passed=True,repair_capacity_exceeded=artifact['repair_capacity_exceeded'])
        features=spectral_features(windows)
        row={'subject':person,'vid':vid,'role':'source','region':region,'sample_interval':interval,
             'native_rate_hz':rate,'eeg_available':decision['eeg_available'],'reasons':decision['reasons'],
             'artifact_decision':artifact['decision'],'repair_count':len(artifact['repaired_channels']),
             'count_array_sha256':array_sha(counts),'processed_window_sha256':array_sha(windows),
             'spectral_feature_sha256':array_sha(features),'range_ledger_sha256':access['range_ledger_sha256']}
        audit_segment(row,expected[person,vid,region])
        if snapshots[person]!=(path.stat().st_size,path.stat().st_mtime_ns):
            raise ValueError('Source original changed during read')
        raw_vectors[person,vid,region]=features
        write(PRIVATE/f'{person}_clip{vid:02d}_{region}_quality.json',quality)
        write(PRIVATE/f'{person}_clip{vid:02d}_{region}_access.json',access)
        segments_out.append(row); budget()
        print(json.dumps({'source_segment':len(segments_out),'region':region,'available':decision['eeg_available']}),flush=True)
    try:
        for name,digest in plan['private_bindings'].items():
            if sha(REPO.parent/name,budget)!=digest:
                raise ValueError('Bound private source cache/state changed')
        for row in segments:
            guard.read(subject=row['subject'],vid=row['vid'],region=row['region'],sample_interval=row['sample_interval'],native_rate_hz=row['native_rate_hz'],reader=reader)
        for person in snapshots:
            original=originals[person,'data.bdf']
            if sha(safe_raw_path(person,'data.bdf'),budget)!=original['sha256']:
                raise ValueError('Original changed after extraction')
            integrity_bytes+=original['bytes']
        if decoded!=plan['count_scalar_cap'] or integrity_bytes!=plan['whole_original_integrity_bytes_cap']:
            raise ValueError('Incomplete selective decode/integrity denominator')
        qualification=load(PRIOR/'qualification.json')
        available_pairs=[tuple(p) for p in qualification['available_pairs']]
        snapshot=STATE_ROOT/'available_source_vectors.npz'
        with np.load(snapshot,allow_pickle=False) as data:
            if data.files!=['vectors']:
                raise ValueError('Unexpected snapshot archive')
            vectors=data['vectors'].copy()
        if vectors.shape!=(979,3,150) or vectors.size!=plan['source_snapshot_scalar_cap'] or not np.isfinite(vectors).all():
            raise ValueError('Source snapshot geometry failed')
        if array_sha(vectors)!=qualification['source_snapshot_array_sha256']:
            raise ValueError('Source snapshot numerical hash differs')
        index={p:i for i,p in enumerate(available_pairs)}
        def feature_loader(pair,region):
            budget()
            if pair not in index:
                raise PermissionError('Unavailable source feature callback forbidden')
            if pair in pilot:
                return raw_vectors[*pair,region].copy()
            return vectors[index[pair],REGIONS.index(region)].copy()
        # The exact public source metadata/QC joins are validated by the pure
        # export module before it invokes any numerical feature callback.
        metadata_rows=[{k:meta[p][k] for k in ('subject','vid','role','family','assigned_category','acquisition_group','catalogue_duration_seconds','trial')} for p in pairs]
        quality_rows=[{**{k:r[k] for k in ('subject','vid','role','region','eeg_available')},'technical_passed':True,'review_flags':r['reasons']} for r in expected_rows]
        state_bindings={r['model']:{'path':STATE_ROOT/r['private_state_file'],'sha256':r['private_state_sha256'],'record':r} for r in plan['final_state_records']}
        bundle=load_frozen_bundle(state_bindings,STATE_ROOT/'full_source_category_prior.json',plan['private_bindings']['publication_runs/faced_coverage_missing_2026-10-13/full_source_category_prior.json'],plan['source_identity_sha256'])
        rows=infer_four_methods(bundle,metadata_rows,quality_rows,feature_loader,mode='source')
        fixtures=load(BASE/'faced_saved_state_qualification_2026-10-14/source_prediction_fixtures.json')
        max_discrepancy=audit_predictions(rows,fixtures)
        export_prediction_lock(PUBLIC/'source_prediction_lock',rows,metadata_rows,quality_rows,mode='source',bundle=bundle)
        verified=verify_prediction_lock(PUBLIC/'source_prediction_lock',metadata_rows,quality_rows,mode='source',bundle=bundle)
        exported=load(PUBLIC/'source_prediction_lock/predictions.json')
        export_discrepancy=audit_predictions(exported,fixtures)
        bindings_check(plan['code_and_input_bindings']); preservation(); budget()
        write(PUBLIC/'source_regions_replay.json',segments_out)
        result={'date':'2026-10-14','passed':True,'status':'completed_bounded_source_backend_qualification',
                'plan_sha256':sha(PLAN),'source_trials':980,'raw_replay_trials':9,'raw_replay_regions':27,
                'count_scalars_decoded':decoded,'selected_payload_bytes_decoded':decoded*3,
                'whole_original_integrity_bytes_hashed':integrity_bytes,'source_snapshot_scalars':vectors.size,
                'source_prediction_rows':len(rows),'source_prediction_lock':verified,
                'shared_missing_source_trials':1,'source_shared_fallback_prediction_rows':3,'missing_trial_prior_agreement_rows':4,
                'independent_algebra_max_prediction_discrepancy':max_discrepancy,
                'independent_export_roundtrip_max_prediction_discrepancy':export_discrepancy,
                'raw_region_hashes_exact':True,'prediction_serializer_roundtrip_passed':True,
                'elapsed_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
                'source_state_numeric_qualification_is_separate':True,'new_fits':0,'target_reads':0,
                'validation_access':False,'protected_access':False,'actual_protected_exports_qualified':False,
                'confirmation_activated':False,'preservation':preservation(),
                'limitations':'Source/manufactured path qualification only; protected recordings, outcomes and prediction exports remain unopened/unqualified.'}
        write(PUBLIC/'completion.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('preservation','source_prediction_lock')}),flush=True)
    except Exception as error:
        write(PUBLIC/'failure.json',{'status':'held_or_failed','error_type':type(error).__name__,'reason':str(error),
              'raw_regions_completed':len(segments_out),'count_scalars_decoded':decoded,'new_fits':0,
              'target_reads':0,'validation_access':False,'protected_access':False,'automatic_retry':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('declare','run'))
    action=parser.parse_args().action
    (declare if action=='declare' else run)()
