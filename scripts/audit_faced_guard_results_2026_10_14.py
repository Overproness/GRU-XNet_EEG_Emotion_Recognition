"""Independent public-receipt/serialization audit; no private or original input."""
from pathlib import Path
import hashlib
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.faced_frozen_inference import canonical_bytes, METHODS
from gruxnet.faced_guarded_confirmation import hash_json, validate_scope, verify_bindings
from scripts.qualify_faced_source_export_2026_10_14 import preservation

OUT = REPO / 'results/development/faced_guarded_confirmation_2026-10-14'
def load(path):
    return json.loads(path.read_bytes())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan = load(OUT/'plan.json')
    completed = load(OUT/'completion.json')
    verify_bindings(REPO, plan['public_bindings'])
    verify_bindings(REPO, completed['output_bindings'])
    preserved = preservation()
    if completed['plan_sha256'] != sha(OUT/'plan.json') or not completed['passed']:
        raise ValueError('Frozen plan/completion mismatch')
    for key in ('original_stat_open_hash_reads', 'source_targets_read', 'validation_numeric_reads',
                'reserved_numeric_reads', 'original_rating_reads', 'new_fits', 'actual_protected_transport_bytes'):
        if completed[key] != 0:
            raise ValueError('Forbidden access claim: '+key)
    scope = load(REPO/'results/development/faced_confirmation_backend_2026-10-14/scope.json')
    rows, _ = validate_scope(scope)
    lookup = {(r['subject'],r['vid']):r for r in rows}
    lock = OUT/'manufactured_prediction_lock'
    parts = {name: load(lock/name) for name in ('predictions.json', 'manifest.json', 'LOCK.complete.json')}
    for name,value in parts.items():
        if canonical_bytes(value) != (lock/name).read_bytes():
            raise ValueError('Noncanonical lock: '+name)
    predictions = parts['predictions.json']; manifest = parts['manifest.json']
    if len(predictions) != 3036 or manifest['kind'] != 'manufactured_qualification' or manifest['target_values'] != 0:
        raise ValueError('Incorrect manufactured denominator/provenance')
    if (manifest['prediction_sha256'] != sha(lock/'predictions.json')
            or manifest['prediction_bytes'] != (lock/'predictions.json').stat().st_size
            or parts['LOCK.complete.json'] != {'manifest_sha256':hash_json(manifest),
                                              'prediction_sha256':manifest['prediction_sha256']}):
        raise ValueError('Canonical byte lock failed')
    expected = {(*pair,model) for pair in lookup for model in METHODS}
    if {(p['subject'],p['vid'],p['model']) for p in predictions} != expected:
        raise ValueError('Exact four-method roster failed')
    arms = {arm:sum(p['arm']==arm for p in predictions) for arm in ('confirmation_unseen','confirmation_familiar')}
    if arms != {'confirmation_unseen':1188,'confirmation_familiar':1848} or manifest['arms'] != arms:
        raise ValueError('Arm denominators failed')
    missing = {}
    for p in predictions:
        row = lookup[p['subject'],p['vid']]
        if 'target' in p or p['role'] != 'confirmation' or p['arm'] != row['arm']:
            raise ValueError('Target/role/arm changed')
        if not p['shared_eeg_available']:
            missing.setdefault((p['subject'],p['vid']),[]).append(p)
    if len(missing) != 1:
        raise ValueError('Fabricated missingness fixture changed')
    for ps in missing.values():
        if len({p['prediction'] for p in ps}) != 1 or sum(p['common_category_fallback'] for p in ps) != 3:
            raise ValueError('Four-method common fallback failed')
    budget = scope['resource_scope']
    expected_counts = dict(zip(('integrity_bytes','header_bytes','signal_bytes','count_values','read_calls'),
        (budget['two_data_plus_two_event_hash_transport_bytes'],budget['data_header_bytes_one_read_pass'],
         budget['selected_signal_payload_bytes'],budget['selected_count_values'],budget['selected_channel_payload_read_calls'])))
    if completed['manufactured_resource_counts'] != expected_counts:
        raise ValueError('Simulated full-scope accounting failed')
    raw = load(OUT/'manufactured_raw.json')
    if ({r['rate_hz'] for r in raw} != {250,1000} or sum(r['raw_count_values'] for r in raw) != 1200000
            or sum(r['signal_payload_bytes'] for r in raw) != 3600000
            or sum(r['read_calls'] for r in raw) != 1980
            or not all(r['count_decode_independent_reference_passed'] and r['finite_features']
                       and r['technical_passed'] and not r['engineered_qc_flags'] for r in raw)):
        raise ValueError('Fabricated decoder receipt failed')
    result = {'date':'2026-10-14', 'passed':True, 'independent_public_audit':True,
        'public_output_bindings':completed['output_bindings'], 'prediction_rows':3036, 'arm_rows':arms,
        'manufactured_missing_trials':1, 'canonical_lock_verified':True, 'simulation_accounting_verified':True,
        'actual_private_source_replay_independently_rerun':False, 'actual_reserved_throughput_qualified':False,
        'private_or_original_inputs_opened':False, 'protected_access_authorized':False,
        'preservation':preserved, 'audit_code_sha256':sha(Path(__file__))}
    with (OUT/'independent_public_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','prediction_rows','arm_rows','private_or_original_inputs_opened')}))

if __name__ == '__main__':
    main()
