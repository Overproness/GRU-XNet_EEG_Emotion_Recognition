"""Preserve the audited source-QC hold; never decode targets or fit a model."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.audit_faced_coverage_source_qc_2026_10_13 import preservation

PUBLIC = REPO/'results/development/faced_coverage_selection_2026-10-13'

def load(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def finish():
    names = ('activation.json','activation_v2.json','README.md','preparation_started.json',
             'source_regions.jsonl','source_gate.json','source_qc_independent_audit.json',
             'independent_audit.json','source_geometry_diagnostics.json')
    fit_outputs = ('fitting_started.json','fitting_failure.json','cached_source_targets_receipt.json',
                   'fit_records.jsonl','cv_predictions.json','comparison.json',
                   'final_source_state_qualification.json')
    if any((PUBLIC/name).exists() for name in fit_outputs) or (PUBLIC/'preparation_failure.json').exists():
        raise PermissionError('Only a completed source-QC hold without fitting is finalized here')
    gate = load(PUBLIC/'source_gate.json'); qc = load(PUBLIC/'source_qc_independent_audit.json')
    audit = load(PUBLIC/'independent_audit.json'); activation = load(PUBLIC/'activation_v2.json')
    if (gate['passed'] is not False or gate['source_regions'] != 2940
            or gate['cached_targets_converted'] != 0 or gate['new_fits'] != 0
            or qc['passed'] is not True or qc['source_quality_gate_passed'] is not False
            or audit['passed'] is not True or audit['source_quality_gate_passed'] is not False
            or audit['cv_fits'] != 0 or audit['full_source_fits'] != 0
            or qc['source_regions_sha256'] != sha(PUBLIC/'source_regions.jsonl')
            or qc['source_gate_sha256'] != sha(PUBLIC/'source_gate.json')
            or audit['source_qc_independent_audit_sha256'] != sha(PUBLIC/'source_qc_independent_audit.json')
            or gate['activation_sha256'] != sha(PUBLIC/'activation_v2.json')):
        raise ValueError('Audited strict source hold and exact dependencies required')
    for name,digest in activation['bindings'].items():
        if sha(REPO/name) != digest:
            raise ValueError('Experimental binding changed: '+name)
    receipt = {
        'date':'2026-10-13','completed_utc':datetime.now(timezone.utc).isoformat(),
        'status':'held_no_source_finalist','attempt_completed':True,
        'completed_phases':['complete_source_signal_preparation','independent_source_qc_audit'],
        'comparison_executed':False,'accuracy_or_negative_predictive_result':False,
        'source_trials':980,'source_people':70,'source_regions':2940,
        'available_regions':2940-gate['unavailable_regions'],
        'unavailable_regions':gate['unavailable_regions'],'unavailable_trials':gate['unavailable_trials'],
        'failures':gate['failures'],'by_region':gate['by_region'],
        'late_exact_replays':gate['late_exact_replays'],
        'native_scalar_counts':gate['counts_sample_values_decoded'],
        'preparation_wall_seconds':gate['elapsed_seconds'],'preparation_cpu_seconds':gate['cpu_seconds'],
        'cached_source_targets_converted':0,'original_rating_decodes':0,
        'ridge_cv_fits':0,'full_source_state_fits':0,'category_prior_constructions':0,
        'fixed_alpha100_coverage_gain':None,'shared_alpha':None,'source_nominee':None,
        'validation_numeric_access':False,'confirmation_access':False,
        'saved_papers_preservation':preservation(),
        'source_quality_arithmetic_audit_passed':True,'source_quality_scientific_gate_passed':False,
        'synthetic_verification':{
            'passed_tests':100,'elapsed_seconds':4.07,
            'command':'python -m pytest -q tests/test_faced_coverage_execution.py tests/test_faced_coverage_selection.py tests/test_faced_temporal_coverage.py tests/test_faced_coverage_qc_audit.py tests/test_faced_coverage_results_audit.py -p no:cacheprovider',
            'scope':'Manufactured inputs, role/QC/byte-range guards, SVD matching, budget and metric/selection arithmetic; no actual target fitting'},
        'worker_freeze_commit':qc['freeze_commit'],
        'public_bindings':{name:sha(PUBLIC/name) for name in names},
        'finalizer_sha256':sha(Path(__file__)),
        'disposition':'Strict unchanged complete-source QC held all target conversion and fitting. No failed region was excluded, imputed, repaired again or used to promote a historical model within this attempt.',
        'next_dependency':'A separately declared, label-blind common missing-EEG handling policy must be qualified before a new matched source comparison. The fixed confirmation proposal remains inactive without a source finalist.',
        'limits':'This completed engineering attempt is not a comparison result, evidence of absent EEG utility, independent confirmation, architectural novelty or a revised paper. Whole-file integrity checks transported bytes without interpreting protected outcomes.'}
    with (PUBLIC/'completion.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({'status':receipt['status'],'source_regions':2940,
                      'unavailable_regions':receipt['unavailable_regions'],'fits':0,
                      'confirmation_access':False}))

if __name__ == '__main__':
    finish()
