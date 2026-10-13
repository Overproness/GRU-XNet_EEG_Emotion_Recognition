"""Preserve the completed audited source comparison; no further fit or access."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.audit_faced_coverage_source_qc_2026_10_13 import preservation

PUBLIC = REPO/'results/development/faced_coverage_missing_2026-10-13'
PRIOR = REPO/'results/development/faced_coverage_selection_2026-10-13'

def load(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def finish():
    if (PUBLIC/'fitting_failure.json').exists() or (PUBLIC/'qualification_failure.json').exists():
        raise PermissionError('A failed attempt cannot be finalized as completed')
    activation = load(PUBLIC/'activation.json'); comparison = load(PUBLIC/'comparison.json')
    audit = load(PUBLIC/'independent_audit.json'); qualification = load(PUBLIC/'qualification.json')
    independent_qualification = load(PUBLIC/'qualification_independent_audit.json')
    for name,digest in activation['bindings'].items():
        if sha(REPO/name) != digest: raise ValueError('Frozen input changed: '+name)
    for name,digest in audit['bindings'].items():
        if sha(PUBLIC/name) != digest: raise ValueError('Audited output changed: '+name)
    if (comparison['status'] != 'completed_source_coverage_missing_selection'
            or audit['passed'] is not True or independent_qualification['passed'] is not True
            or comparison['cv_fit_count'] != 90 or comparison['full_source_fit_count'] != 3
            or comparison['validation_access'] is not False or comparison['protected_access'] is not False
            or load(PRIOR/'source_gate.json')['passed'] is not False):
        raise PermissionError('Exact audited source-only completion required')
    fixed = next(r for r in comparison['contrasts'] if r['primary_fixed_alpha'])
    nominee = comparison['source_nomination']; alpha = nominee['alpha']
    selected = next(r for r in comparison['contrasts'] if r['alpha']==alpha)
    gain_field = 'matched_context_minus_distributed_mae' if nominee['model']=='distributed_joint' else 'matched_context_minus_late_mae'
    incremental_lead = selected[gain_field] > 0
    names = ('activation.json','source_policy_plan.json','qualification_started.json',
        'qualification.json','qualification_independent_audit.json','imputation_fixtures.json',
        'fitting_started.json','cached_source_targets_receipt.json','fit_records.jsonl',
        'cv_predictions.json','comparison.json','final_source_state_qualification.json','independent_audit.json')
    receipt = {'date':'2026-10-13','completed_utc':datetime.now(timezone.utc).isoformat(),
        'status':'source_comparison_complete_incremental_lead' if incremental_lead else 'source_comparison_complete_no_incremental_lead',
        'attempt_completed':True,'comparison_executed':True,'policy_id':comparison['policy_id'],
        'source_people':70,'source_trials':980,'source_regions':2940,'available_trials':979,
        'shared_missing_trials':1,'original_source_rows_or_labels_excluded':0,
        'available_regional_vectors_qualified':2937,'unavailable_trial_arrays_decoded':0,
        'independent_fold_numerical_means':20,'independent_full_source_numerical_means':2,
        'ridge_cv_fits':90,'full_source_state_fits':3,'category_prior_constructions':11,
        'held_distinct_pairs_per_candidate':840,'ridge_prediction_rows':7560,'category_prediction_rows':840,
        'real_held_missing_cases':0,'actual_missing_inference_fallback_evaluated':False,
        'source_targets_unique':980,'original_rating_decodes':0,'new_raw_sample_decodes':0,
        'shared_alpha':comparison['common_alpha_selection']['selected_alpha'],
        'algorithmic_source_nominee':nominee,'incremental_eeg_lead_over_matched_context':incremental_lead,
        'fixed_alpha100_contrasts':fixed,
        'candidate_metrics':comparison['candidate_metrics'],
        'category_median_metrics':comparison['category_median_metrics'],
        'fit_wall_seconds':comparison['elapsed_seconds'],'fit_cpu_seconds':comparison['cpu_seconds'],
        'qualification_wall_seconds':qualification['elapsed_seconds'],
        'qualification_cpu_seconds':qualification['cpu_seconds'],
        'verification':{'new_focused_synthetic_tests_passed':56,'independent_mean_replay_passed':True,
            'post_result_arithmetic_audit_passed':True,'private_saved_state_replay_by_worker_passed':True,
            'independent_private_saved_state_replay':False},
        'old_strict_source_quality_gate_passed':False,'quality_thresholds_changed':False,
        'old_held_attempt_changed':False,'validation_numeric_access':False,'confirmation_access':False,
        'protected_exports_qualified':False,'independent_confirmation':False,
        'saved_papers_preservation':preservation(),
        'source_worker_freeze_commit':'63011548a','pre_label_qualification_commit':'1ca3ca5a0',
        'public_bindings':{n:sha(PUBLIC/n) for n in names},'finalizer_sha256':sha(Path(__file__)),
        'interpretation':'Source-only adaptive comparison of fixed spectral recipes under common quality routing. An algorithmic nominee is not evidence that EEG adds predictive utility. Both positive and adverse contrasts remain reported.',
        'next_decision':'Review contribution and confirmation purpose using the actual source results; do not automatically expand model grids or consume reserved outcomes. Future backend, state/fallback export and precision qualification remain separate dependencies.'}
    with (PUBLIC/'completion.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({'status':receipt['status'],'fits':93,'nominee':nominee['model'],
        'common_alpha':alpha,'protected_access':False}))

if __name__ == '__main__': finish()
