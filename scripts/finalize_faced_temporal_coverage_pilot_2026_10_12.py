"""Seal or verify the public source-only pilot; never access EEG or ratings."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / 'results/development/faced_temporal_coverage_pilot_2026-10-12'
PREFLIGHT = REPO / 'results/development/faced_temporal_coverage_preflight_2026-10-12'
ARCHIVE = REPO / 'docs/paper_archive/2026-10-12-before-improvement'

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_bindings(items):
    for name, expected in items.items():
        assert sha(REPO/name) == expected, name

def checks():
    activation = read(PUBLIC/'activation_v2.json')
    plan = read(PREFLIGHT/'plan.json')
    summary = read(PUBLIC/'summary.json')
    independent = read(PUBLIC/'independent_audit.json')
    assert independent['passed']
    check_bindings(independent['input_bindings']); check_bindings(independent['output_bindings'])
    assert independent['public_segments'] == 24 and independent['counts_sample_values'] == 16560000
    assert independent['independent_availability_decisions'] == 24
    assert independent['channel_rule_evaluations'] == 720
    assert independent['late_count_window_feature_hash_replays'] == 8
    check_bindings(activation['code_bindings']); check_bindings(activation['input_bindings'])
    check_bindings(plan['code_bindings']); check_bindings(plan['input_bindings'])
    assert summary['activation_sha256'] == sha(PUBLIC/'activation_v2.json')
    assert summary['plan_sha256'] == sha(PREFLIGHT/'plan.json')
    assert summary['passed'] and summary['source_quality_gate_passed']
    assert summary['source_trials'] == 8 and summary['segments'] == 24
    assert summary['unavailable_segments'] == 0 and summary['late_exact_replays'] == 8
    assert summary['counts_sample_values_decoded'] == 16560000
    assert summary['elapsed_seconds'] <= activation['seconds_cap']
    assert summary['cpu_seconds'] <= activation['cpu_seconds_cap']
    assert summary['prior_source_qc_files_read'] == 8 and summary['new_fits'] == 0
    for name in ('original_rating_access','cached_target_access','validation_signal_access','confirmation_access','raw_array_persistence'):
        assert summary[name] is False, name
    rows = [read(p) for p in sorted(PUBLIC.glob('segment_*.json'))]
    expected = {(r['subject'],r['vid'],r['region']):r for r in plan['pilot']['segments']}
    assert len(rows) == 24 and len(expected) == 24
    assert len({(r['subject'],r['vid'],r['region']) for r in rows}) == 24
    assert sum(r['samples_decoded'] for r in rows) == 16560000
    for row in rows:
        key = row['subject'],row['vid'],row['region']
        assert row['sample_interval'] == expected[key]['sample_interval']
        assert row['role'] == 'source' and row['eeg_available']
        assert row['extra_boundary_samples_decoded'] == 0 and not row['reasons']
        assert row['repair_count'] == 0 and row['artifact_decision'] == 'unchanged'
    paper = read(REPO/'paper/exploratory_evaluation_2026-10-12/build_receipt.json')
    check_bindings(paper['bindings'])
    copies = read(ARCHIVE/'manifest.json')['files']
    for item in copies:
        assert sha(REPO/item['source']) == item['sha256']
        assert sha(ARCHIVE/item['archive']) == item['sha256']
    return {'passed':True,'frozen_before_execution_revision':'d88c29dd2',
        'source_trials':8,'source_segments':24,'availability_passes':24,
        'late_exact_replays':8,'selected_scalar_counts':16560000,
        'original_paper_bindings_unchanged':31,'archived_files_unchanged':len(copies),
        'new_fits':0,'ratings_accessed':False,'protected_outcomes_accessed':False,
        'validation_waveforms_accessed':False,'source_quality_pilot_only':True,
        'independent_confirmation_claimed':False,'research_question_changed':False,
        'limitations':'Eight-trial source QC feasibility; no full source preparation, prediction utility or novelty established.'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    receipt = PUBLIC/'completion.json'
    result = checks()
    if args.seal:
        assert not receipt.exists(), 'Preserve sealed completion'
        files = [p for p in PUBLIC.rglob('*') if p.is_file()]
        files += [REPO/n for n in (
            'scripts/finalize_faced_temporal_coverage_pilot_2026_10_12.py',
            'docs/publication/FACED_Temporal_Coverage_Pilot_Findings_2026-10-12.md',
            'docs/publication/FACED_Temporal_Coverage_Pilot_Post_Review_2026-10-12.md',
            'docs/publication/FACED_Temporal_Coverage_Pilot_Independent_Review_2026-10-12.md',
            'docs/publication/GRU-XNet_Paper_Improvement_Programme_2026-10-12.md')]
        result['bindings'] = {p.relative_to(REPO).as_posix():sha(p) for p in sorted(files)}
        with receipt.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result,stream,indent=2); stream.write('\n')
    else:
        saved = read(receipt); check_bindings(saved['bindings'])
        for key,value in result.items():
            assert saved[key] == value,key
        result['sealed_bindings_checked'] = len(saved['bindings'])
    print(json.dumps({k:v for k,v in result.items() if k != 'bindings'}))

if __name__ == '__main__':
    main()
