"""Portable source-pilot evidence replay: no originals, samples or ratings needed."""
import argparse
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PUBLIC=REPO/'results/development/faced_source_quality_2026-10-11'
PREFIX='GRU-XNet_EEG_Emotion_Recognition/'


def load(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_bindings(receipt):
    for name,expected in receipt['bindings'].items():
        if sha(REPO/name)!=expected: raise ValueError('Public byte binding changed: '+name)


def check():
    plan=load(PUBLIC/'plan.json')
    for name,expected in plan['bindings'].items():
        if name=='report.tex': path=REPO/'docs/paper_archive/2026-10-05-pre-exploration/report.tex'
        elif name.startswith(PREFIX): path=REPO/name[len(PREFIX):]
        else: raise ValueError('Unknown plan dependency namespace')
        if sha(path)!=expected: raise ValueError('Pilot plan/code/archive dependency changed')
    for name in ('verification.json','analysis_verification.json','line_noise_plan.json','line_noise_verification.json'):
        compare_bindings(load(PUBLIC/name))
    for parent in ('faced_design_proposal_2026-10-11/packet_verification.json','faced_raw_2026-10-11/verification.json'):
        compare_bindings(load(REPO/'results/development'/parent))
    if load(PUBLIC/'verification.json')['script_sha256']!=sha(REPO/'scripts/run_faced_source_quality_pilot.py'):
        raise ValueError('Pilot entry point changed')
    clarification=load(PUBLIC/'line_noise_execution_clarification.json')
    if clarification['line_noise_plan_sha256']!=sha(PUBLIC/'line_noise_plan.json'):
        raise ValueError('Preexecution clarification changed its parent')
    roles=load(REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json')
    design=load(REPO/'results/development/faced_design_proposal_2026-10-11/recommended_proposal.json')
    allowed_clips=set(design['rotations'][0]['clips']['development_source'])
    initial=load(PUBLIC/'records.json')['records'];follow=load(PUBLIC/'line_noise_records.json')['records']
    if len(initial)!=8 or len(follow)!=8: raise ValueError('Expected eight source trials in both phases')
    count=0;groups=set()
    for expected,a,b in zip(plan['trial_identities'],initial,follow):
        identity={'subject':a['subject'],'vid':a['vid']}
        if identity!=expected or identity!={'subject':b['subject'],'vid':b['vid']}:
            raise ValueError('Pilot identity changed')
        if a['subject'] not in roles['development_source'] or a['vid'] not in allowed_clips:
            raise ValueError('Pilot crosses source-only boundary')
        if a['sample_interval']!=b['sample_interval'] or a['access_range_ledger_sha256']!=b['range_ledger_sha256']:
            raise ValueError('Follow-up decoded a new sample range')
        if a['output_shape']!=[7,30,512] or a['decoded_values']!=30*32*a['native_rate_hz']:
            raise ValueError('Pilot geometry/accounting changed')
        if a['ratings_decoded'] or b['ratings_decoded'] or a['extra_boundary_samples_decoded']:
            raise ValueError('Rating/sample seal changed')
        if a['calibration_max_absolute_error_uv']>1e-8 or not a['every_real_count_scalar_reference_checked']:
            raise ValueError('Decoder/calibration check failed')
        # Recompute follow-up's acceptance from published aggregates, not its bool.
        passed=(max(c['notched_processed_line_ratio'] for c in b['channels'])<=.10
                and all(not c['notched_review_flags'] for c in b['channels'])
                and all(c['relative_1_30_power_change']<=.05 for c in b['channels']
                        if c['original_1_30_power_uv2']>1e-6))
        if passed!=b['engineering_diagnostic_passed']:
            raise ValueError('Follow-up acceptance does not replay')
        count+=a['decoded_values'];groups.add(a['acquisition_group'])
    summary=load(PUBLIC/'summary.json')
    if count!=summary['real_count_values_decoded_and_scalar_checked'] or groups!=set(summary['acquisition_groups']):
        raise ValueError('Summary counts/groups do not replay')
    if summary['ratings_decoded'] or summary['models_fitted'] or summary['research_question_changed']:
        raise ValueError('Experiment/question scope changed')
    if load(PUBLIC/'line_noise_records.json')['repeated_real_count_values_decoded']!=count:
        raise ValueError('Repeated decoded-value accounting changed')
    return count


def build():
    count=check();path=PUBLIC/'packet_verification.json'
    if path.exists(): raise ValueError('Preserve final packet receipt')
    paths=[p for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.suffix in ('.json','.csv','.png')]
    paths.append(Path(__file__))
    value={'bindings':{p.relative_to(REPO).as_posix():sha(p) for p in paths},
        'focused_tests_passed':27,'test_failures':0,'source_trials':8,
        'unique_original_sample_values_decoded':count,'repeated_same_sample_values_in_followup':count,
        'ratings_decoded':0,'models_fitted':0,'research_question_changed':False,
        'primary_proposal_changed':False,'supervised_access_cleared':False,
        'public_replay_scope':'Declared identities, code/archive/receipt hashes, sample accounting and aggregate notch criteria; independent numerical sample replay requires originals and a scoped declaration'}
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def verify():
    count=check();receipt=load(PUBLIC/'packet_verification.json');compare_bindings(receipt)
    print({'public_packet_verified':True,'bound_files':len(receipt['bindings']),
           'source_trials':8,'unique_original_sample_values_decoded':count,'ratings_decoded':0,
           'models_fitted':0,'supervised_access_cleared':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('build','verify'))
    args=parser.parse_args();build() if args.mode=='build' else verify()
