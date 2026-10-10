"""Check completed metadata evidence, immutable reservations and optional bytes.

Integrity/source consistency does not certify the unresolved trial endpoints or
authorize opening outcomes. This verifier never parses ratings or EEG samples.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from collect_emo_mc_historical_events import ROOT, PUBLIC
from qualify_emo_mc import REPO, WORK, git_blob, sha, write_json
from verify_emo_mc_qualification import PUBLIC as PRIOR_PUBLIC, invariants as prior_invariants
from analyze_emo_mc_join_resolution import video_endpoint

SOURCES=[
    'scripts/collect_emo_mc_historical_events.py',
    'scripts/collect_emo_mc_historical_events_addendum.py',
    'scripts/collect_emo_mc_identity_maps.py',
    'scripts/analyze_emo_mc_join_resolution.py',
    'scripts/verify_emo_mc_join_resolution.py',
    'tests/test_emo_mc_join_resolution.py',
    'docs/publication/GRU-XNet_EmoEEG_MC_Join_Resolution_2026-10-10.md',
    'docs/publication/EmoEEG_MC_Metadata_Inquiry_Draft_2026-10-10.md',
]


def read(name):
    return json.loads((PUBLIC/name).read_text())


def checks(local=False, allow_pending_binding=False):
    prior=prior_invariants(False)
    prior_proof=json.loads((PRIOR_PUBLIC/'verification.json').read_text())
    for rel,digest in prior_proof['artifact_sha256'].items():
        assert sha((REPO/rel).read_bytes())==digest,'Previous completed evidence changed: '+rel
    plan=read('plan.json');addendum=read('retrieval_addendum.json');identity_plan=read('identity_plan.json')
    for commit,name in [('db1a6b93e','plan.json'),('246da9f8d','retrieval_addendum.json'),('98fc6a71e','identity_plan.json')]:
        published=subprocess.check_output(['git','show',commit+':'+(PUBLIC/name).relative_to(REPO).as_posix()],cwd=REPO)
        assert sha(published)==sha((PUBLIC/name).read_bytes()),'Declaration changed: '+name
    assert plan['collector_sha256']==sha((REPO/SOURCES[0]).read_bytes())
    assert addendum['adapter_sha256']==sha((REPO/SOURCES[1]).read_bytes())
    assert identity_plan['collector_sha256']==sha((REPO/SOURCES[2]).read_bytes())
    reservation=sha((PRIOR_PUBLIC/'reservation.json').read_bytes())
    assert reservation==plan['reservation_sha256']==identity_plan['reservation_sha256']
    assert addendum['original_plan_sha256']==sha((PUBLIC/'plan.json').read_bytes())
    auth=read('historical_authentication.json');identity_auth=read('identity_authentication.json')
    assert auth['plan_sha256']==sha((PUBLIC/'plan.json').read_bytes())
    assert identity_auth['plan_sha256']==sha((PUBLIC/'identity_plan.json').read_bytes())
    assert len(auth['sources'])==auth['source_tables']==60
    original_sources={r['path']:r for r in plan['historical_sources']}
    for record in auth['sources']:
        source=original_sources[record['path']]
        assert record['git_blob']==source['sha'] and record['bytes']==source['size']
        assert addendum['exact_byte_limits'][record['path']]==record['bytes']
    assert len(identity_auth['sources'])==len(identity_plan['sources'])==120
    assert sum(r['status']=='identity_schema_verified' for r in identity_auth['sources'])==identity_auth['schema_verified']==118
    supplements=read('identity_schema_supplements.json')['supplements']
    assert {(r['participant'],r['context']) for r in supplements}=={('sub-25','ima'),('sub-25','vid')}
    assert all(r['trial_ordinal_supplied'] is False and r['material_codes']==21 for r in supplements)

    findings=read('join_resolution.json');manifest=read('complete_pilot_join_candidates.json')
    assert findings['analysis_script_sha256']==sha((REPO/SOURCES[3]).read_bytes())
    assert findings['reservation_sha256']==reservation
    assert findings['historical_markers']==sum(r['event_rows'] for r in auth['sources'])==10693
    assert findings['historical_event_tables']==60
    assert findings['historical_durations_all_zero'] and not findings['historical_run_ids_supplied']
    assert len(findings['participant_checks'])==60 and len(findings['archive_orders_not_exact'])==19
    assert sum(r['candidate_counts_match_readme'] for r in findings['participant_checks'])==47
    coded=[r for r in findings['participant_checks'] if r['behaviour_identity_has_material_codes']]
    assert len(coded)==30 and all(len(r['behaviour_to_archive_order_checks'])==2 and all(c['behaviour_codes_equal_archive_identity_order'] for c in r['behaviour_to_archive_order_checks'].values()) for r in coded)
    assert findings['sub54_metadata_order_recovered'] and findings['sub54_original_quarantine_role_retained']
    assert all(r['ordered_markers_match_at_edf_precision'] for r in findings['historical_export_pilot_checks'])
    assert [r['authenticated_rows'] for r in findings['historical_export_pilot_checks']]==[92,38]
    assert all(r['maximum_onset_difference_s']<=r['tolerance_s'] for r in findings['historical_export_pilot_checks'])
    assert len(findings['video_symbol_number_consistency'])==21
    assert {r['current_spreadsheet_number'] for r in findings['video_symbol_number_consistency']}==set(range(1,22))
    assert len(findings['older_catalogue_duration_changes'])==13
    assert not findings['imagery_symbol_to_spreadsheet_number_verified']
    assert not findings['actual_video_end_or_button_semantics_verified']
    assert not findings['rating_column_semantics_explicitly_numbered']
    assert not findings['protected_roles_changed'] and findings['no_reallocation']
    assert not findings['fitting_ready'] and not findings['research_question_changed']
    for field in ('rating_values_decoded','waveform_samples_decoded','models_fitted'):
        assert findings[field]==auth[field]==identity_auth[field]==0

    trials=manifest['trials'];assert len(trials)==42
    assert {r['behaviour_row_candidate'] for r in trials}==set(range(1,43))
    assert all(r['behaviour_row_equals_raw_trial_ordinal'] and not r['fitting_allowed'] for r in trials)
    video=[r for r in trials if r['context']=='vid'];imagery=[r for r in trials if r['context']=='ima']
    assert len(video)==len(imagery)==21
    assert all(r['material_mask']['status']=='unknown_identity_exclude' for r in imagery)
    assert sum(r['material_mask']['status']=='reserved_exclude' for r in video)==7
    for r in video:
        e=r['endpoint_checks'];recalculated=video_endpoint(r['start_s'],r['rating_start_s'],e['catalogue_duration_s'])
        assert e==recalculated and e['actual_playback_end_verified'] is False and e['fitting_allowed'] is False
    outliers=findings['video_endpoint_mismatches_over_1s'];assert len(outliers)==1
    assert outliers[0]['symbol']=='dis8' and abs(outliers[0]['candidate_end_difference_s']-97.6017)<1e-9
    assert outliers[0]['overlap_between_two_30s_candidates_s']==0
    discovery=read('source_discovery.json')
    assert discovery['original_execution_log_found'] is False
    assert discovery['explicit_numbered_score_codebook_found'] is False
    comparisons=discovery['cross_archive_behaviour_identity_hash_checks']
    assert len(comparisons)==4 and all(len(r['tables'])==1 and r['tables'][0]['matches_current_git_blob'] and r['tables'][0]['matches_current_source_sha256'] and r['tables'][0]['score_values_decoded']==0 and r['tables'][0]['original_table_saved'] is False for r in comparisons)
    if local:
        assert sha((WORK/'report.tex').read_bytes())=='06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0'

    event_files=identity_files=0
    if local:
        assert sha((ROOT/'historical_collection.json').read_bytes())==auth['collection_sha256']
        assert sha((ROOT/'identity_collection.json').read_bytes())==identity_auth['collection_sha256']
        for record in auth['sources']:
            body=(ROOT/'historical_events'/record['path']).read_bytes()
            assert sha(body)==record['sha256'] and git_blob(body)==record['git_blob'] and len(body)==record['bytes']
            event_files+=1
        for record in identity_auth['sources']:
            body=(ROOT/'identity_maps'/record['context']/(record['participant']+'.mat')).read_bytes()
            assert sha(body)==record['sha256'] and len(body)==record['bytes']
            identity_files+=1
        for record in discovery['relevant_sources']:
            body=(ROOT/'source_metadata'/record['name']).read_bytes()
            assert len(body)==record['bytes'] and sha(body)==record['sha256']
        for record in read('archive_crosswalk_checks.json')['root_override_metadata']:
            body=(ROOT/'source_metadata'/record['name']).read_bytes()
            assert len(body)==record['bytes'] and sha(body)==record['sha256']
    links=0
    for document in [PUBLIC/'README.md',REPO/SOURCES[6],REPO/SOURCES[7]]:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            if re.match(r'(?:https?://|#|mailto:)',link):continue
            target=(document.parent/link).resolve()
            if not (allow_pending_binding and target==PUBLIC/'verification.json'):
                assert target.exists(),'Broken phase link: '+link
            links+=1
    return dict(passed=True,historical_event_tables=60,historical_markers=10693,
                identity_map_files=120,primary_strict_schemas=118,supplemental_single_vectors=2,
                coded_behaviour_archive_order_checks=30,complete_pilot_candidate_rows=42,
                local_historical_bytes_rechecked=event_files,local_identity_files_rechecked=identity_files,
                protected_confirmation_participants=20,protected_material_keys=14,local_links_checked=links,
                prior_qualification_bound_blobs=len(prior_proof['artifact_sha256']),prior_frozen_code_files=prior['prior_frozen_source_files'],
                manuscript_unchanged=True,research_question_changed=False,fitting_ready=False,
                rating_values_decoded=0,waveform_samples_decoded=0,models_fitted=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('bind','verify'))
    parser.add_argument('--local',action='store_true')
    args=parser.parse_args();result=checks(args.local,allow_pending_binding=args.command=='bind')
    path=PUBLIC/'verification.json'
    if args.command=='bind':
        assert not path.exists(),'Completed binding is immutable; use a new phase for changes'
        paths=SOURCES+[p.relative_to(REPO).as_posix() for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.name!='verification.json']
        write_json(path,dict(checks=result,artifact_sha256={rel:sha((REPO/rel).read_bytes()) for rel in paths},
                            focused_tests='18 new tests and 10 prior qualification/annotation tests passed in this session; verifier does not rerun them',
                            scope='Source/evidence integrity and protected metadata gates; unresolved execution joins, score semantics and actual endpoints remain unverified'))
    else:
        proof=json.loads(path.read_text())
        for rel,digest in proof['artifact_sha256'].items():
            assert sha((REPO/rel).read_bytes())==digest,'Completed phase changed: '+rel
    print(json.dumps(result,indent=2))
