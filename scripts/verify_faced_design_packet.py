"""Verify a proposal packet and source-only pilot identities, without raw access."""
import argparse
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PUBLIC=REPO/'results/development/faced_design_proposal_2026-10-11'
JSON_FILES=['plan.json','proposal.json','verification.json','preprocessing_plan.json',
    'recommended_proposal.json','supplement_verification.json','source_quality_pilot_candidate.json','phase_verification.json']
CODE=['scripts/prepare_faced_design_proposal.py','scripts/finalize_faced_design_proposal.py',
    'scripts/verify_faced_design_packet.py','gruxnet/faced_proposal_preprocessing.py',
    'gruxnet/faced_proposal_preprocessing_v2.py','tests/test_faced_design_proposal.py',
    'tests/test_faced_proposal_preprocessing.py','tests/test_faced_late_proposal.py']


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pilot():
    pilot=load(PUBLIC/'source_quality_pilot_candidate.json')
    proposal=load(PUBLIC/'recommended_proposal.json')
    reservation=load(REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json')
    if pilot['recommended_proposal_sha256']!=sha(PUBLIC/'recommended_proposal.json'):
        raise ValueError('Pilot references changed proposal')
    identities=pilot['trial_identities']
    source_clips=set(proposal['rotations'][0]['clips']['development_source'])
    if len(identities)!=8 or len({(r['subject'],r['vid']) for r in identities})!=8:
        raise ValueError('Expected eight unique pilot trials')
    if any(r['subject'] not in reservation['development_source'] or r['vid'] not in source_clips for r in identities):
        raise ValueError('Pilot crosses source-only boundary')
    if pilot['status']!='proposed_not_launched' or pilot['outcomes_to_open']!=0:
        raise ValueError('Pilot activation/outcome boundary changed')
    headers={r['subject']:r['data.bdf'] for r in load(REPO/'results/development/faced_raw_2026-10-11/headers.json')['records']}
    groups={(headers[s]['sample_rates_hz'][0],int(s[-3:])<61) for s in pilot['subjects']}
    if len(pilot['subjects'])!=4 or groups!={(250.0,True),(1000.0,True),(1000.0,False)}:
        raise ValueError('Source pilot acquisition coverage changed')


def build():
    check_pilot()
    path=PUBLIC/'packet_verification.json'
    if path.exists():
        raise ValueError('Preserve packet receipt')
    paths=[PUBLIC/name for name in JSON_FILES]+[REPO/name for name in CODE]
    path.write_text(json.dumps({'bindings':{str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in paths},
        'source_quality_pilot_trials':8,'source_quality_pilot_launched':False,
        'raw_files_opened':0,'individual_ratings_opened':0,'eeg_samples_decoded':0,
        'models_fitted':0,'research_question_changed':False,'outcome_access_cleared':False},indent=2)+'\n',encoding='utf-8')


def verify():
    check_pilot();receipt=load(PUBLIC/'packet_verification.json')
    if any(sha(REPO/name)!=expected for name,expected in receipt['bindings'].items()):
        raise ValueError('Packet code/evidence changed')
    print({'verified':True,'bound_packet_files':len(receipt['bindings']),
        'proposed_source_only_pilot_trials':8,'pilot_launched':False,'outcome_access_cleared':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('build','verify'))
    args=parser.parse_args();build() if args.mode=='build' else verify()
