"""Declare/replay development score-storage metadata without decoding ratings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1];WORKSPACE=REPO.parent
sys.path.insert(0,str(REPO))
from gruxnet.faced_targets import audit_score_storage,MAX_BYTES

PUBLIC=REPO/'results/development/faced_target_contract_2026-10-11'
RAW=WORKSPACE/'Data/Data'
PARENT=REPO/'results/development/faced_source_quality_2026-10-11'
QUALIFIED=REPO/'results/development/faced_raw_2026-10-11'
DESIGN=REPO/'results/development/faced_design_proposal_2026-10-11'
ROLE_PATH=REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json'


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    if path.exists():raise ValueError('Preserve previous target receipt: '+path.name)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def previous_bindings():
    for path in (PARENT/'packet_verification.json',QUALIFIED/'verification.json',DESIGN/'packet_verification.json'):
        for name,expected in load(path)['bindings'].items():
            if sha(REPO/name)!=expected:raise ValueError('Earlier evidence changed')


def identities():
    roles=load(ROLE_PATH);design=load(DESIGN/'recommended_proposal.json')
    people=sorted(roles['development_source']+roles['development_validation'])
    clips=sorted(set(design['rotations'][0]['clips']['development_source'])|
                 set(design['rotations'][0]['clips']['development_validation']))
    if len(people)!=90 or len(clips)!=14 or set(people)&set(roles['confirmation']):
        raise ValueError('Development participant/material geometry changed')
    for rotation in design['rotations']:
        c=rotation['clips']
        if set(clips)!=set(c['development_source'])|set(c['development_validation']) or set(clips)&set(c['confirmation']):
            raise ValueError('Development rotations change sealed materials')
    return people,clips


def bindings():
    paths=[Path(__file__),REPO/'gruxnet/faced_targets.py',REPO/'tests/test_faced_targets.py',
        REPO/'scripts/audit_faced_raw.py',PARENT/'packet_verification.json',QUALIFIED/'verification.json',
        DESIGN/'recommended_proposal.json',DESIGN/'packet_verification.json',ROLE_PATH,
        REPO/'results/development/faced_codebooks_2026-10-10/audit.json',
        REPO/'results/development/emo_mc_qualification_2026-10-10/reservation.json',WORKSPACE/'report.tex']
    return {p.relative_to(WORKSPACE).as_posix():sha(p) for p in paths}


def declare():
    previous_bindings();people,clips=identities()
    save(PUBLIC/'plan.json',{
        'date':'2026-10-11','authorization':'User requested item-10 storage and selective decoder verification, then bounded experiment declaration',
        'bindings':bindings(),'participants':people,'clips':clips,
        'allowed':['Opaque original After_remarks.mat bytes and SHA256 for 90 development people only',
            'Existing trial/vid identities, checked against prior authenticated joins',
            'Inner score matrix flags/dimensions/type/length for 14 development clips only',
            'Synthetic numerical decoder tests and manufactured SciPy roundtrip'],
        'forbidden':['Any original score number, including item10',
            'Protected participants, confirmation-film score headers or numbers',
            'Other score items, Accuracy and ResponseTime interpretation','EEG sample access','Fitting or manuscript-question change'],
        'skip_rule':'Read each score outer tag, defer inner parsing until trial/vid are qualified; skip nondevelopment scores without inner metadata parsing',
        'target_contract':{'item_one_based':10,'source_item_name':'valence','scale':[0,7],
            'threshold':None,'eligibility':'No value-dependent filtering; missing/nonfinite/out-of-scale handling remains a future supervised data-quality gate'},
        'original_ratings_to_decode':0,'original_eeg_samples_to_decode':0,'models_to_fit':0,
        'outcome_access_cleared':False,'research_question_changed':False})
    print({'metadata_declared':True,'development_people':90,'development_clips':14,'ratings_to_decode':0})


def check():
    previous_bindings();plan=load(PUBLIC/'plan.json')
    if plan['bindings']!=bindings() or (plan['participants'],plan['clips'])!=identities():
        raise ValueError('Metadata declaration changed')
    return plan


def audit():
    plan=check()
    if (PUBLIC/'storage.json').exists():raise ValueError('Preserve completed structural audit')
    files={r['member']:r for r in load(QUALIFIED/'extraction.json')['records']}
    joins={r['subject']:r for r in load(QUALIFIED/'joins_numeric_projection.json')['records']}
    rows=[];counts=Counter()
    for subject in plan['participants']:
        path=RAW/subject.replace('sub-','sub')/'After_remarks.mat'
        if path.stat().st_size>MAX_BYTES:raise ValueError('Behavioral file exceeds fixed byte budget')
        data=path.read_bytes();expected=files[f'Data/{path.parent.name}/After_remarks.mat']['sha256']
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Original behavioral bytes changed')
        projected,scope=audit_score_storage(data,plan['clips'],joins[subject]['identity_rows'])
        rows.append({'subject':subject,'original_file_sha256':expected,'scores':projected,'scope':scope})
        for r in projected:counts[json.dumps(r['score_storage'],sort_keys=True)]+=1
    save(PUBLIC/'storage.json',{'plan_sha256':sha(PUBLIC/'plan.json'),'records':rows,
        'storage_patterns':[{'descriptor':json.loads(key),'vectors':value} for key,value in sorted(counts.items())],
        'source_files_opened':90,'score_headers_inspected':sum(r['scope']['score_headers_inspected'] for r in rows),
        'protected_score_headers_skipped':sum(r['scope']['protected_score_headers_skipped'] for r in rows),
        'accuracy_and_response_matrices_skipped':sum(r['scope']['accuracy_and_response_matrices_skipped'] for r in rows),
        'original_rating_values_decoded':0,'original_eeg_samples_decoded':0,'models_fitted':0,
        'research_question_changed':False,'outcome_access_cleared':False,
        'limits':'Storage/type/item-position qualification; actual rating range, missingness, variance, validity and predictive signal remain unopened'})
    save(PUBLIC/'verification.json',{'bindings':{
        p.relative_to(REPO).as_posix():sha(p) for p in (PUBLIC/'plan.json',PUBLIC/'storage.json')},
        'original_rating_values_decoded':0,'models_fitted':0,'outcome_access_cleared':False})
    print({'structural_audit_passed':True,'people':len(rows),'score_headers_inspected':90*14,
           'original_ratings_decoded':0,'storage_patterns':len(counts)})


def verify():
    check();receipt=load(PUBLIC/'verification.json')
    if any(sha(REPO/path)!=expected for path,expected in receipt['bindings'].items()):
        raise ValueError('Completed target-storage evidence changed')
    value=load(PUBLIC/'storage.json')
    if value['original_rating_values_decoded'] or value['models_fitted'] or value['outcome_access_cleared']:
        raise ValueError('Target-storage boundary changed')
    print({'storage_verified':True,'score_headers_inspected':value['score_headers_inspected'],
           'original_ratings_decoded':0,'outcome_access_cleared':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','audit','verify'))
    args=parser.parse_args();{'declare':declare,'audit':audit,'verify':verify}[args.mode]()
