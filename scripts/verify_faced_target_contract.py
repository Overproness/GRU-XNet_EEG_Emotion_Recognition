"""Portable target-storage and bounded-programme verification, no originals."""
import argparse
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
TARGET=REPO/'results/development/faced_target_contract_2026-10-11'
PROGRAMME=REPO/'results/development/faced_development_protocol_2026-10-11'
PREFIX='GRU-XNet_EEG_Emotion_Recognition/'


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(receipt,workspace_names=False):
    for name,expected in receipt['bindings'].items():
        if workspace_names:
            if name=='report.tex':path=REPO/'docs/paper_archive/2026-10-05-pre-exploration/report.tex'
            elif name.startswith(PREFIX):path=REPO/name[len(PREFIX):]
            else:raise ValueError('Unexpected workspace dependency namespace')
        else:path=REPO/name
        if sha(path)!=expected:raise ValueError('Target/protocol source binding changed')


def check():
    plan=load(TARGET/'plan.json');compare(plan,True);compare(load(TARGET/'verification.json'))
    for parent in ('faced_source_quality_2026-10-11/packet_verification.json',
                   'faced_raw_2026-10-11/verification.json','faced_design_proposal_2026-10-11/packet_verification.json'):
        compare(load(REPO/'results/development'/parent))
    storage=load(TARGET/'storage.json');rows=storage['records']
    if [r['subject'] for r in rows]!=plan['participants'] or len(rows)!=90:
        raise ValueError('Storage audit participant roles changed')
    descriptors=[]
    for row in rows:
        if set(s['vid'] for s in row['scores'])!=set(plan['clips']) or len(row['scores'])!=14:
            raise ValueError('Storage audit opens a different film set')
        if row['scope']!={'score_values_decoded':0,'score_headers_inspected':14,
                          'protected_score_headers_skipped':14,'accuracy_and_response_matrices_skipped':56}:
            raise ValueError('Structural interpretation boundary changed')
        for score in row['scores']:
            descriptor=score['score_storage'];descriptors.append(descriptor)
            if descriptor['shape']!=[1,12] or descriptor['matlab_class']!=6 or descriptor['mi_numeric_type']!=9 or descriptor['item10_byte_range']!=[72,80] or descriptor['score_values_decoded']:
                raise ValueError('Original qualified score geometry changed')
    if len(descriptors)!=1260 or any(d!=descriptors[0] for d in descriptors):
        raise ValueError('Stored vectors disagree')
    if storage['score_headers_inspected']!=1260 or storage['protected_score_headers_skipped']!=1260 or storage['original_rating_values_decoded'] or storage['original_eeg_samples_decoded']:
        raise ValueError('Audit count/value boundary changed')
    programme=load(PROGRAMME/'plan.json');compare(programme)
    receipt=load(PROGRAMME/'verification.json')
    if receipt['plan_sha256']!=sha(PROGRAMME/'plan.json') or receipt['script_sha256']!=sha(REPO/'scripts/declare_faced_classical.py'):
        raise ValueError('Programme declaration changed')
    roles=load(REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json')
    proposal=load(REPO/'results/development/faced_design_proposal_2026-10-11/recommended_proposal.json')
    union=set()
    for declared,original in zip(programme['rotation_manifests'],proposal['rotations']):
        for arm,people,clips in (
            ('source_fit',roles['development_source'],original['clips']['development_source']),
            ('validation_unseen',roles['development_validation'],original['clips']['development_validation']),
            ('validation_familiar',roles['development_validation'],original['clips']['development_source'])):
            expected=sorted((p,v) for p in people for v in clips)
            actual=[(r['subject'],r['vid']) for r in declared['arms'][arm]]
            if actual!=expected:raise ValueError('Per-fold input/outcome boundary changed')
            union.update(actual)
    if len(programme['rotation_manifests'])!=3 or len(union)!=1260 or union!={(r['subject'],r['vid']) for r in programme['input_access']['future_development_pairs']}:
        raise ValueError('Development union changed')
    folds=programme['source_participant_oof_folds']
    if sorted(p for f in folds for p in f)!=sorted(roles['development_source']) or [len(f) for f in folds]!=[14]*5:
        raise ValueError('OOF participant geometry changed')
    if programme['budget']['maximum_original_ridge_fits']!=90 or programme['budget']['neural_fits']!=0 or programme['status']!='declared_not_launched' or programme['outcome_access_activated'] or programme['original_ratings_decoded'] or programme['research_question_changed']:
        raise ValueError('Bounded experiment status/limit changed')


def build():
    check();path=TARGET/'packet_verification.json'
    if path.exists():raise ValueError('Preserve packet receipt')
    paths=[TARGET/name for name in ('plan.json','storage.json','verification.json')]
    paths += [PROGRAMME/name for name in ('plan.json','verification.json')]+[Path(__file__)]
    path.write_text(json.dumps({'bindings':{p.relative_to(REPO).as_posix():sha(p) for p in paths},
        'synthetic_target_decoder_tests_passed':20,'classical_contract_tests_passed':8,'test_failures':0,
        'original_score_headers_qualified':1260,'original_rating_values_decoded':0,
        'new_original_eeg_samples_decoded':0,'original_data_models_fitted':0,
        'bounded_ridge_fits_declared':90,'experiment_launched':False,
        'outcome_access_activated':False,'research_question_changed':False,
        'scope':'Portable code/archive/identity/type/budget receipt replay; numeric target correctness checked on manufactured bytes only until separately activated original-data access'},indent=2)+'\n',encoding='utf-8')


def verify():
    check();receipt=load(TARGET/'packet_verification.json');compare(receipt)
    print({'target_programme_packet_verified':True,'bound_packet_files':len(receipt['bindings']),
           'original_score_headers_qualified':1260,'original_ratings_decoded':0,
           'bounded_programme_launched':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('build','verify'))
    args=parser.parse_args();build() if args.mode=='build' else verify()
