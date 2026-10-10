"""Verify the protected reservation, public source evidence and optional private bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from qualify_emo_mc import PRIVATE, PUBLIC, REPO, WORK, sha, ranked, family_closure, parse_edf_header, write_json

SOURCE_PATHS = [
    'scripts/qualify_emo_mc.py', 'scripts/audit_emo_mc_pilot.py',
    'scripts/audit_emo_mc_complete.py', 'scripts/analyze_emo_mc_annotations.py',
    'scripts/analyze_emo_mc_metadata.py', 'scripts/verify_emo_mc_qualification.py',
    'tests/test_emo_mc_qualification.py', 'tests/test_emo_mc_annotations.py',
    'docs/publication/GRU-XNet_EmoEEG_MC_Qualification_2026-10-10.md',
]


def invariants(local=False):
    q=json.loads((PUBLIC/'qualification.json').read_text())
    p=json.loads((PUBLIC/'reservation.json').read_text())
    a=json.loads((PUBLIC/'annotation_join_checks.json').read_text())
    technical=json.loads((PUBLIC/'technical_consistency.json').read_text())
    original=subprocess.check_output(['git','show','341696fe8:results/development/emo_mc_qualification_2026-10-10/reservation.json'],cwd=REPO)
    assert sha(original)==sha((PUBLIC/'reservation.json').read_bytes()),'Frozen reservation changed'
    assert p['qualification_sha256']==sha((PUBLIC/'qualification.json').read_bytes())
    for rel,expected in p['source_sha256'].items():assert sha((REPO/rel).read_bytes())==expected,rel
    roles=p['participants'];sets=[set(roles[k]) for k in ('confirmation_reserved','development_validation','development_source','quarantine')]
    assert [len(s) for s in sets]==[20,10,27,2]
    assert sum(len(s) for s in sets)==len(set.union(*sets))==59
    assert set.union(*sets)==set(q['raw_participants'])
    pool=ranked(p['participant_hash_namespace'],set(q['raw_participants'])-sets[-1])
    assert set(pool[:20])==sets[0] and set(pool[20:30])==sets[1] and set(pool[30:])==sets[2]
    assert family_closure(q['materials'],p['materials']['initial_number_selections'])==p['materials']['family_closed_numbers']
    keys={m['material_key'] for m in q['materials']};held=set(p['materials']['reserved_keys']);dev=set(p['materials']['development_keys'])
    assert len(held)==14 and len(dev)==28 and not held&dev and held|dev==keys
    assert q['fitting_ready'] is False and a['fitting_ready'] is False
    assert q['rating_values_decoded']==q['waveform_samples_decoded']==q['models_fitted']==0
    assert a['rating_values_decoded']==a['waveform_samples_decoded']==a['models_fitted']==0
    assert len(q['recording_sources'])==103 and len(q['behaviour_source_hashes'])==59
    indexed={r['path']:r for r in q['recording_sources']}
    complete_objects=[]
    for proof_file,script in [('pilot_authentication.json','audit_emo_mc_pilot.py'),('complete_pilot_authentication.json','audit_emo_mc_complete.py')]:
        proof=json.loads((PUBLIC/proof_file).read_text())
        assert proof['participant'] in sets[2]
        assert proof['rating_values_decoded']==proof['waveform_samples_decoded']==proof['models_fitted']==0
        assert proof['research_question_changed'] is False
        if proof_file.startswith('complete'):
            declaration=json.loads((PUBLIC/'complete_pilot_declaration.json').read_text())
            assert declaration['source_sha256']==sha((REPO/'scripts'/script).read_bytes())
            assert declaration['reservation_sha256']==sha(original)
            assert proof['declaration_sha256']==sha((PUBLIC/'complete_pilot_declaration.json').read_bytes())
        else:
            assert proof['audit_script_sha256']==sha((REPO/'scripts'/script).read_bytes())
        for obj in proof['objects']:
            source=indexed[obj['path']]
            assert obj['full_annex_digest_matches'] and obj['sha256']==source['annex_sha256'] and obj['bytes']==source['file_bytes']
            complete_objects.append(obj)
    assert len(complete_objects)==3
    assert sum(obj['bytes'] for obj in complete_objects)==1055080026
    assert a['analysis_script_sha256']==sha((REPO/'scripts/analyze_emo_mc_annotations.py').read_bytes())
    assert technical['script_sha256']==sha((REPO/'scripts/analyze_emo_mc_metadata.py').read_bytes())
    assert len(technical['split_participant_clock_checks'])==44
    assert technical['overlapping_header_intervals']==sum(c['header_gap_s']<0 for c in technical['split_participant_clock_checks'])==31
    assert len(technical['sidecar_sampling_discrepancies'])==1
    old=json.loads((REPO/'results/development/cbramod_readout_2026-10-09/plan.json').read_text())
    for rel,digest in old['source_sha256'].items():assert sha((REPO/rel).read_bytes())==digest,'Prior frozen source changed: '+rel
    headers=0;full_bytes=0
    if local:
        assert sha((PRIVATE/'collection.json').read_bytes())==q['collection_sha256']
        for r in q['recording_sources']:
            b=PRIVATE.joinpath('headers',r['path']).with_suffix('.header').read_bytes()
            assert sha(b)==r['header_sha256']
            parsed=parse_edf_header(b,r['file_bytes'])
            assert parsed['geometry_consistent']==(r['path'].split('/')[0]!='sub-22')
            headers+=1
        for obj in complete_objects:
            path=PRIVATE/'objects'/obj['path']
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            assert digest==obj['sha256'] and path.stat().st_size==obj['bytes']
            full_bytes+=obj['bytes']
    source_paper=REPO/'docs/paper_archive/2026-10-05-pre-exploration/report.tex'
    pdf=REPO/'docs/paper_archive/2026-10-05-pre-exploration/DL-Report.pdf'
    expected_tex='06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0'
    expected_pdf='921f07f271bc4dadbabaf62ede384c543681b9fe0aec02c490cf7d76ef9de101'
    assert sha(source_paper.read_bytes())==expected_tex and sha(pdf.read_bytes())==expected_pdf
    if local:assert sha((WORK/'report.tex').read_bytes())==expected_tex
    return dict(passed=True,recording_headers_in_source_audit=103,private_headers_rechecked=headers,
                whole_objects_authenticated=3,private_complete_object_bytes_rechecked=full_bytes,
                reserved_confirmation_participants=20,reserved_material_keys=14,
                prior_frozen_source_files=len(old['source_sha256']),fitting_ready=False,
                rating_values_decoded=0,waveform_samples_decoded=0,models_fitted=0,
                manuscript_unchanged=True,research_question_changed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['bind','verify']);parser.add_argument('--local',action='store_true')
    args=parser.parse_args();result=invariants(args.local)
    proof_path=PUBLIC/'verification.json'
    if args.command=='bind':
        paths=SOURCE_PATHS+[p.relative_to(REPO).as_posix() for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.name!='verification.json']
        write_json(proof_path,dict(checks=result,artifact_sha256={p:sha((REPO/p).read_bytes()) for p in paths},
                   focused_tests='10 tests passed in this session; byte/invariant verification does not rerun those tests',
                   scope='Source identity and metadata/reservation integrity; not waveform quality, certified rating joins, model performance or scientific novelty'))
    else:
        saved=json.loads(proof_path.read_text())
        for path,digest in saved['artifact_sha256'].items():assert sha((REPO/path).read_bytes())==digest,path
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
