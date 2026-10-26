"""Bind the reviewed late-viewing design supplement; no participant values read."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
from xml.etree import ElementTree as ET
import numpy as np
import scipy

REPO=Path(__file__).resolve().parents[1]
WORKSPACE=REPO.parent
PUBLIC=REPO/'results/development/faced_design_proposal_2026-10-11'
PRIVATE=WORKSPACE/'publication_runs/faced_backup_2026-10-10/source_metadata'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    if path.exists():
        raise ValueError('Preserve proposal supplement')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def source_facts():
    # Source-authored METHOD paragraphs only; no validation/result sections.
    root=ET.fromstring((PRIVATE/'faced_bioc.response').read_bytes())
    reference=[p.findtext('text','') for p in root.findall('.//passage') if 'reference electrode at CPz' in p.findtext('text','')]
    methods=[p.findtext('text','') for p in root.findall('.//passage') if 're-referenced' in p.findtext('text','') and 'common average' in p.findtext('text','')]
    if len(reference)!=1 or len(methods)!=1:
        raise ValueError('Source method passages changed')
    if not all(token in reference[0] for token in ('ground electrode at AFz','two cohorts','spatial placement of the electrodes in the two cohorts is the same')):
        raise ValueError('Reference/cohort statement changed')
    if not all(token in methods[0] for token in ('last 30','250','0.05 to 47','independent component analysis','common average','offline')):
        raise ValueError('Published processing declaration changed')
    return {'source_paper':'https://www.nature.com/articles/s41597-023-02650-w',
        'source_pmc':'https://pmc.ncbi.nlm.nih.gov/articles/PMC10600242/',
        'bioc_source_sha256':sha(PRIVATE/'faced_bioc.response'),
        'published_supplement_metadata_sha256':sha(PRIVATE/'faced_supplement_metadata_tables.json'),
        'reference':'CPz','ground':'AFz','cohort_channel_placement':'Authors describe common placement with six device-name changes',
        'author_validation_pipeline':{'observation':'Last 30 seconds','sample_rate_hz':250,'bandpass_hz':[.05,47],
            'artifact_steps':['Outlier-based channel interpolation','Trial-level ICA with FP1/FP2 ocular proxy'],
            'reference':'Common average','mode':'Offline'},
        'author_code_fidelity':'Published description inspected; exact author preprocessing code is not authenticated or executed',
        'explicit_scalp_aliases':{'T3':'T7','T4':'T8','T5':'P7','T6':'P8'},
        'alias_primary_explanation':'https://robertoostenveld.nl/electrode/',
        'generic_alias_warning':'Installed MNE CHANNEL_LOC_ALIASES maps T5/T6 to T9/T10; use explicit source-qualified mapping, not generic match_alias=True',
        'auxiliary_channels':'Both A1/A2-labelled and HEOR/HEOL-labelled slots omitted; physical auxiliary roles are not inferred from labels alone'}


def declare():
    facts=source_facts()
    save(PUBLIC/'preprocessing_plan.json',{'date':'2026-10-11','reason':'Published FACED method uses late-viewing data for retrospective whole-clip ratings; supplement the initial early-viewing candidate before any sample or outcome access',
        'script_sha256':sha(Path(__file__)),'initial_proposal_sha256':sha(PUBLIC/'proposal.json'),
        'prototype_sha256':sha(REPO/'gruxnet/faced_proposal_preprocessing.py'),
        'late_prototype_sha256':sha(REPO/'gruxnet/faced_proposal_preprocessing_v2.py'),
        'source_facts':facts,
        'proposed_raw_interval_relative_to_end_seconds':[-32,0],
        'proposed_core_interval_relative_to_end_seconds':[-30,-2],
        'filter':'Fourth-order Butterworth 0.5-45 Hz, SOS forward/backward, odd extension, offline',
        'resampling':'128 Hz polyphase; Kaiser beta5, linear edge extension',
        'reference':'Common average of fixed 30 common scalp channels; no global fitted ICA/interpolation',
        'scope':'Proposed minimal reproducible processing, not the author-cleaned derivative or a demonstrated best recipe',
        'runtime_versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'raw_files_opened':0,'individual_ratings_opened':0,'eeg_samples_decoded':0,
        'models_fitted':0,'research_question_changed':False,'outcome_access_cleared':False})


def check():
    plan=load(PUBLIC/'preprocessing_plan.json')
    if (plan['script_sha256']!=sha(Path(__file__)) or plan['initial_proposal_sha256']!=sha(PUBLIC/'proposal.json')
        or plan['prototype_sha256']!=sha(REPO/'gruxnet/faced_proposal_preprocessing.py')
        or plan['late_prototype_sha256']!=sha(REPO/'gruxnet/faced_proposal_preprocessing_v2.py')):
        raise ValueError('Supplement dependencies changed')
    return plan


def build():
    plan=check()
    value=load(PUBLIC/'proposal.json')
    value['initial_proposal_sha256']=sha(PUBLIC/'proposal.json')
    value['preprocessing_plan_sha256']=sha(PUBLIC/'preprocessing_plan.json')
    value['recommended_target']['population_scope']='Late-viewing EEG predicts retrospective whole-clip valence under non-neutral elicitation'
    value['preprocessing_geometry']['raw_segment_seconds']=[-32,0]
    value['preprocessing_geometry']['core_seconds']=[-30,-2]
    value['preprocessing_geometry']['time_anchor']='Video end; end index floor(end_marker*native_rate), start=end_index-32*native_rate'
    value['source_preprocessing_facts']=plan['source_facts']
    value['proposed_preprocessing']={k:plan[k] for k in ('filter','resampling','reference','scope','runtime_versions')}
    value['primary_contrast']={'endpoint':'Participant-and-source-film-equal MAE on original 0-7 individual valence',
        'comparison':'EEG-plus-context minus the selected context-only control on new confirmation participants and eight held source-film families',
        'gain_definition':'Context MAE minus EEG-plus-context MAE; positive is improvement',
        'proposed_continuation_threshold':.15,
        'threshold_scope':'Engineering continuation requirement; not a validated psychometric minimum meaningful difference',
        'uncertainty':'Paired crossed participant/film bootstrap; eight held film clusters limit precision',
        'secondary_arms':['New participants on development films after locked refit','EEG-only','Within-film across-participant swaps matched on acquisition group','Within-participant across-film swaps matched on assigned valence'],
        'novelty_status':'Candidate diagnostic, not an established new conference contribution'}
    value['source_quality_gate']='Small identity-declared development-source EEG pilot, all ratings still sealed; bind decoder units/header/body, assess feasibility before supervised fitting'
    value['literature_gate']='Gerster FACED full methods/code remain unavailable; no novelty claim from abstract-only inspection. Kong familiar-video priors and mdJPT already overlap broadly'
    save(PUBLIC/'recommended_proposal.json',value)
    save(PUBLIC/'supplement_verification.json',{'preprocessing_plan_sha256':sha(PUBLIC/'preprocessing_plan.json'),
        'recommended_proposal_sha256':sha(PUBLIC/'recommended_proposal.json'),
        'script_sha256':sha(Path(__file__)),'raw_files_opened':0,'individual_ratings_opened':0,
        'eeg_samples_decoded':0,'models_fitted':0,'research_question_changed':False,'outcome_access_cleared':False})
    print({'recommended_window':'end-30 .. end-2','same_material_and_participant_roles':True,'outcomes_opened':0})


def verify():
    check();receipt=load(PUBLIC/'supplement_verification.json')
    if receipt['preprocessing_plan_sha256']!=sha(PUBLIC/'preprocessing_plan.json') or receipt['recommended_proposal_sha256']!=sha(PUBLIC/'recommended_proposal.json') or receipt['script_sha256']!=sha(Path(__file__)):
        raise ValueError('Supplement binding mismatch')
    value=load(PUBLIC/'recommended_proposal.json');original=load(PUBLIC/'proposal.json')
    if value['rotations']!=original['rotations'] or value['participant_reservation_preserved']!=original['participant_reservation_preserved']:
        raise ValueError('Window supplement changed identity roles')
    from gruxnet.faced_proposal_preprocessing_v2 import late_segment_bounds
    header_path=REPO/'results/development/faced_raw_2026-10-11/headers.json'
    join_path=REPO/'results/development/faced_raw_2026-10-11/joins_numeric_projection.json'
    headers={r['subject']:r['data.bdf'] for r in load(header_path)['records']}
    included=set().union(*(set(v) for v in value['rotations'][0]['clips'].values()))
    checked=0
    for participant in load(join_path)['records']:
        h=headers[participant['subject']];rate=int(h['sample_rates_hz'][0])
        for span in participant['spans']:
            if span['vid'] in included:
                late_segment_bounds(span['start_seconds'],span['end_seconds'],rate,int(h['recording_duration_seconds']*rate))
                checked+=1
    print({'verified':True,'included_metadata_intervals_fit':checked,'raw_files_opened':0,'outcome_access_cleared':False})


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(REPO))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','build','verify'))
    args=parser.parse_args();{'declare':declare,'build':build,'verify':verify}[args.mode]()
