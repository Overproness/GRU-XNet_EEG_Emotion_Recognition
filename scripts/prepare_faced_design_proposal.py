"""Metadata-only FACED design proposal; no raw, rating, or model access.

Uses only already qualified public metadata JSON. Candidate material assignments
are proposed, not permission to open outcomes or an adopted paper question.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
WORKSPACE=REPO.parent
PUBLIC=REPO/'results/development/faced_design_proposal_2026-10-11'
SALT='GRU-XNet_FACED_metadata_design_proposal_v1_2026-10-11'
INPUTS={
    'codebook':REPO/'results/development/faced_codebooks_2026-10-10/audit.json',
    'raw_summary':REPO/'results/development/faced_raw_2026-10-11/summary.json',
    'raw_headers':REPO/'results/development/faced_raw_2026-10-11/headers.json',
    'raw_joins':REPO/'results/development/faced_raw_2026-10-11/joins_numeric_projection.json',
    'participant_reservation':REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json',
    'manuscript':WORKSPACE/'report.tex',
    'emo_reservation':REPO/'results/development/emo_mc_qualification_2026-10-10/reservation.json'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    if path.exists():
        raise ValueError('Preserve declared proposal evidence: '+path.name)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def rank(kind,key):
    return hashlib.sha256(f'{SALT}/{kind}/{key}'.encode()).hexdigest()


def families(rows,excluded=()):
    by_family={}
    for row in rows:
        if row['clip'] in excluded:
            continue
        key=row['source_film_sha256']
        entry=by_family.setdefault(key,{'family':key,'clips':[],'assigned_category':row['assigned_category'],
            'assigned_valence':row['assigned_valence']})
        if (entry['assigned_category'],entry['assigned_valence'])!=(row['assigned_category'],row['assigned_valence']):
            raise ValueError('Shared source title has incompatible category assignments; manual grouping review required')
        entry['clips'].append(row['clip'])
    for entry in by_family.values():
        entry['clips'].sort()
    return sorted(by_family.values(),key=lambda r:r['family'])


def eligibility(subject,vid,reservation,rotation,stage):
    # Pure identity predicate; it never receives or decodes an outcome value.
    if not isinstance(subject,str) or type(vid) is not int:
        raise ValueError('Identity types changed')
    material=next((role for role,ids in rotation['clips'].items() if vid in ids),None)
    if material is None:
        return False
    if stage=='source_training':
        return subject in reservation['development_source'] and material=='development_source'
    if stage=='selection':
        return subject in reservation['development_validation'] and material=='development_validation'
    if stage=='familiar_validation_diagnostic':
        return subject in reservation['development_validation'] and material=='development_source'
    if stage=='locked_development_refit':
        return subject in reservation['development_source']+reservation['development_validation'] and material!='confirmation'
    if stage in ('proposal','confirmation'):
        # Confirmation access requires a separate locked execution declaration.
        return False
    raise ValueError('Unknown access stage')


def rotations(rows):
    grouped=families(rows,excluded=[13,14,15,16,22])
    by_category={}
    for row in grouped:
        if row['assigned_valence'] not in ('Negative','Positive'):
            raise ValueError('Primary population includes a neutral elicitation clip')
        by_category.setdefault(row['assigned_category'],[]).append(row)
    if len(grouped)!=22 or len(by_category)!=8 or any(len(v) not in (2,3) for v in by_category.values()):
        raise ValueError('Expected eight non-neutral categories with two/three film families')
    ordered={category:sorted(values,key=lambda r:rank('family-order',r['family']))
             for category,values in sorted(by_category.items())}
    fixed_confirm=[values[1] for values in ordered.values()]
    switchable=[category for category,values in ordered.items() if len(values)==3]
    patterns=list(itertools.product((0,1),repeat=len(switchable)))
    patterns.sort(key=lambda bits:rank('development-rotation',''.join(map(str,bits))))
    # Complement maximizes source/validation material changes while holding all
    # confirmation identities and two low-family categories fixed.
    first=patterns[0]
    second=tuple(1-bit for bit in first)
    third=next(bits for bits in patterns if bits not in (first,second) and sum(a!=b for a,b in zip(bits,first))==3)
    output=[]
    for index,bits in enumerate((first,second,third)):
        source,validation=[],[]
        mapping=dict(zip(switchable,bits))
        for category,values in ordered.items():
            if len(values)==2:
                source.append(values[0])
            else:
                candidates=[values[0],values[2]]
                source.append(candidates[mapping[category]])
                validation.append(candidates[1-mapping[category]])
        material={'development_source':source,'development_validation':validation,'confirmation':fixed_confirm}
        clips={role:sorted(c for family in group for c in family['clips']) for role,group in material.items()}
        output.append({'rotation':index,'families':material,'clips':clips,
            'family_counts':{k:len(v) for k,v in material.items()},
            'assigned_valence_family_counts':{k:dict(Counter(r['assigned_valence'] for r in v)) for k,v in material.items()},
            'assigned_category_support':{k:sorted({r['assigned_category'] for r in v}) for k,v in material.items()},
            'status':'proposed_only_not_outcome_access_authorization'})
    return output,{'possible_development_rotations':len(patterns),'selected_rotations':3,
        'confirmation_roles_identical_across_rotations':True,'switchable_categories':switchable,
        'categories_always_in_source_without_validation_materials':[c for c,v in ordered.items() if len(v)==2]}


def declare():
    save(PUBLIC/'plan.json',{'date':'2026-10-11','timezone':'Asia/Karachi',
        'scope':'Metadata-only target/split/preprocessing proposal; no individual outcomes or EEG numbers',
        'inputs':{key:{'path':str(p.relative_to(WORKSPACE)).replace('\\','/'),'sha256':sha(p)} for key,p in INPUTS.items()},
        'script_sha256':sha(Path(__file__)),
        'salt':SALT,'all_faced_material_outcomes_remain_sealed':True,
        'family_rule':'Use existing normalized source-title hash; exact-title grouping is a minimum, not media/franchise independence authentication',
        'primary_candidate':'Individual continuous valence, original 0-7 scale; non-neutral elicitation clips excluding version-ambiguous clip22',
        'candidate_split':'Per each of eight non-neutral elicitation categories, one family source and one confirmation; third family validation where available',
        'extra_development_rotations':'Two fixed metadata-only source/validation permutations; confirmation unchanged',
        'participant_roles':'Preserve existing 70 source / 20 validation / 33 confirmation identities',
        'not_an_adopted_paper_question':True,'research_question_changed':False,
        'individual_ratings_opened':0,'eeg_samples_decoded':0,'models_fitted':0,'outcome_access_cleared':False})


def check_plan():
    plan=load(PUBLIC/'plan.json')
    if plan['script_sha256']!=sha(Path(__file__)) or any(sha(INPUTS[k])!=v['sha256'] for k,v in plan['inputs'].items()):
        raise ValueError('Declared design inputs changed')
    return plan


def calculate():
    check_plan()
    rows=load(INPUTS['codebook'])['stimuli']
    reservation=load(INPUTS['participant_reservation'])
    headers=load(INPUTS['raw_headers'])['records']
    joins=load(INPUTS['raw_joins'])['records']
    primary_rotations,rotation_info=rotations(rows)
    all_families=families(rows)
    source_counts=Counter(r['assigned_category'] for r in all_families)
    variants=[]
    for name,excluded,target in (
        ('nine_assigned_emotions',[], 'Assigned elicitation category; not individual experience'),
        ('assigned_binary_valence',[13,14,15,16],'Assigned positive/negative elicitation; not individual self-report'),
        ('all_clips_continuous_valence',[],'Individual continuous rating; no category coverage requirement'),
        ('recommended_continuous_primary',[13,14,15,16,22],'Individual continuous rating under non-neutral elicitation; version-ambiguous clip excluded')):
        selected=families(rows,excluded)
        variants.append({'candidate':name,'excluded_clips':excluded,'clips':sum(len(r['clips']) for r in selected),
            'minimum_exact_title_families':len(selected),'target':target,
            'structurally_feasible_for_own_target':name!='nine_assigned_emotions',
            'families_per_category':dict(sorted(Counter(r['assigned_category'] for r in selected).items())),
            'three_way_all_nine_categories_supported':not any(n<3 for n in source_counts.values()),
            'continuous_target_numeric_support_known':False,
            'structural_caveat':('Neutral has one family; Fear has two. Full category coverage in every material role is impossible'
                if name in ('nine_assigned_emotions','all_clips_continuous_valence') else
                'Coarse elicitation balance does not establish individual valence balance or rating variance')})
    included=set(range(1,29))-{13,14,15,16,22}
    durations=[s['duration_seconds'] for r in joins for s in r['spans'] if s['vid'] in included]
    if len(durations)!=123*23 or min(durations)<32.0:
        raise ValueError('Proposed 32-second trial-local segment exceeds an included marker span')
    acquisition={r['subject']:(str(r['data.bdf']['sample_rates_hz'][0]),'legacy_A1_A2' if int(r['subject'][-3:])<61 else 'HEOR_HEOL') for r in headers}
    participant_summary={role:{'participants':len(reservation[role]),
        'acquisition_groups':dict(Counter('/'.join(acquisition[s]) for s in reservation[role]))}
        for role in ('development_source','development_validation','confirmation')}
    trial_counts=[]
    for rotation in primary_rotations:
        clips=rotation['clips']
        trial_counts.append({'rotation':rotation['rotation'],
            'initial_source_trials':70*len(clips['development_source']),
            'unseen_material_validation_trials':20*len(clips['development_validation']),
            'familiar_validation_trials':20*len(clips['development_source']),
            'locked_development_refit_trials':90*(len(clips['development_source'])+len(clips['development_validation'])),
            'final_unseen_confirmation_trials':33*len(clips['confirmation']),
            'final_familiar_confirmation_trials':33*(len(clips['development_source'])+len(clips['development_validation']))})
    uncertainty=[]
    for participant_sd,film_sd,interaction_sd in ((.1,.1,.2),(.3,.1,.6),(.3,.3,.6),(.5,.5,1.0)):
        se=(participant_sd**2/33+film_sd**2/8+interaction_sd**2/(33*8))**.5
        uncertainty.append({'assumed_participant_gain_sd':participant_sd,'assumed_film_gain_sd':film_sd,
            'assumed_interaction_gain_sd':interaction_sd,'approximate_standard_error':se,
            'normal_approximate_80pct_power_5pct_two_sided_detectable_gain':(1.959964+.841621)*se,
            'scope':'Hypothetical independent crossed random effects, not empirical variance/power or a promise; eight film clusters limit precision'})
    return {'date':'2026-10-11','status':'proposal_only_not_adopted_paper_question',
        'target_candidates':variants,'recommended_target':{'name':'individual_continuous_valence',
            'item_one_based':10,'scale':[0,7],'threshold':None,'eligibility':'Public non-neutral elicitation metadata only; all individual rating values remain eligible',
            'population_scope':'Early-viewing EEG predicts retrospective whole-clip valence under non-neutral elicitation',
            'excluded_clips':{'13':'One shared neutral source family; outside declared primary elicitation population',
                '14':'Same neutral family','15':'Same neutral family','16':'Same neutral family',
                '22':'Unresolved content alignment across known clip versions; exclude globally, not selected people'}},
        'rotations':primary_rotations,'rotation_audit':rotation_info,'participant_reservation_preserved':participant_summary,
        'proposed_trial_counts':trial_counts,'included_original_trial_population':123*23,
        'minimum_included_marker_span_seconds':min(durations),
        'preprocessing_geometry':{'raw_segment_seconds':[0,32],'core_seconds':[2,30],
            'target_sample_rate_hz':128,'common_scalp_channels':30,'nonoverlapping_window_seconds':4,
            'windows_per_trial':7,'core_samples_per_trial':3584,'window_shape':[7,30,512]},
        'hypothetical_precision_scenarios':uncertainty,
        'individual_rating_support_and_signal_quality':'Unknown; no outcomes or samples inspected',
        'family_authentication':'Exact source-title grouping only; physical clip/franchise independence not certified',
        'models_fitted':0,'individual_ratings_opened':0,'eeg_samples_decoded':0,
        'research_question_changed':False,'outcome_access_cleared':False}


def build():
    save(PUBLIC/'proposal.json',calculate())
    save(PUBLIC/'verification.json',{'plan_sha256':sha(PUBLIC/'plan.json'),
        'proposal_sha256':sha(PUBLIC/'proposal.json'),'script_sha256':sha(Path(__file__)),
        'individual_ratings_opened':0,'eeg_samples_decoded':0,'models_fitted':0,'research_question_changed':False})
    print({'proposed':True,'family_roles':[8,6,8],'development_rotations':3,'outcome_access_cleared':False})


def verify():
    check_plan();receipt=load(PUBLIC/'verification.json')
    if receipt['plan_sha256']!=sha(PUBLIC/'plan.json') or receipt['proposal_sha256']!=sha(PUBLIC/'proposal.json') or receipt['script_sha256']!=sha(Path(__file__)):
        raise ValueError('Frozen proposal binding changed')
    if calculate()!=load(PUBLIC/'proposal.json'):
        raise ValueError('Metadata proposal replay mismatch')
    print({'verified':True,'candidate_rotations':3,'raw_or_rating_files_opened':0,'outcome_access_cleared':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','build','verify'))
    args=parser.parse_args();{'declare':declare,'build':build,'verify':verify}[args.mode]()
