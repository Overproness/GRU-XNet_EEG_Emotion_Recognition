"""Declare bounded FACED development controls; no original-outcome or EEG I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import scipy
import sklearn
from gruxnet.faced_classical_contract import ALPHAS,BANDS,CATEGORIES,ACQUISITIONS,role_pairs

PUBLIC=REPO/'results/development/faced_development_protocol_2026-10-11'
TARGET=REPO/'results/development/faced_target_contract_2026-10-11'
DESIGN=REPO/'results/development/faced_design_proposal_2026-10-11'
PILOT=REPO/'results/development/faced_source_quality_2026-10-11'
ROLES=REPO/'results/development/faced_backup_2026-10-10/participant_reservation.json'


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    if path.exists():raise ValueError('Preserve prior programme declaration')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def bindings():
    paths=[Path(__file__),REPO/'gruxnet/faced_classical_contract.py',
        REPO/'tests/test_faced_classical_contract.py',REPO/'gruxnet/faced_targets.py',
        REPO/'gruxnet/faced_source_notch.py',REPO/'gruxnet/faced_proposal_preprocessing.py',
        REPO/'gruxnet/faced_proposal_preprocessing_v2.py',ROLES,
        DESIGN/'recommended_proposal.json',TARGET/'plan.json',TARGET/'storage.json',TARGET/'verification.json',
        PILOT/'packet_verification.json',PILOT/'line_noise_plan.json',PILOT/'line_noise_records.json',
        REPO/'results/development/faced_codebooks_2026-10-10/audit.json']
    return {p.relative_to(REPO).as_posix():sha(p) for p in paths}


def manifests():
    roles=load(ROLES);design=load(DESIGN/'recommended_proposal.json')
    result=[];union=set()
    for rotation in design['rotations']:
        pairs=role_pairs(roles,rotation)
        result.append({'rotation':rotation['rotation'],
            'arms':{k:[{'subject':p,'vid':v} for p,v in values] for k,values in pairs.items()},
            'counts':{k:len(v) for k,v in pairs.items()}})
        union.update(p for values in pairs.values() for p in values)
    if len(union)!=1260:raise ValueError('Development union differs from structural audit')
    salt='faced-classical-source-oof-v1-2026-10-11'
    people=sorted(roles['development_source'],key=lambda p:hashlib.sha256((salt+'|'+p).encode()).hexdigest())
    folds=[people[i::5] for i in range(5)]
    return result,[{'subject':p,'vid':v} for p,v in sorted(union)],folds,salt


def declare():
    receipt=load(TARGET/'verification.json')
    if any(sha(REPO/p)!=expected for p,expected in receipt['bindings'].items()):
        raise ValueError('Target metadata qualification changed')
    storage=load(TARGET/'storage.json')
    descriptor=storage['storage_patterns']
    if len(descriptor)!=1 or descriptor[0]['vectors']!=1260 or descriptor[0]['descriptor']['shape']!=[1,12] or descriptor[0]['descriptor']['matlab_class']!=6:
        raise ValueError('Original storage differs from verified double-vector contract')
    m,union,folds,salt=manifests()
    save(PUBLIC/'plan.json',{
        'date':'2026-10-11','status':'declared_not_launched','bindings':bindings(),
        'author_authorization':'Explore alternatives and prepare bounded experiments; manuscript-question adoption still requires explicit author approval',
        'candidate_question':'Does scalp EEG improve individual retrospective valence prediction beyond declared stimulus/acquisition context on new participants and held source-film families?',
        'manuscript_question_changed':False,
        'target':{'field':'score','item_one_based':10,'name':'valence','scale':[0,7],
            'threshold':None,'actual_geometry':[1,12],'actual_class':'mxDOUBLE/miDOUBLE',
            'target_value_eligibility':'Retain 0, 3.5, 7 and every finite in-range item10 value; no value-derived balancing or exclusion',
            'invalid_value_rule':'Stop and preserve the exact failure; no silently removed row, imputed target or substituted item'},
        'input_access':{'future_development_pairs':union,'unique_pairs':1260,
            'current_original_ratings_decoded':0,'current_new_eeg_samples_decoded':0,
            'per_pair':'Only item10 and its declared 30-scalp-channel late trial interval',
            'confirmation':'All 33 reserved people and every confirmation film in every person remain unopened',
            'excluded':'Neutral clips13-16 and version-ambiguous clip22 remain outside the primary experiment',
            'per_fold_source_rule':'A cache union does not grant fitting access: only the current rotation source_fit pairs may enter its scaler, targets, priors or estimator',
            'activation':'A separate exact original-data runner/activation receipt must bind implementation hashes and identities before real outcomes or broader EEG. This declaration alone opens none.'},
        'rotation_manifests':m,'source_participant_oof_folds':folds,'oof_salt':salt,
        'preprocessing':{'base':'Existing source-qualified ADC/common30/late32-second adapter; original files immutable',
            'notch':'Fixed native iirnotch(50,Q=30), forward/backward odd padding before unchanged bandpass/resampling',
            'bandpass':'SciPy butter(N=4,[0.5,45],bandpass,sos) forward/backward: eight-pole bandpass per pass',
            'resampling':'128Hz resample_poly, Kaiser beta5, line edges',
            'core':'end-30..end-2; seven nonoverlapping four-second windows',
            'fitted_processing':'Only source-pair StandardScaler; no global ICA, interpolation, target-adaptive normalization or augmentation',
            'scaler_scope':'Separate source-only scaler for each rotation/feature family; OOF scalers refitted excluding held participants'},
        'features':{'spectral':{'dimension':150,'bands_hz':BANDS,'power':'Welch hann512/noverlap0/nfft512 at128Hz, constant detrend, density sum*0.25Hz, half-open bands',
                'aggregation':'log10(max(power,1e-8 microvolt^2)), mean across all seven windows, canonical channel-major/band-minor order'},
            'context':{'dimension':13,'assigned_categories':CATEGORIES,'acquisition_groups':ACQUISITIONS,
                'numeric':['Catalogue clip duration','Presentation ordinal1..28'],
                'scope':'Experimenter-provided elicitation metadata, not a prediction from EEG or an individual emotion measurement',
                'excluded_features':['Video-ID one-hot in primary unseen arm','Individual labels or evaluated-person/film consensus']},
            'joint':'Concatenate the13context and150EEG features; no target-derived prior fed into the joint model'},
        'models':{'matched_ridge_families':['context_only','spectral_eeg_only','context_plus_spectral_eeg'],
            'ridge_alphas':ALPHAS,'solver':'svd','fit_intercept':True,'sample_weighting':'Equal original source trials; complete source rectangle has one clip per film, so participant/film weighting is uniform',
            'prediction':'Clip all regressor predictions to0..7 before scoring',
            'fixed_context_priors':['Source global median','Source assigned-coarse-valence median with global fallback',
                'Source assigned-category median with global fallback','Source video median only for familiar videos; global fallback for unknown videos'],
            'prior_scope':'OOF priors exclude the held participant group; unseen-film priors never consult that film or targets from evaluated people'},
        'selection':{'per_rotation_per_ridge_family':'Primary unseen-validation participant/film-equal MAE, then unseen MSE, then declared first alpha; all five candidate results retained',
            'familiar_validation':'Diagnostic only; never selects alpha',
            'primary_context_comparator':'Context Ridge selected from its matched five-alpha grid',
            'additional_strong_context_guard':'Also compare joint with the best of global/coarse/category fixed priors on unseen validation; the familiar video prior is diagnostic only',
            'no_model_family_search':'Report all three heads and all priors; no global winning rotation, representation or target selected',
            'tuning_budget_note':'Each Ridge family has five alpha candidates; fixed priors are additional EEG-free controls, not a matched extra EEG search'},
        'capacity_diagnostic':{'scope':'Five participant-disjoint source OOF panels for the selected alpha in each family; exact held-person exclusion, fixed clips',
            'interpretation':'Descriptive source-learning check, conditioned on development-selected alpha; neither independent confirmation nor stimulus-transfer performance',
            'fixed_prior_oof':'Recompute all four priors from the other source people, including familiar source-video medians'},
        'quality_and_failures':{'source':'Authenticate inputs and run all source numerical/QC checks before rating access; numerical failures or review flags hold fitting for documented source-only diagnosis',
            'quality_flags':'Retain all original source-pilot engineering flags and add per-channel processed49-51/1-45Hz ratio>0.10 from the fixed source follow-up; not validated artifact labels',
            'validation_numerical_failure':'Retain evaluated trial and coverage: joint uses declared selected context prediction; EEG-only uses source global median. Source scalers never absorb that input.',
            'validation_quality_flags':'Record flags and coverage without deleting rows or retuning preprocessing from validation values',
            'frozen_settings':'No retrospective filter/channel/target/identity changes from development performance; any continuation is separately preserved'},
        'metrics':{'primary':'One target/prediction per trial; average clips within participant/film, films within participant, then participants',
            'gain':'Context MAE minus joint MAE; positive favors EEG',
            'secondary':['Trial-pooled MAE and MSE','Familiar materials','Selected source OOF','Acquisition breakdown','Leave-one-validation-film-out sensitivity'],
            'uncertainty':'Fixed4000 paired crossed participant/film bootstrap draws, seed425401 per rotation; descriptive intervals with only6held films and no correction for tuning',
            'rotation_dependence':'Same20validation people and overlapping films/labels reused across rotations; do not count rotations as independent replications or pool them as new trials'},
        'continuation':{'threshold_rating_units':.15,'scope':'Preexisting engineering continuation requirement, not a psychometric minimum or conference-worthiness proof',
            'development_lead':'Mean joint gain >=0.15 versus selected context Ridge, positive in at least2of3rotations, also exceeds strongest fixed unseen-context prior; source capacity and coverage must be interpretable',
            'uncertain_or_no_lead':'Retain full results and reassess target/data suitability; do not expand encoder grids automatically',
            'confirmation':'No final confirmation access until a later locked analysis; any paper question change requires findings, author approval and fresh manuscript archive'},
        'budget':{'ridge_candidate_fits':45,'selected_source_oof_ridge_fits':45,'maximum_original_ridge_fits':90,
            'fixed_prior_estimations_base_plus_oof':72,'neural_fits':0,'gpu_hours':0,
            'wall_time_cap_hours':2,'cache':'At most1260trial feature rows and item10 targets; no persisted raw waveform arrays',
            'truncation':'Record any failure/time-cap truncation; never declare an incomplete grid a successful lead'},
        'implementation_status':'Pure numerical/identity/head contracts are tested. No original-data execution runner, active outcome access or experiment output is created by this declaration.',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'original_data_models_fitted':0,'original_ratings_decoded':0,'research_question_changed':False,
        'outcome_access_activated':False})
    save(PUBLIC/'verification.json',{'plan_sha256':sha(PUBLIC/'plan.json'),'script_sha256':sha(Path(__file__)),
        'original_ratings_decoded':0,'original_data_models_fitted':0,'outcome_access_activated':False})
    print({'bounded_programme_declared':True,'unique_future_development_pairs':1260,
           'maximum_ridge_fits':90,'neural_fits':0,'original_ratings_decoded':0,'launched':False})


def verify():
    plan=load(PUBLIC/'plan.json');receipt=load(PUBLIC/'verification.json')
    if plan['bindings']!=bindings() or receipt['plan_sha256']!=sha(PUBLIC/'plan.json') or receipt['script_sha256']!=sha(Path(__file__)):
        raise ValueError('Bounded programme declaration changed')
    m,union,folds,salt=manifests()
    if plan['rotation_manifests']!=m or plan['input_access']['future_development_pairs']!=union or plan['source_participant_oof_folds']!=folds:
        raise ValueError('Programme identity geometry changed')
    print({'programme_verified':True,'rotations':3,'source_fit_trials_per_rotation':560,
           'unseen_validation_trials_per_rotation':120,'original_ratings_decoded':0,'launched':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('declare','verify'))
    args=parser.parse_args();declare() if args.mode=='declare' else verify()
