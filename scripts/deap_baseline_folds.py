"""Declare, prepare and resume the authorized matched DEAP feature/control grid."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import scipy
import sklearn
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.deap_baseline_folds import (STUDY, PARENT, SOURCES, MODELS, REPRESENTATIONS,
    cell_id, full_job, prepare, load_inputs, fit_model, fit_prior)
from gruxnet.preprocessing_diagnostic_v2 import atomic, sha, stamp
from gruxnet.heldout_tuning import partitions
from gruxnet.train import seed_everything
from scripts.audit_deap_baseline_folds import audit_model, audit_complete, certificate_valid
from scripts.export_deap_baseline_folds import export


def environment():
    return {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'scikit_learn': sklearn.__version__, 'platform': platform.platform(),
        'device': 'CPU', 'torch_cpu_threads': 4, 'random_state': 42}


def declare(root):
    output = root/STUDY
    if output.exists(): raise FileExistsError('Preserve frozen declaration')
    old = json.loads((root/'heldout_tuning_2026-10-06/plan.json').read_text())
    for name, expected in old['source_sha256'].items():
        if sha(REPO/name) != expected: raise ValueError('Changed historical fitted source')
    upstream = [f'{study}/{name}' for study in (PARENT, 'prestimulus_control_2026-10-09', 'heldout_tuning_2026-10-06')
                for name in ('plan.json', 'verification.json')]
    upstream += ['cache_full_context_deap/trials.csv', 'cache_common14/lineage.json', f'{PARENT}/inputs/deap/prepared.json']
    for study in (PARENT, 'prestimulus_control_2026-10-09', 'heldout_tuning_2026-10-06'):
        verified = json.loads((root/study/'verification.json').read_text())
        if not verified['passed'] or not verified['complete']: raise ValueError('Complete verified predecessor required')
    jobs = [job for job in old['contexts'] if job['dataset'] == 'DEAP']
    if len(jobs) != 160: raise ValueError('Wrong inherited matched grid')
    base = pd.read_csv(root/'cache_full_context_deap/trials.csv'); feasibility = []
    covered = {}
    for job in jobs:
        table, idx = partitions(base, full_job(job))
        covered.setdefault((job['group'], job['arm']), []).extend(table.iloc[idx['test']].trial_id)
        feasibility.append({'job': job, 'counts': {role: len(indexes) for role, indexes in idx.items()},
            'class_counts': {role: table.iloc[indexes].label.value_counts().sort_index().to_dict() for role, indexes in idx.items()},
            'split_trials': {role: table.iloc[indexes].trial_id.tolist() for role, indexes in idx.items()}})
    for trials in covered.values():
        if len(trials) != 1264 or len(set(trials)) != 1264 or set(trials) != set(base.trial_id):
            raise ValueError('Incomplete once-per-trial coverage')
    output.mkdir()
    atomic(output/'feasibility.json', feasibility)
    plan = {'created_utc': stamp(), 'development_only': True, 'research_question_change_approved': False,
        'known_development_evidence': 'All preceding held-out/source diagnostics and baseline-alone results were inspected, including DEAP baseline-relative validation BA72.35/57.29%. This new fixed full grid uses the SAME people/videos; it is development sensitivity, not pristine confirmation or new independent human data.',
        'jobs': jobs, 'models': list(MODELS), 'representations': list(REPRESENTATIONS), 'C': [.01, .1, 1., 10.],
        'counts': {'matched_cells': 160, 'candidate_fits': 5760, 'selected_logistic_heads': 1440,
            'raw_priors': 160, 'test_probability_rows': 50560, 'regular_paired_contrasts': 240, 'within_video_contrasts': 40},
        'source_sha256': {name: sha(REPO/name) for name in SOURCES},
        'upstream_binding': {name: sha(root/name) for name in upstream}, 'feasibility_sha256': sha(output/'feasibility.json'),
        'hypothesis': 'Baseline-relative stimulus features improve DEAP individual-valence prediction over stimulus-only and pre-stimulus-only controls across matched participant/video folds, and retain useful prediction beyond source-only video context. Authorized measurement/baseline exploration; not an adopted new paper question or novel method.',
        'input': 'All1264 corrected binary-valence trials from32people/40videos,16rating5 midpoints excluded. All32named EEG electrodes,128Hz,first40s after measured3s baseline. Maintain full60s offline4..40Hz stimulus filtering before prefix; baseline filtered separately. Each per-trial transform is label-free/fixed. Preparing every observation including target features before fitting is allowed; no test features enter scaler/fitting/source selection, and each test query occurs only after that heads source selection is sealed.',
        'features': 'Natural-log4-band Welch power, Hann256/overlap128/FFT256,128Hz,densitybinwidth.5Hz,half-open4..8/8..14/14..31/31..40Hz,floor1e-12. Stimulus absolute=mean logpower over10nonoverlapping4s windows; stimulus relative=within-window/channel logpower minus logsumexp across4bands then windowmean; baseline-only=3s measured-baseline logpower; baseline-relative=subtract3sbaseline logpower before windowmean.128features per representation; exact earlier740native/baseline and1264common-prefix comparisons. No waveform-baseline subtraction recipe or invented SEED baseline.',
        'partitions': 'Reuse2fixed independent participant/video GROUPS,5video rotations,8participantfolds,both exposed/unexposed arms,session1. Participant train/validation/test disjoint,allroles trial-disjoint,allclasses present. Same test and unseen validation rows across arms; train counts matched perperson/class. In unexposed arm no source panel contains test videos. Familiar validation contains same validation people on actual source-train videos. Exposure differences also include familiar-validation selection regime, not a causal video-identity intervention.',
        'fitting': 'For each of4representations,EEG-only and concatenated EEG+2logvideo-prior features. Context-only logistic uses the same2logprior features. Training-only StandardScaler;balanced LogisticRegression,C.01/.1/1/10,maxiter4000,tol1e-6,random_state42,torchCPUthreads4. Deterministic linear models deliberately have one initialization; extra optimizer seeds would not add useful independent fits. Independent fresh scaler/model refit for every candidate,exactcoefs/directlogisticlink replay. No neural augmentation/target-adaptation/budget extension.',
        'context': 'FixedLaplace1 per-video class counts,unseen video smoothed global source rate. Every TRAINING receiving row excludes ALL labels from that receiving participant,including sourceglobalfallback; validation/test counts use source training labels only. Preserve raw prior and calibrated context withoutEEG. Concatenated context is a matched established linear control, not new fusion method. Raw/calibrated context test probabilities must match all320earlier controls.',
        'selection': 'Equal.5/.5source familiar/unseen class-balanced logloss,minimize;meanpanelBA breaks ties;stablefirstC. Retain everyC train/validation probability andmetric. Write immutable selection/params/split/sha before retrieving that heads test feature orprior. Test outcomes never choose C,group,featurevariant orstop. All8representationmodels andbothcontextcontrols retained;no globally selected winning recipe.',
        'verification': 'All5760candidates independently refit,exact training-only means/scales/coefs/intercepts;directexpit and independent sklearn weightedloss/BA. Raw prior independently compared to previously audited row-wise participant exclusion.32exact older sourcecandidate/scaler/coef sentinels;320older contexttestmatches. All50,560outerrows checked for identical metadata/once-per-original coverage. Testdata remain reuseddevelopment,notfirstparty authenticated.',
        'analysis': 'Only after complete160cellverification. Report combined and each grouping,all20model/arm scores,240paired contrasts: all10unexposed-minus-exposed;baseline-relative minusstimulusabsolute/relative/baselineonly botharms;eachEEGcontext minusitsEEG/rawprior/calibratedcontext botharms;BAandbalancedlogloss in3scopes. Combined averages correctness/loss by observed person/video across2groups,NOTprobabilities orunequalfoldBA.',
        'uncertainty': '10000seed20261006 paired person-only and crossedperson/video percentile draws on32people/40videos,actualobserved classdenominators. Shared weights across models/arms/groupings;fixedfits,unadjusted exploratory intervals. Independent flatobservedtrial computation rechecks allregular points andboth endpointsets. Groupings reuse people/videos and are not independent human replications.',
        'alignment': 'For each selected cell/group/video,exchange allother held-out participants entire featureprediction keeping recipient context/label. Sourceprior identical withinvideo/cell. Report all10models*2arms*2metrics=40aligned-minus-exchanged contrasts. Dyadicrecipient*donor*video bootstrap weights withsame10000draws; disclose ineligible rows/nonestimable draws,neverresampleuntilsignificant. Bothcontextcontrols must be invariant. Sameperson baseline/stimulus associations do not causally identify emotion physiology.',
        'decision': 'Balancedlogloss primary,BAsecondary;inspectbothgroups/alloutcomes. A single favourable grouping/accuracy score,or established preprocessing advantage alone,does not justify novelty/pivot. Evidence for this lead requires stable loss improvement versusstimulus/rest and relevant contextcontrols withcrosseduncertainty;failure stops escalation ofthislead,notselectingsuccessfulseeds. Any conference contribution needs distinctpriorworkgap andlocked independentconfirmation.',
        'stop_resume_publish': 'Fail/preserve timestampedfailure on changed source/data/config,missingclass/coverage,nonconvergence,refit/replay orpublisherfailure. OSexclusiveworkerlock. Resumeonlysealed matchingcases;nooverwriteofunsealedpartialfit. Publishonlyverifiedderivedprobabilities/declarations,excludeEEG/features/coefs. Authorized Git milestones every10new completecells andcompletion;exactownedpaths,main/knownremoteonly,no forcepush.',
        'gate': 'Manuscript/mainquestion unchanged. Show findings/proposal,obtainexplicitauthorapproval andarchive then-current paper immediately before adopting another researchquestion.'}
    atomic(output/'plan.json', plan)
    atomic(output/'progress.json', {'state': 'declared', 'updated_utc': stamp(), 'cells_completed': 0,
        'cells_total': 160, 'research_question_change_approved': False})
    export(output)
    print(json.dumps({'declared': plan['counts'], 'plan_sha256': sha(output/'plan.json')}), flush=True)


def validate(root, output):
    plan = json.loads((output/'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if sha(REPO/name) != expected: raise ValueError(f'Changed frozen source {name}')
    for name, expected in plan['upstream_binding'].items():
        if sha(root/name) != expected: raise ValueError('Changed upstream evidence')
    if sha(output/'feasibility.json') != plan['feasibility_sha256']:
        raise ValueError('Changed declared feasibility')
    return plan


def configure(output):
    if (output/'config.json').exists(): raise FileExistsError('Preserve fitted configuration')
    atomic(output/'config.json', {'plan_sha256': sha(output/'plan.json'), 'environment': environment(),
        'prepared_sha256': sha(output/'inputs/prepared.json')})


def publish(output, message):
    export(output); paths = [f'results/development/{STUDY}']
    def git(*args): return subprocess.run(['git', *args], cwd=REPO, text=True, capture_output=True, check=True)
    if git('branch', '--show-current').stdout.strip() != 'main': raise ValueError('Wrong publisher branch')
    if git('remote', 'get-url', 'origin').stdout.strip() != 'https://github.com/Overproness/GRU-XNet_EEG_Emotion_Recognition.git':
        raise ValueError('Wrong publication destination')
    staged = git('diff', '--cached', '--name-only').stdout.splitlines()
    if any(not any(name == path or name.startswith(path+'/') for path in paths) for name in staged):
        raise ValueError('Unrelated staged changes; preserve them')
    git('add', '--', *paths); git('diff', '--cached', '--check')
    if git('diff', '--cached', '--name-only').stdout.strip():
        git('commit', '-m', message); git('push', 'origin', 'main')


def run(root, limit=None, push=False):
    output = root/STUDY; plan = validate(root, output)
    config = json.loads((output/'config.json').read_text())
    if config['plan_sha256'] != sha(output/'plan.json') or config['environment'] != environment():
        raise ValueError('Changed fitting declaration/environment')
    if config['prepared_sha256'] != sha(output/'inputs/prepared.json'):
        raise ValueError('Changed input binding')
    base, features = load_inputs(output); seed_everything(42)
    started = time.perf_counter(); completed = 0; newly = 0; candidates = 0
    for job in plan['jobs']:
        validate(root, output); fresh = False
        for model in MODELS:
            folder = output/'cells'/cell_id(job)/model
            if (folder/'verification.json').exists(): certificate_valid(folder, output)
            else:
                if model == 'prior': fit_prior(output, base, job)
                else: fit_model(output, base, features, job, model)
                audit_model(output, root, base, features, job, model); fresh = True
            candidates += 0 if model == 'prior' else 4
        completed += 1; newly += int(fresh)
        progress = {'state': 'running', 'updated_utc': stamp(), 'worker_pid': os.getpid(),
            'cells_completed': completed, 'cells_total': 160, 'candidates_verified': candidates,
            'last_cell': cell_id(job), 'new_cells_this_process': newly,
            'elapsed_seconds_this_process': time.perf_counter()-started, 'research_question_change_approved': False}
        atomic(output/'progress.json', progress); print(json.dumps(progress), flush=True)
        if push and fresh and newly % 10 == 0: publish(output, f'Checkpoint DEAP baseline controls: {completed}/160 cells verified')
        if limit is not None and newly >= limit:
            progress['state'] = 'bounded_run_finished'; atomic(output/'progress.json', progress)
            export(output)
            if push: publish(output, f'Verify initial DEAP baseline-control cells: {completed}/160')
            return
    verified = audit_complete(output)
    from scripts.analyze_deap_baseline_folds import analyze
    analyze(output)
    progress['state'] = 'complete'; progress['updated_utc'] = stamp()
    progress['elapsed_seconds_this_process'] = time.perf_counter()-started
    atomic(output/'progress.json', progress); export(output)
    if push: publish(output, 'Complete and verify matched full-fold DEAP baseline and context controls')
    print(json.dumps(verified), flush=True)


def exclusive(root, limit=None, push=False):
    output = root/STUDY
    with (output/'worker.lock').open('a+b') as lock:
        if lock.tell() == 0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, limit, push)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'run', 'export'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    parser.add_argument('--data-root', type=Path, default=REPO.parent/'emotion-recognition-eeg-datasets')
    parser.add_argument('--limit', type=int); parser.add_argument('--push', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); output = root/STUDY
    try:
        if args.command == 'plan': declare(root)
        elif args.command == 'prepare':
            validate(root, output); prepare(output, root, args.data_root.resolve()); configure(output); export(output)
        elif args.command == 'export': export(output)
        else: exclusive(root, args.limit, args.push)
    except Exception as error:
        if output.exists():
            atomic(output/f'FAILURE_{time.time_ns()}.json', {'created_utc': stamp(), 'worker_pid': os.getpid(),
                'error': str(error), 'traceback': traceback.format_exc()})
            progress = json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
            progress.update(state='failed', updated_utc=stamp(), error=str(error)); atomic(output/'progress.json', progress)
        raise
