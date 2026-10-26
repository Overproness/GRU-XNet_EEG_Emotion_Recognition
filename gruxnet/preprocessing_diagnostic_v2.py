"""Source-only montage, normalization, duration and DEAP baseline diagnostics.

Earlier fitted experiment modules are imported without edits. New outputs must
never overwrite a preceding declaration, cache or fitted result.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from .data import (COMMON_CHANNELS, DEAP_CHANNELS, SEED_LABELS, deap_reference,
                   digest, roots, seed_channels, verify_deap_labels, write_json)
from .eegnet_control import EEGNetControl
from .heldout_tuning import partitions, SOURCES as PREVIOUS_SOURCES
from .learning_controls import draw_stream
from .material_controls import state_digest
from .prepare import preprocess_signal
from .train import seed_everything
from scripts.source_bn_diagnostic import calibrate

REPO = Path(__file__).resolve().parents[1]
STUDY = 'preprocessing_diagnostic_v2_2026-10-09'
RATES = (.001, .0003)
STEPS = (200, 600)
ROLES = ('train', 'validation_unseen', 'validation_familiar')
BANDS = ((4, 8), (8, 14), (14, 31), (31, 40))
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES, 'gruxnet/preprocessing_diagnostic_v2.py',
    'scripts/preprocessing_diagnostic_v2.py', 'scripts/audit_preprocessing_diagnostic_v2.py',
    'scripts/export_preprocessing_diagnostic_v2.py', 'gruxnet/preprocessing_diagnostic.py',
    'scripts/preprocessing_diagnostic.py', 'scripts/audit_preprocessing_diagnostic.py',
    'scripts/export_preprocessing_diagnostic.py')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    write_json(temporary, value)
    temporary.replace(path)


def recipes(dataset):
    result = [dict(montage=m, normalization=n, seconds=s, baseline=False)
              for s in (4, 40) for m in ('common14', 'native')
              for n in ('source_channel', 'trial_zscore')]
    if dataset == 'DEAP':
        result += [dict(montage=m, normalization='source_channel', seconds=4, baseline=True)
                   for m in ('common14', 'native')]
    return result


def recipe_id(recipe):
    return f'{recipe["montage"]}_{recipe["normalization"]}_{recipe["seconds"]}s_b{int(recipe["baseline"])}'


def case_id(job):
    return f'{job["dataset"].lower()}_g{job["group"]}_{recipe_id(job["recipe"])}_lr{job["lr"]}'


def panel(dataset, group):
    return dict(dataset=dataset, model='eegnet', group=group, initialization=42,
                session=1, rotation=0, fold=0, arm='unexposed')


def source_panel(base, dataset, group):
    annotated, idx = partitions(base, panel(dataset, group))
    allowed = np.sort(np.concatenate([idx[r] for r in ROLES]))
    if set(allowed) & set(idx['test']):
        raise ValueError('Outer-test trial in source diagnostic')
    table = annotated.iloc[allowed].reset_index(drop=True)
    remap = {int(old): new for new, old in enumerate(allowed)}
    selected = {r: np.array([remap[int(i)] for i in idx[r]], dtype=int) for r in ROLES}
    return table, selected, sorted(set(annotated.iloc[idx['test']].subject_id))


def bases(root):
    return {d: pd.read_csv(root/f'cache_full_context_{d.lower()}/trials.csv')
            for d in ('DEAP', 'SEEDIV')}


def declare(root):
    output = root/STUDY
    if output.exists():
        raise FileExistsError('Preserve the frozen declaration')
    old = json.loads((root/'heldout_tuning_2026-10-06/plan.json').read_text())
    for name, checksum in old['source_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError(f'Previously frozen source changed: {name}')
    if not json.loads((root/'heldout_tuning_2026-10-06/verification.json').read_text())['complete']:
        raise ValueError('Completed predecessor verification required')
    jobs = [dict(dataset=d, group=g, initialization=42, recipe=r, lr=lr)
            for d in ('DEAP', 'SEEDIV') for g in (1, 2) for r in recipes(d) for lr in RATES]
    assert len(jobs) == 72
    output.mkdir(parents=True)
    bindings = {}
    for dataset, base in bases(root).items():
        for group in (1, 2):
            table, idx, excluded = source_panel(base, dataset, group)
            folder = output/'panels'/f'{dataset.lower()}_g{group}'
            folder.mkdir(parents=True)
            table.to_csv(folder/'trials.csv', index=False)
            record = {'dataset': dataset, 'group': group,
                      'roles': {r: table.iloc[idx[r]].trial_id.tolist() for r in ROLES},
                      'counts': {r: len(idx[r]) for r in ROLES},
                      'class_counts': {r: table.iloc[idx[r]].label.value_counts().sort_index().to_dict() for r in ROLES},
                      'excluded_test_participants': excluded, 'trials_sha256': sha(folder/'trials.csv')}
            atomic(folder/'panel.json', record)
            bindings[folder.name] = sha(folder/'panel.json')
    plan = {'created_utc': stamp(), 'development_only': True,
            'revision': 2,
            'superseded_attempt': {'study': 'preprocessing_diagnostic_2026-10-09',
                'plan_sha256': sha(root/'preprocessing_diagnostic_2026-10-09/plan.json'),
                'failure_records': {p.name: sha(p) for p in (root/'preprocessing_diagnostic_2026-10-09').glob('FAILURE_*.json')},
                'reason': 'Initial context control wrongly exponentiated an existing probability vector; the probability-sum check stopped the first classical panel before any EEG trajectory. Revision2 uses the existing prior probabilities directly and adds an integration test. All initial attempt artifacts preserved and excluded; montage/normalization/duration/selection/budget/seed choices unchanged.'},
            'research_question_change_approved': False, 'jobs': jobs,
            'counts': {'neural_trajectories': 72, 'neural_recipe_panels': 36,
                       'selected_classical_heads': 20, 'classical_C_fits': 80,
                       'neural_candidates': 288},
            'source_sha256': {f: sha(REPO/f) for f in SOURCES}, 'panel_bindings': bindings,
            'upstream_binding': {f: sha(root/f) for f in (
                'heldout_tuning_2026-10-06/plan.json', 'heldout_tuning_2026-10-06/verification.json',
                'eegnet_author_audit_2026-10-06/port_verification.json',
                'cache_full_context_deap/trials.csv', 'cache_full_context_seediv/trials.csv',
                'cache_common14/lineage.json', 'cache_reve_input_seediv/prepared.json')},
            'scope': 'Source-only diagnostic on fixed session1/rotation0/fold0, unexposed arm, existing groups1/2, initialization42. Every fit retains train and two source-validation panels. No test inference or new human cohorts. Earlier results were inspected; this is exploratory and cannot serve as pristine confirmation.',
            'question': 'Does changing electrode coverage, input normalization, or compact temporal input alter source learning/validation? Does subtracting the measured DEAP baseline help within these controls? These are possible explanations to test, not established causes or a new main research question.',
            'input': 'First40s,128Hz,full-trial offline Butterworth4..40Hz before prefix. Named common14 versus all32DEAP/all62SEED electrodes. Native cache is regenerated from checked source files; all common14 overlapping source prefixes must match earlier verified waveforms exactly. Corrected labels unchanged: DEAP individual binary valence excluding5; SEED assigned coarse3 includingneutral.',
            'normalization': 'source_channel: training-only channel mean/std over all training trial/time points; floor1e-6. trial_zscore: each receiving trial uses its own whole40s channel mean/std, with no other trial or label; floor1e-6. Same whole40s normalizer before either duration. This is offline per-observation normalization, not causal4s acquisition or target-population adaptation.',
            'duration': '40s: full5120-sample EEGNet. 4s:512-sample EEGNet; one deterministic uniformly chosen nonoverlapping window per sampled training trial, all10windows at inference. Trial probability=softmax(mean10windowlogits), not mean probabilities or window-level accuracy. Short and long have different head sizes and consumed training samples; matched update counts do not isolate duration or equate signal/compute exposure.',
            'baseline': 'DEAP-only local diagnostic, not a claimed author-protocol reproduction: separately filter3s measured baseline4..40Hz; average its three1s blocks in float64, round once tofloat32 and tile the128-sample waveform over the40s trial before source_channel normalization. Native/common14,4s only. Classical baseline features subtract natural-log Welch power of full filtered3s baseline from mean log power of ten4s stimulus windows. No rating enters any baseline transform. SEED has no analogous measured baseline in this release, so none is invented.',
            'training': 'Both constant LR.001/.0003,600AdamWupdates,balanced12trial draws,wd.01,clip1,dropout.5,FP32/noAMP/scheduler/earlystop/augmentation. Same ordered train draws/init across same-shape recipes; one fixed initialization deliberately limits this pilot. Source EMA learning metrics before training and every50updates. No outcome-dependent budget extension. This is finite-budget learning evidence, not convergence proof.',
            'selection': 'At200/600keep EMA and three-pass source-population BN, actual training examples only (all ten windows for4s), float64 moments. Save every candidate and restore exact EMA state/RNG before continuing trajectory. Within each recipe/panel, choose among both rates/steps/BN variants by equal .5/.5 validation familiar/unseen class-balanced log loss; meanBA breaks ties then first declared candidate. Report each panel separately; no globally selected recipe or unseen-test metrics.',
            'classical': 'For each montage: mean ten-window absolute log4-band Welch power and within-channel relative logpower. DEAP also baseline-subtracted logpower. Hann256/overlap128/FFT256 at128Hz, density integration sum*.5Hz, half-open4..8/8..14/14..31/31..40Hz,floor1e-12. StandardScaler fitted to training rows only; balanced LogisticRegression C.01/.1/1/10,max_iter4000,tol1e-6,random_state42. Same two-validation-panel selection. Raw source-video prior retained as an EEG-free diagnostic, train rows exclude their whole participant.',
            'verification': 'Per-trajectory saved candidate states replay on fresh models from checked native caches; independent sklearn/manual class metrics from complete source predictions. Reconstruct normalization and window aggregation. All saved classical candidates independently refit; selected rule/hash/participant/trial disjointness checked. Original frozen sources remain byte-identical. This does not reproduce an entire stochastic optimization trajectory or authenticate first-party DEAP signals.',
            'interpretation': 'Small reused validation panels, one initialization and fixed firstfold/session/rotation limit inference. No population confidence interval, architecture-only montage conclusion, generic absence-of-EEG statement, joint-negative-transfer claim, new method or main-question change follows. Any broader confirmation requires a new declaration; author approval and fresh manuscript archive precede a paper pivot.'}
    atomic(output/'plan.json', plan)
    atomic(output/'progress.json', {'state': 'declared', 'updated_utc': stamp(),
          'neural_completed': 0, 'neural_total': 72, 'research_question_change_approved': False})
    return output


def validate(output, root):
    plan = json.loads((output/'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if sha(REPO/name) != expected:
            raise ValueError(f'Changed frozen diagnostic source: {name}')
    for name, expected in plan['upstream_binding'].items():
        if sha(root/name) != expected:
            raise ValueError(f'Changed upstream evidence: {name}')
    for name, expected in plan['panel_bindings'].items():
        folder = output/'panels'/name
        if sha(folder/'panel.json') != expected:
            raise ValueError('Changed panel')
        record = json.loads((folder/'panel.json').read_text())
        if sha(folder/'trials.csv') != record['trials_sha256']:
            raise ValueError('Changed source table')
    return plan


def prepare(output, root, data_root):
    validate(output, root)
    sources = roots(data_root)
    lineage = json.loads((root/'cache_common14/lineage.json').read_text())
    common = {r['trial_id']: r for r in lineage}
    reference = np.load(root/'cache_reve_input_seediv/prefix128.npy', mmap_mode='r', allow_pickle=False)
    reference_table = pd.read_csv(root/'cache_reve_input_seediv/trials.csv')
    reference_positions = dict(zip(reference_table.trial_id, reference_table.index))
    info = json.loads((root/'cache_reve_input_seediv/prepared.json').read_text())
    if sha(root/'cache_reve_input_seediv/prefix128.npy') != info['prefix128_sha256']:
        raise ValueError('Changed previously verified SEED prefix')
    for dataset in ('DEAP', 'SEEDIV'):
        cache = output/'inputs'/dataset.lower()
        if (cache/'prepared.json').exists():
            check_cache(cache)
            continue
        if cache.exists():
            raise FileExistsError('Incomplete cache preserved; use explicit recovery before preparing')
        panels = [pd.read_csv(output/'panels'/f'{dataset.lower()}_g{g}'/'trials.csv') for g in (1, 2)]
        table = pd.concat(panels).drop_duplicates('trial_id').sort_values('trial_id').reset_index(drop=True)
        channels = DEAP_CHANNELS if dataset == 'DEAP' else seed_channels(sources[dataset])
        picks = [channels.index(c) for c in COMMON_CHANNELS]
        cache.mkdir(parents=True)
        wave = np.lib.format.open_memmap(cache/'native.npy', mode='w+', dtype=np.float32,
                                        shape=(len(table), len(channels), 5120))
        baseline = None
        if dataset == 'DEAP':
            baseline = np.lib.format.open_memmap(cache/'baseline.npy', mode='w+', dtype=np.float32,
                                                shape=(len(table), len(channels), 384))
        checked = {}
        if dataset == 'DEAP':
            ratings, rating_info = deap_reference(sources[dataset])
            rating_info = dict(rating_info, file=Path(rating_info['file']).name)
            for subject, rows in table.groupby('subject_id', sort=True):
                first = common[rows.iloc[0].trial_id]
                path = Path(first['source'])
                if sha(path) != first['source_sha256']:
                    raise ValueError('Changed DEAP source')
                checked[path.name] = first['source_sha256']
                with path.open('rb') as stream:
                    values = pickle.load(stream, encoding='latin1')
                if values['data'].shape != (40, 40, 8064):
                    raise ValueError('Invalid DEAP recordings')
                original = ratings[int(subject.rsplit('S', 1)[1])]
                verify_deap_labels(values['labels'], original)
                for i, row in rows.iterrows():
                    trial = int(row.trial_id.rsplit('T', 1)[1])-1
                    if not np.isclose(row.original_label, original[trial, 0], atol=1e-12, rtol=0):
                        raise ValueError('Recovered source rating changed')
                    raw = values['data'][trial, :32]
                    filtered, _ = preprocess_signal(raw[:, 384:], 128, channels, channels)
                    wave[i] = filtered[:, :5120]
                    baseline[i], _ = preprocess_signal(raw[:, :384], 128, channels, channels)
                    old = root/'cache_common14'/common[row.trial_id]['cache_file']
                    if sha(old) != common[row.trial_id]['cache_sha256']:
                        raise ValueError('Changed common cache')
                    expected = np.load(old, allow_pickle=False)[:10].transpose(1, 0, 2).reshape(14, 5120)
                    np.testing.assert_array_equal(wave[i, picks], expected)
                del values
        else:
            rating_info = {'ReadMe_sha256': sha(sources[dataset]/'ReadMe.txt'),
                           'channel_order_sha256': sha(sources[dataset]/'Channel Order.xlsx')}
            for filename, rows in table.groupby('source_file', sort=True):
                path = sources[dataset]/filename
                expected_sha = rows.iloc[0].source_sha256
                if not rows.source_sha256.eq(expected_sha).all() or sha(path) != expected_sha:
                    raise ValueError('Changed SEED source')
                checked[filename] = expected_sha
                values = loadmat(path, variable_names=rows.source_key.tolist())
                for i, row in rows.iterrows():
                    trial = int(row.trial_id.rsplit('T', 1)[1])-1
                    if row.original_label != SEED_LABELS[int(row.session)][trial]:
                        raise ValueError('Changed native SEED label')
                    filtered, _ = preprocess_signal(values[row.source_key], 200, channels, channels)
                    wave[i] = filtered[:, :5120]
                    np.testing.assert_array_equal(wave[i, picks], reference[reference_positions[row.trial_id]])
                del values
        wave.flush()
        if baseline is not None:
            baseline.flush()
        table.to_csv(cache/'trials.csv', index=False)
        files = ('native.npy', 'trials.csv', 'baseline.npy') if baseline is not None else ('native.npy', 'trials.csv')
        atomic(cache/'prepared.json', {'dataset': dataset, 'channels': channels,
              'shape': list(wave.shape), 'source_files': checked, 'metadata': rating_info,
              'common_prefixes_exact': len(table), 'file_sha256': {f: sha(cache/f) for f in files},
              'plan_sha256': sha(output/'plan.json'), 'first_party_DEAP_recording_authentication': False})
        del wave, baseline
        print(json.dumps({'prepared': dataset, 'source_trials': len(table)}), flush=True)


def check_cache(cache):
    record = json.loads((cache/'prepared.json').read_text())
    for name, expected in record['file_sha256'].items():
        if sha(cache/name) != expected:
            raise ValueError('Changed native waveform cache')
    return record


def load_panel(output, dataset, group):
    folder = output/'panels'/f'{dataset.lower()}_g{group}'
    table = pd.read_csv(folder/'trials.csv')
    record = json.loads((folder/'panel.json').read_text())
    positions = {trial: i for i, trial in enumerate(table.trial_id)}
    idx = {r: np.array([positions[t] for t in record['roles'][r]], dtype=int) for r in ROLES}
    people = {r: set(table.iloc[idx[r]].subject_id) for r in ROLES}
    if people['train'] & (people['validation_unseen'] | people['validation_familiar']):
        raise ValueError('Participant leakage')
    if set(table.subject_id) & set(record['excluded_test_participants']):
        raise ValueError('Outer-test participant accessed')
    cache = output/'inputs'/dataset.lower()
    info = json.loads((cache/'prepared.json').read_text())
    if info['plan_sha256'] != sha(output/'plan.json'):
        raise ValueError('Wrong input plan')
    all_rows = pd.read_csv(cache/'trials.csv')
    positions = {trial: i for i, trial in enumerate(all_rows.trial_id)}
    rows = [positions[t] for t in table.trial_id]
    raw = np.load(cache/'native.npy', mmap_mode='r', allow_pickle=False)[rows]
    baseline = np.load(cache/'baseline.npy', mmap_mode='r', allow_pickle=False)[rows] if dataset == 'DEAP' else None
    return table, idx, raw, baseline, info


def transform(raw, baseline, channels, recipe, train):
    picks = [channels.index(c) for c in COMMON_CHANNELS] if recipe['montage'] == 'common14' else list(range(len(channels)))
    values = np.array(raw[:, picks], dtype=np.float32, copy=True)
    if recipe['baseline']:
        if baseline is None:
            raise ValueError('Measured baseline is required')
        pattern = baseline[:, picks].reshape(len(values), len(picks), 3, 128).mean(2, dtype=np.float64).astype(np.float32)
        values -= np.tile(pattern, (1, 1, 40))
    if recipe['normalization'] == 'source_channel':
        source = values[train].astype(np.float64)
        mean = source.mean((0, 2), keepdims=True)
        scale = np.maximum(source.std((0, 2), keepdims=True), 1e-6)
    elif recipe['normalization'] == 'trial_zscore':
        mean = values.mean(-1, keepdims=True, dtype=np.float64)
        scale = np.maximum(values.std(-1, keepdims=True, dtype=np.float64), 1e-6)
    else:
        raise ValueError('Unknown normalization')
    values = ((values-mean)/scale).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite model input')
    return values, mean, scale


def windows(values):
    if values.shape[-1] != 5120:
        raise ValueError('Expected first40s at128Hz')
    return values.reshape(len(values), values.shape[1], 10, 512).transpose(0, 2, 1, 3)


@torch.no_grad()
def predict(model, data, rows, short):
    model.eval()
    values = data[rows]
    if short:
        examples = values.flatten(0, 1)
        logits = torch.cat([model(examples[s:s+12]) for s in range(0, len(examples), 12)])
        logits = logits.reshape(len(rows), 10, -1).mean(1)
    else:
        logits = torch.cat([model(values[s:s+12]) for s in range(0, len(values), 12)])
    return torch.softmax(logits, -1).cpu().numpy()


def metric(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1., atol=1e-6):
        raise ValueError('Invalid probabilities')
    predicted = p.argmax(1)
    correct = predicted == labels
    loss = -np.log(np.maximum(p[np.arange(len(p)), labels], 1e-12))
    classes = range(p.shape[1])
    if set(labels) != set(classes):
        raise ValueError('Missing class in panel')
    return {'n': len(labels), 'balanced_accuracy': float(np.mean([correct[labels == c].mean() for c in classes])),
            'balanced_log_loss': float(np.mean([loss[labels == c].mean() for c in classes]))}


def selection_key(candidate):
    values = [candidate['metrics'][r] for r in ROLES[1:]]
    return (float(np.mean([v['balanced_log_loss'] for v in values])),
            -float(np.mean([v['balanced_accuracy'] for v in values])))


def save_predictions(folder, label, table, idx, predictions):
    paths = []
    for role in ROLES:
        rows = table.iloc[idx[role]][['trial_id', 'subject_id', 'material_key', 'label', 'original_label']].copy()
        for c in range(predictions[role].shape[1]):
            rows[f'p{c}'] = predictions[role][:, c]
        name = f'{label}_{role}.csv'
        rows.to_csv(folder/name, index=False)
        paths.append(name)
    return paths


def fit_neural(job, output):
    folder = output/'fits'/case_id(job)
    if (folder/'verification.json').exists():
        result = json.loads((folder/'record.json').read_text())
        for name, expected in result['artifact_sha256'].items():
            if sha(folder/name) != expected:
                raise ValueError('Changed resumed neural artifact')
        return result
    if folder.exists():
        raise FileExistsError('Incomplete trajectory preserved; explicit recovery required')
    folder.mkdir(parents=True)
    table, idx, raw, baseline, info = load_panel(output, job['dataset'], job['group'])
    values, mean, scale = transform(raw, baseline, info['channels'], job['recipe'], idx['train'])
    short = job['recipe']['seconds'] == 4
    data = torch.from_numpy(np.ascontiguousarray(windows(values) if short else values)).cuda()
    target = torch.tensor(table.label.to_numpy(dtype=np.int64), device='cuda')
    classes = 2 if job['dataset'] == 'DEAP' else 3
    sampled, draw_hash = draw_stream(table, idx['train'], job['initialization'], classes, STEPS[-1])
    window_draws = np.random.default_rng(job['initialization']+2000000).integers(0, 10, sampled.shape)
    seed_everything(job['initialization'])
    model = EEGNetControl(classes, channels=values.shape[1], samples=job['recipe']['seconds']*128).cuda()
    initial = state_digest(model.state_dict())
    seed_everything(job['initialization']+1000000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job['lr'], weight_decay=.01)
    np.savez(folder/'normalizer.npz', mean=mean, scale=scale)
    artifacts = ['normalizer.npz']
    history, candidates = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(STEPS[-1]+1):
        if step:
            batch = sampled[step-1]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            inputs = data[batch, window_draws[step-1]] if short else data[batch]
            loss = nn.functional.cross_entropy(model(inputs), target[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            model.constrain()
        if step % 50:
            continue
        p = {r: predict(model, data, idx[r], short) for r in ROLES}
        metrics = {r: metric(table.iloc[idx[r]].label, p[r]) for r in ROLES}
        paths = save_predictions(folder, f'curve{step}', table, idx, p)
        artifacts.extend(paths)
        history.append({'step': step, 'metrics': metrics,
                        'training_minibatch_loss': float(loss.detach()) if step else None})
        if step not in STEPS:
            continue
        ema = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        for normalization in ('ema', 'source_population'):
            label = f'step{step}_{normalization}'
            if normalization == 'source_population':
                training = data[idx['train']]
                calibrate(model, training.flatten(0, 1) if short else training)
                p = {r: predict(model, data, idx[r], short) for r in ROLES}
                metrics = {r: metric(table.iloc[idx[r]].label, p[r]) for r in ROLES}
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state': state}, folder/f'{label}.pt')
            paths = save_predictions(folder, label, table, idx, p)
            artifacts.extend([f'{label}.pt', *paths])
            candidates.append({'id': label, 'step': step, 'normalization': normalization,
                               'metrics': metrics, 'state_digest': state_digest(state)})
        model.load_state_dict(ema)
        if not torch.equal(torch.get_rng_state(), rng[0]) or not torch.equal(torch.cuda.get_rng_state(), rng[1]):
            raise ValueError('Calibration altered training RNG')
    atomic(folder/'history.json', history)
    artifacts.append('history.json')
    result = {'job': job, 'plan_sha256': sha(output/'plan.json'), 'initial_state_digest': initial,
              'training_draw_digest': draw_hash, 'window_draw_digest': digest(window_draws.tolist()),
              'parameters': sum(p.numel() for p in model.parameters()), 'candidates': candidates,
              'selected': min(candidates, key=selection_key)['id'],
              'seconds': time.perf_counter()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'input_cache_sha256': sha(output/'inputs'/job['dataset'].lower()/'prepared.json'),
              'artifact_sha256': {name: sha(folder/name) for name in artifacts}}
    atomic(folder/'record.json', result)
    del data, target, model, optimizer
    torch.cuda.empty_cache()
    return result


def logpower(values):
    frequency, density = welch(values, fs=128, window='hann', nperseg=256,
                               noverlap=128, nfft=256, detrend='constant', scaling='density', axis=-1)
    power = np.stack([density[..., (frequency >= a) & (frequency < b)].sum(-1)*.5 for a,b in BANDS], -1)
    return np.log(np.maximum(power, 1e-12))


def classical_features(raw, baseline, channels, montage, representation):
    picks = [channels.index(c) for c in COMMON_CHANNELS] if montage == 'common14' else list(range(len(channels)))
    result = []
    for i, row in enumerate(raw):
        blocks = row[picks].reshape(len(picks), 10, 512).transpose(1, 0, 2)
        values = logpower(blocks)
        if representation == 'relative':
            values -= logsumexp(values, axis=-1, keepdims=True)
        elif representation == 'baseline':
            if baseline is None:
                raise ValueError('Measured baseline required')
            values -= logpower(baseline[i, picks])[None]
        elif representation != 'absolute':
            raise ValueError('Unknown representation')
        result.append(values.mean(0).reshape(-1))
    return np.asarray(result, dtype=np.float64)


def fit_classical(dataset, group, output):
    from .full_context_controls_v2 import prior
    table, idx, raw, baseline, info = load_panel(output, dataset, group)
    classes = 2 if dataset == 'DEAP' else 3
    folder = output/'classical'/f'{dataset.lower()}_g{group}'
    if (folder/'verification.json').exists():
        result = json.loads((folder/'record.json').read_text())
        for name, expected in result['artifact_sha256'].items():
            if sha(folder/name) != expected:
                raise ValueError('Changed classical artifact')
        return result
    if folder.exists():
        raise FileExistsError('Incomplete classical panel preserved')
    folder.mkdir(parents=True)
    records, artifacts = [], []
    for montage in ('common14', 'native'):
        for representation in (('absolute', 'relative', 'baseline') if dataset == 'DEAP' else ('absolute', 'relative')):
            x = classical_features(raw, baseline, info['channels'], montage, representation)
            scaler = StandardScaler().fit(x[idx['train']])
            z = scaler.transform(x)
            candidates = []
            for c in (.01, .1, 1., 10.):
                model = LogisticRegression(C=c, class_weight='balanced', max_iter=4000, tol=1e-6, random_state=42)
                model.fit(z[idx['train']], table.iloc[idx['train']].label)
                if np.any(model.n_iter_ >= 4000):
                    raise ValueError('Classical optimization did not converge')
                np.testing.assert_array_equal(model.classes_, np.arange(classes))
                label = f'{montage}_{representation}_C{c}'
                p = {r: model.predict_proba(z[idx[r]]) for r in ROLES}
                metrics = {r: metric(table.iloc[idx[r]].label, p[r]) for r in ROLES}
                paths = save_predictions(folder, label, table, idx, p)
                np.savez(folder/f'{label}.npz', mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_, intercept=model.intercept_)
                artifacts.extend([*paths, f'{label}.npz'])
                candidates.append({'id': label, 'C': c, 'metrics': metrics})
            records.append({'montage': montage, 'representation': representation,
                            'candidates': candidates, 'selected': min(candidates, key=selection_key)['id']})
    p = {r: prior(table, idx['train'], idx[r], classes, crossfit=r == 'train') for r in ROLES}
    artifacts.extend(save_predictions(folder, 'context_prior', table, idx, p))
    result = {'dataset': dataset, 'group': group, 'plan_sha256': sha(output/'plan.json'),
              'recipes': records, 'context': {r: metric(table.iloc[idx[r]].label, p[r]) for r in ROLES},
              'artifact_sha256': {name: sha(folder/name) for name in artifacts}}
    atomic(folder/'record.json', result)
    return result
