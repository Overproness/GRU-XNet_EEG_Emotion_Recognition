"""Native-montage, source-only frozen CBraMod diagnostic; no outer-test inference."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import types
import warnings
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample, welch
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from .data import roots, deap_reference, verify_deap_labels, SEED_LABELS
from .material_controls import state_digest
from .preprocessing_diagnostic_v2 import (SOURCES as OLD_SOURCES, ROLES, sha,
    atomic, stamp, metric, selection_key)
from .train import seed_everything

REPO = Path(__file__).resolve().parents[1]
STUDY = 'cbramod_source_probe_2026-10-09'
PARENT = 'preprocessing_diagnostic_v2_2026-10-09'
ASSETS = 'cbramod_audit_2026-10-09'
WEIGHT_SHA = '0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178'
C_VALUES = (.01, .1, 1., 10.)
MODELS = ('pretrained_average', 'pretrained_flatten', 'random42_average',
          'random42_flatten', 'band_absolute', 'band_relative')
SOURCES = tuple(dict.fromkeys((*OLD_SOURCES, 'gruxnet/cbramod_probe.py',
    'scripts/cbramod_source_probe.py', 'scripts/audit_cbramod_source_probe.py',
    'scripts/prepare_cbramod_assets.py', 'tests/test_cbramod_probe.py')))


def assets_check(folder):
    manifest = json.loads((folder/'download_manifest.json').read_text())
    if manifest['github_revision'] != 'b9e961003214326972c567eff390e75b0287e32a' or \
            manifest['checkpoint_revision'] != '500543c7e30bda1b22bfd51a49301b238dee21fd':
        raise ValueError('Wrong author/checkpoint revisions')
    for item in manifest['files']:
        if sha(folder/item['file']) != item['sha256']:
            raise ValueError('Changed downloaded asset')
    if sha(folder/'checkpoint/pretrained_weights.pth') != WEIGHT_SHA:
        raise ValueError('Unexpected checkpoint bytes')
    review = json.loads((folder/'code_review.json').read_text())
    for name, checksum in review['imported_sha256'].items():
        if sha(folder/'author'/name) != checksum:
            raise ValueError('Changed reviewed inference code')
    return manifest


def local_model(folder, pretrained):
    assets_check(folder)
    # The unchanged author source has an absolute models import. Provide only a
    # temporary local package, then restore the import table; no remote code.
    names = ('models', 'models.criss_cross_transformer', 'models.cbramod')
    saved = {name: sys.modules.get(name) for name in names}
    package = types.ModuleType('models')
    package.__path__ = [str(folder/'author/models')]
    sys.modules['models'] = package
    try:
        for name in names[1:]:
            spec = importlib.util.spec_from_file_location(name, folder/'author/models'/f'{name.rsplit(".", 1)[1]}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        seed_everything(42)
        model = sys.modules['models.cbramod'].CBraMod()
        if pretrained:
            weights = torch.load(folder/'checkpoint/pretrained_weights.pth',
                                 map_location='cpu', weights_only=True)
            if not isinstance(weights, dict) or not all(isinstance(v, torch.Tensor) for v in weights.values()):
                raise ValueError('Checkpoint must be a tensor-only state dictionary')
            model.load_state_dict(weights, strict=True)
        model.proj_out = nn.Identity()
        return model.eval().requires_grad_(False)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def pilot(root):
    folder = root/ASSETS
    model = local_model(folder, True).to('cuda')
    before = state_digest(model.state_dict())
    rng = np.random.default_rng(20261009)
    records = []
    with torch.inference_mode():
        for channels in (32, 62):
            wave = torch.tensor(rng.normal(0, .1, (4, channels, 10, 200)).astype(np.float32), device='cuda')
            reference = model(wave[:1]).cpu().numpy()
            for batch in (1, 2, 4):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                actual = model(wave[:batch]).cpu().numpy()
                discrepancy = float(np.max(np.abs(reference-actual[:1])))
                if actual.shape != (batch, channels, 10, 200) or discrepancy > 2e-5:
                    raise ValueError('Pilot output or batch-invariance discrepancy')
                records.append({'channels': channels, 'batch_windows': batch,
                    'seconds': time.perf_counter()-start, 'batch_max_abs': discrepancy,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    if state_digest(model.state_dict()) != before:
        raise ValueError('Frozen encoder changed')
    result = {'synthetic_only': True, 'created_utc': stamp(), 'selected_batch_windows': 4,
              'encoder_parameters': sum(p.numel() for p in model.parameters()),
              'torch_version': torch.__version__, 'mne_version': mne.__version__,
              'device': torch.cuda.get_device_name(), 'records': records,
              'scope': 'FP32 frozen inference only; not fine-tuning memory or clinical performance'}
    atomic(folder/'feasibility.json', result)
    return result


def declare(root):
    output = root/STUDY
    if output.exists():
        raise FileExistsError('Preserve the frozen declaration')
    assets_check(root/ASSETS)
    parent = root/PARENT
    previous = json.loads((parent/'plan.json').read_text())
    for name, checksum in previous['source_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError('Changed predecessor source')
    if not json.loads((parent/'verification.json').read_text())['complete']:
        raise ValueError('Verified source-panel predecessor required')
    output.mkdir(parents=True)
    bindings = {f'{ASSETS}/{name}': sha(root/ASSETS/name) for name in
                ('download_manifest.json', 'code_review.json', 'feasibility.json', 'environment_preflight.json')}
    bindings[f'{PARENT}/plan.json'] = sha(parent/'plan.json')
    for dataset in ('deap', 'seediv'):
        for group in (1, 2):
            for name in ('trials.csv', 'panel.json'):
                key = f'{PARENT}/panels/{dataset}_g{group}/{name}'
                bindings[key] = sha(root/key)
        for name in ('prepared.json', 'trials.csv'):
            key = f'{PARENT}/inputs/{dataset}/{name}'
            bindings[key] = sha(root/key)
    bindings['cache_common14/lineage.json'] = sha(root/'cache_common14/lineage.json')
    plan = {'created_utc': stamp(), 'development_only': True,
        'research_question_change_approved': False, 'models': MODELS, 'C_values': C_VALUES,
        'jobs': [{'dataset': d, 'group': g, 'model': model}
                 for d in ('DEAP', 'SEEDIV') for g in (1, 2) for model in MODELS],
        'counts': {'selected_heads': 24, 'candidate_fits': 96, 'independent_refits': 96},
        'source_sha256': {f: sha(REPO/f) for f in SOURCES}, 'upstream_sha256': bindings,
        'scope': 'Existing session1/rotation0/fold0 unexposed source-only panels, groups1/2. No outer-test inference or new cohort; small reused validation panels, one random-encoder initialization. No context fusion or joint training in this diagnostic.',
        'representation': 'Unchanged pinned official CBraMod encoder, tensor-only strict checkpoint load, remove reconstruction projection exactly as author classifier. Frozen eval mode, no task gradients, FP32/noAMP. Native32/62 channel order retained; channel order matters for conditional convolutions. Published pretraining is TUEG; checkpoint card is minimal, training-membership exclusion is not independently certified.',
        'input': 'Checked original mirror files, first40s stimulus. DEAP discard384baseline samples, Fourier-resample entire60s32channel stimulus128->200Hz then prefix8000samples, matching FACED Fourier resampling operator; retain already preprocessed bandwidth, no extra filter. Resampling cannot restore information removed by acquisition/release preprocessing. SEED original200Hz62channel full trials: MNE1.10 default zero-phase FIR .3..75Hz, then prefix8000samples. Continuous CNT recordings are unavailable so independently filtered trials differ from author SEED-V continuous-file preprocessing. Do not add TUEG-only60Hznotch, artifact rejection, baseline subtraction, zscore or re-reference.',
        'units': 'Interpret release numerical EEG values as microvolts and divide by100, as author FACED/SEED-V loaders. This assumption is explicit: the local SEED-IV ReadMe does not certify physical units/calibration, and full first-party DEAP recording authentication remains outstanding. Amplitude statistics are reported without outcome-dependent rescaling or dropping recordings.',
        'readout': 'Each40s observation has4disjoint10s windows, each channel has10one-second200sample patches. Two fixed author-supported readouts: average all channel/patch tokens ->200, or flatten channel/patch/token ->64000DEAP/124000SEED. Average four window representations in float64, then castfloat32. Flatten keeps within-window time positions but averages offsets across four windows. Both retained; not the author full supervised fine-tuning or original multi-layer classifier.',
        'band_controls': 'Same new200Hz native input before /100. Four half-open4..8/8..14/14..31/31..40Hz bands, Welch Hann400/overlap200/FFT400,density*.5Hz over ten4s windows. Mean logabsolute power or mean within-window/channel logrelative power, floor1e-12. No baseline subtraction/context. This controls the input and source panels; different feature dimension/computation limits architecture-only inference.',
        'fit_selection': 'Training-only StandardScaler and balanced LogisticRegression lbfgs C.01/.1/1/10,maxiter4000,tol1e-6,random_state42; no target adaptation, window labels, augmentation or earlystop. Equal.5/.5familiar/unseen validation class-balanced logloss, then meanBA, stablefirsttie. Every train/validation candidate retained; no globally winning readout/encoder selected. Convergence warning stops the study and preserves failure.',
        'verification': 'Source/asset bindings and participant/trial/material exclusions checked. All96candidatefreshscaler/estimator refits, coefficient/scaler equality, independent probability link and weighted metrics. First/last2source-union trials per dataset reprocessed from raw; pretrained/random encodings replayed using batch1 instead of4 for those trials. This does not reprocess every original sample or independently retrain the external checkpoint.',
        'decision': 'Report both groups/allreadouts/models,train learning/sourcevalidation. Frozen probe failure does not disprove full fine-tuning. A consistent pretrained loss advantage over random and band controls justifies a new source-only training diagnostic, not yet a large held-out grid or novelty. Do not adapt the research question without explicit author approval and a fresh manuscript archive.'}
    atomic(output/'plan.json', plan)
    atomic(output/'progress.json', {'state': 'declared', 'heads_completed': 0, 'heads_total': 24,
           'updated_utc': stamp(), 'research_question_change_approved': False})
    return output


def validate(root, output):
    plan = json.loads((output/'plan.json').read_text())
    for name, checksum in plan['source_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError(f'Changed frozen source {name}')
    for name, checksum in plan['upstream_sha256'].items():
        if sha(root/name) != checksum:
            raise ValueError(f'Changed frozen upstream {name}')
    assets_check(root/ASSETS)
    return plan


def preprocess(raw, dataset):
    if raw.ndim != 2 or raw.shape[1] < (7680 if dataset == 'DEAP' else 8000):
        raise ValueError('Wrong native stimulus duration')
    if dataset == 'DEAP':
        # Data already contain release preprocessing. No new bandpass is imposed.
        x = resample(np.asarray(raw, dtype=np.float64), raw.shape[1]*25//16, axis=-1)
    elif dataset == 'SEEDIV':
        x = mne.filter.filter_data(np.asarray(raw, dtype=np.float64), 200., .3, 75.,
                                  n_jobs=1, verbose=False)
    else:
        raise ValueError('Unsupported dataset')
    x = x[:, :8000].astype(np.float32)
    if x.shape[1] != 8000 or not np.isfinite(x).all():
        raise ValueError('Invalid prefix')
    return x


def prepare(root, output, data_root):
    validate(root, output)
    locations = roots(data_root)
    lineage = {r['trial_id']: r for r in json.loads((root/'cache_common14/lineage.json').read_text())}
    for dataset in ('DEAP', 'SEEDIV'):
        cache = output/'inputs'/dataset.lower()
        if (cache/'prepared.json').exists():
            check_cache(cache)
            continue
        if cache.exists():
            raise FileExistsError('Preserve incomplete preparation')
        cache.mkdir(parents=True)
        old = root/PARENT/'inputs'/dataset.lower()
        table = pd.read_csv(old/'trials.csv')
        channels = json.loads((old/'prepared.json').read_text())['channels']
        wave = np.lib.format.open_memmap(cache/'native200.npy', mode='w+', dtype=np.float32,
                                        shape=(len(table), len(channels), 8000))
        checked = {}; amplitudes = []
        if dataset == 'DEAP':
            ratings, _ = deap_reference(locations[dataset])
            for subject, rows in table.groupby('subject_id', sort=True):
                source = lineage[rows.iloc[0].trial_id]
                path = Path(source['source'])
                if sha(path) != source['source_sha256']:
                    raise ValueError('Changed raw DEAP source')
                checked[path.name] = source['source_sha256']
                with path.open('rb') as stream:
                    pack = pickle.load(stream, encoding='latin1')
                original = ratings[int(subject.rsplit('S', 1)[1])]
                verify_deap_labels(pack['labels'], original)
                if pack['data'].shape != (40, 40, 8064):
                    raise ValueError('Wrong DEAP release shape')
                for i, row in rows.iterrows():
                    trial = int(row.trial_id.rsplit('T', 1)[1])-1
                    if not np.isclose(row.original_label, original[trial, 0], atol=1e-12, rtol=0):
                        raise ValueError('Changed DEAP rating')
                    wave[i] = preprocess(pack['data'][trial, :32, 384:], dataset)
        else:
            for name, rows in table.groupby('source_file', sort=True):
                path = locations[dataset]/name
                expected = rows.iloc[0].source_sha256
                if not rows.source_sha256.eq(expected).all() or sha(path) != expected:
                    raise ValueError('Changed SEED raw source')
                checked[name] = expected
                pack = loadmat(path, variable_names=rows.source_key.tolist())
                for i, row in rows.iterrows():
                    trial = int(row.trial_id.rsplit('T', 1)[1])-1
                    if row.original_label != SEED_LABELS[int(row.session)][trial]:
                        raise ValueError('Changed SEED label')
                    wave[i] = preprocess(pack[row.source_key], dataset)
        for i in range(len(table)):
            x = wave[i].astype(np.float64)
            amplitudes.append({'trial_id': table.iloc[i].trial_id, 'rms_numeric': float(np.sqrt(np.mean(x*x))),
                               'max_abs_numeric': float(np.max(np.abs(x))),
                               'fraction_abs_over100': float(np.mean(np.abs(x) > 100))})
        wave.flush()
        table.to_csv(cache/'trials.csv', index=False)
        pd.DataFrame(amplitudes).to_csv(cache/'amplitude_statistics.csv', index=False)
        atomic(cache/'prepared.json', {'dataset': dataset, 'channels': channels, 'shape': list(wave.shape),
            'source_files': checked, 'plan_sha256': sha(output/'plan.json'), 'mne_version': mne.__version__,
            'unit_calibration_independently_authenticated': False,
            'file_sha256': {n: sha(cache/n) for n in ('native200.npy', 'trials.csv', 'amplitude_statistics.csv')}})
        del wave
        print(json.dumps({'prepared': dataset, 'source_union_trials': len(table)}), flush=True)


def check_cache(cache):
    info = json.loads((cache/'prepared.json').read_text())
    for name, checksum in info['file_sha256'].items():
        if sha(cache/name) != checksum:
            raise ValueError('Changed native input cache')
    return info


def patches(trials):
    n, channels, samples = trials.shape
    if samples != 8000:
        raise ValueError('Exactly40seconds required')
    return (trials/np.float32(100.)).reshape(n, channels, 4, 10, 200).transpose(0, 2, 1, 3, 4).reshape(n*4, channels, 10, 200)


def pool(tokens):
    # Input is windows in four-window trial order. Mean in float64, one cast.
    n, c, s, d = tokens.shape
    if n % 4 or s != 10 or d != 200:
        raise ValueError('Invalid author token shape')
    grouped = tokens.reshape(n//4, 4, c, s, d)
    return {'average': grouped.mean((1, 2, 3), dtype=np.float64).astype(np.float32),
            'flatten': grouped.mean(1, dtype=np.float64).reshape(n//4, c*s*d).astype(np.float32)}


def encode(model, raw, batch=4):
    x = patches(raw)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            y = model(torch.tensor(x[start:start+batch], device='cuda')).cpu().numpy()
            if y.shape != (min(batch, len(x)-start), x.shape[1], 10, 200) or not np.isfinite(y).all():
                raise ValueError('Invalid frozen features')
            chunks.append(y)
    return pool(np.concatenate(chunks))


def band_features(raw):
    n, c, _ = raw.shape
    x = raw.reshape(n, c, 10, 800).transpose(0, 2, 1, 3).astype(np.float64)
    freq, density = welch(x, fs=200, window='hann', nperseg=400, noverlap=200,
                          nfft=400, detrend='constant', scaling='density', axis=-1)
    powers = np.stack([density[..., (freq >= lo) & (freq < hi)].sum(-1)*.5
                      for lo, hi in ((4, 8), (8, 14), (14, 31), (31, 40))], -1)
    logs = np.log(np.maximum(powers, 1e-12))
    return {'band_absolute': logs.mean(1).reshape(n, c*4).astype(np.float32),
            'band_relative': (logs-logsumexp(logs, axis=-1, keepdims=True)).mean(1).reshape(n, c*4).astype(np.float32)}


def extract(root, output):
    validate(root, output)
    for dataset in ('DEAP', 'SEEDIV'):
        cache = output/'inputs'/dataset.lower()
        info = check_cache(cache)
        if (cache/'features.json').exists():
            check_features(cache)
            continue
        raw = np.load(cache/'native200.npy', mmap_mode='r', allow_pickle=False)
        for name, x in band_features(raw).items():
            np.save(cache/f'{name}.npy', x, allow_pickle=False)
        states = {}
        for name, pretrained in (('pretrained', True), ('random42', False)):
            model = local_model(root/ASSETS, pretrained).to('cuda')
            before = state_digest(model.state_dict())
            average = np.lib.format.open_memmap(cache/f'{name}_average.npy', mode='w+',
                        dtype=np.float32, shape=(len(raw), 200))
            flatten = np.lib.format.open_memmap(cache/f'{name}_flatten.npy', mode='w+',
                        dtype=np.float32, shape=(len(raw), raw.shape[1]*10*200))
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for i in range(len(raw)):
                values = encode(model, np.asarray(raw[i:i+1]), batch=4)
                average[i] = values['average'][0]; flatten[i] = values['flatten'][0]
                if (i+1) % 100 == 0:
                    print(json.dumps({'extracting': dataset, 'encoder': name, 'completed': i+1, 'total': len(raw)}), flush=True)
            average.flush(); flatten.flush()
            if state_digest(model.state_dict()) != before:
                raise ValueError('Frozen encoder changed during extraction')
            states[name] = {'state_sha256': before, 'seconds': time.perf_counter()-start,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            del model, average, flatten
        atomic(cache/'features.json', {'input_prepared_sha256': sha(cache/'prepared.json'),
            'plan_sha256': sha(output/'plan.json'), 'encoder_states': states,
            'file_sha256': {f'{name}.npy': sha(cache/f'{name}.npy') for name in MODELS}})
        del raw


def check_features(cache):
    info = json.loads((cache/'features.json').read_text())
    if sha(cache/'prepared.json') != info['input_prepared_sha256']:
        raise ValueError('Changed feature input binding')
    for name, checksum in info['file_sha256'].items():
        if sha(cache/name) != checksum:
            raise ValueError('Changed features')
    return info


def load_panel(root, output, dataset, group, model):
    folder = root/PARENT/'panels'/f'{dataset.lower()}_g{group}'
    table = pd.read_csv(folder/'trials.csv')
    panel = json.loads((folder/'panel.json').read_text())
    by_id = dict(zip(table.trial_id, table.index))
    idx = {role: np.array([by_id[t] for t in panel['roles'][role]], dtype=int) for role in ROLES}
    if set(table.subject_id) & set(panel['excluded_test_participants']):
        raise ValueError('Outer-test participant in panel')
    if set(table.iloc[idx['train']].subject_id) & set(table.iloc[np.r_[idx[ROLES[1]], idx[ROLES[2]]]].subject_id):
        raise ValueError('Source/validation participant overlap')
    if set(table.iloc[idx[ROLES[1]]].material_key) & set(table.iloc[idx['train']].material_key):
        raise ValueError('Unseen validation material overlap')
    cache = output/'inputs'/dataset.lower()
    union = pd.read_csv(cache/'trials.csv')
    positions = dict(zip(union.trial_id, union.index))
    x = np.load(cache/f'{model}.npy', mmap_mode='r', allow_pickle=False)[[positions[t] for t in table.trial_id]]
    return table, idx, x


def fit_estimator(x, table, idx, C):
    scaler = StandardScaler().fit(x[idx['train']])
    z = scaler.transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator = LogisticRegression(C=C, solver='lbfgs', class_weight='balanced',
                                      max_iter=4000, tol=1e-6, random_state=42)
        estimator.fit(z[idx['train']], table.label.to_numpy()[idx['train']])
    p = {role: estimator.predict_proba(z[rows]) for role, rows in idx.items()}
    measures = {role: metric(table.label.to_numpy()[idx[role]], values) for role, values in p.items()}
    return scaler, estimator, p, measures


def fit_job(root, output, job):
    identifier = f'{job["dataset"].lower()}_g{job["group"]}_{job["model"]}'
    folder = output/'fits'/identifier
    if (folder/'verification.json').exists():
        return
    if folder.exists():
        raise FileExistsError('Preserve unverified partial fit')
    folder.mkdir(parents=True)
    table, idx, x = load_panel(root, output, **job)
    candidates = []
    for k, C in enumerate(C_VALUES):
        scaler, model, p, measures = fit_estimator(x, table, idx, C)
        candidate = {'id': k, 'C': C, 'metrics': measures, 'iterations': model.n_iter_.tolist()}
        np.savez(folder/f'candidate{k}.npz', mean=scaler.mean_, scale=scaler.scale_,
                 coef=model.coef_, intercept=model.intercept_, classes=model.classes_,
                 **{role: values for role, values in p.items()})
        candidate['parameters_sha256'] = sha(folder/f'candidate{k}.npz')
        candidates.append(candidate)
    selected = min(candidates, key=selection_key)
    record = {'job': job, 'plan_sha256': sha(output/'plan.json'), 'candidates': candidates,
              'selected_id': selected['id'], 'selected_C': selected['C'], 'metrics': selected['metrics'],
              'feature_sha256': sha(output/'inputs'/job['dataset'].lower()/f'{job["model"]}.npy')}
    atomic(folder/'record.json', record)
    pack = np.load(folder/f'candidate{selected["id"]}.npz', allow_pickle=False)
    for role in ROLES:
        rows = table.iloc[idx[role]][['trial_id', 'subject_id', 'material_key', 'original_label', 'label']].copy()
        rows['role'] = role
        for c in range(pack[role].shape[1]):
            rows[f'p{c}'] = pack[role][:, c]
        rows.to_csv(folder/f'{role}.csv', index=False)
    record['artifact_sha256'] = {p.name: sha(p) for p in folder.iterdir() if p.suffix in ('.npz', '.csv')}
    atomic(folder/'record.json', record)


def export(root, output):
    import shutil
    destination = REPO/'results/development'/STUDY
    records = []
    wanted = [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.md')]
    for path in wanted:
        relative = path.relative_to(output)
        # Do not export incomplete/unverified fitted records.
        if relative.parts[0] == 'fits' and not (path.parent/'verification.json').exists():
            continue
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != sha(path):
            raise ValueError('Export byte mismatch')
        records.append({'file': relative.as_posix(), 'sha256': sha(target)})
    audit_dest = REPO/'results/development'/ASSETS
    audit_dest.mkdir(parents=True, exist_ok=True)
    for name in ('download_manifest.json', 'code_review.json', 'feasibility.json', 'environment_preflight.json'):
        shutil.copyfile(root/ASSETS/name, audit_dest/name)
    atomic(destination/'export_manifest.json', {'files': records,
        'excluded': 'EEG, foundation checkpoint/source assets, embeddings, fitted coefficients stay local'})
    return destination
