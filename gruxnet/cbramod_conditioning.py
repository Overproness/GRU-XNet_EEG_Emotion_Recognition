"""Matched source-only embedding conditioning and encoder-dropout controls."""
from pathlib import Path
import json
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from scipy.special import softmax

from .cbramod_adaptation import Adaptation, new_model as previous_model
from .cbramod_learning import (
    STUDY as PRIOR, PREDECESSOR, SOURCES as PREVIOUS_SOURCES, ROLES, REPO,
    validate as previous_validate, source_data, sampling, state_digest, sha,
    atomic, stamp, scores, save_prediction, grad_l2, exposure_records,
    disable_capacity_dropout, identifier as previous_identifier,
)
from .train import seed_everything
from .data import digest

STUDY = 'cbramod_conditioning_2026-10-09'
STEPS = (0, 200, 600, 1200)
SCALE_FLOOR = 1e-6
INFERENCE_BATCH = 16
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES,
    'gruxnet/cbramod_conditioning.py', 'scripts/cbramod_conditioning.py',
    'scripts/audit_cbramod_conditioning.py', 'tests/test_cbramod_conditioning.py')))


def jobs():
    return [dict(dataset=d, group=g, pretrained=p, trainable=t, scaled=s, dropout=q)
            for d in ('DEAP', 'SEEDIV') for g in (1, 2) for p in (True, False)
            for t in (False, True) for s in (False, True) for q in (True, False)]


def normalizer_jobs():
    return [dict(dataset=d, group=g, pretrained=p) for d in ('DEAP', 'SEEDIV')
            for g in (1, 2) for p in (True, False)]


def normalizer_id(job):
    return f'{job["dataset"].lower()}_g{job["group"]}_{"pretrained" if job["pretrained"] else "random42"}'


def identifier(job):
    return normalizer_id(job) + ('_finetune' if job['trainable'] else '_frozen') + \
        ('_standardized' if job['scaled'] else '_raw') + ('_dropout_on' if job['dropout'] else '_dropout_off')


def is_anchor(job):
    return not job['scaled'] and job['dropout']


def anchor_job(job):
    return {k: job[k] for k in ('dataset', 'group', 'pretrained', 'trainable')}


def statistics(features, training_rows):
    """Equal weight for each original training trial and each of its four windows."""
    x = np.asarray(features[training_rows], dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (4, 200) or not np.isfinite(x).all():
        raise ValueError('Expected finite source-only four-window 200D features')
    flat = x.reshape(-1, 200)
    mean = flat.mean(0)
    scale = np.maximum(np.sqrt(np.mean((flat - mean)**2, axis=0)), SCALE_FLOOR)
    return mean, scale


class Conditioned(Adaptation):
    def __init__(self, backbone, classes, trainable, mean=None, scale=None):
        super().__init__(backbone, classes, trainable)
        self.scaled = mean is not None
        if self.scaled:
            if scale is None or np.shape(mean) != (200,) or np.shape(scale) != (200,):
                raise ValueError('Expected a complete 200D conditioner')
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
                raise ValueError('Invalid conditioner')
            self.register_buffer('embedding_mean', torch.tensor(mean, dtype=torch.float32))
            self.register_buffer('embedding_scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        if not self.scaled:
            # Preserve the exact predecessor operator path, with no identity arithmetic.
            return super().forward(x)
        if self.trainable_encoder:
            tokens = self.backbone(x)
        else:
            with torch.no_grad():
                tokens = self.backbone(x)
        pooled = tokens.mean((1, 2))
        return self.head((pooled - self.embedding_mean) / self.embedding_scale)


def new_model(root, classes, job, mean=None, scale=None):
    base = previous_model(root, classes, job['pretrained'], job['trainable'])
    model = Conditioned(base.backbone, classes, job['trainable'], mean, scale).to('cuda')
    if not job['dropout']:
        disable_capacity_dropout(model)
    return model


def dropout_schema(model):
    return [{'name': name, 'type': type(m).__name__, 'p': float(m.p if isinstance(m, nn.Dropout) else m.dropout)}
            for name, m in model.backbone.named_modules() if isinstance(m, (nn.Dropout, nn.MultiheadAttention))]


def pooled_features(model, data, rows, batch=INFERENCE_BATCH):
    x = data[rows].reshape(len(rows)*4, data.shape[2], 10, 200)
    device = next(model.parameters()).device
    parts = []
    model.eval()
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        for start in range(0, len(x), batch):
            tokens = model.backbone(torch.tensor(x[start:start+batch], device=device))
            parts.append(tokens.mean((1, 2)).cpu().numpy())
    features = np.concatenate(parts).reshape(len(rows), 4, 200)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite initial source features')
    return features


def prepare_normalizer(root, output, job):
    folder = output/'normalizers'/normalizer_id(job)
    if (folder/'verification.json').exists():
        return
    if folder.exists():
        raise FileExistsError('Preserve partial normalizer')
    folder.mkdir(parents=True)
    table, idx, data = source_data(root, job)
    model = previous_model(root, int(table.label.max()+1), job['pretrained'], False)
    before = state_digest(model.backbone.state_dict())
    features = pooled_features(model, data, idx['train'])
    mean, scale = statistics(features, np.arange(len(features)))
    np.savez(folder/'statistics.npz', features=features, mean64=mean, scale64=scale,
             mean32=mean.astype(np.float32), scale32=scale.astype(np.float32))
    if before != state_digest(model.backbone.state_dict()):
        raise ValueError('Encoder changed while preparing source statistics')
    atomic(folder/'record.json', {'job': job, 'plan_sha256': sha(output/'plan.json'),
        'initial_encoder_digest': before, 'training_trial_ids': table.iloc[idx['train']].trial_id.tolist(),
        'training_trials': len(features), 'windows_per_trial': 4, 'feature_dimension': 200,
        'encoder_mode': 'eval', 'extraction_batch': INFERENCE_BATCH, 'scale_floor': SCALE_FLOOR,
        'statistics_sha256': sha(folder/'statistics.npz')})


def load_statistics(output, job):
    folder = output/'normalizers'/normalizer_id(job)
    record = json.loads((folder/'record.json').read_text())
    if sha(folder/'statistics.npz') != record['statistics_sha256']:
        raise ValueError('Changed source statistics')
    with np.load(folder/'statistics.npz', allow_pickle=False) as state:
        return state['mean32'].copy(), state['scale32'].copy(), record


def checkpoint_path(root, folder, record, item):
    if record['reused']:
        return root/PRIOR/'long'/previous_identifier('long', anchor_job(record['job']))/item['checkpoint']
    return folder/item['checkpoint']


def import_anchor(root, output, job):
    from scripts.audit_cbramod_learning import read_record as previous_record
    folder = output/'runs'/identifier(job)
    if (folder/'verification.json').exists():
        return
    if folder.exists() or not is_anchor(job):
        raise FileExistsError('Preserve imported anchor or reject incorrect anchor')
    source, record = previous_record(root/PRIOR, 'long', anchor_job(job))
    proof = json.loads((source/'verification.json').read_text())
    if not proof['complete'] or proof['record_sha256'] != sha(source/'record.json'):
        raise ValueError('Unverified predecessor anchor')
    folder.mkdir(parents=True)
    for path in source.iterdir():
        if path.name == 'history.json' or path.suffix == '.csv':
            shutil.copyfile(path, folder/path.name)
    extra = {k: record[k] for k in ('initial_encoder_digest', 'initial_head_digest', 'sampling_digest',
        'candidates', 'exposure', 'seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
    atomic(folder/'record.json', {'job': job, 'plan_sha256': sha(output/'plan.json'), **extra,
        'reused': True, 'source_record_sha256': sha(source/'record.json'),
        'source_verification_sha256': sha(source/'verification.json'), 'normalizer': None,
        'dropout_schema': None, 'artifact_sha256': {p.name: sha(p) for p in folder.iterdir()}})


def infer(model, data, rows):
    x = data[rows].reshape(len(rows)*4, data.shape[2], 10, 200)
    device = next(model.parameters()).device
    parts = []
    model.eval()
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        for start in range(0, len(x), INFERENCE_BATCH):
            parts.append(model(torch.tensor(x[start:start+INFERENCE_BATCH], device=device)).cpu().numpy())
    logits = np.concatenate(parts).reshape(len(rows), 4, -1).mean(1, dtype=np.float64)
    return softmax(logits, axis=1)


def snapshot(model, data, table, idx, folder, step):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    name = f'checkpoint{step}.pt'
    torch.save(state, folder/name)
    probabilities = {r: infer(model, data, rows) for r, rows in idx.items()}
    for role, rows in idx.items():
        save_prediction(folder, role, table, rows, probabilities[role], prefix=f'step{step}_')
    return {'step': step, 'checkpoint': name, 'checkpoint_sha256': sha(folder/name),
        'state_digest': state_digest(state), 'encoder_digest': state_digest(model.backbone.state_dict()),
        'head_digest': state_digest(model.head.state_dict()),
        'metrics': {r: scores(table.iloc[idx[r]].label.to_numpy(), p) for r, p in probabilities.items()}}


def fit(root, output, job):
    if is_anchor(job):
        import_anchor(root, output, job)
        return
    folder = output/'runs'/identifier(job)
    if (folder/'verification.json').exists():
        return
    if folder.exists():
        raise FileExistsError('Preserve partial conditioning fit')
    folder.mkdir(parents=True)
    table, idx, data = source_data(root, job)
    mean, scale, normalizer = load_statistics(output, job) if job['scaled'] else (None, None, None)
    model = new_model(root, int(table.label.max()+1), job, mean, scale)
    device = next(model.parameters()).device
    initial_encoder = state_digest(model.backbone.state_dict())
    initial_head = state_digest(model.head.state_dict())
    if normalizer and normalizer['initial_encoder_digest'] != initial_encoder:
        raise ValueError('Normalizer fitted on a different initial encoder')
    draws, windows, signature = sampling(table, idx['train'], updates=STEPS[-1])
    groups = [{'params': model.head.parameters(), 'lr': .001}]
    if job['trainable']:
        groups.append({'params': model.backbone.parameters(), 'lr': 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS[-1], eta_min=1e-6)
    seed_everything(424242)
    history = []; losses = []; norms = []; candidates = []
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    candidates.append(snapshot(model, data, table, idx, folder, 0))
    for step in range(1, STEPS[-1]+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        rows = draws[step-1]
        x = torch.tensor(data[rows, windows[step-1]], device=device)
        y = torch.tensor(table.label.to_numpy(dtype=np.int64)[rows], device=device)
        with sdpa_kernel(SDPBackend.MATH):
            loss = nn.functional.cross_entropy(model(x), y, label_smoothing=.1)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite conditioning loss')
            loss.backward()
        if step % 100 == 0:
            head_norm = grad_l2(model.head.parameters()); encoder_norm = grad_l2(model.backbone.parameters())
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step(); scheduler.step()
        losses.append(float(loss.item())); norms.append(norm)
        if step % 100 == 0:
            history.append({'step': step, 'mean_last100_minibatch_loss': float(np.mean(losses[-100:])),
                'gradient_L2_before_clip': norm, 'head_gradient_L2_before_clip': head_norm,
                'encoder_gradient_L2_before_clip': encoder_norm, 'clipped_updates_last100': sum(v > 1 for v in norms[-100:]),
                'mean_global_norm_last100': float(np.mean(norms[-100:])), 'max_global_norm_last100': max(norms[-100:]),
                'head_lr': optimizer.param_groups[0]['lr'], 'encoder_lr': optimizer.param_groups[-1]['lr'] if job['trainable'] else 0.})
        if step in STEPS:
            cpu = torch.get_rng_state(); gpu = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
            candidates.append(snapshot(model, data, table, idx, folder, step))
            torch.set_rng_state(cpu)
            if gpu is not None:
                torch.cuda.set_rng_state_all(gpu)
    atomic(folder/'history.json', history)
    atomic(folder/'record.json', {'job': job, 'plan_sha256': sha(output/'plan.json'), 'reused': False,
        'initial_encoder_digest': initial_encoder, 'initial_head_digest': initial_head,
        'normalizer': normalizer_id(job) if job['scaled'] else None,
        'normalizer_record_sha256': sha(output/'normalizers'/normalizer_id(job)/'record.json') if job['scaled'] else None,
        'dropout_schema': dropout_schema(model), 'sampling_digest': signature, 'candidates': candidates,
        'exposure': exposure_records(idx['train'], draws, windows, STEPS, 'long'),
        'seconds': time.perf_counter()-start,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0,
        'artifact_sha256': {p.name: sha(p) for p in folder.iterdir()}})


def validate(root, output, deep_inputs=False):
    previous_validate(root, root/PRIOR, deep_inputs=deep_inputs)
    plan = json.loads((output/'plan.json').read_text())
    for n, s in plan['source_sha256'].items():
        if sha(REPO/n) != s:
            raise ValueError('Changed frozen conditioning source: '+n)
    for n, s in plan['upstream_sha256'].items():
        if sha(root/n) != s:
            raise ValueError('Changed conditioning predecessor: '+n)
    return plan


def declare(root):
    previous_validate(root, root/PRIOR, deep_inputs=True)
    if not json.loads((root/PRIOR/'verification.json').read_text())['complete']:
        raise ValueError('Completed verified learning predecessor required')
    pilot = root/'cbramod_conditioning_resource_2026-10-09/preflight.json'
    if not pilot.exists() or json.loads(pilot.read_text())['source_sha256'] != sha(Path(__file__)):
        raise ValueError('Matching synthetic conditioning preflight required')
    output = root/STUDY
    if output.exists():
        raise FileExistsError('Preserve conditioning declaration')
    bindings = {f'{PRIOR}/{n}': sha(root/PRIOR/n) for n in ('plan.json', 'summary.json', 'verification.json', 'config.json')}
    bindings[pilot.relative_to(root).as_posix()] = sha(pilot)
    for job in jobs():
        if is_anchor(job):
            folder = root/PRIOR/'long'/previous_identifier('long', anchor_job(job))
            for n in ('record.json', 'verification.json'):
                bindings[(folder/n).relative_to(root).as_posix()] = sha(folder/n)
    output.mkdir()
    atomic(output/'plan.json', {'created_utc': stamp(), 'development_only': True, 'outer_test_inferences': 0,
        'research_question_change_approved': False, 'source_sha256': {n: sha(REPO/n) for n in SOURCES},
        'upstream_sha256': bindings, 'jobs': jobs(), 'normalizer_jobs': normalizer_jobs(), 'steps': STEPS,
        'counts': {'conditions': 64, 'new_trajectories': 48, 'reused_exact_anchors': 16,
                   'normalizers': 8, 'states': 256, 'probability_metric_sets': 768},
        'scope': 'Same native32/62-channel DEAP/SEED-IV groups1/2 session1/rotation0/fold0 unexposed source panels. Same corrected DEAP binary and assigned coarse3 SEED targets, 200Hz40s arrays/units100, four disjoint10s windows. No outer-test inference, original-label change, input calibration or outcome-dependent exclusions. Reused development people/materials; one head/random encoder initialization.',
        'factors': 'Pretrained/random42 x frozen/trainable x raw/initial-source-standardized pooled200D embeddings x author encoder dropout on/all61module+internalattention dropout sites off. Linear head, token pooling, streams, steps, optimizer and labels fixed within each pair. Raw bypasses identity arithmetic and preserves old forward/state layout. Reuse16raw/dropout-on anchors with byte-exact CSV/history/checkpoint references and strict replay;48new trajectories.',
        'normalization': 'Eight source-only initial-eval encoder statistic sets. Equal weight to all4ten-second windows of each original TRAIN trial, never validation/test; featuresFP32, mean/populationstdFP64, scale=max(std,1e-6). StoreFP32buffers, fixed even during finetuning, not optimized or updated on validation/current encoder. Raw has no scaler arithmetic/buffers. Shared statistic set across trainability/dropout conditions for each panel/pretraining. This is feature conditioning, not rawEEG normalization or a novel method.',
        'training': 'Exactly1200balanced6window AdamWupdates, headLR.001/encoderLR1e-4, wd.05/smoothing.1/globalclip1, cosine1200eta_min1e-6. Same heads4242/random encoder42/dropoutseed424242 and NumPy trial20261009/window20261010streams as predecessor. Disabling dropout intentionally changes RNG consumption/masks; observation/window streams remain exact. All conditions finish; no capacity-result-driven budget or optimizer change.',
        'selection': 'Primary matched factor effects at identical200/600/1200checkpoints in both validation roles, loss primary/BAsecondary; include initial and training diagnostics. Normalization effects at each dropout, dropout effects at each scaling, and their interaction; retain all pretraining/trainability/panel blocks. Separately selected checkpoints among3positive steps by equal familiar/unseen balancedloss,thenmeanBA,stablefirst are SECONDARY because durations can differ. No global winning recipe or population intervals.',
        'verification': 'Every normalizer independently reconstructed with sklearn StandardScaler then same floor; training IDs/initial encoder/cache hashes checked. Source features replay at fixedbatch16 and additionally batch8, reporting dimensionless normalized batch sensitivity before fitting. Every256neural state strictly restored and explicit backbone->mean->optionalfixedaffine->functional_linear replay at same canonical inferencebatch16. Fixedbatch avoids amplifying backbone batch-rounding through tiny scales; different-batch feature diagnostic is not whole-model batch invariance. Check frozen/updated module digests, unchanged conditioner buffers,61dropout schema,paired heads/streams,independent CSV metrics/metadata and64selections. No fulloptimizer replay or external recording/pretraining authentication.',
        'publication': 'Publish verified CSV/JSON probabilities/labels/anonymous IDs under existing explicit author approval. No EEG, embeddings, scaler arrays, coefficients/checkpoints or pertrial physical amplitudes. Push protocol/plan before preparation or fitting; normalizer and anchor phases, every4new verified runs and completion. Preserve failures/partial state; exclusive workerlock.',
        'decision': 'Completegrid before aggregate interpretation. Conditioning/dropout of an existing encoder alone is not novelty or approved paper pivot. Need robust matched lead before broader confirmation; actual main-question change requires findings/proposal,explicitauthorapproval and immediate fresh manuscript archive.'})
    shutil.copyfile(pilot, output/'resource_preflight.json')
    progress(output, 'declared')
    return output


def progress(output, state):
    records = [json.loads(p.read_text()) for p in (output/'runs').glob('*/record.json') if (p.parent/'verification.json').exists()]
    atomic(output/'progress.json', {'state': state, 'updated_utc': stamp(), 'normalizers_completed': len(list((output/'normalizers').glob('*/verification.json'))),
        'normalizers_total': 8, 'conditions_completed': len(records), 'conditions_total': 64,
        'new_trajectories_completed': sum(not r['reused'] for r in records), 'new_trajectories_total': 48,
        'anchors_completed': sum(r['reused'] for r in records), 'anchors_total': 16,
        'outer_test_inferences': 0, 'research_question_change_approved': False})


def export(output):
    destination = REPO/'results/development'/STUDY
    files = []
    for path in output.rglob('*'):
        if not path.is_file() or path.suffix not in ('.json', '.csv', '.md'):
            continue
        relative = path.relative_to(output)
        if relative.parts[0] in ('runs', 'normalizers') and not (path.parent/'verification.json').exists():
            continue
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != sha(path):
            raise ValueError('Conditioning publication copy mismatch')
        files.append({'file': relative.as_posix(), 'sha256': sha(target)})
    atomic(destination/'export_manifest.json', {'files': files,
        'scope': 'Approved verified probabilities/labels/anonymous IDs and aggregate diagnostics; no raw EEG, embeddings, scalers or weights.'})
    return destination


def resource_preflight(root):
    output = root/'cbramod_conditioning_resource_2026-10-09'
    if output.exists():
        raise FileExistsError('Preserve conditioning preflight')
    output.mkdir(); results = []
    for channels, classes in ((32, 2), (62, 3)):
        x = torch.tensor(np.random.default_rng(channels).normal(0, .1, (6, channels, 10, 200)).astype(np.float32), device='cuda')
        for scaled in (False, True):
            for dropout in (True, False):
                job = dict(pretrained=True, trainable=True, scaled=scaled, dropout=dropout)
                base = previous_model(root, classes, True, True).eval()
                with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
                    z = base.backbone(x).mean((1, 2)).cpu().numpy().astype(np.float64)
                mean = z.mean(0) if scaled else None
                scale = np.maximum(z.std(0), SCALE_FLOOR) if scaled else None
                del base
                model = new_model(root, classes, job, mean, scale).train()
                before = state_digest(model.backbone.state_dict())
                schema = dropout_schema(model)
                optimizer = torch.optim.AdamW([{'params': model.head.parameters(), 'lr': .001},
                    {'params': model.backbone.parameters(), 'lr': 1e-4}], weight_decay=.05)
                seed_everything(424242); torch.cuda.reset_peak_memory_stats()
                with sdpa_kernel(SDPBackend.MATH):
                    if not dropout:
                        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        loss = nn.functional.cross_entropy(model(x), torch.arange(6, device='cuda') % classes, label_smoothing=.1)
                        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
                if not torch.isfinite(loss) or before == state_digest(model.backbone.state_dict()):
                    raise ValueError('Synthetic conditioner backpropagation failed')
                if len(schema) != 61 or (not dropout and any(s['p'] != 0 for s in schema)):
                    raise ValueError('Incorrect native encoder dropout configuration')
                if scaled:
                    np.testing.assert_array_equal(model.embedding_mean.cpu().numpy(), mean.astype(np.float32))
                    np.testing.assert_array_equal(model.embedding_scale.cpu().numpy(), scale.astype(np.float32))
                results.append({'channels': channels, 'classes': classes, 'scaled': scaled, 'dropout': dropout,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'synthetic_loss': float(loss.item())})
                del model, optimizer
        del x
    result = {'synthetic_only': True, 'task_fits': 0, 'created_utc': stamp(),
        'source_sha256': sha(Path(__file__)), 'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'records': results}
    atomic(output/'preflight.json', result)
    return result
