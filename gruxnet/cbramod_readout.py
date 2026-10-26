"""Matched nonlinear readouts with an isolated head-dropout random stream."""
from pathlib import Path
import json
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from .cbramod_adaptation import Adaptation, new_model as previous_model
from .cbramod_conditioning import (
    STUDY as PRIOR, SOURCES as PREVIOUS_SOURCES, STEPS, INFERENCE_BATCH, ROLES, REPO,
    validate as previous_validate, identifier as prior_identifier,
    checkpoint_path as prior_checkpoint_path, dropout_schema, source_data, sampling,
    state_digest, sha, atomic, stamp, snapshot, grad_l2, exposure_records,
)
from .train import seed_everything

STUDY = 'cbramod_readout_2026-10-09'
HEADS = ('pooled_linear', 'pooled_mlp', 'flattened_mlp')
HEAD_SEED = 4242
HEAD_DROPOUT_SEED = 424243
SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES,
    'gruxnet/cbramod_readout.py', 'scripts/cbramod_readout.py',
    'scripts/audit_cbramod_readout.py', 'scripts/analyze_cbramod_readout.py',
    'tests/test_cbramod_readout.py', 'tests/test_cbramod_readout_analysis.py',
    'scripts/analyze_cbramod_learning.py', 'scripts/audit_cbramod_readout_shapes.py')))


class TokenMean(nn.Module):
    def forward(self, tokens):
        return tokens.mean((1, 2))


class IsolatedDropout(nn.Dropout):
    """Use the author dropout operator without consuming encoder RNG state."""
    update_index = 0

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        devices = [x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if x.is_cuda:
                with torch.cuda.device(x.device):
                    torch.cuda.manual_seed(HEAD_DROPOUT_SEED + self.update_index)
            else:
                torch.random.default_generator.manual_seed(HEAD_DROPOUT_SEED + self.update_index)
            return nn.functional.dropout(x, self.p, training=True, inplace=self.inplace)


def nonlinear_head(head, channels, classes):
    if head not in HEADS[1:]:
        raise ValueError('Expected a declared nonlinear head')
    torch.manual_seed(HEAD_SEED)
    return nn.Sequential(
        TokenMean() if head == 'pooled_mlp' else nn.Flatten(start_dim=1),
        nn.Linear(200 if head == 'pooled_mlp' else channels*10*200, 200),
        nn.ELU(), IsolatedDropout(.1), nn.Linear(200, classes))


class Readout(Adaptation):
    def __init__(self, backbone, classes, head, channels):
        super().__init__(backbone, classes, True)
        self.readout = head
        if head != 'pooled_linear':
            self.head = nonlinear_head(head, channels, classes)

    def forward(self, x):
        if self.readout == 'pooled_linear':
            return super().forward(x)
        return self.head(self.backbone(x))

    def set_update(self, step):
        if self.readout != 'pooled_linear':
            self.head[3].update_index = int(step)


def jobs():
    return [dict(dataset=d, group=g, pretrained=p, head=h)
            for d in ('DEAP', 'SEEDIV') for g in (1, 2)
            for p in (True, False) for h in HEADS]


def identifier(job):
    return f'{job["dataset"].lower()}_g{job["group"]}_{"pretrained" if job["pretrained"] else "random42"}_{job["head"]}'


def is_anchor(job):
    return job['head'] == 'pooled_linear'


def anchor_job(job):
    return {k: job[k] for k in ('dataset', 'group', 'pretrained')} | {
        'trainable': True, 'scaled': False, 'dropout': True}


def new_model(root, classes, job):
    base = previous_model(root, classes, job['pretrained'], True)
    channels = 32 if job['dataset'] == 'DEAP' else 62
    return Readout(base.backbone, classes, job['head'], channels).to('cuda')


def checkpoint_path(root, folder, record, item):
    if record['reused']:
        source = root/PRIOR/'runs'/prior_identifier(anchor_job(record['job']))
        original = json.loads((source/'record.json').read_text())
        return prior_checkpoint_path(root, source, original, item)
    return folder/item['checkpoint']


def import_anchor(root, output, job):
    from scripts.audit_cbramod_conditioning import read_record
    folder = output/'runs'/identifier(job)
    if (folder/'verification.json').exists():
        return
    if folder.exists() or not is_anchor(job):
        raise FileExistsError('Preserve readout anchor or reject wrong head')
    source, record = read_record(root, root/PRIOR, anchor_job(job))
    proof = json.loads((source/'verification.json').read_text())
    if not proof['complete'] or proof['record_sha256'] != sha(source/'record.json'):
        raise ValueError('Unverified conditioning anchor')
    folder.mkdir(parents=True)
    for path in source.iterdir():
        if path.name == 'history.json' or path.suffix == '.csv':
            shutil.copyfile(path, folder/path.name)
    fields = {k: record[k] for k in ('initial_encoder_digest', 'initial_head_digest',
        'sampling_digest', 'candidates', 'exposure', 'seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
    atomic(folder/'record.json', {'job': job, 'plan_sha256': sha(output/'plan.json'),
        'reused': True, 'source_record_sha256': sha(source/'record.json'),
        'source_verification_sha256': sha(source/'verification.json'), **fields,
        'head_parameters': (200+1)*(2 if job['dataset'] == 'DEAP' else 3),
        'head_dropout_rng': None, 'dropout_schema': None,
        'artifact_sha256': {p.name: sha(p) for p in folder.iterdir()}})


def fit(root, output, job):
    if is_anchor(job):
        import_anchor(root, output, job)
        return
    folder = output/'runs'/identifier(job)
    if (folder/'verification.json').exists():
        return
    if folder.exists():
        raise FileExistsError('Preserve partial readout fit')
    folder.mkdir(parents=True)
    table, idx, data = source_data(root, job)
    model = new_model(root, int(table.label.max()+1), job)
    device = next(model.parameters()).device
    initial_encoder = state_digest(model.backbone.state_dict())
    initial_head = state_digest(model.head.state_dict())
    draws, windows, signature = sampling(table, idx['train'], updates=STEPS[-1])
    optimizer = torch.optim.AdamW([
        {'params': model.head.parameters(), 'lr': .001},
        {'params': model.backbone.parameters(), 'lr': 1e-4}], weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS[-1], eta_min=1e-6)
    seed_everything(424242)
    history = []; losses = []; norms = []; candidates = []
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    candidates.append(snapshot(model, data, table, idx, folder, 0))
    for step in range(1, STEPS[-1]+1):
        model.train(); model.set_update(step); optimizer.zero_grad(set_to_none=True)
        rows = draws[step-1]
        x = torch.tensor(data[rows, windows[step-1]], device=device)
        y = torch.tensor(table.label.to_numpy(dtype=np.int64)[rows], device=device)
        with sdpa_kernel(SDPBackend.MATH):
            loss = nn.functional.cross_entropy(model(x), y, label_smoothing=.1)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite readout loss')
            loss.backward()
        if step % 100 == 0:
            head_norm = grad_l2(model.head.parameters()); encoder_norm = grad_l2(model.backbone.parameters())
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step(); scheduler.step()
        losses.append(float(loss.item())); norms.append(norm)
        if step % 100 == 0:
            history.append({'step': step, 'mean_last100_minibatch_loss': float(np.mean(losses[-100:])),
                'gradient_L2_before_clip': norm, 'head_gradient_L2_before_clip': head_norm,
                'encoder_gradient_L2_before_clip': encoder_norm,
                'clipped_updates_last100': sum(v > 1 for v in norms[-100:]),
                'mean_global_norm_last100': float(np.mean(norms[-100:])),
                'max_global_norm_last100': max(norms[-100:]),
                'head_lr': optimizer.param_groups[0]['lr'], 'encoder_lr': optimizer.param_groups[1]['lr']})
        if step in STEPS:
            cpu = torch.get_rng_state(); gpu = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
            candidates.append(snapshot(model, data, table, idx, folder, step))
            torch.set_rng_state(cpu)
            if gpu is not None:
                torch.cuda.set_rng_state_all(gpu)
    atomic(folder/'history.json', history)
    atomic(folder/'record.json', {'job': job, 'plan_sha256': sha(output/'plan.json'), 'reused': False,
        'initial_encoder_digest': initial_encoder, 'initial_head_digest': initial_head,
        'head_parameters': sum(p.numel() for p in model.head.parameters()),
        'head_dropout_rng': {'p': .1, 'seed': HEAD_DROPOUT_SEED, 'index': 'seed + update; encoder global RNG restored'},
        'dropout_schema': dropout_schema(model), 'sampling_digest': signature,
        'candidates': candidates, 'exposure': exposure_records(idx['train'], draws, windows, STEPS, 'long'),
        'seconds': time.perf_counter()-start,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0,
        'artifact_sha256': {p.name: sha(p) for p in folder.iterdir()}})


def validate(root, output, deep_inputs=False):
    previous_validate(root, root/PRIOR, deep_inputs=deep_inputs)
    plan = json.loads((output/'plan.json').read_text())
    for name, checksum in plan['source_sha256'].items():
        if sha(REPO/name) != checksum:
            raise ValueError('Changed frozen readout source: '+name)
    for name, checksum in plan['upstream_sha256'].items():
        if sha(root/name) != checksum:
            raise ValueError('Changed readout upstream: '+name)
    return plan


def declare(root):
    previous_validate(root, root/PRIOR, deep_inputs=True)
    if not json.loads((root/PRIOR/'verification.json').read_text())['complete']:
        raise ValueError('Completed conditioning predecessor required')
    resource = root/'cbramod_readout_resource_2026-10-09/preflight.json'
    if not resource.exists() or json.loads(resource.read_text())['source_sha256'] != sha(Path(__file__)):
        raise ValueError('Matching outcome-free operator/GPU checks required')
    output = root/STUDY
    if output.exists():
        raise FileExistsError('Preserve readout declaration')
    bindings = {f'{PRIOR}/{n}': sha(root/PRIOR/n) for n in ('plan.json', 'summary.json', 'verification.json', 'config.json')}
    bindings[resource.relative_to(root).as_posix()] = sha(resource)
    shape = root/'cbramod_readout_audit_2026-10-09/verification.json'
    bindings[shape.relative_to(root).as_posix()] = sha(shape)
    shape_record = json.loads(shape.read_text())
    for name, checksum in shape_record['author_sha256'].items():
        path = root/'cbramod_audit_2026-10-09/author'/name
        if sha(path) != checksum:
            raise ValueError('Changed inspected author source')
        bindings[path.relative_to(root).as_posix()] = sha(path)
    for job in jobs():
        if is_anchor(job):
            folder = root/PRIOR/'runs'/prior_identifier(anchor_job(job))
            for name in ('record.json', 'verification.json'):
                bindings[(folder/name).relative_to(root).as_posix()] = sha(folder/name)
    output.mkdir()
    atomic(output/'plan.json', {'created_utc': stamp(), 'development_only': True,
        'outer_test_inferences': 0, 'research_question_change_approved': False,
        'source_sha256': {n: sha(REPO/n) for n in SOURCES}, 'upstream_sha256': bindings,
        'jobs': jobs(), 'steps': STEPS,
        'counts': {'conditions': 24, 'new_trajectories': 16, 'reused_exact_anchors': 8,
                   'states': 96, 'probability_metric_sets': 288},
        'scope': 'Same source-only DEAP/SEED-IV group1/2 session1/rotation0/fold0 panels. Native32/62 channels, prepared200Hz40s arrays/divisor100, four disjoint10s windows, corrected individual DEAP binary valence/unchanged assigned coarse3 SEED-IV. No test inference, label/preprocessing/calibration change or outcome-dependent exclusions; repeated development panels and one head/random initialization.',
        'heads': 'Eight exact pooled-linear fulltrain raw/dropout-on anchors; sixteen new pooled/flattened two-layer heads. Pooled mean200 or contiguouschannel,time,dimension flatten -> Linear(input,200)->ELU->Dropout(.1)->Linear(200,2/3). Exact pinned FACED standalone two-layer operators; SEED-V input expands one to ten patches and bypasses wrapper premature flatten. Pooled MLP applies same operators to mean tokens. No large three-layer default or claimed published score reproduction. Head family/dropout/parameter count differ; not a pure pooling comparison.',
        'training': 'All new cases exactly1200balanced6window AdamW updates, headLR.001/encoderLR1e-4,wd.05/smoothing.1/globalclip1,cosine1200eta_min1e-6. Author encoder dropout unchanged/on. Encoder seeds pretrained/random42, head4242; dimension-specific weights, shared exact head within each dataset/head acrosspretraining/groups. Globalencoder/dropoutseed424242, same NumPy observation20261009/window20261010 streams. New head dropout runs author F.dropout at seed424243+update within fork_rng, restoring CPU/activeGPU global states; same hidden200 mask acrossnewheads. Rawanchors consume nohead randomness. FP32CUDAexplicitMATH,noAMP/noaugmentation. Evaluation averages4windowlogitsFP64 then softmax; canonicalbatch16.',
        'analysis': 'Retain all0/200/600/1200 source probabilities/states. Primarysame-step pooledMLP-minus-linear,flattenedMLP-minus-linear,andflattened-minus-pooledMLP; separately matched pretrained-minus-random within each head. Alltrain/familiar/unseen roles retained, lossprimary/BAsecondary, bothgroupings/corpora separately. Secondarypertrajectorycheckpointselectionamong3positivesteps equalfamiliar/unseenbalancedloss,thenmeanBA,stablefirst. No globalwinningrecipe,no populationintervals/no postfithead/optimizersearch. Exposure/clippinghistories retained; anchors lackclippingcounts. Fullgrid before aggregate decision.',
        'verification': 'Executed pinned author standaloneoperator/gradient/dropout equivalence in syntheticCPUchecks; native32/62GPUpreflightincludingbackprop andcanonicalinference. Every96state strictlyrestored, explicitbackbone->mean/flatten->functionalLinearELUfunctionalLinear replay atsamebatch16, tolerance2e-6probabilities/metrics and2e-11publicmetricrecompute. Check unchanged/updateddigests,pairedencoder/head/streams,metadata/boundaries,exposure/selection,all8anchorCSV/history/checkpointreferencesexact. Fulloptimizerreplay/externaldata/pretrainingauthentication notclaimed. Deepinput/cache hashes beforedeclaration/completion. ExclusiveGPUworkerlock; preservefailures.',
        'publication': 'Author approved verified probabilities/labels/anonymous IDs andregularGitHubpush. Freeze/pushcode,protocol,plan beforetaskfits; pushanchors,every4newverifiedruns,completion. RawEEG/embeddings/scalers/checkpoints/pertrialphysicalamplitudes private.',
        'decision': 'A lead needs consistentprimaryunseenloss benefit in bothgroupings of claimedcorpus acrossmatchedsteps andmeaningfulperformanceagainstuniform/source-selectedspectralcontrols. Training-onlygain isfailureofpredictivelead. Repetition/noveltycheckrequired beforeconfirmation. No adoptednewquestion; actualpivotneedsfindings/proposal,explicitauthorapproval andfreshmanuscriptarchive.'})
    shutil.copyfile(resource, output/'resource_preflight.json')
    progress(output, 'declared')
    return output


def progress(output, state):
    records = [json.loads(p.read_text()) for p in (output/'runs').glob('*/record.json') if (p.parent/'verification.json').exists()]
    atomic(output/'progress.json', {'state': state, 'updated_utc': stamp(),
        'conditions_completed': len(records), 'conditions_total': 24,
        'new_trajectories_completed': sum(not r['reused'] for r in records), 'new_trajectories_total': 16,
        'anchors_completed': sum(r['reused'] for r in records), 'anchors_total': 8,
        'outer_test_inferences': 0, 'research_question_change_approved': False})


def export(output):
    destination = REPO/'results/development'/STUDY
    files = []
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json', '.csv', '.md'):
            continue
        relative = path.relative_to(output)
        if relative.parts[0] == 'runs' and not (path.parent/'verification.json').exists():
            continue
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != sha(path):
            raise ValueError('Readout export bytes differ')
        files.append({'file': relative.as_posix(), 'sha256': sha(target)})
    atomic(destination/'export_manifest.json', {'files': files,
        'scope': 'Approved verified probabilities/labels/anonymous references and aggregate diagnostics. No raw EEG, embeddings or weights.'})
    return destination


def resource_preflight(root):
    from scripts.audit_cbramod_readout import author_operator_audit
    output = root/'cbramod_readout_resource_2026-10-09'
    if output.exists():
        raise FileExistsError('Preserve readout preflight')
    output.mkdir()
    operator = author_operator_audit(root)
    records = []
    for dataset, channels, classes in (('DEAP', 32, 2), ('SEEDIV', 62, 3)):
        for head in HEADS[1:]:
            job = dict(dataset=dataset, group=1, pretrained=True, head=head)
            model = new_model(root, classes, job).train()
            x = torch.tensor(np.random.default_rng(channels).normal(0, .1, (6, channels, 10, 200)).astype(np.float32), device='cuda')
            y = torch.arange(6, device='cuda') % classes
            optimizer = torch.optim.AdamW([{'params': model.head.parameters(), 'lr': .001},
                {'params': model.backbone.parameters(), 'lr': 1e-4}], weight_decay=.05)
            before = state_digest(model.backbone.state_dict())
            seed_everything(424242); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            with sdpa_kernel(SDPBackend.MATH):
                for step in range(1, 4):
                    model.set_update(step); optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(x), y, label_smoothing=.1)
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
                model.eval()
                with torch.inference_mode():
                    logits = model(torch.zeros((16, channels, 10, 200), device='cuda'))
            if not torch.isfinite(loss) or not torch.isfinite(logits).all() or before == state_digest(model.backbone.state_dict()):
                raise ValueError('Synthetic readout backpropagation failed')
            records.append({'dataset': dataset, 'head': head, 'channels': channels, 'classes': classes,
                'head_parameters': sum(p.numel() for p in model.head.parameters()),
                'three_updates_and_inference_seconds': time.perf_counter()-start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'synthetic_loss': float(loss.item())})
            del model, optimizer, x, y, logits, loss
            torch.cuda.empty_cache()
    result = {'created_utc': stamp(), 'synthetic_only': True, 'task_fits': 0,
        'source_sha256': sha(Path(__file__)), 'device': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'author_operator_audit': operator, 'records': records}
    atomic(output/'preflight.json', result)
    return result
