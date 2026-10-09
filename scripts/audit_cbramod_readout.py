"""Executed author head equivalence and independent full-state readout replay."""
from pathlib import Path
import importlib.util
import json
import sys
import types
import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from scipy.special import softmax

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_readout import (
    STUDY, PRIOR, STEPS, HEADS, HEAD_DROPOUT_SEED, ROLES, identifier, is_anchor,
    anchor_job, prior_identifier, checkpoint_path, new_model, nonlinear_head,
    source_data, sampling, state_digest, sha, atomic, stamp, validate, dropout_schema,
)
from scripts.audit_cbramod_learning import check_exposure, compare_csv, select


def functional_readout(model, tokens):
    if model.readout == 'pooled_linear':
        return nn.functional.linear(tokens.mean((1, 2)), model.head.weight, model.head.bias)
    features = tokens.mean((1, 2)) if model.readout == 'pooled_mlp' else tokens.flatten(1)
    hidden = nn.functional.linear(features, model.head[1].weight, model.head[1].bias)
    hidden = nn.functional.elu(hidden)
    return nn.functional.linear(hidden, model.head[4].weight, model.head[4].bias)


def explicit_replay(model, data, rows):
    x = data[rows].reshape(len(rows)*4, data.shape[2], 10, 200)
    device = next(model.parameters()).device
    chunks = []; model.eval()
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        for start in range(0, len(x), 16):
            tokens = model.backbone(torch.tensor(x[start:start+16], device=device))
            chunks.append(functional_readout(model, tokens).cpu().numpy())
    logits = np.concatenate(chunks).reshape(len(rows), 4, -1).astype(np.float64)
    return softmax(logits.mean(axis=1), axis=1)


def author_operator_audit(root):
    """Execute pinned standalone heads; modify only documented input dimensions."""
    source = root/'cbramod_audit_2026-10-09/author'
    old_proof = json.loads((root/'cbramod_readout_audit_2026-10-09/verification.json').read_text())
    for name, checksum in old_proof['author_sha256'].items():
        if sha(source/name) != checksum:
            raise ValueError('Changed pinned author operator source')
    class SyntheticBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__(); self.proj_out = nn.Identity()
        def forward(self, x):
            return x
    names = ('models', 'models.cbramod', 'models.model_for_faced', 'models.model_for_seedv')
    previous = {n: sys.modules.get(n) for n in names}
    package = types.ModuleType('models'); package.__path__ = [str(source/'models')]
    stub = types.ModuleType('models.cbramod'); stub.CBraMod = SyntheticBackbone
    sys.modules['models'] = package; sys.modules['models.cbramod'] = stub
    records = []
    try:
        for dataset, channels, classes, author in (('DEAP', 32, 2, 'faced'), ('SEEDIV', 62, 3, 'seedv')):
            name = 'models.model_for_'+author
            spec = importlib.util.spec_from_file_location(name, source/'models'/('model_for_'+author+'.py'))
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
            for head in HEADS[1:]:
                local = nonlinear_head(head, channels, classes)
                param = types.SimpleNamespace(use_pretrained_weights=False, num_of_classes=classes,
                    classifier='all_patch_reps_twolayer', dropout=.1)
                original = module.Model(param).classifier
                original[1] = nn.Linear(200 if head == 'pooled_mlp' else channels*10*200, 200)
                original[1].load_state_dict(local[1].state_dict())
                original[4].load_state_dict(local[4].state_dict())
                for training in (False, True):
                    local.train(training); original.train(training); local[3].update_index = 7
                    torch.manual_seed(20261009)
                    tokens = torch.randn(2, channels, 10, 200, requires_grad=True)
                    mirror = tokens.detach().clone().requires_grad_(True)
                    before = torch.get_rng_state().clone()
                    a = local(tokens)
                    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
                    supplied = mirror.mean((1, 2)).reshape(2, 1, 1, 200) if head == 'pooled_mlp' else mirror
                    torch.manual_seed(HEAD_DROPOUT_SEED+7)
                    b = original(supplied)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    a.sum().backward(); b.sum().backward()
                    torch.testing.assert_close(tokens.grad, mirror.grad, rtol=0, atol=0)
                    for ours, theirs in zip(local.parameters(), original.parameters()):
                        torch.testing.assert_close(ours.grad, theirs.grad, rtol=0, atol=0)
                    records.append({'dataset': dataset, 'head': head, 'training': training,
                        'parameter_count': sum(p.numel() for p in local.parameters()),
                        'outputs_and_token_parameter_gradients_exact': True,
                        'global_cpu_rng_preserved_by_local_dropout': True})
                    local.zero_grad(set_to_none=True); original.zero_grad(set_to_none=True)
                del local, original, tokens, mirror, a, b
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return {'synthetic_only': True, 'records': records, 'author_commit': old_proof['author_commit'],
        'author_sha256': old_proof['author_sha256'],
        'scope': 'Exact executed standalone two-layer operators/gradients/dropout masks. SEED input expands1to10patches; pooled control supplies mean tokens with200input. Premature wrapper flatten bypassed. No EEG or published score reproduction.'}


def read_record(root, output, job):
    folder = output/'runs'/identifier(job)
    record = json.loads((folder/'record.json').read_text())
    if record['job'] != job or record['plan_sha256'] != sha(output/'plan.json') or record['reused'] != is_anchor(job):
        raise ValueError('Wrong readout record binding')
    if [i['step'] for i in record['candidates']] != list(STEPS):
        raise ValueError('Missing readout states')
    for name, checksum in record['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Changed readout artifact')
    for item in record['candidates']:
        if sha(checkpoint_path(root, folder, record, item)) != item['checkpoint_sha256']:
            raise ValueError('Changed readout checkpoint')
    if record['reused']:
        source = root/PRIOR/'runs'/prior_identifier(anchor_job(job))
        original = json.loads((source/'record.json').read_text())
        if sha(source/'record.json') != record['source_record_sha256'] or sha(source/'verification.json') != record['source_verification_sha256']:
            raise ValueError('Changed anchor source proof')
        if original['candidates'] != record['candidates']:
            raise ValueError('Altered anchor outcomes')
        for name in record['artifact_sha256']:
            if sha(source/name) != sha(folder/name):
                raise ValueError('Anchor history/probability bytes differ')
    return folder, record


def audit_case(root, output, job):
    folder, record = read_record(root, output, job)
    table, idx, data = source_data(root, job)
    draws, windows, signature = sampling(table, idx['train'], updates=STEPS[-1])
    if signature != record['sampling_digest']:
        raise ValueError('Changed readout streams')
    check_exposure(record, idx['train'], draws, windows, STEPS, 'long')
    model = new_model(root, int(table.label.max()+1), job)
    if state_digest(model.backbone.state_dict()) != record['initial_encoder_digest'] or state_digest(model.head.state_dict()) != record['initial_head_digest']:
        raise ValueError('Changed readout initialization')
    schema = dropout_schema(model)
    if len(schema) != 61 or not any(i['p'] for i in schema):
        raise ValueError('Incorrect encoder dropout-on configuration')
    if not record['reused'] and schema != record['dropout_schema']:
        raise ValueError('Changed encoder dropout schema')
    if sum(p.numel() for p in model.head.parameters()) != record['head_parameters']:
        raise ValueError('Wrong readout parameter count')
    maximum = metric_maximum = 0.; sets = 0
    for item in record['candidates']:
        state = torch.load(checkpoint_path(root, folder, record, item), map_location='cpu', weights_only=True)
        if state_digest(state) != item['state_digest']:
            raise ValueError('Wrong readout tensor digest')
        model.load_state_dict(state, strict=True)
        encoder = state_digest(model.backbone.state_dict()); head = state_digest(model.head.state_dict())
        if encoder != item['encoder_digest'] or head != item['head_digest']:
            raise ValueError('Wrong component digest')
        if item['step'] == 0:
            if encoder != record['initial_encoder_digest'] or head != record['initial_head_digest']:
                raise ValueError('Wrong initial state')
        elif encoder == record['initial_encoder_digest'] or head == record['initial_head_digest']:
            raise ValueError('Fine-tuning failed to update encoder or head')
        for role, rows in idx.items():
            p = explicit_replay(model, data, rows)
            a, b = compare_csv(folder, item, role, table, rows, p, 2e-6)
            maximum = max(maximum, a); metric_maximum = max(metric_maximum, b); sets += 1
        if state_digest(model.state_dict()) != item['state_digest']:
            raise ValueError('Readout replay mutated state')
    proof = {'complete': True, 'record_sha256': sha(folder/'record.json'), 'states_checked': len(STEPS),
        'metric_sets': sets, 'max_probability_abs': maximum, 'max_metric_abs': metric_maximum,
        'reused_exact_anchor': record['reused'], 'head_parameters': record['head_parameters'],
        'canonical_inference_batch': 16, 'created_utc': stamp(),
        'scope': 'Strict complete-state restore and explicit token/functional linear-ELU-linear readout, not full optimizer replay.'}
    atomic(folder/'verification.json', proof)
    return proof


def finish(root, output):
    plan = validate(root, output, deep_inputs=True)
    records = []; proofs = []; selected = []
    for job in plan['jobs']:
        folder, record = read_record(root, output, job)
        proof = json.loads((folder/'verification.json').read_text())
        if not proof['complete'] or proof['record_sha256'] != sha(folder/'record.json'):
            raise ValueError('Incomplete readout proof')
        records.append(record); proofs.append(proof)
        choices = [i for i in record['candidates'] if i['step'] > 0]
        selected.append({'job': job, 'trajectory': folder.name, 'selected': choices[select(choices)], 'candidates': choices})
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            panel = [r for r in records if (r['job']['dataset'], r['job']['group']) == (dataset, group)]
            if len(panel) != 6 or len({r['sampling_digest'] for r in panel}) != 1:
                raise ValueError('Unmatched readout streams/coverage')
            for pretrained in (True, False):
                block = [r for r in panel if r['job']['pretrained'] == pretrained]
                if len({r['initial_encoder_digest'] for r in block}) != 1:
                    raise ValueError('Unmatched readout encoder initialization')
        for head in HEADS:
            block = [r for r in records if r['job']['dataset'] == dataset and r['job']['head'] == head]
            if len({r['initial_head_digest'] for r in block}) != 1:
                raise ValueError('Head initialization changes across groups/pretraining')
    if len(records) != 24 or sum(r['reused'] for r in records) != 8:
        raise ValueError('Incomplete readout grid')
    atomic(output/'summary.json', {'development_only': True, 'outer_test_inferences': 0,
        'research_question_change_approved': False, 'selected': selected,
        'primary_comparisons': 'Same-step head and pretraining contrasts; individual checkpoint selections secondary.'})
    proof = {'complete': True, 'plan_sha256': sha(output/'plan.json'), 'summary_sha256': sha(output/'summary.json'),
        'conditions': 24, 'new_trajectories': 16, 'reused_exact_anchors': 8,
        'states_checked': sum(p['states_checked'] for p in proofs), 'metric_sets': sum(p['metric_sets'] for p in proofs),
        'maximum_probability_abs': max(p['max_probability_abs'] for p in proofs),
        'maximum_metric_abs': max(p['max_metric_abs'] for p in proofs), 'created_utc': stamp()}
    atomic(output/'verification.json', proof)
    return proof
