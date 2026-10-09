"""Readout information geometry, RNG isolation, gradients and full-state checks."""
import copy
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from gruxnet.cbramod_readout import Readout, IsolatedDropout, HEAD_DROPOUT_SEED
from scripts.audit_cbramod_readout import functional_readout


def test_head_dropout_preserves_global_rng_and_matches_author_operator():
    x = torch.ones(6, 200, requires_grad=True)
    module = IsolatedDropout(.1).train(); module.update_index = 8
    torch.manual_seed(17); before = torch.get_rng_state().clone()
    actual = module(x)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    torch.manual_seed(HEAD_DROPOUT_SEED+8)
    reference = nn.functional.dropout(x, .1, True)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    module.update_index = 9
    assert not torch.equal(actual, module(x))
    module.eval()
    torch.testing.assert_close(module(x), x, rtol=0, atol=0)


@pytest.mark.parametrize('head', ('pooled_mlp', 'flattened_mlp'))
def test_independent_functional_forward_and_gradients(head):
    model = Readout(nn.Identity(), 3, head, 2).eval()
    tokens = torch.arange(2*2*10*200, dtype=torch.float32).reshape(2, 2, 10, 200)/4000
    tokens.requires_grad_(True)
    direct = model(tokens); independent = functional_readout(model, tokens)
    torch.testing.assert_close(direct, independent, rtol=0, atol=0)
    a = torch.autograd.grad(direct.sum(), tokens, retain_graph=True)[0]
    b = torch.autograd.grad(independent.sum(), tokens)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_flattened_head_retains_token_position_discarded_by_pooled_heads():
    tokens = torch.zeros(2, 2, 10, 200)
    # Twenty tokens make the pooled nonzero coordinate exactly one in FP32.
    tokens[0, 0, 0, 0] = 20; tokens[1, 1, 0, 0] = 20
    pooled = Readout(nn.Identity(), 2, 'pooled_mlp', 2).eval()
    linear = Readout(nn.Identity(), 2, 'pooled_linear', 2).eval()
    flat = Readout(nn.Identity(), 2, 'flattened_mlp', 2).eval()
    with torch.no_grad():
        flat.head[1].weight.zero_(); flat.head[1].bias.zero_()
        flat.head[1].weight[0, 0] = 1
        flat.head[4].weight.zero_(); flat.head[4].bias.zero_(); flat.head[4].weight[0, 0] = 1
    torch.testing.assert_close(tokens.mean((1, 2))[0], tokens.mean((1, 2))[1], rtol=0, atol=0)
    # Use identical batch positions; CPU GEMM rows can differ by a few FP32 ulps.
    torch.testing.assert_close(pooled(tokens[:1]), pooled(tokens[1:]), rtol=0, atol=0)
    torch.testing.assert_close(linear(tokens[:1]), linear(tokens[1:]), rtol=0, atol=0)
    assert flat(tokens)[0, 0] == 20 and flat(tokens)[1, 0] == 0


@pytest.mark.parametrize('head', ('pooled_mlp', 'flattened_mlp'))
@pytest.mark.parametrize('classes', (2, 3))
def test_complete_toy_fit_and_independent_state_replay(head, classes, monkeypatch):
    import gruxnet.cbramod_readout as implementation
    import scripts.audit_cbramod_readout as audit
    data = np.random.default_rng(classes).normal(size=(36, 4, 2, 10, 200)).astype(np.float32)
    table = pd.DataFrame({'trial_id': [f'T{i}' for i in range(36)], 'subject_id': [f'S{i//4}' for i in range(36)],
        'material_key': [f'V{i%6}' if i < 24 or i >= 30 else f'U{i%6}' for i in range(36)],
        'original_label': np.arange(36) % classes, 'label': np.arange(36) % classes})
    idx = {'train': np.arange(24), 'validation_unseen': np.arange(24, 30), 'validation_familiar': np.arange(30, 36)}
    def source(*args):
        return table, idx, data
    def create(root, categories, job):
        torch.manual_seed(42)
        return Readout(nn.Linear(200, 200), categories, job['head'], 2)
    for module in (implementation, audit):
        monkeypatch.setattr(module, 'source_data', source)
        monkeypatch.setattr(module, 'new_model', create)
        monkeypatch.setattr(module, 'STEPS', (0, 2, 4, 6))
    monkeypatch.setattr(audit, 'dropout_schema', lambda model: [{'p': .1}]*61)
    monkeypatch.setattr(implementation, 'dropout_schema', lambda model: [{'p': .1}]*61)
    root = Path(__file__).resolve().parents[2]/'publication_runs'/f'cbramod_readout_test_{uuid4().hex}'
    output = root/'new'; output.mkdir(parents=True); (output/'plan.json').write_text('{}\n')
    job = dict(dataset='DEAP' if classes == 2 else 'SEEDIV', group=1, pretrained=True, head=head)
    implementation.fit(root, output, job)
    proof = audit.audit_case(root, output, job)
    assert proof['states_checked'] == 4 and proof['metric_sets'] == 12 and proof['max_probability_abs'] < 1e-12
    folder = output/'runs'/implementation.identifier(job)
    state = torch.load(folder/'checkpoint6.pt', weights_only=True)
    state['head.4.bias'][0] += 1
    torch.save(state, folder/'checkpoint6.pt')
    with pytest.raises(ValueError, match='artifact|checkpoint'):
        audit.audit_case(root, output, job)
