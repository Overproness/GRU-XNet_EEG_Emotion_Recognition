"""Source-only moments, frozen affine buffers, exact anchors and explicit replay."""
import copy
import json
from pathlib import Path
import shutil
from uuid import uuid4
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from gruxnet.cbramod_adaptation import Adaptation
from gruxnet.cbramod_conditioning import (
    Conditioned, statistics, jobs, is_anchor, disable_capacity_dropout,
    sha, identifier, normalizer_id, PRIOR, previous_identifier, anchor_job,
)


@pytest.mark.parametrize('classes', (2, 3))
def test_statistics_exclude_validation_and_weight_all_windows_equally(classes):
    x = np.random.default_rng(classes).normal(size=(36, 4, 200)).astype(np.float32)
    x[:, :, -1] = 4
    mean, scale = statistics(x, np.arange(24))
    flat = x[:24].astype(np.float64).reshape(-1, 200)
    np.testing.assert_allclose(mean, flat.mean(0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(scale[:-1], flat.std(0)[:-1], rtol=0, atol=1e-14)
    assert scale[-1] == 1e-6
    x[24:] = 1e20
    second_mean, second_scale = statistics(x, np.arange(24))
    np.testing.assert_array_equal(mean, second_mean)
    np.testing.assert_array_equal(scale, second_scale)


def test_raw_wrapper_preserves_original_operator_path_and_state_layout():
    torch.manual_seed(42); backbone = nn.Linear(200, 200)
    old = Adaptation(copy.deepcopy(backbone), 2, False)
    new = Conditioned(copy.deepcopy(backbone), 2, False)
    assert set(old.state_dict()) == set(new.state_dict())
    x = torch.randn(6, 3, 10, 200)
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('trainable', (False, True))
def test_fixed_affine_buffers_and_encoder_update_permission(trainable):
    torch.manual_seed(42); backbone = nn.Linear(200, 200)
    mean = np.linspace(-1, 1, 200); scale = np.linspace(.1, 2, 200)
    model = Conditioned(backbone, 3, trainable, mean, scale)
    x = torch.randn(6, 3, 10, 200); y = torch.arange(6) % 3
    expected = nn.functional.linear((backbone(x).mean((1, 2))-model.embedding_mean)/model.embedding_scale,
                                   model.head.weight, model.head.bias)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    before = copy.deepcopy(backbone.state_dict()); initial_mean = model.embedding_mean.clone()
    assert not model.embedding_mean.requires_grad and not model.embedding_scale.requires_grad
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    nn.functional.cross_entropy(model(x), y).backward(); optimizer.step()
    assert (not torch.equal(before['weight'], backbone.weight)) == trainable
    torch.testing.assert_close(initial_mean, model.embedding_mean, rtol=0, atol=0)
    restored = Conditioned(nn.Linear(200, 200), 3, trainable, mean, scale)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_internal_attention_and_module_dropout_disable_together():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__(); self.attention = nn.MultiheadAttention(200, 4, dropout=.5, batch_first=True)
            self.dropout = nn.Dropout(.6)
        def forward(self, x):
            x = x[:, 0]; x = self.attention(x, x, x, need_weights=False)[0]
            return self.dropout(x).unsqueeze(1)
    model = Conditioned(Backbone(), 2, True, np.zeros(200), np.ones(200)).train()
    assert disable_capacity_dropout(model) == 2
    x = torch.randn(6, 3, 10, 200)
    torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)


def test_factorial_has_every_pair_and_exactly_sixteen_reuse_anchors():
    grid = jobs()
    assert len(grid) == 64 and sum(map(is_anchor, grid)) == 16
    assert len({identifier(j) for j in grid}) == 64
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            for pretrained in (True, False):
                for trainable in (False, True):
                    block = [j for j in grid if (j['dataset'], j['group'], j['pretrained'], j['trainable']) ==
                             (dataset, group, pretrained, trainable)]
                    assert {(j['scaled'], j['dropout']) for j in block} == {(False, False), (False, True), (True, False), (True, True)}


def fixture(monkeypatch, classes):
    import gruxnet.cbramod_conditioning as implementation
    import scripts.audit_cbramod_conditioning as audit
    data = np.random.default_rng(classes).normal(size=(36, 4, 3, 10, 200)).astype(np.float32)
    table = pd.DataFrame({'trial_id': [f'T{i}' for i in range(36)], 'subject_id': [f'S{i//4}' for i in range(36)],
        'material_key': [f'V{i%6}' if i < 24 or i >= 30 else f'U{i%6}' for i in range(36)],
        'original_label': np.arange(36) % classes, 'label': np.arange(36) % classes})
    idx = {'train': np.arange(24), 'validation_unseen': np.arange(24, 30), 'validation_familiar': np.arange(30, 36)}
    def source(*args): return table, idx, data
    def previous(*args):
        torch.manual_seed(42)
        return Adaptation(nn.Linear(200, 200), classes, False)
    def model(root, categories, job, mean=None, scale=None):
        torch.manual_seed(42)
        return Conditioned(nn.Linear(200, 200), categories, job['trainable'], mean, scale)
    for module in (implementation, audit):
        monkeypatch.setattr(module, 'source_data', source)
        monkeypatch.setattr(module, 'previous_model', previous)
        monkeypatch.setattr(module, 'new_model', model)
        monkeypatch.setattr(module, 'STEPS', (0, 2, 4, 6))
    root = Path(__file__).resolve().parents[2]/'publication_runs'/f'cbramod_conditioning_test_{uuid4().hex}'
    output = root/'new'; output.mkdir(parents=True); (output/'plan.json').write_text('{}\n')
    job = dict(dataset='DEAP' if classes == 2 else 'SEEDIV', group=1, pretrained=True,
               trainable=True, scaled=True, dropout=False)
    return implementation, audit, root, output, job


@pytest.mark.parametrize('classes', (2, 3))
def test_complete_conditioned_fit_and_independent_readout_replay(classes, monkeypatch):
    implementation, audit, root, output, job = fixture(monkeypatch, classes)
    stat_job = {k: job[k] for k in ('dataset', 'group', 'pretrained')}
    implementation.prepare_normalizer(root, output, stat_job)
    stat_proof = audit.audit_normalizer(root, output, stat_job)
    assert stat_proof['source_windows'] == 96 and not stat_proof['validation_statistics_accessed']
    implementation.fit(root, output, job)
    proof = audit.audit_case(root, output, job)
    assert proof['states_checked'] == 4 and proof['metric_sets'] == 12
    assert proof['max_probability_abs'] < 1e-12 and proof['complete']
    # A modified scaler fails its independent binding before model replay.
    with (output/'normalizers'/normalizer_id(job)/'statistics.npz').open('ab') as stream:
        stream.write(b'altered')
    with pytest.raises(ValueError, match='normalizer binding'):
        audit.audit_case(root, output, job)


def test_anchor_reuses_original_arrays_and_probability_bytes(monkeypatch):
    implementation, audit, root, output, job = fixture(monkeypatch, 2)
    job.update(scaled=False, dropout=False)
    implementation.fit(root, output, job)
    folder = output/'runs'/identifier(job); generated = json.loads((folder/'record.json').read_text())
    original_output = root/PRIOR; original_output.mkdir(); (original_output/'plan.json').write_text('{}\n')
    original = original_output/'long'/previous_identifier('long', anchor_job(job)); original.mkdir(parents=True)
    for path in folder.iterdir():
        if path.name != 'record.json': shutil.copyfile(path, original/path.name)
    record = {**generated, 'job': anchor_job(job), 'plan_sha256': sha(original_output/'plan.json')}
    (original/'record.json').write_text(json.dumps(record))
    (original/'verification.json').write_text(json.dumps({'complete': True, 'record_sha256': sha(original/'record.json')}))
    import scripts.audit_cbramod_learning as old_audit
    monkeypatch.setitem(old_audit.STEPS, 'long', (0, 2, 4, 6))
    job['dropout'] = True
    implementation.import_anchor(root, output, job)
    proof = audit.audit_case(root, output, job)
    imported = output/'runs'/identifier(job)
    assert proof['reused_exact_anchor'] and proof['states_checked'] == 4
    assert not list(imported.glob('*.pt'))
    assert sha(imported/'step6_train.csv') == sha(original/'step6_train.csv')
    with (imported/'step6_train.csv').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'): audit.audit_case(root, output, job)
