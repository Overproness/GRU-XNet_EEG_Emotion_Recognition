"""Independent source normalizers, explicit readout replay and matched selections."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.cbramod_conditioning import (
    PRIOR, STUDY, STEPS, SCALE_FLOOR, ROLES, identifier, normalizer_id, anchor_job,
    is_anchor, new_model, source_data, sampling, sha, state_digest, atomic, stamp,
    checkpoint_path, previous_identifier, validate,
)
from gruxnet.cbramod_adaptation import new_model as previous_model
from scripts.audit_cbramod_learning import compare_csv, check_exposure, select


def read_normalizer(output, job):
    folder = output/'normalizers'/normalizer_id(job)
    record = json.loads((folder/'record.json').read_text())
    if record['job'] != {k: job[k] for k in ('dataset', 'group', 'pretrained')}:
        raise ValueError('Wrong source-normalizer condition')
    if record['plan_sha256'] != sha(output/'plan.json') or sha(folder/'statistics.npz') != record['statistics_sha256']:
        raise ValueError('Changed normalizer binding')
    with np.load(folder/'statistics.npz', allow_pickle=False) as state:
        arrays = {k: state[k].copy() for k in state.files}
    return folder, record, arrays


def audit_normalizer(root, output, job):
    folder, record, state = read_normalizer(output, job)
    table, idx, data = source_data(root, job)
    if record['training_trial_ids'] != table.iloc[idx['train']].trial_id.tolist():
        raise ValueError('Normalizer training membership mismatch')
    if set(record['training_trial_ids']) & set(table.iloc[np.r_[idx[ROLES[1]], idx[ROLES[2]]]].trial_id):
        raise ValueError('Validation trial used for conditioning')
    model = previous_model(root, int(table.label.max()+1), job['pretrained'], False).eval()
    if state_digest(model.backbone.state_dict()) != record['initial_encoder_digest']:
        raise ValueError('Different initial encoder for normalizer')
    source = data[idx['train']].reshape(len(idx['train'])*4, data.shape[2], 10, 200)
    device = next(model.parameters()).device

    def extract(batch):
        chunks = []
        with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
            for start in range(0, len(source), batch):
                tokens = model.backbone(torch.tensor(source[start:start+batch], device=device))
                chunks.append(tokens.mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(chunks).reshape(-1, 4, 200)

    canonical = extract(16)
    np.testing.assert_array_equal(state['features'], canonical)
    scaler = StandardScaler().fit(canonical.astype(np.float64).reshape(-1, 200))
    mean = scaler.mean_; scale = np.maximum(np.sqrt(scaler.var_), SCALE_FLOOR)
    np.testing.assert_allclose(state['mean64'], mean, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state['scale64'], scale, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(state['mean32'], state['mean64'].astype(np.float32))
    np.testing.assert_array_equal(state['scale32'], state['scale64'].astype(np.float32))
    alternate = extract(8)
    if not np.isfinite(alternate).all() or state_digest(model.backbone.state_dict()) != record['initial_encoder_digest']:
        raise ValueError('Nonfinite or mutated initial-feature replay')
    proof = {'complete': True, 'record_sha256': sha(folder/'record.json'), 'source_trials': len(idx['train']),
        'source_windows': len(canonical)*4, 'validation_statistics_accessed': False,
        'fixed_batch_features_exact': True, 'independent_scaler_max_abs': float(max(
            np.max(np.abs(mean-state['mean64'])), np.max(np.abs(scale-state['scale64'])))),
        'batch8_feature_max_abs': float(np.max(np.abs(alternate-canonical))),
        'batch8_normalized_feature_max_abs': float(np.max(np.abs((alternate.astype(np.float64)-canonical)/state['scale64']))),
        'scope': 'Independent source-only moments and exact canonical-batch feature replay; alternate-batch diagnostic, not an invariance claim.',
        'created_utc': stamp()}
    atomic(folder/'verification.json', proof)
    return proof


def read_record(root, output, job):
    folder = output/'runs'/identifier(job)
    record = json.loads((folder/'record.json').read_text())
    if record['job'] != job or record['plan_sha256'] != sha(output/'plan.json') or record['reused'] != is_anchor(job):
        raise ValueError('Wrong conditioning case binding')
    if [i['step'] for i in record['candidates']] != list(STEPS):
        raise ValueError('Missing conditioning states')
    for name, checksum in record['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Conditioning artifact changed')
    for item in record['candidates']:
        if sha(checkpoint_path(root, folder, record, item)) != item['checkpoint_sha256']:
            raise ValueError('Changed conditioning state bytes')
    if record['reused']:
        source = root/PRIOR/'long'/previous_identifier('long', anchor_job(job))
        original = json.loads((source/'record.json').read_text())
        if sha(source/'record.json') != record['source_record_sha256'] or sha(source/'verification.json') != record['source_verification_sha256']:
            raise ValueError('Changed anchor proof')
        if original['candidates'] != record['candidates']:
            raise ValueError('Anchor candidates changed')
        for name in record['artifact_sha256']:
            if sha(source/name) != sha(folder/name):
                raise ValueError('Anchor CSV/history is not byte exact')
    return folder, record


def explicit_replay(model, data, rows, scaled):
    """Bypass the fitting wrapper; preserve canonical batch and precision/order."""
    x = data[rows].reshape(len(rows)*4, data.shape[2], 10, 200)
    device = next(model.parameters()).device
    chunks = []; model.eval()
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        for start in range(0, len(x), 16):
            tokens = model.backbone(torch.tensor(x[start:start+16], device=device))
            features = torch.mean(tokens, dim=(1, 2))
            if scaled:
                features = torch.div(torch.sub(features, model.embedding_mean), model.embedding_scale)
            logits = nn.functional.linear(features, model.head.weight, model.head.bias)
            chunks.append(logits.cpu().numpy())
    logits = np.concatenate(chunks).reshape(len(rows), 4, -1).astype(np.float64)
    return softmax(logits.mean(axis=1), axis=1)


def audit_case(root, output, job):
    folder, record = read_record(root, output, job)
    table, idx, data = source_data(root, job)
    draws, windows, signature = sampling(table, idx['train'], updates=STEPS[-1])
    if signature != record['sampling_digest']:
        raise ValueError('Changed conditioning observation/window stream')
    check_exposure(record, idx['train'], draws, windows, STEPS, 'long')
    mean = scale = None
    if job['scaled']:
        stat_folder, stat_record, state = read_normalizer(output, job)
        proof = json.loads((stat_folder/'verification.json').read_text())
        if not proof['complete'] or proof['record_sha256'] != sha(stat_folder/'record.json') or sha(stat_folder/'record.json') != record['normalizer_record_sha256']:
            raise ValueError('Unverified/changed source conditioner')
        mean, scale = state['mean32'], state['scale32']
    model = new_model(root, int(table.label.max()+1), job, mean, scale)
    if state_digest(model.backbone.state_dict()) != record['initial_encoder_digest'] or state_digest(model.head.state_dict()) != record['initial_head_digest']:
        raise ValueError('Changed paired initialization')
    dropout = [{'name': name, 'type': type(m).__name__, 'p': float(m.p if isinstance(m, nn.Dropout) else m.dropout)}
        for name, m in model.backbone.named_modules() if isinstance(m, (nn.Dropout, nn.MultiheadAttention))]
    if not record['reused'] and dropout != record['dropout_schema']:
        raise ValueError('Wrong encoder dropout schema')
    if not job['dropout'] and any(i['p'] for i in dropout):
        raise ValueError('Internal/module encoder dropout was not disabled')
    maximum = metric_maximum = 0.; sets = 0
    for item in record['candidates']:
        state = torch.load(checkpoint_path(root, folder, record, item), map_location='cpu', weights_only=True)
        if state_digest(state) != item['state_digest']:
            raise ValueError('Wrong checkpoint tensor digest')
        model.load_state_dict(state, strict=True)
        encoder = state_digest(model.backbone.state_dict()); head = state_digest(model.head.state_dict())
        if encoder != item['encoder_digest'] or head != item['head_digest']:
            raise ValueError('Changed conditioning module certificate')
        if job['scaled']:
            np.testing.assert_array_equal(model.embedding_mean.cpu().numpy(), mean)
            np.testing.assert_array_equal(model.embedding_scale.cpu().numpy(), scale)
        if item['step'] == 0:
            if encoder != record['initial_encoder_digest'] or head != record['initial_head_digest']:
                raise ValueError('Wrong initial checkpoint')
        else:
            if (encoder != record['initial_encoder_digest']) != job['trainable']:
                raise ValueError('Frozen/updated encoder invariant failed')
            if head == record['initial_head_digest']:
                raise ValueError('Head never updated')
        for role, rows in idx.items():
            p = explicit_replay(model, data, rows, job['scaled'])
            a, b = compare_csv(folder, item, role, table, rows, p, 2e-6)
            maximum = max(maximum, a); metric_maximum = max(metric_maximum, b); sets += 1
        if state_digest(model.state_dict()) != item['state_digest']:
            raise ValueError('Explicit replay mutated checkpoint')
    proof = {'complete': True, 'record_sha256': sha(folder/'record.json'), 'states_checked': len(STEPS),
        'metric_sets': sets, 'max_probability_abs': maximum, 'max_metric_abs': metric_maximum,
        'canonical_inference_batch': 16, 'readout_replay': 'Independent explicit pooling/affine/functional_linear; same batch/precision.',
        'reused_exact_anchor': record['reused'], 'encoder_dropout_sites': len(dropout), 'created_utc': stamp()}
    atomic(folder/'verification.json', proof)
    return proof


def finish(root, output):
    plan = validate(root, output, deep_inputs=True)
    records = []; proofs = []; normalizers = []
    for job in plan['normalizer_jobs']:
        folder, record, state = read_normalizer(output, job)
        proof = json.loads((folder/'verification.json').read_text())
        if not proof['complete'] or proof['record_sha256'] != sha(folder/'record.json'):
            raise ValueError('Incomplete source-normalizer proof')
        normalizers.append(proof)
    selected = []
    for job in plan['jobs']:
        folder, record = read_record(root, output, job)
        proof = json.loads((folder/'verification.json').read_text())
        if not proof['complete'] or proof['record_sha256'] != sha(folder/'record.json'):
            raise ValueError('Incomplete conditioning case proof')
        records.append(record); proofs.append(proof)
        choices = [i for i in record['candidates'] if i['step'] > 0]
        selected.append({'job': job, 'trajectory': folder.name, 'selected': choices[select(choices)], 'candidates': choices})
    for dataset in ('DEAP', 'SEEDIV'):
        for group in (1, 2):
            panel = [r for r in records if (r['job']['dataset'], r['job']['group']) == (dataset, group)]
            if len({r['initial_head_digest'] for r in panel}) != 1 or len({r['sampling_digest'] for r in panel}) != 1:
                raise ValueError('Unmatched head initialization or training streams')
            for pretrained in (True, False):
                block = [r for r in panel if r['job']['pretrained'] == pretrained]
                if len({r['initial_encoder_digest'] for r in block}) != 1:
                    raise ValueError('Unmatched encoder initialization')
    if len(records) != 64 or sum(r['reused'] for r in records) != 16:
        raise ValueError('Incomplete factorial/anchor coverage')
    atomic(output/'summary.json', {'development_only': True, 'outer_test_inferences': 0,
        'research_question_change_approved': False, 'selected': selected,
        'primary_comparisons': 'Same-step normalization/dropout effects and interaction; checkpoint-selected outcomes are secondary.'})
    proof = {'complete': True, 'plan_sha256': sha(output/'plan.json'), 'summary_sha256': sha(output/'summary.json'),
        'conditions': 64, 'new_trajectories': 48, 'reused_exact_anchors': 16, 'normalizers': 8,
        'states_checked': sum(p['states_checked'] for p in proofs), 'metric_sets': sum(p['metric_sets'] for p in proofs),
        'maximum_probability_abs': max(p['max_probability_abs'] for p in proofs),
        'maximum_metric_abs': max(p['max_metric_abs'] for p in proofs),
        'maximum_normalized_feature_batch_sensitivity': max(p['batch8_normalized_feature_max_abs'] for p in normalizers),
        'created_utc': stamp()}
    atomic(output/'verification.json', proof)
    return proof
