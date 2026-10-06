"""Verify every tuning candidate; replay selected states before bounded deletion."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.data import sha256, digest
from gruxnet.heldout_tuning import (STUDY, RATES, STEPS, NORMS, C_VALUES, case_id, context_id,
    partitions, choose, normalizer, make_model, population, keep_checkpoint, atomic_json, stamp)
from gruxnet.learning_controls import draw_stream
from gruxnet.full_context_controls_v2 import prior
from gruxnet.material_controls import state_digest
from gruxnet.train import seed_everything


def numerical_metrics(rows, dataset):
    neural = 'model' in rows and set(rows.model).issubset({'gru','eegnet','eegnet_context'})
    p = rows[[c for c in rows if c.startswith('p_')]].to_numpy(dtype=np.float32 if neural else float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any(): raise ValueError('Invalid probability')
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    tasks = {'coarse3' if dataset == 'SEEDIV' else 'binary': (rows.label.to_numpy(dtype=int), p)}
    if dataset == 'SEEDIV':
        keep = rows.original_label.ne(0).to_numpy()
        positive = p[keep,2]/np.maximum(p[keep,1:].sum(1), 1e-12)
        tasks['binary'] = ((rows.original_label.to_numpy()[keep] == 3).astype(int), np.stack([1-positive, positive], 1))
    result = {}
    for task, (y,z) in tasks.items():
        predicted = z.argmax(1); losses = -np.log(np.clip(z[np.arange(len(y)), y], 1e-12, 1))
        result[task] = {'n': len(y), 'accuracy': float(accuracy_score(y,predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(y,predicted)),
            'balanced_log_loss': float(np.mean([losses[y == c].mean() for c in range(z.shape[1])])),
            'macro_f1': float(f1_score(y,predicted,labels=range(z.shape[1]),average='macro',zero_division=0)),
            'confusion_matrix': confusion_matrix(y,predicted,labels=range(z.shape[1])).tolist()}
    return result


def compare(actual, expected):
    if actual.keys() != expected.keys(): raise ValueError('Changed metric tasks')
    for task in actual:
        for field, value in actual[task].items():
            if field in ('n','confusion_matrix'):
                if value != expected[task][field]: raise ValueError('Changed population/confusion')
            else: np.testing.assert_allclose(value, expected[task][field], atol=2e-6, rtol=0)


def validate_rows(folder, filename, split, job, expected=None):
    rows = pd.read_csv(folder/filename)
    if rows.trial_id.tolist() != split: raise ValueError('Prediction row population differs from seal')
    for field, value in (('model', job.get('model')), ('arm',job['arm']), ('group',job['group']),
                         ('source_session',job['session']), ('material_rotation',job['rotation']),('fold',job['fold'])):
        if value is not None and not rows[field].eq(value).all(): raise ValueError('Prediction cell identity mismatch')
    actual = numerical_metrics(rows,job['dataset'])
    if expected is not None: compare(actual,expected)
    return rows


def inspect_case(folder, output, context=False):
    r = json.loads((folder/'record.json').read_text()); s = json.loads((folder/'selection.json').read_text())
    if r['config_sha256'] != sha256(output/'config.json') or s['config_sha256'] != r['config_sha256']:
        raise ValueError('Case configuration changed')
    if r['selection_sha256'] != sha256(folder/'selection.json'): raise ValueError('Selection seal changed')
    if r['job'] != s['job']: raise ValueError('Selected case identity changed')
    for source in (r,s):
        for name, expected in source['artifact_sha256'].items():
            if sha256(folder/name) != expected: raise ValueError(f'Changed evidence {folder.name}/{name}')
    candidates = s['candidates']; dataset = s['job']['dataset']
    expected_ids = [str(c) for c in C_VALUES] if context else [f'lr{lr}_step{step}_{norm}' for lr in RATES for step in STEPS for norm in NORMS]
    if [c['id'] for c in candidates] != expected_ids: raise ValueError('Incomplete/reordered candidate grid')
    if choose(candidates,dataset)['id'] != s['selected']: raise ValueError('Source-only selection rule changed')
    for candidate in candidates:
        for panel in ('validation_unseen','validation_familiar'):
            name = (f'candidate_C{candidate["C"]}_{panel}.csv' if context else f'candidate_{candidate["id"]}_{panel}.csv')
            validate_rows(folder,name,s['split_trials'][panel],s['job'],candidate['validation'][panel])
    for role, split in s['split_trials'].items():
        if context:
            for model in ('prior','context_logistic'):
                validate_rows(folder,f'predictions_{model}_{role}.csv',split,dict(s['job'],model=model),r['metrics'][role][model])
        else:
            validate_rows(folder,f'predictions_{role}.csv',split,s['job'],r['metrics'][role])
    if not context:
        source = set().union(*(set(v) for k,v in s['split_trials'].items() if k != 'test'))
        if set(s['source_accessed_trials']) != source or source & set(s['split_trials']['test']):
            raise ValueError('Outer-test inputs entered source tuning')
        if r['checkpoint_retained'] != keep_checkpoint(r['job']): raise ValueError('Retention protocol changed')
        expected_prefix = s['job']['model'] != 'eegnet_context' and s['job']['session'] == 1 and s['job']['rotation'] == 0 and s['job']['fold'] == 0
        if len(s['prefix_replay']) != (2 if expected_prefix else 0): raise ValueError('Missing expected source prefix guards')
        for guard in s['prefix_replay']:
            candidate=next(c for c in candidates if c['lr']==guard['lr'] and c['step']==200 and c['normalization']=='ema')
            if guard['state_digest']!=candidate['state_digest'] or not guard['exact']: raise ValueError('Invalid prefix state guard')
    return r,s


@torch.no_grad()
def fresh_predict(model, values, mean, scale, q, device):
    model.eval(); result = []
    for start in range(0,len(values),12):
        inputs = torch.from_numpy(((values[start:start+12]-mean)/scale).astype(np.float32)).to(device)
        context = torch.tensor(np.log(q[start:start+12]).astype(np.float32),device=device)
        result.append(torch.softmax(model(inputs,context),1).cpu().numpy())
    return np.concatenate(result)


def replay_case(folder, output, base, x=None, wave=None, device='cuda', context=False):
    r,s = inspect_case(folder,output,context)
    job = s['job']; table,idx = partitions(base,dict(job,model=job.get('model','context_logistic'),initialization=job.get('initialization',42)))
    split = {k:table.iloc[v].trial_id.tolist() for k,v in idx.items()}
    if split != s['split_trials']: raise ValueError('Partition reconstruction failed')
    classes = 3 if job['dataset'] == 'SEEDIV' else 2
    q = {role:prior(table,idx['train'],rows,classes,crossfit=role == 'train') for role,rows in idx.items()}
    maximum = 0.
    if context:
        scaler = StandardScaler().fit(np.log(q['train']))
        np.testing.assert_array_equal(scaler.mean_,s['scaler_mean']); np.testing.assert_array_equal(scaler.scale_,s['scaler_scale'])
        models = []
        for candidate in s['candidates']:
            model = LogisticRegression(C=candidate['C'],class_weight='balanced',max_iter=2000,tol=1e-6,random_state=42).fit(scaler.transform(np.log(q['train'])),table.iloc[idx['train']].label)
            if int(model.n_iter_.max()) >= 2000: raise ValueError('Refit logistic failed to converge')
            models.append(model)
            for role in ('validation_unseen','validation_familiar'):
                stored = pd.read_csv(folder/f'candidate_C{candidate["C"]}_{role}.csv').filter(regex='^p_').to_numpy()
                p = model.predict_proba(scaler.transform(np.log(q[role])))
                np.testing.assert_allclose(p,stored,atol=1e-8,rtol=0)
        model = models[next(i for i,c in enumerate(s['candidates']) if c['id'] == s['selected'])]
        np.testing.assert_array_equal(model.coef_,s['coef']); np.testing.assert_array_equal(model.intercept_,s['intercept'])
        for role in idx:
            for name,p in (('prior',q[role]),('context_logistic',model.predict_proba(scaler.transform(np.log(q[role]))))):
                saved = pd.read_csv(folder/f'predictions_{name}_{role}.csv').filter(regex='^p_').to_numpy()
                maximum = max(maximum,float(np.max(np.abs(p-saved))))
                np.testing.assert_allclose(p,saved,atol=1e-8,rtol=0)
    else:
        path = folder/'selected.pt'
        if sha256(path) != s['checkpoint_sha256'] or r['checkpoint_sha256'] != s['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        state = torch.load(path,map_location='cpu',weights_only=False)
        values = x if job['model'] == 'gru' else wave
        mean,scale = normalizer(values,idx['train'],job['model'] != 'gru')
        np.testing.assert_array_equal(state['mean'],mean); np.testing.assert_array_equal(state['scale'],scale)
        init = job['initialization']+1000*job['fold']+10000*job['rotation']
        seed_everything(init); model = make_model(job['model'],classes).to(device)
        if state_digest(model.state_dict()) != s['initial_state_digest']: raise ValueError('Initial state mismatch')
        sampled,signature = draw_stream(table,idx['train'],init,classes,STEPS[-1])
        if signature != s['canonical_draw_signature'] or digest(sampled.tolist()) != s['batch_digest']: raise ValueError('Training streams changed')
        model.load_state_dict(state['state'])
        if state_digest(model.state_dict()) != s['selected_state_digest']: raise ValueError('Selected state digest mismatch')
        chosen = next(c for c in s['candidates'] if c['id'] == s['selected'])
        if chosen['normalization'] == 'source_population':
            training = torch.from_numpy(((values[idx['train']]-mean)/scale).astype(np.float32)).to(device)
            population(model,training); del training
            if state_digest(model.state_dict()) != s['selected_state_digest']: raise ValueError('Source population BN did not reconstruct exactly')
        for role,rows in idx.items():
            p = fresh_predict(model,values[rows],mean,scale,q[role],device)
            saved = pd.read_csv(folder/f'predictions_{role}.csv').filter(regex='^p_').to_numpy()
            maximum = max(maximum,float(np.max(np.abs(p-saved))))
            np.testing.assert_allclose(p,saved,atol=1e-6,rtol=0)
        for role in ('validation_unseen','validation_familiar'):
            selected = pd.read_csv(folder/f'predictions_{role}.csv').filter(regex='^p_').to_numpy()
            candidate = pd.read_csv(folder/f'candidate_{s["selected"]}_{role}.csv').filter(regex='^p_').to_numpy()
            np.testing.assert_allclose(selected,candidate,atol=1e-6,rtol=0)
        del model,state
    certificate = {'passed':True,'created_utc':stamp(),'record_sha256':sha256(folder/'record.json'),
                   'selection_sha256':sha256(folder/'selection.json'),'config_sha256':sha256(output/'config.json'),
                   'maximum_probability_error':maximum,'scope':'All candidate metrics recomputed; source-only choice and splits checked; selected predictions replayed from a fresh model. Context models independently refit for all C values; selected source population moments reconstructed if applicable. Not a full neural optimization rerun.',
                   'checkpoint_sha256':r.get('checkpoint_sha256'),'checkpoint_retained':r.get('checkpoint_retained')}
    atomic_json(folder/'certificate.json',certificate)
    if not context and not r['checkpoint_retained']:
        # Delete only this newly produced, verified checkpoint, within this case.
        path = (folder/'selected.pt').resolve()
        if path.parent != folder.resolve() or not path.is_relative_to((output/'fits').resolve()): raise ValueError('Unsafe checkpoint deletion path')
        if sha256(path) != r['checkpoint_sha256']: raise ValueError('Checkpoint changed before bounded deletion')
        path.unlink()
    return certificate


def certificate_valid(folder, output, context=False):
    r,s = inspect_case(folder,output,context)
    c = json.loads((folder/'certificate.json').read_text())
    if not c['passed'] or c['record_sha256'] != sha256(folder/'record.json') or c['selection_sha256'] != sha256(folder/'selection.json') or c['config_sha256'] != sha256(output/'config.json'):
        raise ValueError('Invalid immediate replay certificate')
    if not context:
        if c['checkpoint_sha256'] != r['checkpoint_sha256'] or c['checkpoint_retained'] != r['checkpoint_retained']: raise ValueError('Checkpoint certificate mismatch')
        if r['checkpoint_retained'] and sha256(folder/'selected.pt') != r['checkpoint_sha256']: raise ValueError('Retained checkpoint changed')
    return r


def audit(output, partial=False, public=False):
    plan = json.loads((output/'plan.json').read_text()); config = json.loads((output/'config.json').read_text())
    if config['plan_sha256'] != sha256(output/'plan.json'): raise ValueError('Changed study declaration')
    for name,expected in plan['source_sha256'].items():
        if sha256(REPO/name) != expected: raise ValueError(f'Changed frozen source {name}')
    count = 0; missing = []; maximum = 0.; prefixes = 0
    for context,jobs,kind in ((False,plan['jobs'],'fits'),(True,plan['contexts'],'contexts')):
        for job in jobs:
            folder = output/kind/(context_id(job) if context else case_id(job))
            if not (folder/'certificate.json').is_file(): missing.append(folder.name); continue
            if public:
                r,s = inspect_case(folder,output,context)
                c = json.loads((folder/'certificate.json').read_text())
                if not c['passed'] or c['record_sha256'] != sha256(folder/'record.json') or c['selection_sha256'] != sha256(folder/'selection.json') or c['config_sha256'] != sha256(output/'config.json'):
                    raise ValueError('Changed public replay certificate')
            else:
                r = certificate_valid(folder,output,context); s = json.loads((folder/'selection.json').read_text()); c = json.loads((folder/'certificate.json').read_text())
            count += 1; prefixes += len(s.get('prefix_replay',[])); maximum = max(maximum,c['maximum_probability_error'])
    if missing and not partial: raise ValueError(f'Incomplete study: {len(missing)} cases missing')
    if not missing and prefixes != 64: raise ValueError('Incomplete exact prior source prefix audit')
    result = {'passed':True,'complete':not missing,'verified_cases':count,'missing_cases':len(missing),
              'exact_prior_200_prefixes':prefixes,'maximum_probability_error':maximum,'created_utc':stamp(),
              'scope':'Portable candidate/selection/probability audit and immediate replay certificate integrity; retained weights additionally hash checked locally. Certificates attest replay performed before declared checkpoint deletion; public data alone cannot replay raw EEG or optimization.'}
    atomic_json(output/('public_verification.json' if public else 'verification.json'),result)
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=REPO.parent/'publication_runs'/STUDY)
    parser.add_argument('--partial',action='store_true'); parser.add_argument('--public',action='store_true')
    args = parser.parse_args(); print(json.dumps(audit(args.output.resolve(),args.partial,args.public)))
