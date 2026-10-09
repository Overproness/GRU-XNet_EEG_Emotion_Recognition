"""Declared source-only baseline-alone control after the completed diagnostic."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.preprocessing_diagnostic_v2 import (STUDY as PARENT, SOURCES, ROLES,
    atomic, sha, stamp, load_panel, logpower, metric, selection_key, save_predictions)
from gruxnet.data import COMMON_CHANNELS
from gruxnet.train import seed_everything
from scripts.audit_preprocessing_diagnostic_v2 import independent

STUDY = 'prestimulus_control_2026-10-09'


def declare(root):
    output = root/STUDY
    if output.exists(): raise FileExistsError('Preserve declaration')
    parent = root/PARENT
    if not json.loads((parent/'verification.json').read_text()).get('complete'):
        raise ValueError('Complete parent required')
    output.mkdir()
    plan = {'created_utc': stamp(), 'development_only': True, 'research_question_change_approved': False,
        'reason': 'Declared after inspecting completed preprocessing outcomes: DEAP native baseline-relative logpower had unseen source-validation BA72.35/57.29%. Test whether the measured pre-stimulus baseline alone predicts individual ratings before attributing the lead to stimulus-response features. This is a postdiagnostic source control, not prospective confirmation.',
        'counts': {'candidate_fits': 16, 'selected_heads': 4},
        'source_sha256': {f: sha(REPO/f) for f in (*SOURCES, 'scripts/prestimulus_control.py')},
        'upstream': {f: sha(parent/f) for f in ('plan.json', 'verification.json', 'inputs/deap/prepared.json',
                    'inputs/deap/baseline.npy', 'inputs/deap/trials.csv')},
        'panels': [1, 2], 'montages': ['common14', 'native'], 'C': [.01, .1, 1., 10.],
        'input': 'Only the separately filtered3-second pre-stimulus baseline,128Hz,384samples,real32electrodes/common14. Same checked source subsets and rating labels as parent. Welch log4-band power using parent FFT/bands/floor. Parent loader can load stimulus arrays, but none is used for features, scaler, fitting, selection or inference.',
        'fitting': 'Train-only StandardScaler,balanced LogisticRegression,maxiter4000,tol1e-6,random_state42. Equal-panel source familiar/unseen balanced logloss,then meanBA,then first declared C. All16 candidates saved and independently refit. Compare matched rows to previously retained stimulus-only and baseline-relative controls; no test inference or population CI.',
        'limits': 'Small reused validation panels, postdiagnostic choice, no new participants. A baseline-alone association can reflect trait, context, carryover or preprocessing; it does not causally identify a mechanism. Better baseline-relative validation is a lead, not demonstrated incremental physiology or a novel method.'}
    atomic(output/'plan.json', plan)
    export(output)
    print(json.dumps({'declared': plan['counts'], 'plan_sha256': sha(output/'plan.json')}))


def export(output):
    destination = REPO/'results/development'/STUDY
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(output.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.csv'):
            relative = p.relative_to(output); target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
            if sha(target) != sha(p): raise ValueError('Failed export')
            files.append({'file': relative.as_posix(), 'sha256': sha(target)})
    atomic(destination/'export_manifest.json', {'files': files,
        'scope': 'Source-only baseline-alone probabilities, declaration and coefficient-refit attestations; raw EEG and coefficient arrays remain local.'})


def run(root):
    output = root/STUDY; parent = root/PARENT
    plan = json.loads((output/'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if sha(REPO/name) != expected: raise ValueError('Changed source')
    for name, expected in plan['upstream'].items():
        if sha(parent/name) != expected: raise ValueError('Changed parent input')
    if list((output/'fits').glob('*')): raise FileExistsError('Preserve existing fits')
    seed_everything(42)
    records = []; error = 0.
    for group in (1, 2):
        table, idx, raw, baseline, info = load_panel(parent, 'DEAP', group)
        del raw
        for montage in ('common14', 'native'):
            folder = output/'fits'/f'deap_g{group}_{montage}'; folder.mkdir(parents=True)
            picks = [info['channels'].index(c) for c in COMMON_CHANNELS] if montage == 'common14' else list(range(32))
            x = logpower(baseline[:, picks]).reshape(len(table), -1).astype(np.float64)
            scaler = StandardScaler().fit(x[idx['train']]); z = scaler.transform(x)
            candidates, artifacts = [], []
            for c in plan['C']:
                model = LogisticRegression(C=c, class_weight='balanced', max_iter=4000, tol=1e-6, random_state=42)
                model.fit(z[idx['train']], table.iloc[idx['train']].label)
                if np.any(model.n_iter_ >= 4000): raise ValueError('Optimization did not converge')
                label = f'C{c}'
                probabilities = {r: model.predict_proba(z[idx[r]]) for r in ROLES}
                metrics = {r: metric(table.iloc[idx[r]].label, probabilities[r]) for r in ROLES}
                artifacts.extend(save_predictions(folder, label, table, idx, probabilities))
                np.savez(folder/f'{label}.npz', mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_, intercept=model.intercept_)
                artifacts.append(f'{label}.npz')
                # Independently refit with fresh scaler and estimator, then
                # replay saved coefficients directly with a logistic link.
                other_scaler = StandardScaler().fit(x[idx['train']])
                other = LogisticRegression(C=c, class_weight='balanced', max_iter=4000, tol=1e-6, random_state=42)
                other.fit(other_scaler.transform(x[idx['train']]), table.iloc[idx['train']].label)
                params = np.load(folder/f'{label}.npz', allow_pickle=False)
                for actual, field in ((other_scaler.mean_, 'mean'), (other_scaler.scale_, 'scale'),
                                      (other.coef_, 'coef'), (other.intercept_, 'intercept')):
                    np.testing.assert_array_equal(actual, params[field])
                for role in ROLES:
                    rows = pd.read_csv(folder/f'{label}_{role}.csv')
                    q = expit(((x[idx[role]]-params['mean'])/params['scale']) @ params['coef'][0]+params['intercept'][0])
                    p = np.column_stack([1-q, q]); expected = rows[['p0','p1']].to_numpy()
                    error = max(error, float(np.max(np.abs(p-expected))))
                    np.testing.assert_allclose(p, expected, atol=1e-6, rtol=0)
                    actual = independent(rows)
                    for field in ('balanced_accuracy','balanced_log_loss'):
                        np.testing.assert_allclose(actual[field], metrics[role][field], atol=1e-6, rtol=0)
                candidates.append({'id': label, 'C': c, 'metrics': metrics})
            record = {'group': group, 'montage': montage, 'candidates': candidates,
                      'selected': min(candidates, key=selection_key)['id'], 'plan_sha256': sha(output/'plan.json'),
                      'coefficients_and_scalers_refitted_exact': True,
                      'artifact_sha256': {f: sha(folder/f) for f in artifacts}}
            atomic(folder/'record.json', record); records.append(record)
    atomic(output/'records.json', records)
    atomic(output/'verification.json', {'passed': True, 'complete': True, 'candidate_fits_refitted': 16,
        'selected_heads': 4, 'probability_metric_sets_recomputed': 48,
        'maximum_probability_error': error, 'plan_sha256': sha(output/'plan.json'),
        'test_predictions_used': 0, 'post_stimulus_features_used': False})
    export(output)
    print(json.dumps({'complete': True, 'candidate_fits': 16, 'maximum_probability_error': error}))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    args = parser.parse_args()
    (declare if args.command == 'plan' else run)(args.root.resolve())
