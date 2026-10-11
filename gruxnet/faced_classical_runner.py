"""Bounded, synthetic-testable FACED controls; this module never opens raw files."""
from __future__ import annotations
from collections import Counter
import numpy as np
from .faced_classical_contract import (ALPHAS, fit_source_ridge, predict_ridge,
                                       participant_film_mae, select_alpha)

FAMILIES = ('context_only', 'spectral_eeg_only', 'context_plus_spectral_eeg')
PRIORS = ('global_median', 'coarse_valence_median', 'category_median', 'familiar_video_median')


def source_gate(records, expected_pairs):
    """Fail closed on missing, duplicated, numerically failed or flagged sources."""
    pairs = [(r['subject'], r['vid']) for r in records]
    expected = set(expected_pairs)
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError('Source quality coverage differs from the declared rectangle')
    failures = [p for p, r in zip(pairs, records) if not r['technical_passed']]
    flagged = [p for p, r in zip(pairs, records) if r['review_flags']]
    return {'passed': not failures and not flagged, 'source_trials': len(records),
            'numerical_failure_trials': len(failures), 'flagged_trials': len(flagged),
            'fitting_held': bool(failures or flagged)}


def activate_after_source(records, expected_pairs, target_loader):
    gate = source_gate(records, expected_pairs)
    if not gate['passed']:
        raise PermissionError('Source quality holds original rating access and fitting')
    return target_loader()


def estimate_priors(source_pairs, targets, metadata):
    """Fetch only permitted source labels, including for held-person diagnostics."""
    pairs = list(source_pairs)
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError('Invalid source-prior identities')
    y = np.asarray([targets[p] for p in pairs], dtype=float)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 7)):
        raise ValueError('Invalid source target')
    result = {'global': float(np.median(y)), 'coarse': {}, 'category': {}, 'video': {}}
    for field, key in [('assigned_valence', 'coarse'), ('assigned_category', 'category'), ('vid', 'video')]:
        groups = {}
        for pair, value in zip(pairs, y):
            group = pair[1] if field == 'vid' else metadata[pair][field]
            groups.setdefault(group, []).append(value)
        result[key] = {group: float(np.median(values)) for group, values in groups.items()}
    return result


def prior_predictions(fitted, pairs, metadata):
    return {
        'global_median': np.asarray([fitted['global'] for p in pairs]),
        'coarse_valence_median': np.asarray([fitted['coarse'].get(metadata[p]['assigned_valence'], fitted['global']) for p in pairs]),
        'category_median': np.asarray([fitted['category'].get(metadata[p]['assigned_category'], fitted['global']) for p in pairs]),
        'familiar_video_median': np.asarray([fitted['video'].get(p[1], fitted['global']) for p in pairs]),
    }


def rows_for(pairs, values, targets, metadata):
    if len(pairs) != len(values):
        raise ValueError('Prediction count mismatch')
    return [{'subject': p, 'vid': v, 'family': metadata[(p, v)]['family'],
             'acquisition_group': metadata[(p, v)]['acquisition_group'],
             'target': float(targets[(p, v)]), 'prediction': float(value)}
            for (p, v), value in zip(pairs, values)]


def metrics(rows):
    errors = np.asarray([r['prediction'] - r['target'] for r in rows])
    return {'participant_film_mae': participant_film_mae(rows),
            'trial_mae': float(np.abs(errors).mean()), 'trial_mse': float((errors**2).mean()),
            'trials': len(rows), 'participants': len({r['subject'] for r in rows}),
            'films': len({r['family'] for r in rows})}


def paired_diagnostics(context_rows, joint_rows, draws=4000, seed=425401):
    """Crossed participant/film resampling of paired, film-equal absolute errors."""
    c = {(r['subject'], r['vid']): r for r in context_rows}
    j = {(r['subject'], r['vid']): r for r in joint_rows}
    if len(c) != len(context_rows) or len(j) != len(joint_rows) or c.keys() != j.keys():
        raise ValueError('Paired comparison identities differ')
    grouped = {}
    for pair, row in c.items():
        other = j[pair]
        if row['target'] != other['target'] or row['family'] != other['family']:
            raise ValueError('Paired target or film differs')
        gain = abs(row['target']-row['prediction'])-abs(other['target']-other['prediction'])
        grouped.setdefault((row['subject'], row['family']), []).append(gain)
    people = sorted({p for p, f in grouped}); films = sorted({f for p, f in grouped})
    if len(grouped) != len(people)*len(films):
        raise ValueError('Crossed bootstrap requires complete participant/film coverage')
    matrix = np.asarray([[np.mean(grouped[p, f]) for f in films] for p in people])
    rng = np.random.default_rng(seed)
    p = rng.integers(len(people), size=(draws, len(people)))
    f = rng.integers(len(films), size=(draws, len(films)))
    samples = matrix[p[:, :, None], f[:, None, :]].mean(axis=(1, 2))
    return {'gain': float(matrix.mean()), 'crossed_bootstrap_draws': draws, 'seed': seed,
            'descriptive_95_percent_interval': np.quantile(samples, [.025, .975]).tolist(),
            'leave_one_film_out_gain': {film: float(np.delete(matrix, i, axis=1).mean()) for i, film in enumerate(films)},
            'acquisition_metrics': {group: {
                'context': metrics([r for r in context_rows if r['acquisition_group'] == group]),
                'joint': metrics([r for r in joint_rows if r['acquisition_group'] == group])}
                for group in sorted({r['acquisition_group'] for r in context_rows})},
            'limits': 'Exploratory: reused validation, six films, development-selected alphas; not independent confirmation or tuning-corrected uncertainty'}


def run_controls(plan, features, targets, metadata, fit_callback=None, row_callback=None, check_budget=None):
    """Run exactly the declared grid, retaining every candidate and OOF prediction."""
    feature_maps = {
        'context_only': features['context'], 'spectral_eeg_only': features['eeg'],
        'context_plus_spectral_eeg': {p: np.r_[x, features['eeg'][p]]
                                    for p, x in features['context'].items() if p in features['eeg']},
    }
    # Missing/nonfinite validation EEG is handled explicitly without source imputation.
    invalid_eeg = {p for p in features['context']
                   if p not in features['eeg'] or not np.isfinite(features['eeg'][p]).all()}
    result = []; fit_count = 0; prior_count = 0

    def fit(pairs, family, alpha, job):
        nonlocal fit_count
        if check_budget is not None: check_budget()
        if fit_count >= 90: raise ValueError('Original Ridge fit cap exceeded')
        fitted = fit_source_ridge(pairs, feature_maps[family], targets, alpha)
        fit_count += 1
        if fit_callback is not None: fit_callback(job, fitted)
        return fitted

    def retain(rows, rotation, model, arm, alpha=None, fold=None, fallback=()):
        fallback = set(fallback)
        enriched = [{**r, 'rotation': rotation, 'model': model, 'arm': arm,
                     'alpha': alpha, 'oof_fold': fold,
                     'fallback': (r['subject'], r['vid']) in fallback} for r in rows]
        if row_callback is not None: row_callback(enriched)
        return enriched

    for manifest in plan['rotation_manifests']:
        rotation = manifest['rotation']
        arms = {k: [(r['subject'], r['vid']) for r in rows] for k, rows in manifest['arms'].items()}
        source = arms['source_fit']
        if invalid_eeg & set(source): raise ValueError('Source numerical failure cannot be imputed')
        source_people = {p for p, v in source}
        eval_pairs = arms['validation_unseen'] + arms['validation_familiar']
        if source_people & {p for p, v in eval_pairs}: raise ValueError('Participant fit/evaluation overlap')
        priors = estimate_priors(source, targets, metadata); prior_count += 4
        selected = {}; candidates = {}; selected_rows = {}; prior_rows = {}
        for arm in ('validation_unseen', 'validation_familiar'):
            prior_rows[arm] = {}
            for name, values in prior_predictions(priors, arms[arm], metadata).items():
                rows = rows_for(arms[arm], values, targets, metadata)
                prior_rows[arm][name] = retain(rows, rotation, name, arm)
        for family in FAMILIES:
            grid = []; candidate_rows = []
            for alpha in ALPHAS:
                fitted = fit(source, family, alpha, {'rotation': rotation, 'family': family, 'alpha': alpha, 'oof_fold': None})
                by_arm = {}
                for arm in ('validation_unseen', 'validation_familiar'):
                    pairs = arms[arm]
                    valid = [p for p in pairs if family == 'context_only' or p not in invalid_eeg]
                    predicted = dict(zip(valid, predict_ridge(fitted, valid, feature_maps[family]))) if valid else {}
                    fallback = [p for p in pairs if p not in predicted]
                    context = {(r['subject'], r['vid']): r['prediction'] for r in selected_rows.get('context_only', {}).get(arm, [])}
                    for pair in fallback:
                        predicted[pair] = context[pair] if family == 'context_plus_spectral_eeg' else priors['global']
                    rows = rows_for(pairs, [predicted[p] for p in pairs], targets, metadata)
                    by_arm[arm] = retain(rows, rotation, family, arm, alpha, fallback=fallback)
                grid.append({'alpha': alpha, 'unseen_mae': participant_film_mae(by_arm['validation_unseen']),
                             'unseen_mse': metrics(by_arm['validation_unseen'])['trial_mse'],
                             'unseen': metrics(by_arm['validation_unseen']), 'familiar': metrics(by_arm['validation_familiar'])})
                candidate_rows.append(by_arm)
            index = select_alpha(grid); selected[family] = grid[index]; candidates[family] = grid
            selected_rows[family] = candidate_rows[index]
        oof = {name: [] for name in FAMILIES + PRIORS}
        covered = []
        for fold, people in enumerate(plan['source_participant_oof_folds']):
            held = [p for p in source if p[0] in people]
            training = [p for p in source if p[0] not in people]
            if not held or not training or {p for p, v in held} & {p for p, v in training}:
                raise ValueError('Invalid source OOF separation')
            covered.extend(held)
            fold_priors = estimate_priors(training, targets, metadata); prior_count += 4
            for name, values in prior_predictions(fold_priors, held, metadata).items():
                oof[name].extend(retain(rows_for(held, values, targets, metadata), rotation, name, 'source_oof', fold=fold))
            for family in FAMILIES:
                alpha = selected[family]['alpha']
                fitted = fit(training, family, alpha, {'rotation': rotation, 'family': family, 'alpha': alpha, 'oof_fold': fold})
                values = predict_ridge(fitted, held, feature_maps[family])
                oof[family].extend(retain(rows_for(held, values, targets, metadata), rotation, family, 'source_oof', alpha, fold))
        if Counter(covered) != Counter(source): raise ValueError('OOF source trials are not held exactly once')
        diag = paired_diagnostics(selected_rows['context_only']['validation_unseen'], selected_rows['context_plus_spectral_eeg']['validation_unseen'])
        best_prior = min(PRIORS[:3], key=lambda name: participant_film_mae(prior_rows['validation_unseen'][name]))
        best_prior_mae = participant_film_mae(prior_rows['validation_unseen'][best_prior])
        result.append({'rotation': rotation, 'selected': selected, 'candidates': candidates,
                       'priors': {arm: {name: metrics(rows) for name, rows in values.items()} for arm, values in prior_rows.items()},
                       'source_oof': {name: metrics(rows) for name, rows in oof.items()},
                       'paired': diag, 'strongest_fixed_unseen_prior': best_prior,
                       'joint_gain_over_strongest_fixed_prior': best_prior_mae-selected['context_plus_spectral_eeg']['unseen_mae'],
                       'validation_eeg_fallback_trials': len(set(eval_pairs) & invalid_eeg)})
    gains = [r['paired']['gain'] for r in result]
    prior_gains = [r['joint_gain_over_strongest_fixed_prior'] for r in result]
    return {'rotations': result, 'ridge_fits': fit_count, 'prior_estimations': prior_count,
            'mean_joint_gain_over_context': float(np.mean(gains)),
            'positive_gain_rotations': sum(g > 0 for g in gains),
            'mean_joint_gain_over_strongest_fixed_prior': float(np.mean(prior_gains)),
            'numerical_continuation_conditions_met': float(np.mean(gains)) >= .15 and sum(g > 0 for g in gains) >= 2 and float(np.mean(prior_gains)) > 0,
            'capacity_and_coverage_review_still_required': True,
            'research_question_changed': False, 'confirmation_access': False}
