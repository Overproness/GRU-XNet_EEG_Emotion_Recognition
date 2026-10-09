"""Complete-grid comparison and within-video association; no interim ranking."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.deap_baseline_folds_v2 import MODELS, REPRESENTATIONS, ARMS, cell_id
from gruxnet.preprocessing_diagnostic_v2 import atomic, sha, stamp
from gruxnet.grouped_material_controls import weights, check_coverage
from gruxnet.full_context_controls_v2 import statistic
from scripts.within_video_alignment import pairs


def collect(output):
    plan = json.loads((output/'plan.json').read_text()); frames = {}
    folder = output/'aggregate'; folder.mkdir(exist_ok=True)
    ordering = ['group', 'source_session', 'material_rotation', 'fold', 'trial_id']
    for model in MODELS:
        for arm in ARMS:
            parts = []
            for job in plan['jobs']:
                if job['arm'] != arm: continue
                rows = pd.read_csv(output/'cells'/cell_id(job)/model/'selected_predictions.csv', float_precision='round_trip')
                parts.append(rows[rows.role.eq('test')])
            rows = pd.concat(parts, ignore_index=True).sort_values(ordering).reset_index(drop=True)
            for _, part in rows.groupby('group'):
                check_coverage(part, part.drop_duplicates('trial_id'))
                if len(part) != 1264 or part.trial_id.nunique() != 1264:
                    raise ValueError('Missing once-per-trial coverage')
            if len(rows) != 2528: raise ValueError('Wrong combined OOF population')
            rows.to_csv(folder/f'{model}_{arm}.csv', index=False)
            frames[(model, arm)] = rows
    columns = ['trial_id', 'subject_id', 'material_key', 'label', 'original_label', *ordering[:-1]]
    reference = next(iter(frames.values()))[columns]
    for rows in frames.values(): pd.testing.assert_frame_equal(reference, rows[columns], check_exact=True)
    return frames


def comparisons():
    result = [(model, 'unexposed', model, 'exposed') for model in MODELS]
    for arm in ARMS:
        result.extend(('baseline_relative', arm, other, arm)
                      for other in ('stimulus_absolute', 'stimulus_relative', 'baseline_only'))
        result.extend((f'{r}_context', arm, other, arm)
                      for r in REPRESENTATIONS for other in (r, 'prior', 'context_logistic'))
    return result


def alignment(frames, subjects, materials, sw, mw):
    reference = next(iter(frames.values())); recipient = []; donor = []; coefficients = []; eligible = []
    for _, part in reference.groupby('group', sort=True):
        indexes = part.index.to_numpy()
        r, d, b, e = pairs(part.reset_index(drop=True))
        recipient.extend(indexes[r]); donor.extend(indexes[d]); coefficients.extend(b); eligible.extend(indexes[e])
    recipient = np.asarray(recipient, dtype=int); donor = np.asarray(donor, dtype=int)
    base = np.asarray(coefficients); eligible = np.asarray(sorted(eligible), dtype=int)
    ri = reference.iloc[recipient].subject_id.map({s: i for i, s in enumerate(subjects)}).to_numpy()
    di = reference.iloc[donor].subject_id.map({s: i for i, s in enumerate(subjects)}).to_numpy()
    vi = reference.iloc[recipient].material_key.map({s: i for i, s in enumerate(materials)}).to_numpy()
    target = reference.label.to_numpy(dtype=int)[recipient]
    keys = []; columns = []
    for (model, arm), rows in frames.items():
        p = rows[['p0', 'p1']].to_numpy()
        for name, indexes in (('aligned', recipient), ('exchanged', donor)):
            for which in ('BA', 'logloss'):
                values = (p[indexes].argmax(1) == target).astype(float) if which == 'BA' else -np.log(np.clip(p[indexes, target], 1e-12, 1.))
                keys.append((model, arm, name, which)); columns.append(values)
    score = np.column_stack(columns)
    points = np.zeros(len(keys)); draws = {}; validity = {}
    for label in (0, 1):
        keep = target == label; fixed = base[keep]
        points += (fixed @ score[keep])/fixed.sum()/2
    for scope, video_weights in (('crossed', mw), ('participant', np.ones_like(mw))):
        sample = np.zeros((len(sw), len(keys))); valid = np.ones(len(sw), dtype=bool)
        for label in (0, 1):
            keep = target == label; fixed = base[keep]; values = score[keep]
            for start in range(0, len(sw), 128):
                stop = min(start+128, len(sw))
                weighted = sw[start:stop, ri[keep]]*sw[start:stop, di[keep]]*video_weights[start:stop, vi[keep]]*fixed
                denominator = weighted.sum(1); ok = denominator > 0; valid[start:stop] &= ok
                sample[start:stop] += np.divide(weighted @ values, denominator[:, None],
                    out=np.zeros((stop-start, len(keys))), where=ok[:, None])/2
        draws[scope] = sample; validity[scope] = valid
    result = {'eligible_rows_per_model': len(eligible), 'total_rows_per_model': len(reference),
        'dyadic_pairs': len(recipient), 'excluded_trials': reference.drop(eligible)[['group', 'trial_id']].to_dict('records'),
        'nonestimable_draws': {scope: int((~valid).sum()) for scope, valid in validity.items()},
        'scope': 'Same selected fold/group/video models, exchange all other held-out participants feature predictions while keeping recipient labels. Video prior is identical within donor groups. Dyadic recipient/donor/video weights; conditional association, not causal emotion or stimulus physiology.',
        'models': {}, 'contrasts': []}
    for index, (model, arm, name, which) in enumerate(keys):
        item = {'point': float(points[index])}
        for scope in draws:
            item[f'{scope}_percentile_95'] = np.quantile(draws[scope][validity[scope], index], [.025, .975]).tolist()
        result['models'].setdefault(model, {}).setdefault(arm, {}).setdefault(name, {})[which] = item
    for model in MODELS:
        for arm in ARMS:
            for which in ('BA', 'logloss'):
                a = keys.index((model, arm, 'aligned', which)); b = keys.index((model, arm, 'exchanged', which))
                item = {'model': model, 'arm': arm, 'metric': which, 'difference': float(points[a]-points[b])}
                for scope in draws:
                    delta = draws[scope][validity[scope], a]-draws[scope][validity[scope], b]
                    item[f'{scope}_percentile_95'] = np.quantile(delta, [.025, .975]).tolist()
                if model in ('prior', 'context_logistic'):
                    np.testing.assert_allclose(points[a], points[b], atol=1e-12, rtol=0)
                    for scope in draws: np.testing.assert_allclose(draws[scope][:, a], draws[scope][:, b], atol=1e-12, rtol=0)
                result['contrasts'].append(item)
    return result


def independent_regular_check(frames, comparison, subjects, materials, sw, mw):
    """Different flat trial-weight computation verifies all points and CI endpoints."""
    errors = []; distributions = {}
    for scope, group in (('combined', None), ('group1', 1), ('group2', 2)):
        for key, full in frames.items():
            rows = full if group is None else full[full.group.eq(group)]
            p = rows[['p0', 'p1']].to_numpy(); y = rows.label.to_numpy(dtype=int)
            work = rows[['trial_id', 'subject_id', 'material_key', 'label']].copy()
            work['BA'] = (p.argmax(1) == y).astype(float)
            work['logloss'] = -np.log(np.clip(p[np.arange(len(p)), y], 1e-12, 1.))
            cell = work.groupby(['trial_id', 'subject_id', 'material_key', 'label'], sort=True)[['BA', 'logloss']].mean().reset_index()
            si = cell.subject_id.map({s: i for i, s in enumerate(subjects)}).to_numpy()
            vi = cell.material_key.map({s: i for i, s in enumerate(materials)}).to_numpy()
            label = cell.label.to_numpy(dtype=int)
            points = np.zeros(2); crossed = np.zeros((len(sw), 2)); participant = np.zeros_like(crossed)
            for c in (0, 1):
                keep = label == c; values = cell.loc[keep, ['BA', 'logloss']].to_numpy()
                points += values.mean(0)/2
                person = sw[:, si[keep]]; both = person*mw[:, vi[keep]]
                crossed += (both @ values)/both.sum(1)[:, None]/2
                participant += (person @ values)/person.sum(1)[:, None]/2
            distributions[(scope, *key)] = points, crossed, participant
            for i, which in enumerate(('BA', 'logloss')):
                saved = comparison['models'][scope][key[0]][key[1]][which]
                errors.append(abs(saved['point']-points[i]))
                errors.extend(np.abs(np.asarray(saved['crossed_percentile_95'])-np.quantile(crossed[:, i], [.025, .975])))
                errors.extend(np.abs(np.asarray(saved['participant_percentile_95'])-np.quantile(participant[:, i], [.025, .975])))
    for saved in comparison['contrasts']:
        i = ('BA', 'logloss').index(saved['metric'])
        pa, ca, sa = distributions[(saved['scope'], saved['model_a'], saved['arm_a'])]
        pb, cb, sb = distributions[(saved['scope'], saved['model_b'], saved['arm_b'])]
        errors.append(abs(saved['difference']-(pa[i]-pb[i])))
        errors.extend(np.abs(np.asarray(saved['crossed_percentile_95'])-np.quantile(ca[:, i]-cb[:, i], [.025, .975])))
        errors.extend(np.abs(np.asarray(saved['participant_percentile_95'])-np.quantile(sa[:, i]-sb[:, i], [.025, .975])))
    maximum = float(max(errors))
    if maximum > 1e-11: raise ValueError('Independent crossed/participant point or interval mismatch')
    return maximum


def plot(output, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Stimulus\nabsolute', 'Stimulus\nrelative', 'Baseline\nonly', 'Stimulus −\nbaseline',
        'Absolute\n+ context', 'Relative\n+ context', 'Baseline\n+ context', 'Difference\n+ context',
        'Calibrated\ncontext', 'Raw\ncontext']
    figure, axes = plt.subplots(2, 2, figsize=(17, 10), sharey='row')
    for column, arm in enumerate(ARMS):
        for row, which in enumerate(('BA', 'logloss')):
            ax = axes[row, column]; factor = 100 if which == 'BA' else 1
            for position, model in enumerate(MODELS):
                item = comparison['models']['combined'][model][arm][which]
                lo, hi = np.asarray(item['crossed_percentile_95'])*factor
                ax.vlines(position, lo, hi, color='#2767a0', linewidth=2)
                ax.scatter(position, item['point']*factor, c='#2767a0', s=35, zorder=3)
                for offset, group, color in ((-.12, 'group1', '#d36b21'), (.12, 'group2', '#448a57')):
                    ax.scatter(position+offset, comparison['models'][group][model][arm][which]['point']*factor,
                               c=color, marker='x', s=28, zorder=4)
            ax.axhline(50 if which == 'BA' else np.log(2), color='#777777', linestyle=':', linewidth=1)
            ax.set_xticks(range(len(MODELS)), labels, rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=.2)
            ax.set_ylabel('Balanced accuracy (%)' if which == 'BA' else 'Balanced log loss')
            if row == 0: ax.set_title('Familiar test videos' if arm == 'exposed' else 'Unseen test videos')
    figure.suptitle('Matched DEAP feature/context controls — reused development cohorts', fontsize=16)
    figure.text(.5, .015, 'Blue: combined point and crossed person/video 95% percentile interval. Orange/green crosses: grouping 1/2 points.\n'
        '32 people, 40 videos, 1,264 retained trials; fixed fitted models, unadjusted exploratory intervals; repeated groupings are paired.',
        ha='center', fontsize=10)
    figure.tight_layout(rect=(0, .075, 1, .95))
    folder = output/'plots'; folder.mkdir(exist_ok=True)
    figure.savefig(folder/'matched_deap_controls.png', dpi=180)
    figure.savefig(folder/'matched_deap_controls.svg')
    svg = folder/'matched_deap_controls.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n', encoding='utf-8')
    plt.close(figure)


def analyze(output):
    verification = json.loads((output/'verification.json').read_text())
    if not verification['passed'] or not verification['complete']: raise ValueError('Complete verification required')
    frames = collect(output); reference = next(iter(frames.values()))
    subjects, materials, sw, mw = weights(reference, 'DEAP')
    result = {'created_utc': stamp(), 'development_only': True, 'research_question_change_approved': False,
        'plan_sha256': sha(output/'plan.json'), 'verification_sha256': sha(output/'verification.json'),
        'test_probability_rows': sum(len(rows) for rows in frames.values()),
        'bootstrap': {'draws': 10000, 'seed': 20261006, 'scope': 'Paired fixed-fit person-only and crossed person/video percentile intervals. Average correctness/loss within observed person/video across repeated groupings; no probability ensemble, new people, multiplicity adjustment or confirmatory cohort.'},
        'models': {}, 'contrasts': [], 'aggregate_sha256': {path.name: sha(path) for path in sorted((output/'aggregate').glob('*.csv'))}}
    distributions = {}
    for scope, group in (('combined', None), ('group1', 1), ('group2', 2)):
        result['models'][scope] = {}
        for (model, arm), full in frames.items():
            rows = full if group is None else full[full.group.eq(group)]
            reports = {}; distributions[(scope, model, arm)] = {}
            for which in ('BA', 'logloss'):
                point, crossed = statistic(rows, 'DEAP', 'binary', subjects, materials, sw, mw, which)
                _, person = statistic(rows, 'DEAP', 'binary', subjects, materials, sw, np.ones_like(mw), which)
                reports[which] = {'point': point, 'crossed_percentile_95': np.quantile(crossed, [.025, .975]).tolist(),
                                 'participant_percentile_95': np.quantile(person, [.025, .975]).tolist()}
                distributions[(scope, model, arm)][which] = point, crossed, person
            result['models'][scope].setdefault(model, {})[arm] = reports
        for a, aa, b, ba in comparisons():
            for which in ('BA', 'logloss'):
                pa, ca, sa = distributions[(scope, a, aa)][which]
                pb, cb, sb = distributions[(scope, b, ba)][which]
                result['contrasts'].append({'scope': scope, 'model_a': a, 'arm_a': aa, 'model_b': b, 'arm_b': ba,
                    'metric': which, 'difference': pa-pb,
                    'crossed_percentile_95': np.quantile(ca-cb, [.025, .975]).tolist(),
                    'participant_percentile_95': np.quantile(sa-sb, [.025, .975]).tolist()})
    assert result['test_probability_rows'] == 50560 and len(result['contrasts']) == 240
    atomic(output/'comparison.json', result)
    aligned = alignment(frames, subjects, materials, sw, mw)
    atomic(output/'alignment.json', aligned)
    maximum = independent_regular_check(frames, result, subjects, materials, sw, mw)
    atomic(output/'analysis_verification.json', {'passed': True, 'plan_sha256': sha(output/'plan.json'),
        'aggregate_metric_sets': 120, 'paired_contrasts': 240, 'within_video_contrasts': len(aligned['contrasts']),
        'maximum_regular_point_and_interval_error': maximum,
        'scope': 'Independent flat observed-trial bootstrap verifies all regular metric/contrast points and both percentile endpoint sets. Within-video vectorized dyadic calculation uses the previously checked pairing design and verifies both context controls are invariant; it is not a second independent alignment-bootstrap implementation.'})
    plot(output, result)
    lines = ['# Matched full-fold DEAP baseline/context controls', '',
        '**Completed development study; no manuscript or research-question change.**', '',
        '160 matched cells, 5,760 independently refitted candidates, 1,440 selected logistic heads and 160 raw priors. '
        'Each model/arm/group covers all 1,264 retained original trials exactly once; 50,560 total test prediction rows. '
        'Existing people/videos were previously examined. These are complete development repetitions, not new independent confirmation.', '',
        '| Model | Familiar-video BA | Unseen-video BA | Familiar loss | Unseen loss |', '| --- | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        known = result['models']['combined'][model]['exposed']; unseen = result['models']['combined'][model]['unexposed']
        lines.append(f'| {model} | {100*known["BA"]["point"]:.2f}% | {100*unseen["BA"]["point"]:.2f}% | {known["logloss"]["point"]:.4f} | {unseen["logloss"]["point"]:.4f} |')
    lines += ['', 'All 240 regular contrasts and 40 within-video contrasts are reported in `comparison.json` and `alignment.json`. '
        'Negative loss differences favour the first model; positive accuracy differences favour it. '
        'Crossed participant/video uncertainty is primary; intervals are unadjusted and conditional on fixed fits. '
        'Baseline associations can reflect person/context/carryover or preprocessing. These analyses do not identify causal emotion physiology.', '',
        'Features use all 32 electrodes, separate 4–40 Hz stimulus/baseline filtering, first 40 seconds, '
        'ten four-second Welch windows and measured three-second baseline. Labels retain the corrected individual-valence policy with sixteen midpoint exclusions. '
        'Every learned scaler and C choice uses training/source validation only. Training priors exclude the recipient participant entirely. '
        'Raw EEG/features and coefficients remain local. First-party DEAP signal authentication remains outstanding.', '',
        'The finite linear control family and reused dataset do not establish a new method, EEG absence, joint negative transfer, '
        'or conference readiness. An adopted research-question change requires author approval and a fresh manuscript archive.', '']
    (output/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
