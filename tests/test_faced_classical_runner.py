"""Manufactured data only: activation, priors, complete budgets and fallback."""
import unittest
import numpy as np
from gruxnet.faced_classical_runner import (source_gate, activate_after_source,
    estimate_priors, prior_predictions, paired_diagnostics, run_controls)


def fixture():
    source_people = [f'source{i}' for i in range(10)]
    validation_people = [f'validation{i}' for i in range(2)]
    rotations = [([3,4,9,11,17,20,23,27], [2,6,12,18,25,28]),
                 ([2,6,9,12,18,20,25,28], [3,4,11,17,23,27]),
                 ([2,6,9,11,17,20,23,28], [3,4,12,18,25,27])]
    clips = sorted(set(rotations[0][0]+rotations[0][1]))
    rng = np.random.default_rng(412)
    features = {'context': {}, 'eeg': {}}; targets = {}; metadata = {}
    for person in source_people+validation_people:
        for vid in clips:
            pair = person, vid
            context = rng.normal(size=3); eeg = rng.normal(size=5)
            features['context'][pair] = context; features['eeg'][pair] = eeg
            targets[pair] = float(np.clip(3.5+.35*context[0]+.35*eeg[0], 0, 7))
            metadata[pair] = {'assigned_valence': 'positive' if vid % 2 else 'negative',
                'assigned_category': f'category{vid%4}', 'family': f'film{vid}',
                'acquisition_group': 'manufactured'}
    manifests = []
    for rotation, (src, val) in enumerate(rotations):
        product = lambda people, vids: [{'subject': p, 'vid': v} for p in people for v in vids]
        manifests.append({'rotation': rotation, 'arms': {
            'source_fit': product(source_people, src),
            'validation_unseen': product(validation_people, val),
            'validation_familiar': product(validation_people, src)}})
    plan = {'rotation_manifests': manifests,
            'source_participant_oof_folds': [source_people[i::5] for i in range(5)]}
    return plan, features, targets, metadata


class RunnerTests(unittest.TestCase):
    def test_source_gate_holds_rating_callback_for_flags_and_numerical_failures(self):
        pairs = [('source', 2)]; called = []
        for technical, flags in [(True, [{'channel': 'Pz', 'flag': 'residual_50hz_power'}]), (False, [])]:
            records = [{'subject': 'source', 'vid': 2, 'technical_passed': technical, 'review_flags': flags}]
            self.assertFalse(source_gate(records, pairs)['passed'])
            with self.assertRaises(PermissionError):
                activate_after_source(records, pairs, lambda: called.append('opened'))
        self.assertEqual(called, [])

    def test_source_gate_rejects_missing_or_duplicate_coverage(self):
        row = {'subject': 'source', 'vid': 2, 'technical_passed': True, 'review_flags': []}
        for rows in ([], [row, row]):
            with self.assertRaises(ValueError): source_gate(rows, [('source', 2)])

    def test_source_gate_pass_activates_once(self):
        records = [{'subject': 'source', 'vid': 2, 'technical_passed': True, 'review_flags': []}]
        called = []
        activate_after_source(records, [('source', 2)], lambda: called.append('opened'))
        self.assertEqual(called, ['opened'])

    def test_priors_never_fetch_excluded_targets_and_unknown_films_use_global(self):
        _, _, targets, metadata = fixture()
        source = [('source0', 3), ('source1', 3), ('source0', 4)]
        restricted = {p: targets[p] for p in source}
        prior = estimate_priors(source, restricted, metadata)
        predictions = prior_predictions(prior, [('validation0', 28)], metadata)
        self.assertEqual(predictions['familiar_video_median'][0], np.median(list(restricted.values())))
        self.assertEqual(predictions['global_median'][0], np.median(list(restricted.values())))

    def test_crossed_bootstrap_constant_gain_and_pair_identity_guard(self):
        c = [{'subject': p, 'vid': v, 'family': f'film{v}', 'acquisition_group': 'test',
              'target': 3., 'prediction': 4.} for p in ('a','b') for v in (1,2)]
        j = [{**r, 'prediction': 3.5} for r in c]
        result = paired_diagnostics(c, j)
        self.assertEqual(result['gain'], .5)
        self.assertEqual(result['descriptive_95_percent_interval'], [.5,.5])
        with self.assertRaises(ValueError): paired_diagnostics(c, j[:-1])

    def test_complete_grid_exact_budget_and_oof_exclusion(self):
        plan, features, targets, metadata = fixture(); jobs = []; rows = []
        def retain(job, fitted):
            people = {p for p,v in fitted['source_pairs']}
            self.assertFalse(any(p.startswith('validation') for p in people))
            if job['oof_fold'] is not None:
                self.assertFalse(people & set(plan['source_participant_oof_folds'][job['oof_fold']]))
                self.assertEqual(len(people), 8)
            jobs.append(job)
        result = run_controls(plan, features, targets, metadata, retain, rows.extend)
        self.assertEqual(result['ridge_fits'], 90); self.assertEqual(result['prior_estimations'], 72)
        self.assertEqual(len(jobs), 90)
        self.assertEqual(sum(j['oof_fold'] is None for j in jobs), 45)
        self.assertEqual(len(rows), 3*(15*28+4*28+7*80))
        self.assertFalse(result['confirmation_access'])
        self.assertTrue(all(r['paired']['gain'] > 0 for r in result['rotations']))

    def test_invalid_validation_eeg_retains_all_trials_with_exact_fallback(self):
        plan, features, targets, metadata = fixture(); rows = []
        missing = ('validation0', 2); features['eeg'].pop(missing)
        result = run_controls(plan, features, targets, metadata, row_callback=rows.extend)
        for rotation in result['rotations']:
            self.assertEqual(rotation['validation_eeg_fallback_trials'], 1)
            selected = rotation['selected']['context_only']['alpha']
            context = [r for r in rows if r['rotation'] == rotation['rotation'] and r['model'] == 'context_only' and r['alpha'] == selected and (r['subject'],r['vid']) == missing][0]['prediction']
            joint = [r for r in rows if r['rotation'] == rotation['rotation'] and r['model'] == 'context_plus_spectral_eeg' and (r['subject'],r['vid']) == missing]
            self.assertEqual(len(joint), 5)
            self.assertTrue(all(r['fallback'] and r['prediction'] == context for r in joint))
            source_pairs = [(r['subject'],r['vid']) for r in plan['rotation_manifests'][rotation['rotation']]['arms']['source_fit']]
            global_median = np.median([targets[p] for p in source_pairs])
            eeg = [r for r in rows if r['rotation'] == rotation['rotation'] and r['model'] == 'spectral_eeg_only' and (r['subject'],r['vid']) == missing]
            self.assertTrue(all(r['fallback'] and r['prediction'] == global_median for r in eeg))

    def test_invalid_source_eeg_stops_without_fits(self):
        plan, features, targets, metadata = fixture(); jobs = []
        features['eeg'][('source0', 3)] = np.full(5, np.nan)
        with self.assertRaises(ValueError):
            run_controls(plan, features, targets, metadata, lambda job, fitted: jobs.append(job))
        self.assertEqual(jobs, [])


if __name__ == '__main__': unittest.main()
