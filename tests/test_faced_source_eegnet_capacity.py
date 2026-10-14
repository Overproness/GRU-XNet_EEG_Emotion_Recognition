import unittest
import numpy as np

from gruxnet.faced_source_eegnet_capacity import (guarded_source_targets, complete_cell, capacity_targets,
    learning_flags, guard_cell, source_offsets, PreparationBudget, verify_replayed_predictions,
    verify_reloaded_state, cache_metadata_projection, guard_source_original_path)


class FloatTrap:
    def __float__(self):
        raise AssertionError('Forbidden numerical field converted')


class Tests(unittest.TestCase):
    def metadata(self):
        return {('source', 1): {'role': 'source', 'family': 's', 'acquisition_group': 'legacy_250'},
                ('validation', 1): {'role': 'validation', 'family': 's', 'acquisition_group': 'legacy_250'}}

    def row(self, person='source', arm='source_fit', target=2.):
        return {'subject': person, 'vid': '1', 'family': 's', 'acquisition_group': 'legacy_250',
                'rotation': '0', 'model': 'category_median', 'arm': arm, 'target': target, 'prediction': FloatTrap()}

    def test_validation_and_all_prediction_values_remain_unconverted(self):
        y = guarded_source_targets([self.row(), self.row('validation', 'validation_unseen', FloatTrap())], self.metadata(), {('source', 1)})
        self.assertEqual(y, {('source', 1): 2.})

    def test_later_protected_row_fails_before_earlier_target_conversion(self):
        rows = [self.row(target=FloatTrap()), self.row('protected')]
        with self.assertRaises(PermissionError):
            guarded_source_targets(rows, self.metadata(), {('source', 1)})

    def test_incomplete_coverage_prevents_numerics(self):
        m = self.metadata(); m[('source', 2)] = m[('source', 1)]
        with self.assertRaises(PermissionError):
            guarded_source_targets([self.row(target=FloatTrap())], m, {('source', 1), ('source', 2)})

    def test_wrong_role_prevents_numerics(self):
        with self.assertRaises(PermissionError):
            guarded_source_targets([self.row('validation', target=FloatTrap())], self.metadata(), {('source', 1)})

    def test_repeated_target_disagreement(self):
        second = self.row(target=3.); second['rotation'] = '1'
        with self.assertRaises(ValueError):
            guarded_source_targets([self.row(), second], self.metadata(), {('source', 1)})

    def test_duplicate_and_unknown_models_before_numerics(self):
        row = self.row(target=FloatTrap())
        with self.assertRaises(PermissionError):
            guarded_source_targets([row, row], self.metadata(), {('source', 1)})
        bad = self.row(); bad['model'] = 'capacity_permuted'
        with self.assertRaises(PermissionError):
            guarded_source_targets([row, bad], self.metadata(), {('source', 1)})

    def test_metadata_first_complete_cell_selection(self):
        cells = [{'cell': i, 'training': [['p', i]], 'held': [['q', i]]} for i in range(10)]
        self.assertEqual(complete_cell(cells, [('p', i) for i in range(6)])['cell'], 6)
        with self.assertRaises(PermissionError):
            complete_cell(list(reversed(cells)), [])

    def test_fixed_source_capacity_permutation(self):
        tiny = [(f'p{i}', 1) for i in range(32)]
        y = dict(zip(tiny, np.linspace(0, 7, 32)))
        real, order, median = capacity_targets(tiny, y, permuted=False)
        permuted, perm, other = capacity_targets(tiny, y, permuted=True)
        self.assertEqual(real, y); self.assertEqual(order, list(range(32)))
        self.assertEqual(sorted(perm), list(range(32)))
        self.assertNotEqual(permuted, real); self.assertEqual(median, other)

    def test_learning_boundaries_and_budget_direction(self):
        self.assertEqual(learning_flags(2., [1., 1.], [.9, .9], [.2, .2]), [])
        self.assertIn('budget_limited_convergence', learning_flags(2., [1., 1.], [.89, .9], [.2, .2]))
        with self.assertRaises(ValueError):
            learning_flags(2., [1.], [1., 1.], [.2, .2])

    def test_whole_cell_roles_guarded_before_data_use(self):
        train = [(f't{i:02d}', v) for i in range(56) for v in range(8)]
        held = [(f'h{i:02d}', v) for i in range(14) for v in range(8, 14)]
        metadata = {p: {'role': 'source', 'family': str(p[1])} for p in train + held}
        cell = {'training': train, 'held': held}
        people = {p[0] for p in train + held}
        result = guard_cell(cell, metadata, people, [])
        self.assertEqual(len(result[2]), 32)
        metadata[held[-1]]['role'] = 'protected'
        with self.assertRaises(PermissionError):
            guard_cell(cell, metadata, people, [])

    def test_missing_or_family_overlap_stops_cell(self):
        train = [(f't{i:02d}', v) for i in range(56) for v in range(8)]
        held = [(f'h{i:02d}', v) for i in range(14) for v in range(8, 14)]
        metadata = {p: {'role': 'source', 'family': str(p[1])} for p in train + held}
        cell = {'training': train, 'held': held}; people = {p[0] for p in train + held}
        with self.assertRaises(PermissionError):
            guard_cell(cell, metadata, people, [held[-1]])
        metadata[held[-1]]['family'] = '0'
        with self.assertRaises(PermissionError):
            guard_cell(cell, metadata, people, [])

    def test_four_inner_source_person_folds_reconstruct_offsets(self):
        people = [f't{i:02d}' for i in range(56)]
        training = [(p, v) for p in people for v in range(8)]
        folds = [people[i:i + 14] for i in range(0, 56, 14)] + [['outer-held-person']]
        features = {p: np.asarray([int(p[0][1:]), p[1]], dtype=float) for p in training}
        targets = {p: float(p[1]) for p in training}
        full, oof, ledger = source_offsets(training, folds, features, targets)
        self.assertEqual(set(oof), set(training)); self.assertEqual(len(ledger), 4)
        for record in ledger:
            self.assertEqual(len(record['training']), 336); self.assertEqual(len(record['held']), 112)
            self.assertFalse({p[0] for p in record['training']} & {p[0] for p in record['held']})
            self.assertNotIn('outer-held-person', {p[0] for p in record['training'] + record['held']})
        with self.assertRaises(PermissionError):
            source_offsets(training, [people], features, targets)

    def test_exact_duplicate_target_conversion_receipt(self):
        second = self.row(); second['rotation'] = '1'
        _, receipt = guarded_source_targets([self.row(), second], self.metadata(), {('source', 1)}, return_receipt=True)
        self.assertEqual(receipt['selected_unique_target_pairs'], 1)
        self.assertEqual(receipt['selected_target_conversion_calls_including_duplicates'], 2)
        self.assertEqual(receipt['duplicate_selected_target_conversion_calls'], 1)
        self.assertEqual(receipt['prediction_conversion_calls'], 0)

    def test_model_alpha_oof_and_canonical_id_tokens_before_conversion(self):
        for field, value in (('alpha', 'nan'), ('oof_fold', '3'), ('vid', '01'), ('rotation', '00')):
            row = self.row(target=FloatTrap()); row[field] = value
            with self.assertRaises(PermissionError):
                guarded_source_targets([row], self.metadata(), {('source', 1)})

    def test_projection_never_converts_outcomes(self):
        a = self.row(target=FloatTrap()); b = self.row(target=FloatTrap()); b['prediction'] = FloatTrap()
        self.assertEqual(cache_metadata_projection([a]), cache_metadata_projection([b]))
        b['alpha'] = '100.0'
        self.assertNotEqual(cache_metadata_projection([a]), cache_metadata_projection([b]))

    def test_cooperative_wall_and_cpu_budgets(self):
        wall, cpu = [0.], [0.]
        budget = PreparationBudget(10., 20., lambda: wall[0], lambda: cpu[0])
        wall[0] = 9.99; cpu[0] = 19.99; budget.check()
        wall[0] = 10.
        with self.assertRaises(TimeoutError):
            budget.check()
        wall[0] = 0.; cpu[0] = 20.
        with self.assertRaises(TimeoutError):
            budget.receipt()

    def test_checkpoint_forward_tolerance_geometry_and_nonfinites(self):
        self.assertLess(verify_replayed_predictions([1., 2.], [1.00001, 2.00001]), .00002)
        for expected, actual in (([1., 2.], [1., 2.01]), ([1., 2.], [1.]), ([1., 2.], [1., np.nan])):
            with self.assertRaises(ValueError):
                verify_replayed_predictions(expected, actual)

    def test_checkpoint_finiteness_and_constraints(self):
        import torch
        from gruxnet.eegnet_regression import EEGNetRegression
        model = EEGNetRegression(); model.constrain(); verify_reloaded_state(model)
        with torch.no_grad():
            model.head.weight[0, 0] = .3
        with self.assertRaises(ValueError):
            verify_reloaded_state(model)
        model.constrain()
        with torch.no_grad():
            model.bn1.running_mean[0] = float('nan')
        with self.assertRaises(ValueError):
            verify_reloaded_state(model)

    def test_escaping_root_or_unlisted_original_denied_before_metadata(self):
        from pathlib import Path
        workspace = Path.cwd()
        allowed = {('sub-001', 'data.bdf')}
        with self.assertRaises(PermissionError):
            guard_source_original_path(workspace, workspace.parent / 'outside', 'sub-001', 'data.bdf', allowed)
        with self.assertRaises(PermissionError):
            guard_source_original_path(workspace, workspace / 'Data', 'sub-002', 'data.bdf', allowed)
        with self.assertRaises(PermissionError):
            guard_source_original_path(workspace, workspace / 'Data', 'sub-001', '../ratings.mat', allowed)
if __name__ == '__main__':
    unittest.main()
