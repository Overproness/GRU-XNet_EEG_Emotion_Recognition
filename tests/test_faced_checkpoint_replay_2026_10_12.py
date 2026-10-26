"""Manufactured checkpoint-replay fixtures only; no original data/cache reads."""
import csv
import io
import unittest

import numpy as np
import torch

from gruxnet.eegnet_regression import EEGNetRegression
from scripts.replay_faced_diagnostic_checkpoints_2026_10_12 import (
    existing_context_prediction, fixture_pairs, guarded_fixture_predictions,
    independent_forward, RATE, EPOCH, SEEDS)


def synthetic_source():
    people = [f'synthetic-source{i:03d}' for i in range(70)]
    pairs = [(p, v) for p in people for v in range(1, 9)]
    programme = {'rotation_manifests': [
        {'rotation': k, 'arms': {'source_fit': [{'subject': p, 'vid': v} for p, v in pairs]}}
        for k in range(3)]}
    metadata = {p: {'role': 'source', 'acquisition_group': f'synthetic-acq{i % 3}',
                    'family': f'synthetic-film{p[1]}'} for i, p in enumerate(pairs)}
    return programme, metadata


class CheckpointReplayTests(unittest.TestCase):
    def test_fixtures_are_deterministic_source_only_and_cover_acquisition(self):
        programme, metadata = synthetic_source()
        result = fixture_pairs(programme, metadata)
        self.assertEqual(result, fixture_pairs(programme, metadata))
        for chosen in result.values():
            self.assertEqual(len(set(chosen)), 4)
            self.assertEqual(len({metadata[p]['acquisition_group'] for p in chosen}), 3)

    def test_protected_identity_rejected_before_numeric_access(self):
        programme, metadata = synthetic_source()
        first = next(iter(metadata))
        metadata[first]['role'] = 'confirmation'
        with self.assertRaises(PermissionError):
            fixture_pairs(programme, metadata)

    def test_independent_functional_forward_matches_model_on_synthetic_windows(self):
        torch.set_num_threads(2)
        torch.manual_seed(731021)
        model = EEGNetRegression().eval()
        with torch.no_grad():
            for name in ('bn1', 'bn2', 'bn3'):
                layer = getattr(model, name)
                layer.running_mean.copy_(torch.randn_like(layer.running_mean) * .15)
                layer.running_var.copy_(torch.rand_like(layer.running_var) + .3)
            trials = torch.randn(3, 7, 30, 512)
            expected = model(trials)
            actual = independent_forward(model.state_dict(), trials)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)

    def test_forward_rejects_nonfinite_or_wrong_geometry(self):
        with self.assertRaises(ValueError):
            independent_forward({}, torch.zeros(1, 6, 30, 512))
        with self.assertRaises(ValueError):
            independent_forward({}, torch.full((1, 7, 30, 512), float('nan')))

    def test_existing_context_prediction_uses_clipped_fitted_state(self):
        state = io.BytesIO()
        coef = np.zeros(13); coef[-1] = 2
        np.savez(state, mean=np.ones(13), scale=np.ones(13) * 2,
                 variance=np.ones(13), coef=coef, intercept=np.array(1.))
        state.seek(0)
        vectors = np.ones((3, 13)); vectors[:, -1] = [-10, 2, 20]
        np.testing.assert_array_equal(existing_context_prediction(state, vectors), [0, 2, 7])

    def test_only_declared_source_fixture_predictions_are_converted(self):
        programme, metadata = synthetic_source()
        fixtures = fixture_pairs(programme, metadata)
        rows = []
        for k, pairs in fixtures.items():
            for seed in SEEDS:
                for arm in ('source_fit', 'source_inference'):
                    for p in pairs:
                        rows.append({'rotation': k, 'readout': 'context_residual', 'learning_rate': RATE,
                            'seed': seed, 'epoch': EPOCH, 'arm': arm, 'subject': p[0], 'vid': p[1],
                            'family': metadata[p]['family'], 'acquisition_group': metadata[p]['acquisition_group'],
                            'target': 'NEVER_CONVERT_TARGET', 'prediction_raw': 1.234,
                            'eeg_available': True, 'fallback': False})
        poison = {**rows[0], 'subject': 'synthetic-protected', 'prediction_raw': 'NEVER_CONVERT_PREDICTION'}
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows([poison] + rows); stream.seek(0)
        result = guarded_fixture_predictions(stream, fixtures, metadata)
        self.assertEqual(len(result), 48)
        self.assertEqual(set(result.values()), {1.234})

    def test_fixture_family_guard_precedes_prediction_conversion(self):
        programme, metadata = synthetic_source()
        fixtures = fixture_pairs(programme, metadata)
        p = fixtures[0][0]
        row = {'rotation': 0, 'readout': 'context_residual', 'learning_rate': RATE,
               'seed': SEEDS[0], 'epoch': EPOCH, 'arm': 'source_inference',
               'subject': p[0], 'vid': p[1], 'family': 'wrong-family',
               'acquisition_group': metadata[p]['acquisition_group'],
               'prediction_raw': 'NEVER_CONVERT_PREDICTION', 'eeg_available': True, 'fallback': False}
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        stream.seek(0)
        with self.assertRaises(PermissionError):
            guarded_fixture_predictions(stream, fixtures, metadata)


if __name__ == '__main__':
    unittest.main()
