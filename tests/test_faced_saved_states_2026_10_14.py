"""Manufactured guards and saved-state stationarity tests; no original data."""
import unittest

import numpy as np

from scripts.qualify_faced_saved_states_2026_10_14 import (
    context_vector, independent_priors, inspect_equations, population_moments,
    source_targets, validate_state_arrays,
)


class FloatTrap:
    def __float__(self):
        raise AssertionError("Numeric conversion occurred before complete guard")


def synthetic_state():
    t = np.linspace(-1, 1, 20)
    x = np.column_stack([t + i for i in range(12)] + [np.ones(20) * 8])
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    scale = np.sqrt(var)
    scale[-1] = 1
    z = (x - mean) / scale
    y = 3 + .4 * z[:, 0]
    # Rank-one standardized design: analytic coefficients, no fitting/solver.
    coef = np.r_[np.repeat(20 * .4 / (20 * 12 + 100), 12), 0.]
    state = {"coefficient": coef, "intercept": np.asarray(3.), "mean": mean,
             "variance": var, "scale": scale, "alpha": np.asarray(100.)}
    return x, y, state


class SavedStateQualificationTests(unittest.TestCase):
    def test_analytic_stationarity_and_prediction_replay(self):
        x, y, state = synthetic_state()
        receipt, prediction = inspect_equations(x, y, state)
        self.assertLess(receipt["normal_equation_relative_residual"], 1e-12)
        self.assertLess(receipt["matrix_vs_row_sum_max_prediction_discrepancy"], 1e-12)
        self.assertEqual(prediction.shape, (20,))

    def test_saved_coefficient_corruption_rejected(self):
        x, y, state = synthetic_state()
        state["coefficient"][0] += .01
        with self.assertRaisesRegex(ValueError, "stationarity"):
            inspect_equations(x, y, state)

    def test_intercept_corruption_rejected(self):
        x, y, state = synthetic_state()
        state["intercept"] = np.asarray(3.01)
        with self.assertRaisesRegex(ValueError, "intercept"):
            inspect_equations(x, y, state)

    def test_scaler_corruption_rejected(self):
        x, y, state = synthetic_state()
        state["scale"][0] *= 2
        with self.assertRaisesRegex(ValueError, "scaler"):
            inspect_equations(x, y, state)

    def test_feature_column_order_corruption_rejected(self):
        x, y, state = synthetic_state()
        x = x[:, list(reversed(range(13)))]
        with self.assertRaisesRegex(ValueError, "scaler"):
            inspect_equations(x, y, state)

    def test_population_variance_uses_ddof_zero_and_constant_scale(self):
        x = np.array([[2., 1.], [4., 1.]])
        mean, variance, scale = population_moments(x)
        np.testing.assert_array_equal(mean, [3, 1])
        np.testing.assert_array_equal(variance, [1, 0])
        np.testing.assert_array_equal(scale, [1, 1])

    def test_extra_saved_array_rejected(self):
        _, _, state = synthetic_state()
        state["unused"] = np.asarray(1.)
        with self.assertRaises(ValueError):
            validate_state_arrays(state, 13)

    def test_nonfinite_coefficient_rejected(self):
        _, _, state = synthetic_state()
        state["coefficient"][0] = np.nan
        with self.assertRaises(ValueError):
            validate_state_arrays(state, 13)

    def test_negative_variance_rejected(self):
        _, _, state = synthetic_state()
        state["variance"][0] = -1
        with self.assertRaises(ValueError):
            validate_state_arrays(state, 13)

    def test_wrong_alpha_rejected(self):
        _, _, state = synthetic_state()
        state["alpha"] = np.asarray(10.)
        with self.assertRaises(ValueError):
            validate_state_arrays(state, 13)

    def test_context_order(self):
        row = {"assigned_category": "Fear", "acquisition_group": "modern_1000",
               "catalogue_duration_seconds": 51, "trial": 5}
        value = context_vector(row)
        self.assertEqual(value.shape, (13,))
        np.testing.assert_array_equal(np.flatnonzero(value[:11]), [3, 10])
        np.testing.assert_array_equal(value[11:], [51, 5])

    def test_context_invalid_duration_rejected(self):
        row = {"assigned_category": "Fear", "acquisition_group": "modern_1000",
               "catalogue_duration_seconds": np.nan, "trial": 5}
        with self.assertRaises(PermissionError):
            context_vector(row)

    def test_complete_target_metadata_guard_before_early_float(self):
        meta = {("source", 2): {"role": "source", "family": "f2", "acquisition_group": "a"},
                ("source", 3): {"role": "source", "family": "f3", "acquisition_group": "a"}}
        rows = [{"subject": "source", "vid": 2, "arm": "source_oof", "family": "f2", "acquisition_group": "a", "target": FloatTrap()},
                {"subject": "source", "vid": 3, "arm": "source_oof", "family": "incorrect", "acquisition_group": "a", "target": 1}]
        with self.assertRaises(PermissionError):
            source_targets(rows, meta)

    def test_non_source_targets_not_converted(self):
        meta = {("source", 2): {"role": "source", "family": "f2", "acquisition_group": "a"}}
        rows = [{"subject": "protected", "vid": 2, "arm": "source_oof", "target": FloatTrap()},
                {"subject": "source", "vid": 2, "arm": "validation_unseen", "target": FloatTrap()},
                {"subject": "source", "vid": 2, "arm": "source_oof", "family": "f2", "acquisition_group": "a", "target": 2}]
        targets, receipt = source_targets(rows, meta)
        self.assertEqual(targets[("source", 2)], 2)
        self.assertEqual(receipt["rows_ignored_before_conversion"], 2)

    def test_conflicting_source_targets_rejected(self):
        meta = {("source", 2): {"role": "source", "family": "f2", "acquisition_group": "a"}}
        base = {"subject": "source", "vid": 2, "arm": "source_oof", "family": "f2", "acquisition_group": "a"}
        with self.assertRaises(ValueError):
            source_targets([{**base, "target": 1}, {**base, "target": 2}], meta)

    def test_source_conversion_row_budget_before_float(self):
        meta = {("source", 2): {"role": "source", "family": "f2", "acquisition_group": "a"}}
        row = {"subject": "source", "vid": 2, "arm": "source_oof", "family": "f2", "acquisition_group": "a", "target": FloatTrap()}
        with self.assertRaises(PermissionError):
            source_targets([row], meta, {"distinct": 1, "numeric_rows": 2, "ignored_rows": 0})

    def test_ignored_row_budget_before_float(self):
        meta = {("source", 2): {"role": "source", "family": "f2", "acquisition_group": "a"}}
        row = {"subject": "source", "vid": 2, "arm": "source_oof", "family": "f2", "acquisition_group": "a", "target": FloatTrap()}
        with self.assertRaises(PermissionError):
            source_targets([row, {"subject": "protected", "target": FloatTrap()}], meta,
                           {"distinct": 1, "numeric_rows": 1, "ignored_rows": 0})

    def test_incomplete_target_rectangle_before_float(self):
        meta = {("source", 2): {"role": "source", "family": "f2", "acquisition_group": "a"},
                ("source", 3): {"role": "source", "family": "f3", "acquisition_group": "a"}}
        row = {"subject": "source", "vid": 2, "arm": "source_oof", "family": "f2", "acquisition_group": "a", "target": FloatTrap()}
        with self.assertRaises(PermissionError):
            source_targets([row], meta)

    def test_prior_medians_even_odd_and_video_json_keys(self):
        pairs = [("a", 2), ("b", 2), ("a", 3)]
        meta = {p: {"assigned_category": "Fear", "assigned_valence": "negative"} for p in pairs}
        prior = independent_priors(pairs, {pairs[0]: 1., pairs[1]: 3., pairs[2]: 6.}, meta)
        self.assertEqual(prior["global"], 3)
        self.assertEqual(prior["category"]["Fear"], 3)
        self.assertEqual(prior["video"]["2"], 2)
        self.assertEqual(prior["video"]["3"], 6)


if __name__ == "__main__":
    unittest.main()
