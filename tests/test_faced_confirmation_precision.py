"""Small deterministic fixtures only; no study simulation or empirical data reads."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/faced_confirmation_precision_2026_10_12.py"
spec = importlib.util.spec_from_file_location("confirmation_precision", SCRIPT)
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


class ConfirmationPrecisionTests(unittest.TestCase):
    def test_crossed_weights_equal_explicit_cartesian_resampling(self):
        matrix = np.array([[0., 2.], [3., 5.], [8., 9.]])
        person_indices = np.array([[0, 0, 2], [1, 2, 1]])
        title_indices = np.array([[1, 1], [0, 1]])
        pc = np.array([[2, 0, 1], [0, 2, 1]])
        tc = np.array([[0, 2], [1, 1]])
        expected = [matrix[p[:, None], t[None, :]].mean() for p, t in zip(person_indices, title_indices)]
        np.testing.assert_allclose(precision.crossed_samples(matrix, pc, tc), expected)

    def test_constant_gain_translation_moves_bootstrap_means_exactly(self):
        matrix = np.array([[0.1, -0.2], [0.3, 0.8]])
        pc = np.array([[2, 0], [1, 1], [0, 2]])
        tc = np.array([[1, 1], [0, 2], [2, 0]])
        np.testing.assert_allclose(precision.crossed_samples(matrix + .15, pc, tc),
                                   precision.crossed_samples(matrix, pc, tc) + .15)

    def test_invalid_or_incomplete_bootstrap_weights_rejected(self):
        for pc, tc in [(np.array([[1, 0]]), np.array([[1, 1]])),
                       (np.array([[3, -1]]), np.array([[1, 1]])),
                       (np.array([[1, 1]]), np.array([[1, 1, 0]]))]:
            with self.assertRaises(ValueError):
                precision.crossed_samples(np.zeros((2, 2)), pc, tc)

    def test_analytic_variance_and_more_titles_floor(self):
        scenario = {"subject_sd": .3, "title_sd": .2, "interaction_sd": .4}
        original = precision.analytic_variance(scenario, 33, 8)
        more_people = precision.analytic_variance(scenario, 66, 8)
        more_titles = precision.analytic_variance(scenario, 33, 16)
        self.assertAlmostEqual(original["mean_variance"], .09 / 33 + .04 / 8 + .16 / 264)
        self.assertEqual(original["infinite_people_title_standard_error_floor"],
                         more_people["infinite_people_title_standard_error_floor"])
        self.assertLess(more_titles["infinite_people_title_standard_error_floor"],
                        original["infinite_people_title_standard_error_floor"])

    def test_aggregate_recount_true_gain_and_zero_exclusion_differ(self):
        records = []
        for sindex in range(len(precision.SCENARIOS)):
            for dindex in range(len(precision.DIMENSIONS)):
                for replicate in range(precision.REPLICATES):
                    records.append({"scenario_index": sindex, "dimension_index": dindex,
                                    "replicate": replicate, "base_mean": .01,
                                    "base_interval_lower": -.02, "base_interval_upper": .08})
        result = precision.summarize(records)
        self.assertEqual(len(result), 48)
        self.assertEqual(result[0]["coverage_count"], 256)
        self.assertEqual(result[0]["zero_exclusion_count"], 0)
        self.assertEqual(result[1]["zero_exclusion_count"], 256)
        self.assertEqual(result[0]["mean_interval_width"], result[2]["mean_interval_width"])

    def test_activation_denies_before_metadata_or_random_draw(self):
        p = SCRIPT.parent
        with patch.object(precision, "load_json", return_value={"status": "declared_not_launched"}), \
             patch.object(precision, "metadata_projection", side_effect=AssertionError("metadata touched")), \
             patch.object(precision, "simulate", side_effect=AssertionError("random draws touched")):
            with self.assertRaises(PermissionError):
                precision.run(p, p / "activation.json", p)

    def test_confirmation_permission_cannot_be_hidden_by_other_fields(self):
        p = SCRIPT.parent
        activation = {
            "status": "active_confirmation_precision_simulation",
            "precision_simulation_authorized": True, "confirmation_access": True,
            "research_question_changed": False,
        }
        with patch.object(precision, "load_json", return_value=activation):
            with self.assertRaises(PermissionError):
                precision.validate_activation(p / "missing_plan.json", p / "activation.json", p)

    def test_missing_or_duplicate_replicates_rejected(self):
        with self.assertRaises(ValueError):
            precision.summarize([])


if __name__ == "__main__":
    unittest.main()
