"""Manufactured arithmetic and pre-conversion role guards, with no data I/O."""
from copy import deepcopy
from math import sqrt
import unittest

from scipy.stats import t
from scripts.assess_faced_benchmark_precision_2026_10_13 import (
    halfwidth, hypothetical_required_n, participant_errors, qualify_rows,
    unknown_category_sd,
)


class FloatTrap:
    def __float__(self):
        raise AssertionError("Forbidden numerical conversion")


def fixture():
    pairs = {("source-1", 1), ("source-2", 1)}
    metadata = {p: {"family": "a", "assigned_category": "Anger", "acquisition_group": "legacy"} for p in pairs}
    cells = {0: {"held": sorted(pairs), "person_fold": 0, "material_assignment": 0}}
    rows = []
    for subject, vid in sorted(pairs):
        for model in ("context_only", "late_joint", "distributed_joint", "category_median"):
            for alpha in ((None,) if model == "category_median" else (.1, 10., 100.)):
                rows.append({"subject": subject, "vid": vid, "family": "a", "assigned_category": "Anger",
                             "acquisition_group": "legacy", "cell": 0, "person_fold": 0,
                             "material_assignment": 0, "role": "source", "model": model,
                             "alpha": alpha, "target": 2., "prediction": 3.})
    return rows, metadata, cells, pairs


class RoleQualificationTests(unittest.TestCase):
    def checked(self, rows):
        _, metadata, cells, pairs = fixture()
        return qualify_rows(rows, metadata, cells, pairs, expected_total=20)

    def test_complete_fixed_source_panel(self):
        rows, _, _, _ = fixture()
        result = self.checked(rows)
        self.assertEqual(len(result["distributed_joint"]), 2)

    def test_denied_roles_rejected_before_float(self):
        for role in ("validation", "protected", "source_oof", None):
            rows, _, _, _ = fixture()
            rows[0].update(role=role, target=FloatTrap(), prediction=FloatTrap())
            with self.assertRaises(PermissionError):
                self.checked(rows)

    def test_unknown_person_rejected_before_float(self):
        rows, _, _, _ = fixture()
        rows[0].update(subject="reserved-person", target=FloatTrap(), prediction=FloatTrap())
        with self.assertRaises(PermissionError):
            self.checked(rows)

    def test_later_denied_row_precedes_first_selected_numeric_conversion(self):
        rows, _, _, _ = fixture()
        rows[2]["target"] = FloatTrap()
        rows[-1].update(role="protected", target=FloatTrap(), prediction=FloatTrap())
        with self.assertRaises(PermissionError):
            self.checked(rows)

    def test_invalid_clip_identity_never_converted(self):
        for value in (FloatTrap(), True, "1", 99):
            rows, _, _, _ = fixture()
            rows[0].update(vid=value, target=FloatTrap(), prediction=FloatTrap())
            with self.assertRaises(PermissionError):
                self.checked(rows)

    def test_wrong_context_rejected_before_float(self):
        for field, value in (("family", "protected-film"), ("assigned_category", "Fear"),
                             ("acquisition_group", "wrong"), ("person_fold", 1), ("material_assignment", 1)):
            rows, _, _, _ = fixture()
            rows[0].update({field: value, "target": FloatTrap(), "prediction": FloatTrap()})
            with self.assertRaises(PermissionError):
                self.checked(rows)

    def test_missing_or_duplicate_candidates_rejected(self):
        rows, _, _, _ = fixture()
        with self.assertRaises(ValueError):
            self.checked(rows[:-1])
        rows[-1] = deepcopy(rows[-2])
        with self.assertRaises(ValueError):
            self.checked(rows)

    def test_paired_targets_agree(self):
        rows, _, _, _ = fixture()
        rows[2]["target"] = 5.
        with self.assertRaises(ValueError):
            self.checked(rows)

    def test_invalid_selected_numeric_values_rejected(self):
        for value in (float("nan"), float("inf"), -1., 8., True):
            rows, _, _, _ = fixture()
            rows[2]["prediction"] = value
            with self.assertRaises(ValueError):
                self.checked(rows)

    def test_other_alphas_are_not_reanalyzed(self):
        rows, _, _, _ = fixture()
        rows[0]["target"], rows[1]["prediction"] = FloatTrap(), FloatTrap()
        self.checked(rows)


class PrecisionArithmeticTests(unittest.TestCase):
    def test_losses_precede_prediction_average(self):
        panel = {("p", 1): {"family": "a", "target": 3., "prediction": 1.},
                 ("p", 2): {"family": "a", "target": 3., "prediction": 5.},
                 ("p", 3): {"family": "b", "target": 0., "prediction": 0.}}
        self.assertEqual(participant_errors(panel), {"p": 1.})

    def test_equal_family_weight_unequal_clip_count(self):
        panel = {("p", 1): {"family": "a", "target": 0., "prediction": 2.},
                 ("p", 2): {"family": "a", "target": 0., "prediction": 4.},
                 ("p", 3): {"family": "b", "target": 0., "prediction": 1.}}
        self.assertEqual(participant_errors(panel), {"p": 2.})

    def test_df32_halfwidth(self):
        self.assertAlmostEqual(halfwidth(.7, 33), t.ppf(.975, 32) * .7 / sqrt(33))

    def test_hypothetical_resolution_minimal_n(self):
        n = hypothetical_required_n(.6, .1)
        self.assertLessEqual(halfwidth(.6, n), .1)
        self.assertGreater(halfwidth(.6, n - 1), .1)
        self.assertEqual(hypothetical_required_n(0., .05), 2)

    def test_hypothetical_count_cap_and_invalid_resolution(self):
        self.assertIsNone(hypothetical_required_n(7., .0001, cap=100))
        for sd, res in ((-1., .1), (.1, 0.), (float("nan"), .1)):
            with self.assertRaises(ValueError):
                hypothetical_required_n(sd, res)

    def test_unknown_family_covariance_edges(self):
        self.assertAlmostEqual(unknown_category_sd(1., 2., 1.), 1.25)
        self.assertAlmostEqual(unknown_category_sd(1., 2., -1.), .25)
        self.assertAlmostEqual(unknown_category_sd(1., 2., 0.), sqrt(.75 ** 2 + .5 ** 2))
        self.assertAlmostEqual(unknown_category_sd(1., 3., -1.), 0.)

    def test_invalid_transport_or_count(self):
        with self.assertRaises(ValueError):
            unknown_category_sd(1., 1., 1.1)
        with self.assertRaises(ValueError):
            halfwidth(1., 1)


if __name__ == "__main__":
    unittest.main()
