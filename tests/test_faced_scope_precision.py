import math
import unittest

import numpy as np

from scripts.analyze_faced_scope_precision import guard_row, paired_matrix, planning, planning_se, variance_components


class ScopePrecisionTests(unittest.TestCase):
    def test_additive_components_match_known_factor_variances(self):
        person = np.array([-2., 0., 2.])
        film = np.array([-3., -1., 1., 3.])
        result = variance_components(5 + person[:, None] + film[None, :])
        self.assertAlmostEqual(result["mean_gain"], 5)
        self.assertAlmostEqual(result["raw_components"]["participant"], person.var(ddof=1))
        self.assertAlmostEqual(result["raw_components"]["film"], film.var(ddof=1))
        self.assertAlmostEqual(result["raw_components"]["residual"], 0)

    def test_negative_estimates_remain_visible(self):
        result = variance_components(np.array([[1., -1.], [-1., 1.]]))
        self.assertEqual(result["raw_components"], {"participant": -2., "film": -2., "residual": 4.})
        self.assertEqual(result["planning_components"], {"participant": 0., "film": 0., "residual": 4.})

    def test_variance_projection_and_inflation(self):
        v = {"participant": 4., "film": 9., "residual": 16.}
        self.assertAlmostEqual(planning_se(v, 33, 8) ** 2, 4/33 + 9/8 + 16/(33*8))
        self.assertAlmostEqual(planning_se(v, 33, 8, 4), 2*planning_se(v,33,8))

    def test_film_floor_prevents_spurious_finite_sample_size(self):
        result = planning({"participant": 0., "film": 1., "residual": 0.}, 0)
        self.assertTrue(all(r["nominal_required_participants_at_eight_films"] is None for r in result["margin_sensitivity"]))
        self.assertAlmostEqual(result["crossed_se"], math.sqrt(1/8))
        self.assertEqual(result["fixed_roster_participant_se"],0)

    def test_protected_person_is_rejected_before_outcome_decode(self):
        row = {"subject":"protected","vid":"2","family":"f", "target":"must not decode", "prediction":"must not decode"}
        with self.assertRaisesRegex(ValueError,"participant"):
            guard_row(row,{"source"},{2:{"source_film_sha256":"f"}},{2})

    def test_protected_film_and_wrong_family_rejected(self):
        row = {"subject":"source","vid":"7","family":"f", "target":"must not decode"}
        with self.assertRaisesRegex(ValueError,"material"):
            guard_row(row,{"source"},{7:{"source_film_sha256":"f"}},{2})
        row["vid"]="2"
        with self.assertRaisesRegex(ValueError,"family"):
            guard_row(row,{"source"},{2:{"source_film_sha256":"other"}},{2})

    def test_missing_trials_cannot_shrink_paired_denominator(self):
        with self.assertRaisesRegex(ValueError,"denominator"):
            paired_matrix({}, {}, [], [])

    def test_target_mismatch_rejected_and_fallback_retained(self):
        people = [f"p{i}" for i in range(20)]
        families = [f"f{j}" for j in range(6)]
        ctx = {(p,j):(f,1.,2.,"false") for p in people for j,f in enumerate(families)}
        joint = {(p,j):(f,1.,1.5,"true") for p in people for j,f in enumerate(families)}
        matrix,cmae,jmae,fallback = paired_matrix(ctx,joint,people,families)
        self.assertEqual(fallback,120)
        self.assertEqual(matrix.shape,(20,6))
        self.assertEqual(cmae,1.)
        self.assertEqual(jmae,.5)
        joint[people[0],0] = (families[0],0.,1.5,"true")
        with self.assertRaisesRegex(ValueError,"target"):
            paired_matrix(ctx,joint,people,families)

    def test_nonfinite_or_unbalanced_variance_input_rejected(self):
        for x in [np.array([[np.nan,1.],[2.,3.]]),np.ones((1,6))]:
            with self.assertRaises(ValueError):
                variance_components(x)


if __name__ == "__main__":
    unittest.main()
