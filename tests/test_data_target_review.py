"""Meaningful preparation/identity boundaries for the metadata-only review."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("review_data_targets", Path(__file__).resolve().parents[1]/"scripts/review_data_targets.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewBoundaries(unittest.TestCase):
    def test_excluded_labels_are_never_inspected(self):
        rows = [
            {"subject_id":"DEAP:S01","material":"T01","trial_id":"DEAP:S01:T01","label":0},
            {"subject_id":"DEAP:S01","material":"T02","trial_id":"DEAP:S01:T02","label":object()},
            {"subject_id":"DEAP:S02","material":"T01","trial_id":"DEAP:S02:T01","label":object()},
        ]
        selected=review.select_source_rows(rows,{"DEAP:S01"},{"T01"})
        self.assertEqual(selected,[rows[0]])

    def test_repeated_predictions_cannot_change_trial_label(self):
        row={"subject_id":"GAMEEMO:S01","trial_id":"GAMEEMO:S01:G1","label":"0","positive_probability":".1"}
        with self.assertRaises(ValueError):
            review.metadata([row,{**row,"label":"1"}],"GAMEEMO")
        self.assertEqual(review.metadata([row],"GAMEEMO"),review.metadata([{**row,"positive_probability":"NaN"}],"GAMEEMO"))

    def test_invalid_native_label_and_identity_rejected(self):
        row={"subject_id":"SEEDIV:S01","trial_id":"SEEDIV:S01:R1:T01","original_label":"4"}
        with self.assertRaises(ValueError):review.metadata([row],"SEEDIV")
        with self.assertRaises(ValueError):review.metadata([{**row,"original_label":"0","subject_id":"SEEDIV:S02"}],"SEEDIV")


if __name__ == "__main__":unittest.main()
