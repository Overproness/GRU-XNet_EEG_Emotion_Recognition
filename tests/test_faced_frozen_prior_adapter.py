"""Manufactured exact-schema and unchanged-state prior-adapter qualification."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import numpy as np

import gruxnet.faced_frozen_inference as frozen
import gruxnet.faced_frozen_prior_adapter as adapter


SOURCE_PAIRS = [(f"source-{p:03d}", v) for p in range(70)
                for v in sorted(frozen.DEVELOPMENT_CLIPS)]
SOURCE_SHA = frozen.sha256_bytes(
    json.dumps(SOURCE_PAIRS, separators=(",", ":")).encode())


def source_shaped_prior():
    return {"global": 3.5, "coarse": {"Negative": 2.25, "Positive": 4.75},
            "category": {c: i * .75 + .125 for i, c in enumerate(frozen.CATEGORIES)},
            "video": {str(v): 3.25 for v in frozen.DEVELOPMENT_CLIPS}}


class FrozenPriorAdapterTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "_tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = scratch / ("prior-adapter-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.bindings = {}
        for model in frozen.METHODS[1:]:
            dimension = 13 if model == "context_only" else 163
            values = {
                "coefficient": np.zeros(dimension, dtype=np.float64),
                "intercept": np.asarray(3., dtype=np.float64),
                "mean": np.zeros(dimension, dtype=np.float64),
                "scale": np.ones(dimension, dtype=np.float64),
                "variance": np.ones(dimension, dtype=np.float64),
                "alpha": np.asarray(100., dtype=np.float64),
            }
            if model != "context_only":
                values["eeg_imputation_mean"] = np.full(150, .5, dtype=np.float64)
            path = self.directory / (model + ".npz")
            np.savez_compressed(path, **values)
            digest = frozen.sha256_bytes(path.read_bytes())
            self.bindings[model] = {"path": path, "sha256": digest, "record": {
                "model": model, "stage": "full_source_state_qualification",
                "alpha": 100., "training_rows": 980,
                "feature_dimension": dimension, "training_pairs": SOURCE_PAIRS,
                "source_identity_sha256": SOURCE_SHA,
                "private_state_sha256": digest,
            }}
        self.prior_path = self.directory / "prior.json"
        self.write_prior(source_shaped_prior())

    def write_prior(self, value):
        payload = frozen.canonical_bytes(value)
        self.prior_path.write_bytes(payload)
        self.prior_sha = frozen.sha256_bytes(payload)

    def load(self, bindings=None, prior_sha=None, source_sha=SOURCE_SHA):
        return adapter.load_frozen_bundle_v2(
            self.bindings if bindings is None else bindings,
            self.prior_path, self.prior_sha if prior_sha is None else prior_sha,
            source_sha)

    def test_exact_uppercase_prior_and_state_values_preserved_without_rewriting(self):
        before = self.prior_path.read_bytes()
        loaded = self.load()
        self.assertIsInstance(loaded, frozen.FrozenBundle)
        self.assertEqual(dict(loaded.category_prior), source_shaped_prior()["category"])
        self.assertEqual(loaded.prior_sha256, self.prior_sha)
        self.assertEqual(loaded.source_identity_sha256, SOURCE_SHA)
        self.assertEqual(self.prior_path.read_bytes(), before)
        for model, state in loaded.states.items():
            self.assertEqual(state.state_sha256, self.bindings[model]["sha256"])
            expected = frozen.load_frozen_state(
                self.bindings[model]["path"], self.bindings[model]["sha256"],
                self.bindings[model]["record"], SOURCE_SHA)
            for field in ("coefficient", "intercept", "mean", "scale", "variance"):
                self.assertEqual(getattr(state, field).tobytes(), getattr(expected, field).tobytes())

    def test_original_loader_remains_strict_and_does_not_accept_uppercase(self):
        with self.assertRaises(ValueError):
            frozen.load_frozen_bundle(self.bindings, self.prior_path,
                                      self.prior_sha, SOURCE_SHA)

    def test_lowercase_mixed_and_extra_coarse_keys_rejected(self):
        for coarse in ({"negative": 2., "positive": 4.},
                       {"Negative": 2., "positive": 4.},
                       {"Negative": 2., "Positive": 4., "extra": 3.}):
            with self.subTest(coarse=coarse):
                value = source_shaped_prior(); value["coarse"] = coarse
                self.write_prior(value)
                with self.assertRaises(ValueError): self.load()

    def test_top_level_schema_extras_missing_or_wrong_container_rejected(self):
        value = source_shaped_prior()
        missing = deepcopy(value); del missing["global"]
        for changed in (dict(value, extra=1), missing, [], None):
            with self.subTest(prior=changed):
                self.write_prior(changed)
                with self.assertRaises(ValueError): self.load()

    def test_exact_eight_categories_and_fourteen_source_video_keys_required(self):
        for field in ("category", "video"):
            for mutation in ("missing", "extra", "container"):
                value = source_shaped_prior()
                if mutation == "missing": value[field].pop(next(iter(value[field])))
                elif mutation == "extra": value[field]["unqualified"] = 3.
                else: value[field] = list(value[field].values())
                self.write_prior(value)
                with self.subTest(field=field, mutation=mutation):
                    with self.assertRaises(ValueError): self.load()

    def test_invalid_numeric_medians_in_every_section_rejected(self):
        for field in ("global", "coarse", "category", "video"):
            for invalid in (-.001, 7.001, True, "3.5", None):
                value = source_shaped_prior()
                if field == "global": value[field] = invalid
                else: value[field][next(iter(value[field]))] = invalid
                self.write_prior(value)
                with self.subTest(field=field, value=invalid):
                    with self.assertRaises(ValueError): self.load()

    def test_zero_and_seven_are_valid_bounded_medians(self):
        value = source_shaped_prior(); value["global"] = 0
        value["coarse"] = {"Negative": 0, "Positive": 7}
        value["category"]["Amusement"] = 7
        value["video"][str(min(frozen.DEVELOPMENT_CLIPS))] = 0
        self.write_prior(value)
        self.assertEqual(self.load().category_prior["Amusement"], 7.)

    def test_nonfinite_json_constants_and_numeric_overflow_rejected(self):
        for spelling in ("NaN", "Infinity", "-Infinity", "1e999"):
            payload = json.dumps(source_shaped_prior()).replace('"global": 3.5', '"global": ' + spelling).encode()
            self.prior_path.write_bytes(payload); self.prior_sha = frozen.sha256_bytes(payload)
            with self.subTest(spelling=spelling):
                with self.assertRaises(ValueError): self.load()

    def test_duplicate_top_level_and_nested_keys_rejected(self):
        payload = json.dumps(source_shaped_prior())
        for changed in (payload.replace('"global": 3.5', '"global": 3.5, "global": 4'),
                        payload.replace('"Negative": 2.25', '"Negative": 2.25, "Negative": 3'),
                        payload.replace('"Amusement": 0.125', '"Amusement": 0.125, "Amusement": 2')):
            encoded = changed.encode(); self.prior_path.write_bytes(encoded)
            self.prior_sha = frozen.sha256_bytes(encoded)
            with self.assertRaises(ValueError): self.load()

    def test_prior_hash_and_file_cap_enforced(self):
        with self.assertRaises(ValueError): self.load(prior_sha="f" * 64)
        payload = self.prior_path.read_bytes() + b" " * 65537
        self.prior_path.write_bytes(payload); self.prior_sha = frozen.sha256_bytes(payload)
        with self.assertRaises(ValueError): self.load()

    def test_invalid_hash_format_rejected_before_state_loading(self):
        with patch.object(adapter, "load_frozen_state", side_effect=AssertionError("Premature state read")):
            with self.assertRaises(ValueError): self.load(prior_sha="INVALID")
            with self.assertRaises(ValueError): self.load(source_sha="INVALID")

    def test_all_binding_shapes_guarded_before_any_state_read(self):
        malformed = deepcopy(self.bindings)
        malformed["distributed_joint"]["unexpected"] = True
        with patch.object(adapter, "load_frozen_state", side_effect=AssertionError("Premature state read")):
            with self.assertRaises(ValueError): self.load(bindings=malformed)

    def test_missing_extra_and_wrong_model_bindings_rejected(self):
        missing = deepcopy(self.bindings); missing.pop("late_joint")
        extra = dict(self.bindings, another=self.bindings["late_joint"])
        wrong = deepcopy(self.bindings); wrong["distributed_joint"]["record"]["model"] = "late_joint"
        for bindings in (missing, extra, wrong):
            with self.subTest(keys=list(bindings)):
                with self.assertRaises((ValueError, PermissionError)): self.load(bindings=bindings)

    def test_unqualified_fit_record_and_state_hash_still_rejected(self):
        for field, invalid in (("alpha", 10.), ("training_rows", 448),
                               ("stage", "source_cv"),
                               ("source_identity_sha256", "e" * 64)):
            bindings = deepcopy(self.bindings)
            bindings["context_only"]["record"][field] = invalid
            with self.subTest(field=field):
                with self.assertRaises(PermissionError): self.load(bindings=bindings)
        bindings = deepcopy(self.bindings)
        bindings["context_only"]["path"].write_bytes(b"changed")
        with self.assertRaises(ValueError): self.load(bindings=bindings)


if __name__ == "__main__":
    unittest.main()
