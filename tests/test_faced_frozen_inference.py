"""Manufactured boundary and hand-calculated frozen inference qualification."""
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

import gruxnet.faced_frozen_inference as module


SOURCE_PAIRS = [(f"source-{person:03d}", vid) for person in range(70) for vid in sorted(module.DEVELOPMENT_CLIPS)]
SOURCE_SHA = module.sha256_bytes(json.dumps(SOURCE_PAIRS, separators=(",", ":")).encode())


def manufactured_bundle():
    states = {}
    for method in module.METHODS[1:]:
        dimension = 13 if method == "context_only" else 163
        coefficient = np.zeros(dimension, dtype=np.float64)
        coefficient[11:13] = (.001, .01)
        if dimension == 163:
            coefficient[13] = .02
            coefficient[-1] = .03
        states[method] = module.FrozenRidge(method, coefficient, np.asarray(2., dtype=np.float64),
            np.zeros(dimension), np.ones(dimension), np.ones(dimension),
            None if dimension == 13 else np.zeros(150),
            module.sha256_bytes(method.encode()), SOURCE_SHA)
    return module.FrozenBundle(states, {c: 3.5 for c in module.CATEGORIES}, "b" * 64, SOURCE_SHA)


def fixture():
    rows = [{"subject": "manufactured-001", "vid": vid, "role": "manufactured",
             "family": f"film-{vid}", "assigned_category": "Amusement",
             "acquisition_group": "legacy_1000", "catalogue_duration_seconds": 90. + vid,
             "trial": vid} for vid in (2, 3, 4)]
    quality = [{"subject": row["subject"], "vid": row["vid"], "role": row["role"],
                "region": region, "technical_passed": True, "review_flags": []}
               for row in rows for region in module.REGIONS]
    quality[4]["review_flags"] = [{"channel": "Fp2", "reason": "high_processed_excursion"}]
    return rows, quality


def feature(pair, region):
    if pair[1] == 3:
        raise AssertionError("No feature callback may access any region of a missing trial")
    return np.full(150, float(module.REGIONS.index(region) + 1), dtype=np.float64)


def bomb(*args):
    raise AssertionError("Unauthorized or premature feature access")


def serialized_states(directory):
    bundle = manufactured_bundle()
    bindings = {}
    for method, state in bundle.states.items():
        values = {"coefficient": state.coefficient, "intercept": state.intercept,
                  "mean": state.mean, "scale": state.scale, "variance": state.variance,
                  "alpha": np.asarray(100., dtype=np.float64)}
        if method != "context_only":
            values["eeg_imputation_mean"] = state.eeg_imputation_mean
        path = directory / (method + ".npz")
        np.savez_compressed(path, **values)
        digest = module.sha256_bytes(path.read_bytes())
        bindings[method] = {"path": path, "sha256": digest, "record": {
            "model": method, "stage": "full_source_state_qualification", "alpha": 100.,
            "training_rows": 980, "feature_dimension": len(state.mean),
            "training_pairs": SOURCE_PAIRS,
            "source_identity_sha256": SOURCE_SHA, "private_state_sha256": digest}}
    prior = {"global": 3.5, "coarse": {"negative": 3., "positive": 4.},
             "category": dict(bundle.category_prior), "video": {str(v): 3.5 for v in module.DEVELOPMENT_CLIPS}}
    prior_path = directory / "prior.json"
    prior_path.write_bytes(module.canonical_bytes(prior))
    return bindings, prior_path, module.sha256_bytes(prior_path.read_bytes())


class FrozenInferenceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "_tmp"
        scratch.mkdir(exist_ok=True)
        # Windows sandbox directories created with tempfile's restrictive
        # mode=0700 inherit an unusable ACL here; normal workspace mkdir keeps
        # the workspace ACL. The manufactured packet stays under ignored _tmp.
        self.directory = scratch / ("frozen-inference-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.bundle = manufactured_bundle()
        self.rows, self.quality = fixture()

    def infer(self, rows=None, quality=None, loader=feature, mode="manufactured"):
        return module.infer_four_methods(self.bundle, self.rows if rows is None else rows,
            self.quality if quality is None else quality, loader, mode=mode)

    def export(self, predictions=None, name="lock"):
        return module.export_prediction_lock(self.directory / name,
            self.infer() if predictions is None else predictions, self.rows, self.quality,
            mode="manufactured", bundle=self.bundle)

    def test_hand_calculated_context_late_distributed_and_shared_fallback(self):
        rows = self.infer()
        self.assertEqual(len(rows), 12)
        # Input-independent hand calculation: duration=.092, trial=.02,
        # late band terms=(.02+.03)*3; distributed=(.02+.03)*2.
        self.assertAlmostEqual(rows[1]["prediction"], 2.112)
        self.assertAlmostEqual(rows[2]["prediction"], 2.262)
        self.assertAlmostEqual(rows[3]["prediction"], 2.212)
        self.assertEqual([r["prediction"] for r in rows[4:8]], [3.5] * 4)
        self.assertEqual([r["common_category_fallback"] for r in rows[4:8]], [False, True, True, True])
        self.assertTrue(all(r["input_sha256"] is None for r in rows[4:8]))

    def test_last_protected_identity_rejected_before_first_feature_callback(self):
        rows = deepcopy(self.rows)
        rows[-1]["role"] = "confirmation"
        with self.assertRaises(PermissionError): self.infer(rows=rows, loader=bomb)

    def test_entire_identity_barrier_precedes_metadata_numeric_conversion(self):
        class FloatBomb:
            def __float__(self): raise AssertionError("Premature numeric conversion")
        rows = deepcopy(self.rows)
        rows[0]["catalogue_duration_seconds"] = FloatBomb()
        rows[-1]["role"] = "confirmation"
        with self.assertRaises(PermissionError): self.infer(rows=rows, loader=bomb)

    def test_incomplete_duplicate_and_extra_rosters_are_rejected(self):
        for rows in (self.rows + [self.rows[0]], [], self.rows[:1]):
            with self.subTest(count=len(rows)):
                with self.assertRaises(PermissionError): self.infer(rows=rows, loader=bomb)

    def test_metadata_target_field_is_rejected_before_callbacks(self):
        rows = deepcopy(self.rows); rows[0]["target"] = 6.
        with self.assertRaises(PermissionError): self.infer(rows=rows, loader=bomb)

    def test_confirmation_and_validation_modes_cannot_be_activated_by_argument(self):
        for mode in ("confirmation", "validation", "protected", "source"):
            with self.subTest(mode=mode):
                with self.assertRaises(PermissionError): self.infer(loader=bomb, mode=mode)

    def test_source_mode_requires_full_exact_rectangle(self):
        rows = deepcopy(self.rows)
        for row in rows: row["role"] = "source"
        with self.assertRaises(PermissionError): self.infer(rows=rows, loader=bomb, mode="source")

    def test_wrong_source_people_cannot_pass_same_geometry_before_numeric_access(self):
        categories = {"Amusement": (2, 3), "Anger": (4, 6), "Disgust": (11, 12),
            "Fear": (9,), "Inspiration": (20,), "Joy": (17, 18), "Sadness": (23, 25), "Tenderness": (27, 28)}
        category_by_title = {v: c for c, videos in categories.items() for v in videos}
        rows = [{"subject": p, "vid": v, "role": "source", "family": f"film-{v}",
                 "assigned_category": category_by_title[v], "acquisition_group": "legacy_1000",
                 "catalogue_duration_seconds": 90. + v, "trial": v} for p, v in SOURCE_PAIRS]
        for row in rows:
            if row["subject"] == "source-069": row["subject"] = "reserved-relabelled-source"
        class FloatBomb:
            def __float__(self): raise AssertionError("Source identity guard must precede numerical conversion")
        rows[0]["catalogue_duration_seconds"] = FloatBomb()
        with self.assertRaises(PermissionError): self.infer(rows=rows, quality=[], loader=bomb, mode="source")

    def test_late_invalid_quality_is_rejected_before_first_callback(self):
        quality = deepcopy(self.quality); quality[-1]["technical_passed"] = False
        with self.assertRaises(PermissionError): self.infer(quality=quality, loader=bomb)

    def test_missing_duplicate_unexpected_quality_records_are_rejected(self):
        for quality in (self.quality[:-1], self.quality + [self.quality[0]],
                        self.quality + [dict(self.quality[0], vid=28)]):
            with self.assertRaises(PermissionError): self.infer(quality=quality, loader=bomb)

    def test_unknown_or_numerical_failure_flag_cannot_be_imputed(self):
        quality = deepcopy(self.quality); quality[4]["review_flags"][0]["reason"] = "decoder_nan"
        with self.assertRaises(PermissionError): self.infer(quality=quality, loader=bomb)

    def test_false_availability_or_bad_channel_rejected(self):
        for mutation in ("availability", "channel"):
            quality = deepcopy(self.quality)
            if mutation == "availability": quality[0]["eeg_available"] = False
            else: quality[4]["review_flags"][0]["channel"] = "unknown"
            with self.assertRaises(PermissionError): self.infer(quality=quality, loader=bomb)

    def test_nonfinite_available_eeg_is_fatal_not_missing(self):
        def bad(pair, region): return np.full(150, np.nan, dtype=np.float64)
        with self.assertRaises(ValueError): self.infer(loader=bad)

    def test_invalid_feature_geometry_and_dtype_fail(self):
        for values in (np.ones(149), np.ones(150, dtype=np.float32), np.ones((1, 150))):
            with self.assertRaises(ValueError): self.infer(loader=lambda *_: values)

    def test_frozen_state_is_immutable_and_clips_predictions(self):
        state = self.bundle.states["context_only"]
        with self.assertRaises(ValueError): state.mean.setflags(write=True)
        inputs = np.zeros((2, 13)); inputs[:, 11] = (1e5, -1e5)
        np.testing.assert_array_equal(state.predict(inputs), (7., 0.))

    def test_numeric_overflow_is_fatal_before_clipping(self):
        inputs = np.zeros((1, 13)); inputs[0, 11] = np.inf
        with self.assertRaises(ValueError): self.bundle.states["context_only"].predict(inputs)

    def test_safe_loader_accepts_exact_numeric_state_schemas(self):
        bindings, prior, digest = serialized_states(self.directory)
        loaded = module.load_frozen_bundle(bindings, prior, digest, SOURCE_SHA)
        self.assertEqual(loaded.category_prior["Fear"], 3.5)
        self.assertEqual(set(loaded.states), set(module.METHODS[1:]))

    def test_byte_hash_mismatch_blocks_state_and_prior(self):
        bindings, prior, digest = serialized_states(self.directory)
        bindings["late_joint"]["path"].write_bytes(b"changed")
        with self.assertRaises(ValueError): module.load_frozen_bundle(bindings, prior, digest, SOURCE_SHA)

    def test_unqualified_state_record_fails_before_numeric_load(self):
        bindings, prior, digest = serialized_states(self.directory)
        for field, wrong in (("alpha", 10.), ("training_rows", 448), ("stage", "source_cv"),
                             ("source_identity_sha256", "c" * 64), ("feature_dimension", 150)):
            changed = deepcopy(bindings); changed["context_only"]["record"][field] = wrong
            with self.assertRaises(PermissionError): module.load_frozen_bundle(changed, prior, digest, SOURCE_SHA)

    def test_npz_object_arrays_and_unexpected_keys_are_rejected(self):
        bindings, _, _ = serialized_states(self.directory)
        binding = bindings["context_only"]
        with np.load(binding["path"], allow_pickle=False) as archive:
            valid = {key: archive[key] for key in archive.files}
        object_values = dict(valid, coefficient=np.asarray([object()] * 13, dtype=object))
        for bad_values in (object_values, {"unexpected": np.ones(1)}):
            np.savez_compressed(binding["path"], **bad_values)
            digest = module.sha256_bytes(binding["path"].read_bytes())
            record = dict(binding["record"], private_state_sha256=digest)
            with self.assertRaises(ValueError): module.load_frozen_state(binding["path"], digest, record, SOURCE_SHA)

    def test_duplicate_zip_members_are_rejected_before_array_loading(self):
        bindings, _, _ = serialized_states(self.directory)
        binding = bindings["context_only"]
        original = binding["path"].read_bytes()
        with zipfile.ZipFile(io.BytesIO(original)) as source:
            members = [(n, source.read(n)) for n in source.namelist()]
        with zipfile.ZipFile(binding["path"], "w") as output:
            for name, payload in members: output.writestr(name, payload)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                output.writestr(members[0][0], members[0][1])
        digest = module.sha256_bytes(binding["path"].read_bytes())
        with self.assertRaises(ValueError):
            module.load_frozen_state(binding["path"], digest, dict(binding["record"], private_state_sha256=digest), SOURCE_SHA)

    def test_context_scaler_mismatch_and_invalid_scale_fail(self):
        late = self.bundle.states["late_joint"]
        altered_mean = late.mean.copy(); altered_mean[0] = .1
        changed = module.FrozenRidge(late.model, late.coefficient, late.intercept, altered_mean,
            late.scale, late.variance, late.eeg_imputation_mean, late.state_sha256, SOURCE_SHA)
        states = dict(self.bundle.states); states[late.model] = changed
        with self.assertRaises(ValueError): module.FrozenBundle(states, self.bundle.category_prior, "b" * 64, SOURCE_SHA)
        with self.assertRaises(ValueError):
            module.FrozenRidge(late.model, late.coefficient, late.intercept, late.mean,
                np.zeros(163), late.variance, late.eeg_imputation_mean, late.state_sha256, SOURCE_SHA)

    def test_prior_duplicate_keys_and_missing_support_fail(self):
        bindings, prior, _ = serialized_states(self.directory)
        for content in (b'{"global":3,"global":4}', module.canonical_bytes({"global": 3.5})):
            prior.write_bytes(content)
            with self.assertRaises(ValueError):
                module.load_frozen_bundle(bindings, prior, module.sha256_bytes(content), SOURCE_SHA)

    def test_full_canonical_export_and_verification_preserve_denominators(self):
        receipt = self.export()
        self.assertTrue(receipt["passed"])
        self.assertEqual((receipt["trials"], receipt["predictions"]), (3, 12))
        manifest = json.loads((self.directory / "lock" / "manifest.json").read_bytes())
        self.assertEqual((manifest["available_trials"], manifest["fallback_trials"]), (2, 1))
        self.assertFalse(manifest["targets_present"])
        self.assertFalse(manifest["confirmation_activated"])

    def test_existing_output_and_partial_reservation_block_retries(self):
        self.export()
        with self.assertRaises(FileExistsError): self.export()
        (self.directory / ".unfinished.reservation").write_bytes(b"partial")
        with self.assertRaises(FileExistsError): self.export(name="unfinished")
        partial = self.directory / "partial"; partial.mkdir()
        with self.assertRaises(PermissionError):
            module.verify_prediction_lock(partial, self.rows, self.quality, mode="manufactured", bundle=self.bundle)

    def test_publication_failure_preserves_evidence_and_blocks_retry(self):
        original = module._write_new
        def fail_manifest(path, data):
            if path.name == "manifest.json": raise OSError("Manufactured publication interruption")
            original(path, data)
        with patch.object(module, "_write_new", fail_manifest):
            with self.assertRaises(OSError): self.export(name="interrupted")
        self.assertFalse((self.directory / "interrupted").exists())
        self.assertTrue((self.directory / ".interrupted.reservation").is_file())
        self.assertEqual(len(list(self.directory.glob(".interrupted.staging-*"))), 1)
        with self.assertRaises(FileExistsError): self.export(name="interrupted")

    def test_prediction_schema_identity_order_duplicates_and_target_fail(self):
        for change in ("missing", "duplicate", "reorder", "family", "target"):
            predictions = self.infer()
            if change == "missing": predictions.pop()
            elif change == "duplicate": predictions[-1] = deepcopy(predictions[0])
            elif change == "reorder": predictions[1], predictions[2] = predictions[2], predictions[1]
            elif change == "family": predictions[0]["family"] = "changed"
            else: predictions[0]["target"] = 6.
            with self.assertRaises(PermissionError): self.export(predictions, name=change)
            self.assertFalse((self.directory / change).exists())

    def test_shared_availability_fallback_and_qc_hash_tampering_fail(self):
        for change in ("availability", "fallback", "qc", "prior"):
            predictions = self.infer()
            if change == "availability": predictions[4]["shared_eeg_available"] = True
            elif change == "fallback": predictions[5]["common_category_fallback"] = False
            elif change == "qc": predictions[0]["qc_sha256"] = "f" * 64
            else: predictions[5]["prediction"] = 3.6
            with self.assertRaises(PermissionError): self.export(predictions, name=change)

    def test_canonical_lock_detects_post_export_byte_tampering(self):
        self.export()
        path = self.directory / "lock" / "predictions.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            module.verify_prediction_lock(path.parent, self.rows, self.quality, mode="manufactured", bundle=self.bundle)

    def test_lock_rejects_different_declared_roster_even_with_same_denominator(self):
        self.export()
        rows = deepcopy(self.rows); rows[-1]["family"] = "other-family"
        with self.assertRaises(PermissionError):
            module.verify_prediction_lock(self.directory / "lock", rows, self.quality, mode="manufactured", bundle=self.bundle)

    def test_generator_quality_input_is_not_consumed_before_final_validation(self):
        self.assertEqual(len(self.infer(quality=(r for r in self.quality))), 12)


if __name__ == "__main__":
    unittest.main()
