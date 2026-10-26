"""Synthetic guards for original-versus-shuffled cached targets."""
import unittest

from gruxnet.faced_cached_targets import cached_original_target


class Poison:
    def __float__(self):
        raise AssertionError("Excluded numeric outcome was interpreted")


class CachedTargetTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {("s", 1): {"family": "f", "acquisition_group": "a"}, ("v", 1): {"family": "f", "acquisition_group": "a"}}
        self.programme = {"rotation_manifests": [{"arms": {"source_fit": [{"subject": "s", "vid": 1}]}} for _ in range(3)]}
        self.row = {"subject": "s", "vid": 1, "family": "f", "acquisition_group": "a", "rotation": 0, "model": "context_kernel", "arm": "source_fit", "target": 2.5}

    def read(self, **change):
        return cached_original_target(dict(self.row, **change), self.metadata, self.programme, "representation")

    def test_original_exact_and_no_alpha_column(self):
        self.assertEqual(self.read(), (("s", 1), 2.5))

    def test_shuffled_targets_never_converted(self):
        for model in ("capacity_spectral_permuted", "capacity_covariance_permuted"):
            self.assertEqual(self.read(model=model, arm="capacity_source_fit", target=Poison(), prediction=Poison()), (("s", 1), None))

    def test_capacity_requires_exact_source_arm_and_model(self):
        for changes in ({"model": "capacity_spectral_permuted"}, {"model": "capacity_unknown", "arm": "capacity_source_fit"}, {"model": "capacity_spectral_permuted", "arm": "capacity_source_fit", "subject": "v"}):
            with self.assertRaises(PermissionError):
                self.read(target=Poison(), **changes)

    def test_identity_before_numeric(self):
        for changes in ({"subject": "protected"}, {"family": "changed"}, {"acquisition_group": "changed"}):
            with self.assertRaises(PermissionError):
                self.read(target=Poison(), **changes)

    def test_real_capacity_is_an_original_target(self):
        self.assertEqual(self.read(model="capacity_covariance_real", arm="capacity_source_fit"), (("s", 1), 2.5))
        with self.assertRaises(ValueError):
            self.read(target=float("nan"))


if __name__ == "__main__":
    unittest.main()
