import unittest
from scripts.continue_faced_stronger_baseline_2026_10_15 import explicit_index_fit


class Tests(unittest.TestCase):
    def test_bare_cuda_is_made_explicit_without_changing_scientific_arguments(self):
        captured = {}
        def fake(*args, **kwargs):
            captured.update(kwargs)
            return args
        returned = explicit_index_fit('source', fit_function=fake, device='cuda', epochs=360,
            checkpoints=(120, 240, 360), rate=.001, seed=20261101)
        self.assertEqual(returned, ('source',))
        self.assertEqual(captured, {'device': 'cuda:0', 'epochs': 360, 'checkpoints': (120, 240, 360), 'rate': .001, 'seed': 20261101})

    def test_explicit_first_device_stays_first(self):
        self.assertEqual(explicit_index_fit(fit_function=lambda **kw: kw['device'], device='cuda:0'), 'cuda:0')

    def test_other_devices_fail_before_fit_callback(self):
        def forbidden(**kwargs):
            raise AssertionError('Forbidden optimizer callback')
        for device in ('cpu', 'cuda:1', None):
            with self.assertRaises(PermissionError):
                explicit_index_fit(fit_function=forbidden, device=device)


if __name__ == '__main__':
    unittest.main()
