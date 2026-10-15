import unittest
import numpy as np
from scripts.qualify_faced_batch8_cuda_2026_10_15 import fabricated_inputs


class Tests(unittest.TestCase):
    def test_exact_representative_fabricated_geometry_and_offsets(self):
        source, held, windows, targets, offsets = fabricated_inputs()
        self.assertEqual((len(source), len(held), len(windows)), (8, 4, 12))
        self.assertEqual(len(source) // 8, 1)
        self.assertFalse(set(source) & set(held))
        self.assertEqual(set(targets), set(source + held))
        for value in windows.values():
            self.assertEqual(value.shape, (7, 30, 512)); self.assertEqual(value.dtype, np.float32)
            self.assertTrue(np.isfinite(value).all())
        self.assertEqual(set(offsets['source_fit']), set(source))
        self.assertEqual(set(offsets['held_source']), set(held))
        self.assertEqual(set(offsets['source_fit'].values()), {3.5})

    def test_fabricated_inputs_are_reproducible(self):
        first, second = fabricated_inputs(), fabricated_inputs()
        for pair in first[2]:
            np.testing.assert_array_equal(first[2][pair], second[2][pair])


if __name__ == '__main__':
    unittest.main()
