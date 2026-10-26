import unittest
import numpy as np

class AuthorIndexingTests(unittest.TestCase):
    def test_squeezed_epoch_first_channel_and_scalar_argwhere_lose_channel_axis(self):
        for bad_index in (0,27):
            data=np.ones((32,7501));data[bad_index]=10000.
            outlier=np.abs(data)>3*np.median(np.abs(data))
            np.testing.assert_array_equal(np.flatnonzero(outlier.mean(axis=1)>.3),[bad_index])
            first=np.where(outlier,0,1)[0]
            bad=np.argwhere(np.mean(1-first,axis=0)>.3)
            self.assertEqual(first.shape,(7501,));self.assertEqual(bad.size,0)

if __name__=='__main__':unittest.main()
