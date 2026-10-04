import unittest
import importlib.util
from pathlib import Path
import numpy as np

class Contract(unittest.TestCase):
    def test_translation(self):
        path=Path(__file__).with_name('motion.py')
        self.assertTrue(path.exists(), 'motion readout implementation missing')
        spec=importlib.util.spec_from_file_location('motion',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        from scipy.ndimage import shift
        rng=np.random.default_rng(88)
        from scipy.ndimage import gaussian_filter
        a=gaussian_filter(rng.normal(size=(3,32,32)),(0,1,1))
        frames=np.stack([shift(a,(0,.2*t,.3*t),order=1,mode='nearest') for t in range(3)])
        v=m.velocity(frames)[0]
        np.testing.assert_allclose(v,[.3,.2],atol=.08)
        np.testing.assert_allclose(m.velocity(frames[::-1])[0],-v,atol=.02)
        self.assertEqual(m.patch_descriptor(np.stack([frames,frames])).shape,(8,))
        self.assertTrue(np.isfinite(m.patch_descriptor(np.zeros((2,3,3,32,32)))).all())

if __name__=='__main__':unittest.main()
