import unittest
from pathlib import Path
import numpy as np

class Contract(unittest.TestCase):
    def test_input_only_access_and_binding(self):
        self.assertTrue(Path(__file__).with_name('factorized.py').exists(), 'missing implementation')
        from factorized import access, relation
        rng=np.random.default_rng(3)
        final=rng.normal(size=(4,3136)); p=rng.normal(size=(3136,32))
        a=access({'memory__final':final}, 'final', p)
        poisoned={'memory__final':final, **{'memory__'+k:np.full_like(final,np.nan) for k in ['cue','baseline','post']}}
        np.testing.assert_array_equal(a, access(poisoned,'final',p))
        self.assertEqual(a.shape,(4,4752))
        c=np.array([0.,1.,0.,1.]); e=np.array([1.,1.,0.,1.]); s=np.array([0.,0.,1.,1.])
        np.testing.assert_array_equal(relation(c,e,s),[1.,0.,0.,1.])

if __name__=='__main__':unittest.main()
