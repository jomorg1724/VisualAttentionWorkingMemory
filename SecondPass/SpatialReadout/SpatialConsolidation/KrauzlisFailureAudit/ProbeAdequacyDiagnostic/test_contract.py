"""Small deterministic algebra/access contracts; synthetic arrays only unit tests."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import unittest
import importlib.util
from pathlib import Path
import numpy as np

class Contract(unittest.TestCase):
    def test_kernel_matches_explicit_and_replay(self):
        p=Path(__file__).with_name('adequacy.py')
        self.assertTrue(p.exists(), 'new diagnostic implementation missing')
        spec=importlib.util.spec_from_file_location('adequacy',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
        rng=np.random.default_rng(1);x=rng.normal(size=(7,4));z=rng.normal(size=(3,4))
        explicit=lambda q:np.concatenate([q/2,np.einsum('ni,nj->nij',q,q).reshape(len(q),-1)/4],1)
        np.testing.assert_allclose(a.kernel(x,z,2),explicit(x)@explicit(z).T,atol=1e-14)
        y=rng.normal(size=(7,2));f=a.fit_candidates(x,y,[2.,4.],2)
        for fit in f:
            pred=a.predict(fit,z)
            self.assertEqual(pred.shape,(3,2));self.assertLess(abs(fit['effective_df']-fit['requested_df']),1e-8)
            np.testing.assert_array_equal(fit['mean'],x.mean(0))

    def test_actual_saved_projection_dimensions(self):
        p=Path(__file__).parent
        if not (p/'train_features.npz').exists():self.skipTest('extraction not yet available')
        import adequacy as a
        data=dict(np.load(p/'train_features.npz'));projections=dict(np.load(p/'projections.npz'))
        for rep in ('pixels','trained','random'):
            self.assertEqual(projections[rep].shape[0],data[rep].shape[1]//2)
            self.assertEqual(a.representation(data,rep,'projected',projections).shape,(600,64))

if __name__=='__main__':unittest.main()
