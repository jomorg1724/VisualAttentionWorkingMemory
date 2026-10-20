"""Numerical tests of analysis definitions, distinct from model inference tests."""
import unittest
import numpy as np
from .statistics import metrics,paired,holm,interval,source_clusters
from .design import phases,onset,retention_boundary,bias,arms
from .deep_analysis import full_probe
from .psychometric_fit import curve

class AnalysisTests(unittest.TestCase):
    def test_signal_detection(self):
        y=np.array([0,0,1,1]);p=np.array([[.9,.1],[.8,.2],[.2,.8],[.1,.9]])
        r=metrics(y,p);self.assertEqual(r['accuracy'],1);self.assertEqual(r['auc'],1);self.assertAlmostEqual(r['criterion'],0);self.assertGreater(r['dprime'],0)
        self.assertLess(r['accuracy_ci95'][0],1)
    def test_paired_and_boundary(self):
        y=np.array([0,1]*64);p=np.eye(2)[y];r=paired(y,p,p)
        self.assertEqual(r['delta_accuracy'],0);self.assertEqual(r['p_exact'],1);self.assertLess(r['ci95'][0],0);self.assertGreater(r['ci95'][1],0)
        q=p.copy();q[:32]=q[:32,::-1];r=paired(y,p,q);self.assertEqual(r['lost'],32);self.assertEqual(r['gained'],0);self.assertAlmostEqual(r['delta_accuracy'],-.25)
    def test_holm(self):self.assertTrue(np.allclose(holm([.01,.04,.03]),[.03,.06,.06]))
    def test_photo_cluster_pairing(self):
        y=np.array([0,1]*32);p=np.eye(2)[y];q=p.copy();q[:32]=q[:32,::-1]
        r=paired(y,p,q,clusters=np.repeat(np.arange(32),2))
        self.assertEqual(r['clusters'],32);self.assertIsNone(r['p_exact']);self.assertLess(r['p_value'],.001)
        self.assertEqual(r['delta_accuracy'],-.5);self.assertEqual(r['lost'],32)
        self.assertLessEqual(r['ci95'][0],-.5);self.assertGreaterEqual(r['ci95'][1],-.5)
    def test_missing_photo_ids_are_independent_trials(self):
        metadata=[{'base_id':None} for _ in range(128)]
        clusters=source_clusters(metadata);self.assertEqual(len(set(clusters)),128)
        y=np.arange(128)%2;p=np.eye(2)[y];q=p.copy();q[:32]=q[:32,::-1]
        ordinary=paired(y,p,q);checked=paired(y,p,q,clusters=clusters)
        self.assertEqual(ordinary,checked)
        same=paired(y,p,p,clusters=clusters);self.assertLess(same['ci95'][1],.03)
        self.assertEqual(list(source_clusters([{'base_id':0},{'base_id':0},{'base_id':None}])),['image:0','image:0','trial:2'])
    def test_psychometric_direction(self):
        v=np.array([0.,0.,.02]);x=np.array([0.,1.,20.]);up=curve(x,v,.5);down=curve(x,v,.5,True)
        self.assertAlmostEqual(up[0],.5);self.assertAlmostEqual(down[0],.98)
        self.assertTrue(np.all(np.diff(up)>0));self.assertTrue(np.all(np.diff(down)<0))
        self.assertTrue(np.allclose(up+down,1.48))
    def test_temporal_masks(self):
        m=dict(phases=['fixation','sample','sample','delay','search','search'],array_onset=4,target_index=0,centers_xy=[[25,25],[75,25],[25,75],[75,75]])
        self.assertEqual(onset('memory_guided_search',m,6),4);self.assertEqual(retention_boundary('memory_guided_search',m,6),3)
        arm=dict(site='target',dose=-4,epoch='evidence');b=bias('memory_guided_search',[m],6,arm)
        self.assertFalse(np.any(b[:,:4]));self.assertTrue(np.any(b[:,4:]));self.assertEqual(set(np.unique(b)),{-4.,0.})
    def test_full_probe_separation(self):
        rng=np.random.default_rng(23);y=np.arange(128)%2;yt=np.arange(64)%2
        x=rng.normal(size=(128,8));xt=rng.normal(size=(64,8));x[:,0]=y*10;xt[:,0]=yt*10
        r=full_probe(x,y,xt,yt,2);self.assertGreater(r['auc'],.99)

if __name__=='__main__':unittest.main()
