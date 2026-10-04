"""Focused behavioral contracts; no checkpoint inference in these tests."""
import importlib.util
from pathlib import Path
import numpy as np

def test_native_metadata_and_counterfactual_contract():
    path=Path(__file__).with_name('diagnostic.py')
    assert path.exists(), 'Diagnostic implementation not present yet'
    spec=importlib.util.spec_from_file_location('diag',path)
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    for b in (12,20,28):
        a=d.SpatialBatteryStream(8675309,'test');z=d.SpatialBatteryStream(8675309,'test')
        x,y,m,angles=d.captured_batch(a,b)
        xx,yy,mm=z.batch(1,d.TASK,dict(baseline_transitions=b))
        assert np.array_equal(x.numpy(),xx.numpy()) and np.array_equal(y.numpy(),yy.numpy())
        assert a.state_dict()==z.state_dict()
        assert angles.shape==(b+8,2,16)
        pair,labels=d.make_pair(x,m[0])
        assert np.array_equal(pair[0,2:].numpy(),pair[1,2:].numpy())
        assert not np.array_equal(pair[0,:2].numpy(),pair[1,:2].numpy())
        assert labels[0]==y.item()
        assert labels[1]==int(m[0]['changed_patch']==1-m[0]['target_location'])
        means=d.circular_targets(angles,b)
        assert means.shape==(2,2)
        assert np.all(np.isfinite(means))
