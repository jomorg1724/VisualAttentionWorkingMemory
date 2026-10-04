"""Focused CPU checks; no accelerator access or scientific test scoring."""
import importlib
import numpy as np
import torch

def test_frozen_extraction_native_logits_and_axial_metric():
    assert importlib.util.find_spec('SecondPass.SpatialReadout.FeatureDiagnostic.run') is not None, 'Diagnostic implementation missing'
    from SecondPass.SpatialReadout.FeatureDiagnostic.run import extract, axial_error, relation_features
    from SecondPass.SpatialReadout.model import SpatialReadout
    from SecondPass.TaskSuite.suite import task_classes
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    torch.set_num_threads(2)
    model=SpatialReadout(task_classes()).eval().requires_grad_(False)
    for task in ['orientation_cued','spatial_binding']:
        x,y,meta=SpatialBatteryStream(19871,'train').batch(2,task,{'delay':0})
        with torch.inference_mode():
            z,features=extract(model,x,task)
            assert torch.equal(z,model(x,task))
        assert features['early'].shape[:3]==(2,x.shape[1],32)
        assert features['late'].shape[2:]==(160,7,7)
        assert features['gru'].shape[2:]==(64,7,7)
        assert features['readout'].shape[2:]==(256,)
    a=np.deg2rad(np.array([[179.,1.,45.,90.]]))
    b=np.deg2rad(np.array([[1.,179.,45.,0.]]))
    np.testing.assert_allclose(axial_error(a,b),[[2,2,0,90]],atol=1e-5)
    # Relations are computable from predicted vectors alone; no metadata API.
    c=np.tile(np.array([1.,0.]),(2,4,1)); p=c.copy(); p[1]=[0.,1.]
    r=relation_features(c,p,np.ones((2,4))/4,np.array([1.,-1.]))
    assert r.shape[0]==2 and np.isfinite(r).all()
