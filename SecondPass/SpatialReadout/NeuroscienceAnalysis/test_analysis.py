"""CPU-only instrumentation contracts; never starts the accelerator budget."""
import importlib
import numpy as np
import torch
from SecondPass.SpatialReadout.model import SpatialReadout
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

def test_recording_preserves_logits_and_reconstructs_reads():
    spec=importlib.util.find_spec('SecondPass.SpatialReadout.NeuroscienceAnalysis.core')
    assert spec is not None, 'Analysis instrumentation not implemented'
    core=importlib.import_module(spec.name)
    torch.set_num_threads(2);torch.manual_seed(471)
    model=SpatialReadout({'orientation_cued':2}).eval().requires_grad_(False)
    x,_,meta=SpatialBatteryStream(78412,'test').batch(2,'orientation_cued',{'delay':4})
    with torch.inference_mode():
        direct=model(x,'orientation_cued')
        wrapped,data=core.observe(model,x,'orientation_cued',meta,record=True)
        zero,_=core.observe(model,x,'orientation_cued',meta,intervention={'kind':'inhibit','site':'cued','epoch':'encoding','dose':0})
    assert torch.equal(direct,wrapped)
    assert torch.equal(direct,zero)
    for s in range(3):
        assert data[f's{s}_beta'].shape[0:2]==(2,8)
        assert data[f's{s}_beta'].shape[-1]==2
        assert data[f's{s}_reconstruction_error'].max()<2e-6
    assert data['gru_write'].shape==(2,8,7,7)


def test_native_parity_zero_and_matched_cue_relocation():
    spec=importlib.util.find_spec('SecondPass.SpatialReadout.NeuroscienceAnalysis.stimuli')
    assert spec is not None, 'Native-law sweep not implemented'
    module=importlib.import_module(spec.name)
    for delay in (0,4):
        x,y,m=SpatialBatteryStream(8932,'test').batch(16,'orientation_cued',{'delay':delay})
        st=module.OrientationSweep(8932)
        xx,yy,mm=st.batch(16,'orientation_cued',{'delay':delay})
        assert torch.equal(x,xx) and torch.equal(y,yy)
        rx,ry,rm=module.relocate(xx,mm,st.raw)
        for i in range(16):
            assert torch.equal(rx[i,3:],xx[i,3:])
            assert rm[i]['rotations_degrees']==mm[i]['rotations_degrees']
            assert ry[i].item()==int(rm[i]['rotations_degrees'][rm[i]['target_location']]*rm[i]['cue_sign']>0)
    x,y,m=module.OrientationSweep(8932,magnitude=0).batch(16,'orientation_cued',{'delay':0})
    assert y.sum()==0 and all(all(z==0 for z in a['rotations_degrees']) for a in m)


def test_runner_cpu_smoke_saves_verified_artifacts(tmp_path):
    spec=importlib.util.find_spec('SecondPass.SpatialReadout.NeuroscienceAnalysis.run')
    assert spec is not None, 'Frozen runner not implemented'
    runner=importlib.import_module(spec.name)
    result=runner.cpu_smoke(tmp_path)
    assert result['direct_wrapper_max_error']==0
    assert result['zero_dose_max_error']==0
    assert result['model_immutable']
    assert (tmp_path/'cpu_smoke.json').exists()
