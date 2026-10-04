import importlib.util
import torch

def test_terminal_transform_shape_mixing():
    name='SecondPass.SpatialReadout.SpatialConsolidation.model'
    assert importlib.util.find_spec(name) is not None, 'terminal spatial block missing'
    from SecondPass.SpatialReadout.SpatialConsolidation.model import SpatialConsolidation
    torch.set_num_threads(2)
    model=SpatialConsolidation({'test':2})
    field=torch.randn(2,64,7,7,requires_grad=True)
    result=model.consolidation(field)
    assert result.shape==field.shape
    grad=torch.autograd.grad(result[0,0,0,0],field)[0]
    assert grad[0,:,6,6].abs().sum()>0
    assert grad[1].abs().sum()==0
    calls=[]
    model.consolidation.register_forward_hook(lambda *args:calls.append(1))
    images=torch.randn(2,3,3,112,112,requires_grad=True)
    model(images,'test').sum().backward()
    assert len(calls)==1
    assert images.grad[:,0].abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
