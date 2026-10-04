from pathlib import Path
import ast
import torch
from torch import nn


def test_true_dense_backward(monkeypatch):
    p=Path(__file__).with_name('model.py')
    assert p.exists(), 'all-dense model missing'
    from .model import DenseObserver
    torch.set_num_threads(2)
    m=DenseObserver({'krauzlis_cued_motion':2})
    assert m.encoder[0].in_features==90000
    assert len(m.kda)==3
    assert not any(isinstance(x,(nn.modules.conv._ConvNd,nn.Unfold,nn.MultiheadAttention,nn.TransformerEncoder)) for x in m.modules())
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node,ast.Attribute):
            assert not any(s in node.attr.lower() for s in ('conv','unfold','pool','interpolate'))
    def forbidden(*args,**kwargs): raise AssertionError('Forbidden local image operation executed')
    from torch.nn import functional as F
    for name in ('conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','unfold','interpolate','avg_pool2d','max_pool2d','adaptive_avg_pool2d'):
        monkeypatch.setattr(F,name,forbidden)
    for name in ('conv1d','conv2d','conv3d','convolution'):
        if hasattr(torch,name): monkeypatch.setattr(torch,name,forbidden)
    x=torch.rand(2,4,3,100,100,requires_grad=True)
    y=m(x,'krauzlis_cued_motion')
    assert y.shape==(2,2)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    frames=m.frames(x.detach())
    assert frames.shape==(2,4,9,100,100)
    assert torch.count_nonzero(frames[:,0,:6])==0
    assert torch.equal(frames[:,2,:3],x.detach()[:,0]-.5)
    assert torch.equal(frames[:,2,6:],x.detach()[:,2]-.5)


def test_dense_kda_native_math_and_initial_gates():
    from .model import DenseKDA
    from PreAttentiveVision.TemporalIntegration.accumulators import kda_update
    from torch.nn import functional as F
    block=DenseKDA()
    x=torch.randn(3,128)
    state=torch.randn(3,2,8,16)
    p=block.inputs(x).reshape(3,2,41)
    assert torch.allclose(p[...,32:40].sigmoid(),torch.full((3,2,8),.9))
    assert torch.equal(p[...,40:].sigmoid(),torch.full((3,2,1),.5))
    q=F.normalize(p[...,:8],dim=-1,eps=1e-6)
    k=F.normalize(p[...,8:16],dim=-1,eps=1e-6)
    read,expected=kda_update(state,q,k,p[...,16:32],p[...,32:40].sigmoid(),p[...,40:].sigmoid())
    actual,new=block(x,state)
    assert torch.equal(new,expected)
    assert torch.equal(actual,(x+block.output(read.flatten(1))).relu())


def test_task_draws_independent_of_model_rng():
    from . import worker as w
    from SecondPass.SpatialReadout.SpatialConsolidation.Krauzlis90Fresh import worker as old
    import random
    import numpy as np
    torch.manual_seed(81);np.random.seed(91);random.seed(101)
    a=w.FreshStream('train').batch(2,w.TASK,'B12')
    torch.manual_seed(181);np.random.seed(191);random.seed(201)
    b=old.FreshStream('train').batch(2,w.TASK,'B12')
    assert w.fresh.tree_equal(a,b)
    assert w.FINAL_NAMESPACE!=old.FINAL_NAMESPACE
