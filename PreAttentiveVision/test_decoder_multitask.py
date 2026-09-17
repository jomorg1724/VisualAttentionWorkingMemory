"""Focused checks for the changed directional comparison semantics (CPU only)."""
import torch
from PreAttentiveVision.decoder_multitask import local_correlation, OrderedTaskDecoder


def test_order_and_shared_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(187)
    a=torch.zeros(1,3,7,7);b=torch.zeros_like(a)
    a[0,0,3,2]=1;b[0,0,3,3]=1
    ab=local_correlation(a,b).sum((0,2,3));ba=local_correlation(b,a).sum((0,2,3))
    assert ab.argmax().item()==13  # dy0, dx+1
    assert ba.argmax().item()==11  # dy0, dx-1
    first=[torch.randn(2,c,s,s,requires_grad=True) for c,s in [(24,11),(48,7),(96,5)]]
    second=[torch.randn_like(f,requires_grad=True) for f in first]
    decoder=OrderedTaskDecoder({'motion_direction':4}).eval()
    logits=decoder(first,second,'motion_direction')
    reverse=decoder(second,first,'motion_direction')
    assert logits.shape==(2,4) and torch.isfinite(logits).all()
    assert not torch.allclose(logits,reverse,atol=1e-6)
    logits.square().sum().backward()
    assert all(f.grad is not None and f.grad.abs().sum()>0 for f in first+second)


if __name__=='__main__':
    test_order_and_shared_gradients()
    print('PASS: signed local displacement, order-sensitive output, gradients to both frames at all scales')
