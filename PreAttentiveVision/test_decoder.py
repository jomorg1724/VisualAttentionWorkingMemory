"""One focused changed-behavior check; encoder/data checks live with their modules."""
import torch
from PreAttentiveVision.decoder import CommonPairDecoder


def test_symmetric_spatial_decoder_reaches_both_frames():
    torch.set_num_threads(1);torch.manual_seed(47)
    decoder=CommonPairDecoder().eval()
    a=[torch.randn(2,c,s,s,requires_grad=True) for c,s in [(24,50),(48,25),(96,13)]]
    b=[torch.randn_like(v,requires_grad=True) for v in a]
    logits=decoder(a,b)
    assert logits.shape==(2,2) and torch.isfinite(logits).all()
    assert torch.allclose(logits,decoder(b,a),atol=1e-6,rtol=1e-6)
    logits.square().sum().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in a+b)
