"""One focused CPU causal-mean, token, streaming and full-gradient check."""
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from SecondPass.WeightedMeanConvGRU.model import WeightedMeanConvGRU
from SecondPass.TwoFrameRViT.model import RecurrentVisualBlock
from .model import WeightedMeanRViT


def main():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(2)
    except RuntimeError: pass
    started = time.monotonic()
    torch.manual_seed(83)
    model = WeightedMeanRViT()
    constant = torch.stack([torch.full((1,3,100,100), v) for v in (0.,1.,.4)], dim=1)
    weighted = model.weighted_frames(constant)
    expected = torch.stack([torch.full((1,3,100,100), v) for v in (0.,.5,.6)], dim=1)
    torch.testing.assert_close(weighted, expected, atol=1e-7, rtol=1e-7)
    assert torch.equal(weighted, WeightedMeanConvGRU.weighted_frames(constant))
    assert torch.equal(model.weighted_frames(constant[:,:1]), constant[:,:1])
    changed = constant.clone(); changed[:,2] = .9
    assert torch.equal(model.weighted_frames(changed)[:,:2], weighted[:,:2])
    assert all(module.stride==(1,1) for module in model.modules() if isinstance(module,torch.nn.Conv2d))
    assert len([module for module in model.modules() if isinstance(module,RecurrentVisualBlock)])==1
    assert not any(isinstance(module,torch.nn.GRU) for module in model.modules())
    images = torch.rand(1,3,3,100,100)
    model.eval()
    with torch.no_grad():
        assert model.encode(images[:,0]).shape == (1,256,13,13)
        assert model.encode_tokens(images[:,0]).shape == (1,169,256)
        sequence = model(images)
        history = memory = None
        for image in images.unbind(1):
            logits, history, memory = model.stream_step(image, history, memory)
        torch.testing.assert_close(sequence, logits, atol=1e-6, rtol=1e-5)
        assert torch.equal(model(images), sequence)
    model.train()
    images.requires_grad_()
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    F.cross_entropy(model(images), torch.tensor([1])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.count_nonzero()>0, name
    assert images.grad[:,0].count_nonzero()>0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    assert all(not torch.equal(before[n],p) for n,p in model.named_parameters())
    record = dict(complete=True, device='cpu', production_steps=0, fixture_optimizer_steps=1,
                  parameters=sum(p.numel() for p in model.parameters()), tensors=len(list(model.parameters())),
                  fixed_mean_matches_previous_formula=True, first_frame_identity=True, causal=True,
                  cnn_field_shape=[256,13,13], token_shape=[169,256], exactly_one_shared_rvit=True,
                  sequence_stream_reset_parity=True, all_parameters_finite_gradients_and_updates=True,
                  earliest_frame_gradient=True, all_weights_fresh=True, seconds=time.monotonic()-started)
    Path(__file__).with_name('check_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__': main()
