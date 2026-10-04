"""Focused CPU arithmetic and full-gradient check; no training launch."""
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import WeightedMeanConvGRU


def main():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(2)
    except RuntimeError: pass
    started=time.monotonic()
    torch.manual_seed(291)
    model=WeightedMeanConvGRU(checkpoint_encoder=True)
    movie=torch.stack([torch.full((1,3,100,100),v) for v in (0.,1.,.4)],dim=1)
    mixed=model.weighted_frames(movie)
    expected=torch.stack([torch.full((1,3,100,100),v) for v in (0.,.5,.6)],dim=1)
    torch.testing.assert_close(mixed,expected,atol=1e-7,rtol=1e-7)
    assert torch.equal(model.weighted_frames(movie[:,:1]),movie[:,:1])
    changed=movie.clone();changed[:,2]=.9
    assert torch.equal(model.weighted_frames(changed)[:,:2],mixed[:,:2])
    assert model.gru.input_size==2704 and model.gru.hidden_size==256 and model.gru.num_layers==1
    images=torch.rand(1,3,3,100,100,requires_grad=True)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    logits=model(images)
    assert logits.shape==(1,2) and torch.isfinite(logits).all()
    F.cross_entropy(logits,torch.tensor([1])).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0,name
    assert images.grad[:,0].count_nonzero()>0
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    optimizer.step()
    assert all(not torch.equal(before[n],p) for n,p in model.named_parameters())
    # The production stream remains the existing modified single-stimulus adapter.
    from . import worker
    from SecondPass.SingleStimulusRViT.stimuli import SingleStimulusStream
    assert worker.ReplayStream.__bases__==(SingleStimulusStream,)
    assert worker._evaluate.__globals__['FreshStream'] is SingleStimulusStream
    result=dict(complete=True,device='cpu',fixture_optimizer_steps=1,production_steps=0,
        fresh_parameters=sum(p.numel() for p in model.parameters()),trainable_tensors=len(list(model.parameters())),
        exact_causal_mean=True,repeat_first_frame_startup=True,earliest_frame_gradient=True,
        all_learned_parameters_finite_and_updated=True,production_single_stimulus_adapter_unchanged=True,
        seconds=time.monotonic()-started)
    Path(__file__).with_name('check_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
