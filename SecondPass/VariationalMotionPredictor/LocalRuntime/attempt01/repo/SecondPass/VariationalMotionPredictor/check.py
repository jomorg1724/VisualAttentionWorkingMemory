"""One short CPU vector-latent, prediction and all-gradient engineering check."""
import json
from pathlib import Path
import time
import torch
from .model import VariationalMotionPredictor, losses


def main():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(2)
    except RuntimeError: pass
    started = time.monotonic()
    torch.manual_seed(93)
    model = VariationalMotionPredictor()
    past = torch.full((1,3,3,100,100), .5)
    for frame in range(3):
        past[:,frame,:,46:50,19+frame:23+frame] = 1.
    target = torch.full((1,3,100,100), .5)
    target[:,:,46:50,22:26] = 1.
    past.requires_grad_()
    before = {name:p.detach().clone() for name,p in model.named_parameters()}
    result = model(past)
    assert result['mu'].shape == result['logvar'].shape == result['z'].shape == (1,512)
    assert result['prediction'].shape == target.shape
    assert torch.equal(result['logvar'], torch.full_like(result['logvar'], -4.))
    assert not torch.equal(result['mu'],result['z'])
    assert torch.equal(model.decode(result['z']),result['prediction'])
    assert not torch.equal(model.encode(past),model.encode(past.flip(1)))
    metrics = losses(result['prediction'],target,past,result['mu'],result['logvar'])
    assert all(torch.isfinite(value) for value in metrics.values())
    gray = losses(torch.full_like(target,.5),target,past,result['mu'].detach(),result['logvar'].detach(),beta=0.)
    assert gray['recon'] > 100*gray['full_mse'] and gray['change_mse'] > 0
    metrics['loss'].backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0,name
    assert all(past.grad[:,frame].count_nonzero()>0 for frame in range(3))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    optimizer.step()
    assert all(not torch.equal(before[name],p) for name,p in model.named_parameters())
    model.eval()
    with torch.no_grad():
        deterministic = model(past)
        assert torch.equal(deterministic['z'],deterministic['mu'])
    record = dict(complete=True,device='cpu',production_steps=0,fixture_optimizer_steps=1,
                  parameters=sum(p.numel() for p in model.parameters()),tensors=len(list(model.parameters())),
                  latent_shape=[512],fourth_frame_shape=[3,100,100],single_vector_latent=True,
                  decoder_latent_only_equivalence=True,ordered_past_frames=True,
                  all_parameters_finite_gradients_and_updates=True,all_three_input_frame_gradients=True,
                  support_union_loss_check=True,deterministic_mu_decode=True,seconds=time.monotonic()-started)
    Path(__file__).with_name('check_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__': main()
