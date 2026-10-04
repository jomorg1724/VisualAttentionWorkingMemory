"""One focused CPU shape, variational and complete-gradient check."""
import json
import time
from pathlib import Path
import torch
from .model import ThreeFrameConvVAE, losses


def main():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(2)
    except RuntimeError: pass
    started = time.monotonic()
    torch.manual_seed(71)
    model = ThreeFrameConvVAE()
    target = torch.full((1, 3, 3, 100, 100), .5)
    for frame in range(3):
        target[:, frame, :, 47:51, 19+frame:23+frame] = 1.
    target.requires_grad_()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = model(target)
    assert result['reconstruction'].shape == target.shape
    assert result['mu'].shape == result['logvar'].shape == result['z'].shape == (1, 256, 13, 13)
    assert result['logvar'].min() >= -8 and result['logvar'].max() <= 4
    assert not torch.equal(result['z'], result['mu'])
    assert torch.equal(model.decode(result['z']), result['reconstruction'])
    assert torch.equal(model.encode(target, sample=False), result['mu'])
    measured = losses(result['reconstruction'], target, result['mu'], result['logvar'])
    assert all(torch.isfinite(value) for value in measured.values())
    # Gray reconstruction retains a penalty despite sparse contrast support.
    gray = losses(torch.full_like(target, .5), target, result['mu'].detach(), result['logvar'].detach())
    assert torch.allclose(gray['recon'], torch.tensor(.125), atol=1e-7)
    assert gray['recon'] > gray['full_mse'] * 100
    measured['loss'].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() > 0, name
    assert all(target.grad[:, frame].count_nonzero() > 0 for frame in range(3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    assert all(not torch.equal(before[name], p) for name, p in model.named_parameters())
    model.eval()
    with torch.no_grad():
        deterministic = model(target)
        assert torch.equal(deterministic['z'], deterministic['mu'])
    record = dict(complete=True, device='cpu', production_steps=0, fixture_optimizer_steps=1,
                  parameters=sum(p.numel() for p in model.parameters()), tensors=len(list(model.parameters())),
                  latent_shape=list(result['mu'].shape[1:]), stochastic_training=True,
                  deterministic_mu_features=True, no_input_decoder_bypass=True,
                  all_parameters_finite_gradients_and_updates=True,
                  all_three_frames_receive_gradients=True, sparse_support_loss_check=True,
                  seconds=time.monotonic()-started)
    Path(__file__).with_name('check_results.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)

if __name__ == '__main__': main()
