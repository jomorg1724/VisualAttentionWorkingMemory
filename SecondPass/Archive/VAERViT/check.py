"""Focused CPU pretrained-mean equivalence and complete temporal-gradient check."""
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from SecondPass.ThreeFrameConvVAE.model import ThreeFrameConvVAE
from .model import VAERViT

SOURCE = Path('/Users/jonathanmorgan/VAWMRuntime/three_frame_conv_vae_local01/resume01/latest.pt')
SHA256 = 'bf92d8c908eb5f1e10188d4ed2b220e352a0b7b1d9435fc11cef0c5c754b1690'


def main():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(2)
    except RuntimeError: pass
    started = time.monotonic()
    torch.manual_seed(73)
    model = VAERViT()
    fresh = {n: p.detach().clone() for n, p in model.named_parameters()}
    receipt = model.load_pretrained_encoder(SOURCE, expected_sha256=SHA256)
    assert receipt['source_step'] == 9900
    assert all(torch.equal(fresh[n], p) for n, p in model.named_parameters()
               if not n.startswith(('encoder.', 'mu_head.')))
    assert not any(n.startswith(('decoder.', 'logvar_head.')) for n, _ in model.named_parameters())
    vae = ThreeFrameConvVAE()
    vae.load_state_dict(torch.load(SOURCE, map_location='cpu', weights_only=False)['model'])
    vae.eval(); model.eval()
    triplets = torch.rand(1, 3, 3, 100, 100)
    with torch.no_grad():
        expected = vae.encode(triplets, sample=False)
        actual = model.encoder_mu(triplets)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (1, 256, 13, 13)
        assert model.encode_tokens(triplets).shape == (1, 169, 256)
        assert not torch.equal(actual, model.encoder_mu(triplets.flip(1)))
        movie = triplets[:, :2]
        sequence = model(movie)
        history = memory = None
        for frame in movie.unbind(1):
            logits, history, memory = model.stream_step(frame, history, memory)
        torch.testing.assert_close(logits, sequence, atol=1e-6, rtol=1e-5)
        assert torch.equal(model(movie), sequence)
    del vae
    model.train()
    movie = triplets[:, :2].clone().requires_grad_()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    F.cross_entropy(model(movie), torch.tensor([1])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.count_nonzero() > 0, name
    assert movie.grad[:, 0].count_nonzero() > 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    assert all(not torch.equal(before[n], p) for n, p in model.named_parameters())
    record = dict(complete=True, device='cpu', production_steps=0, fixture_optimizer_steps=1,
                  source_checkpoint_sha256=receipt['sha256'], source_step=receipt['source_step'],
                  copied_parameters=receipt['copied_parameters'], copied_tensors=receipt['copied_tensors'],
                  parameters=sum(p.numel() for p in model.parameters()), tensors=len(list(model.parameters())),
                  exact_trained_vae_mu_equivalence=True, token_shape=[169,256], ordered_triplets=True,
                  fresh_nonencoder_parameters_preserved=True, sequence_stream_reset_parity=True,
                  all_parameters_finite_gradients_and_updates=True, earliest_frame_gradient=True,
                  seconds=time.monotonic()-started)
    Path(__file__).with_name('check_results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__ == '__main__': main()
