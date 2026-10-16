"""Bounded CPU engineering checks; no claim of trained performance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import dataset
from .model import WeightedSumVAE, NonlinearChangeDecoder, encode_pair, weighted_sum, loss_terms


def main():
    torch.set_num_threads(1)
    torch.manual_seed(190111)
    checks = {}
    frames = torch.stack([torch.full((3, 100, 100), x) for x in (.1, .2, .9)])[None]
    raw = weighted_sum(frames, normalized=False)
    expected = frames[:, 0] + 2 * frames[:, 1] + 3 * frames[:, 2]
    assert torch.equal(raw, expected)
    assert torch.allclose(raw, torch.full_like(raw, 3.2))
    assert torch.allclose(6 * weighted_sum(frames), raw, atol=3e-7)
    assert raw.max() > 1
    assert not torch.equal(weighted_sum(frames), weighted_sum(frames.flip(1)))
    checks['exact_chronological_123_raw_sum_reversible_scaling_no_clip'] = True

    rows = [dataset.generate(i) for i in range(36)]
    assert sum(r[2] for r in rows) == 12
    classified = [dataset.generate_classifier(i) for i in range(24)]
    assert sum(r[2] for r in classified) == 12
    for label in (0, 1):
        assert {r[3]['cell'] for r in classified if r[2] == label} == set(range(6))
    assert len(set(dataset.NAMESPACES.values())) == 3
    assert torch.equal(dataset.generate(5)[0], dataset.generate(5)[0])
    assert not torch.equal(dataset.generate(5)[0], dataset.generate(5, 'val')[0])
    assert not torch.equal(dataset.generate(5)[0], dataset.generate_classifier(5)[0])
    assert not torch.equal(dataset.generate_classifier(5)[0], dataset.generate_classifier(4)[0])
    for i in range(12):
        a, b, label, meta = dataset.generate_evaluation(i, 'test')
        positions = np.asarray(meta['positions_unwrapped_xy'])
        assert np.allclose(np.diff(positions[:4], axis=0), np.asarray(meta['velocity_before_xy'])[None, None])
        assert np.allclose(np.diff(positions[3:], axis=0), np.asarray(meta['velocity_after_xy'])[None, None])
        if label:
            a0, b0, _, _ = dataset.generate_evaluation(i - 1, 'test')
            assert torch.equal(a, a0) and torch.equal(b[0], b0[0])
            assert not torch.equal(b[-1], b0[-1])
    checks['continuous_shared_movie_paired_eval_prefix_all_cells_balanced_classifier_disjoint_modes'] = True

    a = torch.stack([r[0] for r in classified[:2]])
    b = torch.stack([r[1] for r in classified[:2]])
    images = weighted_sum(torch.cat([a, b]))
    model = WeightedSumVAE()
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output = model(images)
    assert output['reconstruction'].shape == (4, 3, 100, 100)
    assert torch.equal(output['raw_reconstruction'], 6 * output['reconstruction'])
    metrics = loss_terms(output, images)
    assert torch.equal(metrics['recon'], F.mse_loss(output['reconstruction'], images))
    assert torch.allclose(metrics['loss'], metrics['recon'] + 1e-4 * metrics['kl'])
    assert all(torch.isfinite(v).all() for v in metrics.values())
    metrics['loss'].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, initial[name]), name
        assert int(optimizer.state[parameter]['step']) == 1, name
    checks['ordinary_image_only_MSE_mean_KL_all_VAE_parameters_finite_changed_Adam'] = True

    model.eval()
    with torch.no_grad():
        first, second = model(images), model(images)
        assert torch.equal(first['mu'], second['mu'])
        assert torch.equal(first['reconstruction'], second['reconstruction'])
        assert torch.equal(first['z'], first['mu'])
        batch32 = model(images[:1].expand(32, -1, -1, -1))
        assert batch32['reconstruction'].shape == (32, 3, 100, 100)
        assert batch32['mu'].shape == (32, 128)
    model.train()
    first, second = model(images), model(images)
    assert torch.equal(first['mu'], second['mu'])
    assert not torch.equal(first['z'], second['z'])
    checks['batch32_correct_shape_deterministic_means_eval_stochastic_train'] = True

    model.eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    mu_a, mu_b = encode_pair(model, a, b)
    assert not mu_a.requires_grad and not mu_b.requires_grad
    changed_b = b.flip(-1)
    same_a, new_b = encode_pair(model, a, changed_b)
    assert torch.equal(mu_a, same_a) and not torch.equal(mu_b, new_b)
    classifier = NonlinearChangeDecoder()
    initial_vae = {n: p.detach().clone() for n, p in model.named_parameters()}
    initial_head = {n: p.detach().clone() for n, p in classifier.named_parameters()}
    head_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    assert torch.equal(classifier(mu_a, mu_b), classifier(torch.cat([mu_a, mu_b], dim=1)))
    assert not torch.equal(classifier(mu_a, mu_b), classifier(mu_b, mu_a))
    F.binary_cross_entropy_with_logits(classifier(mu_a, mu_b), torch.tensor([0., 1.])).backward()
    for name, parameter in classifier.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    head_optimizer.step()
    assert all(p.grad is None and torch.equal(p, initial_vae[n]) for n, p in model.named_parameters())
    assert all(not torch.equal(p, initial_head[n]) and int(head_optimizer.state[p]['step']) == 1
               for n, p in classifier.named_parameters())
    checks['ordered_concat_nonlinear_head_all_Adam_VAE_frozen_independent_clip_encoding'] = True

    directory = Path(__file__).parent
    result = dict(status='passed_cpu_engineering_only', torch_version=torch.__version__,
                  threads=torch.get_num_threads(), VAE_parameters=sum(p.numel() for p in model.parameters()),
                  VAE_parameter_tensors=len(list(model.parameters())),
                  classifier_parameters=sum(p.numel() for p in classifier.parameters()),
                  classifier_parameter_tensors=len(list(classifier.parameters())), checks=checks,
                  source_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                 for name in ('model.py', 'dataset.py', 'check.py')})
    (directory / 'check_results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
