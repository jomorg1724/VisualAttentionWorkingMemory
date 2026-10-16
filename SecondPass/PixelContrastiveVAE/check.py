"""Bounded CPU engineering checks, not evidence of learned performance."""
import json
from pathlib import Path
import numpy as np
import torch
from SecondPass.PixelContrastiveVAE.dataset import generate, generate_evaluation, NAMESPACES
from SecondPass.PixelContrastiveVAE.model import PixelContrastiveVAE, losses, decayed_distance
from SecondPass.VariationalMotionPredictor.dataset import render_positions


def main():
    torch.set_num_threads(2)
    torch.manual_seed(187111)
    checks = {}
    train_rows = [generate(i) for i in range(36)]
    assert sum(row[2] for row in train_rows) == 12
    for label in (0, 1):
        assert {(r[3]['speed'], r[3]['angle']) for r in train_rows if r[2] == label} == {
            (speed, angle) for speed in (.375, 1., 2.) for angle in (26, 28)}
    checks['training_one_third_changes_all_six_cells'] = True
    for index in range(12):
        before, after, label, meta = generate_evaluation(index, 'test')
        p = np.array(meta['positions_unwrapped_xy'])
        vb = np.array(meta['velocity_before_xy'])
        va = np.array(meta['velocity_after_xy'])
        assert np.allclose(np.diff(p[:4], axis=0), vb[None, None])
        assert np.allclose(np.diff(p[3:], axis=0), va[None, None])
        assert torch.equal(before[0], torch.from_numpy(render_positions(p[0])))
        assert torch.equal(after[0], torch.from_numpy(render_positions(p[3])))
        if label:
            b0, a0, _, _ = generate_evaluation(index - 1, 'test')
            assert torch.equal(before, b0)
            assert torch.equal(after[0], a0[0])
            assert not torch.equal(after[-1], a0[-1])
    checks['continuous_positions_and_identical_paired_raster_prefix'] = True
    assert len(set(NAMESPACES.values())) == 3
    assert not torch.equal(generate(0, 'train')[0], generate(0, 'val')[0])
    assert torch.equal(generate(14, 'train')[0], generate(14, 'train')[0])
    checks['reproducible_disjoint_namespaces'] = True

    first = torch.stack([generate_evaluation(i, 'val')[0] for i in (1, 5)])
    second = torch.stack([generate_evaluation(i, 'val')[1] for i in (1, 5)])
    input_with_grad = first.clone().requires_grad_()
    target = decayed_distance(input_with_grad, second)
    assert not target.requires_grad
    assert torch.allclose(decayed_distance(first, first), torch.zeros(2), atol=3e-7)
    manual_a = ((first - .5) * torch.tensor([1., 2., 4.]).reshape(1, 3, 1, 1, 1) / 7).sum(1).flatten(1)
    manual_b = ((second - .5) * torch.tensor([1., 2., 4.]).reshape(1, 3, 1, 1, 1) / 7).sum(1).flatten(1)
    manual = 1 - torch.nn.functional.cosine_similarity(manual_a, manual_b)
    assert torch.allclose(target, manual, atol=3e-7)
    checks['detached_label_free_raw_pixel_target_exact_decay'] = True

    model = PixelContrastiveVAE()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 2000000
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    result_a, result_b = model(first), model(second)
    assert result_a['reconstruction'].shape == first.shape
    metrics = losses(result_a, result_b, first, second, beta=1e-4, pair_weight=1.)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    metrics['loss'].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, initial[name]), name
        assert int(optimizer.state[parameter]['step']) == 1, name
    checks['every_parameter_finite_nonzero_gradient_adam_state_and_update'] = True
    model.eval()
    with torch.no_grad():
        r1, r2 = model(first), model(first)
        assert torch.equal(r1['mu'], r2['mu'])
        assert torch.equal(r1['reconstruction'], r2['reconstruction'])
        assert torch.equal(r1['z'], r1['mu'])
        assert torch.allclose(r1['embedding'].norm(dim=1), torch.ones(2), atol=1e-6)
    model.train()
    r1, r2 = model(first), model(first)
    assert torch.equal(r1['embedding'], r2['embedding'])
    assert not torch.equal(r1['z'], r2['z'])
    assert r1['logvar'].min() >= -8 and r1['logvar'].max() <= 4
    checks['deterministic_mean_pair_eval_stochastic_recon_training'] = True
    model.zero_grad(set_to_none=True)
    a, b = model(first), model(second)
    plain = losses(a, b, first, second, pair_weight=0)
    plain['loss'].backward()
    assert all(p.grad is None for p in model.projection.parameters())
    assert all(p.grad is not None for name, p in model.named_parameters() if not name.startswith('projection.'))
    checks['plain_vae_control_excludes_projection_gradients'] = True
    output = dict(status='passed_cpu_engineering_only', torch_version=torch.__version__,
                  parameters=param_count, parameter_tensors=len(list(model.parameters())),
                  threads=torch.get_num_threads(), checks=checks)
    path = Path(__file__).with_name('check_results.json')
    path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
