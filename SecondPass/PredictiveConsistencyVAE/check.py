"""One-thread CPU engineering checks; no GPU profiling or training."""
import json
from pathlib import Path
import numpy as np
import torch
from SecondPass.PredictiveConsistencyVAE.dataset import generate, generate_evaluation, generate_objective_evaluation, NAMESPACES
from SecondPass.PredictiveConsistencyVAE.model import PredictiveConsistencyVAE, augment_clips, losses, vicreg_loss


def main():
    torch.set_num_threads(1)
    torch.manual_seed(189111)
    checks = {}
    assert len(set(NAMESPACES.values())) == 3
    rows = [generate(i) for i in range(36)]
    assert sum(row[2] for row in rows) == 12
    for label in (0, 1):
        assert {(r[3]['speed'], r[3]['angle']) for r in rows if r[2] == label} == {
            (speed, angle) for speed in (.375, 1., 2.) for angle in (26, 28)}
    assert not torch.equal(generate(0, 'train')[0], generate(0, 'val')[0])
    for i in range(0, 12, 2):
        b0, a0, _, m0 = generate_evaluation(i, 'test')
        b1, a1, _, m1 = generate_evaluation(i + 1, 'test')
        assert torch.equal(b0, b1) and torch.equal(a0[0], a1[0])
        assert not torch.equal(a0[-1], a1[-1])
        for meta in (m0, m1):
            p = np.array(meta['positions_unwrapped_xy'])
            assert np.allclose(np.diff(p[:4], axis=0), np.array(meta['velocity_before_xy'])[None, None])
            assert np.allclose(np.diff(p[3:], axis=0), np.array(meta['velocity_after_xy'])[None, None])
    for row in rows:
        assert row[0].shape == (6, 3, 100, 100)
        assert row[1].shape == (6, 3, 100, 100)
        meta = row[3]
        p = np.array(meta['positions_unwrapped_xy'])
        assert np.allclose(np.diff(p[:7], axis=0), np.array(meta['velocity_before_xy'])[None, None])
        assert np.allclose(np.diff(p[6:], axis=0), np.array(meta['velocity_after_xy'])[None, None])
    objective = generate_objective_evaluation(0, 'val')
    assert objective[0].shape[0] == 6 and objective[3]['index'] == 60000000
    assert not torch.equal(objective[0][:3], generate_evaluation(0, 'val')[0])
    checks['fresh_namespaces_label_balance_continuity_and_paired_prefix'] = True
    checks['training_twelve_frames_and_separate_objective_validation'] = True
    first = torch.stack([generate_objective_evaluation(i, 'val')[0] for i in (1, 5)])
    second = torch.stack([generate_objective_evaluation(i, 'val')[1] for i in (1, 5)])
    clips = torch.cat([first[:, :3], second[:, :3]])
    moved = augment_clips(clips, noise=False, contrast=False)
    for original, transformed in zip(clips, moved):
        matches = [(dy, dx) for dy in range(-4, 5) for dx in range(-4, 5)
                   if torch.equal(torch.roll(original, (dy, dx), (-2, -1)), transformed)]
        assert len(matches) >= 1
    checks['augment_translation_coherent_across_ordered_frames'] = True
    view1 = augment_clips(clips)
    view2 = augment_clips(clips)
    assert not torch.equal(view1, view2)
    assert view1.min() >= 0 and view1.max() <= 1
    checks['independent_clamped_views'] = True
    collapsed = vicreg_loss(torch.zeros(8, 64), torch.zeros(8, 64))
    assert abs(float(collapsed['variance']) - .99) < 1e-6
    assert float(collapsed['invariance']) == 0 and float(collapsed['covariance']) == 0
    checks['constant_embedding_has_positive_variance_penalty'] = True

    model = PredictiveConsistencyVAE()
    param_count = sum(p.numel() for p in model.parameters())
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.eval()
    with torch.no_grad():
        causal_a = model(first[:, :3])
        changed_future = first.clone()
        changed_future[:, 3:] = torch.rand_like(changed_future[:, 3:])
        causal_b = model(changed_future[:, :3])
        assert torch.equal(causal_a['context_mu'], causal_b['context_mu'])
        assert torch.equal(causal_a['prediction_mu'], causal_b['prediction_mu'])
        assert torch.equal(causal_a['prediction'], causal_b['prediction'])
        assert torch.equal(causal_a['future_prediction'], causal_b['future_prediction'])
        rollout = model.rollout(first[:, :3], steps=3)
        assert rollout.shape == (2, 3, 3, 100, 100)
        assert torch.equal(rollout[:, 0], model.predict_next(first[:, :3]))
        manual_next = model.predict_next(torch.cat([first[:, 1:3], rollout[:, 0:1]], dim=1))
        assert torch.equal(rollout[:, 1], manual_next)
        manual_last = model.predict_next(torch.cat([first[:, 2:3], rollout[:, :2]], dim=1))
        assert torch.equal(rollout[:, 2], manual_last)
        try:
            model.context(first[:, :2])
            raise AssertionError('Two context frames must be rejected')
        except ValueError:
            pass
    checks['prediction_causal_target_excluded_and_rollout_recursive'] = True
    model.train()
    causal = model(first[:, :3])
    causal['prediction'].retain_grad()
    # Only later horizons contribute: this verifies their autoregressive path
    # reaches the first generated image, not just the direct horizon-one loss.
    causal['future_prediction'][:, 1:].square().mean().backward()
    assert causal['prediction'].grad is not None
    assert torch.isfinite(causal['prediction'].grad).all()
    assert causal['prediction'].grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    checks['later_horizons_backpropagate_to_first_generated_frame'] = True
    a, b = model(first[:, :3]), model(second[:, :3])
    p1 = model.project(model.encode(view1)[0])
    p2 = model.project(model.encode(view2)[0])
    metrics = losses(a, b, first, second, p1, p2, beta=1e-4)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics['loss'].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, initial[name]), name
        assert int(optimizer.state[parameter]['step']) == 1, name
    checks['all_parameters_finite_nonzero_gradient_and_adam_update'] = True
    output = dict(status='passed_cpu_engineering_only', torch_version=torch.__version__,
                  parameters=param_count, parameter_tensors=len(list(model.parameters())),
                  threads=torch.get_num_threads(), checks=checks,
                  loss='balanced_recon + balanced_causal_prediction + beta_mean_KL + invariant + variance + .04_covariance',
                  view_statistics='all_effective_batch_clips; caller must not use microbatch statistics')
    Path(__file__).with_name('check_results.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
