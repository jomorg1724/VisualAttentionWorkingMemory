"""CPU-only checks of global gradient cache, causal evaluation and queue gates."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import torch
from . import dataset
from .model import PredictiveConsistencyVAE, augment_clips, vicreg_loss
from .train import gradient_cache, update, evaluate
from . import train as runner
from .queue import atomic_json, predecessor_status, process_identity, source_hashes


def main():
    torch.set_num_threads(1)
    torch.manual_seed(188411)
    checks = {}
    # Keep disposable engineering-model evaluations away from the operational
    # high-offset change test draws used by the real scientific run.
    production_indices = runner.evaluation_indices
    runner.evaluation_indices = lambda split, count: range(24, 24 + count)
    rows = [dataset.generate(i, 'train') for i in range(4)]
    clips = torch.stack([clip[:3] for row in rows for clip in row[:2]])
    views1, views2 = augment_clips(clips), augment_clips(clips)
    full = PredictiveConsistencyVAE()
    cached = copy.deepcopy(full)
    projections1 = full.project(full.encode(views1)[0])
    projections2 = full.project(full.encode(views2)[0])
    full_loss = vicreg_loss(projections1, projections2)
    full_loss['view_loss'].backward()
    (gradient1, gradient2), cache_values = gradient_cache(cached, views1, views2, 'cpu', 2)
    for start in range(0, len(clips), 2):
        p1 = cached.project(cached.encode(views1[start:start + 2])[0])
        p2 = cached.project(cached.encode(views2[start:start + 2])[0])
        torch.autograd.backward((p1, p2), (gradient1[start:start + 2], gradient2[start:start + 2]))
    maximum_error = 0.
    checked = 0
    for (name, a), (other, b) in zip(full.named_parameters(), cached.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
            maximum_error = max(maximum_error, float((a.grad - b.grad).abs().max()))
            checked += 1
    assert abs(float(full_loss['view_loss']) - cache_values['view_loss']) < 1e-6
    checks['whole_batch_vicreg_gradient_cache_matches_full_graph'] = dict(
        clips_per_view=len(clips), microclips=2, parameters_checked=checked, max_absolute_error=maximum_error)
    model = PredictiveConsistencyVAE()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
    result = update(model, optimizer, rows, torch.arange(4), 'cpu', 1, 1, (views1, views2))
    assert result['vicreg_clips_per_view'] == 8
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert not torch.equal(parameter, before[name]) or name == 'projector.2.bias', name
        assert int(optimizer.state[parameter]['step']) == 1, name
    checks['all_active_parameters_receive_gradients_and_adam_updates'] = len(optimizer.state)
    rng = torch.get_rng_state().clone()
    val = evaluate(model, 'val', 12, 'cpu', 2, time.time() + 120)
    assert torch.equal(rng, torch.get_rng_state())
    assert val['first_frame_pair_max_difference'] < 1e-6
    assert val['scores']['first_frame_error']['raw_auc'] == .5
    assert all(cell['presentations'] == 2 for cell in val['scores']['rollout_surprise']['cells'])
    calibration = {key: dict(sign=item['sign'], threshold=item['threshold']) for key, item in val['scores'].items()}
    with tempfile.TemporaryDirectory(prefix='vawm_queue_cpu_') as temporary:
        path = Path(temporary)
        test = evaluate(model, 'test', 12, 'cpu', 2, time.time() + 120, calibration, path)
        assert (path / 'forecast_examples.png').exists()
        assert test['first_frame_pair_max_difference'] < 1e-6
        assert all(test['scores'][key]['sign'] == calibration[key]['sign'] for key in calibration)
        assert all(test['scores'][key]['threshold'] == calibration[key]['threshold'] for key in calibration)
        checks['fixed_validation_calibration_and_identical_first_forecast_control'] = True
        pred = path / 'predecessor'
        pred.mkdir()
        assert predecessor_status(pred, 25024)[0] == 'pending'
        (pred / 'pixel_contrastive').mkdir()
        atomic_json(pred / 'pixel_contrastive' / 'failure.json', {'reason': 'fixture'})
        assert predecessor_status(pred, 25024)[0] == 'blocked'
        atomic_json(pred / 'completion_verified.json', {'status': 'complete'})
        for name in ('pixel_contrastive', 'vae_control'):
            arm = pred / name
            arm.mkdir(exist_ok=True)
            atomic_json(arm / 'completion_verified.json', dict(status='complete', stop_reason='planned_complete', step=25024))
            atomic_json(arm / 'config.json', {'target_updates': 25024})
            atomic_json(arm / 'final_test.json', {'n': 3072})
            (arm / 'best.pt').write_bytes(b'fixture placeholder, not a checkpoint')
            (arm / 'latest.pt').write_bytes(b'fixture placeholder, not a checkpoint')
        assert predecessor_status(pred, 25024)[0] == 'ready'
        atomic_json(pred / 'live_status.json', {'pid': os.getpid()})
        assert predecessor_status(pred, 25024)[0] == 'pending'
        checks['queue_requires_both_complete_arms_and_worker_exit'] = True
        assert process_identity(os.getpid())['pid'] == os.getpid()
        checks['safe_read_only_process_identity'] = True
        # Exercise ACTUAL production orchestration through update64 validation,
        # checkpoint selection, fixed-threshold final test and completion. A
        # tiny disposable CPU pool is an engineering fixture, not a GPU profile
        # or a scientific production run. All artifact directories are temporary.
        smoke_run = path / 'cpu_production_smoke'
        smoke_run.mkdir()
        profile_path = path / 'cpu_fixture_pin.json'
        atomic_json(profile_path, dict(started=time.time(), deadline=time.time() + 240,
                    target_updates=64, source_hashes=source_hashes(), engineering_fixture_cpu_only=True))
        arguments = SimpleNamespace(device='cpu', microbatch=1, target_updates=64,
            val_count=12, test_count=12, profile_json=profile_path, budget_origin=None)
        previous_constants = (runner.POOL_SIZE, runner.POOL_EPOCHS, runner.POOL_UPDATES)
        try:
            runner.POOL_SIZE, runner.POOL_EPOCHS, runner.POOL_UPDATES = 1, 2, 2
            runner.production(arguments, smoke_run)
        finally:
            runner.POOL_SIZE, runner.POOL_EPOCHS, runner.POOL_UPDATES = previous_constants
        completion = json.loads((smoke_run / 'completion_verified.json').read_text())
        assert completion['status'] == 'complete' and completion['step'] == 64
        assert (smoke_run / 'validation_000064.json').exists()
        assert (smoke_run / 'final_test.json').exists() and (smoke_run / 'FINAL_REPORT.md').exists()
        latest = torch.load(smoke_run / 'latest.pt', map_location='cpu')
        best = torch.load(smoke_run / 'best.pt', map_location='cpu')
        assert latest['state']['step'] == 64 and best['state']['step'] == 64
        assert len(latest['optimizer']['state']) == len(best['optimizer']['state']) == 38
        assert all(float(s['step']) == 64 for s in latest['optimizer']['state'].values())
        assert all(float(s['step']) == 64 for s in best['optimizer']['state'].values())
        assert all(torch.equal(value, best['model'][key]) for key, value in latest['model'].items())
        assert completion['selected_step'] == 64 and completion['final_n'] == 12
        checks['actual_cpu_production_through_update64_validation_selection_final_completion'] = dict(
            updates=64, adam_states=38, disposable_pool_pairs=1,
            engineering_fixture=True, gpu_profile=False, scientific_run=False,
            latest_best_optimizer_alignment=True)
    # No CUDA API that creates a context was requested by these fixtures.
    assert not torch.cuda.is_initialized()
    runner.evaluation_indices = production_indices
    record = dict(status='passed_cpu_only_no_gpu_profile', checks=checks,
                  source_hashes=source_hashes(), torch_version=torch.__version__, threads=1)
    Path(__file__).with_name('runner_check_results.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
