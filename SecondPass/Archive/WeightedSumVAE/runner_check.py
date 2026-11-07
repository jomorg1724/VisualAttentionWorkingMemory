"""CPU-only complete staged production fixture; not a scientific training run."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import time
import torch
from . import train


def main():
    train.configure('cpu')
    original = (train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES)
    results = {}
    try:
        # Real 64 VAE updates and validation/checkpoint routing; tiny disposable
        # pools reduce stimulus exposure only, not the production call path.
        train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES = 1, 2, 2
        with tempfile.TemporaryDirectory(prefix='weighted_sum_staged_fixture_') as directory:
            run = Path(directory)
            args = Namespace(run=run, profile_json=run / 'profile.json',
                device='cpu', target_updates=64, head_updates=64,
                cache_pairs=12, microbatch=1, val_count=12, test_count=12)
            started = time.time()
            train.atomic_json(args.profile_json, dict(started=started, deadline=started + 14400.,
                target_updates=64, head_updates=64, cache_pairs=12, source_hashes=train.source_hashes(),
                device='cpu', microbatch=1, val_count=12, test_count=12))
            train.production(args, run)
            vae = torch.load(run / 'vae/latest.pt', map_location='cpu')
            head = torch.load(run / 'classifier/latest.pt', map_location='cpu')
            assert vae['state']['step'] == head['state']['step'] == 64
            assert all(float(s['step']) == 64 for s in vae['optimizer']['state'].values())
            assert all(float(s['step']) == 64 for s in head['optimizer']['state'].values())
            assert len(vae['optimizer']['state']) == 20 and len(head['optimizer']['state']) == 6
            assert (run / 'vae/validation_000064.json').exists()
            assert (run / 'classifier/validation_000064.json').exists()
            assert (run / 'vae/best.pt').exists() and (run / 'classifier/best.pt').exists()
            final = json.loads((run / 'final_test.json').read_text())
            assert final['n'] == 12 and final['paired_contexts'] == 6
            assert len(final['cells']) == 6 and all(c['n'] == 2 for c in final['cells'])
            assert final['encoder_immutability_verified'] and not final['test_thresholds_fitted']
            assert final['sign'] == 1
            scaler = json.loads((run / 'latent_standardization.json').read_text())
            assert scaler['train_only'] and len(scaler['mean']) == len(scaler['scale']) == 256
            assert all(s >= 1e-6 for s in scaler['scale'])
            recon = json.loads((run / 'vae_final_test.json').read_text())
            assert abs(recon['raw_sum_mse'] - 36 * recon['reconstruction']) < 1e-8
            completion = json.loads((run / 'completion_verified.json').read_text())
            assert completion['status'] == 'complete'
            assert completion['vae_updates'] == completion['classifier_updates'] == 64
            assert (run / 'weighted_reconstructions.png').exists()
            results = dict(passed=True, device='cpu', scientific_experiment=False,
                real_production_fixture=dict(vae_updates=64, classifier_updates=64,
                    validation_checkpoint64=True, active_vae_adam_states=20,
                    active_classifier_adam_states=6, paired_final_presentations=12,
                    paired_final_contexts=6, selected_load_and_finalization=True,
                    encoder_unchanged_and_gradients_absent=True, six_cells_complete=True,
                    train_only_standardization=True, raw_sum_mse_factor36=True),
                cpu_seconds=time.time() - started, disposable_artifacts_removed=True,
                source_hashes=train.source_hashes())
    finally:
        train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES = original
    train.atomic_json(Path(__file__).with_name('runner_check_results.json'), results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
