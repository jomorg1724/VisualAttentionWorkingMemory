"""CPU-only actual-production fixture; not a scientific training run."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import time
import torch
from . import train


def main():
    train.configure('cpu')
    original = train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES
    try:
        train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES = 1, 2, 2
        with tempfile.TemporaryDirectory(prefix='weighted_transformer_fixture_') as directory:
            run = Path(directory)
            args = Namespace(run=run, profile_json=run / 'profile.json', device='cpu',
                target_updates=64, microbatch=1, val_count=12, test_count=12)
            started = time.time()
            train.atomic_json(args.profile_json, dict(started=started, deadline=started + 14400.,
                target_updates=64, source_hashes=train.source_hashes(), device='cpu',
                microbatch=1, val_count=12, test_count=12))
            train.production(args, run)
            latest = torch.load(run / 'latest.pt', map_location='cpu')
            expected = len(list(train.WeightedSumTransformer().parameters()))
            assert latest['state']['step'] == 64
            assert len(latest['optimizer']['state']) == expected
            assert all(float(item['step']) == 64 for item in latest['optimizer']['state'].values())
            assert (run / 'validation_000064.json').exists() and (run / 'best.pt').exists()
            final = json.loads((run / 'final_test.json').read_text())
            assert final['n'] == 12 and final['paired_contexts'] == 6
            assert len(final['cells']) == 6 and all(cell['n'] == 2 for cell in final['cells'])
            assert final['sign'] == 1 and not final['test_thresholds_fitted']
            assert 'zero_logit_balanced_accuracy' in final
            receipt = json.loads((run / 'completion_verified.json').read_text())
            assert receipt['status'] == 'complete' and receipt['updates'] == 64
            proof = json.loads((run / 'production_verified.json').read_text())
            assert proof['all_weights_finite'] and proof['all_adam_tensors_finite']
            results = dict(passed=True, device='cpu', scientific_experiment=False,
                actual_production_updates=64, validation_and_save64=True,
                all_active_adam_states=expected, selected_load_finalization=True,
                paired_final_presentations=12, paired_contexts=6, all_six_cells_complete=True,
                test_uses_validation_threshold=True, zero_logit_accuracy_reported=True,
                cpu_seconds=time.time() - started, disposable_artifacts_removed=True,
                source_hashes=train.source_hashes())
    finally:
        train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES = original
    train.atomic_json(Path(__file__).with_name('runner_check_results.json'), results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
