"""Actual CPU production fixture and balanced full-batch scheduling check."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import time
import torch
from . import train


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(train.SHUFFLE_SEED)
    order = train.balanced_order(generator)
    assert sorted(order.tolist()) == list(range(1000))
    assert all((batch % 2 == 0).sum() == len(batch)//2 for batch in order.split(32))
    original = train.POOL_SIZE, train.POOL_UPDATES, train.BATCH, train.configure
    try:
        # Same complete production control flow with a disposable tiny data pool.
        train.POOL_SIZE, train.POOL_UPDATES, train.BATCH = 8, 2, 8
        train.configure = lambda device: torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='outcome_rvit_runner_') as folder:
            run = Path(folder); now = time.time()
            args = Namespace(device='cpu', val_count=12, test_count=12)
            train.atomic_json(run/'profile.json', dict(started=now, deadline=now+14400,
                target_updates=2, source_hashes=train.source_hashes(), device='cpu', val_count=12, test_count=12))
            train.production(args, run)
            saved = torch.load(run/'latest.pt', map_location='cpu')
            assert saved['state']['step'] == 2 and saved['state']['presentations'] == 16
            states = saved['optimizer']['state']
            assert len(states) == 36 and all(float(v['step']) == 2 for v in states.values())
            result = json.loads((run/'final_test.json').read_text())
            assert result['n'] == 12 and result['paired_contexts'] == 6 and len(result['cells']) == 6
            assert result['threshold'] == 0 and result['sign'] == 1 and not result['test_thresholds_fitted']
            assert (run/'completion_verified.json').exists()
    finally:
        train.POOL_SIZE, train.POOL_UPDATES, train.BATCH, train.configure = original
    proof = dict(passed=True, cpu_only=True, disposable_fixture=True,
        all_full_batches_16_per_class=True, tail_4_per_class=True,
        actual_production_validation_save_selection_final_test=True,
        all36_adam_states_advanced=True, source_hashes=train.source_hashes())
    Path(__file__).with_name('runner_check_results.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
