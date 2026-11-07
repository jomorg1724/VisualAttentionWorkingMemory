"""Focused CPU exact-state interrupted/uninterrupted production comparison."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import time
import torch
from . import train


class SimulatedShutdown(RuntimeError):
    pass


def exact(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def arguments(run):
    return Namespace(run=run, profile_json=run / 'profile.json', device='cpu',
        target_updates=16, microbatch=1, val_count=12, test_count=12,
        resume=False, recovery_budget=None)


def setup(args):
    started = time.time()
    args.run.mkdir()
    train.atomic_json(args.profile_json, dict(started=started, deadline=started + 14400.,
        target_updates=16, source_hashes=train.source_hashes(), device='cpu',
        microbatch=1, val_count=12, test_count=12, speed_scale=1.))


def main():
    train.configure('cpu')
    original = train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES, train.BATCH_SIZE
    original_append = train.append_json
    started = time.time()
    results = []
    try:
        train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES, train.BATCH_SIZE = 8, 2, 8, 2
        with tempfile.TemporaryDirectory(prefix='transformer_exact_resume_') as directory:
            root = Path(directory)
            reference = arguments(root / 'uninterrupted')
            setup(reference)
            train.production(reference, reference.run)
            expected = torch.load(reference.run / 'latest.pt', map_location='cpu')
            for stop_step in (5, 8):
                args = arguments(root / ('split_%d' % stop_step))
                setup(args)

                def interrupt(path, value):
                    original_append(path, value)
                    if value.get('event') == 'update' and value['step'] == stop_step:
                        raise SimulatedShutdown('CPU engineering interruption after completed update')

                train.append_json = interrupt
                try:
                    train.production(args, args.run)
                    raise AssertionError('Fixture interruption was not reached')
                except SimulatedShutdown:
                    pass
                finally:
                    train.append_json = original_append
                interrupted = torch.load(args.run / 'latest.pt', map_location='cpu')
                assert interrupted['state']['step'] == stop_step
                before = interrupted['state']
                recovery = args.run / 'Recovery/resume01'
                recovery.mkdir(parents=True)
                clock = time.time()
                args.resume, args.recovery_budget = True, recovery / 'budget.json'
                train.atomic_json(args.recovery_budget, dict(started=clock, deadline=clock + 3600.,
                    wall_seconds=3600, source_hashes=train.source_hashes(), authorization='CPU disposable recovery fixture'))
                train.production(args, args.run)
                actual = torch.load(args.run / 'latest.pt', map_location='cpu')
                for key in ('model', 'optimizer', 'state', 'torch_rng', 'shuffle_rng'):
                    assert exact(expected[key], actual[key]), 'Mismatch: ' + key
                assert train.hashlib.sha256(train.pickle.dumps(expected['numpy_rng'])).digest() == train.hashlib.sha256(train.pickle.dumps(actual['numpy_rng'])).digest()
                assert exact(expected['python_rng'], actual['python_rng'])
                receipt = json.loads((recovery / 'resume_receipt.json').read_text())
                assert receipt['exact_rng_restored'] and receipt['restored_step'] == stop_step
                assert json.loads((args.run / 'completion_verified.json').read_text())['updates'] == 16
                assert json.loads((recovery / 'production_verified.json').read_text())['saved_step'] == stop_step + 1
                results.append(dict(stop_step=stop_step, cursor=before['cursor'], epoch=before['epoch'],
                    seam='end_of_pool' if stop_step == 8 else 'mid_epoch',
                    bitwise_equal_model_and_adam=True, identical_final_sampler_and_rng=True,
                    actual_production_resume_and_finalization=True))
    finally:
        train.POOL_SIZE, train.POOL_EPOCHS, train.POOL_UPDATES, train.BATCH_SIZE = original
        train.append_json = original_append
    output = dict(passed=True, device='cpu', scientific_experiment=False, cases=results,
        cpu_seconds=time.time() - started, source_hashes=train.source_hashes())
    train.atomic_json(Path(__file__).with_name('resume_check_results.json'), output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
