"""CPU-only behavioural checks for file resilience and exact continuation."""
import copy
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch
from . import dataset, train


def sharing_error():
    error = PermissionError('simulated Windows sharing violation')
    error.winerror = 5
    return error


def check(checkpoint_path):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'receipt.json'
        original_replace = train.os.replace
        attempts = []
        def transient(source, destination):
            attempts.append(1)
            if len(attempts) < 3:
                raise sharing_error()
            return original_replace(source, destination)
        with mock.patch.object(train.os, 'replace', transient), mock.patch.object(train.time, 'sleep'):
            train.atomic_json(path, dict(check='critical_receipt'))
        assert len(attempts) == 3 and json.loads(path.read_text())['check'] == 'critical_receipt'
        with mock.patch.object(train.os, 'replace', side_effect=sharing_error()), mock.patch.object(train.time, 'sleep'):
            train.atomic_json(Path(temporary) / 'live_status.json', dict(step=99))
            try:
                train.atomic_json(path, dict(check='must_raise'))
            except PermissionError:
                pass
            else:
                raise AssertionError('Critical receipt write exhaustion must remain fatal')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['state']
    shuffle = torch.Generator()
    shuffle.set_state(checkpoint['shuffle_rng'])
    saved_shuffle = shuffle.get_state().clone()
    restored_state = copy.deepcopy(state)
    remaining = list(train.continuation_epochs(restored_state, shuffle, True))
    assert len(remaining) == 1
    assert torch.equal(remaining[0], torch.tensor(state['order'][state['cursor']:]))
    assert sum(len(part) for part in remaining) == 200
    assert len(list(remaining[0].split(32))) == 7
    assert torch.equal(shuffle.get_state(), saved_shuffle)

    # Two separate restores must produce exactly the same next CPU update,
    # including stochastic posterior samples and every Adam state.
    indices = remaining[0][:2]
    rows = [dataset.generate(state['pool_start'] + int(i), 'train') for i in indices]
    results = []
    for _ in range(2):
        model, optimizer = train.build('cpu')
        generator = torch.Generator()
        train.restore_checkpoint(model, optimizer, generator, checkpoint, 'cpu')
        train.update(model, optimizer, rows, torch.arange(2), 'cpu', 1, state['step'] + 1, 1.)
        results.append((train.cpu_tree(model.state_dict()), train.cpu_tree(optimizer.state_dict()), torch.get_rng_state()))
        del model, optimizer
    for key in results[0][0]:
        assert torch.equal(results[0][0][key], results[1][0][key]), key
    for key in results[0][1]['state']:
        for field in results[0][1]['state'][key]:
            first, second = results[0][1]['state'][key][field], results[1][1]['state'][key][field]
            assert torch.equal(first, second), (key, field)
    assert torch.equal(results[0][2], results[1][2])
    assert all(float(value['step']) == state['step'] + 1 for value in results[0][1]['state'].values())
    return dict(status='passed', device='cpu', cpu_threads=1, checkpoint_step=state['step'],
                checkpoint_presentations=state['presentations'], optimizer_states=len(checkpoint['optimizer']['state']),
                remaining_current_epoch_presentations=200, remaining_current_epoch_updates=7,
                unchanged_shuffle_until_next_pool=True, exact_next_cpu_update_reproduced=True,
                transient_retries=True, status_write_nonfatal=True, critical_write_exhaustion_fatal=True,
                native_cuda_continuation_not_executed=True, test_weights_not_saved=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.checkpoint)
    train.atomic_json(args.receipt, result)
    print(json.dumps(result), flush=True)
