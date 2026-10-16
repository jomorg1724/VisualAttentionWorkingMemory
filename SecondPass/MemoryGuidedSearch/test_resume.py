import copy
import random
import numpy as np
import torch
import pytest
from .resume import restore, verify_sources
from pathlib import Path
ROOT = Path(__file__).resolve().parent
from . import train as base
from .dataset import Stream


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope='module',autouse=True)
def configure_cpu_once():
    base.configure('cpu')


@pytest.mark.parametrize('folder,expected_step', [('run01',9120),('continuation01',19000)])
def test_exact_restore_stream_schedule_rng_and_update(folder,expected_step):
    saved = torch.load(ROOT / 'LocalRuntime' / folder / 'latest.pt', map_location='cpu', weights_only=False)
    verify_sources(saved)
    assert saved['step'] == expected_step
    untouched = copy.deepcopy(saved['optimizer'])
    model, opt, schedule, stream = restore(saved)
    assert same(model.state_dict(), saved['model'])
    assert same(opt.state_dict(), saved['optimizer'])
    assert same(schedule.state_dict(), saved['scheduler'])
    assert same(stream.state_dict(), saved['stream'])
    actual_rng = (torch.rand(8), np.random.rand(8), [random.random() for _ in range(8)])
    torch.set_rng_state(saved['torch_rng'])
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    expected_rng = (torch.rand(8), np.random.rand(8), [random.random() for _ in range(8)])
    assert same(actual_rng, expected_rng)

    # Independent reconstruction of the saved scheduler and native stream.
    reference_rng = random.Random()
    reference_rng.setstate(saved['scheduler']['rng'])
    tasks = copy.deepcopy(saved['scheduler']['tasks'])
    cells = copy.deepcopy(saved['scheduler']['cells'])
    reference = Stream('train')
    reference.load_state_dict(saved['stream'])
    for _ in range(19):
        if not tasks:
            tasks = list(base.TASKS)
            reference_rng.shuffle(tasks)
        task = tasks.pop()
        if not cells[task]:
            cells[task] = [c['id'] for c in base.TASKS[task]['conditions']]
            reference_rng.shuffle(cells[task])
        cell = cells[task].pop()
        assert schedule.next() == (task, cell)
        assert same(stream.batch(2, task, cell), reference.batch(2, task, cell))

    # Reloading changes neither the next loss nor the complete Adam update.
    model, opt, schedule, stream = restore(saved)
    task, cell = schedule.next()
    x, y, metadata = stream.batch(2, task, cell)
    first = base.update(model, opt, x, y, metadata, task, micro=1)
    second_model, second_opt, second_schedule, second_stream = restore(saved)
    assert second_schedule.next() == (task, cell)
    xx, yy, mm = second_stream.batch(2, task, cell)
    second = base.update(second_model, second_opt, xx, yy, mm, task, micro=1)
    assert same(first, second)
    assert same(model.state_dict(), second_model.state_dict())
    assert same(opt.state_dict(), second_opt.state_dict())
    assert same(saved['optimizer'], untouched)
    assert len(opt.state) == 92

