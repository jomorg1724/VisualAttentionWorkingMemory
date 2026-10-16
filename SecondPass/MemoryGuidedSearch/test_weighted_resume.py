"""Contracts for the isolated static-memory 2x continuation."""
import copy
from collections import Counter
from pathlib import Path

import torch

from . import train as base
from .test_resume import same
from .weighted_resume import WeightedScheduler, restore, verify_sources

ARCHIVE = Path(__file__).parent / 'Milestones/best_136800_before_static_memory_2x_20261014'


def test_exact_quota_and_condition_coverage():
    scheduler = WeightedScheduler()
    counts = Counter()
    cells = {task: [] for task in base.TASKS}
    expected = Counter({task: 2 if task == 'memory_guided_search' else 1 for task in base.TASKS})
    for _ in range(950):
        block = Counter()
        for _ in range(20):
            task, cell = scheduler.next()
            block[task] += 1
            counts[task] += 1
            cells[task].append(cell)
        assert block == expected
    assert counts == Counter({task: weight * 950 for task, weight in expected.items()})
    for task, sequence in cells.items():
        catalog = {cell['id'] for cell in base.TASKS[task]['conditions']}
        width = len(catalog)
        for start in range(0, len(sequence) - width + 1, width):
            assert set(sequence[start:start + width]) == catalog


def test_migration_and_weighted_resume_next_update():
    base.configure('cpu')
    saved = torch.load(ARCHIVE / 'best.pt', map_location='cpu', weights_only=False)
    verify_sources(saved)
    model, opt, scheduler, stream = restore(saved)
    assert same(model.state_dict(), saved['model'])
    assert same(opt.state_dict(), saved['optimizer'])
    assert same(stream.state_dict(), saved['stream'])
    assert scheduler.rng.getstate() == saved['scheduler']['rng']
    assert scheduler.cells == saved['scheduler']['cells']
    assert scheduler.tasks == []  # Only the old unfinished task bag is replaced.
    assert torch.equal(torch.get_rng_state(), saved['torch_rng'])

    for _ in range(7):
        scheduler.next()
    weighted = copy.deepcopy(saved)
    weighted['scheduler'] = scheduler.state_dict()
    task, cell = scheduler.next()
    x, y, metadata = stream.batch(2, task, cell)
    first = base.update(model, opt, x, y, metadata, task, micro=1)
    model2, opt2, schedule2, stream2 = restore(weighted)
    assert same(schedule2.state_dict(), weighted['scheduler'])
    assert schedule2.next() == (task, cell)
    xx, yy, mm = stream2.batch(2, task, cell)
    assert same((x, y, metadata), (xx, yy, mm))
    second = base.update(model2, opt2, xx, yy, mm, task, micro=1)
    assert same(first, second)
    assert same(model.state_dict(), model2.state_dict())
    assert same(opt.state_dict(), opt2.state_dict())
    for _ in range(60):
        assert scheduler.next() == schedule2.next()
