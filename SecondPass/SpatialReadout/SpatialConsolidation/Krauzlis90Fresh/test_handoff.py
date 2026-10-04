"""CPU-only allocation-to-supervisor regression: no worker is launched."""
import json
from unittest.mock import patch
import pytest
from . import worker as w


def test_pinned_allocation_survives_bounded_handoff(tmp_path):
    budget = dict(cap_started=100, hard_deadline=28900, deadline=28300,
                  wall_cap_seconds=28800, retrieval_reserve_seconds=600, max_usd=5)
    now = 300
    (tmp_path/'profile').mkdir()
    # Explicit timing fixture, not a measurement from the failed cloud pod.
    profile = dict(complete=True,
        rows=[dict(task=w.TASK, cell=c, episodes=32, seconds=4.)
              for _ in range(2) for c in w.CELLS],
        evaluation_cells=[dict(task=w.TASK, cell=c, n=20, seconds=1.) for c in w.CELLS])
    (tmp_path/'profile/profile.json').write_text(json.dumps(profile))
    (tmp_path/'budget.json').write_text(json.dumps(budget))
    with patch.object(w, 'config_for', side_effect=lambda b: dict(b)), patch.object(w.time, 'time', return_value=now):
        plan = w.pin(tmp_path)
    assert plan['max_steps'] >= w.MIN_USEFUL_UPDATES
    assert plan['max_steps'] < w.TARGET
    # Simulate only elapsed process startup/import time; no cloud or GPU work.
    with patch.object(w.time, 'time', return_value=now+5), patch.object(w.cloud, 'supervise', return_value={'returncode': 0}) as child:
        assert w.supervise(tmp_path, 'run')['returncode'] == 0
        assert child.call_args.args[1] == budget['deadline']
    config = json.loads((tmp_path/'config.json').read_text())
    assert all(config[k] == v for k, v in budget.items())
    assert now + 60 + plan['estimated_total_remaining_seconds'] < budget['deadline']


def test_handoff_still_rejects_stale_allocation(tmp_path):
    budget = dict(cap_started=100, hard_deadline=28900, deadline=28300,
                  wall_cap_seconds=28800, retrieval_reserve_seconds=600, max_usd=5)
    (tmp_path/'budget.json').write_text(json.dumps(budget))
    (tmp_path/'config.json').write_text(json.dumps(dict(budget, estimated_total_remaining_seconds=28000)))
    with patch.object(w.time, 'time', return_value=301), patch.object(w.cloud, 'supervise') as child:
        with pytest.raises(RuntimeError, match='no longer fits original cap'):
            w.supervise(tmp_path, 'run')
        child.assert_not_called()
    assert not (tmp_path/'production_claim.json').exists()
