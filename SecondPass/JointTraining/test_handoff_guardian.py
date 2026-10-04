"""Behavior test for adopting an existing bounded supervisor, never training."""
import importlib
import json
import pytest


def test_guardian_returns_existing_completion_without_starting_a_worker(tmp_path):
    try:
        module=importlib.import_module('SecondPass.JointTraining.handoff_guardian')
    except ModuleNotFoundError:
        pytest.fail('Existing-supervisor handoff guardian missing')
    result=dict(returncode=0,deadline=100,hard_cap_triggered=False)
    (tmp_path/'supervisor_result.json').write_text(json.dumps(result))
    (tmp_path/'report.json').write_text(json.dumps(dict(final_coverage_complete=True)))
    assert module.watch(tmp_path)==result
    assert json.loads((tmp_path/'guardian_result.json').read_text())['supervisor']==result
