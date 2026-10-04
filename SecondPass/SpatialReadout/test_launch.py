import importlib.util
import json
from pathlib import Path
import pytest
from SecondPass.SpatialReadout.protocol import new_budget, digest
from SecondPass.SpatialReadout.test_migration import SOURCE


def test_cpu_preparation_records_verified_source_and_no_cap(tmp_path):
    name='SecondPass.SpatialReadout.launch'
    assert importlib.util.find_spec(name) is not None, 'Durable launch entry point missing'
    m=__import__(name,fromlist=['prepare'])
    run=tmp_path/'new'
    result=m.prepare(run,SOURCE)
    assert not (run/'budget.json').exists()
    assert not (run/'migration.pt').exists()
    assert result['parent']['step']==3393 and result['cpu_only']
    assert (run/'preflight_migration.pt').exists()
    assert all(digest(p)==h for p,h in result['source_hashes'].items())
    with pytest.raises(FileExistsError): m.prepare(run,SOURCE)


def test_evaluator_real_cpu_metrics_and_empty_specificity(tmp_path):
    from SecondPass.SpatialReadout.worker import evaluate
    from SecondPass.SpatialReadout.model import SpatialReadout
    from SecondPass.TaskSuite.suite import task_classes
    from SecondPass.SpatialReadout.protocol import FINAL_TEST_NAMESPACE
    assert FINAL_TEST_NAMESPACE==94692763, 'Reserve an untouched final namespace;94592763 was used by an untrained CPU smoke'
    model=SpatialReadout(task_classes())
    out=evaluate(model,'val',4,4,2,'cpu',tmp_path/'val.json',1e20,
        cells=[('contrast','mixed'),('image_recognition','N0_H3')])
    assert out['complete'] and out['complete_cells']==2 and 'final_test_namespace' not in out
    assert out['cells'][0]['n']==4
    assert out['cells'][1]['balanced_accuracy'] is None and out['cells'][1]['auc'] is None
    assert out['cells'][1]['specificity'] is not None
    assert out['summary']['selection_key'] is None
