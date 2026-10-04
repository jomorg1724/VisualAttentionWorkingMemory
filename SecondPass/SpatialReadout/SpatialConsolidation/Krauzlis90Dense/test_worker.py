from pathlib import Path

def test_dense_worker_exists():
    assert Path(__file__).with_name('worker.py').exists(), 'dense worker missing'
    from . import worker as w
    assert w.TARGET==7737
    assert w.provenance()['inherited_weights'] is False
