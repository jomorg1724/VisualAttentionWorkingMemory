import importlib.util
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import torch

def test_fresh_native_cpu():
    path=Path(__file__).with_name('worker.py')
    assert path.exists(), 'Krauzlis-only fresh adapter missing'
    from SecondPass.SpatialReadout.SpatialConsolidation.KrauzlisOnly import worker as w
    w.fresh.cloud.original.cpu_setup()
    with tempfile.TemporaryDirectory() as tmp:
        cfg=dict(device='cpu',cap_started=time.time(),effective_batch=4,microbatch=4,disposable_profile=True)
        with patch('torch.load',side_effect=AssertionError('Initialization cannot load checkpoints')):
            s=w.Session(tmp,cfg)
        torch.manual_seed(w.INIT_SEED)
        expected=w.SpatialConsolidation(w.task_classes()).state_dict()
        assert all(torch.equal(v,expected[k]) for k,v in s.model.state_dict().items())
        assert not s.optimizer.state and s.stream.state_dict()['streams']==[] and s.scheduler.updates==0
        assert all(p.requires_grad for p in s.model.parameters())
        s.checkpoint('initial.pt')
        for _ in range(3): s.train_update(*s.scheduler.next())
        s.checkpoint('checkpoint_000003.pt')
        assert w.verify_progress(tmp)['verified']
        assert set(s.state['exposure'])=={w.TASK}
        assert {r['cell'] for r in s.stream.state_dict()['streams']}==set(w.CELLS)
        assert len({w.FreshStream(split).stream_seed(w.TASK,'B12') for split in ('train','val','test')})==3
        from SecondPass.TaskSuite.suite import SuiteStream
        native=SuiteStream('train'); native.stream_seed=s.stream.stream_seed
        a=w.FreshStream('train').batch(4,w.TASK,'B12'); b=native.batch(4,w.TASK,'B12')
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
        result=w.evaluate(s.model,'val',4,4,4,'cpu',Path(tmp)/'eval.json',time.time()+120)
        assert result['complete'] and len(result['cells'])==3
        assert all('confusion' in e for r in result['cells'] for e in r['events'].values())
        assert w.selection_key(dict(complete=False,cells=[])) is None
