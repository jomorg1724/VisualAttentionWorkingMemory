from pathlib import Path
import json
import time
from unittest.mock import patch
import torch


def test_fresh_cpu(tmp_path):
    assert Path(__file__).with_name('worker.py').exists(), 'fresh cloud adapter missing'
    from . import worker as w
    torch.set_num_threads(1)
    cfg=dict(device='cpu',cap_started=time.time(),effective_batch=32,microbatch=4,disposable_profile=True)
    with patch('torch.load',side_effect=AssertionError('No checkpoint initialization')):
        s=w.Session(tmp_path,cfg)
    torch.manual_seed(w.INIT_SEED)
    direct=w.SpatialConsolidation(w.task_classes()).state_dict()
    assert all(torch.equal(v,direct[k]) for k,v in s.model.state_dict().items())
    assert not s.optimizer.state and s.stream.state_dict()['streams']==[] and s.scheduler.updates==0
    assert s.optimizer.param_groups[0]['lr']==1e-4
    s.checkpoint('initial.pt')
    for _ in range(3): s.train_update(*s.scheduler.next())
    s.checkpoint('checkpoint_000003.pt')
    # The save hook itself must publish the single-task startup verification.
    assert json.loads((tmp_path/'persisted_progress_verification.json').read_text())['step']==3
    assert w.verify_progress(tmp_path)['verified']
    assert set(s.state['exposure'])=={w.TASK}
    seeds={w.FreshStream(split).stream_seed(w.TASK,c) for split in ('train','val','test') for c in w.CELLS}
    assert len(seeds)==9
    _,_,rows=w.FreshStream('train').batch(1,w.TASK,'B12')
    assert rows[0]['event_magnitude_degrees']==90
    for role in ('val','test'):
        result=w.evaluate(s.model,role,4,4,4,'cpu',tmp_path/(role+'.json'),time.time()+180)
        assert result['complete'] and len(result['cells'])==3
        assert all('confusion' in e for r in result['cells'] for e in r['events'].values())
    assert w.selection_key(dict(complete=False,cells=[])) is None
