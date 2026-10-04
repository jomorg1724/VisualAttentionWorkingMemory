import importlib.util
import json
from pathlib import Path
import torch
from SecondPass.JointTraining.core import tree_equal
from SecondPass.SpatialReadout.state import verify_parent,migrate,load_verified
from SecondPass.SpatialReadout.test_migration import SOURCE

def test_cpu_real_optimizer_update_fsynced_progress_full_checkpoint(tmp_path):
    name='SecondPass.SpatialReadout.worker'
    assert importlib.util.find_spec(name) is not None, 'Executing worker missing'
    w=__import__(name,fromlist=['TrainingSession'])
    parent,receipt=verify_parent(SOURCE)
    initial=migrate(parent,receipt)
    config=dict(device='cpu',effective_batch=2,microbatch=1,cap_started=0.,deadline=1e20)
    session=w.TrainingSession(tmp_path,config,initial)
    initial_pointer=session.checkpoint('migration.pt')
    task,cell=session.scheduler.next()
    row=session.train_update(task,cell)
    assert row['step']==1 and row['parent_step']==3393 and row['cumulative_lineage_steps']==3394
    assert row['optimizer_seconds']>0 and row['cumulative_episodes']==2
    assert json.loads((tmp_path/'progress.jsonl').read_text())['step']==1
    pointer=session.checkpoint('checkpoint_000001.pt')
    saved=load_verified(pointer)
    assert saved['state']['step']==1 and saved['scheduler']['updates']==3394
    assert not tree_equal(saved['model']['spatial_gru.candidate.weight'],initial['model']['spatial_gru.candidate.weight'])
    names=saved['optimizer_names']; index=names.index('spatial_gru.candidate.weight')
    assert saved['optimizer']['state'][index]['step']==1
    assert all(k in saved for k in ('rng','stream','scheduler','model','optimizer','migration'))
    assert saved['state']['parent']['episodes']==108576
    assert json.loads((tmp_path/'latest_checkpoint.json').read_text())==pointer
    from SecondPass.SpatialReadout import state
    assert hasattr(state,'verify_updated'), 'Independent post-update verification missing'
    verified=state.verify_updated(initial_pointer,pointer)
    assert verified['branch_step']==1 and len(verified['changed_convgru_tensors'])==4
