import copy
import importlib.util
import json
from pathlib import Path
import torch
import pytest
from SecondPass.JointTraining.core import tree_equal
from SecondPass.TaskSuite.suite import SuiteStream, task_classes
from SecondPass.JointTraining.core import BalancedScheduler
from SecondPass.SpatialReadout.model import SpatialReadout

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/latest_checkpoint.json'


def module():
    name='SecondPass.SpatialReadout.state'
    assert importlib.util.find_spec(name) is not None, 'Migration not implemented'
    return __import__(name, fromlist=['migrate'])


def test_real_source_name_shape_adam_rng_stream_and_roundtrip(tmp_path):
    m=module()
    source, receipt=m.verify_parent(SOURCE)
    rng=torch.get_rng_state().clone()
    migrated=m.migrate(source,receipt,initialization_seed=94592763)
    assert torch.equal(rng,torch.get_rng_state()), 'Initialization leaked into caller RNG'
    assert migrated['state']['step']==0 and migrated['state']['parent']['step']==3393
    assert migrated['state']['best_key'] is None and migrated['state']['selection_history']==[]
    for key in ('scheduler','stream','rng'):
        assert tree_equal(migrated[key],source[key]),key
    old_names=m.legacy_optimizer_names(source)
    old_states=dict(zip(old_names,[source['optimizer']['state'].get(i) for i in source['optimizer']['param_groups'][0]['params']]))
    new_names=migrated['optimizer_names']
    for index,name in zip(migrated['optimizer']['param_groups'][0]['params'],new_names):
        if name in source['model']:
            assert torch.equal(migrated['model'][name],source['model'][name]),name
            assert tree_equal(migrated['optimizer']['state'][index],old_states[name]),name
        else:
            assert index not in migrated['optimizer']['state'],name
    assert set(migrated['migration']['removed_names'])=={n for n in source['model'] if n.startswith(('feat.','gru.'))}
    assert len([n for n in migrated['migration']['carried_names'] if n.startswith('heads.')])==26
    pointer=m.save_payload(tmp_path/'migration.pt',migrated)
    reloaded=m.load_verified(pointer)
    assert tree_equal(reloaded,migrated)
    model=SpatialReadout(task_classes()); optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=BalancedScheduler(0); stream=SuiteStream('train')
    m.restore(reloaded,model,optimizer,scheduler,stream,'cpu')
    expected=BalancedScheduler(0); expected.load_state_dict(source['scheduler'])
    assert scheduler.next()==expected.next()
    replay=SuiteStream('train'); replay.load_state_dict(source['stream'])
    assert tree_equal(stream.batch(2,'contrast','mixed'),replay.batch(2,'contrast','mixed'))
    pointer['sha256']='0'*64
    with pytest.raises(ValueError,match='digest'): m.load_verified(pointer)


def test_adam_transfer_is_named_not_destination_index():
    m=module()
    old={'a':torch.ones(3),'b':torch.ones(2)}
    saved={'state':{8:{'step':torch.tensor(9.),'exp_avg':torch.ones(3),'exp_avg_sq':torch.ones(3)},
                    5:{'step':torch.tensor(4.),'exp_avg':2*torch.ones(2),'exp_avg_sq':3*torch.ones(2)}},
           'param_groups':[{'params':[8,5],'lr':1e-4}]}
    result=m.map_adam(saved,['a','b'],old,['b','fresh','a'],{'b':torch.ones(2),'fresh':torch.ones(1),'a':torch.ones(3)})
    assert result['state'][0]['step']==4 and result['state'][2]['step']==9 and 1 not in result['state']
    malformed=copy.deepcopy(saved); malformed['state'][8]['exp_avg']=torch.zeros(7)
    with pytest.raises(ValueError,match='shape'): m.map_adam(malformed,['a','b'],old,['a'],old)


def test_portable_source_ledger_preserves_digests_without_old_edits(tmp_path):
    m=module()
    assert hasattr(m,'portable_source_config'), 'Launchd cannot read protected Desktop: portable source ledger needed'
    original={'catalog':{'x':1},'source_manifest_sha256':'manifest',
        'source_hashes':{str(ROOT/'SecondPass/JointTraining/worker.py'):'workerhash',str(ROOT/'WorkingMemory/PlainBaseline/accum.py'):'modelhash'}}
    prior=copy.deepcopy(original)
    relocated=m.portable_source_config(original,tmp_path)
    assert relocated['source_hashes'][str(tmp_path/'SecondPass/JointTraining/worker.py')]=='workerhash'
    assert relocated['source_hashes'][str(tmp_path/'WorkingMemory/PlainBaseline/accum.py')]=='modelhash'
    assert original==prior and relocated['source_manifest_sha256']=='manifest'
