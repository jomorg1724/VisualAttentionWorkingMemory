import copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
import pytest
from .dataset import Stream,private_numpy,native_env,adapt_pixels,all_cells,TASKS
from .native.sequences import collect_supervised_sequence
from .model import initialize
from .train import PARENT,Scheduler,update
from SecondPass.StabilizedAttentionSuite.model import StabilizedAttentionSuite

torch.set_num_threads(2)

@pytest.mark.parametrize('size',[4,9,16])
def test_native_sequence_and_invalid_change_semantics(size):
    s=Stream();task=f'vda{size}'
    # Initialize the stream, then replay its exact RNG state independently.
    s.batch(1,task,'TH65');state=s.state_dict();rng=np.random.RandomState();rng.set_state(state['streams'][0]['rng'])
    with private_numpy(rng):
        env=native_env(size,65);raw=collect_supervised_sequence(env);expected=adapt_pixels(raw)
    x,y,meta=s.batch(1,task,'TH65')
    assert torch.equal(x[0],expected) and y.item()==env.change_true
    assert torch.all(x[0,0]==.5) and torch.all(x[0,2]==.5) and x.shape==(1,7,3,100,100)
    assert env.t==6 and x.min()>=0 and x.max()<=1
    # A 0%-validity cue excludes the cue: an uncued change is STILL a positive.
    with private_numpy(np.random.RandomState(57)):
        e=native_env(size,65);e.proportions=[0.]
        for _ in range(100):
            e.reset()
            if e.change_true:assert e.change_index!=e.cue_index and e._is_reportable_change()

def test_stream_restore_and_global_rng_isolation():
    np.random.seed(919);before=copy.deepcopy(np.random.get_state());s=Stream();s.batch(2,'vda9','TH26')
    after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    snap=s.state_dict();x,y,m=s.batch(3,'vda9','TH26');restored=Stream();restored.load_state_dict(snap);xx,yy,mm=restored.batch(3,'vda9','TH26')
    assert torch.equal(x,xx) and torch.equal(y,yy) and m==mm

def test_pinned_native_sources_and_balanced_scheduler():
    root=Path(__file__).parent;p=json.loads((root/'provenance.json').read_text())
    for name,wanted in p['files_sha256'].items():assert hashlib.sha256((root/'native'/name).read_bytes()).hexdigest()==wanted
    scheduler=Scheduler();assert {scheduler.next()[0] for _ in range(16)}==set(TASKS)
    assert len(TASKS)==16 and len(all_cells())==44

def test_migrated_backbone_adam_and_new_head_training():
    before=hashlib.sha256(PARENT.read_bytes()).hexdigest();model,opt=initialize(PARENT)
    saved=torch.load(PARENT,map_location='cpu',weights_only=False);old=StabilizedAttentionSuite();old.load_state_dict(saved['model'])
    x,y,meta=Stream().batch(2,'vda4','TH65')
    with torch.no_grad():assert torch.equal(model(x,'orientation'),old(x,'orientation'))
    assert len(opt.state)==80 and len(list(model.parameters()))==86
    weights=model.heads.vda4.weight.detach().clone();old_head=model.heads.orientation.weight.detach().clone()
    result=update(model,opt,x,y,meta,'vda4',micro=1)
    assert np.isfinite(result['loss']) and not torch.equal(model.heads.vda4.weight,weights)
    assert torch.equal(model.heads.orientation.weight,old_head) and len(opt.state)==82
    assert before==hashlib.sha256(PARENT.read_bytes()).hexdigest()
