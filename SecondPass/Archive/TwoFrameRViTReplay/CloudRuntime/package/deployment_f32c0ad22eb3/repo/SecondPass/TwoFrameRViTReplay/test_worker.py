import random
import torch
from SecondPass.TwoFrameRViTReplay import worker as w

def test_epoch1000_unique_tenfold_and_partial_weights():
    sizes={c:333+int(i==0) for i,c in enumerate(w.CELLS)}
    rng=random.Random(7); seen={}; orders=[]
    for epoch in range(10):
        batches=w.epoch_batches(sizes,rng); orders.append(batches)
        assert len(batches)==33
        ids=[(c,i) for c,indices in batches for i in indices]
        assert len(ids)==len(set(ids))==1000
        assert sorted(map(len,[indices for _,indices in batches]))==[13,13,14]+[32]*30
        assert all(len(ix)==32 for _,ix in batches[:9])
        for identity in ids: seen[identity]=seen.get(identity,0)+1
    assert len(seen)==1000 and set(seen.values())=={10}
    assert orders[0]!=orders[1]
    # Mean losses from partial microbatches reproduce mean loss of actual batch.
    for actual in (13,14,32):
        losses=torch.arange(actual,dtype=torch.float32)
        weighted=sum(losses[i:i+4].mean()*len(losses[i:i+4])/actual for i in range(0,actual,4))
        assert torch.allclose(weighted,losses.mean())

def test_sampler_no_premature_regeneration_and_resume(monkeypatch):
    calls=[]
    def fake_render(self,native):
        calls.append(self.pool_index)
        self.data={c:[(torch.tensor(float(i)),i%2) for i in range(n)] for c,n in self.sizes.items()}
        return {c:[f'{self.pool_index}/{c}/{i}' for i in range(n)] for c,n in self.sizes.items()}
    monkeypatch.setattr(w.ReplayStream,'_render',fake_render)
    stream=w.ReplayStream()
    for step in range(330):
        _,cell=stream.next_plan(); _,indices=stream.current; stream.commit(cell,indices)
        assert stream.pool_index==0
        assert sum(map(len,stream.pool_ids.values()))==1000
    assert stream.presentations==10000 and len(stream.used)==1000 and calls==[0]
    stream.next_plan()
    assert stream.pool_index==1 and stream.sizes['B20']==334 and calls==[0,1]
    # Snapshot has IDs/streams/cursors, never raster tensors.
    state=stream.state_dict()
    assert 'data' not in state['replay']
    assert state['replay']['cursor']==1 and state['replay']['epoch']==0
    restored=w.ReplayStream(); restored.load_state_dict(state)
    for _ in range(35):
        assert restored.next_plan()==stream.next_plan() and restored.current==stream.current
        cell,indices=stream.current
        stream.commit(cell,indices); restored.commit(cell,indices)
    assert restored.state_dict()==stream.state_dict()
