"""Sampler boundary fixture: no movie rendering, GPU or optimizer training."""
import random
from . import cloud_worker as w

class FakePool(w.ReplayStream):
    def _new_pool(self):
        self.pool_index+=1; self.epoch=0; self.cursor=0; self.used=set()
        self.sizes={c:333+int(i==self.pool_index%3) for i,c in enumerate(w.CELLS)}
        self.order=w.original.replay.epoch_batches(self.sizes,self.rng)


def test_refresh_after_two_complete_epochs():
    stream=FakePool(); scheduler=w.ReplayScheduler(stream)
    presentations=0; epochs={0:set(),1:set()}
    for step in range(1,67):
        task,cell=scheduler.next(); indices=stream.current[1]
        assert stream.pool_index==0 and stream.epoch==(step-1)//33
        assert not epochs[stream.epoch].intersection((cell,i) for i in indices)
        epochs[stream.epoch].update((cell,i) for i in indices)
        stream.commit(cell,indices); presentations+=len(indices)
    assert presentations==2000 and len(epochs[0])==len(epochs[1])==1000
    assert scheduler.state_dict()['updates']==66 and scheduler.state_dict()['updates_per_pool']==66
    scheduler.next(); assert stream.pool_index==1 and stream.epoch==0 and stream.cursor==1
    assert stream.sizes[w.CELLS[1]]==334
    assert w.WeightedMeanRViT is w.original.WeightedMeanRViT
    assert w.provenance()['inherited_weights'] is False and w.provenance()['replay_epochs']==2
