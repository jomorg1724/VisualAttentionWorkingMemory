"""Unchanged native Krauzlis renderer, fresh deterministic streams per pool/cell."""
import numpy as np
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

TASK='krauzlis_cued_motion'
CELLS=(12,20,28)
NAMESPACES=dict(train=196101,val=196102,test=196103)


def render(split, cell, count, block=0):
    seed=int(np.random.SeedSequence([NAMESPACES[split],cell,block]).generate_state(1)[0])
    stream=SpatialBatteryStream(seed,split)
    rows=[]
    for _ in range(count):
        x,y,m=stream.batch(1,TASK,dict(baseline_transitions=cell))
        # Exact native RGB floats; lossless removal of repeated channels in RAM.
        assert (x[:,:,0]==x[:,:,1]).all() and (x[:,:,0]==x[:,:,2]).all()
        rows.append((x[0,:,:1].contiguous(),int(y[0]),m[0]))
    return rows


def pool(index):
    return {cell:render('train',cell,333+int(j==index%3),index)
            for j,cell in enumerate(CELLS)}
