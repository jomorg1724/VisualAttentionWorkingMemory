"""Existing evidence with four blank delays; checkpointable balanced scheduling."""
from collections import Counter
import numpy as np
from WorkingMemory.stimuli import BRIDGE_CONDITIONS,TRAIN_CONDITIONS
PARENT_STEP=9840
DELAYS=(0,4,12,24)
def protocol():
    cells={}
    for family,base,label in [('motion_direction',TRAIN_CONDITIONS['integrate_L8'],'motion'),('orientation',BRIDGE_CONDITIONS['bridge_recall'],'orientation')]:
        for d in DELAYS:
            cells[f'{label}_D{d}']=dict(family=family,condition=dict(base,delay=d,post_delay=0,distractor=False),paired_family=label)
    for family in ('motion_direction','orientation'):
        cells[family+'_anchor']=dict(family=family,condition=dict(BRIDGE_CONDITIONS['anchor']),paired_family=None)
    cycle=[name for name in cells for _ in range(4 if name.endswith('_anchor') else 9)]
    assert len(cycle)==80 and sum(v for k,v in Counter(cycle).items() if k.endswith('_anchor'))==8
    return cells,cycle
def scheduled_name(step,cfg):
    offset=step-PARENT_STEP;block,position=divmod(offset,80)
    permutation=np.random.default_rng(cfg['scheduler_seed']+block).permutation(80)
    return cfg['cycle'][int(permutation[position])]
