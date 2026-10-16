"""Versioned 16-task extension; the completed 13-task catalog stays immutable."""
import copy
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from SecondPass.TaskSuite.suite import SuiteStream, CATALOG as ORIGINAL
from .native.tasks import VDAEnv, VDA9Env
from .native.sequences import collect_supervised_sequence

NAMESPACES=dict(train=201101,val=201102,test=201103)
CATALOG=copy.deepcopy(ORIGINAL)
CATALOG['version']='stabilized_vda_suite_v1'
for size in (4,9,16):
    CATALOG['tasks'].append(dict(id=f'vda{size}',classes=2,group='value_directed',
        stream_id=20+size,requires_bsds500=False,
        labels=['no change','change anywhere'],
        question='Did any Gabor change orientation?',
        conditions=[dict(id=f'TH{theta}',kwargs=dict(theta=theta)) for theta in (8,26,65)],
        report_by=['cue_color','displayed_validity','realized_validity','orientation_change']))
TASKS={t['id']:t for t in CATALOG['tasks']}

@contextmanager
def private_numpy(rng):
    """The native renderer uses legacy global NumPy; isolate and restore it."""
    previous=np.random.get_state();np.random.set_state(rng.get_state())
    try:yield
    finally:rng.set_state(np.random.get_state());np.random.set_state(previous)

def native_env(size,theta):
    kw=dict(theta=theta,curriculum=False,noise_multiplier=5.,T=7,frame_repeat=1)
    if size==9:return VDA9Env(**kw)
    return VDAEnv(grid_rows=int(size**.5),grid_cols=int(size**.5),image_size=50 if size==4 else 100,**kw)

def adapt_pixels(native):
    # Native Gabors include additive +/- .11 noise outside nominal [-1,1].
    # Use a reversible affine map, never clip those informative noise samples.
    x=torch.from_numpy(np.stack(native)).permute(0,3,1,2)
    return F.interpolate((x+1.11)/2.22,size=(100,100),mode='nearest')

class OldStream(SuiteStream):
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return NAMESPACES[self.split]*100000+TASKS[task]['stream_id']*1000+index

class Stream:
    def __init__(self,split='train'):
        if split not in NAMESPACES:raise ValueError(split)
        self.split=split;self.old=OldStream(split);self.rngs={};self.counts={}

    def batch(self,n,task,cell):
        if task not in TASKS:raise ValueError(task)
        if not task.startswith('vda'):return self.old.batch(n,task,cell)
        if not isinstance(n,int) or isinstance(n,bool) or n<1:raise ValueError(n)
        spec=TASKS[task];matches=[(i,c) for i,c in enumerate(spec['conditions']) if c['id']==cell]
        if not matches:raise ValueError(cell)
        index,condition=matches[0];key=(task,cell)
        if key not in self.rngs:
            self.rngs[key]=np.random.RandomState((NAMESPACES[self.split]*100000+spec['stream_id']*1000+index)%(2**32))
            self.counts[key]=0
        xx=[];yy=[];metadata=[]
        with private_numpy(self.rngs[key]):
            for _ in range(n):
                env=native_env(int(task[3:]),condition['kwargs']['theta'])
                native=collect_supervised_sequence(env)
                x=adapt_pixels(native);assert len(x)==7 and env.t==6
                row=dict(env.trial_metadata(),cue_color=env.cue_color,value=env._reward_value(),
                    cue_index=env.cue_index,change_index=env.change_index,orientation_change=float(env.orientation_change),
                    theta=env.theta,noise_degrees=env.noise_multiplier,native_size=env.S,
                    frame_times=list(range(7)),phases=['blank','cue','blank','sample','sample','test','test'],
                    sequence_version='blank_first_v1',suite_task=task,suite_cell=cell,
                    trial_id=self.counts[key],suite_trial_id=f'{task}/{cell}/{self.counts[key]}')
                xx.append(x);yy.append(env.change_true);metadata.append(row);self.counts[key]+=1
        return torch.stack(xx),torch.tensor(yy,dtype=torch.long),metadata

    def state_dict(self):
        return copy.deepcopy(dict(version=CATALOG['version'],split=self.split,old=self.old.state_dict(),
            streams=[dict(task=t,cell=c,rng=r.get_state(),count=self.counts[t,c]) for (t,c),r in sorted(self.rngs.items())]))

    def load_state_dict(self,state):
        if state['version']!=CATALOG['version'] or state['split']!=self.split:raise ValueError('Stream contract mismatch')
        old=OldStream(self.split);old.load_state_dict(state['old']);rngs={};counts={}
        for row in state['streams']:
            key=(row['task'],row['cell']);rng=np.random.RandomState();rng.set_state(row['rng']);rngs[key]=rng;counts[key]=row['count']
        self.old=old;self.rngs=rngs;self.counts=counts

def all_cells():return [(t,c['id']) for t,s in TASKS.items() for c in s['conditions']]
