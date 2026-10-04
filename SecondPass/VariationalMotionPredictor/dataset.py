"""New synthetic constant-velocity motion benchmark; not a Krauzlis replication.

Persistent Gaussian dots translate on a100x100 torus for four frames. No
fixation, cue, distractor, direction change, response label, or hidden input.
"""
from __future__ import annotations
import copy
import math
import random
import numpy as np
import torch

SIDE=100
SPEEDS=(.375,1.,2.)
NAMESPACES={'train':184101,'val':184102,'test':184103}
VERSION='fullfield_persistent_toroidal_constant_velocity_gaussian_dots_v1'
POOL_SIZE,EPOCHS,BATCH_SIZE=1000,2,32
UPDATES_PER_POOL=64


def render_positions(positions_xy):
    """Periodic7x7 local Gaussian splats with sigma.7, contrast.42, gray.5."""
    positions=np.asarray(positions_xy,dtype=np.float64)%SIDE
    centers=np.floor(positions+.5).astype(np.int64)
    offsets=np.arange(-3,4,dtype=np.int64)
    x=centers[:,0,None]+offsets[None,:]
    y=centers[:,1,None]+offsets[None,:]
    gx=np.exp(-.5*((x-positions[:,0,None])/.7)**2)
    gy=np.exp(-.5*((y-positions[:,1,None])/.7)**2)
    values=.42*gy[:,:,None]*gx[:,None,:]
    image=np.full((SIDE,SIDE),.5,dtype=np.float32)
    np.add.at(image,(np.broadcast_to(y[:,:,None]%SIDE,values.shape),np.broadcast_to(x[:,None,:]%SIDE,values.shape)),values)
    np.clip(image,0.,1.,out=image)
    return np.repeat(image[None,:,:],3,axis=0)


def generate(index:int,split:str='train'):
    """Stateless independent draw; returns past3, fourth target, analysis metadata."""
    if split not in NAMESPACES or not isinstance(index,(int,np.integer)) or index<0:
        raise ValueError('Require nonnegative index and train/val/test split')
    rng=np.random.default_rng(np.random.SeedSequence([NAMESPACES[split],int(index)]))
    angle=float(rng.uniform(0.,2*math.pi)); speed=SPEEDS[int(index)%len(SPEEDS)]
    count=int(rng.integers(16,49))
    # No rejection, boundary margin, directional conditioning, or respawning.
    initial=rng.uniform(0.,float(SIDE),size=(count,2))
    velocity=speed*np.array([math.cos(angle),math.sin(angle)],dtype=np.float64)
    unwrapped=initial[None,:,:]+np.arange(4,dtype=np.float64)[:,None,None]*velocity[None,None,:]
    wrapped=unwrapped%SIDE
    frames=torch.from_numpy(np.stack([render_positions(p) for p in wrapped]))
    metadata=dict(trial_id=f'{VERSION}/{split}/{NAMESPACES[split]}/{int(index)}',split=split,index=int(index),
        namespace=NAMESPACES[split],angle_radians=angle,speed_pixels_per_frame=speed,dot_count=count,
        velocity_xy=velocity.tolist(),initial_positions_xy=initial.tolist(),
        positions_unwrapped_xy=unwrapped.tolist(),positions_wrapped_xy=wrapped.tolist(),
        input_frame_indices=[0,1,2],target_frame_index=3,analysis_only=True)
    return frames[:3],frames[3],metadata


class MotionStream:
    def __init__(self,split='train'):
        if split not in NAMESPACES: raise ValueError('Unknown split')
        self.split=split; self.next_index=0
    def sample(self):
        result=generate(self.next_index,self.split); self.next_index+=1; return result
    def batch(self,n):
        if n<1: raise ValueError('Require positive batch size')
        rows=[self.sample() for _ in range(n)]
        return torch.stack([x[0] for x in rows]),torch.stack([x[1] for x in rows]),[x[2] for x in rows]
    def state_dict(self):
        return dict(version=VERSION,split=self.split,namespace=NAMESPACES[self.split],next_index=self.next_index,
            randomness='stateless SeedSequence(namespace,index); balanced speed index modulo3')
    def load_state_dict(self,state):
        if state['version']!=VERSION or state['split']!=self.split or state['namespace']!=NAMESPACES[self.split]:
            raise ValueError('Dataset namespace/version mismatch')
        if state['next_index']<0: raise ValueError('Negative counter')
        self.next_index=int(state['next_index'])
    def __iter__(self): return self
    def __next__(self): return self.sample()


class MotionPool:
    """1000 fresh four-frame movies, two global permutations, then replace.

The32nd update of each epoch has8 presentations. Checkpoints store only the
index interval, permutation, RNG and counters; raw pixels are never serialized.
    """
    def __init__(self,seed=184111):
        self.stream=MotionStream('train'); self.rng=random.Random(seed)
        self.pool_index=-1; self.epoch=0; self.cursor=0; self.pool_start=None
        self.order=[]; self.data={}; self.presentations=0; self.updates=0
    def _new_pool(self):
        self.pool_index+=1; self.epoch=0; self.cursor=0; self.pool_start=self.stream.next_index
        self.data={i:self.stream.sample() for i in range(POOL_SIZE)}
        self.order=list(range(POOL_SIZE)); self.rng.shuffle(self.order)
    def next_indices(self):
        if self.pool_index<0: self._new_pool()
        if self.cursor==POOL_SIZE:
            self.epoch+=1; self.cursor=0
            if self.epoch==EPOCHS: self._new_pool()
            else: self.order=list(range(POOL_SIZE)); self.rng.shuffle(self.order)
        indices=self.order[self.cursor:self.cursor+BATCH_SIZE]; self.cursor+=len(indices)
        self.presentations+=len(indices); self.updates+=1
        return indices
    def next_batch(self):
        rows=[self.data[i] for i in self.next_indices()]
        return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows]),[r[2] for r in rows]
    def state_dict(self):
        return copy.deepcopy(dict(version=VERSION,pool_size=POOL_SIZE,epochs=EPOCHS,
            stream=self.stream.state_dict(),rng=self.rng.getstate(),pool_index=self.pool_index,
            epoch=self.epoch,cursor=self.cursor,pool_start=self.pool_start,order=self.order,
            presentations=self.presentations,updates=self.updates))
    def load_state_dict(self,state):
        if state['version']!=VERSION or state['pool_size']!=POOL_SIZE or state['epochs']!=EPOCHS: raise ValueError('Pool protocol mismatch')
        self.stream.load_state_dict(state['stream']); self.rng.setstate(state['rng'])
        for key in ('pool_index','epoch','cursor','pool_start','order','presentations','updates'): setattr(self,key,copy.deepcopy(state[key]))
        self.data={} if self.pool_index<0 else {i:generate(self.pool_start+i,'train') for i in range(POOL_SIZE)}
        assert self.pool_index<0 or self.stream.next_index==self.pool_start+POOL_SIZE
