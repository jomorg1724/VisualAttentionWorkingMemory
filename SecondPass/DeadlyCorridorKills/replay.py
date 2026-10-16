"""Episode replay retaining pixels; never reuse stale learned embeddings."""
import random
from collections import deque
import numpy as np

class Replay:
    def __init__(self,capacity=20000,seed=202211):self.capacity=capacity;self.rng=random.Random(seed);self.episodes=deque();self.transitions=0
    def add(self,episode):
        n=len(episode['actions'])
        if len(episode['observations'])!=n+1 or not n:raise ValueError('Episode observation alignment')
        if not all(len(episode[k])==n for k in ('rewards','done','tics')):raise ValueError('Transition alignment')
        if any(episode['done'][:-1]):raise ValueError('Replay must not cross episode boundaries')
        e={k:np.asarray(v,dtype=np.uint8 if k=='observations' else None).copy() for k,v in episode.items()}
        self.episodes.append(e);self.transitions+=n
        while self.transitions>self.capacity and len(self.episodes)>1:self.transitions-=len(self.episodes.popleft()['actions'])
    def sample(self,batch=2,unroll=8):
        if not self.episodes:raise ValueError('Empty replay')
        sizes=[len(e['actions']) for e in self.episodes];out=[]
        for _ in range(batch):
            # Transition-uniform episode selection, then uniform starting point.
            e=self.rng.choices(list(self.episodes),weights=sizes,k=1)[0];start=self.rng.randrange(len(e['actions']))
            out.append((e,start,min(unroll,len(e['actions'])-start)))
        return out
