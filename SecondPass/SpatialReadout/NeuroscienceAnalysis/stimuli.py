"""Analysis-only magnitude override, otherwise bit-identical native RNG law.
Unlike historical PsychOrientationStream, no extra permutation for keep-sites.
Always draw native magnitude first, then override without extra RNG calls.
Zero magnitude is a negative catch; intended balanced label is recomputed.
"""
import copy
import numpy as np
import torch
from WorkingMemory.SpatialTaskBattery import stimuli as S

class OrientationSweep(S.SpatialBatteryStream):
    def __init__(self,seed,magnitude=None):
        super().__init__(seed,'test');self.magnitude=magnitude;self.raw=[]
    def _orientation(self,rng,label,target,cfg):
        sign=self._cue_sign;angles=rng.uniform(0,np.pi,4);native=int(rng.choice([15,30,45]));magnitude=native if self.magnitude is None else self.magnitude
        pattern=np.array([-1,0,1,int(rng.choice([-1,0,1]))])
        wanted=1 if label else int(rng.choice([-1,0]));i=int(rng.choice(np.flatnonzero(pattern==wanted)));delta=np.empty(4,dtype=int);delta[target]=pattern[i];others=[j for j in range(4) if j!=target];delta[others]=rng.permutation(np.delete(pattern,i));delta=delta*sign*magnitude
        probe=(angles+np.deg2rad(delta))%np.pi;delay=int(cfg.get('delay',0));raw=[S.blank(),self._gabors(rng,angles),self._gabors(rng,angles)]
        frames=[S.local_cue(im,'recall','sample',target,sign) for im in raw]
        frames.extend(S.local_cue(S.blank(),'recall','ignore',target,visible=False) for _ in range(delay));frames.append(S.local_cue(self._gabors(rng,probe),'recall','report',target,visible=False))
        self.raw.append(np.stack(raw))
        return frames,dict(label=int(delta[target]*sign>0),intended_label=label,target_location=target,cue_sign=sign,positions_xy=S.CENTERS.tolist(),sample_angles_radians=angles.tolist(),probe_angles_radians=probe.tolist(),rotations_degrees=delta.tolist(),rotation_magnitude=magnitude,native_drawn_magnitude=native,ood_magnitude=magnitude not in (15,30,45),rotation_multiset_before_target_assignment=pattern.tolist(),label_semantics='1 iff cued target rotates in cued sign; zero magnitude is catch',cue_frames=[0,1,2],sample_frames=[1,2],probe_frame=3+delay,blank_frames=list(range(3,3+delay)))
    def batch(self,n,task,condition):
        self.raw=[]
        return super().batch(n,task,condition)


def relocate(x,meta,raw):
    """Identical pre-overlay scenes, cue sign and rotations; next target location.
    Relocation changes requested report, not classical cue-validity benefit.
    """
    xx=x.clone();mm=copy.deepcopy(meta);labels=[]
    for i,m in enumerate(mm):
        old=m['target_location'];target=(old+1)%4;m['original_target']=old;m['target_location']=target
        m['label']=int(m['rotations_degrees'][target]*m['cue_sign']>0);labels.append(m['label'])
        m['trial_id']+='/relocated';m['cue_relocation']=True
        for t in (0,1,2):xx[i,t]=torch.from_numpy(S.local_cue(raw[i][t],'recall','sample',target,m['cue_sign']))
    return xx,torch.tensor(labels),mm
