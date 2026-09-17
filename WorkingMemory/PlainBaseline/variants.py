"""Difficulty ladder inside the orientation family, for locating where learnability breaks.

Each rung keeps the rendering of the real task (same Gabors, same frame layout, same CENTERS, same corner glyphs)
and removes one interaction at a time from the label:

  orientation_single : one Gabor at CENTERS[0], no location cue, no sign glyph. Label 1 iff it rotates counter-clockwise
                       (delta > 0) between sample and probe. First-order: a fixed rotation detector at one place.
  orientation_ring   : four Gabors, target marked by the ring cue (as in binding/motion), no sign glyph. Label 1 iff the
                       cued Gabor rotates counter-clockwise; the other three rotate by independent random signs. Two-way:
                       location x rotation.
  orientation_sign   : one Gabor at CENTERS[0] with the real sign glyph above it, no location choice. Label 1 iff the Gabor
                       rotates in the glyph's sign (exactly the real task's rule at a fixed location). Two-way: sign x rotation.
  orientation_cued   : the real task (sign glyph gives location and the sign to report). Three-way.

Magnitudes are drawn from `magnitudes` (default the real task's 15/30/45). Labels are balanced through the same pending
queue as the real task. Rendering helpers are reused from stimuli.py so the pixel statistics match.
"""
from __future__ import annotations
import numpy as np
from WorkingMemory.SpatialTaskBattery import stimuli as S

VARIANT_TASKS={'orientation_single':2,'orientation_ring':2,'orientation_sign':2}

class VariantStream(S.SpatialBatteryStream):
    def __init__(self,seed,split='train',magnitudes=(15,30,45)):
        super().__init__(seed,split);self.magnitudes=tuple(magnitudes)
        for t in VARIANT_TASKS:
            self.rng[t]=np.random.default_rng(seed+100003*(len(S.FAMILIES)+1+list(VARIANT_TASKS).index(t)));self.pending[t]=[];self.counts[t]=0
    def _case(self,task):
        if task in VARIANT_TASKS:
            if not self.pending[task]:self.pending[task]=self.rng[task].permutation(8).tolist()
            k=self.pending[task].pop();self._cue_sign=1 if k%2 else -1;return k//4,k%4
        return super()._case(task)
    def _gabors_subset(self,rng,angles,keep):
        """Render only the locations in `keep`; the rest are blank gray (same noise field)."""
        field=rng.normal(0,.008,(100,100));wavelength=float(rng.uniform(5,7));amplitude=float(rng.uniform(.28,.36))
        for k,((x,y),theta) in enumerate(zip(S.CENTERS,angles)):
            if k not in keep:continue
            dx,dy=self.xx-x,self.yy-y;r2=dx*dx+dy*dy;envelope=np.exp(-r2/(2*4.5**2))*(r2<=12**2)
            phase=float(rng.uniform(0,2*np.pi));field+=amplitude*envelope*np.cos(2*np.pi*(dx*np.cos(theta)+dy*np.sin(theta))/wavelength+phase)
        return np.repeat(np.clip(.5+field,0,1)[None],3,axis=0).astype(np.float32)
    def _variant(self,task,rng,label,target,cfg):
        angles=rng.uniform(0,np.pi,4);magnitude=int(rng.choice(self.magnitudes));delay=int(cfg.get('delay',0))
        if task in ('orientation_single','orientation_sign'):target=0;keep={0}
        else:keep={0,1,2,3}
        signs=rng.choice([-1,1],4);signs[target]=1 if label else -1
        cue_sign=self._cue_sign if task=='orientation_sign' else 1
        signs[target]*=cue_sign  # for the sign rung the reported direction is relative to the glyph's sign
        delta=signs*magnitude
        probe=(angles+np.deg2rad(delta))%np.pi
        cue=(task=='orientation_ring');glyph=(task=='orientation_sign')
        frames=[S.local_cue(S.blank(),'recall','sample',target,sign=cue_sign if glyph else None,visible=cue or glyph)]
        frames.extend(S.local_cue(self._gabors_subset(rng,angles,keep),'recall','sample',target,sign=cue_sign if glyph else None,visible=cue or glyph) for _ in range(2))
        frames.extend(S.local_cue(S.blank(),'recall','ignore',target,visible=False) for _ in range(delay))
        frames.append(S.local_cue(self._gabors_subset(rng,probe,keep),'recall','report',target,visible=False))
        assert int(delta[target]*cue_sign>0)==label
        return frames,dict(target_location=target,cue_sign=cue_sign,rotations_degrees=delta.tolist(),rotation_magnitude=magnitude,sample_angles_radians=angles.tolist(),probe_angles_radians=probe.tolist(),
            cue_frames=[0,1,2] if (cue or glyph) else [],sample_frames=[1,2],probe_frame=3+delay,blank_frames=list(range(3,3+delay)),label_semantics='1 iff the target Gabor rotates counter-clockwise (positive delta)')
    def batch(self,n,task,condition):
        if task not in VARIANT_TASKS:return super().batch(n,task,condition)
        import copy,torch
        cfg=dict(condition);rng=self.rng[task];rows=[]
        for _ in range(n):
            label,target=self._case(task);frames,meta=self._variant(task,rng,label,target,cfg);ordinal=self.counts[task];self.counts[task]+=1
            meta.update(version=S.VERSION+'+variants',task=task,split=self.split,label=label,condition=copy.deepcopy(cfg),trial_id=f'Variants/{self.split}/{self.seed}/{task}/{ordinal}',frame_count=len(frames))
            rows.append((np.stack(frames),label,meta))
        x=np.stack([r[0] for r in rows]).astype(np.float32)
        return torch.from_numpy(x),torch.tensor([r[1] for r in rows],dtype=torch.long),[r[2] for r in rows]
