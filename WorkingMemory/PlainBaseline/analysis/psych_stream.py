"""Parametrised orientation_cued generator for psychometric sweeps.

Same rendering as the real task (stimuli.py `_orientation`), with the parameters psychophysics sweeps:
  magnitudes            : set of rotation magnitudes in degrees (real task: 15/30/45)
  delay                 : any non-negative integer (real task trains 0/4/12/24)
  cue_scale             : contrast of the sign glyph, 1.0 = as trained (glyph pixels blended toward gray)
  glyph_jitter_px       : random integer offset of the glyph position in x and y
  n_distractors         : number of uncued Gabors rendered (0..3); the real task has 3
  distractor_same_mag   : if True every uncued Gabor rotates by the target's magnitude (random sign); else as the real task

Labels and targets are balanced through the real task's pending queue. The stream seed should differ from the
train/val/test seeds; PSYCH_SEED below is reserved for this purpose.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from WorkingMemory.SpatialTaskBattery import stimuli as S

PSYCH_SEED=65973001

class PsychOrientationStream(S.SpatialBatteryStream):
    def __init__(self,seed=PSYCH_SEED,magnitudes=(15,30,45),cue_scale=1.0,glyph_jitter_px=0,n_distractors=3,distractor_same_mag=False):
        super().__init__(seed,'test');self.magnitudes=tuple(magnitudes);self.cue_scale=float(cue_scale);self.jitter=int(glyph_jitter_px)
        self.n_distractors=int(n_distractors);self.same_mag=bool(distractor_same_mag)
    def _glyph_frame(self,image,target,sign,visible):
        """local_cue with the glyph optionally attenuated and jittered."""
        out=S.visual_cues(image,'recall','sample')
        if not visible:return out
        x,y=S.CENTERS[int(target)];dx=dy=0
        if self.jitter:dx,dy=[int(v) for v in self._jrng.integers(-self.jitter,self.jitter+1,2)]
        glyph=S.visual_cues(S.blank(),'recall','sample',sign=sign)[:,90:100,0:10]
        glyph=.5+self.cue_scale*(glyph-.5)
        x0=int(x)-5+dx;y0=int(y)-23+dy;out[:,y0:y0+10,x0:x0+10]=glyph
        return out
    def _gabors_subset(self,rng,angles,keep):
        field=rng.normal(0,.008,(100,100));wavelength=float(rng.uniform(5,7));amplitude=float(rng.uniform(.28,.36))
        for k,((x,y),theta) in enumerate(zip(S.CENTERS,angles)):
            if k not in keep:continue
            dx,dy=self.xx-x,self.yy-y;r2=dx*dx+dy*dy;envelope=np.exp(-r2/(2*4.5**2))*(r2<=12**2)
            phase=float(rng.uniform(0,2*np.pi));field+=amplitude*envelope*np.cos(2*np.pi*(dx*np.cos(theta)+dy*np.sin(theta))/wavelength+phase)
        return np.repeat(np.clip(.5+field,0,1)[None],3,axis=0).astype(np.float32)
    def _orientation(self,rng,label,target,cfg):
        self._jrng=rng;sign=self._cue_sign;angles=rng.uniform(0,np.pi,4);magnitude=float(rng.choice(self.magnitudes))
        others=[j for j in range(4) if j!=target];keep={target}|set(rng.permutation(others)[:self.n_distractors].tolist())
        if self.same_mag:pattern=np.array([-1,0,1,int(rng.choice([-1,0,1]))]);wanted=1 if label else int(rng.choice([-1,0]));i=int(np.flatnonzero(pattern==wanted)[0]);delta=np.empty(4);delta[target]=pattern[i];delta[others]=rng.choice([-1,1],3);delta=delta*sign*magnitude
        else:
            pattern=np.array([-1,0,1,int(rng.choice([-1,0,1]))]);wanted=1 if label else int(rng.choice([-1,0]));i=int(rng.choice(np.flatnonzero(pattern==wanted)))
            delta=np.empty(4);delta[target]=pattern[i];delta[others]=rng.permutation(np.delete(pattern,i));delta=delta*sign*magnitude
        probe=(angles+np.deg2rad(delta))%np.pi;delay=int(cfg.get('delay',0))
        frames=[self._glyph_frame(S.blank(),target,sign,True)]
        frames.extend(self._glyph_frame(self._gabors_subset(rng,angles,keep),target,sign,True) for _ in range(2))
        frames.extend(S.local_cue(S.blank(),'recall','ignore',target,visible=False) for _ in range(delay))
        frames.append(S.local_cue(self._gabors_subset(rng,probe,keep),'recall','report',target,visible=False))
        assert int(delta[target]*sign>0)==label
        return frames,dict(target_location=target,cue_sign=sign,rotations_degrees=delta.tolist(),rotation_magnitude=magnitude,rendered_locations=sorted(keep),sample_angles_radians=angles.tolist(),probe_angles_radians=probe.tolist(),
            cue_scale=self.cue_scale,glyph_jitter_px=self.jitter,sample_frames=[1,2],probe_frame=3+delay,blank_frames=list(range(3,3+delay)),cue_frames=[0,1,2])
