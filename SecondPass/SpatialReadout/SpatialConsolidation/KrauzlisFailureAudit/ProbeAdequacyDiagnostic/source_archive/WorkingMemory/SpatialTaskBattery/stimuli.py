"""Five spatial/recognition tasks. Only BTCHW images enter the model."""
from __future__ import annotations
import copy,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from WorkingMemory.stimuli import visual_cues,_motion_schedule,duration_oracle
from PreAttentiveVision.neuroscience_stimuli import DIRECTION_VECTORS
from PreAttentiveVision.natural_stimuli import DATA_ROOT

VERSION='spatial_five_task_battery_v1'
TASK_CLASSES={'orientation_cued':2,'motion_duration_cued':4,'krauzlis_cued_motion':2,'spatial_binding':2,'image_recognition':2}
FAMILIES=tuple(TASK_CLASSES)
CENTERS=np.array([[27.,27.],[73.,27.],[27.,73.],[73.,73.]])
DELAYS=(0,4,12,24)
TRAIN_CONDITIONS={task:[dict(delay=d) for d in DELAYS] for task in ('orientation_cued','motion_duration_cued','spatial_binding')}
TRAIN_CONDITIONS['image_recognition']=[dict(load=n,probe_hold=h) for n in (0,4,12,24) for h in (3,4,5)]
# Krauzlis settings are filled below with the disclosed compressed2018 recipe.
TRAIN_CONDITIONS['krauzlis_cued_motion']=[dict(baseline_transitions=e) for e in (12,20,28)]
EVAL_CONDITIONS={f'{task}_D{d}':dict(task=task,condition=dict(delay=d)) for task in ('orientation_cued','motion_duration_cued','spatial_binding') for d in DELAYS}
EVAL_CONDITIONS.update({f'image_recognition_N{n}_H{h}':dict(task='image_recognition',condition=dict(load=n,probe_hold=h)) for n in (0,4,12,24) for h in (3,4,5)})
EVAL_CONDITIONS.update({f'krauzlis_B{e}':dict(task='krauzlis_cued_motion',condition=dict(baseline_transitions=e)) for e in (12,20,28)})
def frame_count(task,condition):
    if task=='orientation_cued':return 4+int(condition.get('delay',0))
    if task=='spatial_binding':return 5+int(condition.get('delay',0))
    if task=='motion_duration_cued':return 11+int(condition.get('delay',0))
    if task=='image_recognition':return 4+int(condition.get('load',4))+int(condition.get('probe_hold',4))
    if task=='krauzlis_cued_motion':return 17+int(condition.get('baseline_transitions',20))
    raise KeyError(task)
condition_frames=frame_count
def raster_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def blank():return np.full((3,100,100),.5,np.float32)
def local_cue(image,protocol,phase,target,sign=None,visible=True):
    out=visual_cues(image,protocol,phase)
    if visible:
        # Translate the existing10x10plus/minus glyph without changing its pixels.
        x,y=CENTERS[int(target)]
        if sign is not None:
            glyph=visual_cues(blank(),protocol,phase,sign=sign)[:,90:100,0:10];x0=int(x)-5;y0=int(y)-23;out[:,y0:y0+10,x0:x0+10]=glyph
        else:
            yy,xx=np.mgrid[:100,:100];ring=np.abs(np.sqrt((xx-x)**2+(yy-y)**2)-14)<.8;out[:,ring]=.95
    return out

class RecognitionImages:
    """Canonical RGB scenes; source photographs are disjoint across official splits."""
    def __init__(self,split):
        self.root=Path(DATA_ROOT);manifest=json.loads((self.root/'manifest.json').read_text());self.records=[r for r in manifest['images'] if r['split']==split];self.cache={}
        assert len(self.records)>=25
        assert len({r['sha256'] for r in manifest['images']})==len(manifest['images'])
        self.manifest_sha256=raster_hash((self.root/'manifest.json').read_bytes())
    def image(self,index):
        if index not in self.cache:
            rec=self.records[index];path=self.root/rec['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
            with Image.open(path) as image:rgb=np.asarray(ImageOps.fit(image.convert('RGB'),(100,100),method=Image.Resampling.BICUBIC),dtype=np.uint8)
            self.cache[index]=(rgb.transpose(2,0,1).astype(np.float32)/255).copy()
        return self.cache[index]

class SpatialBatteryStream:
    def __init__(self,seed,split='train'):
        if split not in ('train','val','test'):raise ValueError(split)
        self.seed,self.split=int(seed),split;self.rng={task:np.random.default_rng(seed+100003*(i+1)) for i,task in enumerate(FAMILIES)};self.pending={k:[] for k in FAMILIES};self.counts={k:0 for k in FAMILIES};self.photos=None
        self.yy,self.xx=np.mgrid[:100,:100]
    def state_dict(self):return dict(version=VERSION,seed=self.seed,split=self.split,rng={k:copy.deepcopy(v.bit_generator.state) for k,v in self.rng.items()},pending=copy.deepcopy(self.pending),counts=copy.deepcopy(self.counts))
    def load_state_dict(self,s):
        if (s['version'],s['seed'],s['split'])!=(VERSION,self.seed,self.split):raise ValueError('Stream identity mismatch')
        for k,v in self.rng.items():v.bit_generator.state=copy.deepcopy(s['rng'][k])
        self.pending=copy.deepcopy(s['pending']);self.counts=copy.deepcopy(s['counts'])
    def _case(self,task):
        if task=='krauzlis_cued_motion':
            if not self.pending[task]:
                event=np.arange(100);side=(event+self.counts[task]//100)%2;self.pending[task]=self.rng[task].permutation(2*event+side).tolist()
            k=self.pending[task].pop();event=k//2;self._event_kind='target' if event<57 else 'foil' if event<86 else 'catch';return int(event<57),k%2
        if task=='orientation_cued':
            if not self.pending[task]:self.pending[task]=self.rng[task].permutation(16).tolist()
            k=self.pending[task].pop();self._cue_sign=1 if k%2 else -1;return k//8,(k//2)%4
        if not self.pending[task]:self.pending[task]=self.rng[task].permutation(4*TASK_CLASSES[task]).tolist()
        k=self.pending[task].pop();return k//4,k%4
    def _gabors(self,rng,angles):
        field=rng.normal(0,.008,(100,100));wavelength=float(rng.uniform(5,7));amplitude=float(rng.uniform(.28,.36))
        for (x,y),theta in zip(CENTERS,angles):
            dx,dy=self.xx-x,self.yy-y;r2=dx*dx+dy*dy;envelope=np.exp(-r2/(2*4.5**2))*(r2<=12**2)
            phase=float(rng.uniform(0,2*np.pi));field+=amplitude*envelope*np.cos(2*np.pi*(dx*np.cos(theta)+dy*np.sin(theta))/wavelength+phase)
        return np.repeat(np.clip(.5+field,0,1)[None],3,axis=0).astype(np.float32)
    def _orientation(self,rng,label,target,cfg):
        sign=self._cue_sign;angles=rng.uniform(0,np.pi,4);magnitude=int(rng.choice([15,30,45]));pattern=np.array([-1,0,1,int(rng.choice([-1,0,1]))])
        wanted=1 if label else int(rng.choice([-1,0]));i=int(rng.choice(np.flatnonzero(pattern==wanted)));delta=np.empty(4,dtype=int);delta[target]=pattern[i];others=[j for j in range(4) if j!=target];delta[others]=rng.permutation(np.delete(pattern,i));delta=delta*sign*magnitude
        probe=(angles+np.deg2rad(delta))%np.pi;delay=int(cfg.get('delay',0));frames=[local_cue(blank(),'recall','sample',target,sign)]
        frames.extend(local_cue(self._gabors(rng,angles),'recall','sample',target,sign) for _ in range(2));frames.extend(local_cue(blank(),'recall','ignore',target,visible=False) for _ in range(delay));frames.append(local_cue(self._gabors(rng,probe),'recall','report',target,visible=False))
        assert int(delta[target]*sign>0)==label
        return frames,dict(target_location=target,cue_sign=sign,positions_xy=CENTERS.tolist(),sample_angles_radians=angles.tolist(),probe_angles_radians=probe.tolist(),rotations_degrees=delta.tolist(),rotation_magnitude=magnitude,rotation_multiset_before_target_assignment=pattern.tolist(),label_semantics='1 iff cued target rotates in the cued sign;0=zero or opposite target rotation',cue_frames=[0,1,2],sample_frames=[1,2],probe_frame=3+delay,blank_frames=list(range(3,3+delay)),foil_control='Same unsigned/sign-relative delta multiset generated before label assignment; all trials contain aligned, opposite and unchanged locations')
    def _binding(self,rng,label,target,cfg):
        inventory=(rng.uniform(0,np.pi)+np.arange(4)*np.pi/4)%np.pi;angles=inventory[rng.permutation(4)];others=[i for i in range(4) if i!=target];pair=[target,int(rng.choice(others))] if label else rng.choice(others,2,replace=False).tolist();probe=angles.copy();probe[pair]=probe[pair[::-1]];delay=int(cfg.get('delay',0));frames=[local_cue(blank(),'recall','sample',target,visible=False)]
        frames.extend(local_cue(self._gabors(rng,angles),'recall','sample',target,visible=False) for _ in range(2));frames.extend(local_cue(blank(),'recall','ignore',target,visible=False) for _ in range(delay));frames.append(local_cue(blank(),'recall','query',target));frames.append(local_cue(self._gabors(rng,probe),'recall','report',target,visible=False))
        assert np.allclose(np.sort(angles),np.sort(probe)) and int(probe[target]!=angles[target])==label
        return frames,dict(target_location=target,positions_xy=CENTERS.tolist(),sample_angles_radians=angles.tolist(),probe_angles_radians=probe.tolist(),swapped_locations=pair,changed_location_count=2,inventory_preserved=True,label_semantics='1=target item exchanged with another location;0=foil-only exchange,target preserved',cue_frames=[3+delay],cue_timing='retrocue only after retention delay; allfour locations initially eligible',sample_frames=[1,2],probe_frame=4+delay,blank_frames=list(range(3,3+delay)))
    def _disk_positions(self,rng,n,radius):
        angle=rng.uniform(0,2*np.pi,n);r=radius*np.sqrt(rng.uniform(0,1,n));return np.c_[r*np.cos(angle),r*np.sin(angle)]
    def _dots_raster(self,positions,centers,contrast=0.42):
        image=blank()
        for p,c in zip(positions,centers):
            xy=np.rint(p+c).astype(int);valid=(xy[:,0]>=0)&(xy[:,0]<100)&(xy[:,1]>=0)&(xy[:,1]<100);xy=xy[valid];image[:,xy[:,1],xy[:,0]]=.5+contrast
        return image
    def _motion(self,rng,label,target,cfg):
        length=8;winners=rng.integers(0,4,4);winners[target]=label;directions=np.stack([_motion_schedule(rng,length,int(k)) for k in winners]);positions=[self._disk_positions(rng,32,11.5) for _ in range(4)];delay=int(cfg.get('delay',0));frames=[local_cue(blank(),'integration','sample',target)];frames.append(local_cue(self._dots_raster(positions,CENTERS),'integration','sample',target));step=float(rng.choice([.8,1.2,1.6]))
        for t in range(length):
            for k in range(4):
                positions[k]+=step*DIRECTION_VECTORS[directions[k,t]];reset=np.linalg.norm(positions[k],axis=1)>11.5;reset[rng.choice(32,16,replace=False)]=True;positions[k][reset]=self._disk_positions(rng,int(reset.sum()),11.5)
            frames.append(local_cue(self._dots_raster(positions,CENTERS),'integration','sample',target))
        frames.extend(local_cue(blank(),'integration','ignore',target,visible=False) for _ in range(delay));frames.append(local_cue(blank(),'integration','report',target,visible=False));counts=np.stack([np.bincount(d,minlength=4) for d in directions]);assert duration_oracle(directions[target])[0]==label
        return frames,dict(target_location=target,positions_xy=CENTERS.tolist(),directions_by_patch=directions.tolist(),duration_counts_by_patch=counts.tolist(),winner_by_patch=winners.tolist(),motion_transitions=8,dots_per_patch=32,aperture_radius_pixels=11.5,step_pixels=step,randomly_replaced_dots_per_transition=16,cue_frames=list(range(10)),reference_frame=1,moving_frames=list(range(2,10)),blank_frames=list(range(10,10+delay)),report_frame=10+delay,label_semantics='right/up/left/down=0/1/2/3 of the longest-duration direction in the cued patch',foil_control='Each uncued patch has independently drawn winner and direction schedule')
    def _recognition(self,rng,label,target,cfg):
        if self.photos is None:self.photos=RecognitionImages(self.split)
        load=int(cfg.get('load',4));hold=int(cfg.get('probe_hold',4))
        if load not in (0,4,12,24) or hold not in (3,4,5):raise ValueError(cfg)
        if load==0:label=0
        ids=rng.choice(len(self.photos.records),load+1,replace=False);study=[self.photos.image(int(k)).copy() for k in ids[:load]];seen_index=int(rng.integers(load)) if label else None;probe=study[seen_index].copy() if label else self.photos.image(int(ids[-1])).copy();hashes=[raster_hash(x) for x in study];query_hash=raster_hash(probe)
        assert len(set(hashes))==load and (query_hash in hashes)==bool(label)
        frames=[visual_cues(blank(),'recall','sample')]+study+[blank() for _ in range(3)]+[probe.copy() for _ in range(hold)]
        return frames,dict(label=label,study_load=load,probe_hold=hold,study_frames=list(range(1,1+load)),blank_frames=list(range(1+load,4+load)),probe_frames=list(range(4+load,4+load+hold)),study_raster_sha256=hashes,probe_raster_sha256=query_hash,study_source_ids=[self.photos.records[int(k)]['base_id'] for k in ids[:load]],probe_source_id=self.photos.records[int(ids[seen_index] if label else ids[-1])]['base_id'],seen_study_index=seen_index,bit_identical_positive=bool(label),unique_study_scenes=True,source_split=self.split,dataset_manifest_sha256=self.photos.manifest_sha256,label_semantics='1=probe bit-identical to a study raster;0=probe distinct from every study raster;load0 is always negative',raster_adaptation='Canonical central square crop and bicubic downsample to100x100RGB; no glyph overlays on any study/probe frame')
    def _krauzlis(self,rng,label,target,cfg):
        baseline=int(cfg.get('baseline_transitions',20))
        if baseline not in (12,20,28):raise ValueError(cfg)
        centers=np.array([[20.,50.],[80.,50.]]);radius=8.125;positions=[self._disk_positions(rng,16,radius) for _ in range(2)];ages=[rng.permutation(np.arange(16)%10) for _ in range(2)];offsets=[rng.normal(0,np.deg2rad(16),16) for _ in range(2)];theta=float(rng.uniform(0,2*np.pi));means=np.array([theta,theta+np.pi/2]);original=means.copy();sign=int(rng.choice([-1,1]));magnitude=int(rng.choice([26,28]));event=self._event_kind;changed=target if event=='target' else 1-target if event=='foil' else None
        def fixation():
            image=blank();image[:,48:52,49:51]=.1;image[:,49:51,48:52]=.1;return image
        def image():
            field=fixation();density=np.zeros((100,100),np.float32)
            for p,c in zip(positions,centers):
                xy=p+c;lo=np.floor(xy).astype(int);fraction=xy-lo
                for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
                    weight=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1]);np.add.at(density,(lo[:,1]+dy,lo[:,0]+dx),weight)
            return np.clip(field+.48*density[None],0,1).astype(np.float32)
        cue=fixation();cx,cy=centers[target];ring=np.abs(np.sqrt((self.xx-cx)**2+(self.yy-cy)**2)-(radius+1.8))<.7;cue[:,ring]=.95
        frames=[cue.copy() for _ in range(2)]+[fixation() for _ in range(5)]+[image()]
        for t in range(baseline+8):
            if t==baseline and changed is not None:means[changed]+=np.deg2rad(sign*magnitude)
            for k in range(2):
                directions=means[k]+offsets[k];positions[k]+=.375*np.c_[np.cos(directions),np.sin(directions)];ages[k]+=1;reset=(ages[k]>=10)|(np.linalg.norm(positions[k],axis=1)>radius);n=int(reset.sum());positions[k][reset]=self._disk_positions(rng,n,radius);offsets[k][reset]=rng.normal(0,np.deg2rad(16),n);ages[k][reset]=0
            frames.append(image())
        frames.append(fixation());assert label==int(event=='target')
        return frames,dict(target_location=target,positions_xy=centers.tolist(),event_type=event,changed_patch=changed,baseline_transitions=baseline,postevent_transitions=8,baseline_means_degrees=np.rad2deg(original).tolist(),postevent_means_degrees=np.rad2deg(means).tolist(),signed_change_degrees=sign*magnitude if changed is not None else 0,event_sign=sign,event_magnitude_degrees=magnitude,cue_frames=[0,1],fixation_only_frames=list(range(2,7)),reference_frame=7,first_postchange_frame=8+baseline,virtual_event_frame=8+baseline,report_frame=16+baseline,frame_clock_hz=100,dots_per_patch=16,dot_direction_SD_degrees=16,dot_lifetime_frames=10,dot_speed_degrees_per_second=15,pixels_per_degree=2.5,step_pixels=.375,aperture_radius_pixels=radius,aperture_radius_degrees=3.25,eccentricity_degrees=12,event_proportions_exact_per_100_trials=dict(target=.57,foil=.29,catch=.14),rendering='Subpixel bilinear white dots; boundary/lifetime resets; angular offsets persist across event unless dot resets',label_semantics='1=change in cued target;0=foil-only change or catch; end-of-trial report, no reaction-time claim',source='Arcizet and Krauzlis2018,doi:10.1371/journal.pbio.2005930',adaptations='100x100 raster,16dots/patch; cue20ms and gap50ms,baseline120/200/280ms andpost80ms compressed from primary timing; target/foil means90deg apart,Gaussian16deg,lifetime100ms,speed15deg/s retained')
    def batch(self,n,task,condition):
        if task not in TASK_CLASSES or n<=0:raise ValueError((n,task))
        cfg=dict(condition);rng=self.rng[task];rows=[]
        for _ in range(n):
            label,target=self._case(task);fn={'orientation_cued':self._orientation,'spatial_binding':self._binding,'motion_duration_cued':self._motion,'image_recognition':self._recognition,'krauzlis_cued_motion':self._krauzlis}[task];frames,meta=fn(rng,label,target,cfg);label=int(meta.get('label',label));ordinal=self.counts[task];self.counts[task]+=1
            meta.update(version=VERSION,task=task,split=self.split,label=label,condition=copy.deepcopy(cfg),trial_id=f'SpatialBattery/{self.split}/{self.seed}/{task}/{ordinal}',frame_count=len(frames),independent_base_episode=True);rows.append((np.stack(frames),label,meta))
        x=np.stack([r[0] for r in rows]).astype(np.float32);assert x.shape==(n,frame_count(task,cfg),3,100,100) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
        return torch.from_numpy(x),torch.tensor([r[1] for r in rows],dtype=torch.long),[r[2] for r in rows]
