"""Isolated analysis adapters: native nuisance draws preserved; only named strengths vary."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PreAttentiveVision.neuroscience_stimuli import TaskStream, CardinalMotionStream, DIRECTION_VECTORS, DIRECTION_NAMES, MOTION_PROTOCOL
from PreAttentiveVision.natural_stimuli import NaturalSpectrum, PROTOCOL
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream, CENTERS, blank, local_cue


def sensory_orientation(self, label, level):
    degrees = self.analysis_strength
    theta = float(self.rng.uniform(0, np.pi))
    signed = np.deg2rad(degrees) * (1 if label else -1)
    frequency = float(self.rng.uniform(5, 10))
    contrast = float(self.rng.uniform(.2, .38))
    phases = self.rng.uniform(0, 2*np.pi, 2)
    fields = [contrast*self._grating(theta,frequency,phases[0]),
              contrast*self._grating(theta+signed,frequency,phases[1])]
    return np.stack([self._achromatic(f) for f in fields]), {
        'signed_orientation_degrees': degrees*(1 if label else -1),
        'base_orientation_degrees': float(np.rad2deg(theta)),
        'frequency_cycles_per_image': frequency, 'contrast':contrast,
        'difficulty': f'delta_{int(degrees)}deg',
    }


def sensory_contrast(self, label, level):
    increment = self.analysis_strength
    pedestal = float(self.rng.choice([.08, .18, .3]))
    frequency = float(self.rng.uniform(5,10))
    theta, phase = self.rng.uniform(0,np.pi), self.rng.uniform(0,2*np.pi)
    pattern = self._grating(theta, frequency, phase)
    contrasts = [pedestal,pedestal]
    contrasts[label] += increment
    return np.stack([self._achromatic(c*pattern) for c in contrasts]), {
        'contrast_increment':increment, 'pedestal':pedestal,
        'frame_contrasts':contrasts, 'frequency_cycles_per_image':frequency,
        'difficulty':f'increment_{increment:g}_pedestal_{pedestal:g}',
    }


def sensory_spatial_frequency(self, label, level):
    octave = self.analysis_strength
    frequency = float(self.rng.uniform(4,9))
    frequencies = [frequency,frequency]
    frequencies[label] *= 2**octave
    theta = self.rng.uniform(0,np.pi)
    contrast = self.rng.uniform(.2,.38)
    phases = self.rng.uniform(0,2*np.pi,2)
    return np.stack([self._achromatic(contrast*self._grating(theta,f,phase))
                     for f,phase in zip(frequencies,phases)]), {
        'frequency_octave_increment':octave, 'frame_frequencies':frequencies,
        'difficulty':f'octave_{octave:g}',
    }


def sensory_chromatic_increment(self, label, level):
    w = np.array([.2126,.7152,.0722], np.float32)
    axis = np.array([w[1],-w[0],0],np.float32)
    axis /= np.linalg.norm(axis)
    increment = self.analysis_strength
    base = self.rng.uniform(.3,.65,3).astype(np.float32)
    colors = [base.copy(),base.copy()]
    colors[label] += increment*axis
    # Soft equal-geometry patch; linearRGB, no gamma or clamping.
    radius = self.rng.uniform(17,24)
    distance = np.sqrt(self.xx**2+self.yy**2)
    mask = np.clip((radius+.5-distance)/1.,0,1)
    result = []
    for color in colors:
        image = .5 + mask[:,:,None]*(color-.5)[None,None,:]
        image += self.rng.uniform(-.001,.001,image.shape)
        result.append(image.transpose(2,0,1).astype(np.float32))
    return np.stack(result), {
        'chromatic_increment':increment, 'axis_linear_rgb':axis.tolist(),
        'frame_colors': [c.tolist() for c in colors],
        'frame_luminances': [float(w@c) for c in colors],
        'difficulty':f'increment_{increment:g}',
    }


def sensory_contour(self, label, level):
    jitter = self.analysis_strength
    bend = self.rng.uniform(-9,9)
    t = np.linspace(-1,1,7)
    path = np.stack([31*t,bend*t*t],axis=1)
    angle = self.rng.uniform(0,2*np.pi)
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center = 49.5+self.rng.uniform(-3,3,2)
    path = path@rotation.T+center
    positions = [p for p in path]
    for _ in range(2000):
        if len(positions) == 32: break
        p = self.rng.uniform(8,92,2)
        if np.min(np.linalg.norm(np.array(positions)-p,axis=1)) >= 10:
            positions.append(p)
    if len(positions) != 32:
        raise RuntimeError('Unable to place contour clutter')
    positions = np.rint(positions).astype(int)
    tangent = np.arctan2(2*bend*t,31)+angle
    orientations = self.rng.uniform(0,np.pi,32)
    orientations[:7] = tangent + np.pi/2 + np.deg2rad(self.rng.normal(0,jitter,7))
    permutation = self.rng.permutation(32)
    scrambled = orientations[permutation]
    phases = self.rng.uniform(0,2*np.pi,32)
    def render(angles):
        field = np.zeros((100,100),np.float32)
        for (x,y),theta,phase in zip(positions,angles,phases):
            carrier = np.cos(2*np.pi*(self.gx*np.cos(theta)+self.gy*np.sin(theta))/5+phase)
            field[y-5:y+6,x-5:x+6] += .34*self.small_envelope*carrier
        return self._achromatic(field)
    structured,control = render(orientations),render(scrambled)
    result = [control,control]
    result[label] = structured
    return np.stack(result), {
        'alignment_jitter_degrees':jitter, 'path_bend_pixels':float(bend),
        'path_elements':7, 'total_elements':32, 'orientation_multiset_matched':True,
        'difficulty':f'jitter_{int(jitter)}deg',
    }


def motion_batch(self, n, task='motion_direction', displacement=None):
    if task != 'motion_direction':
        raise ValueError(task)
    if displacement is not None and not 0 <= float(displacement) <= 3:
        raise ValueError('Version-2 displacement must be 1, 2 or 3 pixels')
    labels = (np.arange(n, dtype=np.int64) + int(self.rng.integers(4))) % 4
    self.rng.shuffle(labels)
    result = np.empty((n, 2, 3, 100, 100), np.float32)
    metadata = []
    for i, label in enumerate(labels):
        step = float(displacement if displacement is not None else self.rng.choice([1., 2., 3.]))
        before = self.rng.uniform(0, 100, (256, 2)).astype(np.float32)
        ages = np.zeros(256, np.int64)
        ages[self.rng.permutation(256)[:128]] = 1
        survivor = ages == 0
        after = (before + step * DIRECTION_VECTORS[label]) % 100
        after[~survivor] = self.rng.uniform(0, 100, (128, 2))
        result[i, 0], result[i, 1] = self._render(before), self._render(after)
        metadata.append({
            'task': 'motion_direction', 'family': 'motion_direction', 'protocol': MOTION_PROTOCOL['version'],
            'direction': DIRECTION_NAMES[label], 'label': int(label),
            'displacement_pixels': step, 'difficulty': f'displacement_{int(step)}px',
            'coherence': 1., 'coherence_definition': 'fraction of surviving dots following the selected direction',
            'survivor_count_domain': 128, 'reborn_count_domain': 128,
            'base_id': None, 'split': self.split,
            'trial_id': f'{self.split}/{self.seed}/{self.counter}',
        })
        self.counter += 1
    return torch.from_numpy(result), torch.from_numpy(labels), metadata


def spectrum_sample(self, rng, label):
    if label not in (0, 1):
        raise ValueError("Natural spectrum label must be target interval0/1")
    for attempt in range(32):
        index = int(rng.integers(len(self.records)))
        image = self._image(index)
        row = int(rng.integers(image.shape[0] - 99))
        col = int(rng.integers(image.shape[1] - 99))
        crop = image[row:row + 100, col:col + 100].astype(np.float64)
        if crop.std() > 1e-4:
            break
    else:
        raise RuntimeError("32 native crops lacked nonzero contrast")
    rotations = int(rng.integers(4))
    reflected = bool(rng.integers(2))
    crop = np.rot90(crop, rotations)
    if reflected:
        crop = crop[:, ::-1]
    center = float(rng.uniform(-.5, .5))
    delta = float(rng.choice([.15, .30, .60])); delta = getattr(self, "analysis_strength", delta)
    betas = [center + delta / 2, center + delta / 2]
    betas[label] = center - delta / 2
    spectrum = np.fft.fft2(crop - crop.mean())
    spectrum[0, 0] = 0
    standardized = []
    for beta in betas:
        modified = spectrum * np.exp(-beta * self.log_frequency)
        modified[0, 0] = 0
        raster = np.fft.ifft2(modified).real
        raster -= raster.mean()
        std = raster.std()
        if std < 1e-12:
            raise ValueError("Degenerate filtered crop")
        standardized.append(raster / std)
    z = np.stack(standardized)
    rms = min(.15, .45 / np.abs(z).max())
    pair = (.5 + rms * z).astype(np.float32)
    frames = np.repeat(pair[:, None], 3, axis=1)
    metadata = {"task": "natural_spectrum", "family": "natural_spectrum", "protocol": PROTOCOL["version"],
                "label": int(label), "difficulty": f"beta_delta_{delta:.2f}",
                "beta_delta": delta, "betas_by_frame": betas, "target_mean": .5,
                "actual_common_rms": float(rms), "base_id": self.records[index]["base_id"],
                "split": self.split, "crop_xywh": [col, row, 100, 100],
                "quarter_turns": rotations, "horizontal_reflection": reflected,
                "source_contrast_rejections": attempt, "clipping": False}
    return frames, metadata


def cued_orientation(self,rng,label,target,cfg):
    sign=self._cue_sign;angles=rng.uniform(0,np.pi,4);magnitude=int(rng.choice([15,30,45]));magnitude=int(cfg.get("analysis_angle",magnitude));pattern=np.array([-1,0,1,int(rng.choice([-1,0,1]))])
    wanted=1 if label else int(rng.choice([-1,0]));i=int(rng.choice(np.flatnonzero(pattern==wanted)));delta=np.empty(4,dtype=int);delta[target]=pattern[i];others=[j for j in range(4) if j!=target];delta[others]=rng.permutation(np.delete(pattern,i));delta=delta*sign*magnitude
    probe=(angles+np.deg2rad(delta))%np.pi;delay=int(cfg.get('delay',0));frames=[local_cue(blank(),'recall','sample',target,sign)]
    frames.extend(local_cue(self._gabors(rng,angles),'recall','sample',target,sign) for _ in range(2));frames.extend(local_cue(blank(),'recall','ignore',target,visible=False) for _ in range(delay));frames.append(local_cue(self._gabors(rng,probe),'recall','report',target,visible=False))
    assert magnitude==0 or int(delta[target]*sign>0)==label
    return frames,dict(target_location=target,cue_sign=sign,positions_xy=CENTERS.tolist(),sample_angles_radians=angles.tolist(),probe_angles_radians=probe.tolist(),rotations_degrees=delta.tolist(),rotation_magnitude=magnitude,rotation_multiset_before_target_assignment=pattern.tolist(),label_semantics='1 iff cued target rotates in the cued sign;0=zero or opposite target rotation',cue_frames=[0,1,2],sample_frames=[1,2],probe_frame=3+delay,blank_frames=list(range(3,3+delay)),foil_control='Same unsigned/sign-relative delta multiset generated before label assignment; all trials contain aligned, opposite and unchanged locations')


import numpy as np
from WorkingMemory.SpatialTaskBattery.stimuli import blank

def krauzlis(self,rng,label,target,cfg):
    baseline=int(cfg.get('baseline_transitions',20))
    if baseline not in (12,20,28):raise ValueError(cfg)
    centers=np.array([[20.,50.],[80.,50.]]);radius=8.125;positions=[self._disk_positions(rng,16,radius) for _ in range(2)];ages=[rng.permutation(np.arange(16)%10) for _ in range(2)];offsets=[rng.normal(0,np.deg2rad(16),16) for _ in range(2)];theta=float(rng.uniform(0,2*np.pi));means=np.array([theta,theta+np.pi/2]);original=means.copy();sign=int(rng.choice([-1,1]));magnitude=int(rng.choice([26,28]));magnitude=int(cfg.get('analysis_angle',magnitude));sign=int(cfg.get('analysis_sign',sign));event=self._event_kind;changed=target if event=='target' else 1-target if event=='foil' else None
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
