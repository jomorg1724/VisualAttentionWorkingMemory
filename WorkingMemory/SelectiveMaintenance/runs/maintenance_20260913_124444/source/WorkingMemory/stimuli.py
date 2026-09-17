"""Fresh, task-local visual sequences for the declared opponent memory battery.

Only returned image tensors enter the learner. Metadata supplies labels/oracles
and exposure accounting, never hidden instructions or state-update switches.
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from PreAttentiveVision.neuroscience_stimuli import TaskStream, TASK_CLASSES, DIRECTION_VECTORS

VERSION = 'wm_visual_sequences_v1'
FAMILIES = tuple(TASK_CLASSES)
BASE = dict(protocol='integration',length=8,delay=0,load=1,cue_visible=True,
            distractor=False,precue=False,post_delay=2,threshold_level=None)


def setting(**kwargs):
    return dict(BASE,**kwargs)


BRIDGE_CONDITIONS = {
    'anchor': setting(protocol='anchor'),
    'bridge_integrate_visible': setting(length=2),
    'bridge_integrate_transient': setting(length=2,cue_visible=False),
    'bridge_recall': setting(protocol='recall',delay=0,post_delay=0),
}
TRAIN_CONDITIONS = {
    **{f'integrate_L{length}':setting(length=length) for length in (8,16,32)},
    **{f'decision_D{delay}':setting(delay=delay) for delay in (2,4,8,16)},
    'integrate_transient':setting(cue_visible=False),
    'decision_distractor':setting(delay=8,distractor=True),
    **{f'recall_N{load}':setting(protocol='recall',load=load,delay=4) for load in (1,2,4)},
    **{f'recall_D{delay}':setting(protocol='recall',delay=delay) for delay in (0,2,8,16)},
    'recall_distractor':setting(protocol='recall',load=2,delay=8,distractor=True),
    'recall_precue':setting(protocol='recall',load=2,delay=4,precue=True),
    'recall_postcue':setting(protocol='recall',load=2,delay=4,post_delay=0),
    'recall_matched_one':setting(protocol='recall',load=1,delay=4,sample_slots=4),
}
DIAGNOSTIC_CONDITIONS = {**TRAIN_CONDITIONS,
    'extra_integrate_L64':setting(length=64),
    'extra_decision_D32':setting(delay=32),
    'extra_recall_N6':setting(protocol='recall',load=6,delay=4),
    'extra_recall_D32':setting(protocol='recall',delay=32),
}
CONDITIONS = {**BRIDGE_CONDITIONS,**DIAGNOSTIC_CONDITIONS}
CUE_CODES = dict(anchor=1,integration=2,recall=3,sample=4,ignore=5,query=6,report=7)


def condition_frames(condition,family):
    cfg=CONDITIONS[condition] if isinstance(condition,str) else dict(BASE,**condition)
    if cfg['protocol']=='anchor':return 2
    if cfg['protocol']=='integration':return int(cfg['length'])+(2 if family=='contour' else 3)+int(cfg['delay'])
    return 3+2*int(cfg.get('sample_slots',cfg['load']))+int(cfg['delay'])+int(cfg['post_delay'])


def visual_cues(image,protocol,phase='sample',sign=None,threshold=None,identity=None):
    """Four identical reserved 10x10 masks; learned glyph semantics are pixels."""
    image=image.copy()
    for y,x in ((0,0),(0,90),(90,0),(90,90)):
        image[:,y:y+10,x:x+10]=.5
    def glyph(code,y,x):
        # Distinct little binary pictograms, not numerical model inputs.
        for bit in range(3):
            if (int(code)>>bit)&1:
                image[:,y+2:y+4,x+1+3*bit:x+3+3*bit]=.95
        image[:,y+6:y+8,x+2:x+8]=.12
    glyph(CUE_CODES[protocol],0,0)
    glyph(CUE_CODES.get(phase,4),0,90)
    if sign is not None:
        image[:,92:94,2:8]=.1
        if sign>0:image[:,90:96,4:6]=.1
    if threshold is not None:
        # Absolute level-unit threshold, not an implicit length-dependent
        # fraction. Fixed monotone logarithmic bar mapping spans 0..40 units;
        # fractional terminal pixels preserve distinct trained thresholds.
        width=8*math.log1p(float(threshold))/math.log(41)
        for j in range(8):image[:,97:99,91+j]=.5-.4*np.clip(width-j,0,1)
    if identity is not None:
        # All six glyphs are trained; query and samples share this mapping.
        glyph(int(identity)+1,90,90)
    return image


def duration_oracle(directions):
    counts=np.bincount(directions,minlength=4)
    if (counts==counts.max()).sum()!=1:raise ValueError('Motion duration tie')
    return int(counts.argmax()),counts.tolist()


def scalar_oracle(levels,sign,threshold):
    evidence=float(np.maximum(0,sign*np.diff(levels)).sum())
    if evidence==threshold:raise ValueError('Threshold equality')
    return int(evidence>threshold),evidence


def _motion_schedule(rng,length,label,final_direction=None):
    if length==2:
        return np.full(length,label,dtype=np.int64)
    first=int(rng.integers(4));last=int(rng.integers(4)) if final_direction is None else int(final_direction)
    margin=max(1,length//8)
    for _ in range(2048):
        sequence=rng.integers(0,4,length)
        sequence[0],sequence[-1]=first,last
        counts=np.bincount(sequence,minlength=4)
        if counts[label]-max(counts[k] for k in range(4) if k!=label)>=margin:
            return sequence
    # Exact feasible count construction preserves the independently drawn ends.
    middle=np.full(length-2,label,dtype=np.int64)
    return np.concatenate(([first],middle,[last]))


def _walk(rng,length,label,sign,threshold):
    for _ in range(2048):
        current=int(rng.integers(2,7));levels=[current]
        for _ in range(length):
            choices=[d for d in (-1,0,1) if 0<=current+d<=8]
            current+=int(rng.choice(choices));levels.append(current)
        observed,evidence=scalar_oracle(levels,sign,threshold)
        if observed==label:return np.asarray(levels),evidence
    raise RuntimeError('Unable to sample feasible balanced scalar trajectory')


class SequenceStream:
    def __init__(self,seed,split='train'):
        self.seed,self.split=int(seed),split
        self.streams={family:TaskStream(self.seed+100003*(i+1),split) for i,family in enumerate(FAMILIES)}
        self.counters={family:0 for family in FAMILIES}

    def state_dict(self):
        return dict(version=VERSION,seed=self.seed,split=self.split,counters=copy.deepcopy(self.counters),
                    streams={family:stream.state_dict() for family,stream in self.streams.items()})

    def load_state_dict(self,state):
        if (state['version'],state['seed'],state['split'])!=(VERSION,self.seed,self.split):
            raise ValueError('Sequence stream identity mismatch')
        for family in FAMILIES:self.streams[family].load_state_dict(state['streams'][family])
        self.counters=copy.deepcopy(state['counters'])

    def _context(self,stream,family):
        rng=stream.rng
        result=dict(theta=float(rng.uniform(0,np.pi)),phase=float(rng.uniform(0,2*np.pi)),
                    frequency=float(rng.uniform(5,9)),amplitude=float(rng.uniform(.22,.34)),
                    radius=float(rng.uniform(17,24)),base_id=None)
        if family=='natural_spectrum':
            natural=stream._get_natural()
            for _ in range(32):
                index=int(rng.integers(len(natural.records)));photo=natural._image(index)
                row=int(rng.integers(photo.shape[0]-99));col=int(rng.integers(photo.shape[1]-99))
                crop=photo[row:row+100,col:col+100].astype(np.float64)
                if crop.std()>1e-4:break
            else:raise ValueError('Degenerate natural crop')
            crop=np.rot90(crop,int(rng.integers(4)))
            if rng.integers(2):crop=crop[:,::-1]
            spectrum=np.fft.fft2(crop-crop.mean());spectrum[0,0]=0
            candidates=[]
            for beta in np.linspace(.6,-.6,9):
                raster=np.fft.ifft2(spectrum*np.exp(-beta*natural.log_frequency)).real
                raster-=raster.mean();raster/=raster.std()
                candidates.append(raster)
            # Full fixed candidate bank determines gain BEFORE any label,
            # trajectory, target item or eventual probe is selected.
            bank=np.stack(candidates);gain=min(.15,.45/np.abs(bank).max())
            result.update(bank=(.5+gain*bank).astype(np.float32),natural_gain=float(gain),
                          base_id=natural.records[index]['base_id'],crop_xy=[col,row])
        return result

    def _contour(self,stream,level,context,organized=True):
        rng=stream.rng;t=np.linspace(-1,1,7);bend=-9+2.25*float(level)
        angle=context['theta'];rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        path=np.stack((31*t,bend*t*t),1)@rotation.T+49.5
        positions=list(path)
        for _ in range(3000):
            if len(positions)==32:break
            p=rng.uniform(8,92,2)
            if np.linalg.norm(np.asarray(positions)-p,axis=1).min()>=10:positions.append(p)
        if len(positions)!=32:raise RuntimeError('Contour clutter placement failed')
        positions=np.rint(positions).astype(int)
        angles=rng.uniform(0,np.pi,32)
        angles[:7]=np.arctan2(2*bend*t,31)+angle+np.pi/2+np.deg2rad(rng.normal(0,2,7))
        if not organized:angles=angles[rng.permutation(32)]
        phases=rng.uniform(0,2*np.pi,32);field=np.zeros((100,100),np.float32)
        for (x,y),theta,phase in zip(positions,angles,phases):
            carrier=np.cos(2*np.pi*(stream.gx*np.cos(theta)+stream.gy*np.sin(theta))/5+phase)
            field[y-5:y+6,x-5:x+6]+=.34*stream.small_envelope*carrier
        return stream._achromatic(field)

    def _render(self,stream,family,level,context,organized=True):
        level=int(level);rng=stream.rng
        if family=='natural_spectrum':return np.repeat(context['bank'][level][None],3,0)
        if family=='contour':return self._contour(stream,level,context,organized)
        if family=='chromatic_increment':
            w=np.array([.2126,.7152,.0722]);axis=np.array([w[1],-w[0],0]);axis/=np.linalg.norm(axis)
            coordinate=-.12+.03*level;color=.5+coordinate*axis
            mask=np.clip(context['radius']+.5-np.sqrt(stream.xx**2+stream.yy**2),0,1)
            return (.5+mask[None]*(color-.5)[:,None,None]).astype(np.float32)
        theta=context['theta'];frequency=context['frequency'];amplitude=context['amplitude']
        if family=='orientation':theta+=np.deg2rad(7.5*level)
        elif family=='contrast':amplitude=.08*math.exp(math.log(4.5)*level/8)
        elif family=='spatial_frequency':frequency=4*2**(math.log2(2.5)*level/8)
        phase=float(rng.uniform(0,2*np.pi)) if family in ('orientation','spatial_frequency') else context['phase']
        return stream._achromatic(amplitude*stream._grating(theta,frequency,phase))

    def _motion_frames(self,stream,directions):
        rng=stream.rng;positions=rng.uniform(0,100,(256,2)).astype(np.float32)
        ages=np.zeros(256,np.int64);ages[rng.permutation(256)[:128]]=1
        step=float(rng.choice([1.,2.,3.]));frames=[stream._render(positions)]
        for direction in directions:
            survive=ages==0
            positions[survive]=(positions[survive]+step*DIRECTION_VECTORS[int(direction)])%100
            positions[~survive]=rng.uniform(0,100,(128,2))
            ages[survive]=1;ages[~survive]=0
            frames.append(stream._render(positions))
        return frames,step

    def _delay(self,stream,family,count,condition,protocol):
        frames=[]
        context=self._context(stream,family) if condition['distractor'] and count else None
        for _ in range(count):
            if condition['distractor']:
                if family=='motion_direction':
                    image=stream._render(stream.rng.uniform(0,100,(256,2)))
                else:image=self._render(stream,family,int(stream.rng.integers(9)),context)
                phase='ignore'
            else:image=np.full((3,100,100),.5,np.float32);phase='ignore'
            frames.append(visual_cues(image,protocol,phase))
        return frames

    @staticmethod
    def _with_seed(stream,seed,renderer,*args):
        """Independent nuisance draw per sample/probe, irrespective of the
        rejection counts used to place a particular contour's clutter."""
        saved=stream.rng
        stream.rng=np.random.default_rng(int(seed))
        try:return renderer(stream,*args)
        finally:stream.rng=saved

    def _integration(self,stream,family,label,condition,sign,threshold_level):
        rng=stream.rng;length=int(condition['length']);context=self._context(stream,family)
        threshold=(.30 if threshold_level==0 else .45)*length
        if family=='motion_direction':
            directions=np.asarray(condition['directions'],dtype=np.int64) if 'directions' in condition else _motion_schedule(rng,length,label,condition.get('final_direction'))
            evidence,step=self._motion_frames(stream,directions)
            actual,counts=duration_oracle(directions);assert actual==label
            sorted_counts=sorted(counts)
            oracle=dict(directions=directions.tolist(),duration_counts=counts,
                        count_margin=(sorted_counts[-1]-sorted_counts[-2])/length,
                        displacement_pixels=step,final_direction=int(directions[-1]),
                        initial_direction=int(directions[0]),
                        net_displacement=(DIRECTION_VECTORS[directions].sum(0)*step).tolist(),
                        cohort_survivors_per_transition=128)
        elif family=='contour':
            threshold=(.375 if threshold_level==0 else .625)*length
            valid=[k for k in range(length+1) if (k>threshold)==bool(label) and k!=threshold]
            count=int(rng.choice(valid));states=np.full(length,-sign,np.int64);states[:count]=sign;rng.shuffle(states)
            level=int(rng.integers(9))
            evidence=[self._render(stream,family,level,context,organized=(s==1)) for s in states]
            oracle=dict(occupancy_signs=states.tolist(),evidence=float(count),threshold=float(threshold))
        else:
            levels,total=_walk(rng,length,label,sign,threshold)
            evidence=[self._render(stream,family,level,context) for level in levels]
            oracle=dict(levels=levels.tolist(),evidence=total,threshold=float(threshold),
                        level_unit={'orientation':'7.5 degrees','contrast':'ln(4.5)/8 log amplitude',
                        'spatial_frequency':'log2(2.5)/8 octaves','chromatic_increment':'.03 RGB-axis units',
                        'natural_spectrum':'.15 negative-beta units'}[family])
        instruction=visual_cues(np.full((3,100,100),.5,np.float32),'integration','sample',sign,threshold)
        frames=[instruction]
        for image in evidence:
            frames.append(visual_cues(image,'integration','sample',sign if condition['cue_visible'] else None,
                                     threshold if condition['cue_visible'] else None))
        frames+=self._delay(stream,family,int(condition['delay']),condition,'integration')
        frames.append(visual_cues(np.full((3,100,100),.5,np.float32),'integration','report'))
        return frames,dict(oracle,sign=sign,threshold_level=threshold_level,displayed_threshold=threshold,
                          threshold_bar_mapping='width=8*log1p(threshold_level_units)/log(41)',base_id=context['base_id'],
                          natural_gain=context.get('natural_gain'),evidence_events=length,
                          sample_presentations=len(evidence),cue_presentations=1+len(evidence)*int(condition['cue_visible']),
                          probe_presentations=0,report_presentations=1,
                          blank_presentations=condition['delay'] if not condition['distractor'] else 0,
                          distractor_presentations=condition['delay'] if condition['distractor'] else 0)

    def _recall(self,stream,family,label,condition,sample_bit):
        rng=stream.rng;load=int(condition['load']);slots=int(condition.get('sample_slots',load))
        identities=rng.choice(6,load,replace=False).tolist();target=int(rng.integers(load))
        target_id=identities[target]
        contexts=[self._context(stream,family) for _ in range(load)]
        if family!='natural_spectrum':
            # Same nuisance coordinate system makes another item's level a
            # genuine feature-value lure, not a level relative to a new base.
            contexts=[dict(contexts[0]) for _ in range(load)]
        if family=='motion_direction':
            values=rng.integers(0,4,load).tolist();values[target]=label;probe_level=None
            pair_values=None;binding_lure=False
        else:
            # Uniformly unordered distinct pair; four cases A/A,B/B,A/B,B/A
            # have identical sample and probe marginals under either label.
            a,b=sorted(rng.choice(9,2,replace=False).tolist());pair_values=[a,b]
            values=rng.integers(0,9,load).tolist();values[target]=pair_values[sample_bit]
            alternative=pair_values[1-sample_bit]
            if load>1 and rng.integers(2):
                other=int(rng.choice([i for i in range(load) if i!=target]));values[other]=alternative
            probe_level=values[target] if label==0 else alternative
            binding_lure=bool(label and any(values[i]==probe_level for i in range(load) if i!=target))
        instruction=visual_cues(np.full((3,100,100),.5,np.float32),'recall','sample',
                                 identity=target_id if condition['precue'] else None)
        frames=[instruction]
        # Matched-duration load1 control randomly occupies one of four slots;
        # other slots have visibly irrelevant identities/content.
        target_slot=int(rng.integers(slots)) if slots>load else target
        actual_slots=list(range(load)) if slots==load else [target_slot]
        render_seeds=rng.integers(0,2**62,size=2*slots+1)
        sample_offsets=[]
        for slot in range(slots):
            if slot in actual_slots:
                index=actual_slots.index(slot);identity=identities[index]
                if family=='motion_direction':images,_=self._with_seed(stream,render_seeds[2*slot],self._motion_frames,[values[index]])
                else:images=[self._with_seed(stream,render_seeds[2*slot+j],self._render,family,values[index],contexts[index]) for j in range(2)]
                sample_offsets.append(len(frames)+1)
                frames.extend(visual_cues(image,'recall','sample',identity=identity) for image in images)
            else:
                frames+=self._delay(stream,family,2,dict(condition,distractor=True),'recall')
        frames+=self._delay(stream,family,int(condition['delay']),condition,'recall')
        # In precue control, replace the retrocue with equal-format neutral cue.
        frames.append(visual_cues(np.full((3,100,100),.5,np.float32),'recall','query',
                                  identity=None if condition['precue'] else target_id))
        post=int(condition['post_delay'])
        frames+=self._delay(stream,family,post,dict(condition,distractor=False),'recall')
        probe=np.full((3,100,100),.5,np.float32) if family=='motion_direction' else self._with_seed(stream,render_seeds[-1],self._render,family,probe_level,contexts[target])
        frames.append(visual_cues(probe,'recall','report'))
        return frames,dict(base_id=contexts[target]['base_id'],source_base_ids=[c['base_id'] for c in contexts if c['base_id']],
            identities=identities,target_identity=target_id,target_index=target,target_serial_slot=target_slot,
            values=values,probe_value=probe_level,balanced_pair=pair_values,sample_bit=int(sample_bit),
            binding_lure=binding_lure,mismatch=abs(probe_level-values[target]) if probe_level is not None else None,
            natural_gain=contexts[target].get('natural_gain'),target_last_sample_frame=sample_offsets[target],
            target_age_at_report=len(frames)-1-sample_offsets[target],evidence_events=load,
            sample_presentations=2*load,cue_presentations=2,probe_presentations=int(family!='motion_direction'),
            report_presentations=1,blank_presentations=(condition['delay'] if not condition['distractor'] else 0)+post,
            distractor_presentations=(condition['delay'] if condition['distractor'] else 0)+2*(slots-load))

    def matched_motion_pair(self):
        """Two diagnostic histories, different winners, same net and suffix.

        Uses a common nuisance draw: repeated presentations share a group and
        must not be counted as independent fresh episodes in uncertainty.
        """
        stream=self.streams['motion_direction'];rng=stream.rng
        suffix=[0,1,2,3];prefix=[2]
        sequences=[]
        for counts in ([6,3,5,2],[4,5,3,4]):
            middle=np.repeat(np.arange(4),np.asarray(counts)-np.array([1,1,2,1]));rng.shuffle(middle)
            sequences.append(prefix+middle.tolist()+suffix)
        state=self.state_dict();records=[];group=f'matched_motion/{self.seed}/{self.counters["motion_direction"]}'
        for sequence in sequences:
            self.load_state_dict(state)
            records.append(self.batch(1,'motion_direction',setting(length=16,directions=sequence)))
        images=torch.cat([r[0] for r in records]);labels=torch.cat([r[1] for r in records])
        metadata=[r[2][0] for r in records]
        for i,meta in enumerate(metadata):
            meta.update(trial_id=group+f'/{i}',shared_history_group=group,condition='matched_motion_net_suffix')
        return images,labels,metadata

    def batch(self,n,family,condition):
        if family not in FAMILIES:raise ValueError(family)
        if isinstance(condition,str):name=condition;cfg=copy.deepcopy(CONDITIONS[name])
        else:name=condition.get('name','custom');cfg=dict(BASE,**condition)
        stream=self.streams[family];rng=stream.rng
        classes=TASK_CLASSES[family]
        labels=(np.arange(n,dtype=np.int64)+int(rng.integers(classes)))%classes;rng.shuffle(labels)
        if 'directions' in cfg:labels[:]=duration_oracle(cfg['directions'])[0]
        sign=int(rng.choice([-1,1])) if 'sign' not in cfg else int(cfg['sign'])
        threshold_level=int(rng.integers(2)) if cfg.get('threshold_level') is None else int(cfg['threshold_level'])
        frames=[];metadata=[]
        for i,label in enumerate(labels):
            if cfg['protocol']=='anchor':
                # Existing label semantics distinguished by the anchor glyph.
                if family=='natural_spectrum':
                    context=self._context(stream,family);low,high=sorted(rng.choice(9,2,replace=False).tolist())
                    levels=[low,low];levels[int(label)]=high
                    pair_images=[self._render(stream,family,level,context) for level in levels]
                    anchor_meta=dict(base_id=context['base_id'],natural_gain=context['natural_gain'],levels=levels)
                else:
                    pair,y,meta=stream.batch(1,task=family);label=int(y[0]);labels[i]=label
                    pair_images=pair[0].numpy();anchor_meta=meta[0]
                images=[visual_cues(x,'anchor','sample' if j==0 else 'report') for j,x in enumerate(pair_images)]
                info=dict(anchor_meta,evidence_events=1,sample_presentations=2,cue_presentations=2,
                          probe_presentations=0,report_presentations=1,blank_presentations=0,distractor_presentations=0)
            elif cfg['protocol']=='integration':
                images,info=self._integration(stream,family,int(label),cfg,sign,threshold_level)
            elif cfg['protocol']=='recall':
                sample_bit=int(rng.integers(2))
                images,info=self._recall(stream,family,int(label),cfg,sample_bit)
            else:raise ValueError('Unknown visual sequence protocol')
            info.update(protocol=cfg['protocol'],version=VERSION,family=family,task=family,label=int(label),
                        condition=name,condition_config=cfg,split=self.split,
                        trial_id=f'WM/{self.split}/{self.seed}/{family}/{self.counters[family]}',
                        frame_count=len(images),encoder_updates=len(images),
                        repeated_identical_adjacent_frames=sum(np.array_equal(a,b) for a,b in zip(images,images[1:])),
                        difficulty=f"{cfg['protocol']}/L{cfg['length']}/D{cfg['delay']}/N{cfg['load']}")
            self.counters[family]+=1;frames.append(np.stack(images));metadata.append(info)
        result=np.stack(frames).astype(np.float32)
        if not np.isfinite(result).all() or result.min()<0 or result.max()>1:
            raise ValueError('Invalid sequence pixels')
        return torch.from_numpy(result),torch.from_numpy(labels),metadata
