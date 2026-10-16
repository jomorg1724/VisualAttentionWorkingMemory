"""19 tasks with object-memory, pop-out and motion-memory localization."""
import copy
import numpy as np
import torch
from SecondPass.StabilizedAttentionVDA.dataset import Stream as VDAStream, CATALOG as BASE, TASKS as OLD_TASKS
from SecondPass.TaskSuite.suite import SuiteStream
from .stimuli import CENTERS,digest,render_scene,movie,popout_scene
from .motion import motion_trial

TASK='memory_guided_search'
POPOUT='popout_search'
MOTION='motion_memory_search'
DURATIONS=(1,2,4,8)
NAMESPACES=dict(train=204101,val=204102,test=204103)
CONDITIONS=[dict(id=f'{family}_{sim}_S{sample}_D{delay}_H{exposure}',kwargs=dict(family=family,similarity=sim,sample_frames=sample,delay=delay,exposure=exposure))
            for family in ('features','objects','fractals') for sim in ('distinct','similar') for sample in DURATIONS for delay in (0,4,12) for exposure in DURATIONS]
POPOUT_CONDITIONS=[dict(id=f'{feature}_X{difference}_H{exposure}',kwargs=dict(feature=feature,difference=difference,exposure=exposure))
                  for feature,levels in (('color',(12,36,72)),('orientation',(5,15,45))) for difference in levels for exposure in DURATIONS]
MOTION_CONDITIONS=[dict(id=f'motion_A{spacing}_S{sample}_D{delay}_H{exposure}',kwargs=dict(spacing=spacing,sample_frames=sample,delay=delay,exposure=exposure))
                   for spacing in (20,45,90) for sample in (3,6,12) for delay in (0,4,12) for exposure in (3,6,12)]
CATALOG=copy.deepcopy(BASE);CATALOG.update(version='object_popout_motion_search_suite_v3',training_authorized=False,split_seeds=NAMESPACES,
    future_initialization='continue VDA best16000 with three fresh four-class location heads')
CATALOG['tasks'].append(dict(id=TASK,stream_id=40,classes=4,group='memory_guided_search',requires_bsds500=False,
    labels=['top-left','top-right','bottom-left','bottom-right'],question='Where is the object shown in the sample?',
    conditions=CONDITIONS,report_by=['family','similarity','sample_frames','delay','exposure','target_index','minimum_pair_rms']))
CATALOG['tasks'].append(dict(id=POPOUT,stream_id=41,classes=4,group='popout_search',requires_bsds500=False,
    labels=['top-left','top-right','bottom-left','bottom-right'],question='Where is the item that differs from the other three?',
    conditions=POPOUT_CONDITIONS,report_by=['feature','difference','exposure','target_index']))
CATALOG['tasks'].append(dict(id=MOTION,stream_id=42,classes=4,group='motion_memory_search',requires_bsds500=False,
    labels=['top-left','top-right','bottom-left','bottom-right'],question='Which patch moves in the remembered sample direction?',
    conditions=MOTION_CONDITIONS,report_by=['spacing','sample_frames','delay','exposure','target_index']))
TASKS={task['id']:task for task in CATALOG['tasks']}

class FreshOld(SuiteStream):
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return NAMESPACES[self.split]*100000+OLD_TASKS[task]['stream_id']*1000+index

class LegacyStream(VDAStream):
    def __init__(self,split='train'):
        super().__init__(split);self.old=FreshOld(split)
    def batch(self,n,task,cell):
        if task.startswith('vda') and (task,cell) not in self.rngs:
            index=next(i for i,c in enumerate(OLD_TASKS[task]['conditions']) if c['id']==cell)
            self.rngs[task,cell]=np.random.RandomState((NAMESPACES[self.split]*100000+OLD_TASKS[task]['stream_id']*1000+index)%(2**32));self.counts[task,cell]=0
        return super().batch(n,task,cell)
    def load_state_dict(self,state):
        if state['version']!=BASE['version'] or state['split']!=self.split:raise ValueError('Legacy stream mismatch')
        old=FreshOld(self.split);old.load_state_dict(state['old']);rngs={};counts={}
        for row in state['streams']:
            task,cell=row['task'],row['cell'];key=(task,cell)
            if task not in OLD_TASKS or not task.startswith('vda') or cell not in {c['id'] for c in OLD_TASKS[task]['conditions']} or key in rngs:raise ValueError('Invalid legacy stream')
            if type(row['count'])!=int or row['count']<0:raise ValueError('Invalid legacy count')
            rng=np.random.RandomState();rng.set_state(row['rng']);rngs[key]=rng;counts[key]=row['count']
        self.old=old;self.rngs=rngs;self.counts=counts

class Stream:
    def __init__(self,split='train'):
        if split not in NAMESPACES:raise ValueError(split)
        self.split=split;self.legacy=LegacyStream(split);self.counts={}
    def _condition(self,cell,task=TASK):
        for i,c in enumerate(TASKS[task]['conditions']):
            if c['id']==cell:return i,c['kwargs']
        raise ValueError(cell)
    def trial(self,cell,trial_id,target=None):
        index,c=self._condition(cell)
        if trial_id<0:raise ValueError(trial_id)
        # Stable namespaces and counter-based generation make both replay and
        # matched cue switches independent of batching and global RNG state.
        scene_index=('features','objects','fractals').index(c['family'])*2+('distinct','similar').index(c['similarity'])
        seed=[NAMESPACES[self.split],40,scene_index,trial_id]
        sprites,array,desc=render_scene(seed,c['family'],c['similarity'])
        if target is None:
            order=np.random.default_rng([NAMESPACES[self.split],40,scene_index,trial_id//4,818]).permutation(4)
            target=int(order[trial_id%4])
        if target not in range(4):raise ValueError(target)
        raw,phases=movie(sprites,array,target,c['delay'],c['exposure'],c['sample_frames'])
        rms=[float(np.sqrt(np.mean(((sprites[i].astype(float)-sprites[j])/255)**2))) for i in range(4) for j in range(i)]
        meta=dict(c,suite_version=CATALOG['version'],suite_task=TASK,suite_cell=cell,trial_id=trial_id,
            suite_trial_id=f'{self.split}/{TASK}/{cell}/{trial_id}',seed=seed,target_index=target,
            centers_xy=[list(p) for p in CENTERS],sprite_size=28,phases=phases,array_onset=1+c['sample_frames']+c['delay'],
            sample_bbox=[36,36,64,64],object_descriptors=desc,sprite_sha256=[digest(s) for s in sprites],array_sha256=digest(array),
            minimum_pair_rms=min(rms),exact_raster_match=True,target_present=True)
        return torch.from_numpy(raw).permute(0,3,1,2).float()/255,torch.tensor(target,dtype=torch.long),meta
    def popout_trial(self,cell,trial_id):
        _,c=self._condition(cell,POPOUT)
        if trial_id<0:raise ValueError(trial_id)
        # Match nuisance features, target and pixels across exposure durations.
        scene_index=next(i for i,(f,d) in enumerate((f,d) for f,levels in (('color',(12,36,72)),('orientation',(5,15,45))) for d in levels) if (f,d)==(c['feature'],c['difference']))
        seed=[NAMESPACES[self.split],41,scene_index,trial_id]
        order=np.random.default_rng([NAMESPACES[self.split],41,scene_index,trial_id//4,818]).permutation(4)
        target=int(order[trial_id%4]);array,desc=popout_scene(seed,c['feature'],c['difference'],target)
        raw=np.stack([np.full_like(array,128)]+[array]*c['exposure'])
        meta=dict(c,suite_version=CATALOG['version'],suite_task=POPOUT,suite_cell=cell,trial_id=trial_id,
                  suite_trial_id=f'{self.split}/{POPOUT}/{cell}/{trial_id}',seed=seed,target_index=target,
                  centers_xy=[list(p) for p in CENTERS],sprite_size=28,array_onset=1,
                  phases=['fixation']+['search']*c['exposure'],sample_frames=0,delay=0,
                  array_sha256=digest(array),object_descriptors=desc,target_present=True)
        return torch.from_numpy(raw).permute(0,3,1,2).float()/255,torch.tensor(target,dtype=torch.long),meta
    def batch(self,n,task,cell):
        if not isinstance(n,int) or isinstance(n,bool) or n<1:raise ValueError(n)
        if task not in TASKS:raise ValueError(task)
        if cell not in {c['id'] for c in TASKS[task]['conditions']}:raise ValueError(cell)
        if task not in (TASK,POPOUT,MOTION):return self.legacy.batch(n,task,cell)
        self._condition(cell,task);start=self.counts.get(cell,0)
        generator={TASK:self.trial,POPOUT:self.popout_trial,MOTION:self.motion_trial}[task]
        rows=[generator(cell,i) for i in range(start,start+n)];self.counts[cell]=start+n
        return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows]),[r[2] for r in rows]
    def paired_queries(self,cell,trial_id):
        rows=[self.trial(cell,trial_id,target) for target in range(4)]
        return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows]),[r[2] for r in rows]
    def motion_trial(self,cell,trial_id,target=None):
        _,c=self._condition(cell,MOTION)
        raw,label,meta=motion_trial(NAMESPACES[self.split],trial_id,**c,target=target)
        meta.update(suite_version=CATALOG['version'],suite_task=MOTION,suite_cell=cell,trial_id=trial_id,
                    suite_trial_id=f'{self.split}/{MOTION}/{cell}/{trial_id}')
        return torch.from_numpy(raw).permute(0,3,1,2).float()/255,torch.tensor(label,dtype=torch.long),meta
    def paired_motion_queries(self,cell,trial_id):
        rows=[self.motion_trial(cell,trial_id,target) for target in range(4)]
        return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows]),[r[2] for r in rows]
    def state_dict(self):return copy.deepcopy(dict(catalog=CATALOG,split=self.split,counts=self.counts,legacy=self.legacy.state_dict()))
    def load_state_dict(self,state):
        if state['catalog']!=CATALOG or state['split']!=self.split:raise ValueError('Stream contract mismatch')
        for cell,n in state['counts'].items():
            task=next((t for t in (TASK,POPOUT,MOTION) if cell in {c['id'] for c in TASKS[t]['conditions']}),None)
            if task is None:raise ValueError(cell)
            self._condition(cell,task)
            if type(n)!=int or n<0:raise ValueError('Invalid stream count')
        legacy=LegacyStream(self.split);legacy.load_state_dict(state['legacy'])
        self.legacy=legacy;self.counts=copy.deepcopy(state['counts'])

def all_cells():return [(t,c['id']) for t,s in TASKS.items() for c in s['conditions']]
def task_classes():return {t:s['classes'] for t,s in TASKS.items()}
