"""Prospective analysis plan; the previous model is treated as one trained subject."""
from pathlib import Path
import json
import time
import numpy as np
from SecondPass.MemoryGuidedSearch.dataset import TASKS

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).with_name('Results')
BASE=ROOT/'SecondPass/MemoryGuidedSearch/LocalRuntime/static_memory4x01'
CHECKPOINTS={'best':BASE/'best.pt','latest':BASE/'latest.pt'}
EXPECTED={'best':'596bc351783b3ddbfa8d37a9cf8d65146d1333dcc0097d40f0f75665b6451bcf',
          'latest':'c13338ff6958c5dbb617914da538bca139944b89122b848b72885ed3241995d4'}
NS=205903
NATIVE_N=128
PSYCH_N=128
PROBE_TRAIN=192
PROBE_TEST=128
INTERVENTION_N=128
BATCH=8
THRESHOLDS=[.6,.7,.8,.9]

def neural_cells(task):
    cells=[c['id'] for c in TASKS[task]['conditions']]
    if task=='memory_guided_search':return [f'{f}_{s}_S8_D12_H8' for f in ('features','objects','fractals') for s in ('distinct','similar')]
    if task=='motion_memory_search':return ['motion_A20_S12_D12_H12','motion_A90_S12_D12_H12']
    if task=='popout_search':return ['color_X12_H8','orientation_X5_H8']
    if task=='image_recognition':return ['N4_H5','N24_H5']
    return [cells[-1]]

def phases(task,m,t):
    if isinstance(m.get('phases'),list):return m['phases']
    p=['observation']*t
    if t==2:return ['reference','comparison']
    for k,v in [('cue_frames','cue'),('sample_frames','sample'),('study_frames','study'),('blank_frames','delay'),('probe_frames','probe'),('moving_frames','motion'),('fixation_only_frames','fixation')]:
        for i in m.get(k,[]) if isinstance(m.get(k,[]),list) else []:p[i]=v
    for k,v in [('probe_frame','probe'),('report_frame','report'),('reference_frame','reference')]:
        if k in m:p[m[k]]=v
    if task=='krauzlis_cued_motion':
        for i in range(7,m['first_postchange_frame']):p[i]='baseline'
        for i in range(m['first_postchange_frame'],t-1):p[i]='change'
    return p

def onset(task,m,t):
    if 'array_onset' in m:return m['array_onset']
    if task.startswith('vda'):return 5
    if 'first_postchange_frame' in m:return m['first_postchange_frame']
    if task=='motion_duration_cued':return 2
    if 'probe_frame' in m:return m['probe_frame']
    if m.get('probe_frames'):return m['probe_frames'][0]
    return t-1

def retention_boundary(task,m,t):
    p=phases(task,m,t)
    delays=[i for i,v in enumerate(p) if v in ('delay','blank delay')]
    return delays[len(delays)//2] if delays else max(0,onset(task,m,t)-1)

def sites(task,m):
    if task.startswith('vda'):
        from SecondPass.StabilizedAttentionVDA.Analysis.design import masks
        s,off,_=masks(int(task[3:]));a=int(m.get('cue_index',m.get('cue',0)));b=len(s)-1-a
        return s[a],s[b],off
    yy,xx=np.meshgrid(np.arange(10)*10+4.5,np.arange(10)*10+4.5,indexing='ij')
    centers=m.get('positions_xy',m.get('centers_xy'))
    if centers is None and 'target_location' in m:centers=[[27,27],[73,27],[27,73],[73,73]]
    if centers is not None:
        a=int(m.get('target_location',m.get('target_index',0)));b=(a+1)%len(centers)
        # Task-defined location, not the location preferred by the fitted model.
        radius=16 if 'centers_xy' in m else 12
        mask=lambda c:((xx-c[0])**2+(yy-c[1])**2<radius**2).flatten()
        first,second=mask(centers[a]),mask(centers[b])
    else:
        # Full-field tasks have no privileged target site: test fixed opposite quadrants.
        first=((xx<50)&(yy<50)).flatten();second=((xx>=50)&(yy>=50)).flatten()
    off=np.zeros(100,bool);allowed=np.flatnonzero(~(first|second));off[allowed[len(allowed)//2]]=True
    return first,second,off

def arms():
    out=[dict(name='sham',site='target',dose=0,epoch='evidence')]
    for site in ('target','distractor'):
        for dose in (-8,-4,-2,2,4,8):out.append(dict(name=f'{site}_{dose:+d}_evidence',site=site,dose=dose,epoch='evidence'))
        for epoch in ('encoding','delay'):
            for dose in (-4,4):out.append(dict(name=f'{site}_{dose:+d}_{epoch}',site=site,dose=dose,epoch=epoch))
    for site in ('offsite','global'):
        for dose in (-4,4):out.append(dict(name=f'{site}_{dose:+d}_evidence',site=site,dose=dose,epoch='evidence'))
    return out

def bias(task,metas,t,arm):
    result=np.zeros((len(metas),t,100),np.float32)
    for i,m in enumerate(metas):
        ss=sites(task,m);mask=ss[{'target':0,'distractor':1,'offsite':2}.get(arm['site'],0)]
        if arm['site']=='global':mask=np.ones(100,bool)
        phase=phases(task,m,t);start=onset(task,m,t)
        ix=list(range(start,t)) if arm['epoch']=='evidence' else [j for j,v in enumerate(phase) if v in (('sample','study','reference') if arm['epoch']=='encoding' else ('delay','blank delay'))]
        # A missing epoch is not silently replaced by another manipulation.
        result[i,ix]=mask*arm['dose']
    return result

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    # Windows readers may briefly deny replacement; retain atomic publication.
    for attempt in range(50):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.02)
