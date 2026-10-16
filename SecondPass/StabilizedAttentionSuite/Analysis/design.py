"""Prospective finite analysis design for frozen suite checkpoint54808."""
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).with_name('run01')
CHECKPOINT=ROOT/'SecondPass/StabilizedAttentionSuite/LocalRuntime/continuation02/best.pt'
STRENGTHS={
 'motion_direction':[0,.1,.2,.4,.7,1,2,3],
 'orientation':[0,.5,1,2,4,7,10,15,22],
 'contrast':[0,.002,.005,.01,.018,.025,.04,.06,.13],
 'spatial_frequency':[0,.005,.01,.02,.04,.08,.12,.18,.35],
 'chromatic_increment':[0,.001,.003,.006,.01,.018,.03,.045,.1],
 'contour':[0,2,4,8,12,16,24,36,60],
 'natural_spectrum':[0,.01,.025,.05,.1,.15,.3,.6],
 'orientation_ring':[0,2,4,8,15,22,30,45],
 'orientation_cued':[0,2,4,8,15,22,30,45],
 'krauzlis_cued_motion':[0,4,8,12,18,26,28,40,60]}
NATIVE={'motion_direction':[1,2,3],'orientation':[4,10,22],'contrast':[.025,.06,.13],
 'spatial_frequency':[.08,.18,.35],'chromatic_increment':[.018,.045,.1],
 'contour':[2,8,16],'natural_spectrum':[.15,.3,.6],'orientation_ring':[15,30,45],
 'orientation_cued':[15,30,45],'krauzlis_cued_motion':[26,28]}
AXES={'motion_direction':'Displacement (pixels)','orientation':'Rotation magnitude (degrees)',
 'contrast':'Contrast increment','spatial_frequency':'Frequency difference (octaves)',
 'chromatic_increment':'Linear-RGB increment','contour':'Alignment jitter (degrees; harder →)',
 'natural_spectrum':'Spectral slope difference','orientation_ring':'Rotation magnitude (degrees)',
 'orientation_cued':'Rotation magnitude (degrees)','krauzlis_cued_motion':'Motion change (degrees)',
 'motion_duration_cued':'Blank delay (frames)','spatial_binding':'Blank delay (frames)',
 'image_recognition':'Study images'}
DOSES=[-12,-8,-4,-2,2,4,8,12]

def interventions(task):
    arms=[dict(name='sham',site='target',dose=0,epoch='all')]
    for site in ('target','distractor'):
        for dose in DOSES:arms.append(dict(name=f'{site}_{dose:+d}_all',site=site,dose=dose,epoch='all'))
    for site in ('offsite','both','global'):
        for dose in (-12,12):arms.append(dict(name=f'{site}_{dose:+d}_all',site=site,dose=dose,epoch='all'))
    epochs=('cue','baseline','postchange','report') if task=='krauzlis_cued_motion' else ('cue','sample','delay','probe')
    for site in ('target','distractor'):
        for epoch in epochs:
            for dose in (-8,8):arms.append(dict(name=f'{site}_{dose:+d}_{epoch}',site=site,dose=dose,epoch=epoch))
    if task=='orientation_cued':
        for dose in (-8,8):arms.append(dict(name=f'cueglyph_{dose:+d}_cue',site='cueglyph',dose=dose,epoch='cue'))
    return arms

def spatial_mask(center,radius=12):
    yy,xx=np.meshgrid(np.arange(10)*10+4.5,np.arange(10)*10+4.5,indexing='ij')
    return ((xx-center[0])**2+(yy-center[1])**2<radius**2).reshape(100)

def bias_for(task,meta,arm,t):
    centers=np.array(meta['positions_xy']);target=int(meta['target_location']);foil=int(meta.get('analysis_foil_location',(target+1)%len(centers)))
    site=arm['site'];mask=spatial_mask(centers[target])
    if site=='distractor':mask=spatial_mask(centers[foil])
    elif site=='offsite':mask=spatial_mask((50,20))
    elif site=='both':mask=spatial_mask(centers[target])|spatial_mask(centers[foil])
    elif site=='global':mask=np.ones(100,dtype=bool)
    elif site=='cueglyph':mask=spatial_mask(centers[target]+[0,-23],radius=9)
    epoch=arm['epoch'];time_mask=np.ones(t,dtype=bool)
    if task=='krauzlis_cued_motion':
        e=meta['first_postchange_frame'];windows={'cue':range(2),'baseline':range(7,e),'postchange':range(e,t-1),'report':[t-1]}
    else:windows={'cue':[0],'sample':[1,2],'delay':meta['blank_frames'],'probe':[t-1]}
    if epoch!='all':time_mask[:]=False;time_mask[list(windows[epoch])]=True
    return time_mask[:,None]*mask[None,:]*float(arm['dose'])

def phase_labels(task,meta,t):
    a=['observation']*t
    if t==2:return ['reference','comparison']
    a[0]='instruction'
    for key,label in [('cue_frames','cue'),('sample_frames','sample'),('study_frames','study'),('blank_frames','blank'),('probe_frames','probe'),('fixation_only_frames','fixation'),('moving_frames','motion')]:
        for i in meta.get(key,[]):a[i]=label
    for key,label in [('probe_frame','probe'),('report_frame','report'),('reference_frame','reference')]:
        if key in meta:a[meta[key]]=label
    if task=='krauzlis_cued_motion':
        for i in range(7,meta['first_postchange_frame']):a[i]='baseline'
        for i in range(meta['first_postchange_frame'],t-1):a[i]='postchange'
    return a
