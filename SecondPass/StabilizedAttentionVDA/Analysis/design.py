from pathlib import Path
import numpy as np
import torch
from ..dataset import private_numpy, native_env, adapt_pixels

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name('run01')
CHECKPOINT = ROOT/'SecondPass/StabilizedAttentionVDA/LocalRuntime/continuation01/best.pt'
ANGLES = [2,4,8,12,18,26,40,65]
PHASES = ['blank','cue','blank delay','sample','sample','change','change']
COLORS = ['red','green','blue']
EPOCHS = {'all':range(7),'cue':[1],'delay':[2],'sample':[3,4],'change':[5,6]}

def seed(*args):
    return int(np.random.SeedSequence(args).generate_state(1)[0])

def masks(size):
    env = native_env(size,65)
    # Model patch centers mapped through the exact native nearest-neighbor resize.
    yy,xx = np.meshgrid(np.arange(10)*10+4.5,np.arange(10)*10+4.5,indexing='ij')
    ny,nx = np.floor(yy*env.S/100),np.floor(xx*env.S/100)
    sites=[]
    occupancy=np.zeros((env.S,env.S),np.float32)
    for r0,r1,c0,c1 in env.cells:
        sites.append(((ny>=r0)&(ny<r1)&(nx>=c0)&(nx<c1)).reshape(100))
        gy,gx=np.meshgrid(np.linspace(-1,1,r1-r0),np.linspace(-1,1,c1-c0),indexing='ij')
        occupancy[r0:r1,c0:c1]=(gx*gx+gy*gy<=.25)
    # Pick a token with the least Gabor occupancy, outside the two tested cells.
    pix=torch.nn.functional.interpolate(torch.tensor(occupancy)[None,None],(100,100),mode='nearest')
    occ=torch.nn.functional.avg_pool2d(pix,10).flatten().numpy()
    allowed=~(sites[0]|sites[-1]);index=int(np.argmin(np.where(allowed,occ,100)))
    off=np.zeros(100,bool);off[index]=True
    return np.array(sites),off,float(occ[index])

def movie(size,rep,angle,event,validity,cue=None,namespace=203201,overrides=True,theta=65):
    rng=np.random.RandomState(seed(namespace,size,rep))
    with private_numpy(rng):
        env=native_env(size,theta);first=env.reset()
        if overrides:
            positions=[0,size-1] if size==9 else list(range(size))
            site=positions[rep%len(positions)] if cue is None else cue
            env.cue_index=int(site);env.proportion=float(validity)
            env.cue_color=COLORS[(rep//(2*len(positions)))%3] if cue is None else COLORS[(rep//2)%3]
            env.change_true=int(event!='catch' and angle>0)
            env.change_index=site if event=='cued' else size-1-site
            if not env.change_true:env.change_index=-1
            sign_index=(rep//len(positions))%2 if cue is None else rep%2
            env.orientation_change=float(angle)*(1 if sign_index else -1)
        frames=[first]+[env.step(0)[0] for _ in range(6)]
        x=adapt_pixels(frames)
        meta=dict(size=size,rep=rep,seed=seed(namespace,size,rep),cue=int(env.cue_index),
                  uncued=size-1-int(env.cue_index),event=event,angle=float(angle),validity=float(env.proportion),
                  color=env.cue_color,value=float(env._reward_value()),label=int(env.change_true),
                  change_index=int(env.change_index),sign=1 if env.orientation_change>=0 else -1,
                  counterfactual_cue_violation=bool(env.proportion==1 and env.change_true and env.change_index!=env.cue_index),
                  native_size=env.S,cluster=f'{namespace}:{size}:{rep}')
    return x,meta

def arms():
    out=[dict(name='sham',site='cued',dose=0,epoch='all')]
    for site in ('cued','uncued'):
        for dose in (-8,-4,-2,2,4,8):
            out.append(dict(name=f'{site}_{dose:+d}_all',site=site,dose=dose,epoch='all'))
    for site in ('offsite','both','global'):
        for dose in (-8,8):out.append(dict(name=f'{site}_{dose:+d}_all',site=site,dose=dose,epoch='all'))
    for site in ('cued','uncued'):
        for epoch in ('cue','delay','sample','change'):
            for dose in (-4,4):out.append(dict(name=f'{site}_{dose:+d}_{epoch}',site=site,dose=dose,epoch=epoch))
    return out

PRIMARY_ARMS=[dict(name='sham',site='cued',dose=0,epoch='change')]+[
    dict(name=f'{site}_{dose:+d}_change',site=site,dose=dose,epoch='change')
    for site,dose in [('cued',-2),('cued',-4),('cued',-8),('uncued',4),('cued',4),('uncued',-4)]]

def bias(meta,arm,site_masks):
    cells,off,_=site_masks
    site=arm['site'];mask=cells[meta['cue']]
    if site=='uncued':mask=cells[meta['uncued']]
    elif site=='both':mask=cells[meta['cue']]|cells[meta['uncued']]
    elif site=='offsite':mask=off
    elif site=='global':mask=np.ones(100,bool)
    b=np.zeros((7,100),np.float32)
    b[list(EPOCHS[arm['epoch']])]=mask*float(arm['dose'])
    return b
