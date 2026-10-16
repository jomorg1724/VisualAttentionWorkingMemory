"""Pixel-only motion matching: independent random-dot samples and search movies."""
import numpy as np
from .stimuli import CENTERS, SIZE, digest

SPEED=1.5  # input pixels per frame; same for sample and all alternatives
DOTS=12
SIGMA=.65


def dot_movie(seed,degrees,frames):
    """Stationary uniform initial positions, coherent translation on a torus.

    All positions exist before the first visible frame; its pixels do not depend
    on direction. Circular masking is applied after wrapping. No reset at the
    aperture boundary, lifetime event, trail, or direction annotation is shown.
    """
    rng=np.random.default_rng(seed);initial=rng.uniform(0,SIZE,(DOTS,2))
    theta=np.deg2rad(degrees);velocity=SPEED*np.array([np.cos(theta),np.sin(theta)])
    yy,xx=np.mgrid[:SIZE,:SIZE];grid=np.stack([xx+.5,yy+.5],axis=-1)
    aperture=(xx+.5-SIZE/2)**2+(yy+.5-SIZE/2)**2<=12**2
    result=[]
    for t in range(frames):
        positions=(initial+t*velocity)%SIZE
        delta=(grid[:,:,None,:]-positions+SIZE/2)%SIZE-SIZE/2
        ink=np.exp(-np.sum(delta**2,axis=-1)/(2*SIGMA**2)).max(axis=-1)
        gray=np.rint(128+110*ink*aperture).astype(np.uint8)
        result.append(np.repeat(gray[...,None],3,axis=2))
    return np.stack(result)


def motion_trial(namespace,trial_id,spacing,sample_frames,delay,exposure,target=None):
    if trial_id<0:raise ValueError(trial_id)
    if sample_frames not in (3,6,12) or exposure not in (3,6,12) or delay not in (0,4,12):raise ValueError('Invalid motion timing')
    if spacing not in (20,45,90):raise ValueError(spacing)
    seed=[namespace,42,spacing,trial_id]
    rng=np.random.default_rng(seed);base=float(rng.uniform(0,360))
    directions=((base+np.arange(4)*spacing)%360)[rng.permutation(4)]
    if target is None:
        order=np.random.default_rng([namespace,42,spacing,trial_id//4,818]).permutation(4)
        target=int(order[trial_id%4])
    if target not in range(4):raise ValueError(target)
    # Layout seeds do not depend on the query. Across counterfactual queries,
    # only sample dynamics change; even the sample's first frame is identical.
    sample=dot_movie(seed+[100],directions[target],sample_frames)
    alternatives=[dot_movie(seed+[i],angle,exposure) for i,angle in enumerate(directions)]
    blank=np.full((100,100,3),128,np.uint8)
    samples=np.repeat(blank[None],sample_frames,axis=0);samples[:,36:64,36:64]=sample
    arrays=np.repeat(blank[None],exposure,axis=0)
    for tiles,(cx,cy) in zip(alternatives,CENTERS):arrays[:,cy-14:cy+14,cx-14:cx+14]=tiles
    raw=np.concatenate([blank[None],samples,np.repeat(blank[None],delay,axis=0),arrays])
    metadata=dict(spacing=spacing,sample_frames=sample_frames,delay=delay,exposure=exposure,
                  target_index=target,sample_direction=float(directions[target]),directions= directions.tolist(),
                  seed=seed,speed=SPEED,dots=DOTS,dot_sigma=SIGMA,coherence=1.,
                  centers_xy=[list(c) for c in CENTERS],sample_bbox=[36,36,64,64],sprite_size=SIZE,
                  phases=['fixation']+['sample']*sample_frames+['delay']*delay+['search']*exposure,
                  array_onset=1+sample_frames+delay,target_present=True,independent_dot_layouts=True,
                  sample_first_frame_sha256=digest(sample[0]),array_first_frame_sha256=digest(arrays[0]),
                  array_sha256=digest(arrays))
    return raw,target,metadata
