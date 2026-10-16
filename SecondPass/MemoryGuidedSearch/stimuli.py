"""Procedural RGB stimuli. Generator identities and phases are analysis-only.

Fractals here are spectral fractional-noise textures, not Mandelbrot pictures.
Exact raster matches are translated from the sample to a balanced array slot.
"""
import hashlib
import numpy as np
from PIL import Image, ImageDraw

SIZE=28
CENTERS=((25,25),(75,25),(25,75),(75,75))  # x,y; TL,TR,BL,BR
PALETTE=np.array([[220,65,70],[65,180,100],[65,110,225],[215,160,45]])

def digest(image):return hashlib.sha256(image.tobytes()).hexdigest()

def feature_objects(rng,similar):
    # Every hard-array feature value occurs twice; no feature alone locates
    # any of the four possible queries. All objects can be queried equally.
    codes=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)] if similar else [(i,i,i) for i in range(4)]
    colors=rng.permutation(4);shapes=rng.permutation(4);textures=rng.permutation(4)
    stretch=rng.uniform(.8,1.1);radius=rng.uniform(9.5,11.5)
    sprites=[];descriptors=[]
    for c,s,t in codes:
        mask=Image.new('L',(SIZE*4,SIZE*4));draw=ImageDraw.Draw(mask);cx=cy=SIZE*2;r=radius*4
        kind=int(shapes[s]);w=r*stretch
        if kind==0:draw.ellipse((cx-w,cy-r,cx+w,cy+r),fill=255)
        else:
            vertices={1:[(-1,-1),(1,-1),(1,1),(-1,1)],2:[(0,-1),(1,1),(-1,1)],3:[(0,-1),(1,0),(0,1),(-1,0)]}[kind]
            draw.polygon([(cx+x*w,cy+y*r) for x,y in vertices],fill=255)
        alpha=np.array(mask.resize((SIZE,SIZE),Image.Resampling.LANCZOS))/255.
        yy,xx=np.mgrid[:SIZE,:SIZE];tex=int(textures[t])
        patterns=[xx%6<3,yy%6<3,(xx+yy)%7<3,((xx%7-3)**2+(yy%7-3)**2)<5]
        color=PALETTE[colors[c]];ink=color[None,None,:]*(.55+.45*patterns[tex][...,None])
        sprites.append(np.rint(128*(1-alpha[...,None])+ink*alpha[...,None]).astype(np.uint8))
        descriptors.append(dict(color=int(colors[c]),shape=kind,texture=tex))
    return sprites,descriptors

def contour_objects(rng,similar):
    base=rng.normal(0,1,(2,5));sprites=[];descriptors=[]
    yy,xx=np.mgrid[:SIZE*4,:SIZE*4];dx=(xx+.5)/4-SIZE/2;dy=(yy+.5)/4-SIZE/2
    theta=np.arctan2(dy,dx);distance=np.sqrt(dx*dx+dy*dy)
    for _ in range(4):
        coeff=base+.4*rng.normal(size=(2,5)) if similar else rng.normal(size=(2,5))
        rad=np.full_like(theta,9.)
        for k in range(5):rad+=1.5/(k+1)*(coeff[0,k]*np.sin((k+2)*theta)+coeff[1,k]*np.cos((k+2)*theta))
        # Analytic radial fill avoids scan-line cracks in polygon rasterization.
        gray=np.where(distance<=np.clip(rad,4,12),225,128).astype(np.uint8)
        image=Image.fromarray(np.repeat(gray[...,None],3,axis=2))
        sprites.append(np.array(image.resize((SIZE,SIZE),Image.Resampling.LANCZOS)))
        descriptors.append(dict(fourier_coefficients=coeff.tolist()))
    return sprites,descriptors

def fractal_objects(rng,similar):
    fy=np.fft.fftfreq(SIZE)[:,None];fx=np.fft.rfftfreq(SIZE)[None,:]
    freq=np.sqrt(fx*fx+fy*fy);amplitude=np.maximum(freq,1/SIZE)**-1.35;amplitude[0,0]=0
    def field():return np.fft.irfft2(np.fft.rfft2(rng.normal(size=(SIZE,SIZE)))*amplitude,s=(SIZE,SIZE))
    base=field();base/=base.std();yy,xx=np.mgrid[:SIZE,:SIZE];mask=(xx-13.5)**2+(yy-13.5)**2<=12**2
    # Rank normalization gives exactly identical histograms, means and RMS
    # contrast within the same circular support, including close distractors.
    levels=np.rint(np.linspace(30,225,mask.sum())).astype(np.uint8);sprites=[]
    for _ in range(4):
        f=field();f/=f.std();f=base+.35*f if similar else f
        order=np.argsort(f[mask],kind='stable');values=np.empty_like(levels);values[order]=levels
        tile=np.full((SIZE,SIZE),128,np.uint8);tile[mask]=values;sprites.append(np.repeat(tile[...,None],3,axis=2))
    return sprites,[dict(generator='spectral_fractal_noise',amplitude_exponent=1.35,shared_parent=similar) for _ in sprites]

def render_scene(seed,family,similarity):
    rng=np.random.default_rng(seed)
    renderer={'features':feature_objects,'objects':contour_objects,'fractals':fractal_objects}[family]
    for _ in range(20):
        sprites,descriptors=renderer(rng,similarity=='similar')
        if len({digest(s) for s in sprites})==4:break
    else:raise RuntimeError('Could not generate four distinguishable rasters')
    order=rng.permutation(4);sprites=[sprites[i] for i in order];descriptors=[descriptors[i] for i in order]
    array=np.full((100,100,3),128,np.uint8)
    for sprite,(cx,cy) in zip(sprites,CENTERS):array[cy-14:cy+14,cx-14:cx+14]=sprite
    return sprites,array,descriptors

def movie(sprites,array,target,delay,exposure,sample_frames=2):
    blank=np.full_like(array,128);sample=blank.copy();sample[36:64,36:64]=sprites[target]
    frames=[blank]+[sample]*sample_frames+[blank]*delay+[array]*exposure
    return np.stack(frames),['fixation']+['sample']*sample_frames+['delay']*delay+['search']*exposure


def popout_scene(seed,feature,difference,target):
    """Three identical distractors and one singleton; no sample or spatial cue.

    Color levels are RGB opponent amplitudes, not calibrated perceptual units.
    Randomly swap which member of a color pair is the singleton so absolute
    target and distractor color distributions are identical in expectation.
    """
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:SIZE*4,:SIZE*4];dx=(xx+.5)/4-SIZE/2;dy=(yy+.5)/4-SIZE/2
    if feature=='color':
        midpoint=rng.uniform(-20,20);sign=rng.choice([-1,1])
        opponent=np.array([1.,-.2126/.7152,0.])
        # Keep every disk visibly brighter than the background even when its
        # chromatic offset is near zero; otherwise it can disappear entirely.
        colors=[180+(midpoint+sgn*sign*difference/2)*opponent for sgn in (-1,1)]
        mask=dx*dx+dy*dy<=9**2
        masks=[mask,mask]
        descriptor=dict(distractor_rgb=colors[0].tolist(),target_rgb=colors[1].tolist(),
                        difference_units='RGB opponent amplitude; not perceptual distance')
    elif feature=='orientation':
        base=rng.uniform(0,180);angle=(base+rng.choice([-1,1])*difference)%180
        masks=[]
        for degrees in (base,angle):
            theta=np.deg2rad(degrees)
            along=dx*np.cos(theta)+dy*np.sin(theta);across=-dx*np.sin(theta)+dy*np.cos(theta)
            masks.append((np.abs(along)<=10)&(np.abs(across)<=2.5))
        colors=[np.array([225.]*3)]*2
        descriptor=dict(distractor_degrees=float(base),target_degrees=float(angle),difference_units='degrees modulo 180')
    else:raise ValueError(feature)
    tiles=[]
    for mask,color in zip(masks,colors):
        raw=np.rint(np.where(mask[...,None],color,128)).astype(np.uint8)
        tiles.append(np.array(Image.fromarray(raw).resize((SIZE,SIZE),Image.Resampling.LANCZOS)))
    if digest(tiles[0])==digest(tiles[1]):raise RuntimeError('Singleton raster equals distractors')
    array=np.full((100,100,3),128,np.uint8)
    for i,(cx,cy) in enumerate(CENTERS):array[cy-14:cy+14,cx-14:cx+14]=tiles[int(i==target)]
    return array,descriptor
