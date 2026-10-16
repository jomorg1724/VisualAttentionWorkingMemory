"""Deterministic CPU teaching simulations. No checkpoints, optimization, or GPU use."""
from pathlib import Path
import json, hashlib, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'pixels-figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfbf8','axes.facecolor':'#fcfbf8','savefig.facecolor':'#fcfbf8','svg.fonttype':'none'})
teal='#087f81'; coral='#c66249'; violet='#6f58a5';rng=np.random.default_rng(20261009)
checks={};files=[]
def save(fig,name):
    fig.text(.015,.008,'SYNTHETIC TEACHING DATA · prescribed stimuli and hand-set example features',fontsize=9,color='#51636a')
    fig.tight_layout(rect=(0,.07,1,.95),h_pad=2.0,w_pad=2.0)
    for ext in ('png','svg','pdf'):
        path=OUT/f'{name}.{ext}';fig.savefig(path,dpi=150,bbox_inches="tight",pad_inches=.15);files.append(path)
    plt.close(fig)
def image(ax,a,title,cmap='gray',**kw):
    im=ax.imshow(a,cmap=cmap,**kw);ax.set_title(title,fontsize=11);ax.set_xticks([]);ax.set_yticks([]);return im
def shifted(a,dx,dy):
    yy,xx=np.indices(a.shape);sx=np.clip(xx-dx,0,a.shape[1]-1);sy=np.clip(yy-dy,0,a.shape[0]-1)
    x0=np.floor(sx).astype(int);y0=np.floor(sy).astype(int);x1=np.minimum(x0+1,a.shape[1]-1);y1=np.minimum(y0+1,a.shape[0]-1)
    u=sx-x0;v=sy-y0
    return (1-u)*(1-v)*a[y0,x0]+u*(1-v)*a[y0,x1]+(1-u)*v*a[y1,x0]+u*v*a[y1,x1]
yy,xx=np.indices((100,100)); gray=.5+.23*np.sin(.45*xx+.22*yy)*np.exp(-((xx-65)**2+(yy-45)**2)/350)
rgb=np.repeat(gray[...,None],3,-1);rgb[15:30,14:29]=[.9,.3,.3];rgb[66:81,72:87]=[.3,.3,.9]
fig,axs=plt.subplots(1,3,figsize=(11,4));image(axs[0],rgb,'One 100 × 100 RGB frame')
for n in range(0,101,10):axs[0].axhline(n-.5,color='white',alpha=.3,lw=.4);axs[0].axvline(n-.5,color='white',alpha=.3,lw=.4)
axs[0].add_patch(Rectangle((59.5,39.5),10,10,fill=False,ec=teal,lw=2));axs[0].add_patch(Rectangle((52.5,32.5),24,24,fill=False,ec=coral,lw=2))
image(axs[1],rgb[40:50,60:70],'10 × 10 statistics block\nrows 40–49, columns 60–69',interpolation='nearest')
image(axs[2],rgb[33:57,53:77],'24 × 24 motion crop\nrows 33–56, columns 53–76',interpolation='nearest')
save(fig,'01-spatial-neighborhoods')

fig,axs=plt.subplots(2,3,figsize=(10,5.6));colors=np.array([[.9,.3,.3],[.3,.3,.9]])
for i,c in enumerate(colors):
    image(axs[0,i],np.tile(c,(10,10,1)),f'RGB = {tuple(c)}');image(axs[1,i],np.full((10,10),c.mean()),f'Grayscale mean = {c.mean():.2f}',vmin=0,vmax=1)
axs[0,2].bar(np.arange(3)-.17,colors[0]-.5,.34,color=coral,label='Red patch');axs[0,2].bar(np.arange(3)+.17,colors[1]-.5,.34,color=teal,label='Blue patch');axs[0,2].set_xticks(range(3),['R','G','B']);axs[0,2].set_ylim(-.3,.5);axs[0,2].set_title('Centered RGB means\nremain different');axs[0,2].legend(fontsize=8)
contrast=np.indices((10,10)).sum(0)%2*.4+.3
axs[1,2].bar(['Flat 0.5','Flat 0.7','0.3 / 0.7\ncheckerboard'],[0,.2,np.sqrt(((contrast-.5)**2).mean())],color=[teal,coral,violet]);axs[1,2].set_ylim(0,.25);axs[1,2].set_title('RMS relative to 0.5');save(fig,'02-what-grayscale-loses')
checks['equal_grayscale_distinct_color']=bool(np.isclose(colors.mean(1)[0],colors.mean(1)[1]))

edge=(xx>=50).astype(float);ky=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])/3;kx=ky.T
def conv(a,k):
    padded=np.pad(a,1);return sum(k[i,j]*padded[i:i+a.shape[0],j:j+a.shape[1]] for i in range(3) for j in range(3))
toy=.5+.3*((xx>25)&(xx<75)&(yy>25)&(yy<75)).astype(float)
fig,axs=plt.subplots(1,4,figsize=(12,3.7));image(axs[0],toy,'Toy square',vmin=0,vmax=1)
for ax,k,title in zip(axs[1:3],[kx,ky],['Hand-set horizontal derivative','Hand-set vertical derivative']):
    response=conv(toy-.5,k);image(ax,response,title,cmap='RdBu_r',vmin=-.3,vmax=.3)
gelux=np.linspace(-3,3,250);geluy=np.array([x*.5*(1+math.erf(x/math.sqrt(2))) for x in gelux]);axs[3].plot(gelux,geluy,color=teal);axs[3].plot(gelux,gelux,'--',color='#999',lw=1);axs[3].set_title('GELU versus identity');axs[3].set_xlabel('Input');axs[3].set_ylabel('Output');save(fig,'03-convolution-and-nonlinearity')

y,x=np.indices((24,24));a=sum(np.exp(-((x-cx)**2+(y-cy)**2)/1.6) for cx,cy in [(7,7),(15,9),(10,16),(18,16)])*.4
w=np.outer(np.hanning(24),np.hanning(24));pa=(a-a.mean())*w
fig,axs=plt.subplots(1,4,figsize=(12,3.4));image(axs[0],a+.5,'24 × 24 toy crop',vmin=0,vmax=1);image(axs[1],a-a.mean(),'Subtract its own mean',cmap='RdBu_r',vmin=-.4,vmax=.4);image(axs[2],w,'Hann weights',vmin=0,vmax=1,cmap='viridis');image(axs[3],pa,'Multiply: windowed patch',cmap='RdBu_r',vmin=-.4,vmax=.4);save(fig,'04-centering-and-window')

def corr(a,b):
    aa=(a-a.mean())*w;bb=(b-b.mean())*w
    norm=np.sqrt((aa*aa).sum()*(bb*bb).sum());fft=np.fft.irfft2(np.fft.rfft2(aa,s=(32,32)).conj()*np.fft.rfft2(bb,s=(32,32)),s=(32,32))
    c=fft[np.arange(-4,5)%32][:,np.arange(-4,5)%32]/max(norm,1e-8)
    return c,norm/(norm+.01),aa,bb
def probability(c,tau=.04):
    z=(c-c.max())/tau;e=np.exp(z);return e/e.sum()
dotlater=shifted(a,2,1);examples=[('Distinct dots',a,dotlater),('Long vertical edge',(x>11)*.4,((x-2)>11)*.4),('Repeating stripes',.3*np.cos(2*np.pi*x/6),.3*np.cos(2*np.pi*(x-2)/6))]
fig,axs=plt.subplots(3,4,figsize=(12,8.6));saved=[]
for row,(label,earlier,later) in enumerate(examples):
    c,r,aa,bb=corr(earlier,later);p=probability(c);saved.append({'label':label,'correlation':c.tolist(),'probability':p.tolist(),'reliability':float(r)})
    image(axs[row,0],earlier+.5,label+' · earlier',vmin=0,vmax=1);image(axs[row,1],later+.5,'Later · +2 horizontal\n(+1 vertical only for dots)',vmin=0,vmax=1)
    for col,z,title,cmap,vmin,vmax in [(2,c,'Correlation over candidate shifts','RdBu_r',-1,1),(3,p,'Softmax mass · temperature 0.04','magma',0,1)]:
        im=axs[row,col].imshow(z,origin='lower',extent=(-4.5,4.5,-4.5,4.5),vmin=vmin,vmax=vmax,cmap=cmap);axs[row,col].set_title(title,fontsize=10);axs[row,col].set_xticks([-4,0,4]);axs[row,col].set_yticks([-4,0,4]);axs[row,col].set_xlabel('Horizontal shift');axs[row,col].set_ylabel('Vertical shift (+ = image down)');fig.colorbar(im,ax=axs[row,col],fraction=.04,pad=.04)
save(fig,'05-correlation-is-not-an-image-map')
c,r,aa,bb=corr(a,dotlater); pad_a=np.pad(aa,((0,8),(0,8)));pad_b=np.pad(bb,((0,8),(0,8)))
direct=np.array([[np.sum(pad_a*np.roll(pad_b,(-dy,-dx),(0,1))) for dx in range(-4,5)] for dy in range(-4,5)])/np.sqrt((aa*aa).sum()*(bb*bb).sum())
checks['fft_direct_max_abs_error']=float(np.abs(c-direct).max());checks['dot_peak_dx_dy']=[int(np.argmax(c)%9-4),int(np.argmax(c)//9-4)]
assert checks['fft_direct_max_abs_error']<1e-12 and checks['dot_peak_dx_dy']==[2,1]

fig,axs=plt.subplots(1,3,figsize=(12,3.8));amps=np.logspace(-4,0,100);reliabilities=[];entropies=[]
for amp in amps:
    costs,rel,_,_=corr(a*amp,dotlater*amp);mass=probability(costs);reliabilities.append(rel);entropies.append(-np.sum(mass*np.log(np.maximum(mass,1e-300))))
axs[0].semilogx(amps,reliabilities,color=teal);axs[0].set(title='Energy reliability\nrises with contrast',xlabel='Amplitude multiplier',ylabel='Reliability r',ylim=(-.03,1.03))
axs[1].semilogx(amps,entropies,color=violet);axs[1].set(title='Population spread\nis a different quantity',xlabel='Amplitude multiplier',ylabel='Entropy (nats)')
positions=np.arange(-4,5);p1=np.zeros(9);p1[1]=p1[7]=.5;p2=np.zeros(9);p2[4]=1
axs[2].bar(positions-.18,p1,.36,color=coral,label='Two opposing alternatives');axs[2].bar(positions+.18,p2,.36,color=teal,label='Stationary alternative');axs[2].set(title='Both mean displacements\nequal zero',xlabel='Horizontal displacement',ylabel='Mass');axs[2].legend(fontsize=8);save(fig,'06-confidence-and-ambiguity')
checks['opposing_and_stationary_means']=[float(p1@positions),float(p2@positions)]

prior=.5+a;pred=shifted(prior,2,0);stable=shifted(prior,2,0);changed=shifted(prior,0,2)
fig,axs=plt.subplots(2,4,figsize=(11,6));mse=[]
for row,(actual,label) in enumerate([(stable,'Continues right'),(changed,'Turns downward')]):
    for col,im,title in [(0,prior,'Previous image'),(1,pred,'Prediction: +2 right'),(2,actual,label),(3,(actual-pred)**2,'Squared pixel error')]:image(axs[row,col],im,title,cmap='magma' if col==3 else 'gray',vmin=0 if col==3 else .5,vmax=.16 if col==3 else .9)
    mse.append(float(((actual-pred)**2).mean()))
save(fig,'07-prediction-is-a-causal-test');checks['prescribed_flow_mse_stable_changed']=mse;assert mse[0]==0 and mse[1]>0

steps=50;pop=np.zeros((steps,81));pop[:16,42]=1;pop[16:,58]=1 # (2,0) then (0,2)
rel=np.ones(steps);reference=[];recent=[];strength=[];deltas={};traces={}
for reliability in [1.,.2]:
    z=np.ones(81)/81;ds=[];zs=[]
    for cur in pop:ds.append(abs(cur-z).sum());z=z+.1*reliability*(cur-z);zs.append(z[42])
    deltas[str(reliability)]=ds;traces[str(reliability)]=zs
for t in range(steps):
    ref=pop[max(0,t-15):max(0,t-7)];obs=pop[max(0,t-3):t+1];pr=ref.mean(0) if len(ref) else np.zeros(81);po=obs.mean(0)
    q=np.sqrt(len(ref)/8*len(obs)/4);reference.append(pr[42]);recent.append(po[42]);strength.append(q*abs(po-pr).sum())
fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
axs[0].step(range(steps),pop[:,42],where='post',color='#999',label='Current mass at rightward bin');axs[0].plot(traces['1.0'],color=teal,label='Trace: reliability 1');axs[0].plot(traces['0.2'],color=coral,label='Trace: reliability 0.2');axs[0].set_ylabel('Population mass');axs[0].legend(fontsize=9,ncol=2)
axs[1].plot(deltas['1.0'],color=teal,label='Mismatch to trace, reliability 1');axs[1].plot(deltas['0.2'],color=coral,label='Mismatch to trace, reliability 0.2');axs[1].set_ylabel('Absolute difference sum');axs[1].legend(fontsize=9)
axs[2].plot(reference,color=violet,label='Reference mass at rightward bin');axs[2].plot(recent,color=teal,label='Recent mass at rightward bin');axs[2].plot(strength,color=coral,label='Confidence × window difference');axs[2].set(xlabel='Frame index',ylabel='Mass / difference');axs[2].legend(fontsize=9)
for ax in axs:ax.axvline(16,color='#666',ls='--');ax.axvspan(0,7,color='#dde7e0',alpha=.45);ax.grid(alpha=.15)
axs[0].set_title('Prescribed population changes at frame 16 · initial trace uniform over 81 bins')
save(fig,'08-two-kinds-of-temporal-comparison');checks['seven_frame_window_difference_zero']=bool(np.all(np.array(strength)[:7]==0));checks['trace_mass_preserved']=bool(np.isclose(z.sum(),1))

fig,axs=plt.subplots(1,3,figsize=(11,3.6));before=np.array([[1.,0],[0,1.]]);after=before[::-1]
for ax,v,title in [(axs[0],before,'Before: site A / site B'),(axs[1],after,'After: both sites swap')]:
    image(ax,v,title,cmap='viridis',vmin=0,vmax=1);ax.set_yticks([0,1],['Site A','Site B']);ax.set_xticks([0,1],['Leftward','Rightward'])
axs[2].bar(['Pool first','Site A','Site B'],[abs(before.mean(0)-after.mean(0)).sum(),2,2],color=[violet,teal,coral]);axs[2].set(title='Compare locally\nbefore pooling',ylabel='Sum of absolute differences',ylim=(0,2.4));save(fig,'09-compare-before-pooling')
checks['pooling_hides_swap']=bool(np.array_equal(before.mean(0),after.mean(0)))

# Compare the teaching computations against the fixed CPU implementation.
sys.path.insert(0,str(ROOT.parents[2]));import torch
torch.set_num_threads(1)
from SecondPass.StabilizedAttentionKrauzlis.model import LocalMotionBank,causal_average,warp
movie=np.stack([.5+a,.5+dotlater]);canvas=np.full((1,2,1,100,100),.5,np.float32);canvas[0,:,0,33:57,53:77]=movie
with torch.no_grad():
    costs,reliability=LocalMotionBank()(torch.from_numpy(canvas));model_c=costs[0,1,0,46].numpy()
    # Crop index (row4,col6) is exactly this 24x24 region.
    checks['numpy_correlation_vs_model_cpu']=float(np.abs(model_c-c.flatten()).max())
    pops=torch.tensor(pop[None,:,None,None,:],dtype=torch.float32);rs=torch.ones((1,steps,1,1))
    rp,rc=causal_average(pops,rs,8,8);op,oc=causal_average(pops,rs,4,0)
    measures=((rp-op).abs().sum(-1)*(rc*oc).sqrt()).flatten().numpy()
    checks['window_example_vs_model_cpu']=float(np.abs(measures-strength).max())
    src=torch.tensor(prior[None,None],dtype=torch.float32);flow=torch.zeros(1,2,10,10);flow[:,0]=2
    checks['warp_example_vs_model_cpu']=float(np.abs(warp(src,flow).numpy()[0,0]-pred).max())
assert checks['numpy_correlation_vs_model_cpu']<2e-6
assert checks['window_example_vs_model_cpu']<2e-6
assert checks['warp_example_vs_model_cpu']<2e-6
source=ROOT.parents[2]/'SecondPass/StabilizedAttentionKrauzlis/model.py'
data={
    'seed':20261009,'scope':'Synthetic teaching examples with fixed numerical inputs and CPU implementation checks.',
    'spatial':{'rgb':rgb.tolist(),'statistics_block':rgb[40:50,60:70].tolist(),'motion_crop':rgb[33:57,53:77].tolist()},
    'color':{'rgb_patches':colors.tolist(),'centered_means':(colors-.5).tolist(),'checkerboard':contrast.tolist()},
    'convolution':{'image':toy.tolist(),'kernel_x':kx.tolist(),'kernel_y':ky.tolist(),'response_x':conv(toy-.5,kx).tolist(),'response_y':conv(toy-.5,ky).tolist(),'gelu_x':gelux.tolist(),'gelu_y':geluy.tolist()},
    'preparation':{'raw':(a+.5).tolist(),'centered':(a-a.mean()).tolist(),'hann':w.tolist(),'windowed':pa.tolist()},
    'correlation_examples':saved,
    'confidence':{'amplitudes':amps.tolist(),'reliabilities':reliabilities,'entropies':entropies,'positions':positions.tolist(),'opposing_population':p1.tolist(),'stationary_population':p2.tolist()},
    'prediction':{'prior':prior.tolist(),'prescribed_prediction':pred.tolist(),'stable_actual':stable.tolist(),'changed_actual':changed.tolist(),'changed_squared_error':((changed-pred)**2).tolist(),'mse':mse},
    'temporal':{'population':pop.tolist(),'trace_differences':deltas,'trace_mass_at_rightward_bin':traces,'reference_mass':reference,'recent_mass':recent,'window_difference':strength},
    'pooling':{'before':before.tolist(),'after':after.tolist()},'checks':checks}
for item,(_,earlier,later) in zip(data['correlation_examples'],examples):
    item.update(earlier=(earlier+.5).tolist(),later=(later+.5).tolist())
data_path=OUT/'simulation_data.json';data_path.write_text(json.dumps(data,indent=2),encoding='utf-8');files.append(data_path)
(ROOT/'research/pixels_simulations_verified.json').write_text(json.dumps({'status':'passed','checks':checks,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2),encoding='utf-8')
print(json.dumps({'figures':sum(p.suffix=='.png' for p in files),'checks':checks},indent=2))
