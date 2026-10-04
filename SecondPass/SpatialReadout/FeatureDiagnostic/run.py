"""Bounded frozen native D0 representation probes and diagnostic readouts.
No writes outside this directory; no main-model gradients or optimizer.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
import argparse, hashlib, json, signal, time, copy, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.linalg import eigh
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from threadpoolctl import threadpool_limits
from SecondPass.SpatialReadout.model import SpatialReadout
from SecondPass.TaskSuite.suite import task_classes
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream

OUT=Path(__file__).resolve().parent
CKPT=Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/terminal.pt')
EXPECTED='1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841'
TASKS=['orientation_cued','spatial_binding']
LAYERS=['early','late','gru','readout']
CENTERS=[(27,27),(73,27),(27,73),(73,73)]
ALPHAS=[.1,1.,10.,100.]
DEADLINE=None

def dump(name,obj):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,allow_nan=False)); q.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard(reserve=45):
    if DEADLINE is not None and time.time()>DEADLINE-reserve: raise TimeoutError('Nonrenewable budget/report reserve reached')
def log(**kw):
    kw['unix']=time.time()
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
    print(json.dumps(kw),flush=True)

def extract(model,images,task):
    """Reuse native encode_frame unchanged, capture first post-ReLU block."""
    early=[]
    def hook(module,args,y): early.append(y)
    handle=model.blocks[0].register_forward_hook(hook)
    frames=model.frames(images); states=None; hidden=None
    features={k:[] for k in LAYERS+['early_cue']}
    try:
        for t in range(frames.shape[1]):
            field,states=model.encode_frame(frames[:,t],states)
            hidden=model.spatial_gru(model.spatial_input(field),hidden)
            readout=model.readout(hidden.flatten(1)).relu()
            e=early.pop()
            # Spatial 2x2 pooling only. Preserve raw tiny-glyph regions separately.
            features['early'].append(F.avg_pool2d(e,2).cpu())
            glyph=[]
            for x,y in CENTERS:
                cx=int(round(x/2)); cy=int(round((y-18)/2))
                glyph.append(e[:,:,cy-3:cy+3,cx-3:cx+3])
            features['early_cue'].append(torch.stack(glyph,1).cpu())
            features['late'].append(field.cpu())
            features['gru'].append(hidden.cpu())
            features['readout'].append(readout.cpu())
        logits=model.heads[task](readout)
    finally: handle.remove()
    return logits,{k:torch.stack(v,1) for k,v in features.items()}

def axial_error(a,b): return np.abs((a-b+np.pi/2)%np.pi-np.pi/2)*180/np.pi

def relation_features(sample,probe,cue,sign):
    """All arguments predictions, never true target/angle/sign at evaluation."""
    s=sample/(np.linalg.norm(sample,axis=-1,keepdims=True)+1e-8)
    p=probe/(np.linalg.norm(probe,axis=-1,keepdims=True)+1e-8)
    dot=(s*p).sum(-1); cross=s[...,0]*p[...,1]-s[...,1]*p[...,0]
    angle=np.arctan2(cross,dot)/2
    per=np.stack([dot,cross,np.abs(angle),angle,angle*sign[:,None],np.abs(cross)],-1)
    weighted=(per*cue[:,:,None]).sum(1)
    return np.concatenate([weighted,per.reshape(len(s),-1),cue,sign[:,None]],1).astype('float32')

def local_feature(a,layer,t,loc):
    x=a[layer][:,t]
    if layer=='readout':return x.reshape(len(x),-1).astype('float32')
    side=x.shape[-1]
    cx=int(round(CENTERS[loc][0]/(4 if layer=='early' else 16)))
    cy=int(round(CENTERS[loc][1]/(4 if layer=='early' else 16)))
    radius=2 if layer=='early' else 1
    z=x[:,:,cy-radius:cy+radius+1,cx-radius:cx+radius+1]
    return z.reshape(len(z),-1).astype('float32')

def cue_feature(a,layer,t):
    if layer=='early':
        # Raw glyph ROI + coarse spatial map covers signed glyph AND ring query.
        z=torch.from_numpy(a[layer][:,t].astype('float32'))
        pool=F.adaptive_avg_pool2d(z,5).numpy().reshape(len(z),-1)
        return np.concatenate([a['early_cue'][:,t].reshape(len(z),-1),pool],1).astype('float32')
    return a[layer][:,t].reshape(len(a[layer]),-1).astype('float32')

def target_arrays(meta):
    return dict(label=np.array([m['label'] for m in meta]),location=np.array([m['target_location'] for m in meta]),sign=np.array([m.get('cue_sign',1) for m in meta]),sample=np.array([m['sample_angles_radians'] for m in meta]),probe=np.array([m['probe_angles_radians'] for m in meta]))

def ridge_grid(x,y,xv):
    """Train-only per-column standardization; exact dual ridge shared spectrum."""
    guard(); x=np.asarray(x,dtype='float64'); y=np.asarray(y,dtype='float64')
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-6]=1
    z=(x-mean)/scale; zv=(xv-mean)/scale; ym=y.mean(0); yc=y-ym
    # Divide kernel by feature width so alpha grid has comparable interpretation.
    dim=z.shape[1]; kernel=z@z.T/dim
    vals,vecs=eigh(kernel,check_finite=False); cross=vecs.T@yc
    for alpha in ALPHAS:
        coeff=vecs@(cross/(np.maximum(vals,0)[:,None]+alpha))
        w=z.T@coeff/dim
        state=dict(mean=mean.astype('float32'),scale=scale.astype('float32'),weight=w.astype('float32'),bias=ym.astype('float32'),alpha=alpha,feature_dim=dim,n_fit=len(x))
        yield state,(zv@w+ym).astype('float32')

def predict_ridge(state,x):return ((x-state['mean'])/state['scale'])@state['weight']+state['bias']
def softmax(x):
    e=np.exp(x-x.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
def metric(y,p):
    pred=(p>=.5).astype(int); cm=confusion_matrix(y,pred,labels=[0,1]); n=len(y);acc=float((pred==y).mean());zz=1.95996398454;d=1+zz*zz/n;mid=(acc+zz*zz/(2*n))/d;half=zz*np.sqrt(acc*(1-acc)/n+zz*zz/(4*n*n))/d
    return dict(n=n,accuracy=acc,balanced_accuracy=float(balanced_accuracy_score(y,pred)),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,confusion=cm.tolist(),accuracy_wilson95=[mid-half,mid+half])
def binary_ba(y,p):return float(balanced_accuracy_score(y,p>=.5))

def generate(model,task,split,n,seed):
    st=SpatialBatteryStream(seed,split); allmeta=[]; ys=[]; zs=[]; chunks={k:[] for k in LAYERS+['early_cue']}; parity=[]; hashes=[]
    for start in range(0,n,8):
        guard(400); x,y,meta=st.batch(min(8,n-start),task,{'delay':0})
        if start==0:
            xx,yy,mm=SpatialBatteryStream(seed,split).batch(min(8,n-start),task,{'delay':0})
            assert torch.equal(x,xx) and torch.equal(y,yy) and meta==mm
        for raster in x:hashes.append(hashlib.sha256(raster.numpy().tobytes()).hexdigest())
        with torch.inference_mode():
            xm=x.to('mps');z,feat=extract(model,xm,task)
            if start==0:
                direct=model(xm,task);err=float((z-direct).abs().max().cpu());assert err<=1e-6;parity.append(err)
        for k in chunks:chunks[k].append(feat[k].numpy().astype('float16'))
        ys.append(y.numpy());zs.append(z.cpu().numpy());allmeta.extend(meta)
        if start%128==0:log(stage='extract',task=task,split=split,done=start+len(x),total=n)
    a={k:np.concatenate(v) for k,v in chunks.items()};a['logits']=np.concatenate(zs);a['label']=np.concatenate(ys)
    p=OUT/'features'/f'{task}_{split}.npz';p.parent.mkdir(exist_ok=True);np.savez(p,**a)
    dump(f'features/{task}_{split}_metadata.json',dict(seed=seed,split=split,n=n,metadata=allmeta,raster_sha256=hashes,native_equality=True,wrapper_max_abs_errors=parity,feature_dtype='float16 storage; float32 fitting',feature_shapes={k:list(v.shape) for k,v in a.items()}))
    return dict(task=task,split=split,n=n,seed=seed,path=str(p),sha256=sha(p),parity=parity)

def load(task,split):
    with np.load(OUT/'features'/f'{task}_{split}.npz') as f:a={k:f[k] for k in f.files}
    meta=json.loads((OUT/'features'/f'{task}_{split}_metadata.json').read_text())['metadata']
    return a,target_arrays(meta)

def angle_summary(truth,pred,locs):
    err=axial_error(truth,pred);cu=np.arange(4)[None,:]==locs[:,None]
    out=dict(mean_degrees=float(err.mean()),median_degrees=float(np.median(err)),cued_mean_degrees=float(err[cu].mean()),uncued_mean_degrees=float(err[~cu].mean()),fraction_within_7_5_degrees=float((err<7.5).mean()),by_location=[])
    for j in range(4):
        mask=locs==j
        out['by_location'].append(dict(location=j,n=len(err),mean_degrees=float(err[:,j].mean()),cued_n=int(mask.sum()),cued_mean_degrees=float(err[mask,j].mean()),uncued_n=int((~mask).sum()),uncued_mean_degrees=float(err[~mask,j].mean())))
    return out

def fit_probes(task,train,targ,val,vt):
    """Fit and select without loading TEST. Models saved before final test."""
    T=train['early'].shape[1];cue_t=0 if task=='orientation_cued' else 3
    # No pre-query location probe for binding: randomized target is not yet present.
    stages=[('sample',2),('preprobe',T-2),('report',T-1)]
    if task=='orientation_cued': stages=[('sample',2),('report',T-1)] # preprobe == sample
    models={}; selections=[]
    for layer in LAYERS:
        for stage,t in stages:
            for content in ['sample','probe']:
                if content=='probe' and stage!='report':continue
                outputs=[]
                for loc in range(4):
                    x=local_feature(train,layer,t,loc);xv=local_feature(val,layer,t,loc)
                    theta=targ[content][:,loc];yv=vt[content][:,loc]
                    yy=np.stack([np.cos(2*theta),np.sin(2*theta)],1)
                    best=None
                    for state,p in ridge_grid(x,yy,xv):
                        loss=float(axial_error(yv,np.arctan2(p[:,1],p[:,0])/2).mean())
                        if best is None or loss<best[0]:best=(loss,state)
                    name=f'{layer}_{stage}_{content}_loc{loc}';models[name]=dict(kind='angle',layer=layer,t=t,loc=loc,content=content,stage=stage,state=best[1]);outputs.append(best[0])
                selections.append(dict(layer=layer,stage=stage,content=content,val_mean_angle_degrees=float(np.mean(outputs))))
            if task=='spatial_binding' and stage=='sample':continue
            for label,k in [('location',4)]+([('sign',2)] if task=='orientation_cued' else []):
                yy=targ[label] if label=='location' else (targ[label]>0).astype(int)
                vy=vt[label] if label=='location' else (vt[label]>0).astype(int)
                best=None
                for state,p in ridge_grid(cue_feature(train,layer,t),np.eye(k)[yy],cue_feature(val,layer,t)):
                    score=float(balanced_accuracy_score(vy,p.argmax(1)))
                    if best is None or score>best[0]:best=(score,state)
                name=f'{layer}_{stage}_{label}';models[name]=dict(kind='cue',layer=layer,t=t,label=label,stage=stage,state=best[1]);selections.append(dict(layer=layer,stage=stage,content=label,val_balanced_accuracy=best[0]))
        log(stage='probes_fit',task=task,layer=layer,models=len(models))
    torch.save(models,OUT/'models'/f'{task}_probes.pt');dump(f'{task}_probe_selection.json',selections)
    return models,selections

class RelationNet(nn.Module):
    """Shared local relation and soft cue routing, supervised ONLY by task label."""
    def __init__(self,d,c,h):
        super().__init__();self.enc=nn.Sequential(nn.Linear(d,h),nn.ReLU());self.cue=nn.Sequential(nn.Linear(c,16),nn.Tanh());self.gate=nn.Linear(16,1)
        self.rel=nn.Sequential(nn.Linear(4*h+16,32),nn.ReLU(),nn.Linear(32,1));self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,s,p,c):
        a,b=self.enc(s),self.enc(p);u=self.cue(c)
        r=self.rel(torch.cat([a,b,b-a,a*b,u],-1)).squeeze(-1)
        return (self.gate(u).squeeze(-1).softmax(-1)*r).sum(-1)+self.bias

def relation_inputs(a,task):
    s=np.stack([local_feature(a,'early',2,j) for j in range(4)],1)
    p=np.stack([local_feature(a,'early',a['early'].shape[1]-1,j) for j in range(4)],1)
    if task=='orientation_cued':c=a['early_cue'][:,0].reshape(len(s),4,-1).astype('float32')
    else:c=np.stack([local_feature(a,'early',3,j) for j in range(4)],1)
    return [s,p,c]

def fit_relation(task,a,y,v,vy,shuffle=False):
    raw=relation_inputs(a,task);rawv=relation_inputs(v,task)
    # Shared across positions and sample/probe: no metadata-based selection.
    pooled=np.concatenate(raw[:2],0);mu=pooled.mean((0,1));sd=pooled.std((0,1));sd[sd<1e-5]=1
    cm=raw[2].mean((0,1));cs=raw[2].std((0,1));cs[cs<1e-5]=1
    means=[mu,mu,cm];stds=[sd,sd,cs]
    xx=[torch.tensor((z-m)/s) for z,m,s in zip(raw,means,stds)];xv=[torch.tensor((z-m)/s) for z,m,s in zip(rawv,means,stds)]
    target=np.random.default_rng(7182).permutation(y) if shuffle else y
    yy=torch.tensor(target,dtype=torch.float32);vy=torch.tensor(vy)
    best=None;records=[]
    configs=[(16,1e-3),(32,1e-3),(32,1e-2)] if not shuffle else [(32,1e-3)]
    for h,wd in configs:
        guard(120);torch.manual_seed(4701);net=RelationNet(xx[0].shape[-1],xx[2].shape[-1],h);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=wd)
        for epoch in range(1,121):
            order=torch.randperm(len(y));net.train()
            for ids in order.split(128):
                opt.zero_grad();loss=F.binary_cross_entropy_with_logits(net(*[z[ids] for z in xx]),yy[ids]);loss.backward();opt.step()
            if epoch%10==0:
                guard(100);net.eval()
                with torch.no_grad():pv=net(*xv).sigmoid().numpy()
                ba=binary_ba(vy.numpy(),pv);records.append(dict(h=h,wd=wd,epoch=epoch,val_ba=ba,loss=float(loss)))
                if best is None or ba>best[0]:best=(ba,copy.deepcopy(net.state_dict()),h,wd,epoch)
        log(stage='label_only_comparator_fit',task=task,shuffle=shuffle,h=h,wd=wd,best_val_ba=best[0])
    state=dict(state=best[1],h=best[2],wd=best[3],epoch=best[4],val_ba=best[0],d=xx[0].shape[-1],c=xx[2].shape[-1],means=means,stds=stds,shuffled=shuffle,records=records)
    torch.save(state,OUT/'models'/f'{task}_relation'+('_shuffle' if shuffle else '')+'.pt') if False else None
    torch.save(state,OUT/'models'/f'{task}_relation_{"shuffle" if shuffle else "true"}.pt')
    return state

def predict_relation(state,a,task):
    net=RelationNet(state['d'],state['c'],state['h']);net.load_state_dict(state['state']);net.eval()
    xx=[torch.tensor((z-m)/s) for z,m,s in zip(relation_inputs(a,task),state['means'],state['stds'])]
    with torch.no_grad():return net(*xx).sigmoid().numpy()

def fit_structured(task,a,y,v,vy):
    """Auxiliary-supervised compositional diagnostic, NOT label-only comparator.
    Decoder FIT-A / calibration FIT-B disjoint. Selection on validation only.
    """
    mid=len(y['label'])//2;idx=np.arange(mid);cal=np.arange(mid,len(y['label']));T=a['early'].shape[1]
    dec={};predcal={};predval={}
    for content,t in [('sample',2),('probe',T-1)]:
        pcs=[];pvs=[]
        for j in range(4):
            x=local_feature(a,'early',t,j);xv=local_feature(v,'early',t,j);ang=y[content][:,j];target=np.c_[np.cos(2*ang),np.sin(2*ang)]
            best=None
            for state,p in ridge_grid(x[idx],target[idx],xv):
                score=float(axial_error(vy[content][:,j],np.arctan2(p[:,1],p[:,0])/2).mean())
                if best is None or score<best[0]:best=(score,state,p)
            dec[f'{content}_{j}']=best[1];pcs.append(predict_ridge(best[1],x[cal]));pvs.append(best[2])
        predcal[content]=np.stack(pcs,1);predval[content]=np.stack(pvs,1)
    ct=0 if task=='orientation_cued' else 3
    for label,k in [('location',4)]+([('sign',2)] if task=='orientation_cued' else []):
        target=y[label] if label=='location' else (y[label]>0).astype(int);vtarget=vy[label] if label=='location' else (vy[label]>0).astype(int)
        x=cue_feature(a,'early',ct);xv=cue_feature(v,'early',ct);best=None
        for state,p in ridge_grid(x[idx],np.eye(k)[target[idx]],xv):
            score=float(balanced_accuracy_score(vtarget,p.argmax(1)))
            if best is None or score>best[0]:best=(score,state,p)
        dec[label]=best[1];predcal[label]=predict_ridge(best[1],x[cal]);predval[label]=best[2]
    def make(pred):
        cue=np.eye(4)[pred['location'].argmax(1)]
        sign=2*pred['sign'].argmax(1)-1 if 'sign' in pred else np.ones(len(cue))
        return relation_features(pred['sample'],pred['probe'],cue,sign)
    fc,fv=make(predcal),make(predval);best=None
    for state,p in ridge_grid(fc,y['label'][cal,None],fv):
        score=binary_ba(vy['label'],p[:,0])
        if best is None or score>best[0]:best=(score,state)
    out=dict(decoders=dec,calibrator=best[1],val_ba=best[0],decoder_fit_indices=idx.tolist(),calibrator_fit_indices=cal.tolist(),cue_time=ct)
    torch.save(out,OUT/'models'/f'{task}_structured.pt');return out

def predict_structured(state,a,task):
    pcs=[];T=a['early'].shape[1]
    for content,t in [('sample',2),('probe',T-1)]:pcs.append(np.stack([predict_ridge(state['decoders'][f'{content}_{j}'],local_feature(a,'early',t,j)) for j in range(4)],1))
    x=cue_feature(a,'early',state['cue_time']);cue=np.eye(4)[predict_ridge(state['decoders']['location'],x).argmax(1)]
    sign=2*predict_ridge(state['decoders']['sign'],x).argmax(1)-1 if 'sign' in state['decoders'] else np.ones(len(x))
    return predict_ridge(state['calibrator'],relation_features(*pcs,cue,sign))[:,0]

def evaluate(task,models,rel,shuffled,structured,test,tt):
    # Called once after every selection is frozen and selection receipt persisted.
    summary=[];predictions={};groups={}
    for name,m in models.items():
        if m['kind']=='angle':
            p=predict_ridge(m['state'],local_feature(test,m['layer'],m['t'],m['loc']));angle=np.arctan2(p[:,1],p[:,0])/2
            predictions[name]=angle;key=(m['layer'],m['stage'],m['content']);groups.setdefault(key,{})[m['loc']]=angle
        else:
            p=predict_ridge(m['state'],cue_feature(test,m['layer'],m['t']));yp=p.argmax(1);target=tt[m['label']] if m['label']=='location' else (tt[m['label']]>0).astype(int)
            predictions[name]=p;row=dict(layer=m['layer'],stage=m['stage'],content=m['label'],n=len(target),accuracy=float((yp==target).mean()),balanced_accuracy=float(balanced_accuracy_score(target,yp)),class_counts=np.bincount(target).tolist(),confusion=confusion_matrix(target,yp).tolist(),label_strata=[])
            for label in [0,1]:
                mask=tt['label']==label;row['label_strata'].append(dict(task_label=label,n=int(mask.sum()),accuracy=float((yp[mask]==target[mask]).mean())))
            summary.append(row)
    for (layer,stage,content),p in groups.items():summary.append(dict(layer=layer,stage=stage,content=content,**angle_summary(tt[content],np.stack([p[j] for j in range(4)],1),tt['location'])))
    label=tt['label'];deployed=softmax(test['logits'])[:,1]
    ps=dict(deployed=deployed,label_prior=np.full(len(label),.5),label_only_relation=predict_relation(rel,test,task),shuffled_label_relation=predict_relation(shuffled,test,task),aux_supervised_structured=predict_structured(structured,test,task))
    results={};rng=np.random.default_rng(9901);boot=rng.integers(0,len(label),(2000,len(label)))
    for name,p in ps.items():
        r=metric(label,p);dif=((p>=.5)==label).astype(float)-((deployed>=.5)==label)
        r['paired_accuracy_gain_vs_deployed']=float(dif.mean());r['paired_gain_bootstrap95']=np.quantile(dif[boot].mean(1),[.025,.975]).tolist();strata=[]
        for key in ['location','sign','label']:
            if key=='sign' and task!='orientation_cued':continue
            for level in np.unique(tt[key]):
                mask=tt[key]==level;z=dict(stratum=key,value=int(level),n=int(mask.sum()),accuracy=float(((p[mask]>=.5)==label[mask]).mean()))
                if len(np.unique(label[mask]))==2:z['balanced_accuracy']=binary_ba(label[mask],p[mask])
                strata.append(z)
        r['strata']=strata;results[name]=r;predictions['comparator_'+name]=p
    predictions.update(label=label,location=tt['location'],sign=tt['sign'],sample_truth=tt['sample'],probe_truth=tt['probe'])
    np.savez(OUT/'predictions'/f'{task}_test.npz',**predictions);dump(f'{task}_probe_results.json',summary);dump(f'{task}_comparator_results.json',results)
    return dict(task=task,probes=summary,comparators=results)

def report(results,receipt,manifest):
    lines=['# Frozen feature and comparator diagnostic','', '**Native D0 only; terminal6760 unchanged.** Independent 1024/256/512 train/validation/test base episodes per task unless counts below state otherwise.','', '|Task|Readout/control|Test BA|Test AUC|Paired accuracy gain vs deployed (95% interval)|','|---|---|---:|---:|---|']
    for r in results:
        for name,m in r['comparators'].items():lines.append(f"|{r['task']}|{name}|{m['balanced_accuracy']:.4f}|{m['auc']:.4f}|{m['paired_accuracy_gain_vs_deployed']:+.4f} {m['paired_gain_bootstrap95']}|")
    lines+=['','## Feature accessibility','Mean axial error in degrees (uniform-angle uninformed reference 45°, not an empirical control). Each local probe predicts cos(2θ), sin(2θ); all four locations fit separately. Sample-at-report is not clean retention: raw stack3 still contains a sample.','', '|Task|Layer|Time|Content|Cued error°|Uncued error°|All error°|','|---|---|---|---|---:|---:|---:|']
    for r in results:
        for p in r['probes']:
            if 'mean_degrees' in p:lines.append(f"|{r['task']}|{p['layer']}|{p['stage']}|{p['content']}|{p['cued_mean_degrees']:.2f}|{p['uncued_mean_degrees']:.2f}|{p['mean_degrees']:.2f}|")
    lines+=['','|Task|Layer|Time|Cue content|Balanced accuracy|','|---|---|---|---|---:|']
    for r in results:
        for p in r['probes']:
            if 'balanced_accuracy' in p:lines.append(f"|{r['task']}|{p['layer']}|{p['stage']}|{p['content']}|{p['balanced_accuracy']:.4f}|")
    lines+=['','## Scientific scope and fit discipline',
    '- Signed orientation reports positive iff the selected rotation agrees with the instruction sign. Actual rotations are 0/±15/±30/±45°. Error relative to the smallest nonzero change is important: ≥7.5° is already half that separation. Binding swaps exactly one pair; nonzero axial differences are 45° or 90°. Per-location cued/uncued errors and fraction within7.5° are in probe_results.json; no scalar latent-level regression substitutes for angles.',
    '- Cue location/sign are balanced independently of task label by the native queues. Cue tables include per-task-label accuracy and full class counts/confusions, preventing a label-prior decoder being misrepresented as cue decoding. Binding sign is not defined and is not probed; target location before the retrocue is not probed.',
    '- Early visual means first learned Conv/GN/ReLU output32×50×50, stored as local2×2 means32×25×25 plus four unpooled32×6×6 tiny-glyph ROIs. The final visual field is160×7×7 (CNN+KDA output), ConvGRU64×7×7, and post-ReLU readout256. Large maps preserve spatial positions; all local orientation probes use fixed5×5 early or3×3 late/GRU crops, without target-based selection. Readout probes receive all256 units. Cue early probes use all four unpooled glyph ROIs plus fixed5×5 global pooling. Other cue probes flatten the full field.',
    '- Fixed geometry, known task identity and known sample/query/report times are diagnostic privileges. All locations are decoded; none is selected with true target metadata as comparator input. Features are float16 on disk, promoted to float32, while exact dual ridge algebra is float64. A negative probe does not prove information was erased; pooling, linearity, small sample size and independent channel scaling may limit detection.',
    '- Ridge fits standardize using TRAIN only; regularization α∈{0.1,1,10,100} after kernel/feature-width normalization is selected only on validation angular error or cue balanced accuracy. No PCA. Same base episode groups all locations/times. Seeds, rasters hashes, native metadata, fitted scaling/weights and predictions are saved.',
    '- Primary label-only comparator: shared local sample/probe encoder16/32 units, ordered values/difference/product relation32 units, learned16-unit cue representation and softmax routing across all four locations. Receives only early learned activations at fixed times/ROIs; no true cue/sign/angle auxiliary loss. AdamW lr.002, batch128,120 epochs; hidden/weight-decay and 10-epoch checkpoints selected by validation BA. Shuffled-label control uses same feature pipeline and one32-unit configuration; its train labels are independently permuted while validation uses real labels. Thus its tuning opportunity is smaller than the primary grid.',
    '- Auxiliary-supervised structured diagnostic is DISTINCT from the label-only comparator. First half of TRAIN fits early-feature angle/cue decoders with auxiliary metadata targets; second half fits a ridge task-label readout from predicted cue and circular sample/probe relationships. Validation selects α. Calibration examples were not used to fit those decoders. Operational TEST inputs are exclusively predicted quantities. True target/sign/angles never enter the operational score. This diagnoses accessible parts plus an explicitly supplied circular-comparison inductive bias, not spontaneous native task learning.',
    '- TEST is loaded for scoring only after all model/regularization/checkpoint selections for BOTH tasks are serialized and hashed. Test metadata is generated during extraction, but no test outcome is used for selection. No retraining on validation; no test-driven retry. Label-prior rule predicts positive at0.5. Deployed native head predictions use the exact same episodes. Wilson accuracy intervals and paired whole-episode bootstrap gain intervals condition on these fitted probes/checkpoint, not model-training variability; balanced-label queue dependence and multiple descriptive probes are not corrected.',
    '- D0 has no inserted blanks. For signed orientation, sample frame2 is also preprobe; report3 stack contains sample1/sample2/probe. Binding sample2 precedes query3; report4 stack contains sample2/query3/probe. Preprobe3 is not a clean memory assay either. No long-delay, clean-retention, cue-validity benefit, attention allocation, lesion, microstimulation, motion or recognition conclusion is supported.',
    '', '## Verification and coverage',f'- Counts: {json.dumps(manifest["counts"])} per task. Completed tasks: {len(results)}/2.',f'- Exact wrapper parity maximum: {receipt["wrapper_max_abs_error"]}. Main state tensors bitwise unchanged: {receipt["state_unchanged"]}; main-model optimizer updates:0.',f'- Checkpoint/source hashes unchanged: {receipt["hashes_unchanged"]}. CPU cap2, interop1, one MPS worker. Elapsed through verification/report: see completion.json and budget.json; immutable1800-second budget.',
    '- Artifacts: run.py; features/*.npz and *_metadata.json; models/*.pt; predictions/*_test.npz; *_probe_selection.json; *_probe_results.json; *_comparator_results.json; selection_frozen.json; verification.json; budget.json; completion.json; progress.jsonl. All are local to FeatureDiagnostic.',
    '', '## Interpretation boundary / next decision',
    'Positive cue/orientation decoding establishes accessibility to these independent readouts, not use by the deployed network or a causal locus. A stronger structured diagnostic than deployed performance would favor a failure of usable combination/readout over complete absence of component information, within these privileged fixed geometry/timing inputs. Poor label-only comparator performance is inconclusive at this small fitting budget. Do not infer a proven architecture fix, generic learning impossibility or long-delay retention failure. Compare the measured layer/time error progression and the two explicitly different comparator supervision regimes before choosing a next experiment.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    global DEADLINE
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args();assert args.run
    torch.set_num_threads(2);torch.set_num_interop_threads(1);threadpool_limits(2)
    for d in ['features','models','predictions']:(OUT/d).mkdir(exist_ok=True)
    assert not (OUT/'budget.json').exists(),'Immutable run budget already exists; no renewal'
    assert sha(CKPT)==EXPECTED
    cp=torch.load(CKPT,map_location='cpu');model=SpatialReadout(task_classes());model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    sources=['SecondPass/SpatialReadout/model.py','WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py','WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/stimuli.py','SecondPass/TaskSuite/suite.py','SecondPass/TaskSuite/catalog.json']
    hashes={p:sha(p) for p in sources}
    start=time.time();DEADLINE=start+1800
    dump('budget.json',dict(start_unix=start,deadline_unix=DEADLINE,cap_seconds=1800,renewable=False,pid=os.getpid(),cpu_threads=2,interop_threads=1,first_accelerator_operation='model.to(mps)',checkpoint_sha256=EXPECTED))
    def alarm(*unused):
        dump('cap_reached.json',dict(unix=time.time(),elapsed=time.time()-start));os._exit(124)
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,1800)
    model=model.to('mps');pstart=time.time()
    with torch.inference_mode():
        for ti,task in enumerate(TASKS):
            x,y,m=SpatialBatteryStream(879900+ti,'train').batch(8,task,{'delay':0});z,f=extract(model,x.to('mps'),task)
            zz=model(x.to('mps'),task);assert float((z-zz).abs().max().cpu())<=1e-6
    torch.mps.synchronize();profile=time.time()-pstart
    counts=dict(train=1024,val=256,test=512)
    # Profile covers16 episodes +16 parity forwards. Conservative extraction allowance.
    projected=profile/16*sum(counts.values())*2
    if projected>900:counts=dict(train=512,val=128,test=256)
    manifest=dict(counts=counts,profile_seconds=profile,profile_native_episodes=16,profile_includes_direct_parity=True,projected_full_counts_extraction_seconds=projected,policy='Reduce once to512/128/256 only if projected extraction exceeds900s. No outcome-dependent count changes.',seeds={task:{sp:88000000+100000*ti+1000*si for si,sp in enumerate(counts)} for ti,task in enumerate(TASKS)},features=[])
    dump('manifest.json',manifest);log(stage='profile',profile_seconds=profile,counts=counts,projected_full_extraction=projected)
    for task in TASKS:
        for split,n in counts.items():manifest['features'].append(generate(model,task,split,n,manifest['seeds'][task][split]));dump('manifest.json',manifest)
    torch.mps.synchronize();unchanged=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in initial.items());assert unchanged
    del model,cp,initial;torch.mps.empty_cache()
    fitted={};selections={}
    for task in TASKS:
        a,y=load(task,'train');v,vy=load(task,'val');models,sel=fit_probes(task,a,y,v,vy)
        rel=fit_relation(task,a,y['label'],v,vy['label']);shuffled=fit_relation(task,a,y['label'],v,vy['label'],True);structured=fit_structured(task,a,y,v,vy)
        fitted[task]=(models,rel,shuffled,structured);selections[task]=dict(probes=sel,label_only_val_ba=rel['val_ba'],shuffle_val_ba=shuffled['val_ba'],structured_val_ba=structured['val_ba'])
        del a,v;log(stage='task_fits_complete',task=task)
    frozen=dict(unix=time.time(),elapsed=time.time()-start,selections=selections,model_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'models').glob('*.pt')},test_scoring_started=False)
    dump('selection_frozen.json',frozen);results=[]
    for task in TASKS:
        guard(60);a,y=load(task,'test');r=evaluate(task,*fitted[task],a,y);results.append(r);del a
        log(stage='test_complete',task=task,comparators={k:v['balanced_accuracy'] for k,v in r['comparators'].items()})
    immutable=sha(CKPT)==EXPECTED and all(sha(p)==h for p,h in hashes.items());assert immutable
    ids=[];rasters=[]
    for rec in manifest['features']:
        m=json.loads((OUT/'features'/f'{rec["task"]}_{rec["split"]}_metadata.json').read_text());ids.extend(x['trial_id'] for x in m['metadata']);rasters.extend(m['raster_sha256'])
    assert len(set(ids))==len(ids) and len(set(rasters))==len(rasters)
    receipt=dict(checkpoint=str(CKPT),checkpoint_sha256=EXPECTED,step=6760,state_unchanged=unchanged,hashes_unchanged=immutable,source_hashes=hashes,wrapper_max_abs_error=max(e for r in manifest['features'] for e in r['parity']),all_base_ids_unique=True,all_base_rasters_unique=True,n_base_episodes=len(ids),main_optimizer_updates=0,elapsed_seconds=time.time()-start,model_selection_precedes_test=True)
    dump('verification.json',receipt);dump('results.json',results);report(results,receipt,manifest)
    dump('completion.json',dict(status='complete',elapsed_seconds=time.time()-start,finished_unix=time.time(),within_cap=time.time()<DEADLINE,tasks_completed=len(results),test_episodes_per_task=counts['test'],pid=os.getpid()))
    assert time.time()<DEADLINE;signal.setitimer(signal.ITIMER_REAL,0);log(stage='complete',elapsed_seconds=time.time()-start)

if __name__=='__main__':main()
