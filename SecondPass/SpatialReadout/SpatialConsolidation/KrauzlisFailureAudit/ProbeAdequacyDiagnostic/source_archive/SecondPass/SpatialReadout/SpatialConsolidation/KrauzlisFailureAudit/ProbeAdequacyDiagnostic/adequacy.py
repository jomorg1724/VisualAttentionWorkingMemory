"""Input-only finite ridge analysis; no deployed optimizer or state updates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,signal
from pathlib import Path
import numpy as np
OUT=Path(__file__).parent
ROOT=OUT.parents[4]
sys.path.insert(0,str(OUT.parent/'UpstreamMotionDiagnostic'))
from upstream import d,px,pixel_summary,circular_change,flow_angles,torch

def kernel(x,z,degree):
    q=x@z.T/x.shape[1]
    return q if degree==1 else q+q*q

def fit_candidates(x,y,dfs,degree):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
    z=(x-mean)/scale;K=kernel(z,z,degree);km=K.mean(0);grand=K.mean();K=K-km[None,:]-km[:,None]+grand
    eig,q=np.linalg.eigh(K);eig=np.maximum(eig,0);ym=y.mean(0);rhs=q.T@(y-ym);out=[]
    for df in dfs:
        lo,hi=1e-14,max(1.,eig.max()*len(eig))
        for _ in range(80):
            a=(lo+hi)/2
            if np.sum(eig/(eig+a))>df:lo=a
            else:hi=a
        alpha=(lo+hi)/2;coef=q@(rhs/(eig[:,None]+alpha))
        residual=np.linalg.norm((K+alpha*np.eye(len(K)))@coef-(y-ym))/max(1e-15,np.linalg.norm(y-ym))
        out.append(dict(mean=mean,scale=scale,train_z=z,kernel_mean=km,kernel_grand=grand,coef=coef,intercept=ym,alpha=alpha,degree=degree,effective_df=float(np.sum(eig/(eig+alpha))),requested_df=df,normal_residual=float(residual)))
    return out

def predict(fit,x):
    z=(x-fit['mean'])/fit['scale'];K=kernel(z,fit['train_z'],int(fit['degree']))
    return (K-K.mean(1)[:,None]-fit['kernel_mean'][None,:]+fit['kernel_grand'])@fit['coef']+fit['intercept']

def endpoint_features(model,x):
    """Two native stack3 endpoints, BEFORE first KDA. No metadata parameter."""
    b=x.shape[1]-17;frames=model.frames(x);features=[]
    with torch.inference_mode():
        for t in (b+7,b+15):
            h=model.blocks[1](model.blocks[0](frames[:,t]))
            features.append(torch.cat([h[:,:,12:15,c-1:c+2].flatten(1) for c in (5,20)],1).numpy()[0])
    return np.concatenate(features)

def raw_features(x):
    a=x[0].numpy();b=len(a)-17
    assert np.array_equal(a[:,0],a[:,1]) and np.array_equal(a[:,0],a[:,2])
    return np.concatenate([a[t-2:t+1,0,39:62,c-11:c+12].ravel()-.5 for t in (b+7,b+15) for c in (20,80)])

def representation(data,rep,mode,projections):
    x=data[rep].astype(np.float64)
    if mode=='projected':
        half=x.shape[1]//2;p=projections[rep]
        x=np.concatenate([x[:,:half]@p,x[:,half:]@p],1)
    return x

def init_models():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    old=json.loads((OUT.parent/'FrozenDiagnostic/identity.json').read_text())
    assert d.sha(d.CHECKPOINT)==old['checkpoint_sha256']
    assert d.sha(d.__file__)==old['extractor_sha256']
    for p,h in old['source_hashes'].items():assert d.sha(d.ROOT/p)==h and (d.ROOT/p).read_bytes()==(d.RUN.parent/'repo'/p).read_bytes()
    c=torch.load(d.CHECKPOINT,map_location='cpu',weights_only=False);assert c['state']['step']==2297
    classes={k.split('.')[1]:v.shape[0] for k,v in c['model'].items() if k.startswith('heads.') and k.endswith('.weight')}
    torch.manual_seed(674029);random=d.SpatialConsolidation(classes).eval().requires_grad_(False)
    # Independent construction; ONLY trained instance receives checkpoint tensors.
    trained=d.SpatialConsolidation(classes);trained.load_state_dict(c['model'],strict=True);trained.eval().requires_grad_(False)
    assert random.blocks[0][0].weight.data_ptr()!=trained.blocks[0][0].weight.data_ptr()
    assert not torch.equal(random.blocks[0][0].weight,trained.blocks[0][0].weight)
    old.update(random_seed=674029,random_state_sha256=d.state_hash(random),trained_state_sha256=d.state_hash(trained),random_checkpoint_loaded=False,random_architecture=str(random),analysis_source_sha256=d.sha(__file__),pixel_source_sha256=d.sha(px.__file__),upstream_source_sha256=d.sha(OUT.parent/'UpstreamMotionDiagnostic/upstream.py'))
    torch.save(random.state_dict(),OUT/'random_initialized_state.pt');d.dump(OUT/'identity.json',old)
    return trained,random,old

def profile():
    assert not (OUT/'budget.json').exists(),'nonrenewable cap already exists'
    start=time.time();deadline=start+1800;d.dump(OUT/'budget.json',dict(started=start,deadline=deadline,total_seconds=1800,renewable=False,includes='profile, source completion, extraction, fits, report, replay',threads=2,workers=1))
    trained,random,identity=init_models();results=[]
    for b in (12,20,28):
        t=time.time();s=d.SpatialBatteryStream(945031+b,'test');x,y,m,a=d.captured_batch(s,b)
        native=d.SpatialBatteryStream(945031+b,'test');xx,yy,mm=native.batch(1,d.TASK,dict(baseline_transitions=b))
        assert torch.equal(x,xx) and torch.equal(y,yy) and mm==m and s.state_dict()==native.state_dict()
        parity={}
        for name,model in [('trained',trained),('random',random)]:
            ex=d.Extractor(model);f,log=ex(x);ex.close()
            with torch.inference_mode():native_log=model(x,d.TASK).numpy()
            assert np.array_equal(log,native_log)
            desired=np.concatenate([f['cnn25',p][0] for p in ('baseline','post')]);actual=endpoint_features(model,x)
            assert np.array_equal(desired,actual);parity[name]=dict(endpoint_max_abs=0,hook_logits_max_abs=0)
        t=time.time();raw_features(x);endpoint_features(trained,x);endpoint_features(random,x);pixel_summary(x[0].numpy());elapsed=time.time()-t
        results.append(dict(b=b,feature_and_pixel_seconds=elapsed,renderer_parity=True,parity=parity))
    # Profile includes exact modest-size dual algebra but no scientific fit/test.
    rng=np.random.default_rng(31);t=time.time();fit_candidates(rng.normal(size=(600,2304)),rng.normal(size=(600,4)),[16.,64.,128.],2);fitsec=time.time()-t
    d.dump(OUT/'profile.json',dict(profiles=results,fit_seconds=fitsec,elapsed=time.time()-start))
    print(json.dumps(json.loads((OUT/'profile.json').read_text())),flush=True)

if __name__=='__main__':profile()
