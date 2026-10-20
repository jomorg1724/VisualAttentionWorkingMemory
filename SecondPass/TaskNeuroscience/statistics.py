"""Paired behavioral inference and independent-scene population decoding."""
import json, math, warnings
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,norm,spearmanr
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.decomposition import PCA
from .design import *

def interval(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);choice=p.argmax(-1);classes=p.shape[1];n=len(y)
    recalls=[float((choice[y==k]==k).mean()) for k in range(classes) if (y==k).any()]
    auc=[roc_auc_score(y==k,p[:,k]) for k in range(classes) if 0<(y==k).sum()<n]
    out=dict(n=n,accuracy=float((choice==y).mean()),accuracy_ci95=interval(int((choice==y).sum()),n),
        balanced_accuracy=float(np.mean(recalls)) if len(recalls)==classes else None,auc=float(np.mean(auc)) if auc else None,
        confusion=[[int(((y==i)&(choice==j)).sum()) for j in range(classes)] for i in range(classes)],
        brier=float(np.mean(np.sum((p-np.eye(classes)[y])**2,-1))),
        logloss=float(-np.log(np.clip(p[np.arange(n),y],1e-12,1)).mean()))
    if classes==2 and len(np.unique(y))==2:
        hit=((choice==1)&(y==1)).sum();fa=((choice==1)&(y==0)).sum();npos=(y==1).sum();nneg=n-npos
        h=(hit+.5)/(npos+1);f=(fa+.5)/(nneg+1)
        out.update(hit_rate=float(hit/npos),false_alarm_rate=float(fa/nneg),dprime=float(norm.ppf(h)-norm.ppf(f)),criterion=float(-.5*(norm.ppf(h)+norm.ppf(f))))
    return out

def source_clusters(metadata):
    """Cluster repeated real photographs; missing photo IDs are separate trials."""
    out=[]
    for i,m in enumerate(metadata):
        identity=next((m[k] for k in ('probe_source_id','base_id') if m.get(k) is not None),None)
        out.append('trial:'+str(i) if identity is None else 'image:'+str(identity))
    return np.array(out)

def paired(y,p,q,seed=916,clusters=None):
    a=p.argmax(-1)==y;b=q.argmax(-1)==y;d=b.astype(float)-a
    if clusters is not None and len(set(map(str,clusters)))<len(d):
        groups={}
        for key,value in zip(clusters,d):groups.setdefault(str(key),[]).append(value)
        sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(seed);ix=rng.integers(len(sums),size=(3000,len(sums)))
        boot=sums[ix].sum(1)/counts[ix].sum(1);bounds=np.quantile(boot,[.025,.975]).tolist()
        if not np.any(sums):
            bound=1-.025**(1/len(sums));bounds=[-bound,bound]
        null=(rng.choice([-1,1],size=(9999,len(sums)))*sums).sum(1)/len(d)
        pvalue=float((1+sum(abs(null)>=abs(d.mean())-1e-12))/10000)
        return dict(delta_accuracy=float(d.mean()),ci95=bounds,interval_method='paired source-image cluster bootstrap',
            gained=int((~a&b).sum()),lost=int((a&~b).sum()),p_value=pvalue,p_exact=None,test_method='9999 source-image cluster sign flips',clusters=len(sums),
            delta_true_probability=float((q[np.arange(len(y)),y]-p[np.arange(len(y)),y]).mean()))
    rng=np.random.default_rng(seed);boot=np.mean(d[rng.integers(len(d),size=(3000,len(d)))],axis=1)
    lost=int((a&~b).sum());gained=int((~a&b).sum());discordant=lost+gained
    ci=np.quantile(boot,[.025,.975]).tolist();method='paired trial bootstrap'
    if discordant==0:
        bound=1-.025**(1/len(y));ci=[-bound,bound];method='conservative exact upper bound on discordance'
    elif np.all(d==1):ci=interval(len(y),len(y));method='Wilson all-gain boundary'
    elif np.all(d==-1):ci=[-v for v in reversed(interval(len(y),len(y)))];method='Wilson all-loss boundary'
    exact=float(binomtest(gained,discordant,.5).pvalue) if discordant else 1.
    return dict(delta_accuracy=float(d.mean()),ci95=ci,interval_method=method,gained=gained,lost=lost,
        p_exact=exact,p_value=exact,test_method='exact discordant-pair binomial',
        delta_true_probability=float((q[np.arange(len(y)),y]-p[np.arange(len(y)),y]).mean()))

def holm(values):
    v=np.asarray(values);order=np.argsort(v);out=np.zeros(len(v));last=0
    for rank,i in enumerate(order):last=max(last,(len(v)-rank)*v[i]);out[i]=min(1,last)
    return out.tolist()

def bootstrap_equal_cells(data,reps=1500):
    if all(np.all(g) for g in data) or all(not np.any(g) for g in data):
        return interval(sum(int(g.sum()) for g in data),sum(len(g) for g in data))
    rng=np.random.default_rng(916);means=[]
    for good in data:
        means.append(good[rng.integers(len(good),size=(reps,len(good)))].mean(1))
    return np.quantile(np.mean(means,axis=0),[.025,.975]).tolist()

def behavioral(task):
    folder=OUT/task;native=[];good=[];latest=[];byfactor={};clusters={}
    for spec in TASKS[task]['conditions']:
        cell=spec['id'];a=np.load(folder/'native'/f'{cell}.npz');m=json.loads((folder/'native'/f'{cell}.json').read_text())
        row=dict(cell=cell,**metrics(a['y'],a['p']),factors=spec['kwargs']);native.append(row);good.append(a['p'].argmax(1)==a['y'])
        for i,meta in enumerate(m):
            if task=='memory_guided_search':stratum=(meta['family'],meta['similarity']);cluster=meta['trial_id']
            elif task=='motion_memory_search':stratum=(meta['spacing'],);cluster=meta.get('trial_id',i)
            elif task=='popout_search':stratum=(meta['feature'],meta['difference']);cluster=meta['trial_id']
            else:stratum=(cell,);cluster=meta.get('probe_source_id',meta.get('base_id',None)) or meta.get('suite_trial_id',i)
            clusters.setdefault(stratum,{}).setdefault(str(cluster),[]).append(float(good[-1][i]))
        q=np.load(folder/'native'/f'{cell}_latest.npz');assert np.array_equal(q['y'],a['y'][:len(q['y'])])
        cluster=source_clusters(m[:len(q['y'])])
        latest.append(dict(cell=cell,**paired(q['y'],a['p'][:len(q['y'])],q['p'],clusters=cluster)))
        # Both condition factors and trial metadata are retained for marginal curves.
        for field in TASKS[task].get('report_by',[]):
            for i,meta in enumerate(m):
                value=meta.get(field)
                if isinstance(value,(str,int,float,bool)):
                    key=(field,str(value));byfactor.setdefault(key,[]).append(bool(a['p'][i].argmax()==a['y'][i]))
        for field,value in spec['kwargs'].items():
            byfactor.setdefault(('condition_'+field,str(value)),[]).extend(good[-1].tolist())
    factors=[dict(factor=f,value=v,n=len(g),accuracy=float(np.mean(g)),uncertainty='Descriptive marginal; repeated-scene uncertainty is supplied by the matched timing analyses and task-level cluster bootstrap.') for (f,v),g in byfactor.items()]
    psych=[]
    for path in sorted((folder/'psychometric').glob('*.npz')):
        a=np.load(path);level,event=path.stem.split('_',1)
        psych.append(dict(strength=float(level),event=event,**metrics(a['y'],a['p']),report1=float(a['p'].argmax(1).mean()) if a['p'].shape[1]==2 else None))
    chron=[]
    for spec in TASKS[task]['conditions']:
        cell=spec['id'];a=np.load(folder/'native'/f'{cell}.npz');m=json.loads((folder/'native'/f'{cell}.json').read_text());p=a['temporal'];y=a['y']
        for threshold in THRESHOLDS:
            latency=[];early=[];decision=[];dur=[]
            for probs,label,meta in zip(p,y,m):
                begin=onset(task,meta,len(probs));q=probs[begin:];cross=np.flatnonzero(q.max(1)>=threshold)
                dur.append(len(q));early.append(bool((probs[:begin].max(1)>=threshold).any()))
                if len(cross):latency.append(int(cross[0])+1);decision.append(int(q[cross[0]].argmax())==label)
            chron.append(dict(cell=cell,threshold=threshold,n=len(y),crossings=len(latency),crossing_fraction=len(latency)/len(y),
                conditional_median_frames=float(np.median(latency)) if latency else None,
                conditional_accuracy=float(np.mean(decision)) if decision else None,
                premature_fraction=float(np.mean(early)),available_frames=int(np.median(dur)),
                survival=[float(1-sum(z<=t for z in latency)/len(y)) for t in range(1,max(dur)+1)]))
    rng=np.random.default_rng(916);boot=np.zeros(3000);total=sum(len(g) for g in good)
    for group in clusters.values():
        ss=np.array([sum(v) for v in group.values()]);nn=np.array([len(v) for v in group.values()]);ix=rng.integers(len(ss),size=(3000,len(ss)))
        boot+=(ss[ix].sum(1)/nn[ix].sum(1))*nn.sum()/total
    conf=np.quantile(boot,[.025,.975]).tolist()
    if all(np.all(g) for g in good) or all(not np.any(g) for g in good):
        boundary_n=sum(len(g) for g in clusters.values());conf=interval(boundary_n if all(np.all(g) for g in good) else 0,boundary_n)
    primary=dict(task=task,conditions=len(native),trials=sum(r['n'] for r in native),equal_cell_accuracy=float(np.mean([r['accuracy'] for r in native])),
        equal_cell_ci95=conf,cluster_count=sum(len(g) for g in clusters.values()),uncertainty='Stratified cluster bootstrap: matching procedural scenes retained together across timing conditions; photo repeats clustered by probe/base source ID within condition.',equal_cell_auc=float(np.mean([r['auc'] for r in native if r['auc'] is not None])) if any(r['auc'] is not None for r in native) else None,
        cells=native,factors=factors,psychometric=psych,chronometry=chron,latest_comparison=latest)
    write(folder/'behavior.json',primary);return primary

def causal(task):
    folder=OUT/task;rows=[];counter=[];subgroups=[]
    for path in sorted((folder/'interventions').glob('*.npz')):
        a=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());base=a['p'][:,0]
        cluster=source_clusters(meta['metadata'])
        for i,name in enumerate(meta['arms']):
            result=dict(cell=path.stem,arm=name,active_trials=int(a['active'][:,i].sum()),**metrics(a['y'],a['p'][:,i]),**paired(a['y'],base,a['p'][:,i],clusters=cluster))
            rows.append(result)
        groups={}
        for j,m in enumerate(meta['metadata']):
            group=m.get('event_type')
            if task.startswith('vda'):group='catch' if a['y'][j]==0 else 'cued' if m['cue_index']==m['change_index'] else 'uncued'
            if task=='orientation_cued':
                delta=m['rotations_degrees'][m['target_location']];group='unchanged' if delta==0 else 'aligned' if np.sign(delta)==m['cue_sign'] else 'opposite'
            if task=='image_recognition':group='seen' if a['y'][j] else 'novel'
            if group is not None:groups.setdefault(str(group),[]).append(j)
        for group,ix in groups.items():
            for i,name in enumerate(meta['arms']):
                subgroups.append(dict(cell=path.stem,group=group,arm=name,report_positive=float((a['p'][ix,i].argmax(1)==1).mean()),
                    **metrics(a['y'][ix],a['p'][ix,i]),**paired(a['y'][ix],base[ix],a['p'][ix,i],clusters=cluster[ix])))
    for path in sorted((folder/'counterfactual').glob('*.npz')):
        a=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
        for i,name in enumerate(meta['arms']):
            counter.append(dict(event=path.stem,arm=name,report_change=float((a['p'][:,i].argmax(1)==1).mean()),**metrics(a['y'],a['p'][:,i]),**paired(a['y'],a['p'][:,0],a['p'][:,i])))
    allrows=rows+counter
    adjusted=holm([r['p_value'] for r in allrows])
    for r,p in zip(allrows,adjusted):r['holm_within_task']=p
    for r,p in zip(subgroups,holm([r['p_value'] for r in subgroups])):r['holm_subgroup_family']=p
    result=dict(interventions=rows,counterfactual=counter,subgroups=subgroups);write(folder/'causal.json',result);return result

def prepare_features(a,b,seed):
    """One fixed training-only transform shared across time for valid generalization."""
    a=a.astype(np.float32).reshape(len(a),a.shape[1],-1);b=b.astype(np.float32).reshape(len(b),b.shape[1],-1)
    width=a.shape[-1];projected=False
    if width>256:
        rng=np.random.default_rng(seed);proj=rng.choice([-1.,1.],size=(width,256)).astype(np.float32)/np.sqrt(256)
        a=a@proj;b=b@proj;projected=True
    mu=a.mean((0,1));scale=np.maximum(a.std((0,1)),.05);a=(a-mu)/scale;b=(b-mu)/scale
    pca=PCA(n_components=min(32,a.shape[-1],len(a)-1),svd_solver='randomized',random_state=seed)
    aa=pca.fit_transform(a.reshape(-1,a.shape[-1])).reshape(len(a),a.shape[1],-1);bb=pca.transform(b.reshape(-1,b.shape[-1])).reshape(len(b),b.shape[1],-1)
    std=np.maximum(aa.std((0,1)),.05);aa/=std;bb/=std
    return aa,bb,dict(original_width=width,random_projection_to_256=projected,retained_components=pca.n_components_,explained_fraction=float(pca.explained_variance_ratio_.sum())),pca

def ridge(x,y,alpha=10.):
    mu=x.mean(0);ym=y.mean(0);xc=x-mu
    w=np.linalg.solve(xc.T@xc+alpha*np.eye(x.shape[1]),xc.T@(y-ym))
    return w,ym-mu@w

def balanced(y,p):
    labs=np.unique(y);return float(np.mean([(p[y==c]==c).mean() for c in labs]))

def classify(x,xt,y,yt,classes,cross=False):
    target=np.eye(classes)[y];t=x.shape[1];scores=np.zeros((t,t if cross else 1));auc=np.zeros_like(scores);predictions=[]
    for i in range(t):
        w,b=ridge(x[:,i],target)
        times=range(t) if cross else [i];pp=[]
        for j,k in enumerate(times):
            pred=xt[:,k]@w+b;pp.append(pred.argmax(1));scores[i,j]=balanced(yt,pred.argmax(1))
            vals=[roc_auc_score(yt==c,pred[:,c]) for c in range(classes) if 0<(yt==c).sum()<len(yt)]
            auc[i,j]=np.mean(vals) if vals else .5
        predictions.append(pp[-1])
    return scores,auc,np.array(predictions)

def continuous_target(meta):
    if all('sample_direction' in m for m in meta):
        a=np.deg2rad([m['sample_direction'] for m in meta]);return np.stack((np.cos(a),np.sin(a)),1),'remembered_direction_cos_sin'
    if all('sample_angles_radians' in m and 'target_location' in m for m in meta):
        a=np.array([m['sample_angles_radians'][m['target_location']] for m in meta]);return np.stack((np.cos(2*a),np.sin(2*a)),1),'remembered_orientation_cos2_sin2'
    return None,None

def neural(task):
    folder=OUT/task;out=folder/'population';out.mkdir(exist_ok=True)
    summary=[]
    for ci,cell in enumerate(neural_cells(task)):
        if (out/f'{cell}.json').exists():summary.append(json.loads((out/f'{cell}.json').read_text()));continue
        a=np.load(folder/'neural'/f'{cell}_val.npz');b=np.load(folder/'neural'/f'{cell}_test.npz')
        am=json.loads((folder/'neural'/f'{cell}_val.json').read_text());bm=json.loads((folder/'neural'/f'{cell}_test.json').read_text())
        y=a['y'];yt=b['y'];classes=TASKS[task]['classes'];record=dict(cell=cell,training_trials=len(y),test_trials=len(yt),regions={})
        arrays={};ty,tn=continuous_target(am);tyt,_=continuous_target(bm)
        for ri,region in enumerate(('appearance','fusion','spatial','cls','comparison','attention')):
            raw_a=a[region];raw_b=b[region]
            if not np.any(raw_a):record['regions'][region]=dict(constant_zero=True);continue
            x,xt,transform,pca=prepare_features(raw_a,raw_b,1701+ri)
            cross=region in ('cls','spatial')
            ba,auc,preds=classify(x,xt,y,yt,classes,cross=cross)
            diag=np.diag(ba) if cross else ba[:,0];ad=np.diag(auc) if cross else auc[:,0]
            arrays[region+'_ba']=ba;arrays[region+'_auc']=auc;arrays[region+'_diagonal']=diag
            # Pre-specified final-time permutation control (same projection, no test fitting).
            rng=np.random.default_rng(803+ri);null=[]
            for _ in range(199):
                w,bias0=ridge(x[:,-1],np.eye(classes)[rng.permutation(y)])
                null.append(balanced(yt,(xt[:,-1]@w+bias0).argmax(1)))
            rr=dict(**transform,final_balanced_accuracy=float(diag[-1]),final_auc=float(ad[-1]),
                final_permutation_p=float((1+sum(z>=diag[-1] for z in null))/(len(null)+1)),null95=float(np.quantile(null,.95)),
                peak_diagonal=float(max(diag)),diagonal=diag.tolist(),diagonal_auc=ad.tolist())
            # Content reconstruction is compared against training-mean prediction.
            target=a['content'];testtarget=b['content'];den=((testtarget-target.mean(0))**2).sum()
            r2=[]
            for t in range(x.shape[1]):
                w,bias0=ridge(x[:,t],target);pred=xt[:,t]@w+bias0
                r2.append(float(1-((pred-testtarget)**2).sum()/den) if den>1e-10 else None)
            rr['sample_content_r2']=r2
            if ty is not None:
                den=((tyt-ty.mean(0))**2).sum();vals=[]
                for t in range(x.shape[1]):
                    w,bias0=ridge(x[:,t],ty);vals.append(float(1-((xt[:,t]@w+bias0-tyt)**2).sum()/max(den,1e-12)))
                rr[tn+'_r2']=vals
            # Decode the cue separately from the final answer when it is available.
            def loc(m):return m.get('cue_index',m.get('target_location',m.get('target_index',-1)))
            q=np.array([loc(m) for m in am]);qt=np.array([loc(m) for m in bm])
            if len(np.unique(q))>1 and (q>=0).all() and (qt>=0).all():
                n=max(q.max(),qt.max())+1;qa,_,_=classify(x,xt,q,qt,n);rr['location_diagonal']=qa[:,0].tolist()
            choice=a['p'].argmax(1);choicet=b['p'].argmax(1)
            if len(np.unique(choice))>1:
                ca,_,_=classify(x,xt,choice,choicet,classes);rr['choice_diagonal']=ca[:,0].tolist()
            record['regions'][region]=rr
            if region in ('cls','spatial'):
                # Trajectories and dimensionality use a common training-derived coordinate system.
                arrays[region+'_trajectories']=np.stack([xt[yt==k,:,:3].mean(0) if (yt==k).any() else np.zeros_like(xt[0,:,:3]) for k in range(classes)])
                dim=[];drift=[];sep=[]
                for t in range(xt.shape[1]):
                    eig=np.linalg.eigvalsh(np.cov(xt[:,t].T));dim.append(float(eig.sum()**2/max((eig**2).sum(),1e-12)))
                    means=np.array([xt[yt==k,t].mean(0) for k in range(classes) if (yt==k).any()]);sep.append(float(np.var(means,axis=0).sum()))
                    drift.append(float(np.sqrt(np.mean((xt[:,t]-xt[:,max(0,t-1)])**2))))
                rr.update(participation_ratio=dim,state_step_rms=drift,class_centroid_variance=sep)
        save=dict(**arrays);np.savez_compressed(out/f'{cell}.npz',**save)
        write(out/f'{cell}.json',record);summary.append(record)
    write(folder/'neural_summary.json',summary);return summary

def global_correction():
    records=[]
    for task in TASKS:
        path=OUT/task/'causal.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        for r in data['interventions']:
            if r['arm'] in ('target_-4_evidence','target_+4_evidence','distractor_-4_evidence','distractor_+4_evidence'):
                records.append(dict(task=task,cell=r['cell'],arm=r['arm'],p=r.get('p_value',r['p_exact']),delta_accuracy=r['delta_accuracy']))
    for r,p in zip(records,holm([r['p'] for r in records])):r['holm_global']=p
    write(OUT/'primary_family.json',records)

def main():
    import argparse,os,time,traceback
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();done=set()
    while True:
        for task in TASKS:
            if task in done:continue
            folder=OUT/task
            if not (folder/'collection_complete.json').exists():continue
            if not (folder/'statistics_complete.json').exists():
                write(OUT/'statistics_status.json',dict(task=task,phase='analyzing',pid=os.getpid(),time=time.time()))
                behavioral(task);causal(task);neural(task);write(folder/'statistics_complete.json',dict(time=time.time()))
            done.add(task)
        global_correction()
        if len(done)==len(TASKS):write(OUT/'statistics_complete.json',dict(tasks=list(done),time=time.time()));break
        if not a.watch:break
        time.sleep(20)

if __name__=='__main__':main()
