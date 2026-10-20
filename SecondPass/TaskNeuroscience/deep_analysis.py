"""Full-state probe controls, unit selectivity, task-specific error structure."""
import json,time,os
import numpy as np
from scipy.stats import f_oneway
from sklearn.metrics import roc_auc_score
from .design import *
from .statistics import metrics,balanced,holm

def kernel_fit(x,y,xt,alpha):
    mu=x.mean(0);sd=np.maximum(x.std(0),.05);x=(x-mu)/sd;xt=(xt-mu)/sd;width=x.shape[1]
    ym=y.mean(0);dual=np.linalg.solve(x@x.T/width+alpha*np.eye(len(x)),y-ym)
    return xt@x.T@dual/width+ym

def full_probe(x,y,xt,yt,classes):
    x=x.reshape(len(x),-1).astype(np.float64);xt=xt.reshape(len(xt),-1).astype(np.float64)
    if np.max(np.std(x,axis=0))<1e-8:return dict(constant=True)
    rng=np.random.default_rng(770);fit=[];val=[]
    for c in np.unique(y):
        ix=rng.permutation(np.flatnonzero(y==c));cut=max(1,int(.75*len(ix)));fit.extend(ix[:cut]);val.extend(ix[cut:])
    if not val:return dict(insufficient_classes=True)
    eye=np.eye(classes);scores=[];alphas=[.001,.01,.1,1.,10.]
    for alpha in alphas:
        p=kernel_fit(x[fit],eye[y[fit]],x[val],alpha);scores.append(balanced(y[val],p.argmax(1)))
    alpha=alphas[int(np.argmax(scores))];p=kernel_fit(x,eye[y],xt,alpha)
    auc=[roc_auc_score(yt==c,p[:,c]) for c in range(classes) if 0<(yt==c).sum()<len(yt)]
    return dict(dimensions=x.shape[1],alpha=alpha,inner_validation_accuracy=scores,
        balanced_accuracy=balanced(yt,p.argmax(1)),auc=float(np.mean(auc)) if auc else None)

def extra(task):
    folder=OUT/task;result=dict(task=task,full_state_probes=[],event_groups=[],feature_errors={})
    for cell in neural_cells(task):
        a=np.load(folder/'neural'/f'{cell}_val.npz');b=np.load(folder/'neural'/f'{cell}_test.npz')
        m=json.loads((folder/'neural'/f'{cell}_test.json').read_text());record=dict(cell=cell,regions={})
        for region in ('appearance','fusion','spatial','cls','comparison','attention'):
            rr={};times={'final':a[region].shape[1]-1,'retention_boundary':retention_boundary(task,m[0],a[region].shape[1])}
            for name,t in times.items():
                rr[name]=full_probe(a[region][:,t],a['y'],b[region][:,t],b['y'],TASKS[task]['classes'])
            record['regions'][region]=rr
        # Unit selectivity at final time: fit-sign choice, evaluate on fresh test trials.
        x=a['cls'][:,-1];xt=b['cls'][:,-1];y=a['y'];yt=b['y'];units=[]
        for j in range(64):
            groups=[xt[yt==c,j] for c in np.unique(yt)]
            if all(len(g)>1 for g in groups) and np.var(xt[:,j])>1e-10:
                f,p=f_oneway(*groups)
                between=sum(len(g)*(g.mean()-xt[:,j].mean())**2 for g in groups)
                eta=float(between/max(np.sum((xt[:,j]-xt[:,j].mean())**2),1e-12))
                units.append(dict(unit=j,p=float(p),eta_squared=eta))
        for r,p in zip(units,holm([r['p'] for r in units])):r['holm_p']=p
        record['cls_unit_selectivity']=units
        # Cross-validated squared distances between class centroids, without diagonal noise bias.
        for region in ('cls','spatial'):
            aa=a[region].astype(float).reshape(len(y),a[region].shape[1],-1);bb=b[region].astype(float).reshape(len(yt),b[region].shape[1],-1)
            nclass=TASKS[task]['classes'];distance=np.zeros((aa.shape[1],nclass,nclass))
            for t in range(aa.shape[1]):
                sd=np.maximum(aa[:,t].std(0),.05)
                for i in range(nclass):
                    for j in range(nclass):
                        if not ((y==i).any() and (y==j).any() and (yt==i).any() and (yt==j).any()):continue
                        d1=(aa[y==i,t].mean(0)-aa[y==j,t].mean(0))/sd;d2=(bb[yt==i,t].mean(0)-bb[yt==j,t].mean(0))/sd
                        distance[t,i,j]=np.mean(d1*d2)
            record[region+'_crossvalidated_centroid_distance']=distance.tolist()
        result['full_state_probes'].append(record)
    for cell in TASKS[task]['conditions']:
        name=cell['id'];a=np.load(folder/'native'/f'{name}.npz');m=json.loads((folder/'native'/f'{name}.json').read_text())
        fields=['event_type','cue_color','displayed_validity','seen_study_index']
        for field in fields:
            values=list(dict.fromkeys(str(z[field]) for z in m if field in z))
            for value in values:
                mask=np.array([str(z.get(field))==value for z in m]);result['event_groups'].append(dict(cell=name,field=field,value=value,**metrics(a['y'][mask],a['p'][mask])))
        if task.startswith('vda'):
            val=np.array([z['value'] for z in m]);correct=a['p'].argmax(1)==a['y'];result['event_groups'].append(dict(cell=name,field='reward',value='fraction_available',score=float((correct*val).sum()/val.sum())))
        if task=='memory_guided_search':
            for i,z in enumerate(m):
                if z['family']!='features' or z['similarity']!='similar':continue
                choice=int(a['p'][i].argmax());target=int(a['y'][i]);rr=result['feature_errors'];rr['trials']=rr.get('trials',0)+1
                if choice!=target:
                    rr['errors']=rr.get('errors',0)+1
                    for f in ('color','shape','texture'):
                        rr['wrong_shared_'+f]=rr.get('wrong_shared_'+f,0)+int(z['object_descriptors'][choice][f]==z['object_descriptors'][target][f])
    write(folder/'deep_analysis.json',result)

def main():
    import argparse
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();done=set()
    while True:
        for task in TASKS:
            if task in done or not (OUT/task/'statistics_complete.json').exists():continue
            if not (OUT/task/'deep_analysis.json').exists():
                write(OUT/'deep_status.json',dict(task=task,pid=os.getpid(),time=time.time()));extra(task)
            done.add(task)
        if len(done)==len(TASKS) or not a.watch:break
        time.sleep(20)
    if len(done)==len(TASKS):write(OUT/'deep_complete.json',dict(tasks=list(done),time=time.time()))

if __name__=='__main__':main()
