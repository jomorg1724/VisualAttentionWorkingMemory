"""Held-out unit profiles and matched-query causal statistics."""
import json,textwrap
import numpy as np
from .design import OUT,TASKS,neural_cells,sites,phases,onset,retention_boundary,write
from .statistics import metrics,paired,holm,interval,prepare_features,ridge

def geometry(task):
    import matplotlib.pyplot as plt
    from .figures import save
    root=OUT/task;record=json.loads((root/'deep_analysis.json').read_text())['full_state_probes'][0];cell=record['cell'];meta=json.loads((root/'neural'/f'{cell}_test.json').read_text())
    first=np.array(record['cls_crossvalidated_centroid_distance']);boundary=retention_boundary(task,meta[0],len(first));k=first.shape[-1];off=~np.eye(k,dtype=bool);fig,axes=plt.subplots(2,3,figsize=(10,6.8),constrained_layout=True)
    for i,region in enumerate(('cls','spatial')):
        distance=np.array(record[region+'_crossvalidated_centroid_distance']);mean=distance[:,off].mean(1)
        axes[i,0].plot(mean,color='#164d70');axes[i,0].axhline(0,ls='--',color='#999999');axes[i,0].axvline(boundary,ls=':',color='#a84458');axes[i,0].set(title=region+' class separation',xlabel='Model frame',ylabel='Mean cross-validated distance')
        vmax=max(float(abs(distance[[boundary,-1]]).max()),.01)
        for j,t in enumerate((boundary,len(distance)-1),1):
            im=axes[i,j].imshow(distance[t],vmin=-vmax,vmax=vmax,cmap='RdBu_r');axes[i,j].set(title=('Memory-test' if j==1 else 'Final')+f' frame {t}',xlabel='Correct-answer class',ylabel='Correct-answer class');axes[i,j].set_xticks(range(k));axes[i,j].set_yticks(range(k));fig.colorbar(im,ax=axes[i,j],shrink=.7)
    save(fig,root/'figures/full_state_geometry')

def orientation_inventory(task):
    if task not in ('spatial_binding','orientation_cued'):return None
    import matplotlib.pyplot as plt
    from .figures import save
    cell=neural_cells(task)[0];root=OUT/task;a=np.load(root/'neural'/f'{cell}_val.npz');b=np.load(root/'neural'/f'{cell}_test.npz')
    ma=json.loads((root/'neural'/f'{cell}_val.json').read_text());mb=json.loads((root/'neural'/f'{cell}_test.json').read_text())
    aa=np.array([m['sample_angles_radians'] for m in ma]);bb=np.array([m['sample_angles_radians'] for m in mb]);y=np.stack((np.cos(2*aa),np.sin(2*aa)),-1);yt=np.stack((np.cos(2*bb),np.sin(2*bb)),-1)
    result=dict(cell=cell,target='Four remembered orientations at their original locations, each encoded as cos(2 angle),sin(2 angle)',regions={});fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,region,seed in zip(axes,('cls','spatial'),(1704,1703)):
        x,xt,transform,_=prepare_features(a[region],b[region],seed);values=[]
        den=((yt-y.mean(0))**2).sum((0,2))
        for t in range(x.shape[1]):
            w,bias0=ridge(x[:,t],y.reshape(len(y),-1));pred=(xt[:,t]@w+bias0).reshape(yt.shape);values.append(1-((pred-yt)**2).sum((0,2))/den)
        values=np.array(values);result['regions'][region]=dict(r2_by_frame_and_location=values.tolist(),transform=transform)
        for i,label in enumerate(('top left','top right','bottom left','bottom right')):ax.plot(values[:,i],label=label)
        ax.axhline(0,ls='--',color='#888888');ax.set(title=region+' memory',xlabel='Model frame',ylabel='Held-out orientation reconstruction R-squared');ax.legend(fontsize=8)
    save(fig,root/'figures/orientation_inventory');write(root/'orientation_inventory.json',result);return result

def condition_maps(task):
    import matplotlib.pyplot as plt
    from .figures import save
    cells=neural_cells(task)
    if len(cells)==1:return []
    rows=[]
    for cell in cells:
        p=OUT/task/'native'/f'{cell}.npz';a=np.load(p);meta=json.loads(p.with_suffix('.json').read_text());m=meta[0];t=a['attention'].shape[1];ph=phases(task,m,t)
        encoding=[i for i,v in enumerate(ph) if v in ('sample','study','reference')];times=[encoding[-1] if encoding else 0,retention_boundary(task,m,t),onset(task,m,t),t-1]
        mask=a['y']==1 if task=='image_recognition' else np.array([z.get('target_index',z.get('target_location',-1))==0 for z in meta])
        maps=a['attention'][mask].mean(0)[times];rows.append(dict(cell=cell,times=times,n=int(mask.sum()),maps=maps.tolist(),selection='seen probes' if task=='image_recognition' else 'top-left targets'))
    vmax=max(np.array(r['maps']).max() for r in rows);names=[]
    for start in range(0,len(rows),3):
        batch=rows[start:start+3];fig,axes=plt.subplots(len(batch),4,figsize=(10,2.8*len(batch)),squeeze=False,constrained_layout=True)
        for i,r in enumerate(batch):
            for j,phase in enumerate(('Last encoding','Memory test','Evidence onset','Final')):
                im=axes[i,j].imshow(r['maps'][j],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');axes[i,j].set_title(f'{phase}: t={r["times"][j]}',fontsize=9);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            axes[i,0].set_ylabel(textwrap.fill(r['cell'].replace('_S8_D12_H8',''),20)+f'\nn={r["n"]}',fontsize=9)
        fig.colorbar(im,ax=list(axes.flat),label='Mean probability per patch',shrink=.7);name=f'condition_maps_{start:02d}';save(fig,OUT/task/'figures'/name);names.append(name)
    write(OUT/task/'condition_maps.json',rows);return names

def memory_examples(task):
    if task!='memory_guided_search':return []
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .collect import new_stream
    from .design import NS
    from .figures import save
    stream=new_stream('test',NS);rows=[];labels=['TL','TR','BL','BR']
    for cell in neural_cells(task):
        path=OUT/task/'native'/f'{cell}.npz';a=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());wrong=np.flatnonzero(a['p'].argmax(1)!=a['y']);i=int(wrong[0]) if len(wrong) else 0
        x,y,m=stream.trial(cell,i);assert int(y)==int(a['y'][i]) and m['array_sha256']==meta[i]['array_sha256']
        rows.append(dict(cell=cell,index=i,selected='first error' if len(wrong) else 'first trial; no errors observed',choice=int(a['p'][i].argmax()),target=int(y),sample=x[1,:,36:64,36:64].permute(1,2,0).numpy(),array=x[-1].permute(1,2,0).numpy(),attention=a['attention'][i,-1],centers=m['centers_xy']))
    names=[]
    for start in range(0,6,3):
        batch=rows[start:start+3];fig,axes=plt.subplots(3,4,figsize=(10,8.2),constrained_layout=True)
        for i,r in enumerate(batch):
            axes[i,0].imshow(r['sample']);axes[i,0].set_title('Sample crop',fontsize=9);axes[i,0].set_ylabel(r['cell'].replace('_S8_D12_H8','').replace('_','\n')+f'\ntrial {r["index"]}',fontsize=9)
            axes[i,1].imshow(r['array']);axes[i,1].set_title(f'True {labels[r["target"]]}, choice {labels[r["choice"]]}',fontsize=9)
            for loc,color in ((r['target'],'#164d70'),(r['choice'],'#d65a4a')):
                cx,cy=r['centers'][loc];axes[i,1].add_patch(Rectangle((cx-15,cy-15),30,30,fill=False,edgecolor=color,linewidth=1.5))
            axes[i,2].imshow(r['array']);axes[i,2].imshow(r['attention'],extent=(-.5,99.5,99.5,-.5),cmap='magma',alpha=.65,vmin=0,vmax=r['attention'].max(),interpolation='nearest');axes[i,2].set_title('Own attention overlay',fontsize=9)
            im=axes[i,3].imshow(r['attention'],vmin=0,vmax=r['attention'].max(),cmap='magma',interpolation='nearest');axes[i,3].set_title('Own final attention',fontsize=9);fig.colorbar(im,ax=axes[i,3],shrink=.6)
            for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
        name=f'memory_examples_{start:02d}';save(fig,OUT/task/'figures'/name);names.append(name)
    write(OUT/task/'memory_examples.json',[{k:v for k,v in r.items() if k not in ('sample','array','attention')} for r in rows]);return names

def recognition(task):
    if task!='image_recognition':return None
    import matplotlib.pyplot as plt
    from .figures import save
    groups={};positions={}
    for path in (OUT/task/'native').glob('*.npz'):
        if path.stem.endswith('_latest'):continue
        a=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
        for y,p,m in zip(a['y'],a['p'],meta):
            load=m['study_load'];identity=m['probe_source_id'];positive=int(p.argmax()==1)
            groups.setdefault((load,int(y)),{}).setdefault(identity,[]).append(positive)
            if y:
                quartile=min(3,int(4*m['seen_study_index']/load));positions.setdefault((load,quartile),{}).setdefault(identity,[]).append(positive)
    def summary(group):
        sums=np.array([sum(v) for v in group.values()]);ns=np.array([len(v) for v in group.values()]);rng=np.random.default_rng(824);ix=rng.integers(len(ns),size=(3000,len(ns)))
        bounds=np.quantile(sums[ix].sum(1)/ns[ix].sum(1),[.025,.975]).tolist();rate=sums.sum()/ns.sum()
        if rate in (0,1):bounds=interval(int(rate*len(ns)),len(ns))
        return dict(rate=float(rate),ci95=bounds,n=int(ns.sum()),unique_probe_images=len(ns))
    rows=[dict(load=l,report='hit' if y else 'false_alarm',**summary(g)) for (l,y),g in sorted(groups.items())]
    serial=[dict(load=l,quartile=q,**summary(g)) for (l,q),g in sorted(positions.items())]
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for report in ('hit','false_alarm'):
        r=[z for z in rows if z['report']==report];y=np.array([z['rate'] for z in r]);bounds=np.array([z['ci95'] for z in r]);axes[0].errorbar([z['load'] for z in r],y,yerr=[y-bounds[:,0],bounds[:,1]-y],fmt='o-',label=report.replace('_',' '),capsize=2)
    axes[0].set(xlabel='Study images',ylabel='Positive-report fraction',ylim=(0,1.04),title='Sensitivity and response tendency');axes[0].legend()
    for load in (4,12,24):
        r=[z for z in serial if z['load']==load];y=np.array([z['rate'] for z in r]);bounds=np.array([z['ci95'] for z in r]);axes[1].errorbar([z['quartile']+1 for z in r],y,yerr=[y-bounds[:,0],bounds[:,1]-y],fmt='o-',label=f'{load} study images',capsize=2)
    axes[1].set(xlabel='Study-position quartile (earliest to latest)',ylabel='Hit fraction',ylim=(0,1.04),title='Serial-position profile');axes[1].set_xticks([1,2,3,4]);axes[1].legend(fontsize=8)
    save(fig,OUT/task/'figures/recognition_errors');result=dict(load_curves=rows,serial_position=serial,intervals='Bootstrap by probe photograph across probe-hold conditions; all-perfect/zero boundaries use Wilson on unique image count.')
    write(OUT/task/'recognition_summary.json',result);return result

def weak_vda(task):
    import matplotlib.pyplot as plt
    from .figures import save
    rows=[]
    for path in sorted((OUT/task/'weak_counterfactual').glob('*.npz')):
        a=np.load(path);m=json.loads(path.with_suffix('.json').read_text())
        for i,arm in enumerate(m['arms']):
            rows.append(dict(strength=m['strength'],event=m['event'],arm=arm,report_change=float((a['p'][:,i].argmax(-1)==1).mean()),**metrics(a['y'],a['p'][:,i]),**paired(a['y'],a['p'][:,0],a['p'][:,i])))
    if not rows:return []
    for r,p in zip(rows,holm([r['p_exact'] for r in rows])):r['holm_followup']=p
    fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,event in zip(axes,('cued','uncued','catch')):
        for arm in ('sham','cued_-4_change','cued_+4_change','uncued_-4_change','uncued_+4_change'):
            rs=sorted([r for r in rows if r['event']==event and r['arm']==arm],key=lambda r:r['strength'])
            yy=np.array([r['report_change'] for r in rs]);bounds=np.array([interval(round(r['report_change']*r['n']),r['n']) for r in rs])
            ax.errorbar([r['strength'] for r in rs],yy,yerr=[yy-bounds[:,0],bounds[:,1]-yy],fmt='o-',capsize=2,label=arm.replace('_change',''))
        ax.set(xlabel='Rotation magnitude (degrees)',ylabel='Change-report fraction',ylim=(0,1.04),title=event);ax.legend(fontsize=8)
    save(fig,OUT/task/'figures/weak_vda');write(OUT/task/'weak_counterfactual_summary.json',rows);return rows

def units(task):
    import matplotlib.pyplot as plt
    from .figures import save
    cell=neural_cells(task)[0];root=OUT/task
    a=np.load(root/'neural'/f'{cell}_val.npz');b=np.load(root/'neural'/f'{cell}_test.npz')
    x=a['cls'].astype(float);xt=b['cls'].astype(float);mu=x.mean((0,1));sd=np.maximum(x.std((0,1)),.05)
    x=(x-mu)/sd;xt=(xt-mu)/sd;labels=TASKS[task]['labels'];k=len(labels)
    means=np.array([x[a['y']==i].mean(0) for i in range(k)]);testmeans=np.array([xt[b['y']==i].mean(0) for i in range(k)])
    preference=means[:,-1].argmax(0);strength=means[:,-1].var(0);order=np.lexsort((-strength,preference));top=np.argsort(strength)[-4:][::-1]
    ncols=min(k,2);nrows=int(np.ceil(k/ncols));fig,axes=plt.subplots(nrows,ncols,figsize=(10,4.5*nrows),squeeze=False,constrained_layout=True)
    for i,ax in enumerate(axes.flat):
        if i>=k:ax.set_visible(False);continue
        im=ax.imshow(testmeans[i,:,order],aspect='auto',vmin=-2,vmax=2,cmap='RdBu_r',origin='lower')
        ax.set_title(textwrap.fill(labels[i],25),fontsize=10);ax.set(xlabel='Model frame',ylabel='Unit (training-derived order)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Held-out mean standardized activation',shrink=.8);save(fig,root/'figures/unit_population')
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for unit,ax in zip(top,axes.flat):
        for i,label in enumerate(labels):
            values=xt[b['y']==i,:,unit];mean=values.mean(0);sem=values.std(0,ddof=1)/np.sqrt(len(values));time=np.arange(len(mean))
            ax.plot(time,mean,label=textwrap.fill(label,28));ax.fill_between(time,mean-1.96*sem,mean+1.96*sem,alpha=.12)
        ax.set(title=f'CLS coordinate {unit}',xlabel='Model frame',ylabel='Standardized activation');ax.legend(fontsize=8)
    save(fig,root/'figures/unit_examples');write(root/'unit_panels.json',dict(cell=cell,unit_order=order.tolist(),example_units=top.tolist(),selection='Largest final class-mean variance on fitting trials only',evaluation='Independent test trials; shaded bands are 1.96 standard errors of mean activation'))

def query(task):
    import matplotlib.pyplot as plt
    from .figures import save
    folder=OUT/task/'query_switch';result=[]
    if not folder.exists():return []
    fig,axes=plt.subplots(2,3 if task=='memory_guided_search' else 2,figsize=(10,7),squeeze=False,constrained_layout=True)
    for n,path in enumerate(sorted(folder.glob('*.npz'))):
        a=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());p=a['p'];attention=a['attention'].astype(float);correct=p.argmax(-1)==np.arange(4)[None,:]
        probability_gain=np.array([np.diag(v).mean()-(v.sum()-np.trace(v))/12 for v in p]);maps=[]
        for i,ms in enumerate(meta['metadata']):
            masks=np.array([sites(task,m)[0] for m in ms]);final=attention[i,:,-1].reshape(4,100);maps.append(final@masks.T)
        maps=np.array(maps);allocation_gain=np.array([np.trace(v)/4-(v.sum()-np.trace(v))/12 for v in maps])
        rng=np.random.default_rng(1863);ix=rng.integers(len(p),size=(3000,len(p)))
        ci=lambda v:np.quantile(np.asarray(v)[ix].mean(1),[.025,.975]).tolist()
        accuracy_bounds=ci(correct.mean(1))
        if correct.all() or not correct.any():accuracy_bounds=interval(int(correct.all())*len(p),len(p))
        record=dict(cell=path.stem,scenes=len(p),queries_per_scene=4,accuracy=float(correct.mean()),accuracy_ci95=accuracy_bounds,
            all_four_correct=float(correct.all(1).mean()),choice_changes=float(np.mean([len(set(v))>1 for v in p.argmax(-1)])),
            template_probability_gain=float(probability_gain.mean()),template_probability_gain_ci95=ci(probability_gain),
            template_attention_gain=float(allocation_gain.mean()),template_attention_gain_ci95=ci(allocation_gain),
            blank_sample_balanced_accuracy=.25,mean_probability_matrix=p.mean(0).tolist(),mean_attention_matrix=maps.mean(0).tolist())
        result.append(record)
        ax=axes.flat[n];im=ax.imshow(p.mean(0),vmin=0,vmax=1,cmap='viridis');ax.set(title=path.stem.replace('_S8_D12_H8','').replace('_S12_D12_H12',''),xlabel='Reported location',ylabel='Sampled object location');ax.set_xticks(range(4));ax.set_yticks(range(4))
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{p[:,i,j].mean():.2f}',ha='center',va='center',color='white' if p[:,i,j].mean()<.6 else '#173344')
    for ax in list(axes.flat)[len(result):]:ax.set_visible(False)
    fig.colorbar(im,ax=list(axes.flat),label='Mean report probability',shrink=.65);save(fig,OUT/task/'figures/query_switch');write(OUT/task/'query_switch_summary.json',result);return result
