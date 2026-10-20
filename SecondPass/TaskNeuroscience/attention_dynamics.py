"""Condition-by-time attention panels from frozen, already recorded native trials."""
import argparse,csv,html,json,textwrap,time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from .design import OUT,TASKS,NS,phases,onset,write,neural_cells
from .figures import plt,save,COLORS
from .statistics import source_clusters

def read(path):return json.loads(path.read_text(encoding='utf8'))

def load(task,cell):
    p=OUT/task/'native'/f'{cell}.npz'
    with np.load(p) as z:
        return {k:z[k] for k in ('y','p','attention','example')},read(p.with_suffix('.json'))

def selections(task):
    """Select by task design and recorded metadata, never by preferred attention."""
    out=[]
    def add(cell,title,predicate=lambda m,y:True):out.append((cell,title,predicate))
    if task=='motion_direction':
        for name in ('right','up','left','down'):add('mixed',name.title(),lambda m,y,v=name:m['direction']==v)
    elif task in ('orientation','contrast','spatial_frequency','chromatic_increment','contour','natural_spectrum'):
        keys={'orientation':'signed_orientation_degrees','contrast':'contrast_increment','spatial_frequency':'frequency_octave_increment','chromatic_increment':'chromatic_increment','contour':'alignment_jitter_degrees','natural_spectrum':'beta_delta'}
        a,m=load(task,'mixed');key=keys[task]
        # The spectrum renderer calls its exponent difference beta_delta.
        if key not in m[0]:
            key=next(k for k in m[0] if 'delta' in k or 'increment' in k)
        vals=sorted({abs(float(z[key])) for z in m})
        for v in vals:add('mixed',f'{key.replace("_"," ")}: {v:g}',lambda m,y,k=key,v=v:abs(float(m[k]))==v)
    elif task=='orientation_ring':
        for y in (0,1):add('D0','Clockwise' if y==0 else 'Counterclockwise',lambda m,z,y=y:m['target_location']==0 and z==y)
    elif task=='orientation_cued':
        for sign,title in ((1,'Matches instructed rotation'),(0,'Target unchanged'),(-1,'Opposes instructed rotation')):
            add('D24',title,lambda m,y,s=sign:m['target_location']==0 and np.sign(m['rotations_degrees'][0]*m['cue_sign'])==s)
    elif task=='motion_duration_cued':
        for cell in ('D0','D12','D24'):add(cell,cell.replace('D','Delay: ')+' frames',lambda m,y:m['target_location']==0)
    elif task=='krauzlis_cued_motion':
        for event in ('target','foil','catch'):add('B28',{'target':'Change at right cue','foil':'Change at left foil','catch':'No change'}[event],lambda m,y,e=event:m['target_location']==1 and m['event_type']==e)
    elif task=='spatial_binding':
        for y in (0,1):add('D24','Foil-only swap' if y==0 else 'Target-involving swap',lambda m,z,y=y:m['target_location']==0 and z==y)
    elif task=='image_recognition':
        for n in (4,24):
            for y in (0,1):add(f'N{n}_H5',f'{n} study images; '+('seen probe' if y else 'novel probe'),lambda m,z,y=y:z==y)
    elif task.startswith('vda'):
        cells=[c['id'] for c in TASKS[task]['conditions']]
        if task=='vda16':
            # Fixed physical site, with enough catch trials in each pooled bin.
            for lo,title in ((True,'Cue validity 25-50%'),(False,'Cue validity 75-100%')):
                add(cells,title+'; cue row 2, column 3',lambda m,y,lo=lo:y==0 and m['cue_index']==6 and (m['displayed_validity']<=.5)==lo)
        else:
            cue=int(task[3:])-1
            vals=(.25,.5,.75,1.) if task=='vda4' else (0.,.25,.5,.75,1.)
            for v in vals:add(cells,f'Cue validity {100*v:g}%; no change',lambda m,y,v=v,cue=cue:y==0 and m['cue_index']==cue and m['displayed_validity']==v)
    elif task=='memory_guided_search':
        for cell in neural_cells(task):add(cell,cell.replace('_S8_D12_H8','').replace('_',' ').title(),lambda m,y:m['target_index']==0)
    elif task=='motion_memory_search':
        for v in (20,45,90):add(f'motion_A{v}_S12_D12_H12',f'Direction separation {v} degrees',lambda m,y:m['target_index']==0)
    elif task=='popout_search':
        for cell in ('color_X12_H8','color_X72_H8','orientation_X5_H8','orientation_X45_H8'):
            add(cell,cell.replace('_H8','').replace('_X',' difference ').capitalize(),lambda m,y:m['target_index']==0)
    return out

def regions(task,m):
    yy,xx=np.meshgrid(np.arange(10)*10+4.5,np.arange(10)*10+4.5,indexing='ij')
    if task.startswith('vda'):
        from SecondPass.StabilizedAttentionVDA.Analysis.design import masks
        cells=masks(int(task[3:]))[0];cue=m['cue_index'];other=len(cells)-1-cue
        return ['Cued cell','Opposite cell','Other cells (mean)'],np.array([cells[cue],cells[other],np.mean(np.delete(cells,[cue,other],axis=0),axis=0)])
    centers=m.get('positions_xy',m.get('centers_xy'))
    if centers is None and 'target_location' in m:centers=[[27,27],[73,27],[27,73],[73,73]]
    if centers is not None:
        rad=16 if 'centers_xy' in m else 12
        cells=np.array([(((xx-x)**2+(yy-y)**2)<rad**2).ravel() for x,y in centers])
        target=m.get('target_location',m.get('target_index',0));other=np.delete(cells,target,axis=0).mean(0)
        central=(((xx-49.5)**2+(yy-49.5)**2)<16**2).ravel() & ~cells.any(0)
        return ['Target region','Other regions (mean)','Central sample region'],np.array([cells[target],other,central])
    if task=='image_recognition':
        center=((xx>=30)&(xx<70)&(yy>=30)&(yy<70)).ravel()
        return ['Central 4 x 4 sites','Remaining 84 sites'],np.array([center,~center])
    return ['Top left','Top right','Bottom left','Bottom right'],np.array([((xx<50)&(yy<50)).ravel(),((xx>=50)&(yy<50)).ravel(),((xx<50)&(yy>=50)).ravel(),((xx>=50)&(yy>=50)).ravel()])

def bootstrap(values,metadata,seed):
    clusters=source_clusters(metadata);unique=list(dict.fromkeys(clusters));groups=[np.flatnonzero(clusters==k) for k in unique]
    sums=np.array([values[ix].sum(0) for ix in groups]);sizes=np.array([len(ix) for ix in groups])
    rng=np.random.default_rng(seed);draw=rng.integers(len(groups),size=(2000,len(groups)))
    means=sums[draw].sum(1)/sizes[draw].sum(1).reshape(-1,1,1)
    return np.quantile(means,[.025,.975],axis=0),len(groups)

def replay(task,cell,index,data,metadata):
    """Recreate displayed pixels only; no network inference or weight loading."""
    from .collect import new_stream,draw
    stream=new_stream('test',NS)
    for start in range(0,index+1,4):
        x,y,m=draw(stream,4,task,cell)
        assert np.array_equal(y.numpy(),data['y'][start:start+4])
        assert json.loads(json.dumps(m))==metadata[start:start+4],(task,cell,start,'metadata replay differs')
        if start==0:assert np.array_equal(x[0].numpy(),data['example']),(task,cell,'pixel replay differs')
        if start<=index<start+4:return x[index-start].numpy()

def assemble(task):
    cache={};rows=[]
    for number,(cells,title,predicate) in enumerate(selections(task)):
        refs=[];maps=[];metadata=[];labels=[];preds=[]
        for cell in [cells] if isinstance(cells,str) else cells:
            if cell not in cache:cache[cell]=load(task,cell)
            a,mm=cache[cell];ix=[i for i,(m,y) in enumerate(zip(mm,a['y'])) if predicate(m,int(y))]
            for i in ix:refs.append(dict(cell=cell,index=i));maps.append(a['attention'][i]);metadata.append(mm[i]);labels.append(a['y'][i]);preds.append(a['p'][i].argmax())
        assert len(refs)>=2,(task,title,'too few trials')
        a=np.asarray(maps,dtype=np.float32);n,t,_,_=a.shape
        assert np.allclose(a.sum((2,3)),1,atol=1e-5)
        names,mask=regions(task,metadata[0]);weights=[]
        for m in metadata:
            nn,rr=regions(task,m);assert nn==names and np.array_equal(rr,mask),'Spatial geometry differs within a condition'
            weights.append(rr)
        masses=np.einsum('ntp,nrp->ntr',a.reshape(n,t,100),np.asarray(weights))
        bands,nclusters=bootstrap(masses,metadata,52900+number)
        ref=refs[0];native,mm=cache[ref['cell']];movie=replay(task,ref['cell'],ref['index'],native,mm)
        assert movie.shape[0]==t
        path=OUT/task/'figures'/f'attention_movie_{number:02d}.png'
        pixels=np.concatenate([frame.transpose(1,2,0) for frame in movie],axis=1)
        Image.fromarray(np.round(np.clip(pixels,0,1)*255).astype(np.uint8)).save(path)
        ph=phases(task,metadata[0],t);anchor=onset(task,metadata[0],t)
        assert all(phases(task,m,t)==ph and onset(task,m,t)==anchor for m in metadata)
        # Site mass relative to a geometric alternative, or central-vs-uniform
        # concentration for recognition; no privileged location in full fields.
        rows.append(dict(title=title,n=n,clusters=nclusters,refs=refs,accuracy=float(np.mean(np.array(labels)==preds)),
            time=list(range(t)),aligned_time=list(range(-anchor,t-anchor)),anchor=anchor,phases=ph,
            mean_map=a.mean(0).tolist(),example_map=a[0].tolist(),example=ref,sprite=path.name,
            region_names=names,region_weights=mask.tolist(),uniform_mass=mask.mean(1).tolist(),
            mean_mass=masses.mean(0).tolist(),ci_low=bands[0].tolist(),ci_high=bands[1].tolist()))
    return rows

PHASE_COLORS={'cue':'#eedcb0','sample':'#d8e9f1','study':'#d8e9f1','motion':'#d8e9f1','baseline':'#d8e9f1',
    'delay':'#e7e7ee','blank':'#f0f0f0','fixation':'#f0f0f0','search':'#d9ece3','change':'#d9ece3','test':'#d9ece3','probe':'#d9ece3','comparison':'#d9ece3'}

def shade(ax,r):
    ph=r['phases'];start=0
    for i in range(1,len(ph)+1):
        if i==len(ph) or ph[i]!=ph[start]:
            ax.axvspan(start-r['anchor']-.5,i-r['anchor']-.5,color=PHASE_COLORS.get(ph[start],'#f1f3f4'),alpha=.6,zorder=-3)
            if i-start>=2 or len(ph)<=9 or ph[start]=='cue':ax.text((start+i-1)/2-r['anchor'],1.02,ph[start],ha='center',fontsize=7,transform=ax.get_xaxis_transform())
            start=i
    ax.axvline(0,color='#657784',ls='--',lw=.8)

def plots(task,rows):
    from matplotlib.colors import PowerNorm
    vmax=max(np.max(r['mean_map']) for r in rows);norm=PowerNorm(.5,vmin=0,vmax=vmax)
    names=[];folder=OUT/task/'figures';start=min(r['aligned_time'][0] for r in rows);end=max(r['aligned_time'][-1] for r in rows)
    for block in range(0,len(rows),3):
        group=rows[block:block+3]
        for left in range(start,end+1,7):
            times=list(range(left,min(left+7,end+1)))
            if not any(any(v in r['aligned_time'] for v in times) for r in group):continue
            fig,axes=plt.subplots(len(group),len(times),figsize=(max(5,len(times)*1.42),len(group)*1.75+.6),squeeze=False,constrained_layout=True)
            for i,r in enumerate(group):
                for j,v in enumerate(times):
                    ax=axes[i,j];ax.set_xticks([]);ax.set_yticks([])
                    if v in r['aligned_time']:
                        k=v+r['anchor'];im=ax.imshow(r['mean_map'][k],cmap='magma',norm=norm,interpolation='nearest')
                        ax.set_title(f'{v:+d} | t={k}\n{r["phases"][k]}',fontsize=8)
                    else:ax.set_facecolor('#edf0f2');ax.text(.5,.5,'Outside\nthis trial',ha='center',va='center',transform=ax.transAxes,fontsize=8)
                axes[i,0].set_ylabel(textwrap.fill(r['title'],24)+f'\nn={r["n"]}',fontsize=8)
            fig.colorbar(im,ax=list(axes.flat),shrink=.65,label='Mean probability / site (sqrt color scale)')
            fig.supxlabel('Frames relative to task evidence onset (0); t = original frame',fontsize=9)
            name=f'attention_conditions_maps_{block:02d}_{left-start:03d}';save(fig,folder/name);names.append(name)
    curve_names=[]
    for block in range(0,len(rows),3):
        group=rows[block:block+3];fig,axes=plt.subplots(len(group),1,figsize=(10,2.65*len(group)+.5),squeeze=False,constrained_layout=True)
        for ax,r in zip(axes[:,0],group):
            x=np.array(r['aligned_time']);mean=np.array(r['mean_mass']);lo=np.array(r['ci_low']);hi=np.array(r['ci_high']);shade(ax,r)
            for j,label in enumerate(r['region_names']):
                ax.plot(x,mean[:,j],'.-',color=COLORS[j],label=label,lw=1.6,ms=4)
                ax.fill_between(x,lo[:,j],hi[:,j],color=COLORS[j],alpha=.15)
                ax.axhline(r['uniform_mass'][j],color=COLORS[j],ls=':',alpha=.45,lw=.8)
            ax.set(xlim=(start-.5,end+.5),ylim=(-.025,1.025),ylabel='Attention mass')
            ax.set_title(f'{r["title"]} | n={r["n"]}, clusters={r["clusters"]}',pad=25,fontsize=11)
            ax.legend(fontsize=8,loc='upper left',ncol=min(4,len(r['region_names'])),bbox_to_anchor=(0,-.23))
            ax.set_xlabel('Frames relative to evidence onset',labelpad=2)
        name=f'attention_conditions_curves_{block:02d}';save(fig,folder/name);curve_names.append(name)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for k,r in enumerate(rows):
        x=np.array(r['aligned_time']);mean=np.array(r['mean_mass']);lo=np.array(r['ci_low']);hi=np.array(r['ci_high']);color=COLORS[k%len(COLORS)]
        axes[0].plot(x,mean[:,0],'.-',color=color,label=r['title'],ms=3)
        axes[0].fill_between(x,lo[:,0],hi[:,0],color=color,alpha=.12)
        maps=np.array(r['mean_map']);h=-(maps*np.log(np.maximum(maps,1e-12))).sum((1,2))
        axes[1].plot(x,np.exp(h),'.-',color=color,ms=3)
    for ax in axes:ax.axvline(0,color='#657784',ls='--',lw=.8);ax.set_xlabel('Frames relative to evidence onset')
    axes[0].set(ylabel=rows[0]['region_names'][0]+' attention mass',ylim=(-.025,1.025),title='Same region, different conditions')
    axes[1].set(ylabel='Effective sites in condition-average map',ylim=(0,102),title='Spatial concentration of the mean map')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.19),ncol=2,fontsize=8)
    name='attention_conditions_compare';save(fig,folder/name)
    overview=[]
    for block in range(0,len(rows),3):
        group=rows[block:block+3];fig,axes=plt.subplots(len(group),6,figsize=(11,2*len(group)+.4),squeeze=False,constrained_layout=True)
        for i,r in enumerate(group):
            ph=r['phases'];encoding=[t for t,p in enumerate(ph) if p in ('sample','study','motion','reference','baseline')];delay=[t for t,p in enumerate(ph) if p in ('delay','blank') and t>0]
            times=[0,encoding[-1] if encoding else max(0,r['anchor']-1),delay[len(delay)//2] if delay else max(0,r['anchor']-1),max(0,r['anchor']-1),r['anchor'],len(ph)-1]
            labels=('Start','Last encoding','Middle delay / pre-event','Before evidence','Evidence onset','Final')
            if task=='motion_duration_cued':
                last_motion=max(t for t,p in enumerate(ph) if p=='motion')
                times=[0,r['anchor'],last_motion,delay[len(delay)//2] if delay else last_motion,len(ph)-2,len(ph)-1]
                labels=('Start','Motion onset','Last motion','Middle retention / motion','Before report','Report')
            for j,(t,label) in enumerate(zip(times,labels)):
                ax=axes[i,j];im=ax.imshow(r['mean_map'][t],cmap='magma',norm=norm,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{label}\nt={t}; {ph[t]}',fontsize=7)
            axes[i,0].set_ylabel(textwrap.fill(r['title'],23)+f'\nn={r["n"]}',fontsize=8)
        fig.colorbar(im,ax=list(axes.flat),shrink=.65,label='Mean probability / site (sqrt color scale)')
        name=f'attention_conditions_landmarks_{block:02d}';save(fig,folder/name);overview.append(name)
    return dict(maps=names,curves=curve_names,comparison='attention_conditions_compare',landmarks=overview,color_max=float(vmax))

def tables(task,rows):
    folder=OUT/task/'source_tables';folder.mkdir(exist_ok=True)
    with (folder/'attention_condition_timecourses.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['condition','n','clusters','frame','relative_frame','phase','region','mean_attention_mass','ci95_low','ci95_high','uniform_mass'])
        for r in rows:
            for t in r['time']:
                for j,name in enumerate(r['region_names']):w.writerow([r['title'],r['n'],r['clusters'],t,r['aligned_time'][t],r['phases'][t],name,r['mean_mass'][t][j],r['ci_low'][t][j],r['ci_high'][t][j],r['uniform_mass'][j]])

def run(task):
    rows=assemble(task);manifest=plots(task,rows);tables(task,rows)
    write(OUT/task/'attention_dynamics.json',dict(task=task,groups=rows,figures=manifest,
        source='Previously recorded native trials; no new model inference',
        selection='Condition definitions and physical coordinates fixed; outcomes pooled',
        uncertainty='Pointwise 95% whole-trial bootstrap, 2000 draws; repeated photo identities clustered; descriptive within-condition bands',
        reference='Morgan, Albanna and Herman (2025), Figure 4; https://arxiv.org/html/2502.10955v1'))
    return len(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',default=list(TASKS));args=p.parse_args()
    for task in args.tasks:
        if (OUT/'stop_requested.json').exists():return
        write(OUT/'publication_status.json',dict(phase='attention_condition_time_analysis',task=task,time=time.time()))
        n=run(task);print(task,n,'conditions finished',flush=True)

if __name__=='__main__':main()
