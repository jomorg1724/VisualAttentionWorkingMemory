"""Publication figures from completed frozen-model experiments only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .design import *
from .statistics import interval
from SecondPass.StabilizedAttentionSuite.Analysis.design import AXES

COLORS=['#164d70','#d65a4a','#359a82','#a77727','#8067a9','#6a8593']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':'#1d3446','text.color':'#1d3446','axes.prop_cycle':plt.cycler(color=COLORS),
    'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})

def save(fig,path):
    from matplotlib.ticker import MaxNLocator
    for ax in fig.axes:
        if ax.images:continue  # A condition label can contain "frames" on a spatial map.
        if 'frame' in ax.get_xlabel().lower():ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
        if 'frame' in ax.get_ylabel().lower():ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(path.with_suffix('.png'),dpi=140,bbox_inches='tight');plt.close(fig)

def read(path):return json.loads(path.read_text())

def figures(task):
    root=OUT/task;folder=root/'figures';folder.mkdir(exist_ok=True)
    b=read(root/'behavior.json');c=read(root/'causal.json');nn=read(root/'neural_summary.json');cell=neural_cells(task)[0]
    native=np.load(root/'native'/f'{cell}.npz');meta=read(root/'native'/f'{cell}.json');n=len(native['y'])
    # 1. Full native factorial, psychometric challenge and calibration.
    fig,ax=plt.subplots(1,3,figsize=(10,3.8),constrained_layout=True)
    rows=b['cells'];acc=np.array([r['accuracy'] for r in rows]);lo=np.array([r['accuracy_ci95'][0] for r in rows]);hi=np.array([r['accuracy_ci95'][1] for r in rows])
    ax[0].errorbar(range(len(rows)),acc,yerr=[acc-lo,hi-acc],fmt='o',ms=3,elinewidth=.7,color=COLORS[0]);ax[0].set(xlabel='Native condition index',ylabel='Correct fraction',ylim=(0,1.04),title=f"All {len(rows)} native conditions")
    if len(rows)<=12:ax[0].set_xticks(range(len(rows)),[r['cell'] for r in rows],rotation=45,ha='right')
    points=b['psychometric']
    if points:
        for event in dict.fromkeys(r['event'] for r in points):
            series=sorted([r for r in points if r['event']==event],key=lambda z:z['strength']);x=[r['strength'] for r in series]
            response=task.startswith('vda') or task=='krauzlis_cued_motion'
            y=[r['report1'] if response else r['accuracy'] for r in series]
            limits=[interval(int(round(v*r['n'])),r['n']) for v,r in zip(y,series)];lo=np.array([v[0] for v in limits]);hi=np.array([v[1] for v in limits]);yy=np.array(y)
            ax[1].errorbar(x,y,yerr=[yy-lo,hi-yy],fmt='o-',ms=4,label=event,capsize=2)
        ax[1].set(xlabel=AXES.get(task,'Orientation change (degrees)'),ylabel='Change-report fraction' if response else 'Correct fraction',ylim=(0,1.04),title='Difficulty challenge');ax[1].legend(fontsize=9)
    else:
        keys=[k for k in ('condition_delay','condition_sample_frames','condition_exposure','condition_load','condition_difference','condition_spacing') if any(r['factor']==k for r in b['factors'])]
        for key in keys[:3]:
            sr=sorted([r for r in b['factors'] if r['factor']==key],key=lambda r:float(r['value']));ax[1].plot([float(r['value']) for r in sr],[r['accuracy'] for r in sr],'o-',label=key.replace('condition_',''),ms=4)
        ax[1].set(xlabel='Factor value (frames / task units)',ylabel='Marginal correct fraction',ylim=(0,1.04),title='Native factorial margins');ax[1].legend(fontsize=9)
    pp=[];yy=[]
    for r in rows:
        z=np.load(root/'native'/f"{r['cell']}.npz");pp.extend(z['p']);yy.extend(z['y'])
    pp=np.array(pp);yy=np.array(yy);confidence=pp.max(1);correct=pp.argmax(1)==yy
    xs=[];ys=[]
    for low,high in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
        take=(confidence>=low)&(confidence<=high if high>=1 else confidence<high)
        if take.sum():xs.append(confidence[take].mean());ys.append(correct[take].mean())
    ax[2].plot([0,1],[0,1],'--',color='#9cabb3');ax[2].plot(xs,ys,'o-');ax[2].set(xlabel='Predicted confidence',ylabel='Observed correct fraction',xlim=(0,1),ylim=(0,1.04),title='Confidence calibration')
    save(fig,folder/'behavior')
    # 2. Threshold sensitivity and exact native trajectory.
    fig,ax=plt.subplots(1,3,figsize=(10,3.8),constrained_layout=True)
    cr=[r for r in b['chronometry'] if r['cell']==cell]
    ax[0].plot([r['threshold'] for r in cr],[r['crossing_fraction'] for r in cr],'o-',label='Crossed');ax[0].plot([r['threshold'] for r in cr],[r['conditional_accuracy'] if r['conditional_accuracy'] is not None else np.nan for r in cr],'s-',label='Accuracy among crossings');ax[0].set(xlabel='Confidence threshold',ylabel='Fraction',ylim=(0,1.04),title='Threshold sensitivity');ax[0].legend(fontsize=9)
    for r in cr:ax[1].step(np.arange(1,len(r['survival'])+1),r['survival'],where='post',label=str(r['threshold']))
    ax[1].set(xlabel='Frames since evidence onset',ylabel='Not yet crossed',ylim=(0,1.04),title='Uncommitted fraction');ax[1].legend(fontsize=9)
    p=native['temporal'];t=np.arange(p.shape[1]);correctprob=p[np.arange(n)[:,None],t[None,:],native['y'][:,None]]
    ax[2].plot(t,correctprob.mean(0),label='Probability of correct answer');ax[2].plot(t,(p.argmax(-1)==native['y'][:,None]).mean(0),label='Prefix-readout accuracy');ax[2].axvline(onset(task,meta[0],len(t)),ls='--',color='#888888');ax[2].set(xlabel='Model frame',ylabel='Fraction',ylim=(0,1.04),title=cell);ax[2].legend(fontsize=9)
    save(fig,folder/'chronometry')
    # 3. Spatial priority, entropy and correct/error associations.
    a=native['attention'];t=np.arange(a.shape[1]);masses=[];area=[]
    for aa,mm in zip(a,meta):
        s=sites(task,mm);masses.append(np.stack([(aa.reshape(len(aa),100)*mask).sum(1) for mask in s],1));area.append([mask.mean() for mask in s])
    masses=np.array(masses);area=np.array(area);entropy=-(a*np.log(np.maximum(a,1e-12))).sum((2,3));g=native['p'].argmax(1)==native['y']
    fig,ax=plt.subplots(1,3,figsize=(10,3.8),constrained_layout=True)
    for i,name in enumerate(('Designated site','Alternative site','Off-site')):ax[0].plot(t,masses[:,:,i].mean(0),label=name)
    ax[0].set(xlabel='Model frame',ylabel='Attention probability mass',title='Location allocation');ax[0].legend(fontsize=9)
    ax[1].plot(t,np.exp(entropy).mean(0));ax[1].set(xlabel='Model frame',ylabel='Effective patches: exp(entropy)',ylim=(0,101),title='Spatial concentration')
    for good,label in [(True,'Correct'),(False,'Error')]:
        take=g==good
        if take.any():ax[2].plot(t,(masses[take,:,0]-masses[take,:,1]).mean(0),label=f'{label} (n={take.sum()})')
    ax[2].axhline(0,lw=.7,color='#aaaaaa');ax[2].set(xlabel='Model frame',ylabel='Site minus alternative mass',title='Outcome-conditioned allocation');ax[2].legend(fontsize=9)
    save(fig,folder/'attention')
    write(root/'attention_summary.json',dict(cell=cell,n=n,mean_final_site_mass=float(masses[:,-1,0].mean()),mean_final_alternative_mass=float(masses[:,-1,1].mean()),mean_site_area_fraction=float(area[:,0].mean()),mean_final_effective_patches=float(np.exp(entropy[:,-1]).mean()),correct_trials=int(g.sum()),error_trials=int((~g).sum())))
    # 4. Paired causal effects: dose and memory perturbation.
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for site in ('target','distractor'):
        rr=sorted([r for r in c['interventions'] if r['cell']==cell and r['arm'].startswith(site+'_') and r['arm'].endswith('_evidence')],key=lambda r:float(r['arm'].split('_')[1]))
        x=[float(r['arm'].split('_')[1]) for r in rr];y=np.array([r['delta_accuracy'] for r in rr]);lo=np.array([r['ci95'][0] for r in rr]);hi=np.array([r['ci95'][1] for r in rr])
        ax[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',label=site,ms=4,capsize=2)
    ax[0].axhline(0,color='#999999',ls='--');ax[0].set(xlabel='Added priority logit',ylabel='Accuracy change from paired sham',title='Evidence-epoch spatial intervention');ax[0].legend(fontsize=9)
    rr=[r for r in c['interventions'] if r['cell']==cell and '_gain' in r['arm']];y=np.array([r['delta_accuracy'] for r in rr]);lo=np.array([r['ci95'][0] for r in rr]);hi=np.array([r['ci95'][1] for r in rr]);ax[1].barh(range(len(rr)),y,color=[COLORS[0] if 'gain0' in r['arm'] else COLORS[1] for r in rr]);ax[1].errorbar(y,range(len(rr)),xerr=[y-lo,hi-y],fmt='none',color='#273d4b',capsize=2);ax[1].set_yticks(range(len(rr)),[r['arm'].replace('_',' ') for r in rr],fontsize=9);ax[1].axvline(0,color='#999999');ax[1].set(xlabel='Accuracy change from paired sham',title='Memory-state perturbations')
    save(fig,folder/'causal')
    # 5. Independent-scene decoding and sample reconstruction.
    neural_result=nn[0];fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for region,r in neural_result['regions'].items():
        if r.get('constant_zero'):continue
        ax[0].plot(r['diagonal_auc'],label=region);ax[1].plot([np.nan if v is None else v for v in r['sample_content_r2']],label=region)
    ax[0].axhline(.5,color='#999999',ls='--');ax[0].set(xlabel='Model frame',ylabel='Held-out macro AUC',ylim=(0,1.04),title='Correct-answer information');ax[0].legend(fontsize=9,ncol=2)
    ax[1].axhline(0,color='#999999',ls='--');ax[1].set(xlabel='Model frame',ylabel='Held-out content reconstruction R-squared',title='Last sample / study raster descriptor')
    if ax[1].get_ylim()[0]<-1:ax[1].set_yscale('symlog',linthresh=.1);ax[1].set_ylabel('Content R-squared (symmetric-log scale)')
    ax[1].legend(fontsize=9,ncol=2)
    save(fig,folder/'decoding')
    # 6. Cross-temporal decoding and population trajectories, fixed training axes.
    pop=np.load(root/'population'/f'{cell}.npz');fig,ax=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for i,region in enumerate(('cls','spatial')):
        mat=pop[region+'_auc'];im=ax[0,i].imshow(mat,origin='lower',vmin=.5,vmax=1,cmap='viridis',aspect='auto');ax[0,i].set(xlabel='Test frame',ylabel='Decoder training frame',title=region+' temporal generalization');fig.colorbar(im,ax=ax[0,i],label='Macro AUC',shrink=.7)
        traj=pop[region+'_trajectories']
        for label,z in enumerate(traj):
            ax[1,i].plot(z[:,0],z[:,1],'-o',ms=3,label=f'Answer {label}');ax[1,i].scatter(z[0,0],z[0,1],marker='s',s=45);ax[1,i].scatter(z[-1,0],z[-1,1],marker='*',s=90)
        ax[1,i].set(xlabel='Training PCA coordinate 1 (standardized)',ylabel='Training PCA coordinate 2',title=region+' held-out class trajectories');ax[1,i].legend(fontsize=9)
    save(fig,folder/'population')
    # 7. Memory dimensionality, drift and feature code.
    fig,ax=plt.subplots(1,3,figsize=(10,4),constrained_layout=True)
    for region in ('cls','spatial'):
        r=neural_result['regions'][region];ax[0].plot(r['participation_ratio'],label=region);ax[1].plot(r['state_step_rms'],label=region)
        keys=[k for k in r if k.startswith('remembered_')]
        if keys:ax[2].plot(r[keys[0]],label=region+' feature R2')
        elif 'location_diagonal' in r:ax[2].plot(r['location_diagonal'],label=region+' location accuracy')
        else:ax[2].plot(r['class_centroid_variance'],label=region+' centroid variance')
    for aa in ax:aa.set_xlabel('Model frame');aa.legend(fontsize=9)
    ax[0].set(ylabel='Participation ratio in retained subspace',title='Population dimensionality');ax[1].set(ylabel='Root-mean-square step displacement',title='State evolution');ax[2].set(ylabel='Feature R2 / accuracy / variance (legend)',title='Memory-related variable')
    save(fig,folder/'memory')
    # Atlas: show a real trial, its own overlay, and a homogeneous-group mean.
    def group(m,y):return (m.get('target_location',m.get('target_index',m.get('cue_index',-1))),int(y))
    key=group(meta[0],native['y'][0]);take=np.array([group(m,y)==key for m,y in zip(meta,native['y'])]);mean=a[take].mean(0);image=native['example'];ph=phases(task,meta[0],len(image));atlas=[]
    for start in range(0,len(image),6):
        tt=list(range(start,min(start+6,len(image))));fig,ax=plt.subplots(3,len(tt),figsize=(max(3,len(tt)*1.8),5.6),squeeze=False,constrained_layout=True)
        vmax=max(float(a[0,tt].max()),float(mean[tt].max()),.01)
        for j,timestep in enumerate(tt):
            rgb=np.clip(image[timestep].transpose(1,2,0),0,1);ax[0,j].imshow(rgb);ax[1,j].imshow(rgb);ax[1,j].imshow(a[0,timestep],extent=(-.5,99.5,99.5,-.5),cmap='magma',alpha=.65,vmin=0,vmax=vmax,interpolation='nearest');im=ax[2,j].imshow(mean[timestep],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
            ax[0,j].set_title(f't={timestep}\n{ph[timestep]}',fontsize=11)
            for i in range(3):ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
        ax[0,0].set_ylabel('Native RGB');ax[1,0].set_ylabel('Own attention overlay');ax[2,0].set_ylabel(f'Group mean (n={take.sum()})');fig.colorbar(im,ax=ax[2,:].tolist(),label='Probability per patch',shrink=.65,pad=.02)
        name=f'atlas_{start:03d}';save(fig,folder/name);atlas.append(name)
    if c['counterfactual']:
        fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True);arms0=list(dict.fromkeys(r['arm'] for r in c['counterfactual']));xx=np.arange(len(arms0))
        for i,event in enumerate(('cued','uncued','catch')):
            sr=[next(r for r in c['counterfactual'] if r['event']==event and r['arm']==arm) for arm in arms0];ax.bar(xx+(i-1)*.24,[r['report_change'] for r in sr],.24,label=event)
        ax.set_xticks(xx,[a.replace('_',' ') for a in arms0],rotation=25,ha='right',fontsize=9);ax.set(ylabel='Change-report fraction',ylim=(0,1.04),title='100% bottom-right cue: cued change, top-left change, and catch');ax.legend();save(fig,folder/'counterfactual')
    extra=[]
    subgroup_rows=[r for r in c.get('subgroups',[]) if r['cell']==cell]
    if subgroup_rows:
        groups=list(dict.fromkeys(r['group'] for r in subgroup_rows));fig,axes=plt.subplots(1,len(groups),figsize=(4*len(groups),3.6),squeeze=False,constrained_layout=True)
        for ax,group in zip(axes[0],groups):
            for site in ('target','distractor'):
                rr=sorted([r for r in subgroup_rows if r['group']==group and r['arm'].startswith(site+'_') and r['arm'].endswith('_evidence')],key=lambda r:float(r['arm'].split('_')[1]))
                ax.plot([float(r['arm'].split('_')[1]) for r in rr],[r['report_positive'] for r in rr],'o-',label=site,ms=4)
            sham=next(r for r in subgroup_rows if r['group']==group and r['arm']=='sham');ax.axhline(sham['report_positive'],ls='--',color='#999999');ax.set(xlabel='Priority logit dose',ylabel='Positive-response fraction',ylim=(0,1.04),title=f'{group} (n={sham["n"]})');ax.legend(fontsize=9)
        save(fig,folder/'event_causality');extra.append('event_causality')
    if len(nn)>1:
        names=['appearance','fusion','spatial','cls','comparison','attention'];matrix=np.array([[z['regions'][region].get('final_auc',.5) for region in names] for z in nn])
        fig,ax=plt.subplots(figsize=(9,max(3,len(nn)*.5)),constrained_layout=True);im=ax.imshow(matrix,vmin=.5,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(6),names);ax.set_yticks(range(len(nn)),[z['cell'] for z in nn],fontsize=9)
        for i in range(len(nn)):
            for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<.78 else '#172a3b',fontsize=10)
        fig.colorbar(im,ax=ax,label='Held-out final macro AUC');ax.set_title('Condition-resolved population decoding');save(fig,folder/'condition_decoding');extra.append('condition_decoding')
    if task in ('memory_guided_search','motion_memory_search'):
        groups=[(f,s) for f in ('features','objects','fractals') for s in ('distinct','similar')] if task=='memory_guided_search' else [(a,) for a in (20,45,90)]
        for variable in ('exposure','sample_frames'):
            fig,axes=plt.subplots(2 if len(groups)==6 else 1,3,figsize=(12,7 if len(groups)==6 else 3.8),constrained_layout=True);axes=np.array(axes).reshape(-1)
            for ax,group in zip(axes,groups):
                rs=[r for r in b['cells'] if ((r['factors']['family'],r['factors']['similarity'])==group if task=='memory_guided_search' else (r['factors']['spacing'],)==group)]
                xs=sorted(set(r['factors'][variable] for r in rs));ds=sorted(set(r['factors']['delay'] for r in rs))
                matrix=np.array([[np.mean([r['accuracy'] for r in rs if r['factors'][variable]==x and r['factors']['delay']==d]) for x in xs] for d in ds])
                im=ax.imshow(matrix,vmin=.25,vmax=1,aspect='auto',cmap='viridis');ax.set_xticks(range(len(xs)),xs);ax.set_yticks(range(len(ds)),ds);ax.set(xlabel=variable.replace('_',' ')+' (frames)',ylabel='Delay (frames)',title=' / '.join(map(str,group)))
                for i in range(len(ds)):
                    for j in range(len(xs)):ax.text(j,i,f'{100*matrix[i,j]:.0f}',ha='center',va='center',color='white' if matrix[i,j]<.65 else '#173344')
            fig.colorbar(im,ax=axes.tolist(),label='Correct fraction',shrink=.7);name='factorial_'+variable;save(fig,folder/name);extra.append(name)
    write(root/'figure_manifest.json',dict(main=['behavior','chronometry','attention','causal','decoding','population','memory']+(['counterfactual'] if c['counterfactual'] else [])+extra,extra=extra,atlas=atlas,cell=cell,atlas_group=list(key),atlas_group_n=int(take.sum())))
