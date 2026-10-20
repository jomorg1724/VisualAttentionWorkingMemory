"""Full-resolution remembered-object reconstruction on independent scenes."""
import argparse,json,time
import numpy as np
from .design import OUT,neural_cells,retention_boundary,write
from .deep_analysis import kernel_fit
from .statistics import paired,interval,holm
from SecondPass.MemoryGuidedSearch.stimuli import render_scene,digest

def targets(meta):
    out=[]
    for m in meta:
        sprites,_,_=render_scene(m['seed'],m['family'],m['similarity']);sprite=sprites[m['target_index']]
        assert digest(sprite)==m['sprite_sha256'][m['target_index']]
        out.append(sprite.astype(float)/255)
    return np.array(out)

def fit(x,y,xt,yt):
    x=x.reshape(len(x),-1).astype(float);xt=xt.reshape(len(xt),-1).astype(float);rng=np.random.default_rng(711);order=rng.permutation(len(x));cut=int(.75*len(x));ii,jj=order[:cut],order[cut:];alphas=[.001,.01,.1,1,10]
    losses=[]
    for alpha in alphas:losses.append(float(np.mean((kernel_fit(x[ii],y[ii],x[jj],alpha)-y[jj])**2)))
    alpha=alphas[int(np.argmin(losses))];pred=kernel_fit(x,y,xt,alpha);den=((yt-y.mean(0))**2).sum();r2=1-((yt-pred)**2).sum()/den
    return dict(alpha=alpha,r2=float(r2),inner_losses=losses),pred

def candidate_panel(rows,root):
    import matplotlib.pyplot as plt
    from .figures import save
    fig,ax=plt.subplots(figsize=(10,4.7),constrained_layout=True);xx=np.arange(len(rows))
    for j,region in enumerate(('agent','cls','spatial')):
        values=np.array([r['agent_accuracy'] if region=='agent' else r['regions'][region]['mid_delay']['candidate_matching_accuracy'] for r in rows])
        bounds=np.array([interval(round(r['agent_accuracy']*128),128) if region=='agent' else r['regions'][region]['mid_delay']['candidate_matching_ci95'] for r in rows])
        ax.bar(xx+(j-1)*.25,values,.25,label='Agent choice' if region=='agent' else ('CLS' if region=='cls' else 'Spatial')+' memory matcher')
        ax.errorbar(xx+(j-1)*.25,values,yerr=[values-bounds[:,0],bounds[:,1]-values],fmt='none',color='#233846',capsize=2,lw=1)
    ax.axhline(.25,ls='--',color='#888888');ax.set_xticks(xx,[r['cell'].replace('_S8_D12_H8','').replace('_','\n') for r in rows],fontsize=10);ax.set(ylabel='Correct location fraction',ylim=(0,1.04),title='Can reconstructed delay content identify the matching candidate?');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,fontsize=10);save(fig,root/'figures/reconstructed_template_matching')

def run():
    import matplotlib.pyplot as plt
    from .figures import save
    task='memory_guided_search';root=OUT/task;rows=[];examples=[]
    for cell in neural_cells(task):
        a=np.load(root/'neural'/f'{cell}_val.npz');b=np.load(root/'neural'/f'{cell}_test.npz');ma=json.loads((root/'neural'/f'{cell}_val.json').read_text());mb=json.loads((root/'neural'/f'{cell}_test.json').read_text())
        y=targets(ma);yt=targets(mb);flat=y.reshape(len(y),-1);test=yt.reshape(len(yt),-1);m=mb[0];times={'encoding':m['sample_frames'],'mid_delay':retention_boundary(task,m,a['cls'].shape[1]),'first_search':m['array_onset'],'final':a['cls'].shape[1]-1};row=dict(cell=cell,times=times,regions={},agent_accuracy=float((b['p'].argmax(1)==b['y']).mean()));example=dict(cell=cell,truth=yt[0])
        candidates=np.array([render_scene(m['seed'],m['family'],m['similarity'])[0] for m in mb],dtype=float).reshape(len(yt),4,-1)/255
        assert np.array_equal(candidates[np.arange(len(yt)),b['y']],test), 'Candidate labels must identify the exact remembered raster'
        for region in ('appearance','cls','spatial'):
            row['regions'][region]={}
            ar=a[region];br=b[region]
            for name,t in times.items():
                result,pred=fit(ar[:,t],flat,br[:,t],test)
                choice=np.mean((pred[:,None]-candidates)**2,axis=-1).argmin(1);correct=choice==b['y']
                result.update(candidate_matching_accuracy=float(correct.mean()),candidate_matching_ci95=interval(int(correct.sum()),len(correct)),paired_vs_agent=paired(b['y'],np.eye(4)[b['p'].argmax(1)],np.eye(4)[choice]))
                row['regions'][region][name]=result
                if name=='mid_delay' and region in ('cls','spatial'):example[region]=pred[0].reshape(28,28,3)
        rows.append(row);examples.append(example)
    fig,axes=plt.subplots(1,3,figsize=(12,4.7),constrained_layout=True)
    for ax,region in zip(axes,('appearance','cls','spatial')):
        matrix=np.array([[r['regions'][region][k]['r2'] for k in ('encoding','mid_delay','first_search','final')] for r in rows]);im=ax.imshow(matrix,vmin=-.2,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(4),['encoding','delay','search 1','final'],rotation=35,ha='right');ax.set_yticks(range(6),[r['cell'].replace('_S8_D12_H8','') for r in rows],fontsize=8);ax.set_title(region)
        for i in range(6):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<.65 else '#172f40',fontsize=8)
    fig.colorbar(im,ax=axes.tolist(),label='Held-out full-crop R-squared',shrink=.7);save(fig,root/'figures/full_crop_reconstruction')
    candidate_panel(rows,root)
    for start in (0,3):
        fig,axes=plt.subplots(3,3,figsize=(8,8),constrained_layout=True)
        for i,r in enumerate(examples[start:start+3]):
            for j,key in enumerate(('truth','cls','spatial')):
                axes[i,j].imshow(np.clip(r[key],0,1));axes[i,j].set_xticks([]);axes[i,j].set_yticks([]);axes[i,j].set_title('Original sample' if key=='truth' else key+' delay reconstruction',fontsize=10)
            axes[i,0].set_ylabel(r['cell'].replace('_S8_D12_H8','').replace('_','\n'),fontsize=9)
        save(fig,root/f'figures/reconstructed_examples_{start:02d}')
    comparisons=[v['paired_vs_agent'] for r in rows for region in r['regions'].values() for v in region.values()]
    for r,p in zip(comparisons,holm([r['p_exact'] for r in comparisons])):r['holm_reconstruction_family']=p
    write(root/'full_crop_reconstruction.json',dict(results=rows,train_trials=192,test_trials=128,target_shape=[28,28,3],fit='Full-coordinate linear ridge, alpha selected on an inner 75/25 fitting split; independent held-out R-squared; regenerated target hashes verified',candidate_match='External probe chooses the candidate raster with minimum squared pixel distance to reconstructed sample; compared with original agent on the same neural test trials. This is an information readout, not the agent policy.',display='First held-out example in each condition; display clipped to [0,1], metrics use unbounded predictions'))

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();deadline=time.time()+16*3600
    needed=[OUT/'memory_guided_search/neural'/f'{cell}_{split}.npz' for cell in neural_cells('memory_guided_search') for split in ('val','test')]
    while not all(p.exists() for p in needed):
        if not a.wait:return
        if time.time()>deadline:raise TimeoutError('Neural recording wait')
        time.sleep(20)
    run()

if __name__=='__main__':main()
