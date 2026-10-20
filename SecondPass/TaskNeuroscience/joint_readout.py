"""Exploratory template-to-array matching using two independently fitted readouts."""
import json
import numpy as np
from .design import OUT,neural_cells,retention_boundary,write
from .deep_analysis import kernel_fit
from .sample_reconstruction import targets,fit
from .statistics import paired,holm,interval
from SecondPass.MemoryGuidedSearch.stimuli import render_scene,digest


def candidate_targets(meta):
    rows=[]
    for m in meta:
        sprites,_,_=render_scene(m['seed'],m['family'],m['similarity'])
        assert [digest(s) for s in sprites]==m['sprite_sha256']
        rows.append(sprites)
    return np.array(rows,dtype=float).reshape(len(rows),4,-1)/255


def local_features(a,meta):
    t=meta[0]['array_onset'];field=a['appearance'][:,t].reshape(len(meta),10,10,32);rows=[]
    for i,m in enumerate(meta):
        values=[]
        for x,y in m['centers_xy']:
            col=int(np.argmin(abs(np.arange(10)*10+4.5-x)));row=int(np.argmin(abs(np.arange(10)*10+4.5-y)))
            patch=field[i,row-1:row+2,col-1:col+2];assert patch.shape==(3,3,32);values.append(patch.reshape(-1))
        rows.append(values)
    return np.array(rows,dtype=float)


def candidate_fit(x,y,xt,yt):
    rng=np.random.default_rng(711);order=rng.permutation(len(x));cut=int(.75*len(x));ii,jj=order[:cut],order[cut:];alphas=[.001,.01,.1,1,10];losses=[]
    flat=lambda a:a.reshape(-1,a.shape[-1])
    for alpha in alphas:losses.append(float(np.mean((kernel_fit(flat(x[ii]),flat(y[ii]),flat(x[jj]),alpha)-flat(y[jj]))**2)))
    alpha=alphas[int(np.argmin(losses))];pred=kernel_fit(flat(x),flat(y),flat(xt),alpha).reshape(yt.shape);den=((yt-flat(y).mean(0))**2).sum()
    return pred,dict(alpha=alpha,r2=float(1-((yt-pred)**2).sum()/den),inner_losses=losses)


def run():
    root=OUT/'memory_guided_search';rows=[]
    for cell in neural_cells('memory_guided_search'):
        a=np.load(root/'neural'/f'{cell}_val.npz');b=np.load(root/'neural'/f'{cell}_test.npz');ma=json.loads((root/'neural'/f'{cell}_val.json').read_text());mb=json.loads((root/'neural'/f'{cell}_test.json').read_text())
        cy=candidate_targets(ma);ct=candidate_targets(mb);predicted,cr=candidate_fit(local_features(a,ma),cy,local_features(b,mb),ct)
        sample=targets(ma).reshape(len(ma),-1);st=targets(mb).reshape(len(mb),-1);assert np.array_equal(ct[np.arange(len(mb)),b['y']],st)
        t=retention_boundary('memory_guided_search',mb[0],b['cls'].shape[1]);row=dict(cell=cell,candidate_decoder=cr,agent_accuracy=float((b['p'].argmax(1)==b['y']).mean()),readouts={})
        for region in ('raw_sample','cls','spatial'):
            if region=='raw_sample':template=st
            else:_,template=fit(a[region][:,t],sample,b[region][:,t],st)
            choice=np.mean((predicted-template[:,None])**2,-1).argmin(1);correct=choice==b['y'];row['readouts'][region]=dict(accuracy=float(correct.mean()),ci95=interval(int(correct.sum()),len(correct)),paired_vs_agent=paired(b['y'],np.eye(4)[b['p'].argmax(1)],np.eye(4)[choice]))
        rows.append(row)
        print(cell,cr['r2'],{k:v['accuracy'] for k,v in row['readouts'].items()},flush=True)
    comparisons=[r['readouts']['spatial']['paired_vs_agent'] for r in rows]
    for r,p in zip(comparisons,holm([r['p_value'] for r in comparisons])):r['holm_joint_family']=p
    write(root/'joint_readout.json',dict(results=rows,fit_scenes=192,test_scenes=128,candidate_features=288,candidate_outputs=2352,method='First-array-frame appearance features from the 3x3 grid neighborhood centered on each candidate are decoded by one shared linear ridge map. Four candidates per fitting scene provide 768 examples; penalty selection keeps all candidates from a scene together in a 75/25 scene split. Test scenes are independent. Candidate decoders never see behavioral choice labels. A separately fitted full-state middle-delay template decoder supplies the remembered object. Joint matching uses minimum squared distance between the two decoded pixel vectors; test sample/candidate rasters are used only as reconstruction targets and in the explicitly named raw-sample control. Six spatial joint-readout versus agent comparisons receive Holm correction.'))
    plot(rows)


def plot(rows=None):
    root=OUT/'memory_guided_search'
    if rows is None:rows=json.loads((root/'joint_readout.json').read_text())['results']
    import matplotlib.pyplot as plt
    from .figures import save
    fig,ax=plt.subplots(figsize=(10,4.7),constrained_layout=True);xx=np.arange(len(rows))
    for j,key in enumerate(('agent','raw_sample','cls','spatial')):
        y=np.array([r['agent_accuracy'] if key=='agent' else r['readouts'][key]['accuracy'] for r in rows]);bounds=np.array([interval(round(128*v),128) for v in y])
        ax.bar(xx+(j-1.5)*.2,y,.2,yerr=[y-bounds[:,0],bounds[:,1]-y],capsize=2,error_kw={'elinewidth':.8},label={'agent':'Agent choice','raw_sample':'Exact sample + decoded array','cls':'Decoded CLS + decoded array','spatial':'Decoded spatial + decoded array'}[key])
    ax.axhline(.25,ls='--',color='#888888');ax.set_xticks(xx,[r['cell'].replace('_S8_D12_H8','').replace('_','\n') for r in rows],fontsize=10);ax.set(ylabel='Correct location fraction',ylim=(0,1.04),title='Matching with decoded candidate representations');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,fontsize=10);save(fig,root/'figures/joint_readout')


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    run()
