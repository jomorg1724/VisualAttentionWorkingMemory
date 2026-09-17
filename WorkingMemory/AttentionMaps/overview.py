"""CPU-only corrections from already captured attention tensors."""
import json,base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def main():
    groups=json.loads((HERE/'condition_maps.json').read_text());reps=json.loads((HERE/'representatives.json').read_text());phases=['sample','blank','probe'];maps={};counts={}
    for phase in phases:
        rows=[g for g in groups if g['family']=='orientation_single' and g['delay']==24 and g['phase']==phase];n=sum(g['n'] for g in rows);counts[phase]=n
        maps[phase]={k:sum(np.array(g[k])*g['n'] for g in rows)/n for k in ('receiver_map','key_map')}
    for key,title,name in [('receiver_map','Every receiving memory-grid site: visual vs memory allocation','phase_allocation_overview'),('key_map','Source-key usage, averaged over all169 receiving queries','phase_source_keys_overview')]:
        vmax=1 if key=='receiver_map' else max(float(maps[p][key].max()) for p in phases)
        fig,axs=plt.subplots(4,3,figsize=(12,14),layout='constrained')
        for row in range(4):
            h,s=divmod(row,2)
            for col,phase in enumerate(phases):
                ax=axs[row,col];a=maps[phase][key][h,s].reshape(13,13);im=ax.imshow(a,cmap='viridis',vmin=0,vmax=vmax,interpolation='nearest');ax.set_xticks(range(13));ax.set_yticks(range(13));ax.tick_params(labelsize=7);ax.set_xticks(np.arange(-.5,13,1),minor=True);ax.set_yticks(np.arange(-.5,13,1),minor=True);ax.grid(which='minor',color='white',lw=.35,alpha=.7);ax.tick_params(which='minor',bottom=False,left=False)
                if col==0:ax.set_ylabel(f'Head{h+1} · '+('visual source' if s==0 else 'memory source')+'\nGrid row',fontsize=11)
                mass=float(a.mean() if key=='receiver_map' else a.sum());ax.set_title(f'{phase.capitalize()} · source mass {mass:.1%}',fontsize=12);ax.set_xlabel('Grid column')
        fig.suptitle(title+'\nSingle orientation, D24; '+str(counts['blank'])+' episodes; actual13×13grid, no smoothing',fontsize=16)
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.5,label='Source fraction per receiving query' if key=='receiver_map' else 'Joint attention weight per source key')
        fig.savefig(HERE/'figures'/f'{name}.png',dpi=145);fig.savefig(HERE/'figures'/f'{name}.svg');plt.close(fig)
    xy=np.stack(np.meshgrid(np.arange(13),np.arange(13)),axis=-1).reshape(169,2);distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);cheb=np.max(np.abs(xy[:,None]-xy[None,:]),axis=-1);stats=[]
    for r in reps:
        w=np.frombuffer(base64.b64decode(r['weights']),dtype='<f4').reshape(2,169,2,169);joint=w.sum(2)
        for h in (0,1):stats.append(dict(id=r['id'],family=r['family'],phase=r['phase'],head=h+1,same_coordinate_mass=float(np.mean((joint[h]*(distance==0)).sum(-1))),within_3x3_mass=float(np.mean((joint[h]*(cheb<=1)).sum(-1))),mean_distance_grid_cells=float(np.mean((joint[h]*distance).sum(-1))),scope='one captured representative trial, mean over169 queries'))
    result=dict(status='completed_CPU_only',source='existing representative_full_weights and per-trial phase summaries',trained_locality_coefficients=[4.03160143,4.02760696],coefficient_provenance='root read frozencheckpoint8400 onCPU',locality_penalty='-softplus(raw_locality[h]) times squared grid distance; nearest-neighbor preference ratio exp(-4.03) before content≈0.0178',representative_locality=stats,overview_n=counts,interpretation='Strong local connectivity: the old single-query display showed mostly one grid cell because around94% of joint weight stays at the same coordinate. Whole-grid source allocation is the relevant overview; across-query key means are not object saliency.')
    (HERE/'locality_summary.json').write_text(json.dumps(result,indent=2))
    note='\n\n## Corrected overview: why the first display looked like one cell\n\nThe first promoted figure displayed connectivity from one receiving query, not the full13×13population. That was an unsuitable default. The single cell also reflects real strongly local connectivity: across representative orientation frames, approximately94% of joint weight stays at the same spatial coordinate. The trained squared-distance penalty coefficient is about4.03, giving an adjacent-coordinate factor exp(-4.03)≈0.0178 before content scores.\n\nThe corrected default shows all169 receiving memory-grid sites and their visual-versus-memory allocation, with one fixed0–1scale. [Sample/blank/probe overview](figures/phase_allocation_overview.png) shows all64orientation episodes; [query-averaged source keys](figures/phase_source_keys_overview.png) are a separate view and should not be called object saliency. [Locality measurements](locality_summary.json) use cached matrices only. The earlier blank_query_maps figure is preserved solely as advanced single-query connectivity. No new capture, GPU work or model experiment was performed for this correction.\n'
    p=HERE/'report.md';text=p.read_text(encoding='utf-8');mark='\n\n## Corrected overview:'
    if mark in text:text=text.split(mark)[0]
    p.write_text(text+note,encoding='utf-8')
if __name__=='__main__':main()
