"""Saved-score figure; no model loading or new inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
a=json.loads((P/'analysis.json').read_text());p=json.loads((P/'parent_reference.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,task,delays,chance,title in [(axes[0],'single',[0,4,12,24],50,'Existing single-item orientation'),(axes[1],'motion',[0,24],25,'Existing motion-duration decision')]:
    for arm,label,color in [('parent','Unchanged parent14800','#666666'),('dense_comparator','Dense + comparator','#21618c'),('spatial_ei','Spatial E/I + comparator','#a93226')]:
        rows=[p['cells'][f'{task}_D{d}']['parent'] if arm=='parent' else a['cells'][f'{task}_D{d}'][arm] for d in delays]
        y=np.array([r['ba'] for r in rows])*100;ci=np.array([r['ba_ci95'] for r in rows])*100
        ax.errorbar(delays,y,yerr=np.maximum(0,np.stack((y-ci[:,0],ci[:,1]-y))),marker='o',capsize=3,color=color,label=label)
    ax.axhline(chance,ls=':',color='.6');ax.set(title=title,xlabel='Inserted blank frames',ylabel='Balanced accuracy (%)',xticks=delays,ylim=(18,102));ax.legend(fontsize=8)
fig.savefig(P/'parent_tradeoff.png',dpi=180);fig.savefig(P/'parent_tradeoff.svg');plt.close(fig)
