"""Publication-style saved-score retention figure; CPU only."""
import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;data=json.loads((p/'analysis.json').read_text(encoding='utf-8'))
fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),sharey=True)
colors={'parent':'#8c6475','trained':'#147b8d'};delays=[0,4,12,24]
for ax,family,title,chance in zip(axes,['motion','orientation'],['Motion-duration decision','One-item orientation comparison'],[25,50]):
    for model,label in [('parent','Before retention training'),('trained','After retention training')]:
        rows=[data['cells'][family+'_D'+str(d)] for d in delays];mean=np.array([r[model+'_ba'] for r in rows])*100;ci=np.array([r[model+'_ba_ci95'] for r in rows])*100
        ax.errorbar(delays,mean,yerr=np.maximum(0,np.stack((mean-ci[:,0],ci[:,1]-mean))),fmt='o-',capsize=4,lw=2,ms=6,label=label,color=colors[model])
    ax.axhline(chance,color='#858585',ls=':',lw=1,label='Chance' if family=='motion' else None)
    ax.set(title=title,xlabel='Inserted blank frames',xticks=delays,ylim=(0,104));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Balanced accuracy (%)');axes[0].legend(frameon=False,fontsize=8,loc='lower left')
fig.suptitle('Retention with the existing E/I architecture',fontsize=15,y=.99)
fig.text(.5,.015,'Same evidence across delays; 512 paired test groups per task. Error bars: paired-group bootstrap 95% intervals.',ha='center',fontsize=8,color='#555555')
fig.tight_layout(rect=(0,.065,1,.95));fig.savefig(p/'retention_curves.svg');fig.savefig(p/'retention_curves.png',dpi=180);plt.close(fig)
