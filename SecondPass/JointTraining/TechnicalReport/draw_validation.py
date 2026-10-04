"""Build figures only from complete, saved validation artifacts (no inference)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
# Fixed document cutoff: use the audited snapshot, never advancing live logs.
snapshot=json.loads((HERE/'metrics_snapshot.json').read_text())
evaluations=[e for e in snapshot['evaluations'] if e['role'].startswith('v3_validation')]
evaluations.sort(key=lambda e:e['step'])
rows=[]
for evaluation in evaluations:
    row=evaluation['result']
    assert row['complete'] and row['complete_cells']==35 and len(row['summary']['tasks'])==13
    rows.append(row)
steps=[e['step'] for e in evaluations]
assert steps==[715,1248,1781,2314]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42})
fig, axes=plt.subplots(1,2,figsize=(7.05,1.85),sharey=True)
series=[('All tasks',[r['summary']['equal_task_mean_auc'] for r in rows],'#19304A'),
        ('Sensory',[r['summary']['groups']['sensory']['auc'] for r in rows],'#B56716'),
        ('Spatial / sequence',[r['summary']['groups']['spatial']['auc'] for r in rows],'#317C7A')]
for label,values,color in series: axes[0].plot(steps,values,'o-',label=label,color=color,lw=1.4,ms=3)
for task,label,color in [('orientation','Signed orientation','#9D4B62'),('motion_direction','Motion direction','#386AA6'),('contour','Contour','#927340'),('image_recognition','Recognition','#527D62')]:
    axes[1].plot(steps,[r['summary']['tasks'][task]['auc'] for r in rows],'o-',label=label,color=color,lw=1.2,ms=3)
for ax in axes:
    ax.axhline(.5,color='#777777',ls='--',lw=.8)
    ax.set_ylim(.4,1.02); ax.set_xticks(steps); ax.tick_params(axis='x',labelsize=6)
    ax.set_xlabel('Cumulative optimizer update'); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15); ax.legend(loc='upper left',fontsize=6,frameon=False)
axes[0].set_ylabel('Validation AUC'); axes[0].set_title('Group averages: gains are concentrated',fontsize=8)
axes[1].set_title('Emerging score-level signals',fontsize=8)
fig.tight_layout(pad=.4,w_pad=1.2)
fig.savefig(HERE/'validation_curves.pdf',bbox_inches='tight'); fig.savefig(HERE/'validation_curves.png',dpi=180,bbox_inches='tight')
print(json.dumps({'steps':steps,'all_task_auc':[r['summary']['equal_task_mean_auc'] for r in rows],'source_files':[e['source']['path'] for e in evaluations]},indent=2))
