"""Replot frozen-model measurements for the neuroscience synthesis; no inference."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'SecondPass/TaskNeuroscience/Results'
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
used={}
def read(task,name):
    p=DATA/task/(name+'.json') if task else DATA/(name+'.json')
    used[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
BLUE='#21618c';ORANGE='#bd5b36';GREEN='#297b67';PURPLE='#775799';GRAY='#6a7480'
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def errs(rows,key='accuracy',ci='accuracy_ci95',scale=100):
    v=np.array([r[key] for r in rows])*scale;c=np.array([r[ci] for r in rows])*scale
    return v,np.maximum(0,np.stack([v-c[:,0],c[:,1]-v]))
def arm(task,name,cell=None):
    return next(r for r in read(task,'causal')['interventions'] if r['arm']==name and (cell is None or r['cell']==cell))

# Sensory challenge curves retain physical axes and native label accuracy.
fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained')
for ax,t,label,chance in zip(axs.flat,['motion_direction','orientation','contrast','spatial_frequency'],
 ['Dot displacement (pixels/transition)','Orientation difference (degrees)','Contrast increment (image units)','Spatial-frequency difference (octaves)'],[25,50,50,50]):
    rows=sorted([r for r in read(t,'behavior')['psychometric'] if r['event']=='native'],key=lambda r:r['strength'])
    v,e=errs(rows);ax.errorbar([r['strength'] for r in rows],v,yerr=e,fmt='o-',color=BLUE,capsize=2,ms=4)
    ax.axhline(chance,color=GRAY,ls=':',lw=1);ax.set(xlabel=label,ylabel='Correct (%)',ylim=(0,105),title=t.replace('_',' ').capitalize())
save(fig,'sensory_curves')

# Different information in successive processing stages and reset costs.
tasks=['motion_direction','orientation','contrast','spatial_frequency','chromatic_increment','contour','natural_spectrum']
names=['Motion direction','Orientation order','Contrast increment','Spatial frequency','Chromatic increment','Contour arrangement','Natural spectrum']
fig,axs=plt.subplots(1,2,figsize=(7.2,3.7),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
mat=[];resets=[]
for t in tasks:
    n=read(t,'neural_summary')[0]['regions'];mat.append([100*n[k]['final_balanced_accuracy'] for k in ['appearance','fusion','spatial','cls']])
    rows=read(t,'causal')['interventions'];resets.append([-100*next(r['delta_accuracy'] for r in rows if r['arm'].startswith(k+'_gain0')) for k in ['spatial','cls']])
im=axs[0].imshow(mat,vmin=0,vmax=100,cmap='Blues',aspect='auto');axs[0].set_yticks(range(7),names);axs[0].set_xticks(range(4),['Appearance','Fusion','Spatial','Global'],rotation=30,ha='right');axs[0].set_title('A  Answer decoding (%)')
im2=axs[1].imshow(resets,vmin=0,vmax=50,cmap='Oranges',aspect='auto');axs[1].set_yticks(range(7),['']*7);axs[1].set_xticks(range(2),['Spatial reset','Global reset'],rotation=30,ha='right');axs[1].set_title('B  Accuracy loss (pp)')
for ax,m in zip(axs,[mat,resets]):
 for i,row in enumerate(m):
  for j,v in enumerate(row):ax.text(j,i,f'{v:.0f}',ha='center',va='center',color='white' if (v>70 if ax==axs[0] else v>32) else '#18212b',fontsize=9)
save(fig,'sensory_processing')

items=[('orientation_ring',None,'Ring-cued orientation'),('orientation_cued',None,'Rule-cued orientation'),('spatial_binding',None,'Spatial binding'),('motion_duration_cued',None,'Motion accumulation'),('krauzlis_cued_motion',None,'Cued motion change'),('motion_memory_search','motion_A20_S12_D12_H12','Motion search: 20 degrees'),('motion_memory_search','motion_A90_S12_D12_H12','Motion search: 90 degrees')]
fig,ax=plt.subplots(figsize=(7.2,3.6),layout='constrained');y=np.arange(len(items))
for offset,an,color,label in [(-.15,'target_-4_evidence',ORANGE,'Suppress target priority'),(.15,'distractor_+4_evidence',BLUE,'Increase distractor priority')]:
 rows=[arm(t,an,c) for t,c,_ in items];v,e=errs(rows,'delta_accuracy','ci95');ax.errorbar(v,y+offset,xerr=e,fmt='o',color=color,label=label,capsize=2)
ax.set_yticks(y,[r[2] for r in items]);ax.invert_yaxis();ax.axvline(0,color=GRAY,ls=':');ax.set(xlabel='Accuracy change from sham (percentage points)',xlim=(-102,8));ax.legend(loc='lower left',fontsize=8)
save(fig,'priority_causality')

# Condition-averaged maps: maps and curves use exactly the same trial subset.
fig=plt.figure(figsize=(7.2,5.0),layout='constrained');gs=fig.add_gridspec(2,6,width_ratios=[1,1,1,1,1.1,1.1])
for row,(t,idx,title,times) in enumerate([
 ('orientation_cued',0,'Cued orientation',[1,2,15,27]),
 ('motion_memory_search',0,'Motion-template search',[2,12,19,36])]):
 g=read(t,'attention_dynamics')['groups'][idx];maps=np.array(g['mean_map']);mass=np.array(g['mean_mass']);lo=np.array(g['ci_low']);hi=np.array(g['ci_high'])
 for col,tm in enumerate(times):
  ax=fig.add_subplot(gs[row,col]);im=ax.imshow(maps[tm],vmin=0,vmax=.12,cmap='magma',origin='upper');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f"{g['phases'][tm]}\nt = {tm}",fontsize=9)
  if col==0:ax.set_ylabel(title+'\n'+f"n = {g['n']}",fontsize=9)
 ax=fig.add_subplot(gs[row,4:]);
 for j,color in [(0,BLUE),(1,GRAY),(2,GREEN)]:
  if j>=mass.shape[1]:continue
  label=g['region_names'][j].replace(' region','').replace('regions (mean)','sites (mean)')
  ax.plot(g['time'],mass[:,j],color=color,label=label,lw=1.5);ax.fill_between(g['time'],lo[:,j],hi[:,j],color=color,alpha=.12)
 ax.set(xlabel='Model frame',ylabel='Regional priority mass',ylim=(0,1));ax.legend(fontsize=6.5,loc='best');
fig.colorbar(im,ax=fig.axes[:4]+fig.axes[5:9],shrink=.65,label='Priority per patch',location='bottom',pad=.02)
save(fig,'attention_time')

fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
for ax,event,title in zip(axs,['cued','uncued'],['Change at inhibited cue','Change at opposite location']):
 rows=[next(r for r in read(t,'weak_counterfactual_summary') if r['strength']==8 and r['event']==event and r['arm']=='cued_-4_change') for t in ['vda4','vda9','vda16']]
 v,e=errs(rows,'delta_accuracy','ci95');ax.errorbar(range(3),v,yerr=e,fmt='o',color=ORANGE if event=='cued' else BLUE,capsize=4)
 ax.axhline(0,color=GRAY,ls=':');ax.set_xticks(range(3),['VDA4','VDA9','VDA16']);ax.set(title=title,ylabel='Change in detection (pp)',ylim=(-50,18));ax.set_xlim(-.5,2.5)
save(fig,'vda_dual')

fig,axs=plt.subplots(2,2,figsize=(7.2,4.9),layout='constrained')
for ax,t,key,title,delay in [(axs[0,0],'orientation_cued','remembered_orientation_cos2_sin2_r2','A  Retained cued orientation',(3,26)),(axs[0,1],'motion_memory_search','remembered_direction_cos_sin_r2','B  Retained motion direction',(13,24))]:
 d=read(t,'neural_summary')[0]
 for reg,col,label in [('cls',BLUE,'Global memory'),('spatial',GREEN,'Spatial memory')]:ax.plot(d['regions'][reg][key],color=col,label=label)
 ax.axvspan(*delay,color='#dce4ec',alpha=.45);ax.set(title=title,xlabel='Model frame',ylabel='Held-out content R-squared',ylim=(-.1,1.05));ax.legend(fontsize=7)
d=read('spatial_binding','orientation_inventory')
for reg,col,ls,label in [('spatial',GREEN,'-','Spatial: each location'),('cls',BLUE,'--','Global: each location')]:
 vals=np.array(d['regions'][reg]['r2_by_frame_and_location'])
 for j in range(4):axs[1,0].plot(vals[:,j],color=col,ls=ls,alpha=.65,label=label if j==0 else None)
axs[1,0].axvspan(3,26,color='#dce4ec',alpha=.45);axs[1,0].set(title='C  Four-location orientation inventory',xlabel='Model frame',ylabel='Held-out content R-squared',ylim=(-.2,1.05));axs[1,0].legend(fontsize=7)
for j,(t,label) in enumerate([('orientation_cued','Orientation'),('spatial_binding','Binding'),('motion_memory_search','Motion search')]):
 rows=read(t,'causal')['interventions'];cell=rows[0]['cell']
 for off,reg,col in [(-.17,'cls',BLUE),(.17,'spatial',GREEN)]:
  r=next(r for r in rows if r['cell']==cell and r['arm'].startswith(reg+'_gain0'));v,e=errs([r],'delta_accuracy','ci95');axs[1,1].errorbar(j+off,v,yerr=e,fmt='o',color=col,capsize=2)
axs[1,1].axhline(0,color=GRAY,ls=':');axs[1,1].set_xticks(range(3),['Orientation','Binding','Motion\nsearch']);axs[1,1].set(title='D  Reset during retention',ylabel='Accuracy change (pp)',ylim=(-90,8))
save(fig,'memory_content')

fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
rows=sorted([r for r in read('image_recognition','literature_analysis')['rows'] if r['probe_hold']==5 and r['load']>0],key=lambda r:r['load'])
for k,col,label in [('hit_rate',BLUE,'Repeat hits'),('false_alarm_rate',ORANGE,'Novel false alarms')]:axs[0].plot([r['load'] for r in rows],[100*r[k] for r in rows],'o-',color=col,label=label)
axs[0].set(xlabel='Studied images',ylabel='Positive reports (%)',ylim=(0,105));axs[0].legend(fontsize=8)
axs[1].plot([r['load'] for r in rows],[r['dprime'] for r in rows],'o-',color=GREEN,label="Sensitivity d-prime");axs[1].plot([r['load'] for r in rows],[r['criterion'] for r in rows],'s--',color=PURPLE,label='Criterion c');axs[1].axhline(0,color=GRAY,ls=':');axs[1].set(xlabel='Studied images',ylabel='Signal-detection coordinate');axs[1].legend(fontsize=8)
save(fig,'recognition')

fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained');rows=read('motion_duration_cued','literature_analysis')['rows']
for strat,col,label in [('first_transition',GRAY,'First direction'),('last_transition',ORANGE,'Last direction'),('last_three_transitions',PURPLE,'Last three directions')]:
 r=sorted([x for x in rows if x['strategy']==strat],key=lambda x:x['delay']);xx=[x['delay'] for x in r]
 axs[0].plot(xx,[100*x['heuristic_accuracy'] for x in r],'o--',color=col,label=label)
 v,e=errs(r,'model_conflict_accuracy','model_conflict_ci95');axs[1].errorbar(xx,v,yerr=e,fmt='o-',color=col,label=label,capsize=2)
r=sorted([x for x in rows if x['strategy']=='first_transition'],key=lambda x:x['delay']);axs[0].plot([x['delay'] for x in r],[100*x['model_accuracy'] for x in r],'o-',color=BLUE,label='Trained model')
for ax in axs:ax.axhline(25,color=GRAY,ls=':');ax.set(xlabel='Retention delay (frames)',ylabel='Correct (%)',ylim=(15,105));ax.legend(fontsize=7)
axs[0].set_title('A  All trials');axs[1].set_title('B  Model when shortcut is wrong');save(fig,'temporal_evidence')

fig,axs=plt.subplots(2,3,figsize=(7.2,4.8),layout='constrained')
for j,(t,cell,title) in enumerate([('memory_guided_search','features_distinct_S8_D12_H8','Distinct features'),('memory_guided_search','objects_distinct_S8_D12_H8','Distinct contour objects'),('motion_memory_search','motion_A20_S12_D12_H12','Motion template, 20 degrees')]):
 r=next(r for r in read(t,'query_switch_summary') if r['cell']==cell)
 for i,key in enumerate(['mean_probability_matrix','mean_attention_matrix']):
  ax=axs[i,j];im=ax.imshow(r[key],vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),['TL','TR','BL','BR']);ax.set_yticks(range(4),['TL','TR','BL','BR']);ax.set_xlabel('Candidate location');ax.set_ylabel('Sample matches location' if j==0 else '');ax.set_title(title if i==0 else f"Priority gain: {r['template_attention_gain']:.3f}",fontsize=9)
  for y in range(4):
   for x in range(4):
    v=r[key][y][x];ax.text(x,y,f'{v:.2f}',ha='center',va='center',fontsize=6.5,color='white' if v>.55 else '#18212b')
fig.colorbar(im,ax=axs,location='right',shrink=.75,label='Probability / regional priority mass');save(fig,'query_switch')

rows=read('memory_guided_search','joint_readout')['results'];fig,ax=plt.subplots(figsize=(7.2,3.3),layout='constrained');xx=np.arange(6)
ax.bar(xx-.18,[100*r['agent_accuracy'] for r in rows],width=.34,color=GRAY,label='Agent choice')
v,e=errs([r['readouts']['spatial'] for r in rows],'accuracy','ci95');ax.bar(xx+.18,v,width=.34,color=GREEN,label='External decoded-template matcher');ax.errorbar(xx+.18,v,yerr=e,fmt='none',color='#243746',capsize=3)
ax.axhline(25,color=GRAY,ls=':');ax.set_xticks(xx,['Features\ndistinct','Features\nconjunction','Objects\ndistinct','Objects\nsimilar','Fractals\ndistinct','Fractals\nsimilar']);ax.set(ylabel='Correct location (%)',ylim=(0,114));ax.legend(loc='upper center',ncol=2,fontsize=8);save(fig,'available_information')

fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained');rows=read('popout_search','behavior')['cells']
for ax,feature in zip(axs,['color','orientation']):
 diffs=sorted({r['factors']['difference'] for r in rows if r['factors']['feature']==feature})
 for diff,col in zip(diffs,[ORANGE,BLUE,GREEN]):
  r=sorted([r for r in rows if r['factors']['feature']==feature and r['factors']['difference']==diff],key=lambda x:x['factors']['exposure']);v,e=errs(r)
  ax.errorbar([x['factors']['exposure'] for x in r],v,yerr=e,fmt='o-',color=col,capsize=2,label=f'{diff}'+(' degrees' if feature=='orientation' else ' RGB units'))
 ax.axhline(25,color=GRAY,ls=':');ax.set(title=feature.capitalize()+' singleton',xlabel='Array exposure (frames)',ylabel='Correct location (%)',ylim=(15,105),xticks=[1,2,4,8]);ax.legend(fontsize=8)
save(fig,'popout_time')

# Query-selected neurons versus information needed at different times.
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
for ax,t,title in zip(axs,['orientation_cued','motion_duration_cued'],['Rule-cued orientation','Accumulated motion direction']):
 d=read(t,'neural_summary')[0]
 for reg,col,label in [('appearance',GRAY,'Appearance'),('spatial',GREEN,'Spatial memory'),('cls',BLUE,'Global memory')]:ax.plot(100*np.array(d['regions'][reg]['diagonal']),color=col,label=label)
 ax.set(title=title,xlabel='Model frame',ylabel='Answer decoding (%)',ylim=(0,105));ax.legend(fontsize=7)
save(fig,'answer_emergence')

# Compact, fully traceable tabular evidence used by the manuscript.
alltasks=[p.name for p in DATA.iterdir() if (p/'behavior.json').exists()]
summary=[]
for t in alltasks:
 d=read(t,'behavior');summary.append({k:d[k] for k in ['task','conditions','trials','equal_cell_accuracy','equal_cell_ci95']})
(OUT/'task_summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'source_manifest.json').write_text(json.dumps({'checkpoint_step':204289,'checkpoint_sha256':'596bc351783b3ddbfa8d37a9cf8d65146d1333dcc0097d40f0f75665b6451bcf','source_files':used,'new_model_inference':False},indent=2))
print('Created',len(list(FIG.glob('*.pdf'))),'figures from',len(used),'saved source files.')
