"""Replot completed measurements only; no inference, training or source-result edits."""
from pathlib import Path
import json, csv, hashlib, collections, statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[4]
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/independent_20261011'
OUT.mkdir(parents=True,exist_ok=True)
sources={}
def source(name):
    p=REPO/name
    sources[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    return p
def js(name):return json.loads(source(name).read_text(encoding='utf-8'))
def rows(name):return list(csv.DictReader(source(name).open(encoding='utf-8',newline='')))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white','pdf.fonttype':42})
TEAL='#087f81'; BLUE='#315a94'; CORAL='#c66249'; GREY='#66737b'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

latest=js('SecondPass/MemoryGuidedSearch/LocalRuntime/continuation03/final_test.json')
ready=js('SecondPass/MemoryGuidedSearch/LocalRuntime/continuation03/continuation_ready.json')
assert ready['best']['step']==55100 and len(latest['cells'])==437
groups=collections.defaultdict(list)
for c in latest['cells']:groups[c['task']].append(c)
titles={'motion_direction':'Motion direction','orientation':'Orientation difference','contrast':'Contrast difference','spatial_frequency':'Spatial frequency','chromatic_increment':'Chromatic increment','contour':'Contour organization','natural_spectrum':'Natural-image spectrum','orientation_ring':'Ring-cued orientation','orientation_cued':'Signed cued orientation','motion_duration_cued':'Cued motion-duration memory','krauzlis_cued_motion':'Native cued motion change','spatial_binding':'Spatial binding','image_recognition':'Image-list recognition','vda4':'VDA-4','vda9':'VDA-9','vda16':'VDA-16','memory_guided_search':'Object-memory search','popout_search':'Pop-out search','motion_memory_search':'Motion-memory search'}
table=[]
for task,cells in groups.items():
    ba=[x['ba'] for x in cells if x['ba'] is not None]
    auc=[x['auc'] for x in cells if x['auc'] is not None]
    classes=len(cells[0]['predictions'][0]['probabilities'])
    table.append({'task':task,'title':titles[task],'cells':len(cells),'n':sum(x['n'] for x in cells),'ba':statistics.mean(ba),'auc':statistics.mean(auc),'classes':classes,'chance':1/classes})
fig,axs=plt.subplots(1,2,figsize=(11.4,8.6),sharey=True)
y=np.arange(len(table)); color=[CORAL if x['task'] in ('memory_guided_search','popout_search','motion_memory_search') else TEAL for x in table]
axs[0].barh(y,[x['ba'] for x in table],color=color,height=.68)
axs[0].scatter([x['chance'] for x in table],y,marker='|',s=180,color='#172d36',label='Class-balanced chance')
axs[1].barh(y,[x['auc'] for x in table],color=color,height=.68)
axs[1].axvline(.5,color=GREY,ls='--',lw=1)
for ax,key in zip(axs,('ba','auc')):
    ax.set_xlim(0,1.15);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xticklabels(['0','25','50','75','100']);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for i,x in enumerate(table):ax.text(x[key]+.015,i,f'{100*x[key]:.1f}',va='center',fontsize=9)
axs[0].set_yticks(y,[titles[x['task']] for x in table]);axs[0].invert_yaxis()
axs[0].set_title('Balanced accuracy (%)');axs[1].set_title('AUC (%)');axs[0].legend(loc='lower right',fontsize=8)
fig.suptitle('One shared backbone across 19 trained tasks',fontsize=16,y=.99)
fig.text(.03,.015,'Selected checkpoint 55,100; 111,872 fresh test presentations in 437 conditions. Equal-condition task means.\nEmpty-list recognition contributes specificity only and is excluded from BA/AUC. No across-seed uncertainty; new search tasks in orange.',fontsize=9)
fig.subplots_adjust(left=.28,right=.97,top=.93,bottom=.10,wspace=.14)
save(fig,'01_task_capabilities')

parent=js('SecondPass/StabilizedAttentionSuite/LocalRuntime/continuation02/final_test.json')
def cell_ba(c):return c.get('ba',c.get('balanced_accuracy'))
fig,axs=plt.subplots(1,2,figsize=(11.4,4.8))
old=[c for c in parent['cells'] if c['task']=='krauzlis_cued_motion'];new=groups['krauzlis_cued_motion']
old.sort(key=lambda c:int(c['cell'][1:]));new.sort(key=lambda c:int(c['cell'][1:]))
rng=np.random.default_rng(20261011)
def ba_ci(c):
    pred=c['predictions'];labs=np.array([r['label'] for r in pred]);choices=np.array([np.argmax(r.get('probabilities',r.get('probs'))) for r in pred]); cls=np.unique(labs)
    correct=[(choices[labs==k]==k).astype(float) for k in cls]
    dist=np.mean([rng.choice(a,(2000,len(a)),replace=True).mean(1) for a in correct],axis=0)
    return np.quantile(dist,[.025,.975])
for offset,data,label,col in [(-.17,old,'13-task checkpoint 54,808',BLUE),(.17,new,'19-task checkpoint 55,100',TEAL)]:
    values=np.array([cell_ba(c) for c in data]);cis=np.array([ba_ci(c) for c in data]);xx=np.arange(3)+offset
    axs[0].bar(xx,values*100,.32,label=label,color=col)
    axs[0].errorbar(xx,values*100,yerr=np.vstack((values-cis[:,0],cis[:,1]-values))*100,fmt='none',color='#172d36',capsize=3)
axs[0].set_xticks(range(3),[c['cell']+' baseline' for c in new]);axs[0].set_ylim(45,101);axs[0].set_ylabel('Balanced accuracy (%)');axs[0].axhline(50,color=GREY,ls='--');axs[0].legend(fontsize=8,loc='lower right');axs[0].set_title('Native cued motion change')
duration=groups['motion_duration_cued']; duration.sort(key=lambda c:int(c['cell'][1:]))
v=np.array([c['ba'] for c in duration]);ci=np.array([ba_ci(c) for c in duration]);axs[1].bar(range(4),v*100,color=TEAL,width=.6)
axs[1].errorbar(range(4),v*100,yerr=np.vstack((v-ci[:,0],ci[:,1]-v))*100,fmt='none',color='#172d36',capsize=3)
axs[1].set_xticks(range(4),[c['cell']+'\n'+c['cell'][1:]+' blank frames' for c in duration]);axs[1].set_ylim(20,101);axs[1].axhline(25,color=GREY,ls='--');axs[1].set_title('Cued motion-duration memory (four choices)')
fig.suptitle('Moving dots: discrimination, spatial relevance, and retention',fontsize=15)
fig.text(.03,.02,'Pointwise 95% class-stratified trial bootstrap intervals (2,000 resamples). Independent test banks and different training exposure;\nthe checkpoint comparison is descriptive, not a matched causal test of architecture. Latest mean BA: motion change 94.45%, duration memory 98.54%.',fontsize=9)
fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.82,wspace=.25)
save(fig,'02_motion_generalization')

swap=js('SecondPass/LocalComparisonKrauzlis/Stability/run01/attention_swap.json')['results']
fig,ax=plt.subplots(figsize=(10.2,5))
keys=['good','bad','bad_with_good_attention','good_with_bad_attention']
labels=['Good weights\nnative attention','Collapsed weights\nnative attention','Collapsed weights\ngood donor attention','Good weights\ncollapsed donor attention']
values=[100*swap[k]['balanced_accuracy'] for k in keys]
ax.bar(range(4),values,color=[BLUE,GREY,TEAL,CORAL],width=.65);ax.set_xticks(range(4),labels);ax.set_ylim(0,100);ax.axhline(50,color=GREY,ls='--');ax.set_ylabel('Balanced accuracy (%)')
for i,v in enumerate(values):ax.text(i,v+2,f'{v:.2f}%',ha='center')
ax.set_title('Attention trajectory changes can rescue or abolish performance',pad=16)
fig.text(.03,.025,'Frozen diagnostic checkpoints 600 and 1,700; same 300 validation movies. Donor maps are computed from identical pixel trials.\nMaps replaced at every frame, affecting evidence selection AND memory writes; no weight updates. One seed; validation-set diagnostic.',fontsize=9)
fig.subplots_adjust(bottom=.25,top=.85,left=.10,right=.97)
save(fig,'03_attention_routing_rescue')

effects=rows('SecondPass/StabilizedAttentionVDA/Analysis/run01/primary_effects.csv')
tests=rows('SecondPass/StabilizedAttentionVDA/Analysis/run01/primary_hypothesis_tests.csv')
primary=source('SecondPass/StabilizedAttentionVDA/Analysis/run01/primary.jsonl')
recs=[json.loads(line) for line in primary.open(encoding='utf-8') if line.strip()]
fig,axs=plt.subplots(1,2,figsize=(11.4,5.4),gridspec_kw={'width_ratios':[1,1.4]})
arms=['sham','cued_-4_change']; rate=[]
for arm in arms:
    rr=[]
    for event in ('uncued','cued','catch'):
        a=[r['choice'] for r in recs if r['arm']==arm and r['event']==event];assert len(a)==384;rr.append(statistics.mean(a))
    rate.append(rr)
for i,(rr,col,label) in enumerate(zip(rate,(BLUE,CORAL),('Sham','Cued inhibition -4'))):
    axs[0].bar(np.arange(3)+(i-.5)*.34,np.array(rr)*100,width=.34,color=col,label=label)
axs[0].set_xticks(range(3),['Opposite\nchange','Cued\nchange','Catch\n(no change)']);axs[0].set_ylim(0,100);axs[0].set_ylabel('P(report change), %');axs[0].legend(fontsize=8);axs[0].set_title('Response components')
chosen=['cued_-4_change','uncued_+4_change','cued_+4_change','uncued_-4_change']
labelmap={'cued_-4_change':'Cued inhibition -4','uncued_+4_change':'Opposite stimulation +4','cued_+4_change':'Cued stimulation +4','uncued_-4_change':'Opposite inhibition -4'}
for yi,arm in enumerate(chosen):
    for dy,event,col in [(-.12,'uncued',TEAL),(.12,'cued',CORAL)]:
        e=next(x for x in effects if x['arm']==arm and x['event']==event);val=100*float(e['hit_delta']);lo=100*float(e['hit_low']);hi=100*float(e['hit_high'])
        axs[1].errorbar(val,yi+dy,xerr=[[val-lo],[hi-val]],fmt='o',color=col,capsize=3,label=('Opposite change' if event=='uncued' else 'Cued change') if yi==0 else None)
axs[1].set_yticks(range(4),[labelmap[a] for a in chosen]);axs[1].invert_yaxis();axs[1].axvline(0,color=GREY,ls='--');axs[1].set_xlabel('Hit-rate change from sham (percentage points)');axs[1].set_title('Paired effects with scene-bootstrap 95% intervals');axs[1].legend(fontsize=8,loc='center right',bbox_to_anchor=(1,.5))
fig.suptitle('VDA-4: spatial selection redistributes behavioural influence',fontsize=15)
fig.text(.02,.025,'Frozen checkpoint 16,000; 384 matched scenes/event. Eight planned directional hit-rate tests passed Holm correction.\nThe opposite change deliberately violates a displayed 100%-valid cue. Seven-frame trials: explicit long-window comparison branch is zero.\nPointwise intervals concern this checkpoint and scene distribution, not independent training seeds.',fontsize=9)
fig.subplots_adjust(bottom=.25,top=.82,left=.07,right=.98,wspace=.85)
save(fig,'04_vda_reciprocal_effects')

motionrows=rows('SecondPass/StabilizedAttentionSuite/Analysis/run01/psychometric_points.csv')
print('motion psychometric series',sorted(set((r['task'],r['series'],r['metric']) for r in motionrows if 'motion' in r['task'])))
print('task summary',[(x['task'],round(x['ba'],4)) for x in table])
for x in table:
    if x['task']=='motion_memory_search':assert abs(x['ba']-.8115)<.001
with (OUT/'task_performance.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
summary={'checkpoint':55100,'selected_by':'validation, before full test','conditions':437,'presentations':sum(x['n'] for x in latest['cells']),'equal_task_mean_auc':latest['equal_task_mean_auc'],'tasks':table,'sources':sources,'scope':'Completed saved predictions and measurements only. No inference/training. Historical comparisons are not matched architecture ablations.','figures':['01_task_capabilities','02_motion_generalization','03_attention_routing_rescue','04_vda_reciprocal_effects']}
(OUT/'evidence.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Saved four PNG/PDF figure pairs and source ledger.')
