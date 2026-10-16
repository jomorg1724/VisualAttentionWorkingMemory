"""Descriptive psychometrics, paired causal contrasts, and complete attention atlas."""
import json,math,time,html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from scipy.stats import norm
from PIL import Image,ImageDraw,ImageFont
from .design import OUT,ROOT,NATIVE,AXES,STRENGTHS,spatial_mask
from WorkingMemory.BatteryAudit.observers import auc_binary
from SecondPass.TaskSuite.suite import TASKS

TITLES={'motion_direction':'Motion direction','orientation':'Orientation','contrast':'Contrast','spatial_frequency':'Spatial frequency','chromatic_increment':'Chromatic increment','contour':'Contour integration','natural_spectrum':'Natural spectrum','orientation_ring':'Ring-cued orientation','orientation_cued':'Sign-cued orientation','motion_duration_cued':'Motion-duration memory','krauzlis_cued_motion':'Krauzlis selection','spatial_binding':'Spatial binding','image_recognition':'Image recognition'}
COLORS=['#087e8b','#e76f51','#6156a6','#d89922','#337fb5']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.labelcolor':'#364852','text.color':'#20343e','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#adbbc1','xtick.color':'#526873','ytick.color':'#526873','axes.prop_cycle':plt.cycler(color=COLORS),'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def wilson(k,n):
    if n==0:return [np.nan,np.nan]
    z=1.95996398454;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);w=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [c-w,c+w]
def load_rows(path):
    rows=[]
    with path.open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);m=r['metadata'];t=r['task'];p=r['probability']
            target=m.get('target_location',-1);rot=m.get('rotations_degrees',[]);foil=m.get('analysis_foil_location',-1)
            row={k:v for k,v in r.items() if k not in ('metadata','probability','attention_mass_by_location')}
            row.update(correct=int(r['label']==r['choice']),p1=p[1],cluster=m.get('cluster',m.get('suite_trial_id','')),condition=str(m.get('condition','')),event=m.get('event',m.get('event_type','')),target=target,cue_sign=m.get('cue_sign',m.get('event_sign',0)),hold=m.get('probe_hold',0),load=m.get('study_load',0),seen_index=m.get('seen_study_index'),foil_relation=(int(np.sign(rot[foil]*m.get('cue_sign',1))) if rot and foil>=0 else 0))
            if 'duration_counts_by_patch' in m:
                cc=sorted(m['duration_counts_by_patch'][target]);row['duration_margin']=cc[-1]-cc[-2]
            rows.append(row)
    return pd.DataFrame(rows)

def balanced_curve(g):
    # Observed balanced accuracy, stratified binomial bootstrap for bounded bands.
    per=[v['correct'].to_numpy() for _,v in g.groupby('label')]
    if len(per)==1:
        vals=per[0];lo,hi=wilson(vals.sum(),len(vals));return vals.mean(),lo,hi
    means=np.array([v.mean() for v in per]);n=np.array([len(v) for v in per]);rng=np.random.default_rng(6029)
    samples=np.column_stack([rng.binomial(nn,pp,3000)/nn for pp,nn in zip(means,n)]).mean(1)
    # Bootstrap degenerates at ceilings. Wilson bounds remain informative there.
    if np.all((means==0)|(means==1)):
        bounds=np.array([wilson(v.sum(),len(v)) for v in per]);return means.mean(),bounds[:,0].mean(),bounds[:,1].mean()
    lo,hi=np.quantile(samples,[.025,.975]);return means.mean(),lo,hi

def psychometrics(df):
    records=[]
    for task,g in df.groupby('task',sort=False):
        if task=='krauzlis_cued_motion':continue
        keys=['strength']
        if task=='orientation_cued':keys+=['cell']
        if task=='image_recognition':keys+=['hold']
        for key,d in g.groupby(keys):
            key=key if isinstance(key,tuple) else (key,)
            acc,lo,hi=balanced_curve(d)
            records.append(dict(task=task,strength=float(key[0]),series=str(key[1]) if len(key)>1 else 'all',value=acc,low=lo,high=hi,n=len(d),metric='specificity' if task=='image_recognition' and key[0]==0 else 'balanced_accuracy'))
    cp=pd.DataFrame(records);cp.to_csv(OUT/'psychometric_points.csv',index=False)
    fig,axs=plt.subplots(4,4,figsize=(17,15),layout='constrained')
    for ax,task in zip(axs.flat,TITLES):
        ax.set_title(TITLES[task],loc='left',pad=12);ax.grid(axis='y',alpha=.16)
        if task=='krauzlis_cued_motion':
            d=df[df.task==task]
            for event,color in zip(['target','foil'],COLORS):
                gg=d[d.event==event].groupby('strength').choice.agg(['mean','sum','count']);lohi=np.array([wilson(k,n) for k,n in zip(gg['sum'],gg['count'])])
                ax.plot(gg.index,gg['mean'],'o-',c=color,label='Target report' if event=='target' else 'Distractor false alarm',ms=4);ax.fill_between(gg.index,lohi[:,0],lohi[:,1],color=color,alpha=.13)
            catch=d[d.event=='catch'].choice.mean();ax.axhline(catch,color=COLORS[2],ls=':',label='Catch false alarm');ax.legend(fontsize=8)
            ax.set_ylabel('P(report change)')
        else:
            grouped=list(cp[cp.task==task].groupby('series',sort=False))
            grouped.sort(key=lambda pair:float(pair[0]) if pair[0]!='all' else 0)
            for i,(series,d) in enumerate(grouped):
                label='Delay '+series if task=='orientation_cued' else 'Probe hold '+series if task=='image_recognition' else None
                col=COLORS[i%len(COLORS)]
                if task=='image_recognition':
                    empty=d[d.strength==0];ax.plot(empty.strength,empty.value,'s',ms=5,mfc='white',mec=col)
                    d=d[d.strength>0]
                ax.plot(d.strength,d.value,'o-',ms=4,c=col,label=label);ax.fill_between(d.strength,d.low,d.high,alpha=.12,color=col)
                # Native strength locations marked as solid black ticks at foot.
            for v in NATIVE.get(task,[]):ax.plot(v,.04,'|',c='#20343e',ms=8)
            chance=.25 if task in ('motion_direction','motion_duration_cued') else .5;ax.axhline(chance,c='#8d9ba3',ls=':',lw=1)
            ax.set_ylabel('Balanced accuracy' if task!='image_recognition' else 'BA; load 0 = specificity')
            if task in ('orientation_cued','image_recognition'):ax.legend(fontsize=8)
        ax.set_ylim(0,1.04);ax.set_xlabel(AXES[task],fontsize=9)
    for ax in list(axs.flat)[13:]:ax.axis('off')
    axs.flat[13].text(0,1,'Frozen checkpoint 54,808\n\nShading: pointwise 95% intervals.\nBlack ticks: original training strengths.\nNew strengths test graded sensitivity.\n\nZero-angle points have scheduled labels;\nthere is no physical direction change.\nContour jitter increases difficulty.\n\nOne trained model, fresh trial seeds.\nCurves connect observed points;\nno sigmoid has been imposed.',va='top',linespacing=1.7,fontsize=11)
    fig.suptitle('How performance changes with evidence and memory demand',fontsize=23,fontweight='bold');save(fig,'psychometrics_all_tasks')
    # Memory-specific inspection, without conflating zero-load with a two-class task.
    fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    d=df[df.task=='image_recognition']
    for label,name,col in [(1,'Hit rate',COLORS[0]),(0,'False-alarm rate',COLORS[1])]:
        for hold,dd in d[d.label==label].groupby('hold'):
            gg=dd.groupby('load').choice.agg(['mean','sum','count']);axes=axs[0]
            axes.plot(gg.index,gg['mean'],marker='o',color=col,ls={3:'-',4:'--',5:':'}[hold],label=f'{name}, H{hold}')
    axs[0].set(title='Recognition: hits and false alarms',xlabel='Study images',ylabel='P(report seen)',ylim=(0,1.04));axs[0].legend(fontsize=8)
    pos=d[d.label==1].copy();pos['position_bin']=np.minimum(3,((pos.seen_index/pos.load)*4).astype(int))
    for load,dd in pos.groupby('load'):
        gg=dd.groupby('position_bin').correct.agg(['mean','sum','count']);bounds=np.array([wilson(k,n) for k,n in zip(gg['sum'],gg['count'])])
        axs[1].plot(gg.index,gg['mean'],'o-',label=f'{load} images');axs[1].fill_between(gg.index,bounds[:,0],bounds[:,1],alpha=.12)
    axs[1].set(title='Recognition: serial position',xlabel='Position of repeated study image',ylabel='Hit rate',ylim=(0,1.04),xticks=[0,1,2,3],xticklabels=['First ¼','Second ¼','Third ¼','Last ¼']);axs[1].legend()
    d=df[df.task=='motion_duration_cued'].copy()
    d.groupby(['strength','duration_margin']).correct.agg(['mean','sum','count']).to_csv(OUT/'motion_duration_exact_margin_points.csv')
    # Sparse large margins are pooled for a readable descriptive display; exact counts retained.
    d['margin_display']=np.minimum(d.duration_margin,3)
    for delay,dd in d.groupby('strength'):
        gg=dd.groupby('margin_display').correct.agg(['mean','sum','count']);bounds=np.array([wilson(k,n) for k,n in zip(gg['sum'],gg['count'])]);axs[2].plot(gg.index,gg['mean'],'o-',label=f'Delay {int(delay)}');axs[2].fill_between(gg.index,bounds[:,0],bounds[:,1],alpha=.12)
    axs[2].set(title='Motion memory: evidence margin',xlabel='Winner minus runner-up frames (3+ pooled)',ylabel='Accuracy',ylim=(0,1.04),xticks=[1,2,3],xticklabels=['1','2','≥3']);axs[2].legend(fontsize=8)
    fig.suptitle('Memory demand, response bias, and temporal evidence',fontsize=19,fontweight='bold');save(fig,'memory_psychometrics')
    # Interval-bracket thresholds only: don't turn extrapolated fits into findings.
    thresholds=[]
    for (task,series),d in cp.groupby(['task','series']):
        if task in ('contour','motion_duration_cued','spatial_binding','image_recognition'):continue
        criterion=.625 if task=='motion_direction' else .75;d=d.sort_values('strength');pair=None
        for i in range(1,len(d)):
            if d.iloc[i-1].value<criterion<=d.iloc[i].value:pair=[float(d.iloc[i-1].strength),float(d.iloc[i].strength)];break
        thresholds.append(dict(task=task,series=series,criterion=criterion,observed_crossing_bracket=pair,status='bracketed' if pair else 'not bracketed; no extrapolation'))
    (OUT/'threshold_brackets.json').write_text(json.dumps(thresholds,indent=2),encoding='utf8')
    # Separate sign errors from failing to detect a rotation at all.
    d=df[(df.task=='orientation_cued')&(df.strength>0)];detail=[]
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,delay in zip(axs.flat,['0','4','12','24']):
        for event,label,col in [('aligned','Target rotates in cue sign',COLORS[0]),('opposite','Target rotates opposite to cue',COLORS[1]),('unchanged','Target does not rotate',COLORS[2])]:
            gg=d[(d.cell==delay)&(d.event==event)].groupby('strength').choice.agg(['mean','sum','count']);b=np.array([wilson(k,n) for k,n in zip(gg['sum'],gg['count'])]);ax.plot(gg.index,gg['mean'],'o-',c=col,label=label,ms=4);ax.fill_between(gg.index,b[:,0],b[:,1],color=col,alpha=.12)
            for strength,r in gg.iterrows():detail.append(dict(delay=int(delay),strength=strength,event=event,p_yes=r['mean'],n=int(r['count'])))
        ax.set(title=f'{delay} blank frames',xlabel='Rotation magnitude (degrees)',ylabel='P(report cue-aligned rotation)',ylim=(0,1.04));ax.grid(axis='y',alpha=.12)
    axs[0,0].legend(fontsize=8,loc='upper left');fig.suptitle('Seeing a rotation and applying the cue sign are different demands',fontsize=19,fontweight='bold');save(fig,'orientation_rule_psychometrics');pd.DataFrame(detail).to_csv(OUT/'orientation_rule_psychometrics.csv',index=False)
    return cp

def bootstrap_difference(values,strata,reps=3000,max_abs=1.):
    rng=np.random.default_rng(6071);values=np.asarray(values,float);strata=np.asarray(strata)
    if np.all(values==0):
        # A zero-width bootstrap interval would falsely imply population certainty.
        # If no cluster differs, the 95% upper bound on a nonzero-cluster rate is
        # 1 - .05**(1/n). |E(delta)| is bounded by that rate times max |delta|.
        bound=float(max_abs*(1-.05**(1/len(values))))
        return 0.,-bound,bound
    boot=np.zeros(reps)
    for s in np.unique(strata):
        v=values[strata==s];boot+=v[rng.integers(len(v),size=(reps,len(v)))].sum(1)/len(values)
    return float(values.mean()),*np.quantile(boot,[.025,.975]).tolist()

def signflip_p(v):
    v=np.asarray(v);rng=np.random.default_rng(786);stat=v.mean()
    if not np.any(v):return 1.
    # One-sided positive-benefit test with matched scene-level differences.
    draws=(rng.choice([-1,1],size=(9999,len(v)))*v).mean(1)
    return float((1+(draws>=stat-1e-12).sum())/10000)

def intervention_cluster_table(g,angles):
    if g.task.iloc[0]=='krauzlis_cued_motion':
        d=g[(g.strength.isin(angles))|(g.event=='catch')]
        tab=d.groupby(['cluster','arm','event']).choice.mean().unstack('event')
        tab['ba']=.5*(tab.target+1-(29*tab.foil+14*tab['catch'])/43);tab['hit']=tab.target;tab['false_alarm']=tab.foil
        return tab.reset_index()
    d=g[g.strength.isin(angles)];tab=d.groupby(['cluster','arm']).agg(ba=('correct','mean'),label=('label','first'),positive=('choice','mean')).reset_index()
    tab['hit']=np.where(tab.label==1,tab.positive,np.nan);tab['false_alarm']=np.where(tab.label==0,tab.positive,np.nan)
    return tab

def interventions_report(df):
    effects=[];curves=[];sdts=[]
    for task,g in df.groupby('task'):
        for arm,d in g.groupby('arm'):
            for angle in sorted(d.strength.unique()):
                if task=='krauzlis_cued_motion':
                    if angle==0:continue
                    dd=d[d.strength==angle];hit=dd[dd.event=='target'];foil=dd[dd.event=='foil'];catch=d[d.event=='catch']
                    h=hit.choice.mean();f=foil.choice.mean();c=catch.choice.mean();value=.5*(h+1-(29*f+14*c)/43)
                    zh=norm.ppf((hit.choice.sum()+.5)/(len(hit)+1));zf=norm.ppf((catch.choice.sum()+.5)/(len(catch)+1))
                    pair1=pd.concat([hit,foil]);pair2=pd.concat([hit,catch]);auc=(29*auc_binary(pair1.label,pair1.p1)+14*auc_binary(pair2.label,pair2.p1))/43
                    sdts.append(dict(task=task,arm=arm,strength=angle,dprime=zh-zf,criterion=-.5*(zh+zf),hit=h,catch_fa=c,foil_fa=f,auc=auc))
                else:
                    dd=d[d.strength==angle];h=dd[dd.label==1].choice.mean();f=dd[dd.label==0].choice.mean();c=np.nan;value=.5*(h+1-f)
                curves.append(dict(task=task,arm=arm,strength=angle,ba=value,hit=h,foil_fa=f,catch_fa=c,n=len(dd),site=d.site.iloc[0],dose=d.dose.iloc[0],epoch=d.epoch.iloc[0]))
        sets={'near threshold':[8,18] if task=='krauzlis_cued_motion' else [8,15], 'trained':[26,28] if task=='krauzlis_cued_motion' else [15,30,45], 'all':[8,18,26,28,40] if task=='krauzlis_cued_motion' else [8,15,30,45]}
        meta=g.groupby('cluster')[['target','cue_sign','label','condition']].first()
        # For Krauzlis label varies across events, so do not stratify by first label.
        strat=meta.apply(lambda r:f'{r.condition}:{r.target}:{r.cue_sign}'+(f':{r.label}' if task=='orientation_cued' else ''),axis=1)
        for band,angles in sets.items():
            tab=intervention_cluster_table(g,angles)
            for metric in ('ba','hit','false_alarm'):
                p=tab.pivot(index='cluster',columns='arm',values=metric)
                for arm in p:
                    if arm=='sham':continue
                    v=(p[arm]-p.sham).dropna();mean,low,high=bootstrap_difference(v.to_numpy(),strat.loc[v.index].to_numpy())
                    effects.append(dict(task=task,band=band,metric=metric,arm=arm,delta=mean,low=low,high=high,n_clusters=len(v),interval_method='bounded zero-difference interval' if np.all(v==0) else 'stratified cluster bootstrap',p_positive=signflip_p(v.to_numpy()) if band=='near threshold' and metric=='ba' and arm in ['target_+4_all','target_+8_all','distractor_-4_all','distractor_-8_all'] else np.nan))
    ef=pd.DataFrame(effects);primary=ef[ef.p_positive.notna()].sort_values('p_positive');adj=np.maximum.accumulate(np.minimum(1,primary.p_positive.to_numpy()*np.arange(len(primary),0,-1)));ef.loc[primary.index,'p_holm']=adj
    ef.to_csv(OUT/'paired_intervention_effects.csv',index=False);cp=pd.DataFrame(curves);cp.to_csv(OUT/'intervention_curves.csv',index=False);pd.DataFrame(sdts).to_csv(OUT/'signal_detection.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    select=['sham','target_+4_all','distractor_-4_all','target_-8_all','distractor_+8_all']
    names=['Control','Target +4','Distractor −4','Target −8','Distractor +8']
    fig2,ax2=plt.subplots(1,3,figsize=(16,4.8),layout='constrained');sdt=pd.DataFrame(sdts)
    for ax,metric,label in zip(ax2,['dprime','criterion','auc'],['Catch-based d′','Criterion (+ = conservative)','ROC AUC (native negative mixture)']):
        for arm,name in zip(select,names):
            d=sdt[sdt.arm==arm].sort_values('strength');ax.plot(d.strength,d[metric],'o-',ms=4,label=name)
        ax.set(xlabel='Motion change (degrees)',ylabel=label);ax.grid(axis='y',alpha=.15)
    ax2[0].legend(fontsize=8);ax2[1].axhline(0,c='#a5b4bb',ls=':');ax2[2].axhline(.5,c='#a5b4bb',ls=':');ax2[2].set_ylim(0,1.04)
    fig2.suptitle('Krauzlis: separability and response bias',fontsize=21,fontweight='bold');save(fig2,'sensitivity_and_bias')
    for row,task in enumerate(['krauzlis_cued_motion','orientation_cued']):
        for col,(metric,label) in enumerate([('ba','Balanced accuracy'),('hit','Target hit rate'),('foil_fa','Distractor FA' if task=='krauzlis_cued_motion' else 'Incorrect “aligned” report')]):
            ax=axs[row,col]
            for arm,name in zip(select,names):
                d=cp[(cp.task==task)&(cp.arm==arm)].sort_values('strength');ax.plot(d.strength,d[metric],'o-',ms=4,label=name)
            ax.set(xlabel='Change magnitude (degrees)',ylabel=label,ylim=(0,1.04),title=TITLES[task]+' · '+label);ax.grid(axis='y',alpha=.15)
            if col==0:ax.axhline(.5,color='#9aadb4',ls=':');ax.legend(fontsize=8)
    fig.suptitle('Changing the shared priority map changes behavior',fontsize=22,fontweight='bold');save(fig,'intervention_psychometrics')
    fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,task in zip(axs,['krauzlis_cued_motion','orientation_cued']):
        d=ef[(ef.task==task)&(ef.band=='near threshold')&(ef.metric=='ba')&ef.arm.isin(['target_+4_all','target_+8_all','distractor_-4_all','distractor_-8_all'])].set_index('arm').loc[['target_+4_all','target_+8_all','distractor_-4_all','distractor_-8_all']]
        yy=np.arange(4)
        ax.errorbar(d.delta*100,yy,xerr=[(d.delta-d.low)*100,(d.high-d.delta)*100],fmt='o',capsize=4,color=COLORS[0],ms=7)
        ax.axvline(0,color='#81979f',ls=':');ax.set(yticks=yy,yticklabels=['Target stimulation +4','Target stimulation +8','Distractor inhibition −4','Distractor inhibition −8'],xlabel='Change in balanced accuracy (percentage points)',title=TITLES[task]);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        for i,(_,r) in enumerate(d.iterrows()):ax.annotate(f'  Holm p={r.p_holm:.3g}',(r.high*100,i),fontsize=8,va='center')
        lo=min(0,float(d.low.min()*100));hi=max(0,float(d.high.max()*100));ax.set_xlim(lo-1,hi+max(8,.3*(hi-lo)))
    fig.suptitle('The two proposed benefits: paired tests near threshold',fontsize=21,fontweight='bold');save(fig,'dual_benefit_tests')
    fig,axs=plt.subplots(2,3,figsize=(17,11),layout='constrained')
    for row,task in enumerate(['krauzlis_cued_motion','orientation_cued']):
        d=ef[(ef.task==task)&(ef.band=='all')&(ef.metric=='ba')].set_index('arm')
        for site,color in zip(['target','distractor'],COLORS):
            doses=[-12,-8,-4,-2,0,2,4,8,12]
            # Zero is the identical sham-vs-sham contrast, exactly zero by definition.
            dd=pd.DataFrame([dict(delta=0.,low=0.,high=0.) if v==0 else d.loc[f'{site}_{v:+d}_all'].to_dict() for v in doses])
            axs[row,0].errorbar(doses,dd.delta*100,yerr=[(dd.delta-dd.low)*100,(dd.high-dd.delta)*100],marker='o',capsize=3,label=site,c=color)
        axs[row,0].axhline(0,color='#8d9ba3',ls=':');axs[row,0].set(title=TITLES[task]+' · dose',xlabel='Priority-logit bias',ylabel='Δ balanced accuracy (pp)');axs[row,0].legend()
        epochs=['cue','baseline','postchange','report'] if task=='krauzlis_cued_motion' else ['cue','sample','delay','probe']
        matrix=np.array([[d.loc[f'{site}_{dose:+d}_{e}','delta']*100 for e in epochs] for site,dose in [('target',8),('target',-8),('distractor',8),('distractor',-8)]])
        im=axs[row,1].imshow(matrix,cmap='RdBu_r',vmin=-45,vmax=45,aspect='auto');axs[row,1].set(xticks=range(4),xticklabels=epochs,yticks=range(4),yticklabels=['Target +8','Target −8','Distractor +8','Distractor −8'],title='When does the perturbation matter?')
        for y in range(4):
            for x in range(4):axs[row,1].text(x,y,f'{matrix[y,x]:+.1f}',ha='center',va='center',color='white' if abs(matrix[y,x])>25 else '#20343e',fontsize=11)
        fig.colorbar(im,ax=axs[row,1],label='Δ BA (pp)',shrink=.7)
        arms=[f'{site}_{dose:+d}_all' for site in ['offsite','both','global'] for dose in [-12,12]]
        if task=='orientation_cued':arms+=['cueglyph_-8_cue','cueglyph_+8_cue']
        dd=d.loc[arms];axs[row,2].errorbar(dd.delta*100,np.arange(len(dd)),xerr=[(dd.delta-dd.low)*100,(dd.high-dd.delta)*100],fmt='o',capsize=3);axs[row,2].set(yticks=np.arange(len(dd)),yticklabels=[a.replace('_all','').replace('_cue','').replace('_',' ') for a in arms],xlabel='Δ BA (pp)',title='Location and normalization controls');axs[row,2].axvline(0,color='#8d9ba3',ls=':');axs[row,2].invert_yaxis()
    fig.suptitle('Dose, timing, and site: a broad intervention screen',fontsize=22,fontweight='bold');save(fig,'dose_timing_controls')
    return ef,cp

def attention_rgb(alpha):
    # Fixed across tasks, groups, modes and time. Alpha=.01 is uniform.
    scale=np.log1p(np.maximum(alpha,0)*100)/np.log1p(100)
    return (matplotlib.colormaps['magma'](scale)[...,:3]*255).round().astype('uint8')

def project(alpha):return np.repeat(np.repeat(attention_rgb(alpha),10,axis=-3),10,axis=-2)

def attention_legend(fig,axes):
    sm=matplotlib.cm.ScalarMappable(norm=Normalize(0,1),cmap='magma')
    cb=fig.colorbar(sm,ax=axes,orientation='horizontal',shrink=.6,fraction=.035,pad=.025,aspect=60)
    values=np.array([0,.01,.1,1]);cb.set_ticks(np.log1p(100*values)/np.log(101));cb.set_ticklabels(['0','0.01','0.1','1'])
    cb.set_label('Patch attention · uniform = 0.01 · fixed log color scale',fontsize=8);cb.ax.tick_params(labelsize=7)

def routing_timecourses():
    """Use stored actual intervention maps' ROI masses; no additional inference."""
    groups={};arms=['sham','target_+4_all','distractor_-4_all','target_-8_all','distractor_+8_all']
    with (OUT/'intervention_trials.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);m=r['metadata'];task=r['task']
            if r['arm'] not in arms:continue
            if task=='krauzlis_cued_motion' and not (m['condition']==20 and m['angle']==18 and m['event']=='target'):continue
            if task=='orientation_cued' and not (m['condition']==12 and m['angle']==15):continue
            mass=np.asarray(r['attention_mass_by_location']);k=(task,r['arm']);target=m['target_location'];foil=m['analysis_foil_location']
            groups.setdefault(k,[]).append(mass[:,[target,foil]])
    fig,axs=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for row,task in enumerate(['krauzlis_cued_motion','orientation_cued']):
        for arm,label,col in zip(arms,['Control','Target +4','Distractor −4','Target −8','Distractor +8'],COLORS):
            a=np.asarray(groups[(task,arm)]);mean=a.mean(0);se=a.std(0,ddof=1)/np.sqrt(len(a))
            for j in range(2):
                ax=axs[row,j];tt=np.arange(len(mean));ax.plot(tt,mean[:,j],label=label,c=col,lw=1.8);ax.fill_between(tt,np.maximum(0,mean[:,j]-1.96*se[:,j]),np.minimum(1,mean[:,j]+1.96*se[:,j]),color=col,alpha=.08)
        for j in range(2):
            ax=axs[row,j];ax.set(title=TITLES[task]+(' · target mass' if j==0 else ' · selected distractor mass'),xlabel='Model timestep',ylabel='Attention probability mass',ylim=(0,1));ax.grid(axis='y',alpha=.12)
            if row==0:ax.axvline(28,c='#adb9bd',ls=':',lw=1);ax.text(28,.98,'Change',fontsize=8,rotation=90,va='top')
            else:ax.axvline(15,c='#adb9bd',ls=':',lw=1);ax.text(15,.98,'Probe',fontsize=8,rotation=90,va='top')
            if j==0:ax.legend(fontsize=8,ncol=2)
    fig.suptitle('The intervention changes routing throughout the recurrent sequence',fontsize=20,fontweight='bold');save(fig,'intervention_attention_timecourses')
    pd.DataFrame([dict(task=task,arm=arm,n=len(a),target_final_mass=float(np.asarray(a)[:,-1,0].mean()),distractor_final_mass=float(np.asarray(a)[:,-1,1].mean())) for (task,arm),a in groups.items()]).to_csv(OUT/'intervention_attention_summary.csv',index=False)
    phase_rows=[]
    for (task,arm),a in groups.items():
        a=np.asarray(a);windows={'cue':slice(0,2),'baseline':slice(7,28),'postchange':slice(28,36),'report':slice(36,37)} if task=='krauzlis_cued_motion' else {'cue':slice(0,1),'sample':slice(1,3),'delay':slice(3,15),'probe':slice(15,16)}
        for phase,sl in windows.items():phase_rows.append(dict(task=task,arm=arm,phase=phase,n=len(a),target_mass=float(a[:,sl,0].mean()),distractor_mass=float(a[:,sl,1].mean())))
    pd.DataFrame(phase_rows).to_csv(OUT/'intervention_attention_by_phase.csv',index=False)

def geometry_figure():
    from .experiment import controlled
    from .design import bias_for
    from matplotlib.patches import Rectangle
    fig,axs=plt.subplots(2,5,figsize=(15,6.6),layout='constrained')
    for row,task in enumerate(['krauzlis_cued_motion','orientation_cued']):
        x,m=controlled(task,20 if row==0 else 12,0 if row==0 else 8,18 if row==0 else 15)
        for col,site in enumerate(['target','distractor','both','offsite','global' if row==0 else 'cueglyph']):
            frame=x[7 if row==0 else 1].numpy().transpose(1,2,0);ax=axs[row,col];ax.imshow(frame)
            mask=bias_for(task,m,dict(site=site,dose=1,epoch='all'),len(x))[0].reshape(10,10)
            for y,j in zip(*np.nonzero(mask)):
                ax.add_patch(Rectangle((10*j-.5,10*y-.5),10,10,facecolor=COLORS[col],edgecolor=COLORS[col],alpha=.4,lw=.8))
            ax.set(xticks=[],yticks=[],title={'target':'Target site','distractor':'One distractor','both':'Both sites','offsite':'Off-site control','global':'Global shift control','cueglyph':'Upper cue region'}[site])
            if col==0:ax.set_ylabel(TITLES[task],fontsize=11)
    fig.suptitle('Where the intervention acts · shaded patches are masks, not measured attention',fontsize=17,fontweight='bold');save(fig,'intervention_geometry')

def orientation_conflict(df):
    """Exploratory stratification by whether the selected foil implies the wrong answer."""
    g=df[(df.task=='orientation_cued')&df.strength.isin([8,15])]
    meta=g.groupby('cluster')[['label','foil_relation','condition','target','cue_sign']].first()
    meta['conflict']=(meta.foil_relation>0).astype(int)!=meta.label
    scores=g.groupby(['cluster','arm']).correct.mean().unstack('arm')
    arms=['target_+4_all','target_+8_all','distractor_-4_all','distractor_-8_all','target_-8_all','distractor_+8_all']
    labels=['Target +4','Target +8','Distractor −4','Distractor −8','Target −8','Distractor +8'];rows=[]
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for conflict,offset,col,name in [(True,-.12,COLORS[0],'Foil implies the wrong answer'),(False,.12,COLORS[1],'Foil agrees with the answer')]:
        # This post-hoc subset has sparse location/sign/delay cells. Resampling
        # singleton fine strata would produce falsely zero-width intervals.
        # Preserve class balance, but resample whole scenes within each class.
        mm=meta[meta.conflict==conflict];strata=mm.label.astype(str).to_numpy();counts=mm.label.value_counts()
        for j,arm in enumerate(arms):
            v=(scores.loc[mm.index,arm]-scores.loc[mm.index,'sham']).to_numpy();weights=np.array([len(v)/(2*counts[y]) for y in mm.label])
            mean,low,high=bootstrap_difference(v*weights,strata,max_abs=float(weights.max()))
            rows.append(dict(arm=arm,foil_conflict=conflict,n_clusters=len(v),delta_ba=mean,low=low,high=high,interval_method='class-stratified scene-cluster bootstrap'))
            ax.errorbar(mean*100,j+offset,xerr=[[(mean-low)*100],[(high-mean)*100]],fmt='o',c=col,capsize=3,label=name if j==0 else None)
    ax.axvline(0,color='#9aadb4',ls=':');ax.set(yticks=np.arange(len(arms)),yticklabels=labels,xlabel='Change in balanced accuracy (percentage points)',title='Sign-cued orientation: does the selected distractor carry conflicting evidence?');ax.invert_yaxis();ax.legend(loc='best',fontsize=9);ax.grid(axis='x',alpha=.12)
    fig.text(.5,-.035,'Exploratory, harder angles 8°/15°. Paired scene-cluster 95% intervals; each subgroup balances its two target classes.',ha='center',fontsize=9)
    save(fig,'orientation_foil_conflict');pd.DataFrame(rows).to_csv(OUT/'orientation_foil_conflict.csv',index=False)

def atlas_report():
    atlas=OUT/'atlas';entries=json.loads((atlas/'index.json').read_text());web=OUT/'gallery';web.mkdir(exist_ok=True);assets=web/'assets';assets.mkdir(exist_ok=True)
    exports=[];example_by_task={}
    for entry in entries:
        if time.time()>json.loads((OUT/'budget.json').read_text())['deadline']:raise TimeoutError('Original analysis deadline reached during atlas rendering')
        if entry['timesteps']==2 and entry['task'] not in ('motion_direction','orientation'):
            entry['phases']=['interval 1','interval 2']
        key=entry['id'];d=np.load(atlas/(key+'.npz'));t=entry['timesteps']
        notes={'orientation_cued':'Fixed delay, top-left target and cue sign; native rotation sizes and both response classes pooled.','orientation_ring':'Top-left target; native rotation sizes and both rotation signs pooled.','motion_duration_cued':'Fixed delay and top-left target; all four winning directions pooled.','spatial_binding':'Fixed delay and top-left target; target and distractor swaps pooled.','krauzlis_cued_motion':'Fixed baseline, target side and event; native 26°/28° sizes and both change signs pooled.','image_recognition':'Fixed study load, probe hold and seen/novel status; different photographs and study positions pooled.'}
        item=dict(entry,question=TASKS[entry['task']]['question'],label_names=TASKS[entry['task']]['labels'],averaging_note=notes.get(entry['task'],'Fixed evidence-strength group; response classes and native nuisance variation pooled.'))
        for mode,ik,ak in [('example','images','attention'),('mean','mean_images','mean_attention')]:
            images=d[ik];alpha=d[ak];assert alpha.shape==(t,10,10) and np.allclose(alpha.sum((1,2)),1,atol=1e-5)
            raw=(np.clip(images.transpose(0,2,3,1),0,1)*255).round().astype('uint8');heat=project(alpha)
            overlay=(.58*raw+.42*heat).round().astype('uint8')
            # Each row a panel; each column one exact model timestep. No interpolation.
            sprite=np.concatenate([np.concatenate(v,axis=1) for v in (raw,overlay,heat)],axis=0)
            Image.fromarray(sprite).save(assets/(key+'--'+mode+'.png'))
        item['mean_attention']=d['mean_attention'].round(7).tolist();item['example_attention']=d['attention'].round(7).tolist()
        exports.append(item)
        # Save every timestep in contact sheets, split into <=10-column pages.
        sheetdir=OUT/'timesteps';sheetdir.mkdir(exist_ok=True)
        for start in range(0,t,10):
            end=min(t,start+10);fig,axs=plt.subplots(3,end-start,figsize=(1.45*(end-start)+.5,5),squeeze=False,layout='constrained')
            for j,i in enumerate(range(start,end)):
                im=d['images'][i].transpose(1,2,0);alpha=d['attention'][i];heat=project(alpha)
                axs[0,j].imshow(np.clip(im,0,1));axs[1,j].imshow(np.clip(.58*im+.42*heat/255,0,1));axs[2,j].imshow(heat,interpolation='nearest')
                axs[0,j].set_title(f't={i}\n{entry["phases"][i]}',fontsize=8)
                for ax in axs[:,j]:ax.set_xticks([]);ax.set_yticks([])
            for ax,label in zip(axs[:,0],['Model image','Attention overlay','Attention only']):ax.set_ylabel(label,fontsize=8)
            attention_legend(fig,list(axs[2]))
            fig.suptitle(f'{TITLES[entry["task"]]} · {entry["cell"]} · {entry["group"]}\nIndividual trial; fixed log color scale, uniform attention = .01',fontsize=10)
            fig.savefig(sheetdir/f'{key}--{start:02d}.png',dpi=140,bbox_inches='tight');plt.close(fig)
        # Representative is first preselected trial in the first collected condition/group.
        preferred={'image_recognition':('N12_H4','seen'),'orientation_cued':('D12','target_top_left_cueplus'),'spatial_binding':('D12','target_top_left'),'motion_duration_cued':('D12','target_top_left'),'krauzlis_cued_motion':('B20','side0_target')}
        if entry['task'] not in example_by_task or preferred.get(entry['task'])==(entry['cell'],entry['group']):example_by_task[entry['task']]=(entry,d['images'].copy(),d['attention'].copy(),d['mean_attention'].copy())
    (web/'data.js').write_text('window.ATLAS='+json.dumps(exports,separators=(',',':'))+';\nwindow.TITLES='+json.dumps(TITLES)+';',encoding='utf8')
    # A readable thirteen-task summary, using representative meaningful timesteps.
    fig,axs=plt.subplots(13,4,figsize=(12,31),layout='constrained')
    for row,task in enumerate(TITLES):
        entry,images,alpha,mean=example_by_task[task];m=entry['metadata'];i=m.get('probe_frame',m.get('report_frame',len(images)-1))
        if task=='image_recognition':i=m['probe_frames'][0]
        if task=='krauzlis_cued_motion':i=m['first_postchange_frame']+4
        if task=='motion_duration_cued':i=9
        im=images[i].transpose(1,2,0);heat=project(alpha[i]);axs[row,0].imshow(np.clip(im,0,1));axs[row,1].imshow(np.clip(.58*im+.42*heat/255,0,1));axs[row,2].imshow(heat);axs[row,3].imshow(project(mean[i]))
        axs[row,0].set_ylabel(f'{TITLES[task]}\n{entry["cell"]} · t={i}',fontsize=9)
        for ax in axs[row]:ax.set_xticks([]);ax.set_yticks([])
        if row==0:
            for ax,title in zip(axs[row],['Actual frame','Its attention overlay','Its attention map','Condition mean map']):ax.set_title(title,fontsize=11)
    attention_legend(fig,list(axs[:,2:].flat))
    fig.suptitle('One shared spatial priority map across thirteen tasks',fontsize=21,fontweight='bold');save(fig,'attention_all_tasks')
    # Export all individual representative timesteps as an animation per task.
    gifdir=OUT/'animations';gifdir.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for task,(entry,images,alpha,mean) in example_by_task.items():
        frames=[]
        for i,(im,a) in enumerate(zip(images,alpha)):
            raw=(np.clip(im.transpose(1,2,0),0,1)*255).astype('uint8');heat=project(a);overlay=(.58*raw+.42*heat).astype('uint8');panel=Image.new('RGB',(720,300),'#f5f8fa');draw=ImageDraw.Draw(panel)
            draw.text((12,7),f'{TITLES[task]} | {entry["cell"]} | frame {i+1}/{len(images)} | {entry["phases"][i]}',font=font,fill='#20343e')
            for j,(pic,name) in enumerate([(raw,'Model input'),(overlay,'Attention overlay'),(heat,'Attention only')]):
                panel.paste(Image.fromarray(pic).resize((220,220),Image.Resampling.NEAREST),(10+j*240,40));draw.text((10+j*240,265),name,font=font,fill='#20343e')
            frames.append(panel)
        frames[0].save(gifdir/(task+'.gif'),save_all=True,append_images=frames[1:],duration=400,loop=0,disposal=2)
    # Timecourses of allocation with spatial conditions preserved.
    selected=[e for e in entries if e['task'] in ('orientation_cued','krauzlis_cued_motion','spatial_binding','motion_duration_cued')]
    selected=[e for e in selected if (e['task']!='krauzlis_cued_motion' or e['group']=='side0_target') and (e['task']!='orientation_cued' or e['group'].endswith('plus'))]
    fig,axs=plt.subplots(4,4,figsize=(17,13),layout='constrained')
    for ax,e in zip(axs.flat,selected):
        d=np.load(atlas/(e['id']+'.npz'));m=e['metadata'];a=d['mean_attention'];target=m['target_location'];centers=m['positions_xy']
        for loc,c in enumerate(centers):
            mass=(a*spatial_mask(c).reshape(10,10)).sum((1,2));ax.plot(mass,label='Target' if loc==target else f'Distractor {loc}',lw=2 if loc==target else 1,alpha=1 if loc==target else .6)
        ax.set(title=TITLES[e['task']]+' · '+e['cell'],xlabel='Model timestep',ylabel='Attention mass',ylim=(0,1));ax.grid(alpha=.1);ax.legend(fontsize=7,ncol=2)
    for ax in list(axs.flat)[len(selected):]:ax.axis('off')
    fig.suptitle('Attention moves through cues, observations, and memory delays',fontsize=20,fontweight='bold');save(fig,'attention_timecourses')
    return entries

def write_summary(ef,cp,entries):
    primary=ef[ef.p_holm.notna()].copy();records=[]
    for _,r in primary.iterrows():
        h=ef[(ef.task==r.task)&(ef.arm==r.arm)&(ef.band=='near threshold')&(ef.metric=='hit')].iloc[0]
        f=ef[(ef.task==r.task)&(ef.arm==r.arm)&(ef.band=='near threshold')&(ef.metric=='false_alarm')].iloc[0]
        records.append(dict(task=r.task,arm=r.arm,delta_pp=r.delta*100,low_pp=r.low*100,high_pp=r.high*100,p_holm=r.p_holm,n_clusters=int(r.n_clusters),hit_delta_pp=h.delta*100,hit_low_pp=h.low*100,hit_high_pp=h.high*100,false_alarm_delta_pp=f.delta*100))
    summary=dict(primary_benefits=records,atlas_groups=len(entries),atlas_timesteps=sum(e['timesteps'] for e in entries),figures=[p.name for p in FIG.glob('*.png')],uncertainty='Paired stratified cluster bootstrap; all-zero differences receive conservative +/-[1-.05**(1/n)] bounds rather than a zero-width bootstrap interval.',completion=json.loads((OUT/'completion.json').read_text()))
    (OUT/'krauzlis_primary_completed.json').write_text(json.dumps([r for r in records if r['task']=='krauzlis_cued_motion'],indent=2),encoding='utf8')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');return summary

def main():
    assert (OUT/'completion.json').exists(),'Inference must complete before final reporting'
    print('Reading completed trial records',flush=True)
    p=load_rows(OUT/'psychometric_trials.jsonl');i=load_rows(OUT/'intervention_trials.jsonl')
    print('Rendering psychometrics',flush=True);psychometrics(p);geometry_figure()
    print('Computing paired intervention effects and figures',flush=True);ef,cp=interventions_report(i);orientation_conflict(i)
    print('Rendering intervention attention timecourses',flush=True);routing_timecourses()
    print('Rendering all attention timesteps and animations',flush=True);entries=atlas_report();summary=write_summary(ef,cp,entries)
    import shutil
    web=OUT/'gallery'
    for name in ('gallery.html','gallery.css','gallery.js'):shutil.copy2(Path(__file__).with_name(name),web/('index.html' if name=='gallery.html' else name))
    (web/'summary.js').write_text('window.SUMMARY='+json.dumps(summary)+';',encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
