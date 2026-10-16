"""Matched inference statistics and offline visual report; no model fitting."""
import json,html,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm,binomtest
from PIL import Image,ImageDraw,ImageFont
from .design import OUT,PHASES,masks

FIG=OUT/'figures';GALLERY=OUT/'gallery'
COLORS=['#087e8b','#e76f51','#6055a5','#c99929','#4378a2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})

def rows(name):
    with (OUT/f'{name}.jsonl').open() as f:return pd.DataFrame(json.loads(line) for line in f)

def save(fig,name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def wilson(values):
    n=len(values);p=float(np.mean(values));z=1.95996398454
    center=(p+z*z/(2*n))/(1+z*z/n);width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return p,max(0,center-width),min(1,center+width)

def paired_interval(values,draws=4000):
    values=np.asarray(values,float);rng=np.random.default_rng(203909)
    sims=values[rng.integers(0,len(values),(draws,len(values)))].mean(1)
    lo,hi=np.quantile(sims,[.025,.975]);return float(values.mean()),float(lo),float(hi)

def choice_array(df,arm,event):
    d=df[(df.arm==arm)&(df.event==event)].sort_values('rep')
    assert len(d)==384 and len(d.rep.unique())==384
    return d.choice.to_numpy(float)

def primary_statistics(df):
    records=[];tests=[]
    base_fa=choice_array(df,'sham','catch')
    for arm in df.arm.unique():
        fa=choice_array(df,arm,'catch')
        for event in ('cued','uncued'):
            hits=choice_array(df,arm,event);base_hits=choice_array(df,'sham',event)
            delta,lo,hi=paired_interval(hits-base_hits)
            ba_delta,ba_lo,ba_hi=paired_interval(.5*((hits-base_hits)-(fa-base_fa)))
            h=(hits.sum()+.5)/(len(hits)+1);f=(fa.sum()+.5)/(len(fa)+1)
            records.append(dict(arm=arm,event=event,n=len(hits),hit_rate=float(hits.mean()),false_alarm_rate=float(fa.mean()),
                ba=float(.5*(hits.mean()+1-fa.mean())),hit_delta=delta,hit_low=lo,hit_high=hi,
                ba_delta=ba_delta,ba_low=ba_lo,ba_high=ba_hi,dprime=float(norm.ppf(h)-norm.ppf(f)),
                criterion=float(-.5*(norm.ppf(h)+norm.ppf(f)))))
    # Predeclared reciprocal tradeoffs, corrected as a single eight-test family.
    for arm,benefit in [('cued_-4_change','uncued'),('uncued_+4_change','uncued'),
                        ('cued_+4_change','cued'),('uncued_-4_change','cued')]:
        for event,direction in [(benefit,1),('cued' if benefit=='uncued' else 'uncued',-1)]:
            d=(choice_array(df,arm,event)-choice_array(df,'sham',event))*direction
            wins=int((d>0).sum());losses=int((d<0).sum())
            p=binomtest(wins,wins+losses,.5,alternative='greater').pvalue if wins+losses else 1.
            tests.append(dict(arm=arm,event=event,hypothesis='benefit' if direction==1 else 'cost',
                p=float(p),wins=wins,losses=losses,delta=float(d.mean())*direction))
    order=np.argsort([t['p'] for t in tests]);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,tests[i]['p']*(len(tests)-rank)));tests[i]['p_holm']=running
    pd.DataFrame(records).to_csv(OUT/'primary_effects.csv',index=False)
    pd.DataFrame(tests).to_csv(OUT/'primary_hypothesis_tests.csv',index=False)
    return records,tests

def primary_plot(df,records):
    names=['sham','cued_-2_change','cued_-4_change','cued_-8_change','uncued_+4_change','cued_+4_change','uncued_-4_change']
    labels=['No intervention','BR inhibit −2','BR inhibit −4','BR inhibit −8','TL stimulate +4','BR stimulate +4','TL inhibit −4']
    fig,axs=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
    for event,color,label in [('cued',COLORS[0],'Change at cued bottom-right'),('uncued',COLORS[1],'Unexpected change at top-left'),('catch',COLORS[2],'No change: false alarm')]:
        vals=np.array([wilson(choice_array(df,a,event)) for a in names]);x=np.arange(len(names))
        axs[0].errorbar(x,vals[:,0],yerr=[vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]],fmt='o-',c=color,capsize=3,label=label)
    axs[0].set(xticks=np.arange(len(names)),xticklabels=labels,ylabel='Probability of reporting change',ylim=(0,1.03),title='The exact requested condition')
    axs[0].tick_params(axis='x',rotation=38);axs[0].legend(fontsize=8)
    selected=['cued_-4_change','uncued_+4_change','cued_+4_change','uncued_-4_change']
    for j,event in enumerate(('uncued','cued')):
        rr=[next(r for r in records if r['arm']==a and r['event']==event) for a in selected]
        vals=np.array([[r['hit_delta'],r['hit_low'],r['hit_high']] for r in rr])*100
        y=np.arange(4)+(j-.5)*.18
        axs[1].errorbar(vals[:,0],y,xerr=[vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]],fmt='o',capsize=4,
            c=COLORS[1] if event=='uncued' else COLORS[0],label='Top-left change' if event=='uncued' else 'Bottom-right change')
    axs[1].axvline(0,c='#9ba9b2',lw=1);axs[1].set(yticks=np.arange(4),yticklabels=['Inhibit bottom-right −4','Stimulate top-left +4','Stimulate bottom-right +4','Inhibit top-left −4'],
        xlabel='Change in detection rate (percentage points)',title='Benefit at one location; cost at the other?')
    axs[1].invert_yaxis();axs[1].legend(fontsize=8)
    fig.suptitle('100% bottom-right cue · 18° change · manipulation only at frames 5–6',fontsize=16,fontweight='bold')
    fig.supxlabel('384 matched scenes per event and arm. Left: Wilson 95% intervals. Right: paired scene-bootstrap 95% intervals.\nTop-left changes deliberately violate the 100% cue; they are not false alarms.',fontsize=10)
    save(fig,'primary_dual_effect')

def psychometrics(df):
    points=[]
    for key,d in df.groupby(['size','validity','event','angle']):
        p,lo,hi=wilson(d.choice.to_numpy())
        points.append(dict(size=int(key[0]),validity=key[1],event=key[2],angle=key[3],rate=p,low=lo,high=hi,n=len(d)))
    ps=pd.DataFrame(points);ps.to_csv(OUT/'psychometric_points.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for col,size in enumerate((4,9,16)):
        d=ps[(ps['size']==size)&(ps.validity==.75)]
        catch=d[d.event=='catch'].iloc[0]
        for event,color,label in [('cued',COLORS[0],'Cued change'),('uncued',COLORS[1],'Uncued change')]:
            q=d[d.event==event].sort_values('angle')
            x=np.r_[0,q.angle];y=np.r_[catch.rate,q.rate];lo=np.r_[catch.low,q.low];hi=np.r_[catch.high,q.high]
            axs[0,col].plot(x,y,'o-',c=color,label=label,ms=4);axs[0,col].fill_between(x,lo,hi,color=color,alpha=.12)
        axs[0,col].axhline(catch.rate,c=COLORS[2],ls=':',label='Catch false alarm')
        axs[0,col].set(title=f'VDA-{size} · 75% cue',xlabel='Imposed rotation magnitude (degrees)',ylabel='P(report change)',ylim=(0,1.03))
        d=ps[ps['size']==size]
        for event,color in [('cued',COLORS[0]),('uncued',COLORS[1]),('catch',COLORS[2])]:
            q=d[(d.event==event)&(d.angle==(0 if event=='catch' else 18))].sort_values('validity')
            axs[1,col].errorbar(q.validity,q.rate,yerr=[q.rate-q.low,q.high-q.rate],fmt='o-',color=color,capsize=3,label=event)
        axs[1,col].set(title='Cue reliability at a fixed 18° change',xlabel='Displayed cue validity',ylabel='P(report change)',ylim=(0,1.03),xticks=[0,.25,.5,.75,1])
    axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
    fig.suptitle('VDA psychometrics: strength of evidence and spatial expectation',fontsize=18,fontweight='bold')
    fig.supxlabel('96 scenes per point. Shading/error bars: pointwise Wilson 95% intervals. Zero is a no-change catch.\nUnsupported native combinations (uncued at 100%, cued at 0%) are omitted here and tested separately as counterfactuals.',fontsize=10)
    save(fig,'psychometrics')
    return points

def matched_main(df):
    records=[]
    for (size,validity,angle,event),g in df[df.event!='catch'].groupby(['size','validity','angle','event']):
        catch=df[(df['size']==size)&(df.validity==validity)&(df.event=='catch')]
        base=g[g.arm=='sham'].set_index(['geometry','rep']).choice
        base_fa=catch[catch.arm=='sham'].set_index(['geometry','rep']).choice
        for arm,q in g.groupby('arm'):
            hits=q.set_index(['geometry','rep']).choice.reindex(base.index)
            fa=catch[catch.arm==arm].set_index(['geometry','rep']).choice.reindex(base.index)
            assert hits.notna().all() and fa.notna().all()
            # Mirrored geometries reuse the same scenes, so bootstrap scenes,
            # averaging both geometries before sampling; not 96 independent rows.
            dh=(hits-base).groupby('rep').mean().to_numpy()
            dba=(.5*((hits-base)-(fa-base_fa))).groupby('rep').mean().to_numpy()
            delta,lo,hi=paired_interval(dba,1000)
            records.append(dict(size=int(size),validity=validity,angle=angle,event=event,arm=arm,
                hit_rate=float(hits.mean()),false_alarm_rate=float(fa.mean()),hit_delta=float(dh.mean()),
                ba=float(.5*(hits.mean()+1-fa.mean())),ba_delta=delta,low=lo,high=hi,independent_scenes=48))
    rs=pd.DataFrame(records);rs.to_csv(OUT/'all_intervention_effects.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for col,size in enumerate((4,9,16)):
        for row,event in enumerate(('uncued','cued')):
            d=rs[(rs['size']==size)&(rs.validity==1)&(rs.angle==18)&(rs.event==event)]
            matrix=np.array([[float(d[d.arm==f'{site}_{dose:+d}_all'].iloc[0].ba_delta)*100 for dose in (-8,-4,-2,2,4,8)] for site in ('cued','uncued')])
            im=axs[row,col].imshow(matrix,cmap='RdBu_r',vmin=-50,vmax=50,aspect='auto')
            for i in range(2):
                for j in range(6):axs[row,col].text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=9)
            axs[row,col].set(title=f'VDA-{size} · {event} change',xticks=range(6),xticklabels=['−8','−4','−2','+2','+4','+8'],
                yticks=[0,1],yticklabels=['Cued site','Opposite site'],xlabel='Attention-logit dose (all frames)')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Change in balanced accuracy (pp)',shrink=.8)
    fig.suptitle('Location × dose: benefits and costs across all three VDA sizes',fontsize=17,fontweight='bold')
    fig.supxlabel('100% cue; 18° change. Both diagonal cue arrangements, paired across 48 scene clusters.\nThese are exploratory effects; the exact requested change-time-only test has its own larger sample.',fontsize=10)
    save(fig,'dose_site_sweep')
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for col,size in enumerate((4,9,16)):
        d=rs[(rs['size']==size)&(rs.validity==1)&(rs.angle==18)&(rs.event=='uncued')]
        names=[('cued',-4),('uncued',4),('cued',4),('uncued',-4)]
        matrix=np.array([[float(d[d.arm==f'{site}_{dose:+d}_{epoch}'].iloc[0].ba_delta)*100 for epoch in ('cue','delay','sample','change')] for site,dose in names])
        im=axs[col].imshow(matrix,cmap='RdBu_r',vmin=-40,vmax=40,aspect='auto')
        for i in range(4):
            for j in range(4):axs[col].text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=9)
        axs[col].set(title=f'VDA-{size}',xticks=range(4),xticklabels=['Cue','Blank','Sample','Change'],yticks=range(4),
            yticklabels=['Cued −4','Opposite +4','Cued +4','Opposite −4'])
    fig.colorbar(im,ax=axs.tolist(),label='Change in balanced accuracy (pp)',shrink=.8)
    fig.suptitle('When does the intervention matter? Unexpected opposite-location changes',fontsize=16,fontweight='bold')
    fig.supxlabel('100% cue; 18° change; effects relative to identical unmanipulated movies.',fontsize=10)
    save(fig,'timing_sweep')
    return records

ATTENTION_NORM=LogNorm(vmin=1e-4,vmax=1,clip=True)

def map_scale(a):
    a=np.asarray(a)
    return np.asarray(ATTENTION_NORM(np.maximum(a,1e-30).ravel())).reshape(a.shape)

def pixels(x):
    return np.rint(np.clip(x.transpose(0,2,3,1),0,1)*255).astype(np.uint8)

def map_pixels(a):
    colored=plt.get_cmap('magma')(map_scale(a))[...,:3]
    return np.rint(np.repeat(np.repeat(colored,10,axis=1),10,axis=2)*255).astype(np.uint8)

def overlay(x,a):
    im=pixels(x).astype(float);heat=map_pixels(a).astype(float)
    opacity=np.repeat(np.repeat(map_scale(a),10,axis=1),10,axis=2)[...,None]*.65
    return np.rint(im*(1-opacity)+heat*opacity).astype(np.uint8)

def sprite(frames,path):
    Image.fromarray(np.concatenate(list(frames),axis=1)).save(path)

def atlas_report():
    GALLERY.mkdir(exist_ok=True);assets=GALLERY/'assets';assets.mkdir(exist_ok=True)
    entries=json.loads((OUT/'atlas/index.json').read_text());manifest=[]
    for e in entries:
        z=np.load(OUT/'atlas'/f'{e["id"]}.npz');item=dict(e);item['examples']=[]
        for j in range(len(z['images'])):
            paths={}
            for name,data in [('image',pixels(z['images'][j])),('map',map_pixels(z['attention'][j])),('overlay',overlay(z['images'][j],z['attention'][j]))]:
                filename=f'{e["id"]}_e{j}_{name}.png';sprite(data,assets/filename);paths[name]='assets/'+filename
            item['examples'].append(paths)
        item['mean']={}
        for name,data in [('image',pixels(z['mean_images'])),('map',map_pixels(z['mean_attention'])),('overlay',overlay(z['mean_images'],z['mean_attention']))]:
            filename=f'{e["id"]}_mean_{name}.png';sprite(data,assets/filename);item['mean'][name]='assets/'+filename
        manifest.append(item)
    (GALLERY/'atlas.js').write_text('window.ATLAS='+json.dumps(manifest)+';',encoding='utf8')
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    for size in (4,9,16):
        key=f'vda{size}_cue{size-1}_uncued_'
        sham=np.load(OUT/'atlas'/f'{key}sham.npz');inhibit=np.load(OUT/'atlas'/f'{key}cued_-4_change.npz')
        assert np.array_equal(sham['images'],inhibit['images'])
        x=sham['images'][0];sa=sham['attention'][0];ia=inhibit['attention'][0]
        panels=[pixels(x),overlay(x,sa),map_pixels(sa),overlay(x,ia),map_pixels(ia),map_pixels(inhibit['mean_attention'])]
        labels=['Actual input','Unchanged: overlay','Unchanged: map','BR inhibited: overlay','BR inhibited: map','Inhibited: condition mean']
        fig,axs=plt.subplots(6,7,figsize=(14,11),layout='constrained')
        for row in range(6):
            for t in range(7):
                ax=axs[row,t];ax.imshow(panels[row][t],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(f't={t} · {PHASES[t]}',fontsize=10)
                if t==0:ax.set_ylabel(labels[row],fontsize=9)
        fig.suptitle(f'VDA-{size}: 100% bottom-right cue, top-left change\nBottom-right inhibition −4 only during frames 5–6',fontsize=17,fontweight='bold')
        fig.supxlabel('Rows 1–5: the same individual movie. Last row: 48-scene condition mean, pooling colors and signs.\nAll maps use the same logarithmic color scale: 0.0001 to 1 attention mass per patch; dark to bright.',fontsize=10)
        save(fig,f'attention_timeline_vda{size}')
        frames=[]
        for t in range(7):
            canvas=Image.new('RGB',(800,248),'#f7f9fc');draw=ImageDraw.Draw(canvas)
            for j,(data,label) in enumerate(zip([panels[0],panels[1],panels[3],panels[4]],['Actual input','Sham overlay','BR inhibited overlay','Inhibited map'])):
                canvas.paste(Image.fromarray(data[t]).resize((200,200),Image.Resampling.NEAREST),(j*200,28))
                draw.text((j*200+5,4),label,font=font,fill='#263544')
            draw.text((8,228),f'VDA-{size}  |  t={t}: {PHASES[t]}  |  same scene, fixed map color scale',font=font,fill='#263544')
            frames.append(canvas)
        frames[0].save(FIG/f'attention_vda{size}.gif',save_all=True,append_images=frames[1:],duration=600,loop=0,disposal=2)
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for col,size in enumerate((4,9,16)):
        cells,_,_=masks(size)
        for arm,color,label in [('sham',COLORS[0],'No intervention'),('cued_-4_change',COLORS[1],'Inhibit cued site −4'),('uncued_+4_change',COLORS[2],'Stimulate opposite site +4')]:
            z=np.load(OUT/'atlas'/f'vda{size}_cue{size-1}_uncued_{arm}.npz')
            a=z['mean_attention'].reshape(7,100)
            axs[col].plot(range(7),(a*cells[size-1]).sum(1),'o-',color=color,label=label+' · cued',ms=4)
            axs[col].plot(range(7),(a*cells[0]).sum(1),'s--',color=color,label=label+' · top-left',ms=3)
        axs[col].axvspan(4.5,6.5,color='#e7bb76',alpha=.18)
        axs[col].set(title=f'VDA-{size}',xticks=range(7),xlabel='Model timestep',ylabel='Attention mass in stimulus cell',ylim=(0,1.02))
    axs[0].legend(fontsize=7);fig.suptitle('Where attention goes when the change arrives',fontsize=18,fontweight='bold')
    fig.supxlabel('100% bottom-right cue; unexpected 18° top-left change. Mean of 48 matched scenes.\nSolid: cued bottom-right cell. Dashed: opposite top-left cell. Shaded frames contain the intervention.',fontsize=10)
    save(fig,'attention_allocation')
    return manifest

def native_summary(df):
    result=[]
    from WorkingMemory.BatteryAudit.observers import auc_binary
    for (size,theta),g in df.groupby(['size','theta']):
        correct=(g.choice==g.label)
        ba=float(np.mean([correct[g.label==i].mean() for i in (0,1)]))
        result.append(dict(size=int(size),theta=int(theta),n=len(g),ba=ba,auc=float(auc_binary(g.label.to_numpy(),g.p1.to_numpy())),
                           hit_rate=float(g[g.label==1].choice.mean()),false_alarm_rate=float(g[g.label==0].choice.mean())))
    pd.DataFrame(result).to_csv(OUT/'fresh_native_results.csv',index=False)
    groups=[]
    for (size,theta,validity,color,event),g in df.groupby(['size','theta','validity','color','event']):
        p,lo,hi=wilson(g.choice.to_numpy());groups.append(dict(size=int(size),theta=int(theta),validity=validity,color=color,event=event,n=len(g),report_rate=p,low=lo,high=hi))
    pd.DataFrame(groups).to_csv(OUT/'native_value_validity_strata.csv',index=False)
    return result

def value_plot(primary):
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for j,event in enumerate(('uncued','cued')):
        for arm,color,label in [('sham',COLORS[0],'No intervention'),('cued_-4_change',COLORS[1],'Inhibit bottom-right −4')]:
            values=[]
            for cue_color in ('blue','green','red'):
                q=primary[(primary.event==event)&(primary.arm==arm)&(primary.color==cue_color)]
                values.append(wilson(q.choice.to_numpy()))
            v=np.array(values)
            axs[j].errorbar([1,3,5],v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt='o-',c=color,label=label,capsize=3)
        axs[j].set(title='Top-left change' if event=='uncued' else 'Bottom-right change',xlabel='Displayed correct-response value',ylabel='P(report change)',ylim=(0,1.03),xticks=[1,3,5])
    axs[0].legend(fontsize=9);fig.suptitle('Does the effect survive across cue values?',fontsize=17,fontweight='bold')
    fig.supxlabel('VDA-4 · 100% bottom-right cue · 18° change · 128 scenes per color/value.\nColor and value are inseparable in this task; this is a stratified check, not a color-independent value manipulation.',fontsize=10)
    save(fig,'value_strata')

def write_report(summary,atlas):
    e=summary['primary_effects'];get=lambda arm,event:next(r for r in e if r['arm']==arm and r['event']==event)
    base=get('sham','uncued');benefit=get('cued_-4_change','uncued');cost=get('cued_-4_change','cued');cbase=get('sham','cued')
    title='Suppressing the cued location reveals an unexpected change'
    claim=f'Top-left detection rose from {base["hit_rate"]:.1%} to {benefit["hit_rate"]:.1%} when bottom-right attention was inhibited during the change. Bottom-right detection fell from {cbase["hit_rate"]:.1%} to {cost["hit_rate"]:.1%}.'
    def figure(name,caption):
        return f'<figure><a href="../figures/{name}.pdf"><img src="../figures/{name}.png" alt="{html.escape(caption)}"></a><figcaption>{caption} Click for PDF.</figcaption></figure>'
    table='<table><thead><tr><th>Intervention (frames 5–6)</th><th>Top-left detection</th><th>Bottom-right detection</th><th>False alarms</th></tr></thead><tbody>'
    arm_labels=dict(zip(primary_arm_order(),['No intervention','Inhibit bottom-right −2','Inhibit bottom-right −4','Inhibit bottom-right −8','Stimulate top-left +4','Stimulate bottom-right +4','Inhibit top-left −4']))
    for arm in primary_arm_order():
        a=get(arm,'uncued');b=get(arm,'cued')
        table+=f'<tr><td>{arm_labels[arm]}</td><td>{a["hit_rate"]:.1%}</td><td>{b["hit_rate"]:.1%}</td><td>{a["false_alarm_rate"]:.1%}</td></tr>'
    table+='</tbody></table>'
    native='<table><thead><tr><th>Task</th><th>Training-style θ maximum</th><th>Fresh BA</th><th>AUC</th><th>Trials</th></tr></thead><tbody>'
    for r in summary['native']:native+=f'<tr><td>VDA-{r["size"]}</td><td>{r["theta"]}°</td><td>{r["ba"]:.1%}</td><td>{r["auc"]:.3f}</td><td>{r["n"]}</td></tr>'
    native+='</tbody></table>'
    broad=pd.read_csv(OUT/'all_intervention_effects.csv')
    transfer='<table><thead><tr><th>Task</th><th>Opposite-change detection</th><th>Cued-change detection</th><th>Opposite/catch BA gain (95% interval)</th></tr></thead><tbody>'
    for size in (4,9,16):
        d=broad[(broad['size']==size)&(broad.validity==1)&(broad.angle==18)]
        select=lambda arm,event:d[(d.arm==arm)&(d.event==event)].iloc[0]
        u0=select('sham','uncued');u1=select('cued_-4_change','uncued')
        c0=select('sham','cued');c1=select('cued_-4_change','cued')
        transfer+=f'<tr><td>VDA-{size}</td><td>{u0.hit_rate:.1%} → {u1.hit_rate:.1%}</td><td>{c0.hit_rate:.1%} → {c1.hit_rate:.1%}</td><td>{100*u1.ba_delta:+.1f} pp ({100*u1.low:+.1f} to {100*u1.high:+.1f})</td></tr>'
    transfer+='</tbody></table>'
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>VDA attention experiments</title>
<style>body{{margin:0;background:#f4f7f9;color:#20343e;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1220px;margin:auto;padding:42px 28px}}h1{{font-size:42px;line-height:1.12;max-width:950px}}h2{{font-size:28px;margin-top:50px}}h3{{font-size:17px}}a{{color:#087e8b}}.eyebrow{{color:#087e8b;font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:13px}}.lead{{font-size:23px;max-width:1000px}}.card,figure{{background:white;border:1px solid #dfe7eb;border-radius:12px;padding:22px;margin:22px 0}}figure img{{width:100%;display:block}}figcaption,.small{{font-size:14px;color:#526873}}.facts{{display:flex;gap:12px;flex-wrap:wrap}}.facts span{{background:#e5eeef;padding:8px 14px;border-radius:30px;font-size:14px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:11px;text-align:left;border-bottom:1px solid #dde5e8}}th{{color:#087e8b}}.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center}}select{{max-width:100%}}.controls label{{max-width:100%;min-width:0}}.panels h3{{min-height:3em}}select,button{{font:inherit;padding:7px;border:1px solid #bdcdd5;border-radius:6px;background:white}}button{{cursor:pointer}}.panels{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}canvas{{width:100%;height:auto;background:#ddd;image-rendering:pixelated}}#time{{width:min(600px,80%)}}.legend{{height:18px;background:linear-gradient(90deg,#000004,#51127c,#b73779,#fc8961,#fcfdbf);margin:15px 0 4px}}.ticks{{display:flex;justify-content:space-between;font-size:12px}}@media(max-width:800px){{.panels{{grid-template-columns:repeat(2,minmax(0,1fr))}}h1{{font-size:30px}}main{{padding:22px 14px}}}}</style>
<main><div class="eyebrow">Frozen checkpoint 16,000 · VDA4 / VDA9 / VDA16</div><h1>{title}</h1><p class="lead">{claim}</p>
<div class="facts"><span>{summary['presentations']:,} frozen inference presentations</span><span>384 scenes in the primary test</span><span>90 attention groups · all 7 timesteps</span><span>No retraining</span></div>
<p><a href="#primary">Exact hypothesis</a> · <a href="#psychometrics">Psychometrics</a> · <a href="#sweep">Dose and timing</a> · <a href="#atlas">Explore attention</a> · <a href="#methods">Methods and downloads</a></p>
<section id="primary"><h2>The exact test you proposed</h2><p>VDA-4. A 100% cue at bottom-right, followed by an 18° change at top-left. Bottom-right priority logits are reduced only during frames 5 and 6. Matched movies also place the change at bottom-right, or contain no change.</p>
<p><strong>The predicted reciprocal effect is present in this model.</strong> Bottom-right inhibition −4 improved top-left detection by {benefit['hit_delta']*100:.1f} percentage points (paired 95% interval {benefit['hit_low']*100:.1f} to {benefit['hit_high']*100:.1f}). False alarms changed from {base['false_alarm_rate']:.1%} to {benefit['false_alarm_rate']:.1%}. For top-left changes plus catches, balanced accuracy rose from {base['ba']:.1%} to {benefit['ba']:.1%}; d′ rose from {base['dprime']:.2f} to {benefit['dprime']:.2f}.</p>
{figure('primary_dual_effect','The same 384 scenes are repeated across interventions. Benefit/cost tests use exact paired binary tests, Holm-corrected across 8 planned directional comparisons.')}
<div class="card">{table}</div><p>This is a deliberate violation of a perfectly reliable cue. An uncued change never follows a 100% cue in native training. In VDA, however, any physical change is reportable: top-left detections are correct responses, not false alarms. The result measures how an intervention changes selection when spatial expectation is violated; it does not by itself reproduce a biological inactivation experiment.</p></section>
<section id="psychometrics"><h2>Evidence strength and cue reliability</h2>{figure('psychometrics','Observed rates with pointwise 95% intervals; no fitted sigmoid. Uncued controlled curves use the diagonally opposite location.')}
<p>There is a clear cued-location advantage, but changing the displayed validity produces only modest behavioral changes at 18°. The intervention result establishes spatial competition; it does not establish that the model has learned a calibrated interpretation of the ring probabilities.</p>
{figure('value_strata','The main VDA-4 effect split by the three color/value cues.')}
<details><summary>Fresh native-task checks: 512 trials per condition</summary>{native}</details></section>
<section id="sweep"><h2>Location, dose and timing</h2>{figure('dose_site_sweep','35 exploratory arms per VDA size include site, dose, timing, both-site, blank-area and global-shift controls.')}
{figure('timing_sweep','Cue, blank delay, sample and change interventions are distinct. Main sweeps also include 75% validity and 40° changes; see the complete effect table.')}
<h3>The same change-time inhibition across display sizes</h3><p>100% cue, 18° change, cued-site inhibition −4 only during frames 5–6. These exploratory checks use a separate set of 48 scenes, each tested in both diagonal cue arrangements. They are additional stimulus tests of this checkpoint, not independently trained models.</p><div class="card">{transfer}</div>
{figure('attention_allocation','The attention distribution is recorded at every model timestep, using the actual single CLS-query weights.')}</section>
<section id="atlas"><h2>Explore every attention frame</h2><div class="card"><div class="controls"><label>Task <select id="task"><option value="4">VDA-4</option><option value="9">VDA-9</option><option value="16">VDA-16</option></select></label><label>Condition <select id="group"></select></label><label>View <select id="example"><option value="0">Example 1</option><option value="1">Example 2</option><option value="2">Example 3</option><option value="mean">Condition mean</option></select></label></div>
<p id="description" class="small"></p><div class="panels"><div><h3 id="title1">Actual model input</h3><canvas id="original" width="100" height="100"></canvas></div><div><h3>Attention overlay</h3><canvas id="overlay" width="100" height="100"></canvas></div><div><h3 id="title3">Its projected attention</h3><canvas id="map" width="100" height="100"></canvas></div><div><h3>Condition mean attention</h3><canvas id="meanmap" width="100" height="100"></canvas></div></div>
<div class="legend"></div><div class="ticks"><span>0.0001</span><span>0.001</span><span>0.01</span><span>0.1</span><span>1</span></div><p class="small">Fixed logarithmic color scale: attention mass per patch. Nearest-neighbor projection preserves the actual 10×10 grid. Saturated low values are ≤0.0001; maps sum to 1 before projection.</p>
<div class="controls"><button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button><input id="time" type="range" min="0" max="6" value="0"><span id="frame"></span></div><p class="small" id="caution"></p></div>
<p>Animated comparisons: <a href="../figures/attention_vda4.gif">VDA-4 GIF</a> · <a href="../figures/attention_vda9.gif">VDA-9 GIF</a> · <a href="../figures/attention_vda16.gif">VDA-16 GIF</a>. Playback is slowed for inspection.</p>
{figure('attention_timeline_vda4','VDA-4: one movie shown with and without bottom-right inhibition, plus the independently labeled condition mean.')}
<details><summary>VDA-9 and VDA-16 every-timestep contact sheets</summary>{figure('attention_timeline_vda9','VDA-9 matched intervention timeline.')}{figure('attention_timeline_vda16','VDA-16 matched intervention timeline.')}</details></section>
<section id="methods"><h2>Methods and limits</h2><p>One validation-selected checkpoint, with every parameter frozen and no optimizer. Native rendering, per-frame 5° orientation jitter and pixel noise are retained. Controlled analysis fixes cue, validity, sign and change location after consuming native trial draws. Native-equivalence, zero-bias and global-softmax-shift controls are checked. Identical stimulus arrays are reused across interventions.</p>
<p>Inhibition/stimulation means an additive bias before the spatial softmax, applied to every token whose center lies in the selected stimulus cell. A −4 dose multiplies that cell's unnormalized attention weights by exp(−4), about 1/55, before attention is redistributed across the image. The first manipulated frame is checked against this exact formula. Later frames can also reflect changed recurrent memory. This is not literal neural silencing. Cell footprints differ across VDA sizes, so equal doses are not equal biological currents or a pure set-size manipulation.</p>
<p>The model has one CLS query over 100 image patches. Averaging incoming attention over its one query gives the actual attention row; there is no 100×100 patch-query matrix to average. The atlas includes individual stimuli/maps and separately labeled condition means across 48 scenes, pooling the three colors and both rotation signs. Mean images blur varying Gabor orientations.</p>
<p>Primary tests: 384 independent scenes, all three event types paired within scene, 8 planned one-sided exact binary comparisons with Holm correction. Paired bootstrap intervals resample whole scenes. Exploratory sweeps reuse 48 scenes in mirrored geometries and cluster both mirrors together; intervals are pointwise. These results establish a computational intervention effect in one trained model, not replication across training seeds or biological causation.</p>
<p><a href="../REPORT.md">Written report</a> · <a href="../protocol.json">Prospective protocol</a> · <a href="../primary_effects.csv">Primary effects</a> · <a href="../primary_hypothesis_tests.csv">Corrected hypothesis tests</a> · <a href="../all_intervention_effects.csv">All intervention effects</a> · <a href="../psychometric_points.csv">All psychometric points</a> · <a href="../native_value_validity_strata.csv">Native value/validity strata</a> · <a href="../completion_verified.json">Audit receipt</a></p></section></main><script src="atlas.js"></script><script src="viewer.js"></script></html>'''
    (GALLERY/'index.html').write_text(body,encoding='utf8')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    text=f'''# Frozen VDA analysis: checkpoint 16000

{claim}

The primary VDA-4 test uses 384 matched scenes per event/arm, a 100% bottom-right
cue, 18-degree changes, and manipulation only in frames 5–6. With bottom-right
inhibition −4, top-left detection improves by {benefit['hit_delta']*100:.2f}pp
(paired 95% interval {benefit['hit_low']*100:.2f} to{benefit['hit_high']*100:.2f}).
False alarms fall from {base['false_alarm_rate']:.2%} to{benefit['false_alarm_rate']:.2%}.
Top-left/catch balanced accuracy improves from {base['ba']:.2%} to{benefit['ba']:.2%}.
The reciprocal loss of bottom-right detection is {cost['hit_delta']*100:.2f}pp.

This deliberately violates the perfectly predictive cue; such uncued changes
are absent from native 100%-validity training. Uncued physical changes are still
positive VDA labels. False alarms refer only to no-change catches.

{summary['presentations']:,} inference presentations completed, including
8,064  primary, 4,608  native, 18,144  psychometric, 100,800 exploratory intervention
presentations. 90 attention groups contain every one of 7 model frames, three
individual examples each, and a 48-scene condition mean. Original checkpoint
and learned tensors unchanged; no training or threshold tuning occurred.

Eight predeclared directional comparisons use exact paired binary tests with
Holm correction. Full results: primary_hypothesis_tests.csv. All other effects
are exploratory; cluster-bootstrap intervals resample scenes, treating mirrored
geometries as repeated observations. Rendering adapters passed native equality;
zero and global-shift attention controls passed. Maps use the actual single
CLS-query row, projected without inventing an NxN attention matrix.

The model intervention is additive priority-logit bias, not biological current.
This is one trained checkpoint, not a training-seed replication. The 100×100 input
adaptation and different stimulus-cell sizes preclude a pure capacity comparison.

Open gallery/index.html for figures, interactive attention frames, GIFs and data.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf8')

def primary_arm_order():
    return ['sham','cued_-2_change','cued_-4_change','cued_-8_change','uncued_+4_change','cued_+4_change','uncued_-4_change']

def main():
    FIG.mkdir(exist_ok=True);GALLERY.mkdir(exist_ok=True)
    completion=json.loads((OUT/'completion.json').read_text())
    primary=rows('primary');effects,tests=primary_statistics(primary);primary_plot(primary,effects);value_plot(primary)
    native=native_summary(rows('native'));points=psychometrics(rows('psychometrics'));exploratory=matched_main(rows('interventions'))
    atlas=atlas_report()
    summary=dict(presentations=completion['presentations'],primary_effects=effects,primary_tests=tests,native=native,
        atlas_groups=len(atlas),atlas_timesteps=len(atlas)*7,psychometric_points=len(points),exploratory_effects=len(exploratory))
    write_report(summary,atlas)
    import shutil
    shutil.copy2(Path(__file__).with_name('viewer.js'),GALLERY/'viewer.js')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['primary_effects','primary_tests','native']}))

if __name__=='__main__':main()
