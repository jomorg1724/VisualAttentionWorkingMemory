#!/usr/bin/env python3
"""Render the committed KDA records; no training, inference, or invented spatial maps.
Run: python3 SecondPass/KDA_paper/figures/build_figures.py
"""
from pathlib import Path
import csv, hashlib, html, json, subprocess, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from scipy.special import ndtr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT/'WorkingMemory/PlainBaseline/runs/local_kda_program_20260917'
CLOUD = ROOT/'WorkingMemory/PlainBaseline/runs/cloud_20260917_010358/pulled/results'
DATA = OUT/'data'; DATA.mkdir(exist_ok=True)
SOURCES = {}
def load(p):
    p=Path(p); raw=p.read_bytes(); SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
def csvload(p):
    SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)
probe=load(RUN/'analysis/kda_probe/kda_probe.json')
psych=load(RUN/'analysis/psychometric/psychometric.json')
program=load(RUN/'program_receipt.json')
STAGES=['ring','cued','delayA','delayB','delayC']
NAMES=['Ring cue / D0','Signed cue / D0','Delay A / D0,1,2','Delay B / D0,2,4','Delay C / D0,4,12,24']
receipts={s:load(RUN/'kda_s1'/s/'receipt.json') for s in STAGES}
metrics={s:csvload(RUN/'kda_s1'/s/'metrics.csv') for s in STAGES}
validation={s:load(RUN/'kda_s1'/s/'validation.json') for s in STAGES}
cloud={p.parent.parent.name:load(p) for p in sorted(CLOUD.glob('*/*/delayC/receipt.json'))}
assert len(cloud)==8
assert len(probe['delays'])==4 and all(len(d['scales'])==3 for d in probe['delays'])
assert psych['n_per_point']==512 and probe['n']==128
for s in STAGES:
    assert len(metrics[s])==receipts[s]['updates_done']
    assert metrics[s].episodes.iloc[-1]==receipts[s]['episodes']
for d in probe['delays']:
    for sc in d['scales']:
        assert all(len(v)==d['delay']+4 for v in sc['attention_abs_mean'].values())
        assert len(sc['state_probe_r2_by_frame'])==d['delay']+4

# Long-form source tables are directly reusable without Python or the plotting code.
rows=[]
for sweep,sw in psych['sweeps'].items():
    groups=sw.items() if sweep=='magnitude' else [(sweep,sw)]
    for group,v in groups:
        for p in v['points']:
            r={'sweep':sweep,'condition':group,**{k:x for k,x in p.items() if k!='ba_ci'},'ci_low':p['ba_ci'][0],'ci_high':p['ba_ci'][1]}
            rows.append(r)
pd.DataFrame(rows).to_csv(DATA/'psychometric_points.csv',index=False)
pd.DataFrame([{'delay':int(d[1:]),**v['fit']} for d,v in psych['sweeps']['magnitude'].items()]).to_csv(DATA/'psychometric_fits.csv',index=False)
gates=[]; att=[]; state=[]
for d in probe['delays']:
    for sc in d['scales']:
        for phase,g in sc['gates'].items():
            for key,value in g.items():
                gate,region=key.split('_',1); gates.append(dict(delay=d['delay'],scale=sc['scale'],phase=phase,gate=gate,region=region,value=value))
        for region,vals in sc['attention_abs_mean'].items():
            for frame,value in enumerate(vals):att.append(dict(delay=d['delay'],scale=sc['scale'],region=region,source_frame=frame,mean_absolute_weight=value))
        for frame,value in enumerate(sc['state_probe_r2_by_frame']):state.append(dict(delay=d['delay'],scale=sc['scale'],frame=frame,r2=value))
pd.DataFrame(gates).to_csv(DATA/'gate_phase_means.csv',index=False)
pd.DataFrame(att).to_csv(DATA/'implicit_attention_profiles.csv',index=False)
pd.DataFrame(state).to_csv(DATA/'state_angle_decoding.csv',index=False)
pd.DataFrame([{'delay':int(d[1:]),**v} for d,v in probe['interventions'].items()]).to_csv(DATA/'interventions.csv',index=False)
tr=[]; va=[]; tests=[]
for s in STAGES:
    df=metrics[s].copy();df.insert(0,'stage',s);tr.append(df)
    for v in validation[s]:
        for cell,m in v['cells'].items():va.append(dict(stage=s,update=v['update'],episodes=v['episodes'],cell=cell,**{k:x for k,x in m.items() if k!='confusion'}))
    for cell,m in receipts[s]['final']['terminal']['test'].items():tests.append(dict(run='local_kda_s1',stage=s,cell=cell,**{k:x for k,x in m.items() if k!='confusion'}))
for run,r in cloud.items():
    for cell,m in r['final']['terminal']['test'].items():tests.append(dict(run=run,stage='delayC',cell=cell,**{k:x for k,x in m.items() if k!='confusion'}))
pd.concat(tr).to_csv(DATA/'training_metrics.csv',index=False)
pd.DataFrame(va).to_csv(DATA/'validation_metrics.csv',index=False)
pd.DataFrame(tests).to_csv(DATA/'terminal_tests.csv',index=False)

INK='#192b3b'; MUTED='#526577'; BLUE='#176b9b'; TEAL='#14806e'; GOLD='#b17817'; PURPLE='#855aa5'; RED='#c04b49'
COLORS=[BLUE,TEAL,GOLD,PURPLE]; REGIONS=['cued','uncued','background']; RC=[BLUE,GOLD,'#8796a4']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':11,'axes.edgecolor':'#b6c2cc','axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#dce3e9','grid.alpha':.7,'grid.linewidth':.65,'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'lines.linewidth':2.0})
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','sans-serif']
manifest=[]
PDF=PdfPages(OUT/'KDA_results_atlas.pdf')
PDF.infodict().update(Title='KDA results atlas',Subject='Saved orientation-study results; spatial maps unavailable',Author='Visual Attention and Working Memory — figures from recorded results')

def canvas(rows=1,cols=1,figsize=(13.3,8.6)):
    fig,axs=plt.subplots(rows,cols,figsize=figsize,squeeze=False)
    fig.subplots_adjust(left=.08,right=.96,bottom=.22,top=.83,wspace=.32,hspace=.6)
    return fig,axs

def save(fig,slug,title,caption,category,sources):
    num=len(manifest)+1; fid=f'{num:02d}_{slug}'
    fig.suptitle(title,x=.055,y=.962,ha='left',fontsize=20,fontweight='bold')
    lines=textwrap.wrap(caption,145)
    fig.text(.055,.135,'\n'.join(lines),fontsize=10,color=MUTED,va='top',linespacing=1.4)
    fig.text(.055,.025,f'KDA / ORIENTATION STUDY     •     FIGURE {num:02d}     •     Source: {sources}',fontsize=8,color=MUTED)
    fig.savefig(OUT/f'{fid}.png',dpi=160)
    fig.savefig(OUT/f'{fid}.svg')
    fig.savefig(OUT/f'{fid}.pdf')
    PDF.savefig(fig)
    manifest.append(dict(id=fid,title=title,caption=caption,category=category,source=sources,png=f'{fid}.png',svg=f'{fid}.svg',pdf=f'{fid}.pdf'))
    plt.close(fig)

def ba_axis(ax,lo=.45):
    ax.set_ylim(lo,1.035);ax.axhline(.5,color=MUTED,ls=':',lw=1);ax.set_ylabel('Balanced accuracy')

def errors(ax,pts,xkey,color=BLUE,label=None,offset=0.0):
    x=np.array([p[xkey] for p in pts])+offset;y=np.array([p['ba'] for p in pts]);ci=np.array([p['ba_ci'] for p in pts])
    ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-ci[:,0],ci[:,1]-y))),fmt='o-',color=color,label=label,capsize=3,ms=5)

def hm(ax,a,xlabels,ylabels,vmin=0.0,vmax=1.0,cmap='viridis',fmt='.3f'):
    arr=np.ma.masked_invalid(np.array(a,dtype=float)); im=ax.imshow(arr,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest');ax.grid(False)
    ax.set_xticks(range(len(xlabels)),xlabels);ax.set_yticks(range(len(ylabels)),ylabels)
    for (i,j),v in np.ndenumerate(np.asarray(a)):
        if np.isfinite(v):
            rgba=im.cmap(im.norm(v));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,format(v,fmt),ha='center',va='center',color='white' if lum<.53 else INK,fontsize=10)
        else:ax.text(j,i,'—',ha='center',va='center',color=MUTED)
    return im

def phase(ax,d,labels=False):
    ax.axvspan(.5,2.5,color='#dceee8',alpha=.7,zorder=0)
    if d:
        ax.axvspan(2.5,d+2.5,color='#edf0f3',alpha=.8,zorder=0)
        ax.axvspan(2.5,min(4.5,d+2.5),color='#f4dfab',alpha=.75,zorder=0)
    ax.axvline(d+3,color=RED,ls=':',lw=1)
    ax.set_xlim(-.3,d+3.4)
    if labels:ax.set_xticks(sorted(set([0,1,2,d+3]+list(range(4,d+3,4)))))
    ax.set_xlabel('Source frame (0-based)')

# 01. Conceptual architecture, sourced from implementation, not a measured activation map.
fig,axs=canvas();ax=axs[0,0];ax.set_axis_off();ax.set_xlim(0,13);ax.set_ylim(0,6)
def box(x,y,w,h,label,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.08',ec=color,fc='#f1f6f9',lw=1.5));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
def arrow(x,y,xx,yy):ax.annotate('',(xx,yy),(x,y),arrowprops=dict(arrowstyle='->',lw=1.5,color=MUTED))
box(.1,4.4,1.7,1.0,'Frame stack\n9 × 100 × 100')
box(2.3,4.4,1.7,1.0,'Conv 1\n32 × 50 × 50')
box(4.5,4.4,1.9,1.0,'Conv 2\n64 × 25 × 25')
box(7,4.4,1.9,1.0,'Conv 3\n96 × 13 × 13')
box(9.5,4.4,2,1.0,'Conv 4\n128 × 7 × 7')
for a,b in [(1.8,2.3),(4,4.5),(6.4,7),(8.9,9.5)]:arrow(a,4.9,b,4.9)
for x,w,label in [(4.5,1.9,'25 × 25'),(7,1.9,'13 × 13'),(9.5,2,'7 × 7')]:
    box(x,2.5,w,1.05,f'32-ch projection\nSpatial KDA\n{label}',TEAL);arrow(x+w/2,4.4,x+w/2,3.55)
    ax.text(x+w/2,2.18,'S(t−1) → S(t)',ha='center',fontsize=10,color=TEAL)
    ax.text(x+w/2,3.93,'concat 32-ch output',ha='center',fontsize=9,color=TEAL)
    junction=x+w+.3 if x<9 else 12.3
    ax.plot([x+w,junction,junction],[3.0,3.0,4.9],color=TEAL,lw=1.5)
    ax.plot(junction,4.9,'o',color=TEAL,ms=4)
box(8.9,.2,2.6,1.1,'Flatten 160 × 7 × 7\nLinear + ReLU → 256')
arrow(11.5,4.8,12.3,4.8);arrow(12.3,4.8,12.3,.75);arrow(12.3,.75,11.5,.75)
box(5.8,.2,2.4,1.1,'Temporal GRU\n256 hidden units');arrow(8.9,.75,8.2,.75)
box(2.8,.2,2.3,1.1,'Final hidden state\nTask-specific head');arrow(5.8,.75,5.1,.75)
ax.text(.15,2.3,'Each KDA site:\n2 heads\n8 × 16 state / head\nLearned decay + delta write',fontsize=12,linespacing=1.6)
save(fig,'architecture','Architecture / spatial memory inside the encoder','Schematic of the implemented KDA arm. Each accumulator output is concatenated with its convolutional feature map before the next block (or final flatten). All learned parameters were trainable. This is not a spatial attention visualization.','Overview','PlainBaseline/accum.py; TemporalIntegration/accumulators.py')

# 02. Trial structure is exact metadata, not a reconstructed trial image.
fig,axs=canvas(4,1);delays=[d['delay'] for d in probe['delays']]
for ax,d in zip(axs.flat,delays):
    for t in range(d+4):
        col=GOLD if t==0 else TEAL if t in [1,2] else RED if t==d+3 else '#dce3e9'
        ax.barh(0,1,left=t-.5,height=.65,color=col,edgecolor='white')
    ax.set_xlim(-.5,28);ax.set_yticks([0],[f'D{d}']);ax.set_ylim(-.65,.65);ax.grid(False)
    ax.set_xticks(range(0,28,4));ax.set_xlabel('Frame index')
fig.legend(handles=[matplotlib.patches.Patch(color=c,label=l) for c,l in [(GOLD,'Cue-only'),(TEAL,'Samples + cue'),('#dce3e9','Blank delay'),(RED,'Probe / report')]],loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,fontsize=11)
save(fig,'trial_timeline','Trial structure / what each time index means','Frame 0 is cue-only; frames 1–2 contain samples and cue; frames 3 through D+2 are blanks; frame D+3 is the probe. The rolling three-frame input still contains sample images at the first two nominal blank frames. Colored bars describe the current frame only and are schematic, not saved stimuli.','Overview','analysis/psych_stream.py, lines 50–57')

# 03. Original experiment comparison: seeds remain separate.
fig,axs=canvas(2,2)
for ax,(arm,col) in zip(axs.flat,zip(['plain','convgru','opponent','kda'],COLORS)):
    for seed,style in [(1,'-'),(2,'--')]:
        m=cloud[f'{arm}_s{seed}']['final']['terminal']['test'];x=[int(k.rsplit('D',1)[1]) for k in m];y=[v['balanced_accuracy'] for v in m.values()]
        ax.plot(x,y,style,marker='o' if seed==1 else 's',color=col,label=f'Seed {seed}',ms=5)
    ax.set_title(arm.upper() if arm=='kda' else arm);ba_axis(ax);ax.set_xticks([0,4,12,24]);ax.set_xlabel('Blank frames');ax.legend(loc='lower right')
save(fig,'cloud_comparison','Original comparison / both model seeds','Terminal delay-C tests, 512 trials per delay. Separate panels and dashed second-seed traces keep overlapping ceiling results visible. These are cloud-run results, not the later local checkpoint used for the diagnostic sweeps.','Training','cloud_20260917_010358 / eight delayC receipts')

# 04. Every local terminal test cell, missing cells explicit.
fig,axs=canvas();ax=axs[0,0];ds=[0,1,2,4,12,24];a=np.full((5,6),np.nan)
fig.subplots_adjust(left=.25)
for i,s in enumerate(STAGES):
    for k,v in receipts[s]['final']['terminal']['test'].items():a[i,ds.index(int(k.rsplit('D',1)[1]))]=v['balanced_accuracy']
im=hm(ax,a,[f'D{d}' for d in ds],NAMES,vmin=.5,vmax=1);fig.colorbar(im,ax=ax,label='Balanced accuracy');ax.set_title('Terminal checkpoint of each training stage')
save(fig,'local_terminal_results','Local rerun / the full curriculum result','Each cell is a recorded terminal-checkpoint test (512 trials). Dashes mean that delay was not tested in that stage. The ring-cue task in the first row differs from the signed-cue task in later rows.','Training','local rerun / five receipt.json files')

# 05–07. All training metrics; smoothing explicit and raw retained.
for key,title,yl,log in [('loss','Training / cross-entropy loss','Cross-entropy (log scale)',True),('accuracy','Training / minibatch accuracy','Minibatch accuracy',False),('grad_norm','Training / gradient norms','Gradient norm (log scale)',True)]:
    fig,axs=canvas(2,3)
    for ax,s,name in zip(axs.flat,STAGES,NAMES):
        df=metrics[s];x=df.episodes/1000;y=df[key].astype(float);yp=np.maximum(y,1e-6) if log else y
        ax.plot(x,yp,color=BLUE,alpha=.19,lw=.6,label='Each update');smooth=y.rolling(50,min_periods=1).mean();ax.plot(x,np.maximum(smooth,1e-6) if log else smooth,color=BLUE,lw=1.8,label='50-update mean')
        if log:ax.set_yscale('log')
        else:ax.set_ylim(0,1.04)
        ax.set_title(name,fontsize=11);ax.set_xlabel('Stage episodes (thousands)');ax.set_ylabel(yl)
    axs.flat[-1].set_axis_off();axs.flat[0].legend(fontsize=9)
    save(fig,f'training_{key}',title,'All recorded updates are shown faintly; the dark line is a trailing 50-update arithmetic mean. Stage episode counts restart at zero. Log plots floor recorded zeros at 1e−6 for display only; source CSVs retain the recorded values.','Training','local rerun / five metrics.csv files')

# 08. All validation curves and all cells.
for key,title,yl,log in [('balanced_accuracy','Validation / acquisition and retention','Balanced accuracy',False),('loss','Validation / cross-entropy by delay','Cross-entropy (log scale)',True)]:
    fig,axs=canvas(2,3)
    for ax,s,name in zip(axs.flat,STAGES,NAMES):
        vv=validation[s];cells=list(vv[0]['cells'])
        for j,cell in enumerate(cells):ax.plot([v['episodes']/1000 for v in vv],[v['cells'][cell][key] for v in vv],marker=['o','s','^','D'][j],ms=3,color=COLORS[j],label=cell.rsplit('_',1)[1])
        if log:ax.set_yscale('log')
        else:ba_axis(ax)
        ax.set_xlabel('Stage episodes (thousands)');ax.set_ylabel(yl);ax.set_title(name,fontsize=11);ax.legend(fontsize=8,loc='best')
    axs.flat[-1].set_axis_off()
    save(fig,f'validation_{key}',title,'Recorded validation looks, 256 trials per cell; lines connect evaluations, not inferred intermediate measurements. Coincident curves can overlap. These are validation results, separate from terminal tests and the psychometric stream.','Training','local rerun / five validation.json files')

# 10. Confusions, terminal vs best distinction explicit.
fig,axs=canvas(1,4)
for ax,(cell,m) in zip(axs.flat,receipts['delayC']['final']['terminal']['test'].items()):
    hm(ax,m['confusion'],['0','1'],['0','1'],vmin=0,vmax=256,fmt='.0f');ax.set_title(cell.rsplit('_',1)[1]);ax.set_xlabel('Predicted label');ax.set_ylabel('True label')
save(fig,'final_confusions','Final local checkpoint / confusion matrices','Terminal delay-C checkpoint, 512 trials per cell, 256 of each label. Label 1 denotes a target rotation aligned with the sign cue; label 0 includes unchanged or oppositely rotated targets. No errors occurred on these finite test sets.','Training','local rerun / delayC/receipt.json → final.terminal.test')

# 11. Full psychometric range plus exact saved fit and confidence intervals.
fig,axs=canvas(2,2)
for ax,(j,(cond,v)) in zip(axs.flat,enumerate(psych['sweeps']['magnitude'].items())):
    pts=v['points'];fit=v['fit'];xx=np.linspace(2.5,45,400);yy=.5+(.5-fit['lapse'])*ndtr((xx-fit['threshold_75'])/fit['spread'])
    ax.plot(xx,yy,color=COLORS[j],alpha=.6,label='Saved Gaussian fit');errors(ax,pts,'magnitude',COLORS[j],label='Recorded BA + 95% CI')
    ax.axvspan(2.5,10,color='#edf0f3',alpha=.6,zorder=0);ax.set_title(f'{cond} / fitted midpoint {fit["threshold_75"]:.2f}°');ba_axis(ax);ax.set_xticks([2.5,10,15,30,45]);ax.set_xlabel('Rotation magnitude (degrees)')
axs[0,0].legend(fontsize=8,loc='lower right')
save(fig,'psychometric_magnitude','Psychometrics / rotation sensitivity','512 trials per point; whiskers are the saved 95% trial-bootstrap intervals. Lines use the saved cumulative-Gaussian parameters without refitting. Gray region contains untrained small rotations (2.5°–10°); training magnitudes were 15°, 30°, 45°. Fits are descriptive, not human-data fits.','Psychometrics','analysis/psychometric/psychometric.json → magnitude')

# 12. Detail near threshold, not hiding full range above.
fig,axs=canvas(1,2)
for j,(cond,v) in enumerate(psych['sweeps']['magnitude'].items()):
    pts=[p for p in v['points'] if p['magnitude']<=10];errors(axs[0,0],pts,'magnitude',COLORS[j],cond)
    axs[0,1].plot([p['magnitude'] for p in pts],[p['auc'] for p in pts],marker=['o','s','^','D'][j],color=COLORS[j],label=cond)
ba_axis(axs[0,0]);axs[0,0].set_title('Balanced accuracy + saved intervals');axs[0,1].set_ylim(.8,1.01);axs[0,1].set_ylabel('AUC');axs[0,1].set_title('Score discrimination (AUC; zoomed axis)')
for ax in axs.flat:ax.set_xlabel('Rotation magnitude (degrees)');ax.set_xticks([2.5,5,7.5,10]);ax.legend()
save(fig,'small_rotation_detail','Psychometrics / the small-angle detail','Same saved observations as the full curves, zoomed to previously untrained rotation magnitudes. AUC confidence intervals were not saved and are not invented here. Delay curves share a stimulus seed; this figure does not establish a significant benefit of longer delays.','Psychometrics','analysis/psychometric/psychometric.json → magnitude')

# 13. Fit metrics, no artificial uncertainty bands.
fig,axs=canvas(1,3)
fits=[v['fit'] for v in psych['sweeps']['magnitude'].values()]
for ax,key,title,yl in zip(axs.flat,['threshold_75','slope','lapse'],['Fitted midpoint (≈75% here)','Inverse spread (not derivative)','Fitted lapse'],['Degrees','1 / degrees','Lapse probability']):
    y=[f[key] for f in fits];ax.plot(delays,y,'o-',color=BLUE);ax.set_xticks(delays);ax.set_xlabel('Blank frames');ax.set_ylabel(yl);ax.set_title(title,fontsize=11)
    for x,v in zip(delays,y):ax.annotate(f'{v:.3g}',(x,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
    ax.margins(y=.3)
    if key=='lapse':ax.set_ylim(0,.1)
save(fig,'fit_parameters','Psychometrics / fitted sensitivity parameters','Parameters are taken directly from the saved fits. The source calls 1/spread a slope; it is not the derivative of the response curve. No fit-parameter confidence intervals were saved. Threshold estimates below 2.5° extend below the smallest tested rotation.','Psychometrics','analysis/psychometric/psychometric.json → magnitude fits')

# 14. Retention curve flat, tau deliberately not interpreted.
fig,axs=canvas();ax=axs[0,0];pts=psych['sweeps']['delay']['points'];ax.axvspan(24,49,color='#edf0f3',zorder=0);errors(ax,pts,'delay');ba_axis(ax);ax.set_xlim(-1,49);ax.set_xticks([0,4,8,12,16,24,32,48]);ax.set_xlabel('Blank frames (not seconds)');ax.text(35,.8,'Beyond largest\ntraining delay',ha='center',fontsize=14,color=MUTED)
save(fig,'retention_sweep','Psychometrics / retention out to 48 blanks','Rotation magnitude 15°; 512 trials per point. All observed balanced accuracies are 1.000. The saved exponential fit is degenerate: its time constant is not identifiable and is not plotted as a biological forgetting rate. Collapsed bootstrap intervals at ceiling do not imply zero population uncertainty.','Psychometrics','analysis/psychometric/psychometric.json → delay')

# 15. Cue condition heatmaps are conditions, not image space.
fig,axs=canvas(1,2);pts=psych['sweeps']['cue']['points'];contrasts=[1,.5,.25,.1];jitters=[0,2,4]
for ax,key in zip(axs.flat,['ba','auc']):
    a=[[next(p[key] for p in pts if p['cue_scale']==c and p['jitter']==j) for j in jitters] for c in contrasts]
    im=hm(ax,a,[f'±{j} px' if j else '0 px' for j in jitters],[f'{c:g}×' for c in contrasts],vmin=.5,vmax=1);ax.set_xlabel('Uniform integer jitter bound per axis');ax.set_ylabel('Cue contrast multiplier');ax.set_title('Balanced accuracy' if key=='ba' else 'AUC');fig.colorbar(im,ax=ax,shrink=.8)
save(fig,'cue_heatmaps','Psychometrics / cue visibility and position','D4, 512 trials per point. These are condition-grid heatmaps, not spatial attention maps. Jitter means independent integer offsets sampled within ±2 or ±4 pixels on each axis, not a fixed displacement. The original glyph defines contrast 1×.','Psychometrics','analysis/psychometric/psychometric.json → cue')

# 16. Cue curves with intervals.
fig,axs=canvas();ax=axs[0,0]
for j,c in enumerate(contrasts):errors(ax,[p for p in pts if p['cue_scale']==c],'jitter',COLORS[j],f'Contrast {c:g}×',offset=(j-1.5)*.035)
ba_axis(ax);ax.set_xticks(jitters);ax.set_xlabel('Jitter bound per axis (pixels)');ax.legend(title='Glyph visibility',loc='lower left')
save(fig,'cue_curves','Psychometrics / cue degradation with uncertainty','Saved 95% trial-bootstrap intervals. Tiny horizontal offsets separate overlapping error bars and do not change the jitter conditions. Cue scale and jitter were changed together in the factorial sweep; the results alone do not locate a unique failure mechanism.','Psychometrics','analysis/psychometric/psychometric.json → cue')

# 17. Distraction includes source BA CIs and AUC.
fig,axs=canvas(1,2);pts=psych['sweeps']['distraction']['points']
for j,same in enumerate([False,True]):
    pp=[p for p in pts if p['same_magnitude']==same];label='Uncued: all nonzero ±magnitude' if same else 'Uncued: original − / 0 / + pattern';errors(axs[0,0],pp,'n_distractors',COLORS[j],label,offset=(j-.5)*.035)
    axs[0,1].plot([p['n_distractors']+(j-.5)*.035 for p in pp],[p['auc'] for p in pp],'-o' if not same else '--s',color=COLORS[j],label=label)
ba_axis(axs[0,0],lo=.97);axs[0,0].set_title('Balanced accuracy (zoomed axis)');axs[0,1].set_ylim(.97,1.004);axs[0,1].set_ylabel('AUC');axs[0,1].set_title('AUC (zoomed axis)')
for ax in axs.flat:ax.set_xlabel('Uncued Gabors present');ax.set_xticks(range(4));ax.legend(fontsize=9,loc='lower left')
save(fig,'distraction','Psychometrics / uncued-Gabor interference','D12, 512 trials per point. Zoomed y axes show the small differences near ceiling. No AUC intervals were saved. These are simultaneous uncued Gabors, not new distractors inserted during retention; absence of an observed cost here is not a general interference-capacity claim.','Psychometrics','analysis/psychometric/psychometric.json → distraction')

# 18. Interventions full source values; no absent D0 blank trials fabricated.
fig,axs=canvas();ax=axs[0,0];keys=['baseline','reset_before_probe','beta_zero_blanks','alpha_one_blanks'];labels=['Unmodified','Reset KDA state before probe','β = 0 on blanks (no writes)','α = 1 on blanks (no diagonal decay)']
x=np.arange(4);width=.18
for j,(key,label,col) in enumerate(zip(keys,labels,[BLUE,RED,TEAL,GOLD])):
    for i,d in enumerate(delays):
        v=probe['interventions'][f'D{d}'].get(key)
        if v is not None:
            ax.bar(i+(j-1.5)*width,v,width,color=col,label=label if i==(0 if j<2 else 1) else None);ax.text(i+(j-1.5)*width,v+.012,f'{v:.3f}',ha='center',fontsize=8,rotation=90,va='bottom')
ax.set_ylim(0,1.16);ax.axhline(.5,color=MUTED,ls=':');ax.set_xticks(x,[f'D{d}' for d in delays]);ax.set_ylabel('Balanced accuracy');ax.legend(loc='upper left',bbox_to_anchor=(0,1.25),ncol=2,fontsize=10)
fig.subplots_adjust(top=.73)
save(fig,'memory_interventions','Memory probes / perturbing the stored state','128 trials per delay. Reset affects all KDA scales, not the GRU. β=0 removes writes and delta correction; α=1 removes diagonal decay, not all possible forgetting. Blank clamps are absent at D0. No paired outcomes or uncertainty were saved. Reset shows dependence under perturbation, not exclusive memory localization.','Memory probes','analysis/kda_probe/kda_probe.json → interventions')

# 19–21. Every gate mean across phase, region, scale and delay.
for si,size in enumerate([25,13,7]):
    fig,axs=canvas(2,4);fig.subplots_adjust(wspace=.4,hspace=.6)
    for ci,d in enumerate(probe['delays']):
        sc=d['scales'][si];phases=['sample','blank','probe']
        for ri,gate in enumerate(['beta','alpha']):
            a=[[sc['gates'].get(p,{}).get(f'{gate}_{r}',np.nan) for p in phases] for r in REGIONS]
            hm(axs[ri,ci],a,['Sample','Blank','Probe'],['Cued','Uncued','BG'],vmin=0,vmax=1);axs[ri,ci].set_title(f'D{d["delay"]} / '+('write β' if gate=='beta' else 'retain α'),fontsize=11)
    save(fig,f'gate_means_scale{si}',f'Gates / {size} × {size} spatial scale','Phase means pooled over trials, masked sites and heads; alpha is also averaged over key dimensions. Color range is fixed 0–1 across every panel. D0 has no blank phase. These summaries do not retain per-frame gate dynamics, per-head differences or individual spatial maps.','Memory probes','analysis/kda_probe/kda_probe.json → scales.gates')

# 22–24. Time traces for every scale and region. Non-normalized absolute coefficients.
for si,size in enumerate([25,13,7]):
    fig,axs=canvas(2,2)
    vmax=max(max(sc['attention_abs_mean'][r]) for d in probe['delays'] for sc in [d['scales'][si]] for r in REGIONS)*1.15
    for ax,d in zip(axs.flat,probe['delays']):
        for r,col,style in zip(REGIONS,RC,['-','--',':']):
            y=d['scales'][si]['attention_abs_mean'][r];ax.plot(range(len(y)),y,style,marker='o',ms=3,color=col,label=r.capitalize())
        phase(ax,d['delay'],True);ax.set_ylim(0,vmax);ax.set_ylabel('Mean |implicit coefficient|');ax.set_title(f'D{d["delay"]} / read at frame {d["delay"]+3}');ax.legend(fontsize=9)
    save(fig,f'attention_profiles_scale{si}',f'Implicit attention / {size} × {size} spatial scale','Read at final probe; x is the source frame. Green: samples; gray: blanks; amber: first two blanks still contain samples in the three-frame input; red line: probe. Values are mean absolute coefficients over trials, sites and heads—not probabilities, signed contributions or decision attribution. No spatial maps are reconstructed.','Attention','analysis/kda_probe/kda_probe.json → attention_abs_mean')

# 25. All D24 temporal heatmaps, exact aggregate data only.
fig,axs=canvas(3,1);d=probe['delays'][-1];vmax=max(max(v) for sc in d['scales'] for v in sc['attention_abs_mean'].values())
for ax,sc in zip(axs.flat,d['scales']):
    arr=np.array([sc['attention_abs_mean'][r] for r in REGIONS]);im=ax.imshow(arr,aspect='auto',vmin=0,vmax=vmax,cmap='magma');ax.grid(False);ax.set_yticks([0,1,2],['Cued','Uncued','BG']);ax.set_xticks([0,1,2,6,10,14,18,22,26,27]);ax.set_xlabel('Source frame');ax.set_title(f'{sc["map"][0]} × {sc["map"][1]} scale',fontsize=11);ax.axvline(2.5,color='white',lw=1);ax.axvline(26.5,color='white',lw=1);fig.colorbar(im,ax=ax,label='Mean |w|',fraction=.035)
save(fig,'temporal_attention_heatmaps','Implicit attention / D24 temporal coefficient heatmaps','Rows are region averages, columns are time—not image coordinates. All panels share the same absolute color scale. Boundaries separate the sample/cue prefix, blank interval and probe. Signed coefficients, individual trials and pixel/site maps are unavailable in this checkout.','Attention','analysis/kda_probe/kda_probe.json → D24 attention_abs_mean')

# 26. Decoding curves including negative R2 and contaminated post-probe time explicitly.
fig,axs=canvas(2,2);rmin=min(min(sc['state_probe_r2_by_frame']) for d in probe['delays'] for sc in d['scales']);rlo=min(-.2,rmin-.05)
for ax,d in zip(axs.flat,probe['delays']):
    for j,sc in enumerate(d['scales']):
        y=sc['state_probe_r2_by_frame'];ax.plot(range(len(y)-1),y[:-1],'-o',ms=3,color=COLORS[j],label=f'{sc["map"][0]} × {sc["map"][1]}');ax.scatter(len(y)-1,y[-1],facecolors='none',edgecolors=COLORS[j],s=50)
    phase(ax,d['delay'],True);ax.set_ylim(rlo,1.08);ax.axhline(0,color=MUTED,ls=':',lw=1);ax.set_ylabel('Held-out R²');ax.set_title(f'D{d["delay"]} / sample-angle readout');ax.legend(fontsize=9)
save(fig,'state_decoding','Memory probes / sample orientation decodable over time','Exploratory same-time least-squares fits: 64 fit / 64 held-out trials, 257 features with intercept; underdetermined, no regularization or intervals. Amber frames still include samples in stacked input, so are not pure recurrent retention. Open final markers follow the probe and may reflect sample–probe correlation.','Memory probes','analysis/kda_probe/kda_probe.json → state_probe_r2_by_frame')

# 27. End-of-delay representation, no interpolation at untested delays.
fig,axs=canvas();ax=axs[0,0];a=[[d['scales'][s]['state_probe_r2_by_frame'][-2] for d in probe['delays']] for s in range(3)]
im=hm(ax,a,[f'D{d}' for d in delays],['25 × 25','13 × 13','7 × 7'],vmin=0,vmax=1);fig.colorbar(im,ax=ax,label='Held-out R²');ax.set_xlabel('Delay condition');ax.set_ylabel('Spatial scale')
save(fig,'preprobe_decoding','Memory probes / the last state before the probe','Last pre-probe frame is the second sample at D0 and the last blank at other delays. Each cell is a separately fitted, same-time held-out decoder of cos(2θ), sin(2θ); it is not a cross-time generalization test or an angular-error estimate.','Memory probes','analysis/kda_probe/kda_probe.json → penultimate R² entries')

# 28. Differences derived, not claims of significant cue selectivity.
fig,axs=canvas(1,2)
for ax,gate in zip(axs.flat,['beta','alpha']):
    a=[[sc['gates']['sample'][f'{gate}_cued']-sc['gates']['sample'][f'{gate}_uncued'] for sc in d['scales']] for d in probe['delays']]
    lim=max(.01,float(np.max(np.abs(a))));im=hm(ax,a,['25×25','13×13','7×7'],[f'D{d}' for d in delays],vmin=-lim,vmax=lim,cmap='RdBu_r');fig.colorbar(im,ax=ax,shrink=.8);ax.set_title('Write β difference' if gate=='beta' else 'Retention α difference')
save(fig,'cue_gate_contrasts','Gates / cued minus uncued during the samples','Differences are computed from saved phase means, separately at every scale. Each panel uses its own labeled symmetric color range. No trial-level variation remains, so these contrasts cannot establish significance or prove that cue selection happens only downstream.','Memory probes','analysis/kda_probe/kda_probe.json → sample gate means (derived difference)')

# 29. Cloud computational comparison (not local/cloud causal benchmark).
fig,axs=canvas(1,2)
arms=['plain','convgru','opponent','kda'];params=[cloud[f'{a}_s1']['params']/1e6 for a in arms]
axs[0,0].bar(arms,params,color=COLORS);axs[0,0].set_ylabel('Trainable/model parameters (millions)');axs[0,0].set_title('Recorded parameter counts')
for i,v in enumerate(params):axs[0,0].text(i,v+.03,f'{v:.3f}M',ha='center',fontsize=10)
for seed,off in [(1,-.18),(2,.18)]:
    vals=[cloud[f'{a}_s{seed}']['seconds']/60 for a in arms];axs[0,1].bar(np.arange(4)+off,vals,.34,label=f'Seed {seed}',color=BLUE if seed==1 else TEAL)
axs[0,1].set_xticks(range(4),arms);axs[0,1].set_ylabel('Delay-C stage wall time (minutes)');axs[0,1].set_title('Recorded stage duration / RTX 3090');axs[0,1].legend()
save(fig,'compute_context','Compute context / original cloud comparison','Delay-C stage receipts: 200,000 training episodes per arm and seed, plus each stage’s recorded overhead. Cloud programs used two concurrent lanes on one GPU, so wall times are contextual observations—not isolated throughput benchmarks or GPU-only timings.','Training','cloud_20260917_010358 / eight delayC receipts')

PDF.close()
assert len(manifest)==29, len(manifest)
for f in manifest:
    for ext in ['png','svg','pdf']:assert (OUT/f[ext]).stat().st_size>1000
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
# Include the implementation used to interpret axes/phases in the provenance list.
for rel in ['WorkingMemory/PlainBaseline/analysis/kda_probe.py','WorkingMemory/PlainBaseline/analysis/psychometric.py','WorkingMemory/PlainBaseline/analysis/psych_stream.py','WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py']:
    SOURCES[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
manifest_doc=dict(source_commit=commit,figure_count=len(manifest),figures=manifest,sources=SOURCES,missing=['Per-trial spatial gates_D*_scale*.npz arrays','Trained terminal.pt weights','Per-trial predictions for paired intervention uncertainty'],limits=['Only orientation family for current analysis; one local seed','No inference or training was performed to generate these graphics','Temporal attention plots are region/head/trial aggregates, not spatial maps','No regenerated stimulus examples are presented as recorded trials'])
(OUT/'manifest.json').write_text(json.dumps(manifest_doc,indent=2))

notes='''# KDA results graphics

Generated exclusively from committed metrics, test receipts and analysis JSON. The architecture and trial timelines are labeled implementation schematics, not observations. No training or inference was run.

## Open
- `index.html`: offline searchable gallery, full-resolution SVGs, PNG/PDF downloads, source links.
- `KDA_results_atlas.pdf`: one figure per page.
- `01_...` through `29_...`: each figure in PNG, SVG and PDF.
- `data/`: normalized CSV tables for every plotted numerical result.
- `manifest.json`: inventory, source paths and SHA256 hashes.
- `build_figures.py`: reproducible plotting script; needs matplotlib, pandas, numpy, scipy.

## What cannot be plotted honestly from this checkout
The source analysis script saves gates_D*_scale*.npz arrays containing per-trial spatial maps and signed implicit coefficients. They are not in the checkout, and neither are trained model weights. A spatial movie or per-trial heatmap therefore cannot be recovered here. The temporal heatmaps in this gallery display the stored region averages only.

## Interpretation limits
- 512 trials per psychometric point; saved 95% intervals are trial-bootstrap intervals. The bootstrap is degenerate at perfect accuracy; it does not quantify uncertainty about all future trials.
- One local seed for the probes and psychometrics; cloud comparison shows both original model seeds separately.
- Gaussian curve parameters are replayed as saved, not refitted. The saved slope is inverse spread, not a derivative. Fit-parameter uncertainty was not saved. Some fitted thresholds lie below the smallest tested magnitude.
- The all-ceiling delay curve cannot identify a forgetting time constant. The degenerate saved tau is not interpreted.
- KDA coefficients are signed and unnormalized before aggregation. Stored attention summaries average their absolute magnitudes across trials, regions and heads; they do not give attention probabilities or final-decision attribution.
- R2 probes are separate, same-time, unregularized least-squares fits with 64 fit / 64 held-out trials. Post-probe values may reflect sample/probe correlations; pre-probe values are identified explicitly. No cross-time decoding or angular error is inferred.
- Those probes have 257 features including the intercept and are underdetermined. No repeated-split or shuffle controls are present. The first two nominal blank frames still contain sample frames in the rolling three-frame input; they are not pure recurrent retention measurements.
- Reset interventions affect all KDA states, not the downstream GRU. Clamping does not prove blank writes are universally irrelevant.
- Beta=0 disables both writes and delta correction; alpha=1 removes diagonal decay, not all forgetting. Same-magnitude distractor mode makes all uncued rotations nonzero; the original pattern also uses a shared nonzero magnitude but permits unchanged Gabors.
- Cue jitter is a uniform integer offset per axis, sampled per cue-bearing frame, bounded by the stated value. Distraction varies uncued Gabors, not delay-inserted distractors.
- No significant improvement with delay, lossless memory, causal readout locus, or general memory capacity is claimed from these plots.
'''
(OUT/'README.md').write_text(notes)
sections=[]
for category in dict.fromkeys(f['category'] for f in manifest):
    cards=[]
    for f in [q for q in manifest if q['category']==category]:
        cards.append(f'''<article id="{f['id']}" data-title="{html.escape(f['title'].lower())}"><header><span class="num">{f['id'][:2]}</span><h3>{html.escape(f['title'])}</h3></header><a class="image" href="{f['svg']}" target="_blank"><img src="{f['svg']}" alt="{html.escape(f['title'])}" loading="lazy"></a><p>{html.escape(f['caption'])}</p><footer><a href="{f['svg']}" download>SVG</a><a href="{f['png']}" download>PNG</a><a href="{f['pdf']}" download>PDF</a><span>{html.escape(f['source'])}</span></footer></article>''')
    sections.append(f'<section id="{category.lower().replace(" ","-")}"><h2>{category}</h2>{"".join(cards)}</section>')
source_links=[]
for rel in SOURCES:
    source_links.append(f'<li><a href="../../../{html.escape(rel)}">{html.escape(rel)}</a></li>')
html_doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KDA — Results atlas</title><style>
:root{--ink:#192b3b;--muted:#526577;--line:#dce3e9;--accent:#176b9b}*{box-sizing:border-box}body{margin:0;background:#f8f9fa;color:var(--ink);font:17px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1250px;margin:auto;padding:48px 32px 100px}.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:700}h1{font:54px/1.1 Georgia,serif;margin:18px 0}h2{font:34px/1.2 Georgia,serif;margin:55px 0 25px}h3{font-size:20px;line-height:1.3;margin:0}p.lead{font-size:22px;max-width:850px}.note{border-left:3px solid #b17817;padding:12px 22px;background:#fff9ed;max-width:1000px;margin:26px 0}nav{display:flex;gap:8px 16px;flex-wrap:wrap;padding:20px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}a{color:var(--accent);text-underline-offset:3px}nav a,.button{padding:8px 0;min-height:44px}.actions{display:flex;gap:22px;flex-wrap:wrap;align-items:center}input{font:inherit;padding:12px 16px;border:1px solid #b6c2cc;border-radius:5px;margin-top:20px;width:min(100%,500px)}article{background:white;border:1px solid var(--line);margin-bottom:32px;border-radius:6px;overflow:hidden;scroll-margin-top:20px}article header{display:flex;gap:15px;align-items:center;padding:22px 28px;border-bottom:1px solid var(--line)}.num{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}.image{display:block;line-height:0}.image img{width:100%;height:auto;display:block}article p{max-width:1000px;padding:0 28px;margin:0 0 22px;font-size:16px}article footer{display:flex;flex-wrap:wrap;gap:20px;padding:16px 28px;background:#f3f6f8;font-size:14px;align-items:center}footer span{color:var(--muted);overflow-wrap:anywhere}details{margin-top:32px}summary{cursor:pointer;font-weight:700;padding:14px 0}li{overflow-wrap:anywhere;font-size:14px}.small{font-size:14px;color:var(--muted)}[hidden]{display:none!important}@media(max-width:700px){main{padding:28px 14px 70px}h1{font-size:40px}p.lead{font-size:20px}article header{padding:17px}article p{padding:0 17px}article footer{padding:14px 17px}.image{overflow-x:auto}.image img{min-width:850px}h3{font-size:18px}}@media print{nav,input,.actions,details{display:none}article{break-inside:avoid}main{padding:0}body{background:white}}
</style><main><div class="eyebrow">Visual attention & working memory · second pass</div><h1>KDA results atlas</h1><p class="lead">The completed orientation study, plotted from the saved records: learning, retention, cue robustness, spatial-memory gates and implicit attention over time.</p><div class="actions"><a class="button" href="KDA_results_atlas.pdf">Open complete PDF atlas</a><a class="button" href="README.md">Methods & limitations</a><a class="button" href="manifest.json">Source manifest</a></div><div class="note"><strong>Spatial-map gap.</strong> The raw per-trial map arrays and trained checkpoint were not committed. The attention figures below are genuine saved temporal profiles and region summaries—not reconstructed spatial heatmaps. No new experiment was run.</div><p class="small">FIGCOUNT figures · PNG / vector SVG / PDF for each · source commit COMMIT · one local seed for diagnostics; both original seeds for the cloud comparison.</p><nav><a href="#overview">Overview</a><a href="#training">Training</a><a href="#psychometrics">Psychometrics</a><a href="#memory-probes">Memory probes</a><a href="#attention">Attention</a></nav><label><input id="filter" type="search" placeholder="Filter figures by title…" aria-label="Filter figures by title"></label>SECTIONS<details><summary>Source records and downloadable data tables</summary><p class="small">Paths refer to the local repository. SHA256 hashes are in manifest.json.</p><ul>SOURCES</ul><h3>CSV exports</h3><ul>TABLES</ul></details></main><script>document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.title.includes(q));document.querySelectorAll('section').forEach(s=>s.hidden=![...s.querySelectorAll('article')].some(a=>!a.hidden));});</script></html>'''
html_doc=html_doc.replace('FIGCOUNT',str(len(manifest))).replace('COMMIT',commit[:7]).replace('SECTIONS',''.join(sections)).replace('SOURCES',''.join(source_links)).replace('TABLES',''.join(f'<li><a href="data/{p.name}">{p.name}</a></li>' for p in sorted(DATA.glob('*.csv'))))
(OUT/'index.html').write_text(html_doc)
print(json.dumps({'figures':len(manifest),'formats':['PNG','SVG','PDF'],'atlas':str(OUT/'KDA_results_atlas.pdf'),'gallery':str(OUT/'index.html'),'csv_tables':len(list(DATA.glob('*.csv'))),'source_commit':commit},indent=2))
