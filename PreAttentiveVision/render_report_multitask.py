"""Offline report for the authorized seven-task run. CPU document work only.

Reads results_multitask.json; never reads or relabels the stopped results.json.
No model imports, GPU execution, training changes or external publication.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
from render_report import CSS, NAMES, LABELS, COLORS, PARAMS, esc, finite, number, percent, interval, embedded, m

HERE=Path(__file__).resolve().parent
TASKS=['motion_direction','orientation','contrast','spatial_frequency','chromatic_increment','contour','natural_spectrum']
TITLES=['Motion direction','Signed orientation','Contrast','Spatial frequency','Chromatic increment','Contour grouping','Natural spectral detail']
SHORT=['Motion','Orientation','Contrast','Frequency','Chromatic','Contour','Natural spectrum']
TASK_LABELS=dict(zip(TASKS,TITLES))


def completed(data):
    return [r for r in data.get('runs',[]) if r.get('model') in NAMES
            and isinstance(r.get('test'),dict) and r['test'].get('status')=='completed'
            and all(finite(r['test'].get('tasks',{}).get(t,{}).get('overall',{}).get('balanced_accuracy')) for t in TASKS)]


def make_figures(data):
    curves=[r for r in data.get('runs',[]) if r.get('model') in NAMES and r.get('val_curve')]
    tested=completed(data)
    if not curves and not tested:return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'text.color':'#293145','axes.labelcolor':'#41485d','axes.edgecolor':'#c9ccd5','figure.facecolor':'#fffefa',
        'axes.facecolor':'#fffefa','savefig.facecolor':'#fffefa','svg.fonttype':'none'})
    out=HERE/'report_multitask_assets';out.mkdir(exist_ok=True)
    figures={}
    def save(fig,key):
        fig.savefig(out/f'{key}.png',dpi=180,bbox_inches='tight')
        fig.savefig(out/f'{key}.svg',bbox_inches='tight')
        figures[key]=embedded(out/f'{key}.png');plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        for r in curves:
            c=r['val_curve'];name=r['model'];label=f"{LABELS[name]} · {r.get('seed','?')}"
            axes[0].plot([v['step'] for v in c],[v['macro_ovr_auc'] for v in c],'-o',ms=4,lw=1.7,color=COLORS[name],label=label)
            axes[1].plot([v['step'] for v in c],[v['mean_normalized_balanced_accuracy'] for v in c],'-o',ms=4,lw=1.7,color=COLORS[name])
        axes[0].set(ylabel='Equal-task macro OVR-AUC',ylim=(.45,1.02),title='Validation score ranking')
        axes[0].axhline(.5,color='#b3b8c5',ls=':',lw=1)
        axes[1].set(ylabel='Mean chance-normalized balanced accuracy',ylim=(-.1,1.02),title='Validation decisions')
        axes[1].axhline(0,color='#b3b8c5',ls=':',lw=1)
        for ax in axes:ax.set_xlabel('Completed training updates');ax.grid(alpha=.13)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.08))
        fig.tight_layout();save(fig,'validation_curves')
    if tested:
        ba=np.array([[r['test']['tasks'][t]['overall']['balanced_accuracy'] for t in TASKS] for r in tested])
        chance=np.array([.25]+[.5]*6)
        normalized=(ba-chance)/(1-chance)
        fig,ax=plt.subplots(figsize=(11,max(3,len(tested)*.65+1.2)))
        im=ax.imshow(normalized,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set(xticks=range(7),xticklabels=[f'{t}\nchance {c:.0%}' for t,c in zip(SHORT,chance)],
               yticks=range(len(tested)),yticklabels=[f"{LABELS[r['model']]} · {r.get('seed','?')}" for r in tested])
        ax.tick_params(length=0,pad=10)
        for i in range(len(tested)):
            for j in range(7):ax.text(j,i,f'{ba[i,j]:.1%}',ha='center',va='center',fontsize=10,color='white' if normalized[i,j]>.65 else '#25334c')
        fig.colorbar(im,ax=ax,fraction=.025,pad=.03,label='Color: chance-normalized BA')
        fig.tight_layout();save(fig,'task_balanced_accuracy')
        fig,axes=plt.subplots(2,4,figsize=(12,7),sharex=True)
        for ti,t in enumerate(TASKS):
            ax=axes.flat[ti]
            for ri,r in enumerate(tested):
                c=r['test']['tasks'][t]['overall'];score=c['macro_ovr_auc'];ci=c.get('macro_ovr_auc_ci95')
                if isinstance(ci,list) and len(ci)==2:ax.plot(ci,[ri,ri],color=COLORS[r['model']],lw=2)
                ax.scatter(score,ri,s=28,color=COLORS[r['model']])
            ax.set(title=SHORT[ti],xlim=(.45,1.02),yticks=range(len(tested)),yticklabels=[LABELS[r['model']] for r in tested],xlabel='Test macro OVR-AUC')
            ax.invert_yaxis();ax.axvline(.5,color='#b9bdc7',ls=':',lw=1);ax.grid(axis='x',alpha=.12);ax.tick_params(labelsize=8)
        axes.flat[7].axis('off');axes.flat[7].text(0,.9,'One row per model–seed.\nBars: supplied 95% task intervals.\nChance AUC = 0.5 for every task.\nNo interval pools training seeds.',fontsize=10,va='top',linespacing=1.8)
        fig.tight_layout();save(fig,'task_auc_intervals')
    return figures


def table(headers,rows):
    return '<div class="table-scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{v}</{"th" if i==0 else "td"}>' for i,v in enumerate(row))+'</tr>' for row in rows)+'</tbody></table></div>'


def result_tables(data):
    tested=completed(data);rows=[]
    for t in TASKS:
        for name in NAMES:
            available=[r for r in tested if r['model']==name]
            if not available:available=[{'model':name}]
            for r in available:
                c=r.get('test',{}).get('tasks',{}).get(t,{}).get('overall',{})
                rows.append([TASK_LABELS[t],f"{LABELS[name]} · {esc(r.get('seed','pending'))}", '25%' if t=='motion_direction' else '50%',
                    percent(c.get('balanced_accuracy'))+f'<small>{interval(c.get("balanced_accuracy_ci95"),True)}</small>',
                    number(c.get('macro_ovr_auc'),4)+f'<small>{interval(c.get("macro_ovr_auc_ci95"))}</small>',str(c.get('n','Pending'))])
    return table(['Task','Model · seed','BA chance','Balanced accuracy [95%]','Macro OVR-AUC [95%]','Test pairs'],rows)


def detail_tables(data):
    rows=[];confusions=''
    for r in completed(data):
        for t in TASKS:
            item=r['test']['tasks'][t];overall=item['overall']
            for d,c in item.get('difficulty',{}).items():
                rows.append([LABELS[r['model']],TASK_LABELS[t],esc(d),str(c.get('n','—')),percent(c.get('balanced_accuracy')),number(c.get('macro_ovr_auc'),4)])
            recalls='; '.join(f'{i}: {percent(v)} [{interval(ci,True)}]' for i,(v,ci) in enumerate(zip(overall.get('recall',[]),overall.get('recall_ci95',[]))))
            matrix=overall.get('confusion_true_rows_pred_columns',[])
            confusions+=f'<details><summary>{LABELS[r["model"]]} · {TASK_LABELS[t]} · confusion & recall</summary><p class="small">Rows are true classes; columns predicted classes. Fixed argmax. Class order: {"right / up / left / down" if t=="motion_direction" else "0 / 1, defined in the task table"}.</p>'+table(['True / predicted']+[str(i) for i in range(len(matrix))],[[str(i)]+[str(n) for n in row] for i,row in enumerate(matrix)])+f'<p class="caption">Per-class recall and Wilson95% interval: {recalls}. Unique evaluation groups: {overall.get("unique_base_groups","not supplied")}. Source photos are clustered; synthetic examples use pair identities.</p></details>'
    difficulty=table(['Encoder','Task','Difficulty','N','BA','OVR-AUC'],rows) if rows else '<p>Difficulty measurements are pending completed tests.</p>'
    return '<details><summary>Per-task difficulty estimates</summary>'+difficulty+'<p class="caption">Difficulty cells are point estimates. The evaluator supplies bootstrap intervals for overall task cells, not these smaller strata.</p></details>'+('<details><summary>Class confusion matrices and recall uncertainty</summary>'+confusions+'</details>' if confusions else '')


def outcome(data):
    valid=completed(data)
    if data.get('status')!='completed' or len({r['model'] for r in valid})<5:
        return f'<h3>Test comparison pending.</h3><p>{len(valid)} of five model evaluations are complete in this saved snapshot. Validation curves are development measurements; they are not held-out test results. No performance winner is assigned while the comparison is incomplete.</p>'
    selected=max(valid,key=lambda r:r.get('best_auc',float('-inf')))
    return f'<h3>{LABELS[selected["model"]]} has the highest saved validation selection score.</h3><p>This is a descriptive within-run selection: equal-task validation macro OVR-AUC {number(selected.get("best_auc"),4)}, checkpoint {selected.get("best_step","—")}. All five held-out test results are shown by task. A single average cannot hide a weak motion, contour or chromatic result; separate confidence intervals do not establish a significant model difference.</p><p class="small">Only one training seed was run. The recommendation remains scoped to this training recipe and exposure; neither acquisition asymptotes nor neural correspondence are established.</p>'


def render(path,output):
    data=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {'status':'pending','config':{},'profiles':{},'runs':[]}
    cfg=data.get('config',{});figs=make_figures(data);now=datetime.now(timezone.utc).strftime('%d %b %Y · %H:%M UTC')
    status=esc(data.get('status','pending'));tested=completed(data)
    rows=[]
    for name in NAMES:
        p=data.get('profiles',{}).get(name,{});r=next((r for r in data.get('runs',[]) if r['model']==name),{})
        sec=p.get('train_step_median');batch=cfg.get('batch_size');speed=batch/sec if finite(batch) and finite(sec) and sec>0 else None
        rows.append([f'<i style="background:{COLORS[name]}"></i>{LABELS[name]}',f'{p.get("encoder_params",PARAMS[name]):,}',esc(p.get('decoder_params','Pending')),esc(p.get('total_params','Pending')),number(speed,1),f'{r.get("step",0)} / {cfg.get("updates","pending")}',esc(r.get('best_step','Pending')),esc(r.get('status','pending'))])
    summary=table(['Candidate','Encoder params','Decoder params','Total params','Profile pairs/s','Updates','Selected update','Saved status'],rows)
    task_rows=[
        ['Motion direction','4 · 25%','0 right · 1 up · 2 left · 3 down','1 / 2 / 3 pixel displacement; 128 survivors + 128 reborn domain dots','<a href="cardinal_motion.md">Lovejoy & Krauzlis adaptation</a>'],
        ['Signed orientation','2 · 50%','0 second counterclockwise · 1 clockwise','4° / 10° / 22°; roving base angle, independent phase','<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3842491/">Zhang et al.2010</a>'],
        ['Contrast','2 · 50%','Index of the higher-amplitude frame','Δc .025 / .06 / .13; pedestal .08 / .18 / .30','<a href="https://opg.optica.org/josa/abstract.cfm?uri=josa-70-12-1458">Legge & Foley1980</a>'],
        ['Spatial frequency','2 · 50%','Index of the higher-frequency frame','.08 / .18 / .35 octaves; lower f4–9 cycles/image','<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3325184/">Two-interval frequency discrimination</a>'],
        ['Chromatic increment','2 · 50%','Index of the higher chromatic-axis coordinate','Δ .018 / .045 / .10; fixed numerical luminance-null axis','<a href="https://www.allpsych.uni-giessen.de/karl/pdf/01.coldisc.pdf">Krauskopf & Gegenfurtner1992</a>'],
        ['Contour grouping','2 · 50%','Index of the generated aligned-contour frame','2° / 8° / 16° tangent jitter; 7 path elements of32','<a href="https://doi.org/10.1016/0042-6989(93)90156-Q">Field, Hayes & Hess1993</a>'],
        ['Natural spectral detail','2 · 50%','Index of smaller β / greater high-frequency weight','Δβ .15 / .30 / .60; native100-pixel BSDS500 crops','<a href="https://doi.org/10.1016/0042-6989(94)90167-8">Tadmor & Tolhurst1994</a>']]
    tasktable=table(['Task','Classes · chance','Ordered label','Declared signal levels','Primary motivation'],task_rows)
    figure_html=''
    for key,title,caption in [('validation_curves','Learning across seven tasks','Actual saved validation checkpoints. Each task has equal weight in the aggregate. Curves remain development evidence used for checkpoint selection.'),('task_balanced_accuracy','Decisions, task by task','Cell text is raw balanced accuracy. Color subtracts each task’s own chance level and divides by its remaining headroom; 0 is chance and1 perfect. This is not pooled raw accuracy.'),('task_auc_intervals','Ranking with evaluation uncertainty','One-versus-rest AUROC is averaged across classes within each task. All tasks have chance AUC0.5. Intervals are supplied overall-task1000-resample bootstrap estimates, conditional on the trained checkpoint.')]:
        if key in figs:figure_html+=f'<figure class="plot"><h3>{title}</h3><img src="{figs[key]}" alt="{title}; real seven-task experiment measurements"><figcaption>{caption} <a href="report_multitask_assets/{key}.svg">SVG</a> · <a href="report_multitask_assets/{key}.png">PNG</a></figcaption></figure>'
    if not figure_html:figure_html='<div class="pending-chart"><span>Measured curves will appear here.</span><p>No scores are fabricated or borrowed from the stopped displacement benchmark.</p></div>'
    contact=embedded(HERE/'sensory_battery_examples.png')
    ordered=m('<msub><mi>I</mi><mi>s</mi></msub><mo>=</mo><mo>[</mo><mi>B</mi><mo>−</mo><mi>A</mi><mo>;</mo><mo>|</mo><mi>B</mi><mo>−</mo><mi>A</mi><mo>|</mo><mo>;</mo><mfrac><mrow><mi>A</mi><mo>+</mo><mi>B</mi></mrow><mn>2</mn></mfrac><mo>;</mo><mi>A</mi><mo>⊙</mo><mi>B</mi><mo>;</mo><mi>C</mi><mo>]</mo>')
    correlation=m('<msub><mi>C</mi><mrow><mi>δ</mi></mrow></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>=</mo><munder><mo>∑</mo><mi>c</mi></munder><msub><mover><mi>A</mi><mo>^</mo></mover><mi>c</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><msub><mover><mi>B</mi><mo>^</mo></mover><mi>c</mi></msub><mo>(</mo><mi>p</mi><mo>+</mo><mi>δ</mi><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>δ</mi><mo>∈</mo><msup><mrow><mo>{</mo><mo>−</mo><mn>2</mn><mo>,</mo><mo>…</mo><mo>,</mo><mn>2</mn><mo>}</mo></mrow><mn>2</mn></msup>')
    orientation=m('<mi>Δ</mi><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mtext>atan2</mtext><mo>(</mo><mo>sin</mo><mo>[</mo><mn>2</mn><mo>(</mo><msub><mi>θ</mi><mn>2</mn></msub><mo>−</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>)</mo><mo>]</mo><mo>,</mo><mo>cos</mo><mo>[</mo><mn>2</mn><mo>(</mo><msub><mi>θ</mi><mn>2</mn></msub><mo>−</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>)</mo><mo>]</mo><mo>)</mo>')
    spectrum=m('<msub><mi>F</mi><mi>β</mi></msub><mo>(</mo><mi>f</mi><mo>)</mo><mo>=</mo><mi>ℱ</mi><mo>[</mo><mi>J</mi><mo>]</mo><mo>(</mo><mi>f</mi><mo>)</mo><msup><mrow><mo>(</mo><mfrac><mrow><mo>‖</mo><mi>f</mi><mo>‖</mo></mrow><msub><mi>f</mi><mtext>ref</mtext></msub></mfrac><mo>)</mo></mrow><mrow><mo>−</mo><mi>β</mi></mrow></msup><mo>,</mo><mspace width="1em"/><mi>f</mi><mo>≠</mo><mn>0</mn>')
    ba_math=m('<msub><mtext>BA</mtext><mi>t</mi></msub><mo>=</mo><mfrac><mn>1</mn><msub><mi>K</mi><mi>t</mi></msub></mfrac><munderover><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>K</mi><mi>t</mi></msub></munderover><mfrac><msub><mi>N</mi><mrow><mi>k</mi><mi>k</mi></mrow></msub><mrow><munder><mo>∑</mo><mi>j</mi></munder><msub><mi>N</mi><mrow><mi>k</mi><mi>j</mi></mrow></msub></mrow></mfrac><mo>,</mo><mspace width="1em"/><msub><mi>q</mi><mi>t</mi></msub><mo>=</mo><mfrac><mrow><msub><mtext>BA</mtext><mi>t</mi></msub><mo>−</mo><mn>1</mn><mo>/</mo><msub><mi>K</mi><mi>t</mi></msub></mrow><mrow><mn>1</mn><mo>−</mo><mn>1</mn><mo>/</mo><msub><mi>K</mi><mi>t</mi></msub></mrow></mfrac>')
    methods=''
    profiles=[('aa_resnet','Residual local filtering with anti-aliasing','Binomial filtering before decimation suppresses high-frequency aliasing, but could also attenuate useful displacement/detail.','https://proceedings.mlr.press/v97/zhang19a.html'),('convnext_grn','Spatial mixing and response normalization','Depthwise7 × 7 convolution plus pointwise expansion and global response normalization preserves maps while modulating channels. GRN is not a fitted biophysical normalization model.','https://arxiv.org/abs/2301.00808'),('inceptionnext','Several spatial scales and shapes','Identity, square3 × 3, horizontal1 × 11 and vertical11 × 1 depthwise branches add heterogeneous receptive fields.','https://arxiv.org/abs/2303.16900'),('mobilenet_se','Expanded depthwise filtering and channel gain','Inverted bottlenecks and squeeze-excitation use image-dependent channel modulation. This is not the later visual attention component.','https://arxiv.org/abs/1905.02244'),('vone_resnet','Explicit oriented simple/complex features','Fixed chromatic Gabors, rectification and quadrature energy feed learned anti-aliased residual layers. Deterministic filter grids replace the original model’s biological distributions and neuronal stochasticity.','https://papers.neurips.cc/paper_files/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html')]
    for name,title,desc,url in profiles:methods+=f'<article class="model" style="--model:{COLORS[name]}"><div class="model-top"><span>{esc(name)}</span><b>{PARAMS[name]:,} encoder parameters</b></div><h3>{title}</h3><p>{desc}</p><p class="source"><a href="{url}">Primary model source</a> · <a href="research.md">Exact adaptations & equations</a> · <a href="models.py">Local code</a></p></article>'
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Seven sensory questions · PAV experiment</title><style>{CSS}
.wrap,.hero-grid>*,.two>*,.grounding-panel>*,.section-head>*{{min-width:0}}.task-sheet{{max-width:780px;margin:30px auto;padding:20px;background:white;border:1px solid var(--line)}}.task-sheet img{{display:block;width:100%;height:auto}}.task-sheet figcaption{{font-size:11px}}.hero-grid{{grid-template-columns:1.3fr 1fr;gap:45px}}.hero h1{{font-size:clamp(60px,7.5vw,99px)}}.hero .dek{{font-size:21px}}.summary-count{{font:72px/1 var(--serif);color:var(--accent);margin:23px 0 12px}}.summary-count small{{font:19px var(--sans);color:var(--muted)}}@media(max-width:620px){{.hero-grid{{grid-template-columns:1fr}}.hero h1{{font-size:66px;letter-spacing:-3px}}.summary-count{{font-size:54px}}.task-sheet{{padding:8px}}}}
</style></head><body><a class="skip" href="#main">Skip to report</a><header><div class="wrap nav"><a class="brand" href="#">PAV / SEVEN-TASK SCREEN</a><nav><a href="#results">Results</a><a href="#tasks">Tasks</a><a href="#decoder">Decoder</a><a href="#meaning">Grounding</a></nav></div></header><main id="main"><section class="hero wrap"><div class="eyebrow">Visual attention & working memory · First component</div><div class="hero-grid"><div><h1>Seven sensory<br><em>questions.</em></h1><p class="dek">Two ordered frames.<br>Five candidate visual encoders.</p><p class="hero-copy">Motion, orientation, contrast, scale, color, grouping and natural-image detail provide distinct tests of an early visual representation.</p></div><div class="hero-right"><div class="status"><span class="dot"></span>{status}<small>Saved report · {now}</small></div><div class="summary-count">{len(tested)}<small> / 5 tests complete</small></div><p class="small muted">One training seed · fixed argmax outputs · per-task results. This report reads only <a href="results_multitask.json">the authorized multitask run</a>. The stopped arbitrary-displacement prototype supplies none of its scores.</p><div class="callout">The new dot task classifies cardinal population-motion direction. It does not ask whether motion direction changed.</div></div></div><div class="scope"><b>PAV is one component.</b><span>This experiment evaluates visual encoding and ordered comparison. Visual attention, visual working memory and integration remain later work. Guided Search6.0 is a functional scaffold, not a completed implementation.</span></div></section>
<section class="section wrap" id="results"><div class="section-head"><span class="index">01</span><div><div class="eyebrow">Current experiment · measured evidence</div><h2>Read the tasks separately.</h2><p class="intro">Motion has four outputs and25% chance balanced accuracy. Each other task has two outputs and50% chance. One pooled raw-accuracy headline would mix different baselines.</p></div></div><div class="finding">{outcome(data)}</div>{summary}<p class="caption">Profile pairs/s = batch32 divided by median measured training-step duration, including stimulus generation, transfer, forward/backward and optimizer update; it excludes startup, checkpoints and evaluations. All candidates use the same ordered decoder topology, trained separately. Size and compute are not matched.</p><div class="facts"><div><strong>{cfg.get('updates','Pending')}</strong><span>target updates per model</span></div><div><strong>{cfg.get('training_pairs_per_model','Pending'):,}</strong><span>target training pairs per model</span></div><div><strong>{cfg.get('training_pairs_per_task_per_model','Pending'):,}</strong><span>target pairs per task per model</span></div><div><strong>50m58s</strong><span>remaining authorized compute allocation</span></div></div>{figure_html}<details open><summary>Held-out scores, task by task</summary>{result_tables(data)}</details>{detail_tables(data)}<div class="two" style="margin-top:30px"><div><h3>What the score means.</h3><div class="equation">{ba_math}</div><p class="small">N is the true-row/predicted-column confusion matrix; K is the task's class count. Balanced accuracy is mean class recall. The normalized score q has chance0 and perfect1; averaging q is descriptive and does not replace the task table.</p></div><div><h3>What the uncertainty covers.</h3><p>Overall task cells use1,000 bootstrap resamples of source-image clusters for natural crops and generated pairs for synthetic stimuli. Wilson intervals summarize each class's recall. These intervals condition on this trained checkpoint.</p><p class="small muted">Repeated crops are not independent source photographs. One seed does not estimate variation over training seeds. Fixed argmax is used everywhere; no test threshold calibration occurs. AUC averages class-wise one-versus-rest rankings, whose chance is0.5 for every task.</p></div></div></section>
<section class="section wrap" id="tasks"><div class="section-head"><span class="index">02</span><div><div class="eyebrow">Sourced paradigms · disclosed adaptations</div><h2>Two images can ask more than “changed?”</h2><p class="intro">The labels are sensory judgments. Task identity selects a readout head; latent parameters, contour membership, direction labels and future information never enter the encoder or decoder as features.</p></div></div>{tasktable}<figure class="task-sheet"><img src="{contact}" alt="Actual seven-task contact sheet: two ordered frame pairs per task, labeled by their target judgment"><figcaption>Actual generated examples from <a href="sensory_battery_examples.json">the current battery</a>, seed2026091203. The displayed first/second labels are1-based for reading; output indices in the task table are0-based. These are inputs, not trained predictions.</figcaption></figure><div class="two"><div><h3>Population motion, with finite lifetimes.</h3><p>Lovejoy & Krauzlis2010 used stochastic-motion patches, two-refresh dot lifetimes and coherent motion pulses. Here exactly one transition is observed in a fixed85-pixel circular aperture. Of256 domain dots,128 persist and128 are independently reborn. All survivors move in the selected cardinal direction.</p><p class="small">The source-derived2-pixel displacement anchor is accompanied by explicit1- and3-pixel raster variations. Coherence1 refers to survivors; rebirth still creates correspondence noise. A periodic underlying domain and a fixed aperture preserve class-independent single-frame spatial distributions. No speed in degrees/second, attention cue or full original trial is claimed.</p><p class="source"><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3412590/">Lovejoy & Krauzlis2010</a> · <a href="cardinal_motion.md">Exact raster adaptation</a></p></div><div><h3>Native natural detail, not upsampled thumbnails.</h3><p>The natural task uses100 × 100 crops from BSDS500 photographs:200 training sources,100 validation sources and200 test sources. Original source identities do not cross those splits. Fresh crop/transform pairs can revisit the same source photograph.</p><div class="equation">{spectrum}</div><p class="small">Fourier phase is preserved. Smaller β raises relative high-frequency weighting. Each version is zero-centered and standardized; one common RMS level, min(0.15,0.45/max|z|), keeps both in gamut around mean0.5 without clipping. Finite-crop periodic FFT boundaries are a disclosed limitation.</p><p class="source"><a href="natural_stimuli.py">Implemented spectrum/source contract</a> · <a href="natural_image_task.md">Psychophysical rationale</a> · <a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html">Official BSDS source</a></p></div></div><details><summary>Orientation, contrast, color and contour: mathematical qualifications</summary><div class="equation">{orientation}</div><p>Orientation is axial: angles separated by180° describe the same orientation. The signed local difference uses doubled angles, avoiding wrap errors. Positive image-coordinate rotation is clockwise because y increases downwards. Independent carrier phases stop phase correspondence from substituting for orientation.</p><p>The grating code's contrast parameter c is signed intensity-modulation amplitude around mean0.5, not calibrated Michelson contrast; symmetric extrema would yield Michelson contrast2c. Discrete mean subtraction and shared-pattern normalization are applied before the desired amplitude difference, with identical small independent sensor noise.</p><p>The chromatic unit axis is proportional to(0.7152,−0.2126,0) in declared linear RGB, orthogonal to numerical luminance weights(0.2126,0.7152,0.0722). This is a defined red–green-like numerical comparison, not calibrated human isoluminance or a cone-isolating experiment.</p><p>Contour comparisons retain the same32 positions and orientation multiset, permuting orientation assignments between elements. A seven-element smooth path provides the generating target. Random controls may retain partial accidental contours; errors alone do not establish absent grouping. Exact raster choices and signal levels: <a href="sensory_battery_parameters.md">implemented task specification</a>.</p></details></section>
<section class="section wrap" id="decoder"><div class="section-head"><span class="index">03</span><div><div class="eyebrow">The comparison retains time order</div><h2>Direction needs a signed relationship.</h2></div></div><div class="two"><div><h3>Same encoder weights, separate frame evidence.</h3><p>Each frame passes through the shared encoder to produce24 × 50 × 50,48 × 25 × 25 and96 × 13 × 13 feature fields. Inputs receive the same fixed normalization(x−0.5)/0.5. No recurrent state, clean previous-frame cache or process noise is introduced.</p><p>A shared per-scale projection forms32-channel A and B. A swap-symmetric difference/mean/product representation cannot retain the sign of motion. The new common decoder includes signed B−A and ordered local matches.</p><div class="equation">{ordered}</div></div><div><h3>Learned features, ordered correspondence.</h3><div class="equation">{correlation}</div><p class="small">Â and B̂ are channel-normalized projected fields, with ε10⁻⁶. The25 correlation channels match first-frame position p to second-frame p+δ, with dy-major offset order and zero padding. They are learned-feature similarities, not privileged direction targets.</p><p class="small">Local3 × 3 convolutions mix153 channels per scale into32. Pooling to13 × 13 precedes64-channel fusion, global mean/max and a128-unit shared trunk. Separate linear task heads produce four motion logits or two logits for each other task. Source: <a href="decoder_multitask.py">ordered decoder</a>.</p></div></div><div class="loss"><div><div class="eyebrow">Equal exposure after measured throughput</div><p>One32-pair task batch per update, in seven-task round robin. AdamW uses lr0.001, weight decay0.0001, gradient clipping5, fp32 and no AMP. The batch mean cross-entropy trains the selected task output; equal task update counts balance exposure, not necessarily task difficulty or gradient size.</p></div><div><p>Targets252,504,756 are multiples of seven. All candidates receive the same stream seed and corresponding draws. The fixed budget is756 updates /24,192 pairs per model,3,456 pairs per task. These are fresh procedural/crop presentations—not24,192 unique natural photographs.</p></div></div><p class="caption">Validation:224 pairs per task, reused at three planned looks. Test:448 new pairs per task (3,136 per model), shared across candidates. Checkpoint selection uses highest equal-task validation macro OVR-AUC, with earliest-checkpoint exact ties. The remaining3057.7762-second allowance includes profiling/evaluation; it is not renewed and is not a sufficient acquisition horizon. <a href="results_multitask.json">Frozen config and actual records</a> · <a href="sweep_multitask.py">Sequential experiment supervisor</a></p></section>
<section class="section wrap" id="meaning"><div class="section-head"><span class="index">04</span><div><div class="eyebrow">Performance and neuroscience are separate</div><h2>Five computational biases.<br>No biological winner by accuracy alone.</h2></div></div><div class="model-grid">{methods}</div><div class="grounding-panel" style="margin-top:30px"><div><span class="pill">Architectural grounding</span><h3>VOne ResNet is the most explicitly V1-inspired candidate.</h3><p>Its oriented Gabor bank and simple/complex response construction are direct commitments to early-visual computations. The other candidates adopt useful engineering biases without this explicit front-end correspondence.</p><p>This implementation omits the original VOneNet's fitted physiological distributions and neuronal noise. No human/monkey neural recordings are fit or predicted here. “Most explicitly inspired” describes the construction, not measured neural validity.</p></div><div><span class="pill">Measured usefulness</span>{outcome(data)}<p class="small">Good scores can establish that an encoder plus ordered decoder supports these sensory judgments. They do not establish visual search guidance, attention selection, long-delay retention or semantic recognition. Specific task strengths and weaknesses matter more than a biologically suggestive model name.</p></div></div></section>
<section class="closing wrap"><div class="eyebrow">A first component, kept in scope</div><h2>Make the representation useful.<br>Keep later mechanisms open.</h2><p>Guided Search6.0 provides the project's functional scaffold. The diffuser is omitted. Activated long-term memory is approximated as synaptic weights by the user's modeling choice, not by a claim that Wolfe assigns it this implementation. Working memory, visual attention and integration follow later.</p><p class="sources"><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/">Wolfe · Guided Search6.0</a> · <a href="two_frame_task_research.md">Primary task literature</a> · <a href="research.md">Encoder rationale and exact equations</a> · <a href="README.md">Current PAV overview</a> · <a href="results_multitask.json">Raw current results</a></p><p class="small">This self-contained HTML embeds its figures and native MathML. Linked code/data and exported SVG/PNG files remain local companion artifacts. Its status is a dated snapshot, not an automatic monitor.</p></section></main><footer class="wrap"><span>Seven-task PAV experiment · {now}<br>Saved status: {status} · one seed20271 · no old benchmark results imported</span><button type="button" onclick="window.print()">Print / save PDF</button></footer></body></html>'''
    # Typography only; the source-derived numerical values and equations above remain unchanged.
    for a,b in {'and25%':'and 25%','and50%':'and 50%','chance0':'chance 0','perfect1':'perfect 1','uses1,000':'uses 1,000','is0.5':'is 0.5','Search6.0':'Search 6.0','seed202':'seed 202','are1-based':'are 1-based','are0-based':'are 0-based','fixed85':'fixed 85','Of256':'Of 256','dots,128':'dots, 128','and128':'and 128','derived2':'derived 2','explicit1-':'explicit 1-','and3-pixel':'and 3-pixel','Coherence1':'Coherence 1','uses100':'uses 100','photographs:200':'photographs: 200','sources,100':'sources, 100','and200':'and 200','mean0.5':'mean 0.5','by180°':'by 180°','contrast2c':'contrast 2c','same32':'same 32','produce24':'produce 24',',48 ×':', 48 ×','and96 ×':'and 96 ×','forms32':'forms 32','ε10⁻⁶':'ε 10⁻⁶','The25':'The 25','Local3':'Local 3','mix153':'mix 153','into32':'into 32','to13':'to 13','precedes64':'precedes 64','a128':'a 128','One32':'One 32','lr0.001':'lr 0.001','decay0.0001':'decay 0.0001','clipping5':'clipping 5','Targets252,504,756':'Targets 252, 504, 756','is756':'is 756','/24,192':'/ 24,192',',3,456':', 3,456','not24,192':'not 24,192','Validation:224':'Validation: 224','Test:448':'Test: 448','remaining3057':'remaining 3057','Depthwise7':'Depthwise 7','square3':'square 3','horizontal1':'horizontal 1','vertical11':'vertical 11','f4–9':'f 4–9','of32':'of 32','native100':'native 100','task1000':'task 1,000','AUC0.5':'AUC 0.5','and1 perfect':'and 1 perfect','Krauzlis2010':'Krauzlis 2010','et al.2010':'et al. 2010','Foley1980':'Foley 1980','Gegenfurtner1992':'Gegenfurtner 1992','Hess1993':'Hess 1993','Tolhurst1994':'Tolhurst 1994'}.items():html=html.replace(a,b)
    output.write_text(html,encoding='utf-8')
    return dict(output=str(output),status=data.get('status'),completed_tests=len(tested),mathml=html.count('<math '),figures=list(figs),bytes=output.stat().st_size)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=HERE/'results_multitask.json')
    ap.add_argument('--output',type=Path,default=HERE/'report_multitask.html')
    args=ap.parse_args();print(json.dumps(render(args.results,args.output),indent=2))
