"""Render the fresh PAV screen as an offline scientific report (CPU/document only).

Consumes results.json or --results; never imports a model or invokes a trainer.
Missing measurements remain pending. PNG/SVG charts are exported for reuse and
embedded into report.html along with the supplied real stimulus contact sheet.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ['aa_resnet', 'convnext_grn', 'inceptionnext', 'mobilenet_se', 'vone_resnet']
LABELS = dict(zip(NAMES, ['AA ResNet', 'ConvNeXt + GRN', 'InceptionNeXt', 'MobileNet + SE', 'VOne ResNet']))
COLORS = dict(zip(NAMES, ['#235c75', '#886dba', '#d17748', '#458b70', '#a44860']))
FAMILIES = ['gabors', 'dots', 'shapes', 'natural']
DISPLAY_FAMILIES = ['Gabors', 'Dot displacement', 'Colored shapes', 'Natural-image edits']
PARAMS = dict(zip(NAMES, [479160, 262032, 245574, 329730, 489528]))
# This specific comparison was halted after the user rejected arbitrary dot
# displacement. Keep its output diagnostic even if its last worker JSON is stale.
# A corrected experiment gets its own run identity and must define its protocol.
SUPERSEDED_RUNS = {'benchmark_20260912_135139'}


def esc(value):
    return html.escape(str(value))


def number(value, precision=3):
    return f'{float(value):.{precision}f}' if finite(value) else 'Pending'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def percent(value):
    return f'{100*value:.1f}%' if finite(value) else 'Pending'


def interval(value, pct=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(finite(v) for v in value):
        return 'not supplied'
    return '–'.join(f'{v*(100 if pct else 1):.{1 if pct else 3}f}' for v in value)+('%' if pct else '')


def embedded(path):
    if not path.exists():
        return ''
    mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/png'
    return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()


def m(content, display=True):
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{"block" if display else "inline"}">{content}</math>'


def load_results(path):
    if not path.exists():
        return {'status': 'pending', 'config': {}, 'profiles': {}, 'runs': []}
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Results must be a JSON object')
    return value


def measured_runs(data):
    if data.get('_report_halted'):
        return []
    return [r for r in data.get('runs', []) if isinstance(r, dict)
            and r.get('model') in NAMES
            and isinstance(r.get('test'), dict)
            and finite(r['test'].get('macro', {}).get('auroc'))]


def val_score(run):
    curve = run.get('val_curve', [])
    best = next((v for v in curve if v.get('step') == run.get('best_step')), None)
    if best and finite(best.get('macro_auc')):
        return best['macro_auc']
    scores = [v['macro_auc'] for v in curve if finite(v.get('macro_auc'))]
    return max(scores) if scores else None


def plots(data, assets):
    runs = measured_runs(data)
    curves=[r for r in data.get('runs',[]) if isinstance(r,dict) and r.get('model') in NAMES and r.get('val_curve')]
    if not runs and not curves:
        return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#3d465b', 'text.color': '#293145',
        'axes.edgecolor': '#c7cbd2', 'xtick.color': '#606779',
        'ytick.color': '#606779', 'svg.fonttype': 'none',
        'savefig.facecolor': '#fffefa', 'axes.facecolor': '#fffefa',
        'figure.facecolor': '#fffefa'})
    assets.mkdir(exist_ok=True)
    made = {}

    def save(fig, name):
        fig.savefig(assets/f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(assets/f'{name}.svg', bbox_inches='tight')
        made[name] = embedded(assets/f'{name}.png')
        plt.close(fig)

    if curves:
        fig,ax=plt.subplots(figsize=(9.5,4.0))
        seeds=sorted({r.get('seed',0) for r in curves},key=str)
        for run in curves:
            curve=[v for v in run['val_curve'] if finite(v.get('step')) and finite(v.get('macro_auc'))]
            if not curve:continue
            style=['-','--',':','-.'][seeds.index(run.get('seed',0))%4]
            ax.plot([v['step'] for v in curve],[v['macro_auc'] for v in curve],marker='o',ms=3,lw=1.8,ls=style,color=COLORS[run['model']],label=f"{LABELS[run['model']]} · {run.get('seed','?')}")
        ax.axhline(.5,lw=1,ls=':',color='#aab1bc')
        ax.set(xlabel='Completed training updates',ylabel='Validation macro AUROC',ylim=(.45,1.02))
        if data.get('_report_halted'):
            ax.set_title('OBSOLETE PROTOCOL · validation diagnostics only',fontsize=11,pad=13)
        ax.grid(alpha=.15)
        ax.legend(ncol=2,fontsize=8,loc='lower right',framealpha=.9)
        fig.tight_layout()
        save(fig,'learning_curves')
    if not runs:
        return made

    fig, ax = plt.subplots(figsize=(9.5, max(3.2, len(runs)*.42+1.1)))
    labels = []
    for index, run in enumerate(runs):
        stats = run['test']['macro']
        score = stats['auroc']
        ci = stats.get('auroc_ci95')
        if isinstance(ci, list) and len(ci) == 2 and all(finite(v) for v in ci):
            ax.plot(ci, [index,index], color=COLORS[run['model']], lw=2)
        ax.scatter(score,index,s=50,color=COLORS[run['model']],zorder=3)
        labels.append(f"{LABELS[run['model']]} · seed {run.get('seed','?')}")
    ax.axvline(.5,color='#aab1bc',lw=1,ls='--')
    ax.set(yticks=range(len(runs)),yticklabels=labels,xlim=(.45,1.015),xlabel='Test macro AUROC · supplied 95% intervals')
    ax.invert_yaxis()
    ax.grid(axis='x',alpha=.16)
    fig.tight_layout()
    save(fig,'test_auroc')

    family_values=[]
    for run in runs:
        family_values.append([run['test'].get('families',{}).get(f,{}).get('overall',{}).get('auroc',float('nan')) for f in FAMILIES])
    fig,ax=plt.subplots(figsize=(9.5,max(3.3,len(runs)*.45+1.1)))
    values=np.asarray(family_values,dtype=float)
    im=ax.imshow(values,aspect='auto',vmin=.5,vmax=1,cmap='YlGnBu')
    ax.set(xticks=range(4),xticklabels=DISPLAY_FAMILIES,yticks=range(len(runs)),yticklabels=labels)
    ax.tick_params(axis='both',length=0,pad=9)
    for i in range(len(runs)):
        for j in range(4):
            if math.isfinite(values[i,j]):
                ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]>.83 else '#253d54',fontsize=11)
    fig.colorbar(im,ax=ax,fraction=.027,pad=.03,label='Test AUROC')
    fig.tight_layout()
    save(fig,'family_auroc')

    return made


def result_table(data):
    runs=measured_runs(data)
    rows=[]
    for name in NAMES:
        found=[r for r in runs if r['model']==name]
        profile=data.get('profiles',{}).get(name,{})
        params=profile.get('encoder_params',PARAMS[name])
        batch=data.get('config',{}).get('batch_size')
        seconds=profile.get('train_step_median')
        speed=batch/seconds if finite(batch) and finite(seconds) and seconds>0 else None
        if not found:found=[{'model':name}]
        for run in found:
            macro=run.get('test',{}).get('macro',{})
            rows.append(f'<tr><th scope="row"><i style="background:{COLORS[name]}"></i>{LABELS[name]}<small>{esc(name)}</small></th><td>{esc(run.get("seed","—"))}</td><td>{number(macro.get("auroc"))}<small>{interval(macro.get("auroc_ci95"))}</small></td><td>{percent(macro.get("balanced_accuracy"))}<small>{interval(macro.get("balanced_accuracy_ci95"),True)}</small></td><td>{int(params):,}</td><td>{number(speed,1)}</td><td>{esc(run.get("best_step","Pending"))}</td></tr>')
    return '<div class="table-scroll"><table><thead><tr><th>Encoder</th><th>Seed</th><th>Test macro AUROC<small>95% interval</small></th><th>Balanced accuracy<small>95% interval</small></th><th>Encoder params</th><th>Profile pairs/s</th><th>Selected update</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def family_details(data):
    rows=[]
    for run in measured_runs(data):
        for family in FAMILIES:
            cell=run['test'].get('families',{}).get(family,{})
            strata=[('all',cell.get('overall',{}))]+list(cell.get('difficulty',{}).items())
            for difficulty,c in strata:
                if not c:continue
                counts=' / '.join(str(c.get(k,'—')) for k in ['hits','misses','false_alarms','correct_rejections'])
                rows.append(f'<tr><th>{LABELS[run["model"]]} · {esc(run.get("seed","?"))}</th><td>{esc(family)}</td><td>{esc(difficulty)}</td><td>{esc(c.get("n","—"))}</td><td>{number(c.get("auroc"))}<small>{interval(c.get("auroc_ci95"))}</small></td><td>{percent(c.get("balanced_accuracy"))}</td><td>{counts}</td><td>{number(c.get("dprime"),2)} / {number(c.get("criterion"),2)}</td></tr>')
    if not rows:return '<p class="muted">No completed held-out family measurements are available yet.</p>'
    return '<div class="table-scroll"><table><thead><tr><th>Model · seed</th><th>Family</th><th>Difficulty</th><th>N pairs</th><th>AUROC [95% CI]</th><th>BA</th><th>H / M / FA / CR</th><th>d′ / c</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def execution_details(data):
    rows=[]
    for name in NAMES:
        profile=data.get('profiles',{}).get(name,{})
        for run in [r for r in data.get('runs',[]) if r.get('model')==name] or [{'model':name}]:
            test=run.get('test') or {}
            peak=profile.get('peak_vram_bytes')
            rows.append(f'<tr><th>{LABELS[name]} · {esc(run.get("seed","—"))}</th><td>{esc(run.get("status","pending"))}</td><td>{esc(run.get("step","—"))}</td><td>{esc(run.get("train_pairs","—"))}</td><td>{esc(profile.get("total_params","—"))}</td><td>{number(peak/1e9 if finite(peak) else None,2)}</td><td>{number(test.get("threshold"),4)}</td><td>{percent(test.get("default_threshold",{}).get("macro",{}).get("balanced_accuracy"))}</td><td>{"Yes" if test.get("engineering_goal_met") is True else "No" if test.get("engineering_goal_met") is False else "Pending"}</td></tr>')
    return '<details><summary>Execution snapshot, total size and threshold behavior</summary><div class="table-scroll"><table><thead><tr><th>Model · seed</th><th>Saved status</th><th>Updates</th><th>Training pairs</th><th>Total trainable params</th><th>Profile GPU GB</th><th>Locked threshold</th><th>Default0.5 macro BA</th><th>Test target met</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p class="caption">GPU memory is peak PyTorch-allocated device memory in the measured profile, expressed in decimal GB. It excludes desktop/other processes and is not total GPU utilization. Production update counts exclude the separate, accounted32-update throughput profiles. Execution statuses are snapshots saved when this report was generated.</p></details>'


def measured_summary(data):
    if data.get('_report_halted'):
        return '<h3>Halted for stimulus redesign.</h3><p>The user rejected arbitrary dot displacement and requested Krauzlis moving-dot stimuli. This comparison does not test that requested task. Its partial validation measurements are preserved only as diagnostics of the superseded prototype; they support no performance ranking or PAV recommendation for the corrected task.</p><p class="small">The temporal formulation is awaiting the user’s choice. No corrected protocol, new run or completed test comparison is claimed here.</p>'
    runs=measured_runs(data)
    if not runs:
        return '<h3>The comparison is underway.</h3><p>No completed test measurements have been supplied to this report. Scores, learning curves and performance recommendations will appear only after the researcher saves actual results.</p>'
    complete=data.get('status') in ('complete','completed')
    model_vals={}
    for name in NAMES:
        vals=[val_score(r) for r in runs if r['model']==name]
        vals=[v for v in vals if finite(v)]
        if vals:model_vals[name]=sum(vals)/len(vals)
    if len(model_vals)==5 and complete:
        selected=max(model_vals,key=model_vals.get)
        subset=[r for r in runs if r['model']==selected]
        test=sum(r['test']['macro']['auroc'] for r in subset)/len(subset)
        return f'<h3>{LABELS[selected]} leads the saved validation comparison.</h3><p>Its mean best-checkpoint validation macro AUROC is <strong>{model_vals[selected]:.3f}</strong>; corresponding mean test macro AUROC is <strong>{test:.3f}</strong> across {len(subset)} saved training seed(s). This descriptive ranking uses validation, not test-score selection. Separate model confidence intervals alone do not resolve close paired differences.</p><p class="small">A practical PAV recommendation must also weigh speed, complexity, family-specific weaknesses and replication. This ranking does not select later attention or working-memory implementations.</p>'
    return f'<h3>{len(runs)} completed model–seed evaluation(s) available.</h3><p>The saved experiment status is <strong>{esc(data.get("status","partial"))}</strong>. The comparison is incomplete; the visible rows describe measured runs, and unmeasured candidates remain pending. No final winner is assigned.</p>'


def build_report(results_path, output):
    data=load_results(results_path)
    run_name=str(data.get('run_root','')).replace('\\','/').rstrip('/').split('/')[-1]
    if run_name in SUPERSEDED_RUNS or data.get('status') in ('halted_for_stimulus_redesign','superseded_protocol'):
        data=dict(data,_report_halted=True)
    config=data.get('config',{})
    runs=measured_runs(data)
    chart_data=plots(data,HERE/'report_assets')
    generated=datetime.now(timezone.utc).strftime('%d %B %Y · %H:%M UTC')
    status=str(data.get('status','pending')).replace('_',' ')
    if data.get('_report_halted'):
        status='halted for stimulus redesign'
    image=embedded(HERE/'stimulus_examples.png')
    if image:
        examples=f'<img class="examples" src="{image}" alt="Actual stimulus contact sheet: change and no-change pairs for Gabors, dots, colored shapes and natural-image edits."><figcaption>Actual generator samples · medium difficulty · validation seed 2026091201. Each pair contains two separately rendered noisy inputs. The natural-image row has no-change on the left and change on the right. These examples illustrate the task; they are not predictions.</figcaption>'
    else:examples='<p>Stimulus contact sheet pending.</p>'
    figures=''
    for key,title,caption in [('test_auroc','Held-out discrimination','One point per trained model–seed, with the supplied 95% test interval. Dashed line: chance ranking. The interval concerns the saved evaluation sample, not the space of possible training seeds.'),('family_auroc','Different images, different demands','Family-specific test AUROC. Rows retain independent training-seed identities; no duplicated frames or checkpoints are treated as seed replication.'),('learning_curves','Acquisition across equal updates','Validation scores at recorded checkpoints. The same validation set supports model/checkpoint selection. These are development trajectories, not independent held-out replications.')]:
        if key in chart_data:figures+=f'<figure class="plot"><h3>{title}</h3><img src="{chart_data[key]}" alt="{title}: measured PAV comparison"><figcaption>{caption} <a href="report_assets/{key}.svg">SVG</a> · <a href="report_assets/{key}.png">PNG</a></figcaption></figure>'
    if not figures:figures='<div class="pending-chart"><span>Results will be plotted here</span><p>Only saved experimental measurements become curves or points. No illustrative performance values are substituted.</p></div>'
    profiles=data.get('profiles',{})
    decoder_counts=sorted({p.get('decoder_params') for p in profiles.values() if isinstance(p,dict) and finite(p.get('decoder_params'))})
    decoder_info=', '.join(f'{int(v):,}' for v in decoder_counts) if decoder_counts else 'pending profile'
    seeds=config.get('seeds',[])
    nseed=len(seeds) if isinstance(seeds,list) else seeds or 'Pending'
    updates=config.get('updates','Pending')
    pairs=config.get('training_pairs_per_model_seed','Pending')
    budget=config.get('budget_seconds',3600)
    evidence_link=results_path.name if results_path.parent==HERE else str(results_path.relative_to(HERE)).replace('\\','/') if HERE in results_path.parents else ''
    raw_link=f'<a href="{esc(evidence_link)}">Full result JSON</a>' if evidence_link and results_path.exists() else 'Result JSON not yet saved'

    pair_math=m('<msubsup><mi>F</mi><mi>s</mi><mrow><mo>(</mo><mi>j</mi><mo>)</mo></mrow></msubsup><mo>=</mo><msub><mi>E</mi><mrow><mi>θ</mi><mo>,</mo><mi>s</mi></mrow></msub><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>j</mi><mo>)</mo></mrow></msup><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>j</mi><mo>∈</mo><mo>{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>}</mo>')
    interaction_math=m('<msub><mi>I</mi><mi>s</mi></msub><mo>=</mo><mo>[</mo><mo>|</mo><msub><mi>A</mi><mi>s</mi></msub><mo>−</mo><msub><mi>B</mi><mi>s</mi></msub><mo>|</mo><mo>;</mo><mfrac><mrow><msub><mi>A</mi><mi>s</mi></msub><mo>+</mo><msub><mi>B</mi><mi>s</mi></msub></mrow><mn>2</mn></mfrac><mo>;</mo><msub><mi>A</mi><mi>s</mi></msub><mo>⊙</mo><msub><mi>B</mi><mi>s</mi></msub><mo>]</mo>')
    loss_math=m('<mi>ℒ</mi><mo>=</mo><mo>−</mo><mfrac><mn>1</mn><mi>B</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><mo>log</mo><msub><mi>p</mi><mrow><mi>θ</mi><mo>,</mo><mi>ψ</mi></mrow></msub><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>|</mo><msubsup><mi>x</mi><mi>i</mi><mn>1</mn></msubsup><mo>,</mo><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup><mo>)</mo>')
    blur_math=m('<mi>y</mi><mo>=</mo><msub><mo>↓</mo><mn>2</mn></msub><mo>(</mo><mi>h</mi><mo>∗</mo><mi>z</mi><mo>)</mo><mo>,</mo><mspace width=".8em"/><mi>h</mi><mo>=</mo><mfrac><mn>1</mn><mn>16</mn></mfrac><msup><mrow><mo>[</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>1</mn><mo>]</mo></mrow><mi>T</mi></msup><mo>[</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>1</mn><mo>]</mo>')
    grn_math=m('<msub><mi>g</mi><mi>c</mi></msub><mo>=</mo><msub><mrow><mo>‖</mo><msub><mi>X</mi><mi>c</mi></msub><mo>‖</mo></mrow><mn>2</mn></msub><mo>,</mo><mspace width=".5em"/><msub><mi>n</mi><mi>c</mi></msub><mo>=</mo><mfrac><msub><mi>g</mi><mi>c</mi></msub><mrow><mover><mi>g</mi><mo>¯</mo></mover><mo>+</mo><mi>ε</mi></mrow></mfrac><mo>,</mo><mspace width=".5em"/><mi>Y</mi><mo>=</mo><mi>X</mi><mo>+</mo><mi>γ</mi><mo>⊙</mo><mi>X</mi><mo>⊙</mo><mi>n</mi><mo>+</mo><mi>β</mi>')
    inception_math=m('<mi>M</mi><mo>(</mo><mi>X</mi><mo>)</mo><mo>=</mo><mo>[</mo><msub><mi>X</mi><mn>0</mn></msub><mo>;</mo><msub><mi>K</mi><mrow><mn>3</mn><mo>×</mo><mn>3</mn></mrow></msub><mo>∗</mo><msub><mi>X</mi><mn>1</mn></msub><mo>;</mo><msub><mi>K</mi><mrow><mn>1</mn><mo>×</mo><mn>11</mn></mrow></msub><mo>∗</mo><msub><mi>X</mi><mn>2</mn></msub><mo>;</mo><msub><mi>K</mi><mrow><mn>11</mn><mo>×</mo><mn>1</mn></mrow></msub><mo>∗</mo><msub><mi>X</mi><mn>3</mn></msub><mo>]</mo>')
    se_math=m('<msub><mi>Y</mi><mi>c</mi></msub><mo>=</mo><msub><mi>X</mi><mi>c</mi></msub><mo>·</mo><msub><mi>g</mi><mi>c</mi></msub><mo>(</mo><msub><mtext>mean</mtext><mrow><mi>h</mi><mo>,</mo><mi>w</mi></mrow></msub><mi>X</mi><mo>)</mo>')
    vone_math=m('<mi>S</mi><mo>=</mo><mtext>ReLU</mtext><mo>(</mo><mi>a</mi><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>C</mi><mo>=</mo><msqrt><msup><mi>a</mi><mn>2</mn></msup><mo>+</mo><msup><mi>b</mi><mn>2</mn></msup><mo>+</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></msqrt><mo>−</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>3</mn></mrow></msup>')

    models=[
      ('aa_resnet','01','Keep local detail. Reduce aliasing.','Residual 3 × 3 filters learn local structure. A fixed binomial low-pass precedes each downsampling step. This addresses aliasing, while potentially smoothing the very positional changes this task asks us to detect.',blur_math,'Residual optimization and signal processing, rather than a fitted early-visual circuit.','https://proceedings.mlr.press/v97/zhang19a.html','Zhang · 2019'),
      ('convnext_grn','02','Mix space, then normalize response.','Depthwise 7 × 7 convolution supplies spatial mixing; pointwise expansion mixes channels. Global response normalization compares each channel’s spatial L2 response with the channel mean, then modulates an intact field.',grn_math,'Relative response modulation is a computational analogy. GRN is not a biophysically validated normalization model.','https://arxiv.org/abs/2301.00808','Woo et al. · 2023'),
      ('inceptionnext','03','Several receptive fields in one block.','Identity, 3 × 3, 1 × 11 and 11 × 1 depthwise branches combine local and elongated spatial evidence. The channels then mix through an MLP and a residual path.',inception_math,'Heterogeneous receptive fields are loosely motivated; axis-specific branches and channel partitions are engineering choices.','https://arxiv.org/abs/2303.16900','Yu et al. · 2023'),
      ('mobilenet_se','04','Spend computation selectively.','Inverted bottlenecks expand channels, filter spatially with depthwise 5 × 5 kernels, then project back. Squeeze-excitation gates channels using image-dependent spatial means.',se_math,'Feedforward channel gain is not the future attention component or a GS6 priority map.','https://arxiv.org/abs/1905.02244','Howard et al. · 2019'),
      ('vone_resnet','05','Start with oriented visual primitives.','A fixed 9 × 9 Gabor bank provides simple-cell-like rectification and complex-cell-like quadrature energy. Twenty-four maps of each type plus raw RGB enter a learned anti-aliased residual hierarchy.',vone_math,'The most direct V1-inspired construction in this set, but only a deterministic subset of VOneNet’s biological commitments.','https://papers.neurips.cc/paper_files/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html','Dapello et al. · 2020')]
    cards=''
    for name,num,title,body,equation,neuro,url,cite in models:
        cards+=f'<article class="model" style="--model:{COLORS[name]}"><div class="model-top"><span>{num} / {esc(name)}</span><b>{PARAMS[name]:,} encoder parameters</b></div><h3>{title}</h3><p>{body}</p><div class="equation">{equation}</div><p class="ground"><strong>Grounding:</strong> {neuro}</p><p class="source"><a href="{url}">{cite}</a> · <a href="research.md">Adopted properties & departures</a> · <a href="models.py">Local implementation</a></p></article>'

    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="An offline scientific results report comparing five lightweight visual encoders on paired image change detection."><title>Before attention · PAV encoder comparison</title><style>{CSS}</style></head><body>
<a class="skip" href="#main">Skip to report</a><header><div class="wrap nav"><a class="brand" href="#">PAV / EXPERIMENT 01</a><nav aria-label="Report sections"><a href="#results">Results</a><a href="#task">Task</a><a href="#models">Encoders</a><a href="#grounding">Grounding</a></nav></div></header>
<main id="main"><section class="hero wrap"><div class="eyebrow">Visual attention & working memory · Component 1</div><div class="hero-grid"><div><h1>Before<br><em>attention.</em></h1><p class="dek">Five lightweight encoders.<br>Two images. One controlled comparison.</p><p class="hero-copy">Which spatial representation best preserves the visual evidence needed to tell whether a scene changed?</p></div><div class="hero-right"><div class="status"><span class="dot"></span> {esc(status)}<small>Report generated {generated}</small></div><div class="pair-art" role="img" aria-label="Two image frames pass through one shared encoder and meet at a comparison decoder"><div class="art-images"><div class="art-frame"><i></i><i></i><i></i><i></i></div><div class="art-frame alt"><i></i><i></i><i></i><i></i></div></div><div class="art-connect">↘ <span>shared weights</span> ↙</div><div class="art-encoder">spatial feature pyramid</div><div class="art-down">↓</div><div class="art-decoder">learned symmetric comparison</div></div><p class="small muted">Architecture schematic. Each image is encoded separately; comparison occurs only in the common decoder.</p></div></div><div class="scope"><b>One component of a larger system.</b><span>Preattentive vision comes first. Visual working memory, visual attention and integration remain future components. This experiment does not implement them.</span></div></section>
<section class="section wrap" id="results"><div class="section-head"><span class="index">01</span><div><div class="eyebrow">Measured performance</div><h2>The evidence, by candidate.</h2><p class="intro">Equal output scales and a common decoder let us compare useful visual representations. Parameter counts and compute still differ: this is a candidate screen, not an isolated operation ablation.</p></div></div><div class="finding">{measured_summary(data)}</div>{result_table(data)}<p class="caption">Profiles report trainable encoder parameters; the common decoder has {decoder_info} trainable parameters. Profile pairs/s = batch size ÷ measured median training-step time, not deployment latency or end-to-end dataset throughput. Intervals are supplied by the evaluator; missing measurements stay pending.</p><div class="facts"><div><strong>{esc(updates)}</strong><span>target updates / model–seed</span></div><div><strong>{esc(pairs)}</strong><span>fresh training pairs / model–seed</span></div><div><strong>{esc(nseed)}</strong><span>configured training seeds</span></div><div><strong>{number(budget/60 if finite(budget) else None,0)} min</strong><span>total local compute ceiling</span></div></div>{figures}<details><summary>Inspect family, difficulty and response counts</summary>{family_details(data)}</details><details><summary>Evaluation uncertainty and how to read a ranking</summary><p>AUROC measures score ranking; balanced accuracy averages sensitivity and specificity at the saved decision threshold. Hits, misses, false alarms and correct rejections expose response bias. A class-balanced benchmark does not make frames or repeated versions of one natural image independent.</p><p>Each table row belongs to one trained seed. CIs supplied for its test data do not cover variation over all possible training runs. Macro averages weight stimulus families equally; their exact CI construction follows the saved evaluator metadata. Repeated natural source identities require base-image-aware uncertainty. Validation selects checkpoints and candidates; the test set describes all five saved candidates without hiding unfavorable outcomes.</p><p class="small">{raw_link} · <a href="train.py">Training/evaluation implementation</a>. The report does not manufacture, recompute or tighten missing confidence intervals.</p></details></section>
<section class="section wrap" id="task"><div class="section-head"><span class="index">02</span><div><div class="eyebrow">A deliberately small benchmark</div><h2>Change across four visual families.</h2><p class="intro">Exactly two RGB100 × 100 frames enter each trial. No phase, source category, edit location, label or previous-frame buffer enters the encoder.</p></div></div><div class="two"><div><h3>Same scene is not the same pixels.</h3><p>Every observed frame gets independent gain, brightness and pixel noise. No-change examples therefore remain noisy comparisons, while changed examples alter the underlying scene variant.</p><p>For every trial, an original A and edited B are generated before the label is applied. No-change presents A/A or B/B; change presents A/B or B/A. Each frame position has the same original/edited marginal under either label.</p><div class="callout">A visible edit artifact in one image alone should not reveal the change label. The comparison must use the pair.</div></div><figure class="examples-box">{examples}</figure></div><div class="four"><div><h4>Gabors</h4><p>Rotate one of nine patches. Angle bins: 35–65°, 15–35°, 5–15°.</p></div><div><h4>Dot displacement</h4><p>Move one of sixteen dots. Displacement bins: 6–9, 3–6, 1.5–3 pixels.</p></div><div><h4>Colored shapes</h4><p>Change one object's color or category. Smaller changes/objects define harder bins.</p></div><div><h4>Natural-image edits</h4><p>Local color, occlusion or translation on CIFAR-10 sources enlarged from32 to100pixels.</p></div></div><p class="caption">Actual generation definitions: <a href="task.md">task contract</a> · <a href="stimuli.py">stimulus implementation</a> · <a href="stimulus_examples.json">example metadata</a> · <a href="https://www.cs.toronto.edu/~kriz/cifar.html">official CIFAR-10 source</a>. Two dot frames measure displacement, not a change in velocity or acceleration. Natural image identities are split across train/validation/test; class labels are unused.</p></section>
<section class="section wrap" id="comparison"><div class="section-head"><span class="index">03</span><div><div class="eyebrow">What every model shares</div><h2>Separate encoding.<br>Expressive comparison.</h2></div></div><div class="two"><div><h3>One encoder, applied twice.</h3><p>The two frames use exactly the same weights and normalization. A computational batch joins the frames for efficiency, but no encoder layer mixes examples across the batch. Each candidate returns three intact feature fields.</p><div class="equation">{pair_math}</div><div class="pyramid"><span>24 × 50 × 50</span><span>48 × 25 × 25</span><span>96 × 13 × 13</span></div><p class="small muted">Channel widths24/48/96; stage depths1/2/2. No global spatial collapse in the encoder, temporal state, pretrained weights or injected process noise.</p></div><div><h3>A common decoder with room to learn.</h3><p>At each scale, a shared projection produces A and B. The decoder combines absolute difference, mean and product, preserving symmetry to swapping the frames.</p><div class="equation">{interaction_math}</div><p class="small muted">Local convolutions learn interactions before pooling to13 × 13. Multiscale fusion precedes global mean/max and an MLP. There is no raw-image bypass. Decoder topology and dropout are shared across candidates, while learned weights are trained separately.</p></div></div><div class="loss"><div><div class="eyebrow">Pair-level supervised learning</div><p>One target per two-image trial. The common classifier is optimized with binary-label cross-entropy over its two logits.</p></div><div class="equation">{loss_math}</div></div><p class="caption">θ denotes encoder parameters, ψ decoder parameters, B batch size and y the pair label. Exact optimization/exposure settings come from the frozen experiment configuration. <a href="decoder.py">Common decoder source</a> · {raw_link}</p></section>
<section class="section wrap" id="models"><div class="section-head"><span class="index">04</span><div><div class="eyebrow">Five computational biases</div><h2>What each encoder contributes.</h2><p class="intro">These are compact paper-inspired adaptations trained from scratch, not canonical pretrained models or reproduced large-scale paper benchmarks. The comparison asks what their adopted computations offer here.</p></div></div><div class="model-grid">{cards}</div><details><summary>Exact VOne front-end and boundaries of the analogy</summary><p>The fixed bank uses four orientations (0, π/4, π/2, 3π/4), two frequency/sigma pairs (0.12cycles/pixel,2pixels) and (0.25cycles/pixel,1.5pixels), and three unit-length luminance/opponent color axes. Each9 × 9 Gabor kernel is mean-subtracted and L2-normalized. The24 quadrature pairs produce24 rectified simple responses and24 energy responses; raw RGB is concatenated to retain DC/color information.</p><p>In the equation above, a and b are0 andπ/2 phase filter responses. The10⁻⁶ term regularizes the square root; subtraction of10⁻³ makes zero input give zero complex energy. The fixed bank contains11,664 buffer coefficients in addition to trainable parameters.</p><p>The grid is hand-chosen rather than fitted to physiological response distributions. The original VOneNet's neuronal stochasticity is omitted. No cortical dynamics, eye movements or human neural-data validation is introduced. The paper supplies a rationale for this prototype, not a guarantee of its biological fidelity.</p></details></section>
<section class="section wrap" id="grounding"><div class="section-head"><span class="index">05</span><div><div class="eyebrow">Two different questions</div><h2>Useful features.<br>Biological grounding.</h2><p class="intro">A performance winner and the most explicitly neuroscience-inspired model need not be the same candidate.</p></div></div><div class="grounding-panel"><div><span class="pill">Architectural grounding</span><h3>VOne ResNet has the most direct V1-inspired front end in this set.</h3><p>Oriented Gabor filtering, simple-response rectification and quadrature energy name explicit early-visual computations. That makes its connection more concrete than a generic residual path, kernel decomposition or learned global gate.</p><p>It is still a deterministic engineering adaptation. This screen provides no neural recordings, population fits or behavioral correspondence test. “Most directly inspired” is an assessment of the adopted construction, not a measured neuroscience result.</p></div><div><span class="pill">Measured usefulness</span>{measured_summary(data)}<p class="small">The test asks whether an encoder plus trained decoder supports noisy two-frame decisions on these distributions. It does not measure attentive guidance, working-memory maintenance, natural search or complete GS6 behavior.</p></div></div><div class="limitations"><h3>What can change the interpretation?</h3><div class="three"><div><h4>Exposure and randomness</h4><p>Different encoders can acquire at different rates. A finite equal-update screen is useful evidence at its budget, not a proof of asymptotic inferiority. Small seed counts leave initialization variation uncertain.</p></div><div><h4>Task-specific advantages</h4><p>A Gabor bank may favor Gabor stimuli. Spatial alignment and synthetic edit structure may favor pair comparison tricks. Family/difficulty scores help reveal uneven strengths.</p></div><div><h4>Representation versus decoder</h4><p>The common decoder is expressive and trained jointly. Scores reflect that whole learned pairing. Equal decoder topology does not isolate one encoder operation causally.</p></div></div></div></section>
<section class="closing wrap"><div class="eyebrow">The broader project</div><h2>Build the first component.<br>Keep the later questions open.</h2><p>Guided Search6.0 supplies the functional scaffold: early visual processing, selective guidance and identification are distinct. This project omits the diffuser. Activated long-term memory is provisionally treated as synaptic weights at the user's request; that is a modeling approximation, not Wolfe's biological claim.</p><p>Measured PAV results can motivate the next small encoder development step. They do not choose or implement visual attention, visual working memory or their integration.</p><p class="sources"><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/">Wolfe · Guided Search6.0</a> · <a href="research.md">Full source rationale</a> · <a href="README.md">PAV scope</a> · <a href="../README.md">Project overview</a></p></section></main><footer class="wrap"><div>PreAttentiveVision · Local research report<br>{generated} · saved status: {esc(status)} · {raw_link}</div><button type="button" onclick="window.print()">Print / save PDF</button></footer></body></html>'''
    page=page.replace('<div class="facts">', '<p class="caption">The production recipe uses equal-family batches (eight pairs per family at batch32). Corresponding procedural pairs are shared across candidates via the same training stream seed:49,152 matched pair draws per model become245,760 model–pair presentations across five candidates, not independent scenes across models. The800 validation pairs are reused at three scheduled looks; the3,200 test pairs are new and shared across models. These are the current fixed allocation, not a universal sufficient learning horizon.</p><div class="facts">')
    page=page.replace('<details><summary>Inspect family, difficulty and response counts</summary>', execution_details(data)+'<details><summary>Inspect family, difficulty and response counts</summary>')
    page=page.replace('not deployment latency or end-to-end dataset throughput.', 'including data generation, transfer, forward/backward and the optimizer step. It excludes interpreter startup and checkpoint/evaluation overhead, and is not inference FPS.')
    page=page.replace('their exact CI construction follows the saved evaluator metadata. Repeated natural source identities require base-image-aware uncertainty.', 'their95% intervals average independent family-bootstrap replicates. The evaluator uses1,000 resamples: natural pairs cluster by CIFAR source-image identity, while synthetic examples resample generated pairs. Repeated natural variants are not counted as independent source images.')
    page=page.replace('at the saved decision threshold.', 'at one validation-calibrated threshold per model–seed, locked across all four test families. Threshold selection maximizes macro validation balanced accuracy; ties choose the largest threshold. The raw result also reports default0.5-threshold scores.')
    page=page.replace('<details><summary>Evaluation uncertainty and how to read a ranking</summary>', '<div class="callout" style="margin:25px 0"><strong>Preset engineering target:</strong> macro balanced accuracy≥90% and every family≥85%, with test confirmation reported at the locked threshold. This heuristic target is not a neuroscience criterion; training does not stop early when it is met.</div><details><summary>Evaluation uncertainty and how to read a ranking</summary>')
    for old,new in {'RGB100':'RGB 100','from32 to100pixels':'from 32 to 100 pixels','to13 × 13':'to 13 × 13','widths24/48/96':'widths 24/48/96','depths1/2/2':'depths 1/2/2','batch32':'batch 32','seed:49,152':'seed: 49,152','become245,760':'become 245,760','The800':'The 800','the3,200':'the 3,200','their95%':'their 95%','uses1,000':'uses 1,000','default0.5':'default 0.5','Default0.5':'Default 0.5','accounted32':'accounted 32','accuracy≥90%':'accuracy ≥ 90%','family≥85%':'family ≥ 85%','Search6.0':'Search 6.0','0.12cycles/pixel,2pixels':'0.12 cycles/pixel, 2 pixels','0.25cycles/pixel,1.5pixels':'0.25 cycles/pixel, 1.5 pixels','Each9 × 9':'Each 9 × 9','The24':'The 24','produce24':'produce 24','and24':'and 24','are0 andπ/2':'are 0 and π/2','The10⁻⁶':'The 10⁻⁶','of10⁻³':'of 10⁻³','contains11,664':'contains 11,664'}.items():
        page=page.replace(old,new)
    if data.get('_report_halted'):
        notice='<div class="finding" style="border-left-color:#b05d3c;background:#f4e6da;margin:28px 0"><div class="eyebrow">Protocol correction · comparison halted</div><h3 style="margin-top:12px">The dot task did not match the request.</h3><p>The requested stimulus is Krauzlis moving dots, not arbitrary displacement of one static dot. Existing validation curves, example images and compute profiles below belong to that rejected prototype. They are obsolete-protocol diagnostics, not results for the requested experiment.</p><p>Protocol research is underway. The temporal choice remains pending; this report does not select it, claim a new run, or authorize a restart.</p></div>'
        page=page.replace('<section class="hero wrap">','<section class="hero wrap">'+notice)
        page=page.replace('Before attention · PAV encoder comparison','PAV comparison halted · stimulus redesign')
        page=page.replace('Before<br><em>attention.</em>','Protocol<br><em>on hold.</em>')
        page=page.replace('Five lightweight encoders.<br>Two images. One controlled comparison.','Five encoder candidates.<br>Moving-dot protocol correction pending.')
        page=page.replace('Which spatial representation best preserves the visual evidence needed to tell whether a scene changed?','The requested moving-dot experiment has no completed comparison yet. This page retains the earlier prototype transparently while its stimulus contract is corrected.')
        page=page.replace('Measured performance','Historical engineering diagnostics')
        page=page.replace('The evidence, by candidate.','No requested-task results yet.')
        page=page.replace('Acquisition across equal updates','Obsolete-protocol validation diagnostics')
        page=page.replace('Validation scores at recorded checkpoints. The same validation set supports model/checkpoint selection. These are development trajectories, not independent held-out replications.','Partial validation scores from the rejected displacement prototype. These measurements cannot rank encoders for the requested Krauzlis moving-dot task. No test results are supplied.')
        page=page.replace('Change across four visual families.','The superseded stimulus suite.')
        page=page.replace('A deliberately small benchmark','Rejected prototype · retained for transparency')
        page=page.replace('What every model shares','Superseded pairwise prototype')
        page=page.replace('Actual generator samples · medium difficulty','OBSOLETE PROTOCOL examples · medium difficulty')
        page=page.replace('Actual generation definitions:','Superseded generation definitions:')
        page=page.replace('The production recipe uses equal-family batches','The halted prototype’s planned production recipe used equal-family batches')
        page=page.replace('target updates / model–seed','superseded target updates / model–seed')
        page=page.replace('fresh training pairs / model–seed','superseded target pairs / model–seed')
        page=page.replace('total local compute ceiling','historical allocation ceiling · not restart permission')
        page=page.replace('Preset engineering target:','Superseded engineering target:')
        page=page.replace('Execution statuses are snapshots saved when this report was generated.','Saved worker statuses may predate the stop. The comparison is halted for redesign; these are retained execution records, not a live running-state claim.')
        page=page.replace('A performance winner and the most explicitly neuroscience-inspired model need not be the same candidate.','No performance winner for the corrected task is available. Architectural inspiration can still be described independently of the rejected prototype’s measurements.')
        page=page.replace('Measured usefulness</span>','Requested-task usefulness · pending</span>')
    output.write_text(page,encoding='utf-8')
    return {'output':str(output),'status':status,'completed_model_seed_evaluations':len(runs),'mathml_expressions':page.count('<math '),'chart_exports':list(chart_data),'bytes':output.stat().st_size}


CSS='''
:root{--paper:#f5f3ed;--white:#fffefa;--ink:#263045;--muted:#72798a;--line:#d9dce1;--accent:#756297;--orange:#be774e;--serif:Georgia,serif;--sans:Inter,"Segoe UI",Arial,sans-serif}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:84px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 var(--sans);-webkit-font-smoothing:antialiased}.wrap{max-width:1180px;width:calc(100% - 72px);margin:auto}a{color:#645083;text-underline-offset:4px}a:hover{color:#ae653e}.skip{position:absolute;left:10px;top:-90px;background:#fff;padding:12px;z-index:30}.skip:focus{top:10px}header{position:sticky;top:0;background:rgba(245,243,237,.97);border-bottom:1px solid var(--line);z-index:5}.nav{height:68px;display:flex;justify-content:space-between;align-items:center;gap:20px}.brand{color:var(--ink);font-size:11px;letter-spacing:2px;font-weight:800;text-decoration:none}.brand:before{content:'◧';color:var(--accent);margin-right:12px;font-size:20px}nav{display:flex;gap:25px;font-size:12px}nav a{color:var(--muted);text-decoration:none}.hero{padding:65px 0 0}.eyebrow{font-size:10px;font-weight:750;text-transform:uppercase;letter-spacing:2.2px;color:var(--accent)}.hero-grid{display:grid;grid-template-columns:1.2fr 1fr;gap:75px;align-items:center;padding:15px 0 55px}h1{font:clamp(68px,8vw,111px)/.99 var(--serif);letter-spacing:-5px;font-weight:400;margin:25px 0}h1 em{color:var(--accent);font-weight:400}.dek{font-size:22px;line-height:1.5;margin:30px 0 20px}.hero-copy{font-size:16px;color:var(--muted);max-width:450px}.hero-right{padding-top:15px}.status{font-size:11px;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--line);padding-bottom:17px}.status small{display:block;font-size:9px;color:var(--muted);letter-spacing:.6px;margin-top:6px}.dot{display:inline-block;width:7px;height:7px;background:var(--orange);border-radius:50%;margin-right:7px}.pair-art{padding:30px 25px;background:#ece8f0;margin:25px 0 15px}.art-images{display:flex;justify-content:center;gap:35px}.art-frame{background:#30334a;display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:17px;width:112px;height:112px;transform:rotate(-5deg);box-shadow:8px 10px 0 #d8d0e1}.art-frame.alt{transform:rotate(5deg)}.art-frame i{background:repeating-linear-gradient(55deg,#c8c1d8 0 2px,transparent 2px 5px);border-radius:50%}.alt i:last-child{background:repeating-linear-gradient(115deg,#dfae7b 0 2px,transparent 2px 5px)}.art-connect{display:flex;justify-content:center;align-items:center;gap:24px;color:#9686ab;font-size:23px;margin:18px 0 6px}.art-connect span{font:10px var(--sans);letter-spacing:1px;text-transform:uppercase}.art-encoder,.art-decoder{border:1px solid #aaa1b8;text-align:center;padding:10px;font-size:11px;letter-spacing:.7px;background:#f7f4fa}.art-down{text-align:center;color:#a095ae}.art-decoder{background:#685c7d;color:white;border:0}.scope{display:grid;grid-template-columns:230px 1fr;gap:24px;background:#293146;color:#e8e9ec;padding:30px 35px;font-size:13px}.scope b{font:21px/1.4 var(--serif);color:#e7d2b2}.scope span{max-width:760px}.section{padding:78px 0;border-bottom:1px solid var(--line)}.section-head{display:grid;grid-template-columns:70px 1fr;gap:20px;margin-bottom:34px}.index{font:43px/1 var(--serif);color:#b4aabf;margin-top:26px}h2{font:clamp(35px,4.2vw,51px)/1.13 var(--serif);font-weight:400;letter-spacing:-1.2px;margin:11px 0 17px}.intro{font-size:17px;color:#727785;max-width:850px;margin:0}h3{font:26px/1.3 var(--serif);font-weight:400;margin:0 0 15px}h4{font-size:14px;margin:0 0 8px}p{margin:0 0 17px}.muted{color:var(--muted)}.small,.caption,figcaption{font-size:12px;line-height:1.7}.caption,figcaption{color:var(--muted);margin-top:15px}.finding{border-left:3px solid var(--accent);background:#eeebf1;padding:27px 30px;margin-bottom:27px}.finding p{max-width:940px;font-size:14px;margin-bottom:10px}.finding p:last-child{margin-bottom:0}.finding h3{font-size:27px}.table-scroll{overflow:auto;background:var(--white);border:1px solid var(--line);padding:6px 19px}table{width:100%;border-collapse:collapse;font-size:12px;text-align:left;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--line);padding:16px 13px;white-space:nowrap}th:first-child,td:first-child{padding-left:0}thead th{font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);font-weight:750}table small{display:block;font-size:10px;color:var(--muted);font-weight:400;letter-spacing:0;text-transform:none}tbody th{font-weight:650}tbody i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:8px}tbody th small{padding-left:15px}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:25px;margin:30px 0 40px}.facts>div{border-top:1px solid var(--line);padding-top:18px}.facts strong{display:block;font:32px var(--serif);margin-bottom:8px}.facts span{font-size:11px;color:var(--muted)}.plot{margin:36px 0;background:var(--white);border:1px solid var(--line);padding:25px}.plot img{display:block;width:100%;height:auto}.plot h3{font-size:25px}.pending-chart{border:1px dashed #bfb6cc;padding:50px 35px;text-align:center;background:linear-gradient(90deg,transparent 49.8%,#e5e0e8 50%,transparent 50.2%)}.pending-chart span{font:29px var(--serif);color:#8b7b9e}.pending-chart p{font-size:13px;color:var(--muted);max-width:510px;margin:15px auto 0}details{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:17px 0;margin:23px 0}details+details{border-top:0;margin-top:-23px}summary{cursor:pointer;font-size:13px;font-weight:650;list-style:none;display:flex;justify-content:space-between;gap:20px}summary:after{content:'+';font-size:21px;color:var(--accent)}details[open] summary{margin-bottom:18px}details[open] summary:after{content:'−'}details p{font-size:13px}.two{display:grid;grid-template-columns:1fr 1fr;gap:45px}.four{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:30px}.four div{border-top:2px solid #c6bbd0;padding-top:17px}.four p{font-size:12px;margin:0}.examples-box{margin:0;border:1px solid var(--line);background:white;padding:17px}.examples{display:block;width:100%;height:auto}.examples-box figcaption{font-size:10px}.callout{border-left:3px solid #c59970;padding:13px 20px;background:#f1e8dc;font-size:13px}.equation{padding:17px 5px;margin:18px 0;overflow:auto;font-size:19px;max-width:100%}math{font-family:'Cambria Math','STIX Two Math',serif;line-height:1.4}math[display=block]{margin:0 auto;min-width:max-content}.pyramid{display:flex;gap:9px;margin:24px 0;flex-wrap:wrap}.pyramid span{background:#ebe7ef;border:1px solid #c9c0d3;padding:10px 13px;font:12px Consolas,monospace}.loss{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:center;background:#eae8ef;padding:28px;margin-top:25px}.loss p{font-size:13px;margin:10px 0 0}.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.model{background:var(--white);border:1px solid var(--line);border-top:3px solid var(--model);padding:27px}.model:last-child{grid-column:1/-1}.model:last-child>.equation{float:right;width:45%;margin:0 0 15px 30px}.model-top{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:9px;color:var(--model);letter-spacing:.7px;text-transform:uppercase;margin-bottom:20px}.model-top b{font-weight:400;color:var(--muted);letter-spacing:0}.model h3{font-size:25px}.model p{font-size:13px}.model .equation{background:#f6f4f7;border:1px solid #e4dfe9;font-size:17px;padding:17px 10px}.ground{color:var(--muted);border-top:1px solid var(--line);padding-top:15px}.source{font-size:10px!important;margin:15px 0 0}.grounding-panel{background:#2e3448;color:#e8e8ec;display:grid;grid-template-columns:1fr 1fr;gap:45px;padding:40px}.grounding-panel h3{color:#fff4e4;font-size:28px;margin-top:20px}.grounding-panel p{font-size:14px;color:#d2d1dc}.pill{display:inline-block;border:1px solid #89839a;color:#decaa7;padding:6px 10px;font-size:9px;text-transform:uppercase;letter-spacing:1.1px}.limitations{margin-top:36px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:30px}.three p{font-size:13px;color:var(--muted)}.closing{padding:65px 0}.closing h2{max-width:700px}.closing p{max-width:830px;color:var(--muted)}.closing .sources{font-size:12px;margin-top:28px}footer{border-top:1px solid var(--line);padding:30px 0 40px;display:flex;justify-content:space-between;gap:25px;font-size:10px;color:var(--muted)}button{font:11px var(--sans);background:none;border:1px solid #bfc2cd;color:var(--ink);padding:10px 17px;cursor:pointer;align-self:start}a:focus-visible,summary:focus-visible,button:focus-visible{outline:3px solid var(--orange);outline-offset:5px}@media(max-width:900px){.wrap{width:calc(100% - 40px)}.hero-grid{gap:35px}h1{font-size:90px}.two{gap:27px}.grounding-panel{padding:28px;gap:27px}.model{padding:22px}.four{grid-template-columns:1fr 1fr}.facts{gap:17px}.model .equation{font-size:15px}.scope{grid-template-columns:180px 1fr}.loss{gap:20px}}@media(max-width:620px){.wrap{width:calc(100% - 32px)}.nav{height:58px}.brand{font-size:9px;letter-spacing:1px}.brand:before{margin-right:6px}nav{gap:12px;font-size:10px}nav a:nth-child(2){display:none}.hero{padding-top:35px}.hero-grid,.two,.model-grid,.grounding-panel,.loss{grid-template-columns:1fr}.hero-grid{gap:20px;padding-bottom:30px}h1{font-size:85px;letter-spacing:-4px;margin:24px 0}.dek{font-size:20px}.hero-right{padding-top:0}.pair-art{max-width:380px;margin:22px auto 15px}.hero-right>.small{max-width:380px;margin:auto}.scope{grid-template-columns:1fr;gap:13px;padding:25px}.scope b{font-size:22px}.section{padding:48px 0}.section-head{grid-template-columns:35px 1fr;gap:15px;margin-bottom:25px}.index{font-size:32px;margin-top:28px}h2{font-size:35px;letter-spacing:-1px}.intro{font-size:15px}.finding{padding:23px}.finding h3{font-size:25px}.facts{grid-template-columns:1fr 1fr;gap:23px}.facts strong{font-size:28px}.plot{padding:17px 10px}.plot figcaption{padding:0 7px}.table-scroll{padding:4px 13px}.model:last-child{grid-column:auto}.model:last-child>.equation{float:none;width:auto;margin:18px 0}.model .equation{font-size:15px}.loss{padding:22px;gap:0}.equation{font-size:17px}.grounding-panel{padding:27px;gap:25px}.three{grid-template-columns:1fr;gap:7px}.closing{padding:45px 0}.four{gap:20px}.examples-box{padding:11px}footer{flex-direction:column;gap:18px}.pending-chart{padding:35px 20px}.pending-chart span{font-size:25px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{header,button,.skip{display:none}.wrap{width:100%;max-width:none}.hero{padding-top:0}h1{font-size:72px}.hero-grid{gap:35px}.section{padding:32px 0}.section-head{break-after:avoid}.model,.plot,.finding,.grounding-panel,.examples-box{break-inside:avoid}.scope,.grounding-panel,.art-frame,.art-decoder,.loss{-webkit-print-color-adjust:exact;print-color-adjust:exact}.table-scroll{overflow:visible}table{font-size:9px}th,td{padding:8px 6px}.caption,figcaption{font-size:9px}.equation{overflow:visible;font-size:15px}.closing{padding:30px 0}a{color:inherit}}
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results.json')
    parser.add_argument('--output',type=Path,default=HERE/'report.html')
    args=parser.parse_args()
    print(json.dumps(build_report(args.results.resolve(),args.output.resolve()),indent=2))
