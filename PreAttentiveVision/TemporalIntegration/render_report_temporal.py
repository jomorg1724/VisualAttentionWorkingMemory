"""Render saved temporal experiment results only; no model or GPU imports."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ('spatial_kda','convgru','opponent')
LABELS = {'spatial_kda':'Spatial KDA','convgru':'ConvGRU','opponent':'Opponent filters','reference':'Direct pair reference'}
TASKS = ('motion_direction','orientation','contrast','spatial_frequency','chromatic_increment','contour','natural_spectrum')
TITLES = ('Motion direction','Orientation','Contrast','Spatial frequency','Color increment','Contour grouping','Natural spectral detail')
COLORS = {'spatial_kda':'#6554ad','convgru':'#178c91','opponent':'#c07837','reference':'#596272'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def finite(x):
    return isinstance(x,(int,float)) and math.isfinite(x)


def pct(x):
    return f'{100*x:.1f}%' if finite(x) else 'Pending'


def num(x,digits=0):
    return f'{x:,.{digits}f}' if finite(x) else 'Pending'


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def mathml(body):
    return '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'+body+'</math>'


def sub(letter,index):
    return f'<msub><mi>{letter}</mi><mrow><mi>{index}</mi></mrow></msub>'


def summary(record):
    candidate = record.get('test') or record.get('summary') or record
    return candidate if isinstance(candidate,dict) else {}


def score(record,task):
    return summary(record).get('tasks',{}).get(task,{}).get('overall',{})


def completed(record):
    test=summary(record)
    return test.get('status')=='completed' and all(finite(score(record,t).get('balanced_accuracy')) for t in TASKS)


def render(source,output):
    data=load(source)
    raw_runs=data.get('runs',[])
    runs=raw_runs if isinstance(raw_runs,dict) else {r.get('model',r.get('arm')):r for r in raw_runs}
    reference=data.get('reference',data.get('pair_reference',{}))
    if isinstance(reference,dict) and reference.get('tasks'):
        reference={'test':reference}
    records={**runs,'reference':reference}
    finished=all(completed(runs.get(a,{})) for a in ARMS) and completed(reference)
    n_completed=sum(completed(runs.get(a,{})) for a in ARMS)
    checks=load(HERE/'focused_checks.json').get('models',{})
    status='Final held-out results' if finished else f'Evaluation pending · {n_completed}/3 arms tested'
    score_rows=[]
    auc_rows=[]
    for task,title in zip(TASKS,TITLES):
        scores=[title];aucs=[title]
        for arm in ('reference',)+ARMS:
            s=score(records.get(arm,{}),task) if completed(records.get(arm,{})) else {}
            value=s.get('balanced_accuracy')
            bar=f'<div class="bar"><i style="width:{100*value:.3f}%;background:{COLORS[arm]}"></i></div>' if finite(value) else ''
            ci=s.get('balanced_accuracy_ci95')
            uncertainty=f'<small>95% CI {pct(ci[0])}–{pct(ci[1])}</small>' if isinstance(ci,list) and len(ci)==2 else ''
            scores.append(f'<b>{pct(value)}</b>{bar}{uncertainty}')
            aucs.append(num(s.get('macro_ovr_auc'),4))
        score_rows.append(scores);auc_rows.append(aucs)
    columns=['Task']+[LABELS[a] for a in ('reference',)+ARMS]
    result_table=table(columns,score_rows)
    auc_table=table(columns,auc_rows)
    cost_rows=[]
    for arm in ARMS:
        r=runs.get(arm,{})
        measured=r.get('profile',data.get('profiles',{}).get(arm,{}))
        check=checks.get(arm,{})
        exposure=r.get('fresh_pairs',r.get('training_pairs',r.get('train_pairs')))
        end=r.get('terminal_step',r.get('step',r.get('final_step')))
        chosen=r.get('selected_step',r.get('best_step',summary(r).get('step')))
        cost_rows.append([LABELS[arm],num(check.get('trainable_parameters',measured.get('trainable_params'))),
                         num(check.get('state_bytes_per_example',0)/(1024**2),3)+' MiB',
                         num(end),num(chosen),num(exposure),
                         num(measured.get('train_step_mean'),3),
                         num(measured.get('peak_vram_bytes')/(1024**3),2)+' GiB' if finite(measured.get('peak_vram_bytes')) else 'Pending'])
    cost=table(['Arm','Trainable parameters','Persistent state / pair','Terminal updates','Selected updates','Fresh training pairs','Profile seconds / update','Peak training VRAM'],cost_rows)
    exposure_rows=[]
    batch_size=data.get('config',{}).get('batch_size',32)
    for task,title in zip(TASKS,TITLES):
        row=[title]
        for arm in ARMS:
            r=runs.get(arm,{})
            for key in ('per_task_updates','selected_per_task_updates'):
                value=r.get(key,{}).get(task)
                row.append(num(value*batch_size) if finite(value) else 'Pending')
        exposure_rows.append(row)
    exposure_html=table(['Task']+[f'{LABELS[a]} · {kind}' for a in ARMS for kind in ('terminal pairs','selected pairs')],exposure_rows)
    diagnostic_parts=[]
    for mode,title in (('reset_before_second','Reset state before frame 2'),('frame_swap','Reverse frame order and invert labels')):
        rows=[]
        for task,task_title in zip(TASKS,TITLES):
            rows.append([task_title]+[pct(summary(runs.get(a,{})).get('diagnostics',{}).get(mode,{}).get('tasks',{}).get(task,{}).get('overall',{}).get('balanced_accuracy')) for a in ARMS])
        diagnostic_parts.append(f'<h3>{title}</h3>'+table(['Task']+[LABELS[a] for a in ARMS],rows))
    paired=data.get('paired_comparisons',data.get('comparisons',[]))
    if isinstance(paired,dict):
        paired=[dict(model=k,tasks=v.get('tasks',v)) for k,v in paired.items()]
    paired_rows=[]
    for comparison in paired:
        arm=comparison.get('model',comparison.get('candidate','unspecified'))
        for task,title in zip(TASKS,TITLES):
            cell=comparison.get('tasks',{}).get(task,{})
            delta=cell.get('delta_ba');interval=cell.get('delta_ba_ci95')
            paired_rows.append([escape(LABELS.get(arm,arm)),title,
                                f'{100*delta:+.2f} pp' if finite(delta) else 'Pending',
                                f'[{100*interval[0]:+.2f}, {100*interval[1]:+.2f}] pp' if isinstance(interval,list) and len(interval)==2 else 'Pending'])
    paired_html=table(['Candidate','Task','Change vs pair reference','Paired 95% interval'],paired_rows) if paired_rows else '<p class="pending">Paired uncertainty will appear when saved-score analysis completes.</p>'
    curves=[]
    for arm in ARMS:
        for item in runs.get(arm,{}).get('val_curve',[]):
            curves.append([LABELS[arm],num(item.get('step')),pct(item.get('min_task_ba')),num(item.get('macro_ovr_auc',item.get('mean_task_auc')),4)])
    curve_html=table(['Arm','Updates','Minimum task BA','Mean task AUC'],curves) if curves else '<p class="pending">Planned validation trajectories will appear as checkpoints are evaluated.</p>'
    minima=[(a,min(score(runs[a],t)['balanced_accuracy'] for t in TASKS)) for a in ARMS if completed(runs.get(a,{}))]
    if finished:
        passed=[LABELS[a] for a,v in minima if v>=.95]
        conclusion=('The 95% point-score screen was met by '+', '.join(passed)+'.') if passed else 'None of the three fitted temporal models met 95% balanced accuracy on every task at this exposure.'
        conclusion+=' Read paired effects and acquisition trajectories before attributing differences to an architecture limitation.'
    else:
        conclusion='The three temporal mechanisms are specified. Held-out comparisons are not yet complete; no winner or performance conclusion is reported here.'
    kda=mathml(sub('S̄','t')+'<mo>=</mo><mi>Diag</mi><mo>(</mo>'+sub('α','t')+'<mo>)</mo>'+sub('S','t−1'))
    kda+=mathml(sub('e','t')+'<mo>=</mo>'+sub('v','t')+'<mo>−</mo><msup>'+sub('S̄','t')+'<mi>⊤</mi></msup>'+sub('k','t'))
    kda+=mathml(sub('S','t')+'<mo>=</mo>'+sub('S̄','t')+'<mo>+</mo>'+sub('β','t')+sub('k','t')+'<msup>'+sub('e','t')+'<mi>⊤</mi></msup>')
    kda+=mathml(sub('o','t')+'<mo>=</mo><msup>'+sub('S','t')+'<mi>⊤</mi></msup>'+sub('q','t'))
    gru=mathml(sub('z','t')+'<mo>,</mo>'+sub('r','t')+'<mo>=</mo><mi>σ</mi><mo>(</mo><mi>Conv</mi><mo>[</mo>'+sub('U','t')+'<mo>,</mo>'+sub('h','t−1')+'<mo>]</mo><mo>)</mo>')
    gru+=mathml(sub('h̃','t')+'<mo>=</mo><mi>tanh</mi><mo>(</mo><mi>Conv</mi><mo>[</mo>'+sub('U','t')+'<mo>,</mo>'+sub('r','t')+'<mo>⊙</mo>'+sub('h','t−1')+'<mo>]</mo><mo>)</mo>')
    gru+=mathml(sub('h','t')+'<mo>=</mo><mo>(</mo><mn>1</mn><mo>−</mo>'+sub('z','t')+'<mo>)</mo><mo>⊙</mo>'+sub('h','t−1')+'<mo>+</mo>'+sub('z','t')+'<mo>⊙</mo>'+sub('h̃','t'))
    neuro=mathml(sub('F','t')+'<mo>=</mo><mn>.25</mn>'+sub('F','t−1')+'<mo>+</mo><mn>.75</mn>'+sub('U','t'))
    neuro+=mathml(sub('L','t')+'<mo>=</mo><mn>.75</mn>'+sub('L','t−1')+'<mo>+</mo><mn>.25</mn>'+sub('U','t'))
    neuro+=mathml(sub('E','+')+'<mo>−</mo>'+sub('E','−')+'<mo>=</mo><mn>4</mn><mo>(</mo>'+sub('f','e')+sub('l','o')+'<mo>−</mo>'+sub('f','o')+sub('l','e')+'<mo>)</mo>')
    updated=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    config=data.get('config',{})
    plan_html=f"<p class='note'>Configured target: <b>{num(config.get('updates'))} updates per arm</b>; {num(config.get('training_pairs_per_arm'))} fresh pairs per arm. Finite allowance: {num(config.get('budget_seconds',0)/60,1)} minutes including profiling and evaluation. Measured total experiment wall time: {num(data.get('wall_seconds'),1)} seconds. Targets are plans; the table below reports saved execution evidence.</p>"
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Causal visual integration · PAV</title><style>
    :root{{--ink:#202a3a;--muted:#647184;--paper:#faf9f5;--line:#dedfd9;--teal:#178c91}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}}header{{background:#172a3a;color:#f6f4ec;padding:68px max(6vw,24px) 54px}}header p{{max-width:840px;color:#ccd6dc;font-size:20px}}.eyebrow{{font:700 12px/1.5 system-ui;letter-spacing:.16em;text-transform:uppercase;color:#66cfbe}}h1{{font:500 clamp(38px,5vw,66px)/1.1 Georgia,serif;letter-spacing:-.025em;max-width:950px;margin:22px 0}}nav{{display:flex;gap:24px;flex-wrap:wrap}}nav a{{color:#d6ece8;text-decoration:none}}main{{max-width:1320px;margin:auto;padding:32px 24px 80px}}section{{padding:35px 0;border-bottom:1px solid var(--line)}}h2{{font:500 32px/1.25 Georgia,serif;margin:0 0 20px}}h3{{margin:8px 0 12px;font-size:20px}}a{{color:#137b80}}.lead{{max-width:1000px;font-size:19px}}.badge{{display:inline-block;background:#e2ece8;color:#25534b;padding:7px 13px;border-radius:30px;font-size:13px;font-weight:650}}.pipeline{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0}}.pipeline div{{background:#edf0eb;border-top:3px solid var(--teal);padding:20px}}.pipeline b,.pipeline span{{display:block}}.pipeline span{{font-size:14px;color:var(--muted);margin-top:6px}}.models{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}.card{{background:white;border:1px solid var(--line);border-top:4px solid;padding:23px;border-radius:6px}}.card p{{font-size:15px}}.card small{{display:block;color:var(--muted)}}math{{font-size:18px;overflow-x:auto;margin:16px 0;min-height:29px}}.scroll{{overflow-x:auto;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.45}}th{{text-align:left;padding:13px 14px;border-bottom:2px solid #b9c4c8;color:#415465;white-space:nowrap;font-size:12px}}td{{padding:15px 14px;border-bottom:1px solid var(--line);vertical-align:top;min-width:100px}}td:first-child{{font-weight:600;min-width:175px}}small{{font-size:11px;color:var(--muted)}}td small{{display:block;white-space:nowrap}}.bar{{height:4px;background:#e9e9e4;margin:7px 0}}.bar i{{height:4px;display:block}}.pending{{background:#f0efe9;padding:18px;color:var(--muted)}}.note{{background:#eaf2ef;border-left:3px solid #429987;padding:18px 24px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:650}}footer{{color:var(--muted);font-size:12px;margin-top:30px}}code{{font-size:12px;overflow-wrap:anywhere}}@media(max-width:850px){{.models,.pipeline,.two{{grid-template-columns:1fr}}header{{padding:38px 24px}}main{{padding:20px}}math{{font-size:17px}}}}@media print{{header{{background:white;color:black}}header p,nav a{{color:black}}section{{break-inside:avoid}}}}
    </style></head><body><header><div class="eyebrow">PreAttentiveVision / temporal integration</div><h1>One frame at a time.<br>Three ways to carry the past.</h1><p>A controlled comparison of associative state, learned spatial recurrence and neuroscience-inspired temporal filters—built on the same competent visual encoder.</p><nav><a href="#question">The question</a><a href="#models">The mechanisms</a><a href="#evidence">The evidence</a><a href="#diagnostics">Temporal diagnostics</a></nav></header><main>
    <section id="question"><span class="badge">{status}</span><h2 style="margin-top:20px">Can temporal state replace the direct pair comparison?</h2><p class="lead">{conclusion}</p><div class="pipeline"><div><b>1 · Current image</b><span>One RGB 100×100 frame. Exactly two observations per example.</span></div><div><b>2 · Frozen PAV</b><span>ConvNeXt/GRN + late SE. Spatial maps at 50², 25² and 13².</span></div><div><b>3 · Causal update</b><span>Current 32-channel fields + explicit prior state → new state and output.</span></div><div><b>4 · Shared readout</b><span>Current fields + emitted state. Score after frame 2.</span></div></div><p>The encoder sees frames separately. No old image or separate first-frame map reaches the new decoder. All candidates retain the same current-frame appearance route, shared initial projections/readout, and contour-focused training schedule.</p></section>
    <section id="models"><h2>Three hypotheses, one sensory interface</h2><div class="models"><article class="card" style="border-top-color:{COLORS['spatial_kda']}"><h3>01 · Spatial KDA</h3><p>Maintain a small key–value matrix at each spatial site. Decay it, correct the current association, then read the updated matrix.</p>{kda}<small>Two heads · keys 8 · values 16 · zero initial state</small><p>Selective association is an engineering hypothesis; it is not an MT circuit model. <a href="https://arxiv.org/html/2510.26692v1#S3">Kimi Linear, equation 1</a>.</p></article><article class="card" style="border-top-color:{COLORS['convgru']}"><h3>02 · ConvGRU</h3><p>Learn how much spatial history to retain, reset and replace using local convolutional gates.</p>{gru}<small>32 state channels · 3×3 kernels · zero initial state</small><p>Recurrence and locality are broad biological analogies; the exact gates are engineered. <a href="https://arxiv.org/abs/1511.06432">Ballas et al.</a></p></article><article class="card" style="border-top-color:{COLORS['opponent']}"><h3>03 · Opponent filters</h3><p>Combine fast and slow traces with spatial quadrature filters. Their opponent response retains the direction of a temporal change.</p>{neuro}<small>First frame: F = L = U · fixed retention .25 / .75</small><p>Adopts energy/opponency and normalization principles from <a href="https://persci.mit.edu/pub_pdfs/spatio85.pdf">Adelson–Bergen</a> and <a href="https://www.cns.nyu.edu/pub/lcv/simoncelli96-reprint.pdf">Simoncelli–Heeger</a>; not a full MT reconstruction.</p></article></div><p>The opponent candidate also retains sustained appearance and signed transients, so motion energy does not become a bottleneck for color, contrast or contour judgments. All three emit 32-channel spatial maps at each scale.</p></section>
    <section id="evidence"><h2>Seven tasks. Every score stays visible.</h2><p>Balanced accuracy on fresh held-out pairs; 95% on every task is the engineering point-score screen. The frozen trained direct pair model (parent checkpoint step 2268) is reevaluated on the same new pairs. Chance is 25% for four-direction motion and 50% for the binary tasks.</p>{result_table}<details><summary>Score ranking: macro one-vs-rest AUC</summary>{auc_table}</details><details><summary>Paired changes relative to the direct pair reference</summary>{paired_html}</details><p class="note">Uncertainty is conditional on one fitted seed. Natural-image uncertainty groups repeated pairs by source photograph. A high point score does not guarantee a population threshold, and a low endpoint does not establish an architectural limit.</p></section>
    <section id="diagnostics"><h2>Did the model use temporal history?</h2><div class="two"><div>{diagnostic_parts[0]}<p>Removes prior state. A performance drop supports history use, although reset can create an unfamiliar state distribution. Single-frame marginal cues can make some tasks easier than chance.</p></div><div>{diagnostic_parts[1]}<p>Right↔left and up↔down; binary interval/sign labels invert. These are repeated presentations of the same held-out pairs, not extra independent examples.</p></div></div></section>
    <section><h2>Exposure, acquisition and cost</h2>{plan_html}{cost}<p>Persistent state is measured separately from peak training memory. Shared output width does not make these models equal in parameter count, state size or runtime. Frozen PAV parameters are excluded from the trainable counts.</p><details><summary>Fresh exposure by task, terminal and selected checkpoint</summary>{exposure_html}</details><details><summary>Validation trajectory and checkpoint selection</summary>{curve_html}<p>Choose the checkpoint with the highest minimum task balanced accuracy; break ties by mean task AUC. Report terminal exposure separately from selected-checkpoint exposure.</p></details><p>All arms use fresh task-local streams and the same interleaved schedule: half the updates on contours, one twelfth on each other task. Exact run manifests and receipts govern the finite budget; design targets are not completed exposure.</p></section>
    <section><h2>The scientific boundary</h2><p class="lead">Two frames test one temporal transition. Success would establish causal sensory comparison, not persistent working memory or sustained evidence accumulation.</p><p>In particular, two fast/slow traces can retain invertible mixtures of the two projected inputs. That makes the opponent candidate a legitimate online comparator, but its success alone would not prove temporal compression. Cardinal dots do not establish MT pattern-motion selectivity or fitted neuronal correspondence.</p><p>PAV remains the first component of the broader visual attention and working-memory project. Longer streams, delays and distractors are separate future questions.</p><p><a href="README.md">Full experimental specification</a> · <a href="neuroscience_design_notes.md">Opponent model derivation</a> · <a href="focused_checks.json">Focused implementation checks</a></p></section>
    <footer>Rendered {updated}. Saved-data source: <code>{escape(str(source))}</code>. This renderer reads JSON only and does not execute training. No live polling or external libraries are required.</footer></main></body></html>'''
    output.write_text(html,encoding='utf-8')
    print(json.dumps({'output':str(output),'status':status,'completed_arms':n_completed,'source_exists':source.exists()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=HERE/'results_temporal.json')
    parser.add_argument('--output',type=Path,default=HERE/'report_temporal.html')
    args=parser.parse_args()
    render(args.results,args.output)

