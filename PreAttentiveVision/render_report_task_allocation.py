"""Offline scientific report for the task-allocation development experiment.

Reads saved JSON only; does not import or execute models, stimuli or training.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
from render_report import CSS, esc, finite, number, percent, interval, embedded, m
from render_report_multitask import TASKS, TITLES, SHORT, table

HERE = Path(__file__).resolve().parent
ARMS = ['uniform', 'contour_focus']
LABELS = {'uniform': 'Uniform continuation', 'contour_focus': 'Contour emphasis'}
COLORS = {'uniform': '#776494', 'contour_focus': '#448c80'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def records(data):
    return {r.get('arm', r.get('model')): r for r in data.get('runs', [])}


def complete(run):
    test = run.get('test') or {}
    return test.get('status') == 'completed' and all(finite(test.get('tasks', {}).get(t, {}).get('overall', {}).get('balanced_accuracy')) for t in TASKS)


def score(run, task):
    return run['test']['tasks'][task]['overall'] if complete(run) else {}


def paired_tasks(paired):
    return next((c.get('tasks', {}) for c in paired.get('comparisons', []) if c.get('model', c.get('arm')) == ARMS[1] and c.get('reference') == ARMS[0]), {})


def pp(value):
    return f'{100 * value:+.2f} pp' if finite(value) else 'Pending'


def ci(value, pct=False):
    if not isinstance(value, list) or len(value) != 2 or not all(finite(v) for v in value): return 'Pending'
    return '[' + ', '.join(f'{v * (100 if pct else 1):+.3f}' for v in value) + ']' + (' pp' if pct else '')


def local_link(path, title):
    return f'<a href="{esc(path)}">{title}</a>' if (HERE / path).exists() else f'<span class="muted">{title} · pending</span>'


def exposure_rows(data):
    runs = records(data)
    rows = []
    for task, title in zip(TASKS, TITLES):
        row = [title]
        for arm in ARMS:
            r = runs.get(arm, {})
            # Explicit reported counts only. Planned counts are shown separately.
            actual = r.get('per_task_additional_updates', {}).get(task)
            selected = r.get('selected_per_task_additional_updates', {}).get(task)
            actual = actual * 32 if finite(actual) else None
            selected = selected * 32 if finite(selected) else None
            row.append(f'{int(actual):,}' if finite(actual) else 'Pending')
            row.append(f'{int(selected):,}' if finite(selected) else 'Pending')
        rows.append(row)
    return table(['Task', 'Uniform · terminal pairs', 'Uniform · selected pairs', 'Focused · terminal pairs', 'Focused · selected pairs'], rows)


def plots(data, paired):
    runs = records(data)
    curves = [a for a in ARMS if runs.get(a, {}).get('val_curve')]
    tested = [a for a in ARMS if complete(runs.get(a, {}))]
    if not curves and not tested: return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'text.color': '#293145', 'axes.labelcolor': '#41485d',
        'axes.edgecolor': '#c9ccd5', 'figure.facecolor': '#fffefa', 'axes.facecolor': '#fffefa', 'savefig.facecolor': '#fffefa'})
    folder = HERE / 'report_task_allocation_assets'; folder.mkdir(exist_ok=True)
    result = {}
    def save(fig, key):
        fig.savefig(folder / f'{key}.png', dpi=180, bbox_inches='tight')
        fig.savefig(folder / f'{key}.svg', bbox_inches='tight')
        result[key] = embedded(folder / f'{key}.png'); plt.close(fig)
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        for arm in curves:
            curve = runs[arm]['val_curve']; x = [v['step'] - 1512 for v in curve]
            task_values = [{t: v.get('tasks', {}).get(t, {}).get('overall', {}).get('balanced_accuracy') for t in TASKS} for v in curve]
            series = [[v['contour'] for v in task_values], [min(v.values()) if all(finite(z) for z in v.values()) else None for v in task_values], [min(v[t] for t in TASKS if t != 'contour') if all(finite(z) for z in v.values()) else None for v in task_values]]
            for ax, ys in zip(axes, series):
                valid = [(a, b) for a, b in zip(x, ys) if finite(b)]
                if valid: ax.plot(*zip(*valid), '-o', ms=4, color=COLORS[arm], label=LABELS[arm])
        for ax, title in zip(axes, ['Contour validation BA', 'Selection: lowest task BA', 'Retention: lowest other-task BA']):
            ax.set(title=title, xlabel='Additional optimizer updates', ylim=(0, 1.03))
            ax.axhline(.95, color='#a2a7b3', lw=1, ls=':'); ax.grid(alpha=.13)
            horizon = data.get('config', {}).get('additional_updates')
            if finite(horizon): ax.set_xlim(0, horizon * 1.04); ax.set_xticks([0, horizon/3, 2*horizon/3, horizon])
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, bbox_to_anchor=(.5, -.05), fontsize=9)
        fig.tight_layout(); save(fig, 'validation_curves')
    if tested:
        values = np.array([[score(runs[a], t)['balanced_accuracy'] for t in TASKS] for a in tested])
        chance = np.array([.25] + [.5]*6); norm = (values - chance) / (1 - chance)
        fig, ax = plt.subplots(figsize=(11.2, 3))
        im = ax.imshow(norm, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        ax.set(xticks=range(7), xticklabels=[f'{t}\nchance {c:.0%}' for t,c in zip(SHORT,chance)], yticks=range(len(tested)), yticklabels=[LABELS[a] for a in tested]); ax.tick_params(length=0, pad=10)
        for i in range(len(tested)):
            for j in range(7): ax.text(j,i,f'{values[i,j]:.1%}',ha='center',va='center',color='white' if norm[i,j]>.65 else '#25334c')
        fig.colorbar(im, ax=ax, fraction=.026, pad=.025, label='Color: chance-normalized BA')
        fig.tight_layout(); save(fig, 'seven_task_accuracy')
    if len(tested) == 2:
        effects = paired_tasks(paired)
        fig, axes = plt.subplots(1, 2, figsize=(10.7, 4))
        for i, task in enumerate(TASKS):
            d = score(runs[ARMS[1]], task)['balanced_accuracy'] - score(runs[ARMS[0]], task)['balanced_accuracy']
            bar = effects.get(task, {}).get('delta_ba_ci95')
            if isinstance(bar, list): axes[0].plot([100*v for v in bar], [i,i], color=COLORS[ARMS[1]], lw=2)
            axes[0].scatter(100*d, i, color=COLORS[ARMS[1]], s=35)
        axes[0].set(yticks=range(7), yticklabels=SHORT, xlabel='Focused − uniform BA (percentage points)', title='Shared-example task differences')
        axes[0].invert_yaxis(); axes[0].axvline(0,color='#aaaeba',lw=1); axes[0].grid(axis='x',alpha=.13)
        for arm in ARMS:
            diff = runs[arm]['test']['tasks']['contour'].get('difficulty', {})
            ys = [diff.get(f'jitter_{j}deg', {}).get('balanced_accuracy') for j in [2,8,16]]
            if all(finite(v) for v in ys): axes[1].plot([2,8,16], ys,'-o',color=COLORS[arm],label=LABELS[arm],ms=5)
        axes[1].set(xticks=[2,8,16],xlabel='Generating contour jitter (degrees)',ylabel='Balanced accuracy',ylim=(0,1.03),title='Contour difficulty remains visible')
        axes[1].axhline(.5,color='#aaaeba',ls=':',lw=1);axes[1].legend(fontsize=9);axes[1].grid(alpha=.13)
        fig.tight_layout(); save(fig, 'paired_effects_and_jitter')
    return result


def result_tables(data, paired):
    runs = records(data); both = all(complete(runs.get(a, {})) for a in ARMS)
    out = ''
    if both:
        qualifiers = [LABELS[a] for a in ARMS if min(score(runs[a], t)['balanced_accuracy'] for t in TASKS) >= .95]
        summary = ', '.join(qualifiers) + ' reached the 95% point-estimate target on every task.' if qualifiers else 'Neither allocation reached 95% balanced accuracy on every task.'
        delta = score(runs[ARMS[1]], 'contour')['balanced_accuracy'] - score(runs[ARMS[0]], 'contour')['balanced_accuracy']
        out += f'<div class="finding"><h3>{summary}</h3><p>Focused minus uniform contour balanced accuracy: <strong>{pp(delta)}</strong>, paired 95% interval {ci(paired_tasks(paired).get("contour", {}).get("delta_ba_ci95"), True)}. The remaining task scores below determine whether that gain comes with a retention cost.</p></div>'
        contour = score(runs[ARMS[1]], 'contour')
        contrast = paired_tasks(paired).get('contrast', {})
        out += f'<p class="callout">Focused contour BA is {percent(contour["balanced_accuracy"])} with a supplied 95% interval of {interval(contour.get("balanced_accuracy_ci95"), True)}. That interval extends below the 95% target: the sample clears the point-estimate screen, not a 95% population guarantee. Contrast changes by {pp(contrast.get("delta_ba"))} versus uniform, paired 95% interval {ci(contrast.get("delta_ba_ci95"), True)}; the improved allocation has a small measured retention cost.</p>'
    else: out += '<div class="pending-chart"><span>Allocation outcome pending.</span><p>Final task judgments require both completed development evaluations. Curves from validation are labeled separately.</p></div>'
    rows = []
    for task, title in zip(TASKS, TITLES):
        rows.append([title + ('<small>Chance BA 25%</small>' if task == 'motion_direction' else '<small>Chance BA 50%</small>')] + [percent(score(runs.get(a, {}), task).get('balanced_accuracy')) + '<small>AUC ' + number(score(runs.get(a, {}), task).get('macro_ovr_auc'), 4) + '</small>' for a in ARMS])
    out += table(['Development test task', *[LABELS[a] for a in ARMS]], rows)
    if both:
        rows = []
        for task, title in zip(TASKS, TITLES):
            a, b = score(runs[ARMS[1]], task), score(runs[ARMS[0]], task); p = paired_tasks(paired).get(task, {})
            rows.append([title, pp(a['balanced_accuracy']-b['balanced_accuracy']), ci(p.get('delta_ba_ci95'),True), f"{a['macro_ovr_auc']-b['macro_ovr_auc']:+.4f}", ci(p.get('delta_auc_ci95'))])
        out += '<details><summary>All seven paired changes and intervals</summary>' + table(['Task','ΔBA','Paired 95% interval','ΔAUC','Paired 95% interval'],rows) + '</details>'
        comparison = next((c for c in paired.get('comparisons', []) if c.get('model') == ARMS[1] and c.get('reference') == ARMS[0]), {})
        strata = comparison.get('contour_difficulty', {})
        if strata:
            rows = []
            for name, degrees in [('jitter_2deg',2),('jitter_8deg',8),('jitter_16deg',16)]:
                s = strata.get(name,{})
                rows.append([f'{degrees}° jitter',str(s.get('n','—')),pp(s.get('delta_ba')),ci(s.get('delta_ba_ci95'),True),number(s.get('delta_auc'),4),ci(s.get('delta_auc_ci95'))])
            out += '<details><summary>Paired contour effects within each jitter condition</summary>' + table(['Generator condition','Pairs','ΔBA','Paired 95% interval','ΔAUC','Paired 95% interval'],rows) + '</details>'
    for arm in ARMS:
        if not complete(runs.get(arm, {})): continue
        rows = [[title, str(score(runs[arm],t).get('n','—')), percent(score(runs[arm],t).get('balanced_accuracy')), interval(score(runs[arm],t).get('balanced_accuracy_ci95'),True), number(score(runs[arm],t).get('macro_ovr_auc'),4), interval(score(runs[arm],t).get('macro_ovr_auc_ci95'))] for t,title in zip(TASKS,TITLES)]
        difficulty = runs[arm]['test']['tasks']['contour'].get('difficulty', {})
        detailrows = [[f'{j}° jitter', str(difficulty.get(k,{}).get('n','—')),percent(difficulty.get(k,{}).get('balanced_accuracy')),number(difficulty.get(k,{}).get('macro_ovr_auc'),4)] for k,j in [('jitter_2deg',2),('jitter_8deg',8),('jitter_16deg',16)]]
        out += f'<details><summary>{LABELS[arm]}: uncertainty and contour jitter</summary>' + table(['Task','Pairs','BA','95% interval','AUC','95% interval'],rows) + '<h4 style="margin-top:22px">Contour difficulty point estimates</h4>' + table(['Generator condition','Pairs','BA','AUC'],detailrows) + '</details>'
    return out


def render(path, paired_path, output):
    data, paired = load(path), load(paired_path)
    cfg = data.get('config', {}); runs = records(data)
    count = sum(complete(runs.get(a, {})) for a in ARMS)
    status = esc(data.get('status','pending')); now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    horizon = cfg.get('additional_updates'); total = horizon * 32 if finite(horizon) else None
    shown_horizon = f'{horizon:,}' if finite(horizon) else 'Pending'
    shown_total = f'{total:,}' if finite(total) else 'Pending'
    planned = table(['Allocation','Contour share','Each other task','Contour pairs','Pairs per other task'], [[LABELS['uniform'],'1/7','1/7',f'{total//7:,}' if total else 'Pending',f'{total//7:,}' if total else 'Pending'],[LABELS['contour_focus'],'1/2','1/12',f'{total//2:,}' if total else 'Pending',f'{total//12:,}' if total else 'Pending']])
    figures = plots(data,paired); figure_html = ''
    for key,title,caption in [('seven_task_accuracy','Seven tasks, each still accountable.','Labels show actual balanced accuracy. Color accounts for four-way motion versus binary tasks. All tasks use fixed argmax decisions.'),('paired_effects_and_jitter','Tradeoffs and difficulty, exposed.','Difference bars use supplied paired intervals, conditional on trained checkpoints. Jitter values describe the stimulus generator; difficulty curves are point estimates.'),('validation_curves','Practice, learning and retention.','Three planned validation looks. The dotted 95% line is the engineering target, not a confidence bound. No training-seed replication is implied.')]:
        if key in figures: figure_html += f'<figure class="plot"><h3>{title}</h3><img src="{figures[key]}" alt="{title}"><figcaption>{caption} <a href="report_task_allocation_assets/{key}.svg">SVG</a> · <a href="report_task_allocation_assets/{key}.png">PNG</a></figcaption></figure>'
    sampling = m('<msub><mi>p</mi><mi>t</mi></msub><mo>=</mo><mfrac><mn>1</mn><mn>7</mn></mfrac><mspace width="2em"/><mtext>or</mtext><mspace width="2em"/><msub><mi>p</mi><mi>t</mi></msub><mo>=</mo><mo>{</mo><mtable columnalign="left"><mtr><mtd><mfrac><mn>1</mn><mn>2</mn></mfrac></mtd><mtd><mtext>contour</mtext></mtd></mtr><mtr><mtd><mfrac><mn>1</mn><mn>12</mn></mfrac></mtd><mtd><mtext>each other task</mtext></mtd></mtr></mtable>')
    exposure = m('<msub><mi>N</mi><mi>t</mi></msub><mo>=</mo><mn>32</mn><mi>U</mi><msub><mi>p</mi><mi>t</mi></msub><mo>,</mo><mspace width="1em"/><mi>U</mi><mo>∈</mo><mn>84</mn><msub><mi>ℕ</mi><mo>+</mo></msub>')
    selection = m('<msup><mi>s</mi><mo>∗</mo></msup><mo>=</mo><msub><mtext>arg max</mtext><mi>s</mi></msub><mo>(</mo><msub><mtext>min</mtext><mi>t</mi></msub><mtext>BA</mtext><mo>(</mo><mi>s</mi><mo>,</mo><mi>t</mi><mo>)</mo><mo>,</mo><mspace width=".5em"/><mfrac><mn>1</mn><mn>7</mn></mfrac><munder><mo>∑</mo><mi>t</mi></munder><mtext>AUC</mtext><mo>(</mo><mi>s</mi><mo>,</mo><mi>t</mi><mo>)</mo><mo>)</mo>')
    runtime_rows = [[LABELS[a],str(runs.get(a,{}).get('step','Pending')),str(runs.get(a,{}).get('best_step','Pending')),number(runs.get(a,{}).get('training_seconds'),1)] for a in ARMS]
    raw = local_link(path.name,'Saved allocation results') + ' · ' + local_link(paired_path.name,'Paired comparison analysis')
    if data.get('run_root'):
        try: relative = Path(data['run_root']).relative_to(HERE).as_posix()
        except ValueError: relative = ''
        if relative: raw += ' · ' + local_link(relative+'/aggregate.json','Run snapshot') + ' · ' + local_link(relative+'/exit.json','Exit receipt')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Practice the weak skill · PAV task allocation</title><style>{CSS}
.wrap,.hero-grid>*,.two>*,.section-head>*{{min-width:0}}.hero-grid{{grid-template-columns:1.2fr 1fr;gap:45px}}.hero h1{{font-size:clamp(65px,7vw,94px)}}.practice{{background:#eeebf2;padding:27px;margin:25px 0}}.practice h3{{font-size:24px}}.batch-strip{{display:flex;height:25px;gap:3px;margin:12px 0 7px}}.batch-strip i{{flex:1;background:#b8afc8}}.batch-strip i.contour{{background:#448c80}}.practice .small{{color:var(--muted);font-size:11px}}.practice hr{{border:0;border-top:1px solid #d2cbdc;margin:25px 0}}.facts strong{{font-size:31px}}@media(max-width:620px){{.hero-grid{{grid-template-columns:1fr}}.hero h1{{font-size:67px;letter-spacing:-3px}}.equation{{font-size:16px}}}}
</style></head><body><a class="skip" href="#main">Skip to report</a><header><div class="wrap nav"><a class="brand" href="#">PAV / TASK ALLOCATION</a><nav><a href="#results">Results</a><a href="#allocation">Allocation</a><a href="#lineage">Lineage</a><a href="#meaning">Meaning</a></nav></div></header><main id="main"><section class="hero wrap"><div class="eyebrow">A training intervention · architecture held fixed</div><div class="hero-grid"><div><h1>Practice the<br><em>weak skill.</em><br>Keep the rest.</h1><p class="dek">Can more contour experience improve a shared representation without losing its other sensory skills?</p><p class="hero-copy">Both arms continue the same trained ConvNeXt + late-SE model. The intervention is how often each task supplies the next optimizer update.</p></div><div><div class="status"><span class="dot"></span>{status} · {count}/2 development tests saved<small>Snapshot {now} · no automatic refresh</small></div><div class="practice"><h3>Uniform continuation</h3><div class="batch-strip" aria-label="One contour batch in seven"><i class="contour"></i>{'<i></i>'*6}</div><p class="small">One contour batch + one from each other task</p><hr><h3>Contour emphasis</h3><div class="batch-strip" aria-label="Six contour batches and six other batches in twelve">{'<i class="contour"></i>'*6}{'<i></i>'*6}</div><p class="small">Six contour batches + one from each other task</p></div><p class="small muted">Green represents contour allocation. Strips illustrate proportions, not the exact presentation order.</p></div></div><div class="scope"><b>Same model.<br>Different practice.</b><span>Equal total updates deliberately produce unequal per-task exposure. Focused training receives 3.5× as many contour examples and 7/12 as many examples from each other task. That difference is the experimental intervention.</span></div></section>
<section class="section wrap" id="results"><div class="section-head"><span class="index">01</span><div><div class="eyebrow">Development results</div><h2>The weakest task sets the challenge.</h2><p class="intro">The ambitious point-estimate target is at least 95% balanced accuracy on every current task, including aggregate contour grouping. A useful partial result would be a contour gain over uniform continuation with strong retention across the other six tasks.</p></div></div>{result_tables(data,paired)}{figure_html}<p class="caption">{raw}. BA is mean class recall; AUC measures score ranking. Chance AUC is 0.5 for every task. Neither metric demonstrates population certainty at a perfect empirical score.</p></section>
<section class="section wrap" id="allocation"><div class="section-head"><span class="index">02</span><div><div class="eyebrow">The intervention is task sampling</div><h2>Spend updates where learning<br>has furthest to go.</h2><p class="intro">At the previous endpoint, the SE parent performed strongly on six task distributions while contour performance remained much lower and its validation trajectory was still rising. That motivates a finite allocation test, without assuming a plateau or a broken architecture.</p></div></div><div class="two"><div><h3>Two fixed schedules.</h3><div class="equation">{sampling}</div><p>The uniform schedule completes one batch from every task in seven updates. The focused schedule completes six contour batches and one batch from each other task in twelve updates. Both use 32 examples per batch and the same optimizer settings.</p><p class="small">This changes the sequence and frequency of task gradients. It is not equivalent to multiplying a static contour loss weight: Adam’s moments, parameter trajectory and the number of distinct training examples also change.</p></div><div><h3>Exposure is counted explicitly.</h3><div class="equation">{exposure}</div><p>U denotes new optimizer updates and Nₜ the new pairs for task t. Multiples of 84 complete both schedule periods. The initially proposed target was 1,008 updates per arm; the executed allocation is read from the saved run, after checking that profiling and evaluation fit the remaining allowance.</p><p class="small">Corresponding task-local draws match across arms up to the shorter per-task prefix. The focused arm explores further into the contour stream and less far into the other streams. These are paired draws, not independent data pools.</p></div></div><div class="facts"><div><strong>1,512</strong><span>Shared trained-parent step</span></div><div><strong>{shown_horizon}</strong><span>Planned new updates per arm</span></div><div><strong>{shown_total}</strong><span>Planned new pairs per arm</span></div><div><strong>2</strong><span>Schedules · one shared architecture</span></div></div>{planned}<h3 style="margin-top:35px">Actual exposure, including the selected checkpoint.</h3>{exposure_rows(data)}<p class="caption">Terminal exposure and selected-checkpoint exposure can differ when an earlier validation checkpoint is selected. Counts above are populated only from explicit saved training records; proposed counts are not relabeled as completed work.</p></section>
<section class="section wrap" id="lineage"><div class="section-head"><span class="index">03</span><div><div class="eyebrow">Matched parent, fresh task-local streams</div><h2>Preserve learned state.<br>Change the sampling contract openly.</h2></div></div><div class="two"><div><h3>Identical architectural starting point.</h3><p>Both arms begin with the late-SE step-1512 checkpoint, retaining learned encoder and ordered-decoder weights and compatible Adam state. They use the same seven heads and the same two-frame stimulus laws. There is no new Gabor branch, decoder redesign or architecture sweep in this experiment.</p><p>The new schedules use shared, freshly initialized task-specific streams. Because presentation order changes, this is not bit-for-bit continuation of the earlier single input stream. The sampler change is explicit and common to both arms.</p><p class="small">The parent’s late gate is image-dependent channel modulation of its 13 × 13 feature field. It is shared across frames and tasks. The preceding <a href="report_hybrids.html">hybrid report</a> explains its identity initialization, equations and measured tradeoffs.</p></div><div><h3>Choose for the weakest task.</h3><div class="equation">{selection}</div><p>The ordered pair is maximized lexicographically: highest minimum task validation BA first, then highest equal-task mean AUC. Three planned validation looks assess learning. The minimum uses raw BA as a practical criterion, despite different chance levels and task difficulty.</p><p>Final fixed-argmax scores are reported per task on fresh shared development pairs. Contour difficulty is broken out at 2°, 8° and 16° generating jitter. Natural-image crops reuse the existing BSDS source-photo pool; fresh crops do not create independent new photographs.</p></div></div><details><summary>Stimulus definitions and the retained ordered comparison</summary><p>The battery asks for four cardinal motion directions and six binary comparisons: signed orientation, contrast, spatial frequency, chromatic increment, contour grouping and natural spectral detail. Each encoder sees two separate 100 × 100 RGB frames through shared weights.</p><p>Signed feature differences and directed local correlations preserve order in the common decoder. The contour generator compares orientation assignments over matched element positions; accidental partial contours can occur in controls. The natural task changes radial spectral weighting while preserving Fourier phase and matching contrast.</p><p><a href="sensory_battery_parameters.md">Exact numerical stimuli</a> · <a href="two_frame_task_research.md">Primary task rationale</a> · <a href="decoder_multitask.py">Common ordered decoder</a> · <a href="sensory_battery_examples.png">Actual generator examples</a></p></details>{table(['Arm','Terminal total step','Selected total step','Training worker wall · s'],runtime_rows)}<p class="caption">Saved experiment wall time: {number(data.get('wall_seconds'),1)} seconds. Worker times include process/checkpoint overhead; they are not isolated GPU kernel timings. The finite remaining compute allowance includes profiling and evaluation and is not renewed by this report.</p></section>
<section class="section wrap" id="meaning"><div class="section-head"><span class="index">04</span><div><div class="eyebrow">Interpretation</div><h2>A better allocation is useful.<br>Its mechanism remains a question.</h2></div></div><div class="grounding-panel"><div><span class="pill">What this can show</span><h3>Whether contour practice earns its cost.</h3><p>A contour gain with strong retained performance supports this training allocation for the current model and distributions. Similar improvement in both arms would favor continued training as the sufficient explanation. A gain with losses elsewhere reveals an empirical specialization tradeoff.</p><p>If performance remains low, the validation trajectory still matters. A short finite run cannot establish saturation or that contour information is absent from the encoder.</p></div><div><span class="pill">What this does not establish</span><h3>Gradient conflict, biology or universal competence.</h3><p>This fixed-sampling experiment does not measure gradient conflict or reproduce an adaptive balancing method. It does not identify a biological attention process, working-memory mechanism or contour-binding circuit.</p><p>The broader project uses Guided Search 6.0 as a functional scaffold, omits its diffuser and approximates activated long-term memory as synaptic weights by the user’s choice. PAV remains the first component; later mechanisms are still separate work.</p></div></div><div class="three limitations"><div><h4>Paired sampling uncertainty</h4><p>Intervals compare predictions on shared pairs. Procedural trials are resampled as pairs; natural-image uncertainty uses source-photo clusters. Intervals condition on the trained models, not a population of training seeds.</p></div><div><h4>Development evidence</h4><p>The earlier results motivated the allocation. New procedural and crop draws provide fresh examples, while reused BSDS source pools limit claims of a fully untouched natural-image test.</p></div><div><h4>Point target, not certainty</h4><p>Reaching 95% on every task is an ambitious engineering screen. It does not certify 95% population performance; uncertainty, jitter dependence and score-ranking changes remain visible.</p></div></div></section>
<section class="closing wrap"><div class="eyebrow">Sources and artifacts</div><h2>Keep the experiment<br>easy to read and reuse.</h2><p class="sources"><a href="next_experiment_task_allocation.md">Approved allocation rationale</a> · <a href="https://proceedings.mlr.press/v80/chen18a.html">Chen et al. · GradNorm (related balancing research)</a> · <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/">Wolfe · Guided Search 6.0</a> · <a href="report_hybrids.html">Prior hybrid report</a> · {raw}</p><p class="small">This standalone HTML embeds scientific figures and native MathML without external scripts, fonts or CDNs. Linked source files and exported charts are companion artifacts. Status and metrics are a dated snapshot.</p></section></main><footer class="wrap"><span>PAV task-allocation experiment · {now}<br>Saved status: {status} · previous reports preserved</span><button type="button" onclick="window.print()">Print / save PDF</button></footer></body></html>'''
    html = html.replace('Three planned validation looks assess learning.', 'Three planned validation looks assess learning; exact ties on both scores select the earliest checkpoint.')
    html = html.replace('compatible Adam state.', 'all 100 populated compatible Adam parameter states.')
    html = html.replace('<a href="next_experiment_task_allocation.md">Approved allocation rationale</a>', '<a href="next_experiment_task_allocation.md">Approved allocation rationale</a> · <a href="allocation_sampler.py">Task-local streams and schedules</a> · <a href="train_allocation.py">Continuation worker</a> · <a href="sweep_allocation.py">Allocation supervisor</a> · <a href="analyze_allocation.py">Paired analysis</a>')
    output.write_text(html,encoding='utf-8')
    return dict(output=str(output),status=data.get('status','pending'),completed_tests=count,figures=list(figures),mathml=html.count('<math '),bytes=output.stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results_allocation.json')
    parser.add_argument('--paired',type=Path,default=HERE/'results_allocation_analysis.json')
    parser.add_argument('--output',type=Path,default=HERE/'report_task_allocation.html')
    args = parser.parse_args()
    print(json.dumps(render(args.results,args.paired,args.output),indent=2))
