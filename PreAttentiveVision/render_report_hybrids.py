"""Render the trained-parent hybrid development comparison, with offline math.

CPU document work only. No model imports, training or old-score substitution.
Usage: python -B PreAttentiveVision/render_report_hybrids.py
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
from render_report import CSS, esc, finite, number, percent, interval, embedded, m
from render_report_multitask import TASKS, TITLES, SHORT, table

HERE = Path(__file__).resolve().parent
ARMS = ['convnext_grn', 'convnext_gabor_residual', 'convnext_se_residual']
LABELS = dict(zip(ARMS, ['Continued ConvNeXt', '+ Gabor side branch', '+ Late SE gate']))
COLORS = dict(zip(ARMS, ['#776494', '#448c80', '#c3834b']))
ADDED = dict(zip(ARMS, [0, 1296, 4728]))


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def runs_by_name(data):
    return {r['model']: r for r in data.get('runs', []) if r.get('model') in ARMS}


def scored(run):
    test = run.get('test') or {}
    return test.get('status') == 'completed' and all(finite(test.get('tasks', {}).get(t, {}).get('overall', {}).get('balanced_accuracy')) for t in TASKS)


def stats(run, task):
    return ((run.get('test') or {}).get('tasks', {}).get(task, {}).get('overall', {})) if scored(run) else {}


def pp(value):
    return f'{value * 100:+.2f} pp' if finite(value) else 'Pending'


def ci_pp(value):
    return f'[{100 * value[0]:+.2f}, {100 * value[1]:+.2f}] pp' if isinstance(value, list) and len(value) == 2 and all(finite(v) for v in value) else 'Paired interval pending'


def link(path, label):
    return f'<a href="{esc(path)}">{label}</a>' if (HERE / path).exists() else f'<span class="muted">{label} · pending</span>'


def paired_stat(paired, arm, task):
    # The researcher supplies paired estimates. Never subtract marginal CIs.
    match = next((c for c in paired.get('comparisons', []) if c.get('model') == arm and c.get('reference') == ARMS[0]), {})
    return match.get('tasks', {}).get(task, {})


def make_figures(data, paired):
    runs = runs_by_name(data)
    tested = [a for a in ARMS if scored(runs.get(a, {}))]
    curves = [a for a in ARMS if runs.get(a, {}).get('val_curve')]
    if not tested and not curves:
        return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
        'text.color': '#293145', 'axes.labelcolor': '#41485d', 'axes.edgecolor': '#c9ccd5',
        'figure.facecolor': '#fffefa', 'axes.facecolor': '#fffefa', 'savefig.facecolor': '#fffefa'})
    folder = HERE / 'report_hybrids_assets'
    folder.mkdir(exist_ok=True)
    figures = {}
    def save(fig, key):
        fig.savefig(folder / f'{key}.png', dpi=180, bbox_inches='tight')
        fig.savefig(folder / f'{key}.svg', bbox_inches='tight')
        figures[key] = embedded(folder / f'{key}.png')
        plt.close(fig)
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for arm in curves:
            curve = runs[arm]['val_curve']
            x = [v['step'] - 756 for v in curve]
            for ax, task in zip(axes[:2], ['contour', 'motion_direction']):
                points = [v.get('tasks', {}).get(task, {}).get('overall', {}).get('balanced_accuracy') for v in curve]
                valid = [(a, b) for a, b in zip(x, points) if finite(b)]
                if valid: ax.plot(*zip(*valid), '-o', color=COLORS[arm], label=LABELS[arm], ms=4)
            axes[2].plot(x, [v['macro_ovr_auc'] for v in curve], '-o', color=COLORS[arm], label=LABELS[arm], ms=4)
        for ax, title in zip(axes, ['Contour: validation BA', 'Motion: validation BA', 'Selection: equal-task AUC']):
            ax.set(title=title, xlabel='Additional updates after parent step 756', ylim=(0, 1.03))
            horizon = data.get('config', {}).get('additional_updates')
            if finite(horizon) and horizon > 0:
                ax.set_xlim(0, horizon * 1.04)
                ax.set_xticks([0, horizon / 3, 2 * horizon / 3, horizon])
            ax.grid(alpha=.15)
        axes[0].axhline(.5, color='#bbb6c7', ls=':', lw=1)
        axes[1].axhline(.25, color='#bbb6c7', ls=':', lw=1)
        axes[2].axhline(.5, color='#bbb6c7', ls=':', lw=1)
        fig.legend(*axes[2].get_legend_handles_labels(), loc='lower center', ncol=3, bbox_to_anchor=(.5, -.07), fontsize=9)
        fig.tight_layout(); save(fig, 'validation_curves')
    if tested:
        values = np.array([[stats(runs[a], t)['balanced_accuracy'] for t in TASKS] for a in tested])
        chance = np.array([.25] + [.5] * 6)
        normalized = (values - chance) / (1 - chance)
        fig, ax = plt.subplots(figsize=(11.4, 3.4))
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        ax.set(xticks=range(7), xticklabels=[f'{t}\nchance {c:.0%}' for t, c in zip(SHORT, chance)], yticks=range(len(tested)), yticklabels=[LABELS[a] for a in tested])
        ax.tick_params(length=0, pad=10)
        for i in range(len(tested)):
            for j in range(7): ax.text(j, i, f'{values[i,j]:.1%}', ha='center', va='center', color='white' if normalized[i,j] > .65 else '#25334c')
        fig.colorbar(im, ax=ax, fraction=.026, pad=.025, label='Color: chance-normalized BA')
        fig.tight_layout(); save(fig, 'seven_task_accuracy')
    if all(a in tested for a in ARMS):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), sharey=True)
        for ax, arm in zip(axes, ARMS[1:]):
            delta = [stats(runs[arm], t)['balanced_accuracy'] - stats(runs[ARMS[0]], t)['balanced_accuracy'] for t in TASKS]
            ax.axvline(0, color='#aeb2bc', lw=1)
            for i, (task, value) in enumerate(zip(TASKS, delta)):
                ci = paired_stat(paired, arm, task).get('delta_ba_ci95')
                if isinstance(ci, list) and len(ci) == 2: ax.plot([100*c for c in ci], [i, i], color=COLORS[arm], lw=2)
                ax.scatter(value * 100, i, color=COLORS[arm], s=38)
            ax.set(title=LABELS[arm], yticks=range(7), yticklabels=SHORT, xlabel='BA change vs continued ConvNeXt (pp)')
            ax.grid(axis='x', alpha=.15)
        axes[0].invert_yaxis()
        fig.tight_layout(); save(fig, 'paired_task_deltas')
    return figures


def results_html(data, paired):
    runs = runs_by_name(data)
    all_done = all(scored(runs.get(a, {})) for a in ARMS)
    out = ''
    if all_done:
        passed_names = []
        for arm in ARMS[1:]:
            contour_delta = stats(runs[arm], 'contour')['balanced_accuracy'] - stats(runs[ARMS[0]], 'contour')['balanced_accuracy']
            motion_delta = stats(runs[arm], 'motion_direction')['balanced_accuracy'] - stats(runs[ARMS[0]], 'motion_direction')['balanced_accuracy']
            if contour_delta >= .03 and motion_delta >= -.02: passed_names.append(LABELS[arm])
        headline = (' and '.join(passed_names) + ' met the preset point-estimate screen.') if passed_names else 'Neither addition met the preset contour-improvement screen.'
        base_motion_auc = stats(runs[ARMS[0]], 'motion_direction')['macro_ovr_auc']
        out += f'<div class="finding"><h3>{headline}</h3><p>Motion balanced accuracy improved with both additions in this saved comparison, but the continued control already had motion AUC {base_motion_auc:.6f}. Near-perfect ranking alongside lower argmax accuracy points to a decision-score distinction; it does not establish that the additions uniquely supplied absent motion information.</p><p>The late SE arm is a useful subsequent development starting point for its strong six-task profile, while its contour result did not improve over the continued baseline. Selecting it for further exploration is separate from passing this experiment’s primary criterion.</p></div>'
        rows = []
        for arm in ARMS[1:]:
            d = {t: stats(runs[arm], t)['balanced_accuracy'] - stats(runs[ARMS[0]], t)['balanced_accuracy'] for t in ['contour', 'motion_direction']}
            passed = d['contour'] >= .03 and d['motion_direction'] >= -.02
            rows.append([LABELS[arm], pp(d['contour']), ci_pp(paired_stat(paired, arm, 'contour').get('delta_ba_ci95')), pp(d['motion_direction']), ci_pp(paired_stat(paired, arm, 'motion_direction').get('delta_ba_ci95')), 'Meets point-estimate screen' if passed else 'Does not meet screen'])
        out += table(['Addition', 'Contour ΔBA', 'Paired 95% interval', 'Motion ΔBA', 'Paired 95% interval', 'Preset practical criterion'], rows)
        out += '<p class="caption">The screen is based on point estimates. Intervals describe sampling uncertainty conditional on these trained checkpoints; they do not turn the −2 pp motion margin into a formal noninferiority test.</p>'
    else:
        out += '<div class="pending-chart"><span>Hybrid outcome pending.</span><p>The comparison needs completed evaluations for the continued baseline and both additions. Validation curves are development measurements, not final task results.</p></div>'
    rows = []
    for task, title in zip(TASKS, TITLES):
        rows.append([title + ('<small>Chance BA 25%; AUC 0.5</small>' if task == 'motion_direction' else '<small>Chance BA 50%; AUC 0.5</small>')] + [percent(stats(runs.get(a, {}), task).get('balanced_accuracy')) + '<small>AUC ' + number(stats(runs.get(a, {}), task).get('macro_ovr_auc')) + '</small>' for a in ARMS])
    out += '<h3 style="margin-top:35px">All seven judgments remain visible.</h3>' + table(['Development test', *[LABELS[a] for a in ARMS]], rows)
    details = []
    for arm in ARMS:
        if not scored(runs.get(arm, {})): continue
        rows = []
        for task, title in zip(TASKS, TITLES):
            s = stats(runs[arm], task)
            rows.append([title, str(s.get('n', '—')), percent(s.get('balanced_accuracy')), interval(s.get('balanced_accuracy_ci95'), True), number(s.get('macro_ovr_auc')), interval(s.get('macro_ovr_auc_ci95')), str(s.get('unique_base_groups', '—'))])
        details.append(f'<h4>{LABELS[arm]}</h4>' + table(['Task', 'Pairs', 'BA', '95% interval', 'AUC', '95% interval', 'Base groups'], rows))
    if details: out += '<details><summary>Per-arm uncertainty and evaluation counts</summary>' + ''.join(details) + '</details>'
    if all_done:
        delta_tables = []
        for arm in ARMS[1:]:
            rows = []
            for task, title in zip(TASKS, TITLES):
                a, b = stats(runs[arm], task), stats(runs[ARMS[0]], task)
                pair = paired_stat(paired, arm, task)
                auc_ci = pair.get('delta_auc_ci95')
                auc_text = '[' + ', '.join(f'{v:+.4f}' for v in auc_ci) + ']' if isinstance(auc_ci, list) and len(auc_ci) == 2 else 'Pending'
                rows.append([title, pp(a['balanced_accuracy'] - b['balanced_accuracy']), ci_pp(pair.get('delta_ba_ci95')), f"{a['macro_ovr_auc'] - b['macro_ovr_auc']:+.4f}", auc_text])
            delta_tables.append(f'<h4>{LABELS[arm]} versus continued ConvNeXt</h4>' + table(['Task', 'ΔBA', 'Paired 95% interval', 'ΔAUC', 'Paired 95% interval'], rows))
        out += '<details><summary>All seven paired effects, including score ranking</summary>' + ''.join(delta_tables) + '</details>'
    parent = data.get('parent_reference', {})
    if parent.get('status') == 'completed' and scored(runs.get(ARMS[0], {})):
        rows = []
        for task, title in zip(TASKS, TITLES):
            old = parent.get('tasks', {}).get(task, {}).get('overall', {})
            current = stats(runs[ARMS[0]], task)
            rows.append([title, percent(old.get('balanced_accuracy')), percent(current.get('balanced_accuracy')), pp(current['balanced_accuracy'] - old['balanced_accuracy'])])
        out += '<details><summary>Context: the frozen parent on the same new examples</summary><p>This optional reference isolates the observed benefit of continued training. Hybrid decisions above still use the continued baseline.</p>' + table(['Task', 'Frozen step 756 BA', 'Continued BA', 'Change'], rows) + '</details>'
    return out


def cost_html(data):
    runs = runs_by_name(data)
    rows = []
    for arm in ARMS:
        p = data.get('profiles', {}).get(arm, {})
        r = runs.get(arm, {})
        rows.append([LABELS[arm], f'{ADDED[arm]:,}', f'{262032+ADDED[arm]:,}', str(p.get('decoder_params', 211952)), number(p.get('train_step_median')), number(p.get('eval_batch_mean')), number(p.get('peak_vram_bytes', 0)/(1024**3), 2) if p.get('peak_vram_bytes') else 'Pending', number(r.get('training_seconds'), 1), str(r.get('best_step', 'Pending'))])
    return table(['Arm', 'New trainable', 'Encoder total', 'Common decoder', 'Median train step · s', 'Mean eval batch · s', 'Peak allocated · GiB', 'Training workers · s', 'Selected step'], rows)


def render(path, paired_path, output):
    data = load(path)
    paired = load(paired_path)
    config = data.get('config', {})
    runs = runs_by_name(data)
    count = sum(scored(runs.get(a, {})) for a in ARMS)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    status = esc(data.get('status', 'pending'))
    figs = make_figures(data, paired)
    figure_html = ''
    for key, title, caption in [
        ('paired_task_deltas', 'The addition must beat continued training.', 'Dots are measured BA differences on shared development examples. Bars appear only when paired intervals have been supplied. These are not differences between marginal interval endpoints.'),
        ('seven_task_accuracy', 'A sensory profile, not a pooled accuracy.', 'Text gives balanced accuracy. Color normalizes for four-class motion versus binary tasks. The same evaluation pairs are shared across arms.'),
        ('validation_curves', 'Watch learning without calling it a test result.', 'Three planned validation looks select each checkpoint by equal-task macro one-vs-rest AUC; exact ties choose the earliest checkpoint. The horizontal axis counts additional updates after the trained parent.')]:
        if key in figs: figure_html += f'<figure class="plot"><h3>{title}</h3><img src="{figs[key]}" alt="{title}"><figcaption>{caption} <a href="report_hybrids_assets/{key}.svg">SVG</a> · <a href="report_hybrids_assets/{key}.png">PNG</a></figcaption></figure>'
    gabor = m('<mi>B</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mi>P</mi><mo>(</mo><msub><mtext>LN</mtext><mi>C</mi></msub><mo>(</mo><msub><mo>↓</mo><mn>2</mn></msub><mo>[</mo><mi>h</mi><mo>∗</mo><mi>G</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>]</mo><mo>)</mo><mo>)</mo>')
    residual = m('<msub><mover><mi>H</mi><mo>~</mo></mover><mn>1</mn></msub><mo>=</mo><msub><mi>H</mi><mn>1</mn></msub><mo>+</mo><mi>α</mi><mo>⊙</mo><mi>B</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo><mspace width="1em"/><msub><mi>α</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn>')
    se = m('<mi>z</mi><mo>=</mo><msub><mtext>mean</mtext><mrow><mi>h</mi><mo>,</mo><mi>w</mi></mrow></msub><mo>(</mo><msub><mi>H</mi><mn>3</mn></msub><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>u</mi><mo>=</mo><msub><mi>W</mi><mn>2</mn></msub><mtext>ReLU</mtext><mo>(</mo><msub><mi>W</mi><mn>1</mn></msub><mi>z</mi><mo>+</mo><msub><mi>b</mi><mn>1</mn></msub><mo>)</mo><mo>+</mo><msub><mi>b</mi><mn>2</mn></msub>')
    gate = m('<msub><mover><mi>H</mi><mo>~</mo></mover><mn>3</mn></msub><mo>=</mo><msub><mi>H</mi><mn>3</mn></msub><mo>⊙</mo><mo>[</mo><mn>1</mn><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mtext>tanh</mtext><mo>(</mo><mi>u</mi><mo>)</mo><mo>]</mo>')
    delta = m('<msub><mi>Δ</mi><mi>t</mi></msub><mo>=</mo><msub><mtext>BA</mtext><mrow><mtext>addition</mtext><mo>,</mo><mi>t</mi></mrow></msub><mo>−</mo><msub><mtext>BA</mtext><mrow><mtext>continued</mtext><mo>,</mo><mi>t</mi></mrow></msub>')
    added_updates = f'{config["additional_updates"]:,}' if 'additional_updates' in config else 'Pending'
    added_pairs = f'{config["additional_training_pairs_per_arm"]:,}' if 'additional_training_pairs_per_arm' in config else 'Pending'
    per_task = f'{config["additional_training_pairs_per_task_per_arm"]:,}' if 'additional_training_pairs_per_task_per_arm' in config else 'pending'
    contact = embedded(HERE / 'sensory_battery_examples.png')
    outcome = 'All three development evaluations are saved.' if count == 3 else 'The experiment is awaiting complete development evaluations.'
    archive = ''
    if data.get('run_root'):
        root = Path(data['run_root'])
        try: rel = root.relative_to(HERE).as_posix()
        except ValueError: rel = ''
        if rel: archive = ' · ' + link(rel + '/aggregate.json', 'Run snapshot') + ' · ' + link(rel + '/exit.json', 'Exit receipt')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A scientific development report comparing continued ConvNeXt with identity-initialized Gabor and SE additions on seven two-frame visual tasks."><title>Keep the motion, improve the contour · PAV hybrids</title><style>{CSS}
.wrap,.hero-grid>*,.two>*,.section-head>*{{min-width:0}}.hero-grid{{grid-template-columns:1.25fr 1fr;gap:45px}}.hero h1{{font-size:clamp(60px,7.5vw,100px)}}.hero .dek{{font-size:21px}}.branch-map{{border:1px solid #c8bfd2;background:#eeebf2;padding:25px;margin:25px 0}}.branch-map .node{{padding:10px 12px;border:1px solid #b8aec5;background:#fffefa;text-align:center;font-size:12px}}.branch-map .arrow{{text-align:center;color:#9383a6;font-size:22px;line-height:1.5}}.branch-map .siblings{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}}.branch-map .siblings .node{{font-size:11px;padding:14px 7px;border-top:3px solid var(--c)}}.branch-map small{{display:block;color:var(--muted);font-size:9px;margin-top:5px}}.task-sheet{{max-width:680px;margin:25px auto}}.task-sheet img{{width:100%;display:block}}.model-grid .model:last-child{{grid-column:auto}}.model-grid .model:last-child>.equation{{float:none;width:auto;margin:18px 0}}.facts strong{{font-size:30px}}@media(max-width:620px){{.hero-grid{{grid-template-columns:1fr}}.hero h1{{font-size:65px;letter-spacing:-3px}}.branch-map{{padding:18px 12px}}.branch-map .siblings{{gap:6px}}.equation{{font-size:16px}}}}
</style></head><body><a class="skip" href="#main">Skip to report</a><header><div class="wrap nav"><a class="brand" href="#">PAV / TRAINED-PARENT HYBRIDS</a><nav><a href="#results">Results</a><a href="#mechanisms">Mechanisms</a><a href="#protocol">Protocol</a><a href="#meaning">Meaning</a></nav></div></header><main id="main"><section class="hero wrap"><div class="eyebrow">Visual attention & working memory · First component</div><div class="hero-grid"><div><h1>Keep the motion.<br><em>Improve the<br>contour?</em></h1><p class="dek">One trained parent. Three continuations.<br>A measured test of two small additions.</p><p class="hero-copy">Can oriented local features or late channel modulation improve grouping without sacrificing the directional evidence already learned?</p></div><div><div class="status"><span class="dot"></span>{status} · {count}/3 development tests saved<small>Snapshot {now} · no automatic refresh</small></div><div class="branch-map" aria-label="The step 756 parent branches into three continued arms, sharing the same ordered decoder"><div class="node">Trained ConvNeXt + ordered decoder<small>Parent step 756 · shared learned state</small></div><div class="arrow">↓</div><div class="siblings"><div class="node" style="--c:#776494">Continue<small>Unchanged architecture</small></div><div class="node" style="--c:#448c80">Gabor branch<small>50 × 50 output</small></div><div class="node" style="--c:#c3834b">SE gate<small>13 × 13 output</small></div></div><div class="arrow">↓</div><div class="node">Same seven tasks · same new pair stream<small>Two 100 × 100 RGB frames · shared weights</small></div></div><p class="small muted">These are separate additions. There is no combined Gabor-plus-SE model in this experiment.</p></div></div><div class="scope"><b>The right control is<br>continued learning.</b><span>The parent was still improving at the end of the initial screen. A hybrid must be compared with an equally trained continued baseline; beating the frozen parent alone would not identify a benefit from the added computation.</span></div></section>
<section class="section wrap" id="results"><div class="section-head"><span class="index">01</span><div><div class="eyebrow">Development evidence</div><h2>More grouping, with motion retained.</h2><p class="intro">{outcome} The primary practical target was specified before the new results: at least +3 percentage points of contour balanced accuracy, with no more than 2 points of motion loss, relative to continued ConvNeXt.</p></div></div><div class="finding"><h3>One criterion, two distinct scientific questions.</h3><p>A useful engineering extension must improve the task tradeoff. A claim about biological contour binding would require additional evidence. Passing the numerical screen addresses the first question only.</p><div class="equation">{delta}</div><p><strong>Screen:</strong> Δ<sub>contour</sub> ≥ +0.03 and Δ<sub>motion</sub> ≥ −0.02. Both are absolute balanced-accuracy differences, not relative percent changes.</p></div>{results_html(data, paired)}{figure_html}<p class="caption">{link(path.name, 'Saved hybrid results')} · {link(paired_path.name, 'Paired comparison analysis')}{archive}. Earlier five-encoder measurements remain in the <a href="report_multitask.html">separate initial report</a>.</p></section>
<section class="section wrap" id="mechanisms"><div class="section-head"><span class="index">02</span><div><div class="eyebrow">Source-faithful mathematical adaptations</div><h2>Add evidence at an output.<br>Retain the learned pathways.</h2><p class="intro">The parent produces H₁, H₂ and H₃ at 50, 25 and 13 pixels per side. Each addition alters one returned feature field after the original pyramid has been computed. The common ordered decoder still receives all three scales.</p></div></div><div class="model-grid"><article class="model" style="--model:#448c80"><div class="model-top"><span>Fixed oriented measurements</span><b>+1,296 trainable parameters</b></div><h3>A Gabor output side branch.</h3><p>The current deterministic bank supplies 24 rectified simple responses and 24 quadrature energies. The branch excludes duplicate raw RGB, applies binomial low-pass filtering and ×2 decimation, normalizes channels at each position, then projects 48 channels to 24.</p><div class="equation">{gabor}</div><div class="equation">{residual}</div><p>P is a learned 1 × 1 projection. α is a learned per-channel scale initialized to zero. The projection begins nonzero: α can learn immediately, while its opening permits later loss gradients into the projection.</p><p class="ground">Only the returned H₁ changes. The modified field is not fed back into deeper ConvNeXt stages. The fixed bank adds 11,664 coefficients, plus 432 blur coefficients; its approximately 120.6 million extra MAC/frame is an analytical count, not a latency measurement.</p><p class="source"><a href="hybrid_models.py">Exact local branch</a> · <a href="https://papers.neurips.cc/paper_files/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html">Dapello et al. · VOneNet</a></p></article><article class="model" style="--model:#c3834b"><div class="model-top"><span>Image-dependent channel gain</span><b>+4,728 trainable parameters</b></div><h3>A centered late SE gate.</h3><p>Spatial means of the final 96-channel field feed a 96 → 24 → 96 MLP. Its output modulates the retained spatial map. Earlier 50 × 50 and 25 × 25 fields remain unchanged by this operation.</p><div class="equation">{se}</div><div class="equation">{gate}</div><p>W₂ and b₂ begin at zero, so initial gain is exactly one. Gains lie between 0.5 and 1.5. The first layer starts with ordinary random weights; the final layer opens learning to it.</p><p class="ground">This is a centered SE-inspired adaptation, not MobileNet’s literal gate or copied MobileNet weights. It adds about 4,608 MLP MAC/frame plus pooling and multiplications. GRN already rescales responses, so this gate could prove redundant.</p><p class="source"><a href="hybrid_models.py">Exact local gate</a> · <a href="https://arxiv.org/abs/1709.01507">Hu et al. · Squeeze-and-excitation</a></p></article></div><p class="callout" style="margin-top:25px">Identity initialization preserves the trained parent’s initial function. Subsequent optimization updates the base, decoder and new parameters, so neither branch guarantees that motion will stay unchanged.</p><details><summary>Why these additions were plausible, without attributing causality</summary><p>The initial screen found complementary contour predictions from Mobile/SE and VOne/residual models. Equal-probability combinations improved contour but reduced motion. That post hoc result motivated two narrowly specified additions; it did not identify SE or fixed Gabors as the cause of the original models’ behavior.</p><p>The Gabor bank uses 0.12 and 0.25 cycles/pixel. Current contour elements use a 5-pixel period (0.20 cycles/pixel), whereas orientation gratings use 0.05–0.10 cycles/pixel. Frequency coverage is a possible contributor, not a demonstrated explanation. Bank bandwidth, learned downstream stages and the retained raw-image pathway also matter.</p><p>Read the <a href="component_combination_research.md">complete rationale and primary sources</a>, <a href="hybrid_models.md">implementation notes</a>, and <a href="combination_analysis.json">historical saved-score complementarity analysis</a>.</p></details></section>
<section class="section wrap" id="protocol"><div class="section-head"><span class="index">03</span><div><div class="eyebrow">Matched continuation and finite exposure</div><h2>A sibling experiment.<br>Not a restart from scratch.</h2></div></div><div class="facts"><div><strong>756</strong><span>Shared parent updates</span></div><div><strong>{added_updates}</strong><span>Planned additional updates per arm</span></div><div><strong>{added_pairs}</strong><span>Planned new training pairs per arm</span></div><div><strong>1</strong><span>Training seed · 20271</span></div></div><div class="two"><div><h3>Preserve learning state.</h3><p>All arms inherit the step-756 ConvNeXt and decoder weights. Compatible Adam states transfer by parameter name; sampler position and Python, NumPy, Torch and CUDA RNG states resume from the parent. Only newly introduced parameter states are initialized anew.</p><p>Training keeps one 32-pair task batch per update, seven-task round robin, mean cross-entropy, AdamW learning rate 0.001, weight decay 0.0001 and gradient clipping 5, in fp32. The same resumed stream provides corresponding new draws to each arm.</p><p class="small">The planned additional exposure is {per_task} pairs per task per arm. Matching draws across models are paired exposure, not independent datasets. A fresh crop is not a unique new photograph.</p><p class="source"><a href="hybrid_transfer.py">Named state transfer</a> · <a href="train_hybrids.py">Continuation worker</a> · <a href="sweep_hybrids.py">Fixed-allocation supervisor</a></p></div><div><h3>Fresh pairs, reused source pool.</h3><p>New evaluation seeds generate shared examples for all arms. The natural task reuses the existing BSDS500 validation/test photo pools, which were already involved in the earlier model exploration. New crop and transform draws therefore provide development evidence on reused sources.</p><p>All seven task scores use fixed argmax decisions. Each arm’s checkpoint is selected by highest equal-task validation macro one-vs-rest AUC, with exact ties choosing the earlier checkpoint. Evaluation is planned for 224 validation pairs and 448 development-test pairs per task.</p><p class="small">Balanced accuracy averages class recalls: chance is 25% for four-way motion and 50% for the six binary judgments. One-vs-rest AUC measures score ranking and has chance 0.5. These answer related but distinct questions.</p></div></div><details><summary>See the unchanged seven-task sensory battery</summary><figure class="task-sheet"><img src="{contact}" alt="Actual two-frame stimulus examples for motion, signed orientation, contrast, spatial frequency, chromatic increment, contour grouping and natural spectral detail"><figcaption>Illustrative generator samples, not selected successes. <a href="sensory_battery_parameters.md">Numerical task definitions</a> · <a href="two_frame_task_research.md">Primary neuroscience and psychophysics rationale</a> · <a href="cardinal_motion.md">Krauzlis-derived cardinal motion adaptation</a></figcaption></figure><p>The ordered decoder retains signed difference and 25 directed local correlations, along with absolute difference, mean and product at each scale. Frame order carries motion direction. There is no attention cue, working-memory delay or later attention-selection system here.</p></details><h3 style="margin-top:35px">Parameter count is not runtime.</h3>{cost_html(data)}<p class="caption">Parameter totals describe the actual compact local implementations; fixed filter buffers are separate. Profile times cover 32-pair batches. Training-step timing and evaluation-batch timing have different boundaries; see the archived worker. Peak allocated GPU memory is PyTorch allocation, not total desktop usage. Training worker wall time includes process/startup/checkpoint overhead. Total saved run wall time: {number(data.get('wall_seconds'), 1)} seconds.</p></section>
<section class="section wrap" id="meaning"><div class="section-head"><span class="index">04</span><div><div class="eyebrow">Interpretation with the right scope</div><h2>Useful evidence.<br>Proportionate conclusions.</h2></div></div><div class="grounding-panel"><div><span class="pill">Neuroscience grounding</span><h3>Gabor responses are an explicit early-vision commitment.</h3><p>Fixed oriented filters, rectification and quadrature energy connect the branch to simple/complex-cell-inspired computations. The deterministic parameter grid is a local adaptation, without the original VOneNet’s fitted physiological distributions or neuronal noise.</p><p>SE and GRN are engineering response-modulation mechanisms. Their presence does not establish biological attention or contour binding. No neural recordings are fit or predicted in this comparison.</p></div><div><span class="pill">Measured engineering value</span><h3>The continued baseline sets the standard.</h3><p>If an addition clears the contour/motion screen, it supports that extension of this trained parent under this exposure. If the baseline improves similarly, the simpler model remains competitive. A contour gain accompanied by excessive motion loss is a tradeoff, not a successful combination of strengths.</p><p>No result from this single-seed continuation establishes a permanent architectural limit, training convergence, or superiority from random initialization.</p></div></div><div class="three limitations"><div><h4>Paired uncertainty</h4><p>Shared examples permit paired comparisons. Resample generated trials, or source-photo clusters for natural images. Marginal intervals cannot be subtracted to make a valid interval for a difference.</p></div><div><h4>Conditional evidence</h4><p>Bootstrap intervals condition on trained checkpoints and sampled evaluation groups; they do not estimate variation across training seeds. Empirical perfect-score intervals at a boundary are not certainty.</p></div><div><h4>Exploratory selection</h4><p>The prior screen informed this design. Fresh generated pairs reduce exact-example reuse, but reused natural sources and one development cycle limit confirmatory claims. The practical margin is not a formal biological criterion.</p></div></div></section>
<section class="closing wrap"><div class="eyebrow">First component of a broader system</div><h2>Improve PAV.<br>Keep the larger questions open.</h2><p>Guided Search 6.0 supplies the functional scaffold. The diffuser is omitted; activated long-term memory is approximated as synaptic weights by the user’s modeling choice. Visual attention, working memory and integration follow later. The present results concern two-frame sensory representations and their common decoder.</p><p class="sources"><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/">Wolfe · Guided Search 6.0</a> · <a href="component_combination_research.md">Hybrid source rationale</a> · <a href="hybrid_models.md">Implemented equations and initialization</a> · <a href="report_multitask.html">Initial five-encoder report</a> · {link(path.name, 'Hybrid raw results')}</p><p class="small">Self-contained HTML with embedded figures and native MathML; no CDN, external font or script dependency. Source and raw-data links remain local companion artifacts. This is a dated report, not a live monitor.</p></section></main><footer class="wrap"><span>PAV trained-parent comparison · {now}<br>Saved status: {status} · seed 20271 · prior reports preserved</span><button type="button" onclick="window.print()">Print / save PDF</button></footer></body></html>'''
    html = html.replace('Training-step timing and evaluation-batch timing have different boundaries; see the archived worker.', 'Training-step timing includes pair generation, device transfer, forward/backward, optimizer update and GPU synchronization. Evaluation timing averages one batch per task, including pair generation, transfer, forward and synchronization; it is not pure GPU inference latency.')
    output.write_text(html, encoding='utf-8')
    return {'output': str(output), 'status': data.get('status', 'pending'), 'completed_tests': count, 'mathml': html.count('<math '), 'figures': list(figs), 'bytes': output.stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results_hybrids.json')
    parser.add_argument('--paired', type=Path, default=HERE / 'results_hybrids_analysis.json')
    parser.add_argument('--output', type=Path, default=HERE / 'report_hybrids.html')
    args = parser.parse_args()
    print(json.dumps(render(args.results, args.paired, args.output), indent=2))
