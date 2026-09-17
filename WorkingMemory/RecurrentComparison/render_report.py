"""Standalone saved-data scientific view, with explicit pending cells."""
import json,html,re
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent


def render():
    r=json.loads((HERE/'results.json').read_text());c=r['config'];runs=r.get('runs',{});arms=c['arms']
    remote_progress=Path(r['run_root'])/'remote_progress.json'
    if 'ei_adaptive' not in runs and remote_progress.exists():
        runs['ei_adaptive']=json.loads(remote_progress.read_text())['runs']['ei_adaptive']
    analysis=json.loads((HERE/'analysis.json').read_text()) if (HERE/'analysis.json').exists() else None
    esc=lambda x:html.escape(str(x))
    pct=lambda x:f'{100*x:.1f}%'
    style=re.search(r'<style>(.*?)</style>',(HERE.parent/'report.html').read_text(encoding='utf-8'),re.S).group(1)
    out=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recurrent memory without attention</title><style>'+style+'</style><main>',
        '<nav><b>Visual working memory</b><a href="#computation">Computation</a><a href="#evidence">Evidence</a><a href="#limits">Interpretation</a></nav>',
        '<header><div><span class="kicker">A focused two-model comparison</span><h1>Memory after<br>perception.</h1><p class="lead">Can a learned recurrent state turn an existing visual systemâ€™s momentary evidence into a usable ongoing computation?</p></div>',
        f'<div class="summary"><span class="pill">{esc(r["status"].replace("_"," "))}</span><p>One shared opponent sensory parent. Two different recurrent dynamics. No attention or addressing.</p><p class="small">Target {c.get("episodes_per_arm","pending"):,} fresh episodes per arm. All learned sensory weights can adapt at the lower learning rate.</p></div></header>' if isinstance(c.get('episodes_per_arm'),int) else '<div class="summary">Profiled exposure pending.</div></header>',
        '<div class="stats">'+''.join(f'<div class="stat"><strong>{v}</strong><small>{k}</small></div>' for k,v in [('LSTM state','256 + 256'),('E/I state','256 + 256'),('Focused cells','6'),('Total allowance','4 hours')])+'</div>',
        '<section id="computation"><div class="intro"><h2>Same sensory evidence.<br>Different state updates.</h2><p>Every image produces an opponent emission and a spatially ordered memory input. Blank, cue and probe images use the same path. Only explicit traces and recurrent vectors carry history.</p></div>',
        '<div class="diagram"><div class="flow"><div>Current image<small>100 Ã—100 Ã—3</small></div><span>â†’</span><div>Opponent fields<small>Current + fast/slow information</small></div><span>â†’</span><div>Spatial projection<small>64 Ã—13 Ã—13 â†’8 Ã—13 Ã—13 â†’128</small></div><span>â†’</span><div>Recurrent memory<small>512 state values</small></div></div><p style="color:#b6cdc3">The final head receives the existing sensory feature plus a small learned memory residual. No previous prediction is cached, and no memory feedback enters the encoder.</p></div>',
        '<div class="grid" style="grid-template-columns:1fr 1fr"><article class="card"><span class="kicker">Engineering baseline</span><h3>Gated LSTM</h3><div class="eq"><math display="block"><msub><mi>c</mi><mi>t</mi></msub><mo>=</mo><msub><mi>f</mi><mi>t</mi></msub><mo>âŠ™</mo><msub><mi>c</mi><mrow><mi>t</mi><mo>âˆ’</mo><mn>1</mn></mrow></msub><mo>+</mo><msub><mi>i</mi><mi>t</mi></msub><mo>âŠ™</mo><msub><mi>g</mi><mi>t</mi></msub></math></div><div class="eq"><math display="block"><msub><mi>h</mi><mi>t</mi></msub><mo>=</mo><msub><mi>o</mi><mi>t</mi></msub><mo>âŠ™</mo><mi>tanh</mi><mo>(</mo><msub><mi>c</mi><mi>t</mi></msub><mo>)</mo></math></div><p>Independent write, retain and output gates depend on current input and the previous recurrent output. Per-gate normalization precedes the gate biases; the stored cell is unnormalized. Nominal initial forget times span4â€“128 frames.</p><p>1,011,872 learned parameters;603,144 newly added.</p></article>',
        '<article class="card"><span class="kicker">Neuroscience-inspired competitor</span><h3>Adaptive E/I rates</h3><div class="eq"><math display="block"><msub><mi>r</mi><mi>t</mi></msub><mo>=</mo><mo>(</mo><mn>1</mn><mo>âˆ’</mo><mi>Î±</mi><mo>)</mo><mo>âŠ™</mo><msub><mi>r</mi><mrow><mi>t</mi><mo>âˆ’</mo><mn>1</mn></mrow></msub><mo>+</mo><mi>Î±</mi><mo>âŠ™</mo><mi>ReLU</mi><mo>(</mo><msub><mi>u</mi><mi>t</mi></msub><mo>)</mo></math></div><div class="eq"><math display="block"><msub><mi>a</mi><mi>t</mi></msub><mo>=</mo><mo>(</mo><mn>1</mn><mo>âˆ’</mo><mi>Î²</mi><mo>)</mo><mo>âŠ™</mo><msub><mi>a</mi><mrow><mi>t</mi><mo>âˆ’</mo><mn>1</mn></mrow></msub><mo>+</mo><mi>Î²</mi><mo>âŠ™</mo><msub><mi>r</mi><mrow><mi>t</mi><mo>âˆ’</mo><mn>1</mn></mrow></msub></math></div><p>Current u combines sensory drive, sign-constrained recurrent E/I drive and adaptation. Updates are synchronous; rates, adaptation and recurrent current are not normalized. Rates start with time constants2â€“8 frames and adaptation16â€“64.</p><p>714,912 learned parameters;306,184 newly added.</p></article></div></section>',
        '<section id="evidence"><div class="intro"><h2>Acquisition, one capability at a time.</h2><p>Held-out results only. Motion chance accuracy is25%, orientation50%. Mean OVR-AUC has chance0.5 in both. Open each model’s details for complete confusion matrices and uncertainty.</p></div><div class="tablewrap"><table><tr><th>Cell</th><th>LSTM BA</th><th>LSTM AUC</th><th>E/I BA</th><th>E/I AUC</th></tr>']
    for name,cell in c['cells'].items():
        out.append('<tr><td>'+esc(name)+'</td>')
        for arm in arms:
            record=runs.get(arm,{}).get('test');m=record['cells'][cell['family']+'/'+name]['overall'] if record else None
            out.append(f'<td>{pct(m["balanced_accuracy"])}</td><td>{m["macro_ovr_auc"]:.3f}</td>' if m else '<td>Pending</td><td>Pending</td>')
        out.append('</tr>')
    out.append('</table></div>')
    for arm in arms:
        record=runs.get(arm,{});test=record.get('test')
        if test:
            out.append(f'<details><summary>{esc(arm)}: selected update {record["selected_step"]:,}, full held-out metrics</summary><pre style="white-space:pre-wrap;font-size:11px">{esc(json.dumps(test["cells"],indent=2))}</pre></details>')
    if analysis:
        out.append('<div class="footnote">Both models acquired the focused tasks. LSTM scored higher on longer motion duration; the small E/I advantage on delayed orientation recall has a paired interval that includes zero. Resetting the new recurrent history substantially reduces motion ranking and decisions. In orientation recall, BA changes coexist with essentially unchanged AUC, so they do not establish extra retained information.</div>')
    out+=['<h3 style="margin-top:35px">Paired recurrent-history interventions</h3><p class="small">Normal minus reset: only the added state resets before every frame. Opponent traces remain intact. This is not a separately trained memory-free control. Intervals condition on these fitted models.</p>']
    if analysis:
        out.append('<div class="tablewrap"><table><tr><th>Comparison / cell</th><th>BA difference</th><th>95% interval</th><th>AUC difference</th></tr>')
        for v in analysis['comparisons']:
            lo,hi=v['delta_ba_ci95'];out.append(f'<tr><td>{esc(v["comparison"])} / {esc(v["condition"])}</td><td>{100*v["delta_ba"]:+.2f}pp</td><td>[{100*lo:+.2f},{100*hi:+.2f}]</td><td>{v["delta_auc"]:+.3f}</td></tr>')
        out.append('</table></div>')
    else:out.append('<p>Paired analysis pending completion of both held-out evaluations.</p>')
    out.append('</section><section><h2>Learning history stays visible.</h2><p class="small">Equal six-cell validation AUC selects checkpoints; training continues to the fixed endpoint. The two curves do not represent independent seed replications.</p><svg class="timeline" viewBox="0 0 1100 215">')
    X=lambda step:65+980*step/max(1,c.get('total_steps',1));Y=lambda auc:180-150*auc
    for y in (.25,.5,.75,1):out.append(f'<line x1="65" y1="{Y(y)}" x2="1045" y2="{Y(y)}" stroke="#d3ddd4"/><text x="20" y="{Y(y)+4}">{y}</text>')
    for arm,color in [('lstm','#056f65'),('ei_adaptive','#b45331')]:
        vals=runs.get(arm,{}).get('validation',[])
        if vals:out.append('<polyline fill="none" stroke="'+color+'" stroke-width="3" points="'+' '.join(f'{X(v["step"])},{Y(v["selection_mean_auc"])}' for v in vals)+'"/>')
        for v in vals:out.append(f'<circle cx="{X(v["step"])}" cy="{Y(v["selection_mean_auc"])}" r="4" fill="{color}"/>')
    out.append('</svg><p class="small">Teal: LSTM Â· Rust: E/I. Horizontal axis: updates to the fixed endpoint; vertical: validation AUC.</p><div class="tablewrap"><table><tr><th>Arm</th><th>Update</th><th>Episodes</th><th>Mean AUC</th></tr>')
    for arm in arms:
        for v in runs.get(arm,{}).get('validation',[]):out.append(f'<tr><td>{esc(arm)}</td><td>{v["step"]:,}</td><td>{v["step"]*c["recipe"]["batch_size"]:,}</td><td>{v["selection_mean_auc"]:.4f}</td></tr>')
    out.append('</table></div></section><section id="limits"><h2>What this can establish.</h2><div class="narrative"><p>Reliable minimal-condition acquisition is needed before a harder-condition deficit can be interpreted as a memory limitation. Strong score ranking with biased decisions does not imply absent information. Equal exposure does not equalize parameterization, initialization or optimization difficulty.</p><p>Both models add a new shared sensory interface and receive focused training. Differences from the older broad battery cannot be assigned solely to the recurrent core. An intervention on trained history measures that interventionâ€™s effect; it does not establish biological correspondence or universal working-memory necessity.</p><div class="footnote">The E/I core starts more contractive than the retention-biased LSTM. Its raw-softplus gradient and actual effective-weight movement are logged explicitly. Both use Adam epsilon1e-10, fresh state, lower sensory LR and full BPTT. At the user’s request, LSTM runs locally and E/I on a separate RTX3090; each device has one fp32 training worker. Exact fresh initialization and scientific library versions match, while platform differences remain a comparison caveat. No attention or extra tuning sweep was added.</div><p><a href="report.md">Full scientific report</a> Â· <a href="results.json">Raw summaries</a> Â· <a href="analysis.json">Paired analysis</a> Â· <a href="README.md">Exact protocol</a> Â· <a href="../Research/recurrent_memory_without_attention.md">Primary-source design</a></p></div></section>')
    out.append(f'<footer>Saved-data report rendered {datetime.now(timezone.utc).isoformat()}. Status: {esc(r["status"])}. Run: <code>{esc(r["run_root"])}</code></footer></main></html>')
    if analysis:
        note='<p class="small">Updates clipped at the global norm limit: '+', '.join(f'{esc(a)} {100*analysis["training"][a]["clipping_fraction"]:.2f}%' for a in arms)+'. Frequent clipping remains an optimization limitation. All work has stopped and the paid pod was deleted after verified retrieval.</p>'
        out.insert(-1,note)
    path=HERE/'report.html';path.write_text('\n'.join(out),encoding='utf-8');print(path)

if __name__=='__main__':render()
