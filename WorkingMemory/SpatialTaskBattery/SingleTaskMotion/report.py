"""Paired saved-score report, conditional on this single continuation."""
import json
from pathlib import Path
import numpy as np
def report(root):
    root=Path(root);a=json.loads((root/'aggregate.json').read_text());lines=['# Motion-only continuation','',f"Added motion episodes: {a['pinned_added_episodes']}. Parent step10000; selected step{a['selected_step']}; terminal step{a['training']['step']}.",'','| Delay | Parent BA | Selected BA | Terminal BA | Selected gain (paired95% interval) |','|---|---:|---:|---:|---:|'];rng=np.random.default_rng(73973001);changes={}
    rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
    parent=rows(a['parent_test']['predictions']);selected=rows(a['selected_test']['predictions'])
    for name in ('D0','D4','D12','D24'):
        aa=[r for r in selected if r['condition']==name];bb=[r for r in parent if r['condition']==name];assert [(r['paired_base_id'],r['label'],r['metadata']) for r in aa]==[(r['paired_base_id'],r['label'],r['metadata']) for r in bb]
        y=np.array([r['label'] for r in aa]);delta=np.array([int(np.argmax(r['probabilities'])==r['label'])-int(np.argmax(s['probabilities'])==s['label']) for r,s in zip(aa,bb)]);groups=[delta[y==k] for k in range(4)];boot=np.mean([g[rng.integers(0,len(g),(2000,len(g)))].mean(1) for g in groups],axis=0);ci=np.quantile(boot,[.025,.975]);gain=float(np.mean([g.mean() for g in groups]));changes[name]=dict(gain=gain,ci95=ci.tolist())
        metrics=[a[k]['cells'][name]['balanced_accuracy'] for k in ('parent_test','selected_test','terminal_test')];lines.append(f"| {name} | {metrics[0]*100:.2f}% | {metrics[1]*100:.2f}% | {metrics[2]*100:.2f}% | {gain*100:+.2f}pp [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |")
    lines += ['','Each delay has512 scored presentations of the same512 base episodes; repeats across delays are paired, not additional independent episodes. The2,000-replicate bootstrap is class-stratified and conditional on these trained models. All confusion matrices and probabilities remain in the summaries/predictions.','', 'The cloud run contributes eight motion episodes per mixed update, matching the local motion sample count per update. Local training uses a full motion loss; the cloud averages five task losses and combines their gradients. Changes therefore do not isolate gradient interference. Cloud training and its deadline were left unchanged.']
    (root/'report.md').write_text('\n'.join(lines));(root/'paired_changes.json').write_text(json.dumps(changes,indent=2))
