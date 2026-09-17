"""Read-only post-retrieval paired comparison; no training or GPU access."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def main(receipt_path=None):
    receipt=read(receipt_path or HERE/'retrieval_receipt.json');fetched=Path(receipt['results']);agg=read(fetched/'aggregate.json');lines=['# Attention without explicit source/distance priors','',f"Run status: {agg['status']}. Source/task definitions and original deadline were fixed before production.",'']
    def local(p):return fetched/str(p).split('/remote_results/')[-1]
    comparisons={}
    control=ROOT/'WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/aggregate.json'
    for arm,r in agg['runs'].items():
        lines += [f'## {arm}',f"Additional episodes: {r['added_episodes']}; selected global step: {r.get('selected_step')}; parent fallback: {r.get('selected_is_parent',False)}.",'']
        for label in ('selected_test','terminal_test'):
            if label not in r:continue
            e=r[label];lines += [f'### {label}','','| Condition | BA | Accuracy | AUC |','|---|---:|---:|---:|']
            for c in e['cells'].values():
                m=c['overall'];fmt=lambda x:'undefined' if x is None else f'{100*x:.2f}%'
                lines.append(f"| {c['condition']} | {fmt(m['balanced_accuracy'])} | {fmt(m['accuracy'])} | {m['macro_ovr_auc']} |")
            lines.append('')
        if arm=='old_unbiased' and control.exists() and 'terminal_test' in r:
            ctrl=read(control)['runs']['control_10'];a=rows(local(r['terminal_test']['predictions']));b=rows(ctrl['terminal_test']['predictions']);rng=np.random.default_rng(65973001)
            lines += ['### Equal-exposure terminal comparison','','Unbiased terminal minus saved biased control terminal, using identical test episodes. The paired bootstrap uses complete four-case blocks for binding and class-stratified episodes for other tasks (2,000 replicates). This reused test is exploratory; hardware differs; intervals are conditional on these trained models.','', '| Condition | BA change | Paired95% interval |','|---|---:|---:|']
            for name in sorted({x['condition'] for x in a}):
                aa=sorted([x for x in a if x['condition']==name],key=lambda x:int(x['paired_base_id'].rsplit('/',1)[-1]));bb=sorted([x for x in b if x['condition']==name],key=lambda x:int(x['paired_base_id'].rsplit('/',1)[-1]))
                assert [(x['paired_base_id'],x['label']) for x in aa]==[(x['paired_base_id'],x['label']) for x in bb]
                y=np.array([x['label'] for x in aa]);delta=np.array([int(np.argmax(x['probabilities'])==x['label'])-int(np.argmax(z['probabilities'])==z['label']) for x,z in zip(aa,bb)]);by=[delta[y==k] for k in np.unique(y)];point=float(np.mean([d.mean() for d in by]));unit='class-stratified paired episodes'
                if name.startswith('binding'):
                    assert all(len(set((aa[j]['label'],aa[j]['metadata']['sample_bit']) for j in range(i,i+4)))==4 for i in range(0,len(aa),4))
                    blocks=delta.reshape(-1,4).mean(1);boot=blocks[rng.integers(0,len(blocks),(2000,len(blocks)))].mean(1);unit='paired four-case counterbalance blocks'
                else:boot=np.mean([d[rng.integers(0,len(d),(2000,len(d)))].mean(1) for d in by],axis=0)
                ci=np.quantile(boot,[.025,.975]).tolist();comparisons[name]=dict(ba_change=point,ci95=ci,n=len(y),uncertainty_unit=unit)
                lines.append(f'| {name} | {point*100:+.2f}pp | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |')
    lines += ['','All per-class confusion matrices, probabilities and per-head attention summaries remain in the unchanged retrieved artifacts. New-task recognition load0 has specificity/accuracy only. The five-task battery has no equally trained biased control and does not isolate an architecture effect.']
    out=Path(receipt['archive']).parent;(out/'report.md').write_text('\n'.join(lines));(out/'paired_terminal_comparison.json').write_text(json.dumps(comparisons,indent=2));print(str(out/'report.md'))
if __name__=='__main__':main()
