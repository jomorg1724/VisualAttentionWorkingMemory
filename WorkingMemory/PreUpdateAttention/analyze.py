"""Paired saved-prediction comparison; no training or GPU work."""
import sys,json,csv,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from WorkingMemory.SelectiveMaintenance.analyze import metrics
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(path):
    result={}
    for line in Path(path).read_text().splitlines():
        row=json.loads(line);result.setdefault(row['condition'],[]).append(row)
    for group in result.values():group.sort(key=lambda r:int(r['paired_base_id'].split('/')[-1]))
    return result
def main():
    receipt=read(HERE/'retrieval_receipt.json');assert receipt['status']=='verified'
    remote=Path(receipt['results']);agg=read(remote/'aggregate.json');assert agg['status']=='completed'
    local=ROOT/'WorkingMemory/SelectiveMaintenance';control=read(local/'results.json');assert control['status']=='completed'
    data={'attention':rows(remote/'test/predictions.jsonl')}
    for arm in ('continuation','controller_feedback'):data[arm]=rows(control['runs'][arm]['test']['predictions'])
    data['parent']=rows(control['parent_reference']['predictions'])
    rng=np.random.default_rng(50973001);cells={};differences={};indices={}
    for cell,base in data['attention'].items():
        y=np.array([r['label'] for r in base]);kind=('binding_locations' if cell.endswith('locations') else 'binding') if cell.startswith('binding') else ('single' if cell.startswith('single') else 'motion')
        if kind not in indices:
            if kind.startswith('binding'):indices[kind]=np.array([np.concatenate([np.arange(4*b,4*b+4) for b in rng.integers(len(y)//4,size=len(y)//4)]) for _ in range(1000)])
            else:indices[kind]=np.array([np.concatenate([rng.choice(np.flatnonzero(y==v),sum(y==v),replace=True) for v in np.unique(y)]) for _ in range(1000)])
        boot=indices[kind];cells[cell]={};draws={}
        for arm,allrows in data.items():
            rs=allrows[cell];assert len(rs)==len(base)
            assert all(a['paired_base_id']==b['paired_base_id'] and a['label']==b['label'] and a['metadata']==b['metadata'] for a,b in zip(base,rs)),cell
            p=np.array([r['probabilities'] for r in rs]);cells[cell][arm]=metrics(y,p)
            guess=p.argmax(1);draws[arm]=np.array([np.mean([np.mean(guess[ix][y[ix]==v]==v) for v in range(p.shape[1])]) for ix in boot])
            cells[cell][arm]['ba_ci95']=np.quantile(draws[arm],[.025,.975]).tolist()
        differences[cell]={arm:dict(ba=cells[cell]['attention']['ba']-cells[cell][arm]['ba'],ba_ci95=np.quantile(draws['attention']-draws[arm],[.025,.975]).tolist()) for arm in ('continuation','controller_feedback','parent')}
    training=list(csv.DictReader((remote/'preupdate_attention/metrics.csv').open()));diag=[json.loads(r['diagnostics']) for r in training if r['diagnostics']!='{}']
    result=dict(status='completed',selected_global_step=agg['runs']['preupdate_attention']['selected_step'],terminal_new_updates=len(training),terminal_new_episodes=8*len(training),cells=cells,attention_minus_baseline=differences,training_diagnostics=diag,clipping_fraction=float(np.mean([int(r['clipped']) for r in training])),platform=agg['platform'],uncertainty='1000 paired resamples; binding resamples128 four-case blocks; other tasks512 class-stratified episodes; intervals conditional on these trained seeds; repeated models/delays do not add independent examples')
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2));(HERE/'results.json').write_text(json.dumps(agg,indent=2))
    lines=['# Pre-update attention: unchanged-teaching competitor','',f"Completed {8*len(training):,} additional episodes from spatial4400; selected global checkpoint{result['selected_global_step']}. Tasks, cues, labels, losses, comparator, heads and90/10schedule remain unchanged. The new arm ran on a single cloud RTX3090; controls ran locally. Package versions match, but this is not bitwise cross-platform equivalence.",'','| Condition | Parent BA% | Continuation BA% | Feedback BA% | Attention BA% | Attention−continuation pp [95%CI] |','|---|---:|---:|---:|---:|---:|']
    for name,c in cells.items():
        d=differences[name]['continuation'];lines.append(f"| {name} | {100*c['parent']['ba']:.2f} | {100*c['continuation']['ba']:.2f} | {100*c['controller_feedback']['ba']:.2f} | {100*c['attention']['ba']:.2f} | {100*d['ba']:+.2f} [{100*d['ba_ci95'][0]:+.2f}, {100*d['ba_ci95'][1]:+.2f}] |")
    lines+=['','The attention package adds27,590 parameters and no persistent state. Both heads jointly select current sensory and old-memory tokens before the existing E/I memory input drive. Existing comparator and sensory decision routes remain. Attention weights alone do not establish retention or a uniquely biological mechanism. Motion has only10% training allocation and does not participate in checkpoint selection. Binding swaps can be solved using one location and therefore do not establish two-item memory capacity. Unequal task renderings/change sizes preclude direct capacity ranking between binding and native single-item orientation.','',result['uncertainty']+'.','', '[Full AUC, confusion matrices, intervals and numerical diagnostics](analysis.json)']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='completed',selected_step=result['selected_global_step'],report=str(HERE/'report.md'))))
if __name__=='__main__':main()
