"""Saved-prediction exposure report, including rejected terminal capabilities."""
import os,sys,json,csv,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from WorkingMemory.SelectiveMaintenance.analyze import metrics
from WorkingMemory.TrainingExposure.sweep import write
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def resolve(evaluation,root):
    p=Path(evaluation['predictions'])
    if p.exists():return p
    text=evaluation['predictions'].replace('\\','/');suffix=text.split('/remote_results/',1)[1];return root/suffix
def rows(evaluation,root):
    groups={}
    for line in resolve(evaluation,root).read_text().splitlines():
        r=json.loads(line);groups.setdefault(r['condition'],[]).append(r)
    for group in groups.values():group.sort(key=lambda x:int(x['paired_base_id'].split('/')[-1]))
    return groups
def roots():
    result={};local=Path(read(HERE/'local_run.json')['run'])
    if (local/'exit.json').exists() and read(local/'exit.json')['status']=='completed':result['control_10']=local
    remote=HERE/'Cloud/retrieval_receipt.json'
    if remote.exists():
        receipt=read(remote)
        if receipt.get('status')=='verified':
            p=Path(receipt['results'])
            if read(p/'exit.json')['status']=='completed':result['focused_50']=p
    return result
def main():
    paths=roots()
    if not paths:raise RuntimeError('No completed exposure arm')
    result=dict(status='completed' if len(paths)==2 else 'partial',arms={},cross_arm={},uncertainty='2000 paired resamples. Binding uses128 independent four-case blocks per location split; other tasks512 class-stratified base episodes. Within each report contrast, one-sided95% simultaneous lower bounds use the bootstrap maximum centered estimation error over all14cells. Intervals are conditional on trained seeds; this is not equivalence proof or a cross-hardware causal isolation.')
    bootstrap={};rng=np.random.default_rng(55973001);all_data={}
    for arm,root in paths.items():
        agg=read(root/'aggregate.json');record=agg['runs'][arm];data={name:rows(e,root) for name,e in [('parent',agg['parent_test']),('selected',record['selected_test']),('terminal',record['terminal_test'])]};all_data[arm]=data
        report=dict(selected_step=record['selected_step'],selected_is_parent=record['selected_is_parent'],terminal_step=12400,platform=agg['platform'],cells={},differences={},validation=[],training={})
        draw_deltas={'selected':[],'terminal':[]};point_deltas={'selected':[],'terminal':[]};names=[]
        for cell,base in data['parent'].items():
            y=np.array([r['label'] for r in base]);kind=('binding_locations' if cell.endswith('locations') else 'binding') if cell.startswith('binding') else ('single' if cell.startswith('single') else 'motion')
            if kind not in bootstrap:
                if kind.startswith('binding'):bootstrap[kind]=np.array([np.concatenate([np.arange(4*b,4*b+4) for b in rng.integers(len(y)//4,size=len(y)//4)]) for _ in range(2000)])
                else:bootstrap[kind]=np.array([np.concatenate([rng.choice(np.flatnonzero(y==v),sum(y==v),replace=True) for v in np.unique(y)]) for _ in range(2000)])
            ix=bootstrap[kind];draws={};report['cells'][cell]={};report['differences'][cell]={};names.append(cell)
            for name,groups in data.items():
                rs=groups[cell];assert len(rs)==len(base) and all(a['paired_base_id']==b['paired_base_id'] and a['label']==b['label'] and a['metadata']==b['metadata'] for a,b in zip(base,rs))
                p=np.array([r['probabilities'] for r in rs]);guess=p.argmax(1);m=metrics(y,p);cm=np.array(m['confusion']);m['class_recalls']=(np.diag(cm)/cm.sum(1)).tolist();report['cells'][cell][name]=m
                draws[name]=np.mean([np.mean((guess[ix]==v)&(y[ix]==v),axis=1)/np.mean(y[ix]==v,axis=1) for v in range(p.shape[1])],axis=0)
            for name in ('selected','terminal'):
                delta=draws[name]-draws['parent'];point=report['cells'][cell][name]['ba']-report['cells'][cell]['parent']['ba'];report['differences'][cell][name]=dict(ba=point,ba_ci95=np.quantile(delta,[.025,.975]).tolist());draw_deltas[name].append(delta);point_deltas[name].append(point)
        for name in ('selected','terminal'):
            estimates=np.array(point_deltas[name]);draws=np.stack(draw_deltas[name],1);radius=float(np.quantile((draws-estimates).max(1),.95));lower=estimates-radius
            report[name+'_preservation']=dict(simultaneous_radius=radius,all14_lower_bounds_above_minus_2pp=bool(np.all(lower>=-.02)),observed_regression_cells=[cell for cell,d in zip(names,estimates) if d<0],observed_losses_over_2pp=[cell for cell,d in zip(names,estimates) if d<-.02],interpretation='A crossing of -2pp means preservation unresolved, not an automatic instruction to expand evaluation.')
            for cell,bound in zip(names,lower):report['differences'][cell][name]['simultaneous_lower95']=float(bound)
        log=list(csv.DictReader((root/arm/'metrics.csv').open()));report['training']=dict(updates=len(log),episodes=8*len(log),logical_frames=sum(8*int(r['frames']) for r in log),clipping_fraction=float(np.mean([int(r['clipped']) for r in log])))
        for val in record['validation']:
            prior=[r for r in log if int(r['step'])<=val['step']];counts={name:sum(r['cell']==name for r in prior)*8 for name in agg['config']['recipe']['cells']}
            report['validation'].append(dict(step=val['step'],added_updates=val['step']-8400,added_episodes=8*len(prior),cell_exposure=counts,assessment=val['assessment'],cells={c['condition']:c['overall'] for c in val['cells'].values()}))
        result['arms'][arm]=report
    if len(paths)==2:
        for kind in ('selected','terminal'):
            result['cross_arm'][kind]={}
            for cell,base in all_data['control_10'][kind].items():
                other=all_data['focused_50'][kind][cell];assert all(a['label']==b['label'] and a['metadata']==b['metadata'] for a,b in zip(base,other));y=np.array([r['label'] for r in base]);family=('binding_locations' if cell.endswith('locations') else 'binding') if cell.startswith('binding') else ('single' if cell.startswith('single') else 'motion');ix=bootstrap[family];pred=[np.array([r['probabilities'] for r in rows]).argmax(1) for rows in (base,other)];k=len(base[0]['probabilities'])
                draws=[np.mean([np.mean((g[ix]==v)&(y[ix]==v),axis=1)/np.mean(y[ix]==v,axis=1) for v in range(k)],axis=0) for g in pred];point=result['arms']['focused_50']['cells'][cell][kind]['ba']-result['arms']['control_10']['cells'][cell][kind]['ba'];result['cross_arm'][kind][cell]=dict(focused_minus_control_ba=point,ci95=np.quantile(draws[1]-draws[0],[.025,.975]).tolist())
    write(HERE/'analysis.json',result)
    lines=['# Training exposure: all capabilities remain visible','', 'Status: '+result['status']+'. No architecture, cue, objective, LR or calibration changes. Selection uses the predeclared ten-cell validation screen; terminal results remain reported even if rejected.','']
    for arm,report in result['arms'].items():
        lines += ['## '+arm,'',f"Trained {report['training']['episodes']:,} additional episodes; selected global{report['selected_step']}"+(' (unchanged parent fallback).' if report['selected_is_parent'] else '.'),'','| Cell | Parent BA% | Selected BA% | Terminal BA% | Selected−parent pp [95%CI] | Selected simultaneous lower95 pp |','|---|---:|---:|---:|---:|---:|']
        for cell,v in report['cells'].items():
            d=report['differences'][cell]['selected'];lines.append(f"| {cell} | {100*v['parent']['ba']:.2f} | {100*v['selected']['ba']:.2f} | {100*v['terminal']['ba']:.2f} | {100*d['ba']:+.2f} [{100*d['ba_ci95'][0]:+.2f},{100*d['ba_ci95'][1]:+.2f}] | {100*d['simultaneous_lower95']:+.2f} |")
        lines += ['', 'Selected preservation across all14cells: '+('supported against the predeclared2pp margin within this bootstrap screen.' if report['selected_preservation']['all14_lower_bounds_above_minus_2pp'] else 'unresolved against the predeclared2pp margin; do not label a universal upgrade.'),'Observed selected declines: '+str(report['selected_preservation']['observed_regression_cells'])+'. Terminal declines>2pp: '+str(report['terminal_preservation']['observed_losses_over_2pp'])+'.','']
    lines += [result['uncertainty'],'','[Full BA/AUC, class recalls, paired differences and validation exposure](analysis.json). Equal added episodes/updates are not equal logical frames. Family evidence prefixes match; per-delay assignments can differ. Hardware differences prevent a strictly hardware-controlled causal interpretation. A flat finite curve does not establish a capacity ceiling. No follow-on residual run is automatic.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for axis in ('updates','cell_exposure'):
        fig,axs=plt.subplots(5,2,figsize=(12,15),constrained_layout=True)
        cells=list(next(iter(result['arms'].values()))['validation'][0]['cells'])
        for ax,cell in zip(axs.flat,cells):
            for arm,report in result['arms'].items():
                val=report['validation'];x=[v['added_updates'] if axis=='updates' else v['cell_exposure'][cell] for v in val];y=[100*v['cells'][cell]['balanced_accuracy'] for v in val];ax.plot(x,y,marker='o',label=arm)
            ax.set(title=cell,xlabel='Added updates' if axis=='updates' else 'Added examples from this cell',ylabel='Validation BA (%)',ylim=(0,102));ax.legend(fontsize=8)
        fig.savefig(HERE/f'learning_{axis}.png',dpi=130);plt.close(fig)
    print(json.dumps(dict(status=result['status'],arms=list(paths),report=str(HERE/'report.md'))),flush=True)
if __name__=='__main__':main()
