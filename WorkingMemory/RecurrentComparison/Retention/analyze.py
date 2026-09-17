"""Paired saved-score retention curves, clustered across delays and models."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,csv,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PreAttentiveVision.evaluate import auc
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def ba(y,correct):return float(np.mean([correct[y==k].mean() for k in np.unique(y)]))
def macro(y,p):return float(np.mean([auc((y==k).astype(int),p[:,k]) for k in range(p.shape[1])]))
def main():
    tick=time.time();result=read(HERE/'results.json');root=Path(result['run_root']);deadline=read(root/'budget.json')['deadline_unix']
    if result['status']!='completed':raise RuntimeError('Requires completed final tests')
    aa=[json.loads(s) for s in Path(result['parent_test']['predictions']).read_text(encoding='utf-8').splitlines()];bb=[json.loads(s) for s in Path(result['test']['predictions']).read_text(encoding='utf-8').splitlines()]
    assert len(aa)==len(bb)==5120
    for a,b in zip(aa,bb):assert a['metadata']==b['metadata'] and a['label']==b['label'] and a['condition']==b['condition']
    rng=np.random.default_rng(25973001);group_boot={};group_ids={};cell_results={};draws={}
    for name,cell in result['config']['cells'].items():
        a=sorted([r for r in aa if r['condition']==name],key=lambda r:r['paired_base_id']);b=sorted([r for r in bb if r['condition']==name],key=lambda r:r['paired_base_id'])
        y=np.asarray([r['label'] for r in a]);p=np.asarray([r['probabilities'] for r in a]);q=np.asarray([r['probabilities'] for r in b]);ids=[r['paired_base_id'] for r in a];g=cell.get('paired_family') or name
        if g not in group_boot:
            groups=[np.flatnonzero(y==k) for k in range(p.shape[1])]
            group_boot[g]=np.array([np.concatenate([rng.choice(v,len(v),replace=True) for v in groups]) for _ in range(1000)]);group_ids[g]=(ids,y.copy())
        else:assert group_ids[g][0]==ids and np.array_equal(group_ids[g][1],y)
        pc=(p.argmax(1)==y).astype(float);qc=(q.argmax(1)==y).astype(float);boot=[]
        for ix in group_boot[g]:
            pba=ba(y[ix],pc[ix]);qba=ba(y[ix],qc[ix]);pa=macro(y[ix],p[ix]);qa=macro(y[ix],q[ix]);boot.append([pba,qba,qba-pba,pa,qa,qa-pa])
        boot=np.asarray(boot);draws[name]=boot;ci=np.quantile(boot,[.025,.975],axis=0)
        cell_results[name]=dict(n=len(y),class_counts=np.bincount(y,minlength=p.shape[1]).tolist(),delay=cell['condition'].get('delay') if not name.endswith('_anchor') else None,
            parent_ba=ba(y,pc),trained_ba=ba(y,qc),delta_ba=ba(y,qc)-ba(y,pc),parent_ba_ci95=ci[:,0].tolist(),trained_ba_ci95=ci[:,1].tolist(),delta_ba_ci95=ci[:,2].tolist(),
            parent_auc=macro(y,p),trained_auc=macro(y,q),delta_auc=macro(y,q)-macro(y,p),parent_auc_ci95=ci[:,3].tolist(),trained_auc_ci95=ci[:,4].tolist(),delta_auc_ci95=ci[:,5].tolist(),
            both_correct=int(np.sum((pc==1)&(qc==1))),both_wrong=int(np.sum((pc==0)&(qc==0))),parent_only_correct=int(np.sum((pc==1)&(qc==0))),trained_only_correct=int(np.sum((pc==0)&(qc==1))))
        if time.time()>deadline-40:raise TimeoutError('Original cap nearing')
    retention={}
    for g in ('motion','orientation'):
        first=draws[g+'_D0'];last=draws[g+'_D24'];v=last-first
        retention[g]=dict(parent_delay_cost_ba=cell_results[g+'_D24']['parent_ba']-cell_results[g+'_D0']['parent_ba'],trained_delay_cost_ba=cell_results[g+'_D24']['trained_ba']-cell_results[g+'_D0']['trained_ba'],
            parent_delay_cost_ba_ci95=np.quantile(v[:,0],[.025,.975]).tolist(),trained_delay_cost_ba_ci95=np.quantile(v[:,1],[.025,.975]).tolist(),training_by_delay_interaction_ba_ci95=np.quantile(v[:,2],[.025,.975]).tolist(),
            training_by_delay_interaction_ba=cell_results[g+'_D24']['delta_ba']-cell_results[g+'_D0']['delta_ba'])
    primary=[n for n in cell_results if not n.endswith('_anchor')];mean_draws=np.mean([draws[n][:,5] for n in primary],axis=0)
    with (root/'retention/metrics.csv').open(encoding='utf-8') as f:metrics=list(csv.DictReader(f))
    assert len(metrics)==result['config']['additional_updates']
    percell={n:dict(updates=sum(m['condition']==n for m in metrics),episodes=8*sum(m['condition']==n for m in metrics)) for n in cell_results}
    diagnostics=[json.loads(m['diagnostics']) for m in metrics if m['diagnostics']!='{}']
    gradient={}
    for key in ('early_r_gradient_norm','early_a_gradient_norm','late_r_gradient_norm','late_a_gradient_norm','core_gradient_norm','raw_recurrent_gradient_rms'):
        vals=[d[key] for d in diagnostics if d.get(key) is not None]
        gradient[key]=dict(recorded=len(vals),median=float(np.median(vals)) if vals else None,min=float(min(vals)) if vals else None,max=float(max(vals)) if vals else None)
    output=dict(status='completed',cells=cell_results,retention_effects=retention,mean_primary_auc_delta=float(np.mean([cell_results[n]['delta_auc'] for n in primary])),mean_primary_auc_delta_ci95=np.quantile(mean_draws,[.025,.975]).tolist(),
        training=dict(updates=len(metrics),new_episodes=len(metrics)*8,cumulative_endpoint=int(metrics[-1]['episodes']),new_logical_frames=sum(int(m['frames'])*8 for m in metrics),per_cell=percell,
            clipping_fraction=float(np.mean([int(m['clipped']) for m in metrics])),median_preclip_norm=float(np.median([float(m['grad_norm']) for m in metrics])),production_seconds=sum(float(m['step_seconds']) for m in metrics),state_gradients=gradient),
        selected_step=result['selected_step'],selected_new_episodes=(result['selected_step']-9840)*8,unique_evidence_groups=2048,model_delay_presentations=10240,
        paired_metadata_exact=True,bootstrap='1000 paired evidence-group draws stratified by class per family; same draws reused across its4delays and both models; anchors separate',analysis_seconds=time.time()-tick)
    write(HERE/'analysis.json',output);write(root/'analysis.json',output)
    lines=['# Existing E/I retention learning','',f"Completed {output['training']['new_episodes']:,} new training episodes. Validation selected step{result['selected_step']} ({output['selected_new_episodes']:,} new episodes at selection).",'',
        '| Task | Inserted blanks | Parent BA | Trained BA | Change, pp (paired95% CI) | Parent/trained AUC |','|---|---:|---:|---:|---:|---:|']
    for n,c in cell_results.items():
        lo,hi=c['delta_ba_ci95'];lines.append(f"|{n}|{c['delay'] if c['delay'] is not None else 'anchor'}|{100*c['parent_ba']:.2f}%|{100*c['trained_ba']:.2f}%|{100*c['delta_ba']:+.2f} [{100*lo:+.2f},{100*hi:+.2f}]|{c['parent_auc']:.4f}/{c['trained_auc']:.4f}|")
    lines+=['','The main cells share identical evidence/query/probe images across delays. D counts inserted cue-marked blanks: motion age isD+1 after final evidence, orientation sample-to-probe ageD+2, with the established query frame. No calibrated milliseconds or biological retention constant is implied.','',
        'The512 evidence groups per primary family are paired across four delays and two model views. Anchors contribute512 independent examples per family. Bootstrap draws preserve these groups;10240 model-delay presentations are2048 underlying evidence groups, not10240 independent examples. Main classes are balanced; anchor BA is explicitly the mean class recall even when anchor labels are imbalanced.','',
        'Fixed opponent traces remain an additional history-bearing route. Training the core and output together does not isolate which state or learned change produces the retention gains. Only existing recurrent-core parameters, memory_output and motion/orientation heads trained. Frozen encoder/opponent/memory_input/sensory trunk parameters remained exactly unchanged. Recurrent r/a updates retained full sequence gradients; no extra memory architecture or attention was added. Compatible Adam histories and task streams resumed from the selected refitted parent; the schedule is an explicitly versioned migration.','',
        f"Training clipping fraction was{100*output['training']['clipping_fraction']:.2f}%; median preclip norm{output['training']['median_preclip_norm']:.4f}. Per-cell exposures, logical frame counts and early/late state-gradient statistics are in analysis.json. All checkpoints, optimizer/sampler/RNG/scheduler states, pinned source/config, logs and raw predictions are retained.",'',
        'Initial baseline used the development set. All primary delays were included during training; validation selected by mean eight-cell AUC, with anchors descriptive. Final parent/trained comparisons used a separate fresh paired test set after selection. The unchanged parent is the baseline, not an equally exposed alternative training arm. One trained lineage and a finite horizon cannot prove architecture failure, optimality, a unique circuit mechanism or biological correspondence. No extrapolated delay, new condition or automatic extension was run.']
    (HERE/'report.md').write_text('\n'.join(lines),encoding='utf-8');(root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
