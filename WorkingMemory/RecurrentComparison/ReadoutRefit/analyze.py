"""Saved paired predictions and training records; no model inference."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json,csv,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PreAttentiveVision.evaluate import auc
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def macro(y,p):return float(np.mean([auc((y==k).astype(int),p[:,k]) for k in range(p.shape[1])]))
def main():
    start=time.time();result=read(HERE/'results.json');root=Path(result['run_root']);deadline=read(root/'budget.json')['deadline_unix']
    if result['status']!='completed':raise RuntimeError('Requires complete paired final tests')
    a=[json.loads(s) for s in Path(result['parent_test']['predictions']).read_text().splitlines()];b=[json.loads(s) for s in Path(result['test']['predictions']).read_text().splitlines()]
    assert len(a)==len(b)==3072
    for x,y in zip(a,b):assert x['metadata']==y['metadata'] and x['label']==y['label'] and x['condition']==y['condition']
    rows=[];all_auc_draws=[];rng=np.random.default_rng(20953001)
    for name in result['config']['cells']:
        aa=[r for r in a if r['condition']==name];bb=[r for r in b if r['condition']==name]
        y=np.array([r['label'] for r in aa]);p=np.array([r['probabilities'] for r in aa]);q=np.array([r['probabilities'] for r in bb]);pc=(p.argmax(1)==y).astype(float);qc=(q.argmax(1)==y).astype(float)
        classes=[np.flatnonzero(y==k) for k in range(p.shape[1])];boot=[]
        for _ in range(1000):
            ix=np.concatenate([rng.choice(c,len(c),replace=True) for c in classes]);pa=macro(y[ix],p[ix]);qa=macro(y[ix],q[ix]);boot.append([pc[ix].mean(),qc[ix].mean(),(qc[ix]-pc[ix]).mean(),pa,qa,qa-pa])
        boot=np.array(boot);all_auc_draws.append(boot[:,5]);ci=np.quantile(boot,[.025,.975],axis=0)
        rows.append(dict(condition=name,n=len(y),parent_ba=float(pc.mean()),refit_ba=float(qc.mean()),delta_ba=float((qc-pc).mean()),parent_ba_ci95=ci[:,0].tolist(),refit_ba_ci95=ci[:,1].tolist(),delta_ba_ci95=ci[:,2].tolist(),
            parent_auc=macro(y,p),refit_auc=macro(y,q),delta_auc=macro(y,q)-macro(y,p),parent_auc_ci95=ci[:,3].tolist(),refit_auc_ci95=ci[:,4].tolist(),delta_auc_ci95=ci[:,5].tolist(),
            both_correct=int(np.sum((pc==1)&(qc==1))),both_wrong=int(np.sum((pc==0)&(qc==0))),parent_only_correct=int(np.sum((pc==1)&(qc==0))),refit_only_correct=int(np.sum((pc==0)&(qc==1)))))
        if time.time()>deadline-30:raise TimeoutError('Analysis nearing original deadline')
    with (root/'readout_refit/metrics.csv').open() as f:metrics=list(csv.DictReader(f))
    cfg=result['config'];assert len(metrics)==cfg['additional_updates']
    training=dict(updates=len(metrics),new_episodes=len(metrics)*8,cumulative_endpoint=int(metrics[-1]['episodes']),clipping_fraction=float(np.mean([int(m['clipped']) for m in metrics])),median_preclip_norm=float(np.median([float(m['grad_norm']) for m in metrics])),observed_production_seconds=sum(float(m['step_seconds']) for m in metrics),per_cell={})
    for name in cfg['cells']:
        rs=[r for r in metrics if r['condition']==name];n=min(100,len(rs)//2)
        training['per_cell'][name]=dict(updates=len(rs),new_episodes=len(rs)*8,first_loss_mean=float(np.mean([float(r['loss']) for r in rs[:n]])),last_loss_mean=float(np.mean([float(r['loss']) for r in rs[-n:]])),loss_window_updates=n)
    output=dict(status='completed',paired_cells=rows,mean_six_cell_auc_delta=float(np.mean([r['delta_auc'] for r in rows])),mean_six_cell_auc_delta_ci95=np.quantile(np.mean(all_auc_draws,axis=0),[.025,.975]).tolist(),training=training,
        selected_step=result['selected_step'],selected_additional_updates=result['selected_step']-5000,selected_additional_episodes=(result['selected_step']-5000)*8,
        bootstrap='1000 paired episode draws stratified by true class within each cell; uncertainty conditional on selected trained checkpoint',all_paired_metadata_exact=True,
        scope='Readout-only before/after versus untouched parent, not comparison with equally exposed full-model training',analysis_seconds=time.time()-start)
    write(HERE/'analysis.json',output);write(root/'analysis.json',output)
    lines=['# Existing readout refit: held-out results','',f"Trained {training['new_episodes']:,} new standard-task episodes with sensory and recurrent dynamics frozen. Validation selected step{result['selected_step']} ({output['selected_additional_episodes']:,} new episodes at that checkpoint).",'',
        '| Standard task cell | Parent BA | Refit BA | Change, percentage points (paired95% CI) | Parent/refit AUC |','|---|---:|---:|---:|---:|']
    for row in rows:
        lo,hi=row['delta_ba_ci95'];lines.append(f"|{row['condition']}|{100*row['parent_ba']:.2f}%|{100*row['refit_ba']:.2f}%|{100*row['delta_ba']:+.2f} [{100*lo:+.2f},{100*hi:+.2f}]|{row['parent_auc']:.4f}/{row['refit_auc']:.4f}|")
    lines+=['',f"The mean six-cell AUC changed by{output['mean_six_cell_auc_delta']:+.4f}, paired95% CI{output['mean_six_cell_auc_delta_ci95']}. Different binary/four-way chance accuracies are not pooled into one raw headline BA.",'',
        'Both model views used exactly the same3072 fresh held-out standard-task movies; complete metadata matched. No validation or test example was used for optimizer fitting. Confidence intervals resample complete paired episodes, stratified by class within each cell; they quantify this selected-checkpoint comparison, not training-seed variation or the uncertainty of an architecture search.', '',
        'Only existing memory_output plus motion/orientation linear heads trained (33,670 parameters). All sensory and recurrent parameters remained exactly unchanged at every training-block check. Parent Adam moments, sampler and RNG resumed in a versioned readout-only protocol. The separate profiling updates were excluded from production initialization/exposure. The selected result can precede the endpoint; both counters are recorded.', '',
        f"Training clipped{100*training['clipping_fraction']:.2f}% of updates; median preclip norm{training['median_preclip_norm']:.4f}. Measured production collection/inference/output-fit time was{training['observed_production_seconds']:.2f}seconds. Full GPU worker/supervisor/evaluation costs are in the run budget/exit receipts. No full sequence BPTT was performed because upstream computation was fixed.",'',
        'This is a targeted readout-fitting intervention compared with the unchanged EI parent. It does not establish superiority over equally exposed end-to-end continuation, a biological memory mechanism, or an adequate horizon for every task. The preceding controlled StateDiagnostic motivated this branch; its suffix-matched task distribution and raw-score AUC differ from this standard-task softmax-score evaluation.', '',
        'See analysis.json for paired effects, intervals, error overlap and training exposure; results.json for validation curves, class confusions, per-class recall and binary hit/miss/false-alarm/correct-rejection statistics. Immutable checkpoints, original source/config snapshots, metrics and raw predictions remain in the run directory. No new experiment is launched automatically.']
    (HERE/'report.md').write_text('\n'.join(lines));(root/'report.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
