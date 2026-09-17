"""Paired acute-routing effects and exact additive decision-branch decomposition."""
from common import *
import numpy as np
from scipy.stats import rankdata
def metrics(y,p):
    k=p.shape[1];guess=p.argmax(1);cm=np.zeros((k,k),int);np.add.at(cm,(y,guess),1)
    auc=[]
    for j in range(k):
        positive=y==j;n=positive.sum();auc.append((rankdata(p[:,j])[positive].sum()-n*(n+1)/2)/(n*(len(y)-n)))
    return dict(ba=float(np.mean(np.diag(cm)/cm.sum(1))),auc=float(np.mean(auc)),confusion=cm.tolist(),predicted_counts=np.bincount(guess,minlength=k).tolist(),n=len(y))
def main():
    root=HERE/'run';groups={};cfg=read(root/'fixed_config.json')
    for line in (root/'predictions.jsonl').read_text().splitlines():
        r=json.loads(line);groups.setdefault(r['cell'],{}).setdefault(r['arm']+'/'+r['variant'],[]).append(r)
    rng=np.random.default_rng(51973001);cells={};effects={};source_stats={};branches={}
    for cell,variants in groups.items():
        base=variants['attention/baseline'];y=np.array([r['label'] for r in base]);k=len(base[0]['probabilities'])
        ix=np.array([np.concatenate([rng.choice(np.flatnonzero(y==j),sum(y==j),replace=True) for j in range(k)]) for _ in range(1000)])
        draws={};cells[cell]={};source_stats[cell]={};branches[cell]={}
        for name,rows in variants.items():
            assert all(a['episode']==b['episode'] and a['label']==b['label'] and a['metadata']==b['metadata'] for a,b in zip(base,rows))
            p=np.array([r['probabilities'] for r in rows]);logits=np.array([r['logits'] for r in rows]);parts=np.array([r['branch_logits'] for r in rows]);bias=np.array([r['head_bias'] for r in rows]);guess=p.argmax(1)
            cells[cell][name]=metrics(y,p);draws[name]=np.array([np.mean([np.mean(guess[z][y[z]==j]==j) for j in range(k)]) for z in ix]);cells[cell][name]['ba_ci95']=np.quantile(draws[name],[.025,.975]).tolist()
            wrong=logits.copy();wrong[np.arange(len(y)),y]=-np.inf;competitor=wrong.argmax(1)
            margins=parts[np.arange(len(y)),:,y]-parts[np.arange(len(y)),:,competitor]
            biasmargin=bias[np.arange(len(y)),y]-bias[np.arange(len(y)),competitor]
            totalmargin=logits[np.arange(len(y)),y]-logits[np.arange(len(y)),competitor]
            branches[cell][name]=dict(branch_names=['sensory','postupdate_memory','oldmemory_currentfield_comparator'],mean_true_minus_strongest_wrong_branch_margin=margins.mean(0).tolist(),mean_head_bias_margin=float(biasmargin.mean()),mean_total_margin=float(totalmargin.mean()),reconstruction_max_abs=max(r['branch_reconstruction_max_abs'] for r in rows),mean_centered_logits_by_branch=(parts-parts.mean(-1,keepdims=True)).mean(0).tolist(),head_bias=rows[0]['head_bias'],interpretation='Exact additive logit contributions, conditional on final trained features; not branch-removal interventions')
            if rows[0]['stage_attention'] is not None:
                source_stats[cell][name]={stage:np.mean([r['stage_attention'][stage] for r in rows],axis=0).tolist() for stage in rows[0]['stage_attention']}
        effects[cell]={}
        for name in variants:
            if name=='attention/baseline':continue
            difference=draws[name]-draws['attention/baseline'];effects[cell][name]=dict(ba=cells[cell][name]['ba']-cells[cell]['attention/baseline']['ba'],ba_ci95=np.quantile(difference,[.025,.975]).tolist())
    result=dict(status='completed',config=cfg,cells=cells,minus_attention_baseline=effects,stage_attention_head_statistics=source_stats,stage_statistics_columns=['memory_attention_mass','visual_value_contribution_rms','memory_value_contribution_rms'],branches=branches,uncertainty='1000 class-stratified paired bootstrap resamples over independent base episodes; conditions/models reuse identical evidence and do not multiply independent n; same heldout test reused for a diagnostic, not new training/modelselection',caveats=['Acute interventions are out of distribution and do not isolate how benefits were learned.','No rescue does not establish irreversible loss of information.','Memory-source exclusion preserves previous-memory queries and native E/I recurrence, so it does not eliminate all memory influence.','Probe-only memory-input intervention leaves old-memory/current-probe comparator unchanged by construction.','The native query carries an identity cue and is a separate phase from inserted blanks and the visual probe.'])
    write(HERE/'results.json',result)
    lines=['# Frozen pre-update attention mechanism diagnostic','',f"Frozen attention8400 and ordinary continuation8400. Primary prefixes: {cfg['n_primary']} orientation and {cfg['n_primary']} motion episodes, reusing existing heldout evidence; orientationD0 uses {cfg['n_orientation_D0']}. No model parameters, task images, cues, labels, losses or sampling rules changed.",'','| Condition | Model / acute intervention | BA% | AUC | Change from attention baseline pp [95%CI] |','|---|---|---:|---:|---:|']
    for cell,vs in cells.items():
        for name,m in vs.items():
            effect=effects[cell].get(name);change='—' if effect is None else f"{effect['ba']*100:+.2f} [{effect['ba_ci95'][0]*100:+.2f},{effect['ba_ci95'][1]*100:+.2f}]"
            lines.append(f"| {cell} | {name} | {100*m['ba']:.2f} | {m['auc']:.3f} | {change} |")
    lines+=['','## Timing and intervention meaning','','Orientation indices: instruction0; sample1–2; D24 inserted blanks3–26; identity query27; probe/report28. D0 query3 and probe4. Memory-source exclusion sets memory-key logits to negative infinity then renormalizes the same visual keys/values. Previous-memory queries and recurrent E/I state updates remain. Motion instruction0; initial dots reference1; transition-bearing frames2–9; D24 blanks10–33; report34. The motion bypass sends H directly through the inherited memory_input normalization and input convolution on frames2–9 only. The ordinary trained model accepts no intervention metadata.','','The sensory field sequence is cached once per frozen model, then reused unchanged across interventions. This is exact because neither tested model feeds memory into the sensory encoder. Existing sensory and comparator decision routes remain unchanged. CPU checks confirm ordinary/wrapped baseline equality, empty-phase no-op equality, source-mass exclusion and unchanged probe-only comparator contributions. Branch logits decompose the final decision into sensory, postupdate-memory and old-memory/current-field comparator projections, adding classifier bias once.','', '## Scope of inference','']
    lines += ['- '+c for c in result['caveats']]
    lines += ['', 'Stage/head attention mass and value RMS are descriptive; high mass or activity is not evidence that orientation/duration content was preserved. Full confusion matrices, source summaries and component margins are in [results.json](results.json). The separate [saved motion audit](../MotionAudit/report.md) examines historical task allocation, class biases and a validation-only calibration; it is not a new main-model training run.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='analyzed',report=str(HERE/'report.md'))),flush=True)
if __name__=='__main__':main()
