"""Independent no-fit replay, group bootstrap, report and journal emitter."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import json,time,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;SRC=OUT.parent/'FrozenDiagnostic';ROOT=OUT.parents[4]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def balanced(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))
def auc(y,s):
    a=s[y==1];b=s[y==0]
    return float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))
def verify_and_report(deadline):
    assert time.time()<deadline
    contract=json.loads((OUT/'split_contract.json').read_text());assert not set(contract['component_groups'])&set(contract['calibration_groups'])
    models=np.load(OUT/'fitted_models.npz');selection=json.loads((OUT/'selection.json').read_text());pred=np.load(OUT/'test_predictions.npz')
    freeze=json.loads((OUT/'test_freeze.json').read_text());assert freeze['fits_sha256']==sha(OUT/'fitted_models.npz') and freeze['selection_sha256']==sha(OUT/'selection.json')
    p=np.load(OUT/'projection.npz')['p'];data=np.load(SRC/'test_features.npz');meta=json.loads((SRC/'test_metadata.json').read_text())
    joint=np.array([2 if m['changed_patch'] is None else m['changed_patch'] for m in meta]);event=(joint<2).astype(int);cue=np.array([m['target_location'] for m in meta]);label=np.array([m['label'] for m in meta]);side=np.minimum(joint,1)
    truth=dict(joint=joint,event=event,cue=cue,label=label,side=side)
    scoreinputs=np.load(OUT/'scoring_inputs.npz')
    for k,v in truth.items():assert np.array_equal(v,scoreinputs['truth/'+k])
    def evaluate(key,x):
        return ((x-models[key+'__mean'])/models[key+'__scale'])@models[key+'__weight']+models[key+'__intercept']
    def two(v):return np.column_stack([1-v,v])
    maxerr=0.
    for kind in ['final','phases']:
        ph=['final']*3 if kind=='final' else ['cue','baseline','post']
        parts=[np.roll(data['memory__'+t],1045*i,axis=1)@p for i,t in enumerate(ph)];z=np.hstack(parts);ii,jj=np.triu_indices(96);x=np.hstack([z,z[:,ii]*z[:,jj]])
        got={t:evaluate(kind+'/'+t,x) for t in ['cue','event','side','joint','label','shuffled_label']}
        c,e,s=[np.clip(got[t][:,1],0,1) for t in ['cue','event','side']];hc=(c>=.5).astype(int);he=(e>=.5).astype(int);hs=(s>=.5).astype(int)
        got['hard']=two((he&(hc==hs)).astype(float));got['soft']=two(e*((1-c)*(1-s)+c*s))
        got['calibrated']=evaluate(kind+'/calibrator',np.column_stack([c,e,s,c*e,c*s,e*s,c*e*s]))
        for name,cc,ee,ss in [('oracle_cue',cue,he,hs),('oracle_event',hc,event,hs),('oracle_side_on_events',hc,he,np.where(event,side,hs)),('oracle_event_side',hc,event,side),('oracle_all',cue,event,side)]:got[name]=two(ee*(cc==ss))
        for name,v in got.items():
            diff=float(np.max(np.abs(v-pred[kind+'/'+name])));maxerr=max(maxerr,diff);assert diff<1e-12
    assert np.array_equal(pred['native/label'],data['native_logits'])
    # One shared bootstrap draw per independent physical episode, keeping variants together.
    draws=np.random.default_rng(60631).integers(0,175,(500,175));metrics={};boots={};pair={}
    for key in pred.files:
        target=key.split('/')[1];target=target if target in truth else 'label'
        y=truth[target];scores=pred[key];dec=scores.argmax(1)
        for population in ['native','paired']:
            indices=np.arange(0,350,2) if population=='native' else np.arange(350)
            if target=='side':indices=indices[event[indices]==1]
            yy=y[indices];dd=dec[indices];ss=scores[indices];m=dict(n_rows=len(indices),n_groups=len(np.unique(indices//2)),balanced_accuracy=balanced(yy,dd),accuracy=float(np.mean(yy==dd)),confusion=[[int(np.sum((yy==i)&(dd==j))) for j in range(scores.shape[1])] for i in range(scores.shape[1])])
            values=[];aucs=[]
            for groups in draws:
                rows=2*groups if population=='native' else (2*groups[:,None]+np.arange(2)).ravel()
                if target=='side':rows=rows[event[rows]==1]
                values.append(balanced(y[rows],dec[rows]))
                if scores.shape[1]==2:aucs.append(auc(y[rows],scores[rows,1]-scores[rows,0]))
            m['ba_ci95']=np.quantile(values,[.025,.975]).tolist();boots[key+'/'+population]=np.array(values)
            if scores.shape[1]==2:m.update(auc=auc(yy,ss[:,1]-ss[:,0]),auc_ci95=np.quantile(aucs,[.025,.975]).tolist())
            metrics[key+'/'+population]=m
        if target=='label':
            cp=dec.reshape(-1,2);yt=label.reshape(-1,2);change=event[::2]==1;score=scores[:,1]-scores[:,0];aligned=(score[1::2]-score[::2])*(label[1::2]-label[::2]);correct=np.all(cp==yt,axis=1)
            pair[key]=dict(groups=175,event_groups=int(change.sum()),action_flips=int(np.sum(cp[:,0]!=cp[:,1])),both_correct=int(correct.sum()),both_correct_event=int(correct[change].sum()),task_aligned_margin_mean=float(aligned[change].mean()),task_aligned_margin_ci95=np.quantile([aligned[g][change[g]].mean() for g in draws],[.025,.975]).tolist())
    deltas={}
    for population in ['native','paired']:
        for t in ['cue','event','side','joint','label','hard','soft','calibrated']:
            a='phases/'+t+'/'+population;b='final/'+t+'/'+population;deltas['phases_minus_final/'+t+'/'+population]=dict(ba_difference=metrics[a]['balanced_accuracy']-metrics[b]['balanced_accuracy'],ci95=np.quantile(boots[a]-boots[b],[.025,.975]).tolist())
        for k in ['final','phases']:
            for t in ['hard','soft','calibrated']:
                a=k+'/'+t+'/'+population;b=k+'/label/'+population;deltas[k+'/'+t+'_minus_direct/'+population]=dict(ba_difference=metrics[a]['balanced_accuracy']-metrics[b]['balanced_accuracy'],ci95=np.quantile(boots[a]-boots[b],[.025,.975]).tolist())
    dump(OUT/'metrics.json',dict(scores=metrics,paired_cue_challenge=pair,paired_gains=deltas));np.savez(OUT/'bootstrap_scores.npz',**boots)
    dump(OUT/'verification.json',dict(independent_prediction_replay_max_abs=maxerr,all_scores_recomputed_from_saved_predictions=True,component_calibrator_groups_disjoint=True,split_noncue_hashes_disjoint=True,final_access_remove_and_nan_invariance=True,oracle_all_perfect=all(metrics[k+'/oracle_all/paired']['balanced_accuracy']==1 for k in ['final','phases']),test_reused=True))
    def row(k):
        m=metrics[k];return f"{m['balanced_accuracy']:.3f} [{m['ba_ci95'][0]:.3f}, {m['ba_ci95'][1]:.3f}]"
    lines=['# Factorized frozen diagnostic — selected2297','', '**Exploratory reused-test follow-up, not a new independent confirmation.** No neural extraction, deployed training, stimulus changes or cloud work.','', '## Measured accessibility', '', 'Balanced accuracy [95% independent-group bootstrap interval]. Native = only original untouched episodes; paired = both cue variants, not twice the independent sample.','', '| Target / comparator | Final native | Phase native | Final paired | Phase paired |','|---|---:|---:|---:|---:|']
    for t in ['event','side','joint','cue','label','hard','soft','calibrated','shuffled_label']:
        lines.append('| '+t+' | '+' | '.join(row(k+'/'+t+'/'+pop) for pop in ['native','paired'] for k in ['final','phases'])+' |')
    lines+=['','Event: catch versus any physical change; side: left versus right **conditional on a real event** (evaluation/training subset only, never an operational input). Joint: left/right/catch. Label: target-only report, not any-change detection. Native n175 (97 target,52 foil,26 catch); event-side n149. Paired n350 rows/175 groups, event-side298 rows/149 groups. AUC, confusion, exact denominators and paired-difference intervals are in metrics.json.','', '## Diagnostic oracle substitutions — NOT deployable', '', 'Hard supplied relation: event AND (changed side = cue). One truth substitution at a time; these are diagnostic ideal-component ceilings, not operational scores or guaranteed monotonic upper bounds. Side substitution applies only to true events; predicted side remains on catches. No truth enters hard/soft/calibrated operational inference.','', '| Substitution | Final native | Phase native | Final paired | Phase paired |','|---|---:|---:|---:|---:|']
    for t in ['oracle_cue','oracle_event','oracle_side_on_events','oracle_event_side','oracle_all']:
        lines.append('| '+t+' | '+' | '.join(row(k+'/'+t+'/'+pop) for pop in ['native','paired'] for k in ['final','phases'])+' |')
    lines+=['','## Paired cue challenge','','| Operational output | Flips /175 | Both correct /175 | Both correct event /149 |','|---|---:|---:|---:|']
    for k in ['native/label','final/label','phases/label','final/hard','phases/hard','final/calibrated','phases/calibrated']:
        v=pair[k];lines.append(f"| {k} | {v['action_flips']} | {v['both_correct']} | {v['both_correct_event']} |")
    lines+=['','## Protocol, interpretation and limitations','', 'Split contracts verified before fitting:425train/100validation/175test independent physical groups with two matched cue variants. Fixed RNG splits training into300 component/direct-probe groups and125 disjoint calibrator groups. Every calibrator training prediction is from a component decoder that never saw that group (including scaling). Components are not refitted afterward. Validation-only alpha selection from1/100/10000, first ties; fixed0.5 decisions for hard/soft. Label-only and group-shuffled controls use the same300 groups. No test-guided choice or sweep.','', 'Final and external phase accesses each use the same4752 quadratic features:96 projected coordinates plus upper-triangle products. Projection is fixed/reused, not supervised. Final uses three fixed permutations of final memory; phase uses cue/baseline/post memory. Same coefficients, fit groups and alpha grids across access pairs. Factorized route has extra auxiliary supervision, three component decoders and seven calibration inputs, unlike the direct label probe; its capacity is matched across temporal access, not claimed equal to direct label-only. Raw ridge class1 scores are clipped to[0,1] for composition and are not claimed calibrated probabilities. Learned calibration uses seven multilinear terms and native target labels on separate groups.','', 'Phase access is external diagnostic memory unavailable to the deployed head. Truth never selects a patch or feature. Input-only final prediction was invariant with all earlier arrays removed and independently poisoned with NaNs, including every component and calibrated output. Checkpoint/source/extraction/feature hashes matched original receipts; metadata hashes recorded. Saved fits independently reconstruct all predictions and all score summaries are recomputed.500 independent-group resamples retain cue pairs; intervals are descriptive, unadjusted for multiple exploratory comparisons and conditional on fixed learned probes. Natural-frequency metrics use only original variant0; counterfactual cue-pair frequencies are not native.','', 'Probe failure is not erasure. Feature projection, small catch counts, ridge capacity, sample size, and auxiliary teaching can limit decoding. Success shows accessible/composable information, not native learned use, a causal lesion, biological attention or a validated architecture remedy. All oracle results are nondeployable. One selected checkpoint only; classical validity benefits, causal inhibition/microstimulation and response timing remain unmeasured.','', '## Reproduction','', 'Use `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python replay.py` for no-fit replay within the recorded deadline. `factorized.py` refuses any existing budget; re-fitting requires a separately authorized empty output and new cap. `test_contract.py` checks input-only final access and relation semantics without fitting. Artifacts: protocol/input_receipt/split_contract/budget/selection/test_freeze/fitted_models/projection/test_predictions/scoring_inputs/bootstrap_scores/metrics/verification/completion, plus these scripts. Original features remain in sibling FrozenDiagnostic/. New nonrenewable600-second cap starts at first fit; at most two CPU numerical threads, no main-model optimizer steps.']
    narrative=(OUT/'INTERPRETATION.md').read_text() if (OUT/'INTERPRETATION.md').exists() else ''
    report='\n'.join(lines[:4])+'\n'+narrative+'\n'+'\n'.join(lines[4:])+'\n';(OUT/'REPORT.md').write_text(report)
    journal='# Factorized frozen Krauzlis diagnostic — selected2297\n\nExploratory reused-test follow-up; not new independent confirmation.\n\n'
    for k in ['event','side','joint','cue','label','calibrated']:
        journal+=f"- {k}, native BA: final {row('final/'+k+'/native')}; external phase {row('phases/'+k+'/native')}.\n"
    journal+='\n300 component-fit and125 disjoint calibrator-fit training groups;100 validation and175 reused-test physical groups. Operational comparator sees predicted cue/event/side only; no in-sample component outputs in calibration. Matched temporal capacity, train-only scaling, validation-only selection,500 group bootstraps, shuffled-pair control and independent fitted-model replay. Original untouched native episodes scored separately from350 counterfactual-pair rows. No neural extraction, training, cloud or stimulus change. Probe failure is not erasure; oracle substitutions are nondeployable diagnostics.\n\n[Full report and artifacts](../SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/FactorizedDiagnostic/REPORT.md).\n'
    journal+='\n'+narrative
    (ROOT/'LabJournal/krauzlis-factorized-diagnostic.md').write_text(journal)
    entry='\n**Frozen selected2297 factorized follow-up completed — exploratory reused-test.** [Results](krauzlis-factorized-diagnostic.md). Original native n175, paired challenge175 groups. Disjoint component/calibrator fitting; matched final/external-phase access; fitted-model replay and grouped uncertainty. No model/stimulus/cloud changes.\n'
    for f in ['CURRENT_STATUS.md','CHRONOLOGY.md','README.md']:
        path=ROOT/'LabJournal'/f;text=path.read_text()
        if entry.strip() not in text:
            first,rest=text.split('\n',1);path.write_text(first+'\n'+entry+rest)
    assert time.time()<deadline
    return {k:v for k,v in metrics.items() if k.endswith('/native') and ('oracle' not in k)}
if __name__=='__main__':
    deadline=json.loads((OUT/'budget.json').read_text())['deadline']
    print(json.dumps(verify_and_report(deadline),indent=2))
