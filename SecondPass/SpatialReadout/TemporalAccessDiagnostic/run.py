"""Matched diagnostic access only; no main-network updates, one immutable cap."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import csv
import hashlib
import json
import signal
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
from SecondPass.SpatialReadout.FeatureDiagnostic import run as prior

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'FeatureDiagnostic'
ROOT=OUT.parents[2]
TASKS=prior.TASKS
LAYERS=['gru','readout','early']
ACCESS=['final_only','time_separated']
SEEDS={'orientation_cued':993170000,'spatial_binding':993270000}
KEYS=['early','early_cue','gru','readout','logits','label']
ALPHAS=prior.ALPHAS
DEADLINE=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(name,obj):
    p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False));q.replace(p)

def guard(reserve=30):
    assert DEADLINE is not None,'Numerical work requires persisted cap'
    if time.time()>DEADLINE-reserve:raise TimeoutError('Immutable deadline/report reserve reached')

def component_feature(a,layer,access,task,component,loc=0):
    """No truth metadata accepted. Final-only works with a one-frame bundle."""
    if access=='final_only':t=-1
    elif access=='time_separated':
        t=2 if component=='sample' else (-1 if component=='probe' else (0 if task=='orientation_cued' else 3))
    else:raise ValueError(access)
    if layer=='early':
        if component in ('sample','probe'):return prior.local_feature(a,layer,t,loc)
        return prior.cue_feature(a,layer,t)
    x=a[layer][:,t]
    return x.reshape(len(x),-1).astype('float32')

def load(root,task,split):
    with np.load(root/'features'/f'{task}_{split}.npz') as f:a={k:f[k] for k in KEYS}
    m=json.loads((root/'features'/f'{task}_{split}_metadata.json').read_text())
    return a,prior.target_arrays(m['metadata']),m

def components(state,a):
    layer,access,task=(state[k] for k in ('layer','access','task'))
    pred={}
    for content in ('sample','probe'):
        pred[content]=np.stack([prior.predict_ridge(state['decoders'][f'{content}_{j}'],component_feature(a,layer,access,task,content,j)) for j in range(4)],1)
    for label in ('location','sign'):
        if label in state['decoders']:
            pred[label]=prior.predict_ridge(state['decoders'][label],component_feature(a,layer,access,task,label))
    return pred

def relations(pred):
    cue=np.eye(4)[pred['location'].argmax(1)]
    sign=2*pred['sign'].argmax(1)-1 if 'sign' in pred else np.ones(len(cue))
    return prior.relation_features(pred['sample'],pred['probe'],cue,sign)

def predict(state,a):
    p=components(state,a)
    return prior.predict_ridge(state['calibrator'],relations(p))[:,0],p

def fit(task,layer,access,a,y,v,vy):
    guard(400)
    state=dict(task=task,layer=layer,access=access,decoders={},selections=[],decoder_fit_indices=list(range(512)),calibrator_fit_indices=list(range(512,1024)))
    for content in ('sample','probe'):
        for j in range(4):
            x=component_feature(a,layer,access,task,content,j);xv=component_feature(v,layer,access,task,content,j)
            ang=y[content][:512,j];target=np.c_[np.cos(2*ang),np.sin(2*ang)]
            best=None;grid=[]
            for model,p in prior.ridge_grid(x[:512],target,xv):
                loss=float(prior.axial_error(vy[content][:,j],np.arctan2(p[:,1],p[:,0])/2).mean())
                grid.append(dict(alpha=model['alpha'],validation_mean_axial_error_degrees=loss))
                if best is None or loss<best[0]:best=(loss,model)
            name=f'{content}_{j}';state['decoders'][name]=best[1]
            state['selections'].append(dict(component=name,selected_alpha=best[1]['alpha'],grid=grid))
    for label,k in [('location',4)]+([('sign',2)] if task=='orientation_cued' else []):
        target=y[label] if label=='location' else (y[label]>0).astype(int)
        vt=vy[label] if label=='location' else (vy[label]>0).astype(int)
        x=component_feature(a,layer,access,task,label);xv=component_feature(v,layer,access,task,label)
        best=None;grid=[]
        for model,p in prior.ridge_grid(x[:512],np.eye(k)[target[:512]],xv):
            score=float(balanced_accuracy_score(vt,p.argmax(1)));grid.append(dict(alpha=model['alpha'],validation_ba=score))
            if best is None or score>best[0]:best=(score,model)
        state['decoders'][label]=best[1];state['selections'].append(dict(component=label,selected_alpha=best[1]['alpha'],grid=grid))
    # Calibration features are predictions on FIT-B, never decoder-training predictions.
    cal={k:z[512:] for k,z in a.items()}
    fc=relations(components(state,cal));fv=relations(components(state,v))
    best=None;grid=[]
    for model,p in prior.ridge_grid(fc,y['label'][512:,None],fv):
        score=prior.binary_ba(vy['label'],p[:,0]);grid.append(dict(alpha=model['alpha'],validation_ba=score))
        if best is None or score>best[0]:best=(score,model)
    state['calibrator']=best[1];state['validation_ba']=best[0]
    state['selections'].append(dict(component='calibrator',selected_alpha=best[1]['alpha'],grid=grid))
    torch.save(state,OUT/'models'/f'{task}_{layer}_{access}.pt')
    prior.log(stage='fit',task=task,layer=layer,access=access,validation_ba=best[0])
    return state

def capacity(state):
    rows=[]
    for name,m in list(state['decoders'].items())+[('calibrator',state['calibrator'])]:
        rows.append(dict(component=name,feature_dim=m['feature_dim'],output_dim=int(m['weight'].shape[1]),weight_bias_parameters=int(m['weight'].size+m['bias'].size),scaler_parameters=int(m['mean'].size+m['scale'].size),n_fit=m['n_fit'],alpha_grid=ALPHAS,selection_candidates=len(ALPHAS),n_validation=256))
    return dict(components=rows,weight_bias_parameters=sum(r['weight_bias_parameters'] for r in rows),scaler_parameters=sum(r['scaler_parameters'] for r in rows),selected_models=len(rows),candidate_models=sum(r['selection_candidates'] for r in rows))

def access_invariance(state,a):
    assert state['access']=='final_only'
    original,po=predict(state,a)
    relevant=[state['layer']]+(['early_cue'] if state['layer']=='early' else [])
    removed={k:a[k][:,-1:].copy() for k in relevant}
    corrupt={k:a[k].copy() for k in relevant}
    for z in corrupt.values():z[:,:-1]=np.nan
    for label,b in [('removed',removed),('nan_corrupted',corrupt)]:
        p,pc=predict(state,b)
        np.testing.assert_array_equal(original,p)
        for k in po:np.testing.assert_array_equal(po[k],pc[k])
    return dict(early_removed_bitwise_equal=True,early_nan_corrupted_bitwise_equal=True,all_component_outputs_bitwise_equal=True,only_layer_feature_keys_supplied=relevant,n=len(original),maximum_score_difference=0.)

def bootstrap_metrics(y,ps):
    """Paired, label-stratified whole-episode bootstrap; ranking handles ties."""
    rng=np.random.default_rng(7319501);zero=np.flatnonzero(y==0);one=np.flatnonzero(y==1)
    iz=rng.choice(zero,(2000,len(zero)),replace=True);io=rng.choice(one,(2000,len(one)),replace=True)
    idx=np.concatenate([iz,io],1);out={};draws={}
    for name,p in ps.items():
        ba=((p[iz]<.5).mean(1)+(p[io]>=.5).mean(1))/2
        ranks=rankdata(p[idx],axis=1,method='average')
        auc=(ranks[:,len(zero):].sum(1)-len(one)*(len(one)+1)/2)/(len(zero)*len(one))
        draws[name]=(ba,auc);out[name]=prior.metric(y,p)
        out[name]['ba_bootstrap95']=np.quantile(ba,[.025,.975]).tolist();out[name]['auc_bootstrap95']=np.quantile(auc,[.025,.975]).tolist()
    def contrast(a,b):
        return dict(ba_gain=out[a]['balanced_accuracy']-out[b]['balanced_accuracy'],ba_gain_bootstrap95=np.quantile(draws[a][0]-draws[b][0],[.025,.975]).tolist(),auc_gain=out[a]['auc']-out[b]['auc'],auc_gain_bootstrap95=np.quantile(draws[a][1]-draws[b][1],[.025,.975]).tolist())
    for name in ps:out[name]['versus_deployed']=contrast(name,'deployed')
    pairs={layer:contrast(layer+'_time_separated',layer+'_final_only') for layer in LAYERS}
    return out,pairs

def evaluate(task,fitted,a,y,meta):
    scores={'deployed':prior.softmax(a['logits'])[:,1]};predictions={};diagnostics={};invariance={}
    changes=np.round(prior.axial_error(y['sample'],y['probe']),6)
    target_change=changes[np.arange(len(changes)),y['location']]
    for name,state in fitted.items():
        p,comp=predict(state,a);scores[name]=p
        for k,z in comp.items():predictions[name+'__'+k]=z
        diag={}
        for content in ('sample','probe'):
            ang=np.arctan2(comp[content][...,1],comp[content][...,0])/2
            d=prior.angle_summary(y[content],ang,y['location']);d['by_actual_change']=[]
            error=prior.axial_error(y[content],ang)
            for delta in np.unique(changes):
                for group,mask in [('all',np.ones_like(changes,dtype=bool)),('target',np.arange(4)[None,:]==y['location'][:,None]),('nontarget',np.arange(4)[None,:]!=y['location'][:,None])]:
                    m=mask&(changes==delta)
                    if m.any():d['by_actual_change'].append(dict(absolute_change_degrees=float(delta),group=group,n=int(m.sum()),mean_axial_error_degrees=float(error[m].mean()),fraction_within_7_5=float((error[m]<7.5).mean())))
            diag[content]=d
        for label in ('location','sign'):
            if label not in comp:continue
            target=y[label] if label=='location' else (y[label]>0).astype(int)
            yp=comp[label].argmax(1)
            diag[label]=dict(accuracy=float((yp==target).mean()),balanced_accuracy=float(balanced_accuracy_score(target,yp)),confusion=confusion_matrix(target,yp).tolist(),label_strata=[dict(task_label=i,n=int((y['label']==i).sum()),accuracy=float((yp[y['label']==i]==target[y['label']==i]).mean())) for i in (0,1)])
        diagnostics[name]=diag
        if state['access']=='final_only':invariance[name]=access_invariance(state,a)
    metrics,pairs=bootstrap_metrics(y['label'],scores)
    for name,p in scores.items():
        strata=[]
        groups={'location':y['location'],'label':y['label'],'target_absolute_change_degrees':target_change}
        if task=='orientation_cued':groups['sign']=y['sign']
        for key,z in groups.items():
            for value in np.unique(z):
                m=z==value;r=dict(stratum=key,value=float(value),n=int(m.sum()),accuracy=float(((p[m]>=.5)==y['label'][m]).mean()))
                if len(np.unique(y['label'][m]))==2:r.update(prior.metric(y['label'][m],p[m]))
                strata.append(r)
        metrics[name]['strata']=strata
        predictions['score__'+name]=p
    predictions.update(label=y['label'],location=y['location'],sign=y['sign'],sample_truth=y['sample'],probe_truth=y['probe'],trial_id=np.array([m['trial_id'] for m in meta['metadata']]),raster_sha256=np.array(meta['raster_sha256']))
    np.savez(OUT/'predictions'/f'{task}_test.npz',**predictions)
    result=dict(task=task,metrics=metrics,pairs_time_separated_minus_final_only=pairs,components=diagnostics,invariance=invariance)
    dump(f'{task}_results.json',result)
    return result

def report(results,verification,capacities):
    lines=['# Matched final-only versus time-separated access','', '**Executed frozen terminal6760 diagnostic, native D0 only.** Primary comparison: whole spatial ConvGRU3136. Secondary: whole post-ReLU256 readout. Early visual ROIs are an explanatory reference.','', '## Held-out decisions','512 fresh episodes per task, shared by all readouts and deployed head. BA/AUC intervals are paired label-stratified whole-episode bootstrap (2000 resamples); they condition on these fixed fits and one checkpoint, not training variability.','', '|Task|Layer/access|BA [95%]|AUC [95%]|','|---|---|---|---|']
    for r in results:
        for name,m in r['metrics'].items():
            lines.append(f"|{r['task']}|{name}|{m['balanced_accuracy']:.4f} [{m['ba_bootstrap95'][0]:.4f}, {m['ba_bootstrap95'][1]:.4f}]|{m['auc']:.4f} [{m['auc_bootstrap95'][0]:.4f}, {m['auc_bootstrap95'][1]:.4f}]|")
    lines+=['','### Matched temporal-access effect','Positive differences favor external sample/cue-time access. All feature dimensions, decoder/calibrator parameter counts, example counts, supervision and four-alpha selection opportunities match within each pair.','', '|Task|Layer|BA gain [95%]|AUC gain [95%]|','|---|---|---|---|']
    for r in results:
        for layer,p in r['pairs_time_separated_minus_final_only'].items():
            lines.append(f"|{r['task']}|{layer}|{p['ba_gain']:+.4f} [{p['ba_gain_bootstrap95'][0]:+.4f}, {p['ba_gain_bootstrap95'][1]:+.4f}]|{p['auc_gain']:+.4f} [{p['auc_gain_bootstrap95'][0]:+.4f}, {p['auc_gain_bootstrap95'][1]:+.4f}]|")
    lines+=['','## Measured interpretation']
    for r in results:
        final=r['metrics']['gru_final_only'];temp=r['metrics']['gru_time_separated'];p=r['pairs_time_separated_minus_final_only']['gru']
        lines.append(f"- **{r['task']}, primary whole-field:** final-only BA {final['balanced_accuracy']:.4f}; time-separated BA {temp['balanced_accuracy']:.4f}; paired difference {p['ba_gain']:+.4f} with 95% interval {p['ba_gain_bootstrap95']}. Final-only versus deployed BA gain is {final['versus_deployed']['ba_gain']:+.4f}, interval {final['versus_deployed']['ba_gain_bootstrap95']}.")
    lines+=['- An above-deployed final-only result establishes accessibility to this auxiliary-supervised circular-comparison diagnostic, not acquisition from native task labels or a trained deployed replacement. The time-separated contrast changes access privilege only within a layer. Failure of one finite linear component-decoding scheme does not prove erasure.', '- Across-layer capacities and ROI geometries are unmatched: layerwise differences cannot establish causal compression or irreversible information loss. No new label-only arm or shuffled-label arm was authorized; the earlier control is not treated as a control on these new fits.', '', '## Instruction and orientation precision','All four locations decoded independently; true target is used only for scoring, never input or ROI selection. Errors are axial degrees. Location/sign columns are balanced accuracy.','', '|Task|Layer/access|Sample error° (target / other)|Probe error° (target / other)|Cue location BA|Cue sign BA|','|---|---|---|---|---|---|']
    for r in results:
        for name,d in r['components'].items():
            s,p=d['sample'],d['probe'];sign=f"{d['sign']['balanced_accuracy']:.4f}" if 'sign' in d else 'not defined'
            lines.append(f"|{r['task']}|{name}|{s['mean_degrees']:.2f} ({s['cued_mean_degrees']:.2f} / {s['uncued_mean_degrees']:.2f})|{p['mean_degrees']:.2f} ({p['cued_mean_degrees']:.2f} / {p['uncued_mean_degrees']:.2f})|{d['location']['balanced_accuracy']:.4f}|{sign}|")
    lines+=['','Orientation changes are 0/±15/±30/±45°; 7.5° is half the smallest nonzero separation. Binding target changes are 0/45/90° axially. `*_results.json` includes actual-change-stratified sample/probe errors, fractions below7.5°, per-location cued/uncued errors, instruction label strata, and decision target-change/sign/location/label strata. Predictions retain truth for scoring separately from the input-only prediction API.','', '## Matched capacity and fitting','|Task|Layer (each access condition)|Weights+biases|Scaler values|Selected fits|Alpha candidates|','|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for layer in LAYERS:
            c=capacities[task][layer]
            lines.append(f"|{task}|{layer}|{c['weight_bias_parameters']}|{c['scaler_parameters']}|{c['selected_models']}|{c['candidate_models']}|")
    lines+=['', '- Whole ConvGRU3136 and readout256 enter every component decoder. Early angle inputs are the identical prior fixed32×5×5 pooled local ROI (800 values); early cue inputs concatenate four raw32×6×6 glyph ROIs and32×5×5 global pooled values (5408). No target-selected crop. The relational calibrator receives35 values in every arm.', '- Each location/sample/probe predicts cos(2θ), sin(2θ). Cue predicts one-hot location and, for orientation, sign. Binding sign is the task constant +1, not a fitted or metadata-supplied instruction. Hard predicted cue/sign enter the exact prior circular relation: normalized axial dot/cross products, absolute/signed half-angle, sign-aligned angle and absolute cross; selected-target relationships, all-location relationships, predicted cue and sign feed a ridge task-label calibrator.', '- Reused immutable train1024 and validation256 per task. TRAIN first512 fit components; TRAIN last512 fit calibration using out-of-component-fit predictions. The same validation256 choose α∈{0.1,1,10,100} independently for each component and calibrator, first grid entry wins ties. Train-only column standardization, feature-width-normalized exact dual ridge; no PCA, no train/validation refit. Float16 features promoted to float32; ridge algebra float64, saved weights/scalers float32.', '- Both tasks and all12 structured models were fit, serialized and hashed before generating any fresh test episode. Test seeds and counts are pinned in protocol.json. No old test predictions are loaded; old metadata/hashes are read only for exclusion. Fresh1024 episode IDs and full-raster hashes are disjoint from all3584 previous FeatureDiagnostic train/validation/test episodes and each other.', '- final_only uses report features for every decoder. time_separated uses sample2, report probe, orientation cue0 or binding query3. Stored earlier activations are external diagnostic memory. Six actual final-only test predictions and every intermediate component were bitwise invariant after earlier timesteps were removed and separately filled with NaNs; the input bundle contained no labels, logits, metadata or other layers.', '', '## Verification and limits', f"- Main checkpoint, saved source dependencies, and all prior diagnostic files unchanged: {verification['hashes_unchanged']}. Main-model state tensors bitwise unchanged: {verification['state_unchanged']}. Native direct/extraction wrapper maximum error: {verification['wrapper_max_abs_error']}. Zero main-model optimizer updates; eval/no-grad only.", '- Independent artifact replay reopens all12 fit states and test features, recomputes component outputs, decisions, BA/AUC and paired bootstrap intervals, rechecks capacities, fit separation, invariance and hashes. See independent_verification.json; completion.json is written only after replay and this report finish.', '- Single local worker; CPU≤2, interop1; MPS used only for fresh frozen test extraction. One immutable1200s wall cap begins before the first numerical fit and includes extraction, scoring, replay and reporting. Exact elapsed time is in completion.json. No restart, cloud use, model update or architecture experiment.', '- **D0 stack3 is not clean retention.** Orientation report3 contains sample1/sample2/probe; binding report4 contains sample2/query3/probe. Removing early *diagnostic feature access* is not removing those samples from the native input stack or recurrent computation. No long-delay memory, biology, attention allocation, cue-validity benefit, inhibition or microstimulation conclusion is measured here.', '- Bootstrap intervals are descriptive, uncorrected for multiple secondary comparisons and do not model native balancing-queue dependence. At a perfect score a nonparametric bootstrap degenerates; saved accuracy Wilson intervals retain finite-sample uncertainty. Ridge outputs are decision scores (threshold0.5), not calibrated probabilities.', '', '## Artifacts and reproduction', '`run.py`, `test_access.py`, `verify.py`, `protocol.json`, `prior_artifacts.json`, `models/*.pt`, `features/*`, `predictions/*`, `*_results.json`, `results.json`, `capacity.json`, `selection_frozen.json`, `verification.json`, `independent_verification.json`, `budget.json`, `completion.json`, `summary.csv`, `progress.jsonl`.', '', 'Run from repo root: `/tmp/vawm-task-suite-venv/bin/python -B -m SecondPass.SpatialReadout.TemporalAccessDiagnostic.run --run`. Existing budget deliberately prevents rerunning; a separately authorized new output directory and cap would be required. `--precheck` and `test_access` do CPU setup/routing checks only. Replay is executable via `verify` only while the same cap remains active; no renewal. Prior artifacts and LabJournal were not edited.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['task','arm','n','BA','BA95_lo','BA95_hi','AUC','AUC95_lo','AUC95_hi'])
        for r in results:
            for name,m in r['metrics'].items():w.writerow([r['task'],name,m['n'],m['balanced_accuracy'],*m['ba_bootstrap95'],m['auc'],*m['auc_bootstrap95']])

def precheck():
    assert not (OUT/'budget.json').exists(),'Run cap already exists; no renewal'
    assert sha(prior.CKPT)==prior.EXPECTED
    manifest=json.loads((OLD/'manifest.json').read_text())
    old_verification=json.loads((OLD/'verification.json').read_text())
    assert old_verification['checkpoint_sha256']==prior.EXPECTED
    assert old_verification['state_unchanged'] and old_verification['hashes_unchanged']
    assert all(sha(ROOT/p)==h for p,h in old_verification['source_hashes'].items())
    assert manifest['counts']==dict(train=1024,val=256,test=512)
    for rec in manifest['features']:
        assert sha(rec['path'])==rec['sha256']
        meta=json.loads((OLD/'features'/f"{rec['task']}_{rec['split']}_metadata.json").read_text())
        assert len(meta['metadata'])==rec['n']
        assert meta['feature_shapes']['gru'][2:]==[64,7,7]
        assert meta['feature_shapes']['readout'][2:]==[256]
        assert meta['wrapper_max_abs_errors']==[0.0]
        assert rec['seed'] not in SEEDS.values()
    assert torch.backends.mps.is_available()
    return manifest

def main():
    global DEADLINE
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--precheck',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);threadpool_limits(2)
    manifest=precheck()
    if args.precheck:print('CPU precheck passed: pinned checkpoint, all prior feature hashes/counts/shapes, MPS available; no fit/accelerator work.');return
    assert args.run
    for d in ('models','features','predictions'):(OUT/d).mkdir(exist_ok=True)
    # Import inspected: no execution at import. Redirect every reused write and guard in MEMORY only.
    prior.OUT=OUT;prior.sha=sha
    oldhash={str(p):sha(p) for p in OLD.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='.DS_Store'}
    dump('prior_artifacts.json',oldhash)
    source_names=['SecondPass/SpatialReadout/model.py','WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py','WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/stimuli.py','SecondPass/TaskSuite/suite.py','SecondPass/TaskSuite/catalog.json','SecondPass/SpatialReadout/FeatureDiagnostic/run.py']
    sources={str(ROOT/p):sha(ROOT/p) for p in source_names}
    sources.update({str(p):sha(p) for p in OUT.glob('*.py')})
    cp=torch.load(prior.CKPT,map_location='cpu',weights_only=False)
    model=prior.SpatialReadout(prior.task_classes());model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    protocol=dict(checkpoint=str(prior.CKPT),checkpoint_sha256=prior.EXPECTED,step=6760,tasks=TASKS,layers=LAYERS,access=ACCESS,train=1024,decoder_fit=[0,512],calibration_fit=[512,1024],validation=256,fresh_test_per_task=512,test_seeds=SEEDS,alphas=ALPHAS,threshold=.5,bootstrap=dict(seed=7319501,resamples=2000,method='paired label-stratified whole-episode percentile'),cap_seconds=1200,renewable=False,main_updates=0,source_hashes=sources,train_val_feature_records=[r for r in manifest['features'] if r['split']!='test'])
    dump('protocol.json',protocol)
    start=time.time();DEADLINE=start+1200;prior.DEADLINE=DEADLINE
    with (OUT/'budget.json').open('x') as f:json.dump(dict(start_unix=start,deadline_unix=DEADLINE,cap_seconds=1200,renewable=False,pid=os.getpid(),first_numerical_work='component ridge fit after train/val read',cpu_threads=2,interop_threads=1,mps_workers=1),f,indent=2)
    def alarm(*args):
        dump('cap_reached.json',dict(elapsed=time.time()-start));os._exit(124)
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,1200)
    fitted={};capacities={}
    for task in TASKS:
        a,y,_=load(OLD,task,'train');v,vy,_=load(OLD,task,'val');fitted[task]={};capacities[task]={}
        for layer in LAYERS:
            for access in ACCESS:fitted[task][layer+'_'+access]=fit(task,layer,access,a,y,v,vy)
            left=capacity(fitted[task][layer+'_final_only']);right=capacity(fitted[task][layer+'_time_separated'])
            assert left==right;capacities[task][layer]=left
        del a,v
    dump('capacity.json',capacities)
    frozen=dict(unix=time.time(),elapsed=time.time()-start,all_selections_frozen=True,test_generated=False,test_loaded=False,protocol_sha256=sha(OUT/'protocol.json'),model_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'models').glob('*.pt')},selections={task:{name:state['selections'] for name,state in states.items()} for task,states in fitted.items()})
    assert len(frozen['model_hashes'])==12;dump('selection_frozen.json',frozen)
    dump('test_start.json',dict(unix=time.time(),selection_frozen_sha256=sha(OUT/'selection_frozen.json')))
    guard(400);model=model.to('mps');features=[]
    for task in TASKS:features.append(prior.generate(model,task,'test',512,SEEDS[task]))
    torch.mps.synchronize()
    unchanged=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in initial.items());assert unchanged
    del model,cp,initial;torch.mps.empty_cache()
    oldids=set();oldrasters=set()
    for p in (OLD/'features').glob('*_metadata.json'):
        m=json.loads(p.read_text());oldids.update(z['trial_id'] for z in m['metadata']);oldrasters.update(m['raster_sha256'])
    ids=[];rasters=[]
    for task in TASKS:
        m=json.loads((OUT/'features'/f'{task}_test_metadata.json').read_text());ids.extend(z['trial_id'] for z in m['metadata']);rasters.extend(m['raster_sha256'])
    assert len(ids)==len(set(ids))==1024 and len(rasters)==len(set(rasters))==1024
    assert not set(ids)&oldids and not set(rasters)&oldrasters
    results=[]
    for task in TASKS:
        guard(100);a,y,m=load(OUT,task,'test');results.append(evaluate(task,fitted[task],a,y,m));del a
    assert sha(prior.CKPT)==prior.EXPECTED
    assert all(sha(p)==h for p,h in sources.items())
    assert all(sha(p)==h for p,h in oldhash.items())
    verification=dict(hashes_unchanged=True,state_unchanged=unchanged,wrapper_max_abs_error=max(e for rec in features for e in rec['parity']),main_model_updates=0,prior_ids=len(oldids),prior_rasters=len(oldrasters),fresh_ids=len(ids),fresh_rasters=len(rasters),no_overlap_prior_all_splits=True,all_pairs_capacity_matched=True,all_final_invariance_passed=True,features=features,source_hashes=sources,selection_precedes_test=frozen['unix']<json.loads((OUT/'test_start.json').read_text())['unix'],elapsed_seconds=time.time()-start)
    dump('verification.json',verification);dump('results.json',results)
    from .verify import verify
    verify()
    report(results,verification,capacities)
    guard(0)
    dump('completion.json',dict(status='complete',tasks=2,structured_models=12,fresh_test_per_task=512,elapsed_seconds=time.time()-start,finished_unix=time.time(),within_cap=True,artifact_replay='passed',pid=os.getpid(),report_sha256=sha(OUT/'REPORT.md')))
    signal.setitimer(signal.ITIMER_REAL,0);prior.log(stage='complete',elapsed_seconds=time.time()-start)

if __name__=='__main__':main()
