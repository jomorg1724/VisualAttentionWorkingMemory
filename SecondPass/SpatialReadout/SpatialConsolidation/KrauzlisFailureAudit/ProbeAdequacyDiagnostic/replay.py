"""No-fit replay: exact serialized predictions, metrics, capacity, identities."""
from adequacy import *
from run_upstream import targets,change_from_scores,change_error,metrics_binary

def main():
    budget=json.loads((OUT/'budget.json').read_text());assert time.time()<budget['deadline'],'original cap expired'
    signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('original cap')));signal.setitimer(signal.ITIMER_REAL,budget['deadline']-time.time())
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    selection=json.loads((OUT/'selection.json').read_text());expected=json.loads((OUT/'metrics.json').read_text());meta={s:json.loads((OUT/f'{s}_metadata.json').read_text()) for s in ('train','val','test')};ys={s:targets(m) for s,m in meta.items()};data={s:dict(np.load(OUT/f'{s}_features.npz')) for s in meta};proj=dict(np.load(OUT/'projections.npz'));fits=np.load(OUT/'fitted_models.npz');pred=np.load(OUT/'test_predictions.npz');boot=np.load(OUT/'bootstrap_groups.npz')['indices'];ey=ys['test'];ev=ey['event'].astype(bool);idx=np.flatnonzero(ev);eboot=[r[ev[r]] for r in boot];emap=np.full(len(ev),-1);emap[idx]=np.arange(len(idx));eb=[emap[r] for r in eboot]
    keys=('mean','scale','train_z','kernel_mean','kernel_grand','coef','intercept','degree','alpha','effective_df','requested_df')
    capacity=[];valcheck=[];cache={};replayed=0
    def change_metric(delta):
        err=change_error(ey['delta'],delta)
        return dict(mean_mae=float(err.mean()),mae_degrees=err.mean(0).tolist(),changed_patch_mae=float(err[idx,ey['side'][ev]].mean()),changed_patch_ci95=np.quantile([err[r,ey['side'][r]].mean() for r in eboot],[.025,.975]).tolist(),catch_mae=float(err[~ev].mean()),predicted_change_std=delta.std(0).tolist())
    for key,z in selection.items():
        f={k:fits[f"{z['index']}_{k}"] for k in keys};x=representation(data['test'],z['representation'],z['mode'],proj);s=predict(f,x);assert np.array_equal(s,pred[key]);replayed+=1
        K=kernel(f['train_z'],f['train_z'],int(f['degree']));K=K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean();e=np.maximum(np.linalg.eigvalsh(K),0);df=float(np.sum(e/(e+f['alpha'])));assert abs(df-z['effective_df'])<1e-9
        capacity.append(dict(key=key,effective_df_recomputed=df,requested_df=z['requested_df'],input_dimensions=z['dimensions'],implicit_features=z['implicit_features']))
        v=predict(f,representation(data['val'],z['representation'],z['mode'],proj));vix=np.flatnonzero(ys['val']['event']) if z['target']=='side' else np.arange(len(v));best=int(np.argmax(z['validation_scores']))
        if z['target']=='change':score=-float(change_error(ys['val']['delta'],change_from_scores(v)).mean());metric=change_metric(change_from_scores(s))
        else:
            score=d.ba(ys['val'][z['target']][vix],(v[vix,0]>z['threshold']).astype(int));mask=ev if z['target']=='side' else np.ones(len(ev),bool);metric=metrics_binary(ey[z['target']][mask],s[mask,0],z['threshold'],eb if z['target']=='side' else boot);cache[key]=(ey[z['target']],s[:,0]-z['threshold'],mask)
        assert score==z['validation_scores'][best];assert metric==expected[key];valcheck.append(dict(key=key,validation_score=score,selected_index=best))
    pix=np.load(OUT/'test_pixel.npz');thresholds=json.loads((OUT/'pixel_thresholds.json').read_text())
    for w,name in enumerate(('full','endpoint')):
        delta=circular_change(pix['angles'][:,w,0],pix['angles'][:,w,1]);assert np.array_equal(delta,pred['pixel_flow/'+name+'/change']);assert change_metric(delta)==expected['pixel_flow/'+name+'/change']
        for task in ('event','side'):
            k='pixel_flow/'+name+'/'+task;score=abs(delta).max(1) if task=='event' else abs(delta[:,1])-abs(delta[:,0]);threshold=thresholds[name]['threshold'] if task=='event' else 0;mask=ev if task=='side' else np.ones(len(ev),bool);assert metrics_binary(ey[task][mask],score[mask],threshold,eb if task=='side' else boot)==expected[k];cache[k]=(ey[task],score-threshold,mask)
    assert change_metric(np.zeros_like(ey['delta']))==expected['zero_change_baseline']
    for key,result in expected['paired_gains'].items():
        a,b=key.split(' MINUS ');yy,sa,mask=cache[a];_,sb,_=cache[b];rs=eboot if a.endswith('side') else boot;vals=[d.ba(yy[r],(sa[r]>0).astype(int))-d.ba(yy[r],(sb[r]>0).astype(int)) for r in rs];calc=dict(ba_gain=d.ba(yy[mask],(sa[mask]>0).astype(int))-d.ba(yy[mask],(sb[mask]>0).astype(int)),ci95=np.quantile(vals,[.025,.975]).tolist());assert calc==result
    identity=json.loads((OUT/'identity.json').read_text());assert d.sha(d.CHECKPOINT)==identity['checkpoint_sha256']
    c=torch.load(d.CHECKPOINT,map_location='cpu',weights_only=False);classes={k.split('.')[1]:v.shape[0] for k,v in c['model'].items() if k.startswith('heads.') and k.endswith('.weight')};torch.manual_seed(identity['random_seed']);random=d.SpatialConsolidation(classes).eval().requires_grad_(False);assert d.state_hash(random)==identity['random_state_sha256'];savedrandom=torch.load(OUT/'random_initialized_state.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,savedrandom[k]) for k,v in random.state_dict().items());trained=d.SpatialConsolidation(classes);trained.load_state_dict(c['model']);trained.eval().requires_grad_(False);assert d.state_hash(trained)==identity['trained_state_sha256']
    for split in meta:
        dots=np.load(OUT/f'{split}_dot_angles.npz')['angles'];examples=np.load(OUT/f'{split}_examples.npz')
        for i,m in enumerate(meta[split]):assert np.array_equal(d.circular_targets(dots[i,:m['baseline_transitions']+8],m['baseline_transitions']),np.array(m['actual_circular_angles_radians']))
        for i in range(3):
            x=torch.from_numpy(examples[str(i)])[None];assert np.array_equal(raw_features(x),data[split]['pixels'][i]);assert np.array_equal(endpoint_features(trained,x),data[split]['trained'][i]);assert np.array_equal(endpoint_features(random,x),data[split]['random'][i])
    for p,h in json.loads((OUT/'test_freeze.json').read_text())['hashes'].items():assert d.sha(OUT/p)==h
    d.dump(OUT/'capacity_replay.json',capacity);d.dump(OUT/'validation_replay.json',valcheck)
    result=dict(fitted_predictors_replayed=replayed,metric_records_exact=len(expected)-2,paired_gains_exact=len(expected['paired_gains']),effective_df_recomputed=True,validation_chosen_scores_exact=True,actual_dot_angle_targets_exact=True,examples_all_representations_exact=9,random_state_regenerated_without_checkpoint_exact=True,random_saved_state_exact=True,checkpoint_hash_unchanged=True,frozen_sources_and_choices_unchanged=True,elapsed_seconds=time.time()-budget['started'],within_cap=time.time()<budget['deadline'],no_fits=True)
    d.dump(OUT/'independent_replay.json',result);print(json.dumps(result));signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
