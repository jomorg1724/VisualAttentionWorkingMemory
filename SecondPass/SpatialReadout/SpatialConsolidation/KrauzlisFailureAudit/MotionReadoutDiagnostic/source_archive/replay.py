"""Independent no-fit process: rerun input-only descriptors and serialized predictions."""
from run_motion import OUT,OLD,d,truth,metric,descriptor,budget,np,json,time,predict

def main():
    from run_motion import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cap=budget();fit_audit={};draws={};fits=np.load(OUT/'fits.npz');selection=json.loads((OUT/'selection.json').read_text());archive=np.load(OUT/'test_predictions.npz');metrics=json.loads((OUT/'metrics.json').read_text());reports={};maxdiff=0.
    # Fail closed if any fitting is accidentally reached during replay.
    import motion
    def forbidden(*a,**k):raise AssertionError('Replay must never fit')
    motion.fit=forbidden;motion.minimize=forbidden
    for split in ('train','val','test'):
        for rep in sorted({k.split('/')[0] for k in selection}):
            z=np.load(OUT/f'{split}_{rep}_maps.npz')['maps'];x=np.load(OUT/f'{split}_{rep}_descriptors.npz')['x']
            calc=np.stack([descriptor(a,1 if rep=='pixels' else 4) for a in z]);np.testing.assert_array_equal(calc,x)
            reports[f'{split}/{rep}']=dict(groups=len(z),descriptor_exact=True,shape=list(z.shape))
    meta=json.loads((OUT/'test_metadata.json').read_text());ev,side,_=truth(meta);boot=np.load(OUT/'test_bootstrap.npz')['indices']
    for name,q in selection.items():
        rep=name.split('/')[0];f={k:fits[name+'__'+k] for k in ('mean','scale','weight','iterations')}
        tx=np.load(OUT/f'train_{rep}_descriptors.npz')['x'].reshape(-1,8);np.testing.assert_array_equal(tx.mean(0),f['mean']);sc=tx.std(0);sc[sc<1e-10]=1;np.testing.assert_array_equal(sc,f['scale'])
        x=np.load(OUT/f'test_{rep}_descriptors.npz')['x'];p=predict(f,x);np.testing.assert_array_equal(p,archive[name]);m,draws[name]=metric(ev,side,p,q['threshold'],boot);assert m==metrics[name]
        _,_,labels=truth(json.loads((OUT/'train_metadata.json').read_text()))
        if q['shuffled']:labels=labels[np.load(OUT/'shuffle_permutation.npz')['indices']]
        labels=labels.ravel().astype(float);weights=np.where(labels==1,len(labels)/(2*labels.sum()),len(labels)/(2*(1-labels).sum()))
        z=np.c_[(tx-f['mean'])/f['scale'],np.ones(len(tx))];reg=f['weight'].copy();reg[-1]=0
        from scipy.special import expit
        gradient=z.T@(weights*(expit(z@f['weight'])-labels))+reg
        residual=float(abs(gradient).max()/len(tx));assert residual<1e-4
        fit_audit[name]=dict(train_groups=600,patch_rows=len(tx),gradient_max_abs_per_row=residual,fresh_initialization='zero, convex logistic; no inherited diagnostic weights',scaler_exact=True)
        maxdiff=max(maxdiff,float(abs(p-archive[name]).max()))
        val=np.load(OUT/f'val_{rep}_descriptors.npz')['x'];vp=predict(f,val);np.testing.assert_array_equal(vp,np.load(OUT/'validation_predictions.npz')[name])
    flow=np.load(OUT/'test_flow.npz')['angles'];from upstream import circular_change
    thresholds=json.loads((OUT/'flow_thresholds.json').read_text())
    for j,name in enumerate(('full','endpoint3','window5')):
        p=abs(circular_change(flow[:,j,0],flow[:,j,1]));np.testing.assert_array_equal(p,archive['flow/'+name]);m,draws['flow/'+name]=metric(ev,side,p,thresholds[name],boot);assert m==metrics['flow/'+name]
    for comparison,gain in metrics['paired_gains'].items():
        a,b=comparison.split(' MINUS ')
        for j,k in enumerate(('event_ba','event_auc','side_ba','side_auc')):
            assert gain[k]['value']==metrics[a][k]['value']-metrics[b][k]['value']
            np.testing.assert_array_equal(gain[k]['ci95'],np.quantile(draws[a][:,j]-draws[b][:,j],[.025,.975]))
    old=json.loads((OLD/'identity.json').read_text())
    from run_motion import pixel_summary
    import upstream
    assert d.sha(d.__file__)==old['extractor_sha256'] and d.sha(upstream.__file__)==old['upstream_source_sha256'] and d.sha(upstream.px.__file__)==old['pixel_source_sha256']
    d.dump(OUT/'fit_audit.json',fit_audit)
    ident=json.loads((OUT/'identity.json').read_text());assert d.sha(d.CHECKPOINT)==ident['checkpoint_sha256']
    for f,h in ident['source_hashes'].items():assert d.sha(d.ROOT/f)==h
    freeze=json.loads((OUT/'test_freeze.json').read_text())
    for f,h in freeze['hashes'].items():assert d.sha(OUT/f)==h
    assert freeze['time']<(OUT/'test_metadata.json').stat().st_mtime
    groups={}
    for split in ('train','val','test'):
        ms=json.loads((OUT/f'{split}_metadata.json').read_text());hashes={m['noncue_sha256'] for m in ms};assert len(hashes)==len(ms);groups[split]=hashes
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    prior=set()
    for folder in ('FrozenDiagnostic','UpstreamMotionDiagnostic','ProbeAdequacyDiagnostic'):
        for f in (OUT.parent/folder).glob('*_metadata.json'):prior.update(m['noncue_sha256'] for m in json.loads(f.read_text()))
    assert not groups['test']&prior
    # Renderer replay: all three initial test durations, raw maps and flow sums.
    from run_motion import raw_maps,pixel_summary
    stream=d.SpatialBatteryStream(106100731,'test');stored=np.load(OUT/'test_pixels_maps.npz')['maps']
    for i,b in enumerate((12,20,28)):
        x,y,mm=stream.batch(1,d.TASK,dict(baseline_transitions=b));np.testing.assert_array_equal(raw_maps(x),stored[i]);np.testing.assert_array_equal(pixel_summary(x[0].numpy())['angles'],flow[i,:2])
    result=dict(independent_process=True,no_fit=True,descriptor_replay=reports,prediction_max_abs=maxdiff,metrics_exact=True,validation_predictions_exact=True,train_only_scalers_exact=True,three_native_test_movies_replayed=True,test_disjoint_all_prior=True,checkpoint_sources_freeze_hashes_unchanged=True,elapsed=time.time()-cap['started'],within_cap=time.time()<cap['deadline'])
    d.dump(OUT/'independent_replay.json',result);print(json.dumps(result))

if __name__=='__main__':main()
