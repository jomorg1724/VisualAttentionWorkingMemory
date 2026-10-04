"""No-fit serialized audit, constrained to the original immutable deadline."""
from run_upstream import *

def audit():
    budget=json.loads((OUT/'budget.json').read_text());remaining=budget['deadline']-time.time();assert remaining>0
    def expired(*args):raise TimeoutError('Original deadline expired during audit')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    f=np.load(OUT/'fitted_models.npz');pr=np.load(OUT/'projections.npz');p=np.load(OUT/'test_predictions.npz');data=dict(np.load(OUT/'test_features.npz'));tr=dict(np.load(OLD/'train_features.npz'));meta=json.loads((OUT/'test_metadata.json').read_text());tm=json.loads((OLD/'train_metadata.json').read_text());ty=targets(tm);ey=targets(meta);selection=json.loads((OUT/'selection.json').read_text());m=json.loads((OUT/'metrics.json').read_text());boot=np.load(OUT/'bootstrap_groups.npz')['indices'];original=np.arange(0,len(meta),2)
    perm=(2*np.random.default_rng(22761).permutation(425)[:,None]+np.arange(2)).ravel();eventgroups=np.flatnonzero(ty['event'][::2]);eventperm=(2*np.random.default_rng(22762).permutation(eventgroups)[:,None]+np.arange(2)).ravel()
    replay=0.;maxres=0.;counts=0;xs={};xt={}
    for key,z in selection.items():
        name=z['site']+'/'+z['access'];proj=pr[z['site']]
        if name not in xs:xs[name]=access(data,z['site'],z['access'],proj);xt[name]=access(tr,z['site'],z['access'],proj)
        fit={k:f[f"{z['index']}_{k}"] for k in ('mean','scale','weight','intercept')};calc=d.predict(fit,xs[name]);replay=max(replay,float(abs(calc-p[key]).max()))
        ix=np.flatnonzero(ty['event']) if z['target']=='side' else np.arange(len(tm));ti=(eventperm if z['target']=='side' else perm) if z['shuffled'] else ix
        x=xt[name][ix];mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
        assert np.array_equal(mean,fit['mean']) and np.array_equal(scale,fit['scale'])
        if z['target']=='change':
            a=np.deg2rad(ty['delta'][ti]);yy=np.stack([np.cos(a),np.sin(a)],-1).reshape(-1,4)
        else:yy=np.eye(2)[ty[z['target']][ti]]
        assert np.array_equal(yy.mean(0),fit['intercept']);xx=(x-mean)/scale;yc=yy-fit['intercept'];rhs=xx.T@yc
        residual=xx.T@(xx@fit['weight']-yc)+z['alpha']*fit['weight'];rel=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1e-12));maxres=max(maxres,rel);assert rel<1e-7
        if z['target']=='change':
            err=change_error(ey['delta'][original],change_from_scores(calc[original]));assert np.allclose(err.mean(0),m[key]['mae_degrees'],rtol=0,atol=1e-12);assert np.allclose(np.quantile(err[boot].mean((1,2)),[.025,.975]),m[key]['ci95'],rtol=0,atol=1e-12)
        else:
            mask=ey['event'][original].astype(bool) if z['target']=='side' else np.ones(len(original),bool);truth=ey[z['target']][original][mask];score=calc[original][mask];assert d.ba(truth,score.argmax(1))==m[key]['balanced_accuracy'];assert d.auc(truth,score[:,1]-score[:,0])==m[key]['auc']
        counts+=1
    oldsel=json.loads((OLD/'selection.json').read_text());oldfit=np.load(OLD/'fitted_probes.npz')
    for site in ('cnn25','kda25','kda7','memory'):
        for phase in ('baseline','post','final'):
            for patch in (0,1):
                prefix=oldsel[f'{site}__{phase}/angle/{patch}']['archive_prefix'];fit={k:oldfit[prefix+'_'+k] for k in ('mean','scale','weight','intercept')};calc=d.predict(fit,data[site+'__'+phase]);replay=max(replay,float(abs(calc-p[f'{site}/direction/{phase}/{patch}']).max()))
    examples=np.load(OUT/'example_frames.npz');legacydiff=0.
    for i in range(3):
        frames=examples[str(i)][0];new=pixel_summary(frames);old=px.krauzlis_observe(frames,len(frames)-17);legacydiff=max(legacydiff,float(np.max(abs(np.rad2deg(new['angles'][0].T)-np.array(old['estimated_angles_degrees'])))));assert new['cue']==old['decoded_target']
    assert legacydiff<1e-10 and replay==0
    freeze=json.loads((OUT/'test_freeze.json').read_text())
    for filename,h in freeze['hashes'].items():assert d.sha(OUT/filename)==h
    assert d.sha(d.CHECKPOINT)==json.loads((OUT/'identity.json').read_text())['checkpoint_sha256']
    result=dict(fits_audited=counts,all_scalers_exact=True,fit_normal_equation_max_relative_residual=maxres,replay_all_new_and_prior_direction_predictions_max_abs=replay,pixel_original_method_max_angle_difference_degrees=legacydiff,metrics_checked=True,change_grouped_ci_replayed=True,all_frozen_selections_unchanged=True,checkpoint_hash_unchanged=True,elapsed_seconds=time.time()-budget['started'],within_cap=time.time()<budget['deadline'])
    d.dump(OUT/'independent_audit.json',result);print(json.dumps(result,indent=2));signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':audit()
