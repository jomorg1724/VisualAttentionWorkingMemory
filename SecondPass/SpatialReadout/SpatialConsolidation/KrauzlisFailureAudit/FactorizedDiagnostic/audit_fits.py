"""No-refit audit: saved normal equations, scaler fit groups, validation scores."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import json,time
import numpy as np
from factorized import OUT,SRC,access,targets,calibration,predict,ba,dump,sha
budget=json.loads((OUT/'budget.json').read_text());assert time.time()<budget['deadline']
contract=json.loads((OUT/'split_contract.json').read_text());sel=json.loads((OUT/'selection.json').read_text());archive=np.load(OUT/'fitted_models.npz');p=np.load(OUT/'projection.npz')['p']
meta={s:json.loads((SRC/f'{s}_metadata.json').read_text()) for s in ['train','val']};ys={s:targets(m) for s,m in meta.items()}
ag=set(contract['component_groups']);bg=set(contract['calibration_groups']);assert not ag&bg
ar=np.array([i for i,m in enumerate(meta['train']) if m['group'] in ag]);br=np.array([i for i,m in enumerate(meta['train']) if m['group'] in bg]);assert len(ar)==600 and len(br)==250
order=np.random.default_rng(4433).permutation(300);shuffle=ar.reshape(-1,2)[order].ravel()
fits={key:{name:archive[key+'__'+name] for name in ['mean','scale','weight','intercept','alpha']} for key in sel}
check={};inputs={};normal_max=0
for kind in ['final','phases']:
    data={s:np.load(SRC/f'{s}_features.npz') for s in ['train','val']};xs={s:access(d,kind,p) for s,d in data.items()}
    for target in ['cue','event','side','joint','label','shuffled_label','calibrator']:
        key=kind+'/'+target;f=fits[key];name='label' if target in ['shuffled_label','calibrator'] else target
        mask=ys['val']['event']==1 if target=='side' else np.ones(200,bool)
        rows=ar[ys['train']['event'][ar]==1] if target=='side' else ar
        if target=='calibrator':
            rows=br
            cv=[np.clip(predict(fits[kind+'/'+t],xs['val'])[:,1],0,1) for t in ['cue','event','side']]
            ct=[np.clip(predict(fits[kind+'/'+t],xs['train'][br])[:,1],0,1) for t in ['cue','event','side']]
            x=calibration(*ct);vx=calibration(*cv)
            inputs[kind+'/calibration_predictions']=np.column_stack(ct)
        else:x=xs['train'][rows];vx=xs['val'][mask]
        y=np.eye(3 if target=='joint' else 2)[ys['train'][name][shuffle if target=='shuffled_label' else rows]]
        np.testing.assert_allclose(x.mean(0),f['mean'],rtol=0,atol=1e-12)
        scale=x.std(0);scale[scale<1e-12]=1;np.testing.assert_allclose(scale,f['scale'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(y.mean(0),f['intercept'],rtol=0,atol=1e-12)
        z=(x-f['mean'])/f['scale'];res=z.T@(z@f['weight']-(y-f['intercept']))+f['alpha']*f['weight']
        relative=float(np.max(np.abs(res))/(1+np.max(np.abs(z.T@(y-f['intercept'])))));assert relative<1e-7;normal_max=max(normal_max,relative)
        got=ba(ys['val'][name][mask],predict(f,vx).argmax(1));idx=[1.,100.,10000.].index(float(f['alpha']));assert got==sel[key]['validation_ba'][idx]
        check[key]=dict(scaler_group_mean_and_std_verified=True,normal_equation_relative_residual=relative,validation_ba_replayed=got)
np.savez(OUT/'calibration_inputs.npz',**inputs,row_indices=br)
# Actual runtime numerical backend information, not an assumed hardware label.
import io,contextlib
stream=io.StringIO()
with contextlib.redirect_stdout(stream):np.show_config()
dump(OUT/'fit_audit.json',dict(checks=check,max_normal_equation_relative_residual=normal_max,component_calibrator_groups_disjoint=True,scalers_bound_to_exact_fit_groups=True,calibration_uses_only_out_of_fit_group_predictions=True,numpy_runtime_config=stream.getvalue(),thread_environment={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},elapsed_from_first_fit=time.time()-budget['started']))
assert time.time()<budget['deadline']
print(json.dumps(dict(audited_models=len(check),max_relative_residual=normal_max,seconds_remaining=budget['deadline']-time.time())))
